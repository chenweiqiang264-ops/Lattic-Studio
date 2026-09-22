/**
 * C++ 距离场计算加速模块
 * 用于加速 Voronoi 晶格生成中的距离场计算
 * 使用 pybind11 绑定到 Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/**
 * 计算点到线段的最短距离
 * 
 * @param point: 查询点 (x, y, z)
 * @param p1: 线段端点1 (x, y, z)
 * @param p2: 线段端点2 (x, y, z)
 * @return: 最短距离
 */
inline double point_to_segment_distance(
    const double* point,
    const double* p1,
    const double* p2
) {
    // 计算边向量
    double edge_x = p2[0] - p1[0];
    double edge_y = p2[1] - p1[1];
    double edge_z = p2[2] - p1[2];
    
    // 边长度的平方
    double edge_length_sq = edge_x * edge_x + edge_y * edge_y + edge_z * edge_z;
    
    // 退化边（两端点重合）
    if (edge_length_sq < 1e-20) {
        double dx = point[0] - p1[0];
        double dy = point[1] - p1[1];
        double dz = point[2] - p1[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    // 计算投影参数 t
    double dx = point[0] - p1[0];
    double dy = point[1] - p1[1];
    double dz = point[2] - p1[2];
    
    double t = (dx * edge_x + dy * edge_y + dz * edge_z) / edge_length_sq;
    
    // 限制在线段范围内 [0, 1]
    t = std::max(0.0, std::min(1.0, t));
    
    // 计算最近点
    double closest_x = p1[0] + t * edge_x;
    double closest_y = p1[1] + t * edge_y;
    double closest_z = p1[2] + t * edge_z;
    
    // 计算距离
    dx = point[0] - closest_x;
    dy = point[1] - closest_y;
    dz = point[2] - closest_z;
    
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * 计算距离场（单线程版本）
 * 
 * @param grid_points: NumPy 数组 (N, 3) - 网格点坐标
 * @param edges: 边列表，每条边是 [(x1,y1,z1), (x2,y2,z2)]
 * @return: NumPy 数组 (N,) - 每个网格点到最近边的距离
 */
py::array_t<double> compute_distance_field_single(
    py::array_t<double> grid_points,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>& edges
) {
    auto points_buf = grid_points.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("grid_points must be (N, 3) array");
    }
    
    size_t n_points = points_buf.shape[0];
    size_t n_edges = edges.size();
    
    double* points_ptr = static_cast<double*>(points_buf.ptr);
    
    // 创建输出数组
    py::array_t<double> distances({(py::ssize_t)n_points});
    auto dist_buf = distances.request();
    double* dist_ptr = static_cast<double*>(dist_buf.ptr);
    
    // 初始化为无穷大
    for (size_t i = 0; i < n_points; ++i) {
        dist_ptr[i] = std::numeric_limits<double>::infinity();
    }
    
    // 对每条边计算距离
    for (size_t edge_idx = 0; edge_idx < n_edges; ++edge_idx) {
        const auto& edge = edges[edge_idx];
        const double* p1 = edge.first.data();
        const double* p2 = edge.second.data();
        
        // 对每个网格点计算到当前边的距离
        for (size_t i = 0; i < n_points; ++i) {
            const double* point = points_ptr + i * 3;
            double dist = point_to_segment_distance(point, p1, p2);
            
            // 更新最小距离
            if (dist < dist_ptr[i]) {
                dist_ptr[i] = dist;
            }
        }
    }
    
    return distances;
}

/**
 * 计算距离场（多线程 OpenMP 版本）
 * 
 * @param grid_points: NumPy 数组 (N, 3) - 网格点坐标
 * @param edges: 边列表，每条边是 [(x1,y1,z1), (x2,y2,z2)]
 * @param num_threads: 线程数（0 = 自动）
 * @return: NumPy 数组 (N,) - 每个网格点到最近边的距离
 */
py::array_t<double> compute_distance_field_parallel(
    py::array_t<double> grid_points,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>& edges,
    int num_threads = 0
) {
    auto points_buf = grid_points.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("grid_points must be (N, 3) array");
    }
    
    size_t n_points = points_buf.shape[0];
    size_t n_edges = edges.size();
    
    double* points_ptr = static_cast<double*>(points_buf.ptr);
    
    // 创建输出数组
    py::array_t<double> distances({(py::ssize_t)n_points});
    auto dist_buf = distances.request();
    double* dist_ptr = static_cast<double*>(dist_buf.ptr);
    
    // 初始化为无穷大
    for (size_t i = 0; i < n_points; ++i) {
        dist_ptr[i] = std::numeric_limits<double>::infinity();
    }
    
#ifdef _OPENMP
    // 设置线程数
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // 并行计算（按边并行）
    #pragma omp parallel
    {
        // 每个线程有自己的临时距离数组
        std::vector<double> local_distances(n_points, std::numeric_limits<double>::infinity());
        
        #pragma omp for schedule(dynamic, 10)
        for (int edge_idx = 0; edge_idx < (int)n_edges; ++edge_idx) {
            const auto& edge = edges[edge_idx];
            const double* p1 = edge.first.data();
            const double* p2 = edge.second.data();
            
            // 对每个网格点计算到当前边的距离
            for (size_t i = 0; i < n_points; ++i) {
                const double* point = points_ptr + i * 3;
                double dist = point_to_segment_distance(point, p1, p2);
                
                // 更新局部最小距离
                if (dist < local_distances[i]) {
                    local_distances[i] = dist;
                }
            }
        }
        
        // 合并结果（临界区）
        #pragma omp critical
        {
            for (size_t i = 0; i < n_points; ++i) {
                if (local_distances[i] < dist_ptr[i]) {
                    dist_ptr[i] = local_distances[i];
                }
            }
        }
    }
#else
    // 如果没有 OpenMP，回退到单线程版本
    for (size_t edge_idx = 0; edge_idx < n_edges; ++edge_idx) {
        const auto& edge = edges[edge_idx];
        const double* p1 = edge.first.data();
        const double* p2 = edge.second.data();
        
        for (size_t i = 0; i < n_points; ++i) {
            const double* point = points_ptr + i * 3;
            double dist = point_to_segment_distance(point, p1, p2);
            
            if (dist < dist_ptr[i]) {
                dist_ptr[i] = dist;
            }
        }
    }
#endif
    
    return distances;
}

/**
 * 计算距离场（优化版本：按点并行）
 * 
 * 这个版本对每个网格点并行计算，避免了临界区的开销
 * 
 * @param grid_points: NumPy 数组 (N, 3) - 网格点坐标
 * @param edges: 边列表，每条边是 [(x1,y1,z1), (x2,y2,z2)]
 * @param num_threads: 线程数（0 = 自动）
 * @return: NumPy 数组 (N,) - 每个网格点到最近边的距离
 */
py::array_t<double> compute_distance_field_optimized(
    py::array_t<double> grid_points,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>& edges,
    int num_threads = 0
) {
    auto points_buf = grid_points.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("grid_points must be (N, 3) array");
    }
    
    size_t n_points = points_buf.shape[0];
    size_t n_edges = edges.size();
    
    double* points_ptr = static_cast<double*>(points_buf.ptr);
    
    // 创建输出数组
    py::array_t<double> distances({(py::ssize_t)n_points});
    auto dist_buf = distances.request();
    double* dist_ptr = static_cast<double*>(dist_buf.ptr);
    
#ifdef _OPENMP
    // 设置线程数
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // 并行计算（按点并行，无需临界区）
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < (int)n_points; ++i) {
        const double* point = points_ptr + i * 3;
        double min_dist = std::numeric_limits<double>::infinity();
        
        // 对当前点，遍历所有边找最小距离
        for (size_t edge_idx = 0; edge_idx < n_edges; ++edge_idx) {
            const auto& edge = edges[edge_idx];
            const double* p1 = edge.first.data();
            const double* p2 = edge.second.data();
            
            double dist = point_to_segment_distance(point, p1, p2);
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        dist_ptr[i] = min_dist;
    }
#else
    // 单线程版本
    for (size_t i = 0; i < n_points; ++i) {
        const double* point = points_ptr + i * 3;
        double min_dist = std::numeric_limits<double>::infinity();
        
        for (size_t edge_idx = 0; edge_idx < n_edges; ++edge_idx) {
            const auto& edge = edges[edge_idx];
            const double* p1 = edge.first.data();
            const double* p2 = edge.second.data();
            
            double dist = point_to_segment_distance(point, p1, p2);
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        dist_ptr[i] = min_dist;
    }
#endif
    
    return distances;
}

/**
 * 获取 OpenMP 信息
 */
py::dict get_openmp_info() {
    py::dict info;
    
#ifdef _OPENMP
    info["available"] = true;
    info["max_threads"] = omp_get_max_threads();
    info["version"] = _OPENMP;
#else
    info["available"] = false;
    info["max_threads"] = 1;
    info["version"] = 0;
#endif
    
    return info;
}

// pybind11 module definition
PYBIND11_MODULE(cpp_distance_field, m) {
    m.doc() = "C++ accelerated distance field computation module with OpenMP support";
    
    m.def("compute_distance_field_single", &compute_distance_field_single,
          "Compute distance field (single-threaded version)",
          py::arg("grid_points"),
          py::arg("edges"));
    
    m.def("compute_distance_field_parallel", &compute_distance_field_parallel,
          "Compute distance field (multi-threaded version, parallel by edges)",
          py::arg("grid_points"),
          py::arg("edges"),
          py::arg("num_threads") = 0);
    
    m.def("compute_distance_field_optimized", &compute_distance_field_optimized,
          "Compute distance field (optimized version, parallel by points) - Recommended",
          py::arg("grid_points"),
          py::arg("edges"),
          py::arg("num_threads") = 0);
    
    m.def("get_openmp_info", &get_openmp_info,
          "Get OpenMP information");
}
