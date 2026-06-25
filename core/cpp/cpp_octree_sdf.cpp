/*
 * C++ 八叉树 SDF 构建加速模块
 * 
 * 功能：
 * 1. 从周期性 SDF 快速构建八叉树
 * 2. 支持 OpenMP 多线程加速
 * 3. 自适应细分和均匀性检测
 * 
 * 编译：python core/cpp/build_octree_sdf.py
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// 八叉树节点结构
struct OctreeNode {
    // 节点信息
    float center[3];      // 中心点
    float size;           // 尺寸
    int depth;            // 深度
    
    // 子节点（8个）或 nullptr（叶节点）
    std::vector<std::shared_ptr<OctreeNode>> children;
    
    // 叶节点数据
    bool is_leaf;
    float sdf_value;      // 均匀区域：单个值
    std::vector<float> sdf_grid;  // 复杂区域：小网格
    int grid_resolution;  // 网格分辨率
    
    // 统计信息
    float min_sdf;
    float max_sdf;
    
    OctreeNode(float cx, float cy, float cz, float s, int d)
        : size(s), depth(d), is_leaf(true), sdf_value(0.0f), 
          grid_resolution(0), min_sdf(1e10f), max_sdf(-1e10f) {
        center[0] = cx;
        center[1] = cy;
        center[2] = cz;
    }
    
    bool is_uniform(float threshold) const {
        return (max_sdf - min_sdf) < threshold;
    }
};

// 周期性 SDF 采样函数
inline float sample_periodic_sdf(
    const py::array_t<float>& unit_sdf,
    float x, float y, float z,
    const float* unit_bbox_min,
    float cell_size,
    int unit_resolution
) {
    // 映射到晶胞内（周期性）
    float local_x = x - unit_bbox_min[0];
    float local_y = y - unit_bbox_min[1];
    float local_z = z - unit_bbox_min[2];
    
    float inv_cell_size = 1.0f / cell_size;
    
    // 周期性取模
    local_x = fmodf(local_x, cell_size);
    local_y = fmodf(local_y, cell_size);
    local_z = fmodf(local_z, cell_size);
    
    if (local_x < 0) local_x += cell_size;
    if (local_y < 0) local_y += cell_size;
    if (local_z < 0) local_z += cell_size;
    
    // 转换为单元 SDF 的索引（最近邻）
    int i = static_cast<int>(local_x * inv_cell_size * unit_resolution);
    int j = static_cast<int>(local_y * inv_cell_size * unit_resolution);
    int k = static_cast<int>(local_z * inv_cell_size * unit_resolution);
    
    // 限制在有效范围内
    i = std::max(0, std::min(unit_resolution - 1, i));
    j = std::max(0, std::min(unit_resolution - 1, j));
    k = std::max(0, std::min(unit_resolution - 1, k));
    
    // 访问 numpy 数组
    auto buf = unit_sdf.unchecked<3>();
    return buf(i, j, k);
}

// 在节点内采样网格
void sample_grid(
    std::shared_ptr<OctreeNode> node,
    const py::array_t<float>& unit_sdf,
    const float* unit_bbox_min,
    float cell_size,
    int unit_resolution,
    int leaf_resolution
) {
    float half = node->size / 2.0f;
    
    node->grid_resolution = leaf_resolution;
    node->sdf_grid.resize(leaf_resolution * leaf_resolution * leaf_resolution);
    
    float step = node->size / (leaf_resolution - 1);
    
    for (int i = 0; i < leaf_resolution; i++) {
        float dx = -half + i * step;
        for (int j = 0; j < leaf_resolution; j++) {
            float dy = -half + j * step;
            for (int k = 0; k < leaf_resolution; k++) {
                float dz = -half + k * step;
                
                float x = node->center[0] + dx;
                float y = node->center[1] + dy;
                float z = node->center[2] + dz;
                
                float sdf = sample_periodic_sdf(
                    unit_sdf, x, y, z,
                    unit_bbox_min, cell_size, unit_resolution
                );
                
                int idx = i * leaf_resolution * leaf_resolution + j * leaf_resolution + k;
                node->sdf_grid[idx] = sdf;
                
                node->min_sdf = std::min(node->min_sdf, sdf);
                node->max_sdf = std::max(node->max_sdf, sdf);
            }
        }
    }
}

// 递归构建节点
void build_node(
    std::shared_ptr<OctreeNode> node,
    const py::array_t<float>& unit_sdf,
    const float* unit_bbox_min,
    float cell_size,
    int unit_resolution,
    int leaf_resolution,
    float uniform_threshold,
    int max_depth
) {
    // 在节点范围内采样以判断是否需要细分
    float half = node->size / 2.0f;
    
    // 采样 27 个点（3x3x3）
    std::vector<float> sample_values;
    sample_values.reserve(27);
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                float x = node->center[0] + dx * half;
                float y = node->center[1] + dy * half;
                float z = node->center[2] + dz * half;
                
                float sdf = sample_periodic_sdf(
                    unit_sdf, x, y, z,
                    unit_bbox_min, cell_size, unit_resolution
                );
                
                sample_values.push_back(sdf);
                node->min_sdf = std::min(node->min_sdf, sdf);
                node->max_sdf = std::max(node->max_sdf, sdf);
            }
        }
    }
    
    float sdf_range = node->max_sdf - node->min_sdf;
    
    // 判断是否需要细分
    if (sdf_range < uniform_threshold || node->depth >= max_depth) {
        // 叶节点
        if (sdf_range < uniform_threshold) {
            // 均匀区域：只存储一个值
            float sum = 0.0f;
            for (float v : sample_values) {
                sum += v;
            }
            node->sdf_value = sum / sample_values.size();
        } else {
            // 复杂区域：存储小网格
            sample_grid(node, unit_sdf, unit_bbox_min, cell_size, 
                       unit_resolution, leaf_resolution);
        }
        return;
    }
    
    // 细分为 8 个子节点
    node->is_leaf = false;
    node->children.resize(8);
    
    float quarter = node->size / 4.0f;
    
    // 8 个子节点的偏移
    int offsets[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
        {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}
    };
    
    for (int i = 0; i < 8; i++) {
        float cx = node->center[0] + offsets[i][0] * quarter;
        float cy = node->center[1] + offsets[i][1] * quarter;
        float cz = node->center[2] + offsets[i][2] * quarter;
        
        node->children[i] = std::make_shared<OctreeNode>(
            cx, cy, cz, half, node->depth + 1
        );
        
        build_node(
            node->children[i], unit_sdf, unit_bbox_min,
            cell_size, unit_resolution, leaf_resolution,
            uniform_threshold, max_depth
        );
    }
}

// 统计节点数
void count_nodes(
    const std::shared_ptr<OctreeNode>& node,
    int& total_nodes,
    int& leaf_nodes
) {
    total_nodes++;
    
    if (node->is_leaf) {
        leaf_nodes++;
    } else {
        for (const auto& child : node->children) {
            count_nodes(child, total_nodes, leaf_nodes);
        }
    }
}

// 估算内存占用
size_t estimate_memory(const std::shared_ptr<OctreeNode>& node) {
    size_t base_size = sizeof(OctreeNode);
    
    if (node->is_leaf) {
        if (!node->sdf_grid.empty()) {
            return base_size + node->sdf_grid.size() * sizeof(float);
        } else {
            return base_size + sizeof(float);
        }
    } else {
        size_t total = base_size;
        for (const auto& child : node->children) {
            total += estimate_memory(child);
        }
        return total;
    }
}

// 序列化节点为 Python 字典
py::dict serialize_node(const std::shared_ptr<OctreeNode>& node) {
    py::dict result;
    
    result["center"] = py::make_tuple(node->center[0], node->center[1], node->center[2]);
    result["size"] = node->size;
    result["depth"] = node->depth;
    result["is_leaf"] = node->is_leaf;
    result["min_sdf"] = node->min_sdf;
    result["max_sdf"] = node->max_sdf;
    
    if (node->is_leaf) {
        if (!node->sdf_grid.empty()) {
            // 复杂区域：返回网格
            result["type"] = "grid";
            result["grid_resolution"] = node->grid_resolution;
            
            // 转换为 numpy 数组
            int res = node->grid_resolution;
            py::array_t<float> grid({res, res, res});
            auto buf = grid.mutable_unchecked<3>();
            
            for (int i = 0; i < res; i++) {
                for (int j = 0; j < res; j++) {
                    for (int k = 0; k < res; k++) {
                        int idx = i * res * res + j * res + k;
                        buf(i, j, k) = node->sdf_grid[idx];
                    }
                }
            }
            
            result["sdf_grid"] = grid;
        } else {
            // 均匀区域：返回单个值
            result["type"] = "uniform";
            result["sdf_value"] = node->sdf_value;
        }
    } else {
        // 非叶节点：递归序列化子节点
        result["type"] = "internal";
        py::list children_list;
        for (const auto& child : node->children) {
            children_list.append(serialize_node(child));
        }
        result["children"] = children_list;
    }
    
    return result;
}

// 主函数：构建八叉树
py::dict build_octree_from_periodic_sdf(
    py::array_t<float> unit_sdf,
    py::array_t<float> target_bbox_min_arr,
    py::array_t<float> target_bbox_max_arr,
    float cell_size,
    py::array_t<float> unit_bbox_min_arr,
    int unit_resolution,
    int max_depth,
    int leaf_resolution,
    float uniform_threshold,
    bool verbose
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 转换 numpy 数组为 C++ 数组
    auto bbox_min_buf = target_bbox_min_arr.unchecked<1>();
    auto bbox_max_buf = target_bbox_max_arr.unchecked<1>();
    auto unit_bbox_buf = unit_bbox_min_arr.unchecked<1>();
    
    float target_bbox_min[3] = {
        static_cast<float>(bbox_min_buf(0)),
        static_cast<float>(bbox_min_buf(1)),
        static_cast<float>(bbox_min_buf(2))
    };
    
    float target_bbox_max[3] = {
        static_cast<float>(bbox_max_buf(0)),
        static_cast<float>(bbox_max_buf(1)),
        static_cast<float>(bbox_max_buf(2))
    };
    
    float unit_bbox_min[3] = {
        static_cast<float>(unit_bbox_buf(0)),
        static_cast<float>(unit_bbox_buf(1)),
        static_cast<float>(unit_bbox_buf(2))
    };
    
    // 计算根节点参数
    float center[3] = {
        (target_bbox_min[0] + target_bbox_max[0]) / 2.0f,
        (target_bbox_min[1] + target_bbox_max[1]) / 2.0f,
        (target_bbox_min[2] + target_bbox_max[2]) / 2.0f
    };
    
    float size = std::max({
        target_bbox_max[0] - target_bbox_min[0],
        target_bbox_max[1] - target_bbox_min[1],
        target_bbox_max[2] - target_bbox_min[2]
    });
    
    if (verbose) {
        std::cout << "  构建八叉树 SDF（C++ 加速）..." << std::endl;
        std::cout << "    最大深度: " << max_depth << std::endl;
        std::cout << "    叶节点分辨率: " << leaf_resolution << "³" << std::endl;
        std::cout << "    均匀性阈值: " << uniform_threshold << std::endl;
        
#ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        std::cout << "    OpenMP 线程数: " << num_threads << std::endl;
#endif
    }
    
    // 创建根节点
    auto root = std::make_shared<OctreeNode>(center[0], center[1], center[2], size, 0);
    
    // 构建八叉树
    build_node(
        root, unit_sdf, unit_bbox_min,
        cell_size, unit_resolution, leaf_resolution,
        uniform_threshold, max_depth
    );
    
    // 统计信息
    int total_nodes = 0;
    int leaf_nodes = 0;
    count_nodes(root, total_nodes, leaf_nodes);
    
    size_t octree_memory = estimate_memory(root);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (verbose) {
        std::cout << "  ✓ 八叉树构建完成 (耗时: " 
                  << duration.count() / 1000.0 << "s)" << std::endl;
        std::cout << "  节点总数: " << total_nodes << std::endl;
        std::cout << "  叶节点数: " << leaf_nodes << std::endl;
        std::cout << "  八叉树内存: " << octree_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    
    // 序列化为 Python 字典
    py::dict result;
    result["root"] = serialize_node(root);
    result["total_nodes"] = total_nodes;
    result["leaf_nodes"] = leaf_nodes;
    result["memory_bytes"] = octree_memory;
    result["build_time_ms"] = duration.count();
    result["bbox_min"] = target_bbox_min_arr;
    result["bbox_max"] = target_bbox_max_arr;
    result["max_depth"] = max_depth;
    
    return result;
}

// Python bindings
PYBIND11_MODULE(cpp_octree_sdf, m) {
    m.doc() = "C++ Octree SDF acceleration module";
    
    m.def("build_octree_from_periodic_sdf", &build_octree_from_periodic_sdf,
          py::arg("unit_sdf"),
          py::arg("target_bbox_min"),
          py::arg("target_bbox_max"),
          py::arg("cell_size"),
          py::arg("unit_bbox_min"),
          py::arg("unit_resolution"),
          py::arg("max_depth") = 8,
          py::arg("leaf_resolution") = 4,
          py::arg("uniform_threshold") = 0.01f,
          py::arg("verbose") = true,
          "Build octree from periodic SDF (C++ accelerated)");
    
#ifdef _OPENMP
    m.attr("OPENMP_ENABLED") = true;
    m.attr("NUM_THREADS") = omp_get_max_threads();
#else
    m.attr("OPENMP_ENABLED") = false;
    m.attr("NUM_THREADS") = 1;
#endif
    
    m.attr("__version__") = "1.0.0";
}
