/**
 * C++ 网格分割加速模块
 * 使用 pybind11 绑定到 Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <tuple>

namespace py = pybind11;

// 3D 向量结构
struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    double& operator[](int idx) {
        if (idx == 0) return x;
        if (idx == 1) return y;
        return z;
    }
    
    const double& operator[](int idx) const {
        if (idx == 0) return x;
        if (idx == 1) return y;
        return z;
    }
};

// 顶点哈希函数（用于去重）
struct Vec3Hash {
    std::size_t operator()(const Vec3& v) const {
        // 简单的哈希组合
        std::size_t h1 = std::hash<double>{}(std::round(v.x * 1e8) / 1e8);
        std::size_t h2 = std::hash<double>{}(std::round(v.y * 1e8) / 1e8);
        std::size_t h3 = std::hash<double>{}(std::round(v.z * 1e8) / 1e8);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// 顶点相等比较（用于去重）
struct Vec3Equal {
    bool operator()(const Vec3& a, const Vec3& b) const {
        const double eps = 1e-8;
        return std::abs(a.x - b.x) < eps && 
               std::abs(a.y - b.y) < eps && 
               std::abs(a.z - b.z) < eps;
    }
};

// 三角面结构
struct Triangle {
    int v0, v1, v2;
    
    Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
};

/**
 * 精准分割网格
 * 
 * @param vertices_np: NumPy 数组 (N, 3) - 顶点坐标
 * @param faces_np: NumPy 数组 (M, 3) - 面索引
 * @param axis_idx: 分割轴索引 (0=x, 1=y, 2=z)
 * @param position: 分割位置
 * @return: (pos_vertices, pos_faces, neg_vertices, neg_faces)
 */
std::tuple<py::array_t<double>, py::array_t<int>, 
           py::array_t<double>, py::array_t<int>>
slice_mesh_precise(
    py::array_t<double> vertices_np,
    py::array_t<int> faces_np,
    int axis_idx,
    double position
) {
    // 获取输入数组信息
    auto vertices_buf = vertices_np.request();
    auto faces_buf = faces_np.request();
    
    if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
        throw std::runtime_error("vertices must be (N, 3) array");
    }
    if (faces_buf.ndim != 2 || faces_buf.shape[1] != 3) {
        throw std::runtime_error("faces must be (M, 3) array");
    }
    
    int n_vertices = vertices_buf.shape[0];
    int n_faces = faces_buf.shape[0];
    
    double* vertices_ptr = static_cast<double*>(vertices_buf.ptr);
    int* faces_ptr = static_cast<int*>(faces_buf.ptr);
    
    // 转换为 Vec3 数组
    std::vector<Vec3> vertices(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        vertices[i] = Vec3(
            vertices_ptr[i * 3 + 0],
            vertices_ptr[i * 3 + 1],
            vertices_ptr[i * 3 + 2]
        );
    }
    
    // 计算所有顶点到平面的距离
    std::vector<double> distances(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        distances[i] = vertices[i][axis_idx] - position;
    }
    
    const double eps = 1e-10;
    
    // 分类顶点符号
    std::vector<int> signs(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        if (distances[i] > eps) signs[i] = 1;
        else if (distances[i] < -eps) signs[i] = -1;
        else signs[i] = 0;
    }
    
    // 结果存储
    std::vector<Vec3> pos_verts;
    std::vector<Triangle> pos_faces;
    std::vector<Vec3> neg_verts;
    std::vector<Triangle> neg_faces;
    
    // 顶点映射（去重）
    std::unordered_map<Vec3, int, Vec3Hash, Vec3Equal> pos_vert_map;
    std::unordered_map<Vec3, int, Vec3Hash, Vec3Equal> neg_vert_map;
    
    // 添加顶点到正侧
    auto add_pos_vert = [&](const Vec3& v) -> int {
        auto it = pos_vert_map.find(v);
        if (it != pos_vert_map.end()) {
            return it->second;
        }
        int idx = pos_verts.size();
        pos_verts.push_back(v);
        pos_vert_map[v] = idx;
        return idx;
    };
    
    // 添加顶点到负侧
    auto add_neg_vert = [&](const Vec3& v) -> int {
        auto it = neg_vert_map.find(v);
        if (it != neg_vert_map.end()) {
            return it->second;
        }
        int idx = neg_verts.size();
        neg_verts.push_back(v);
        neg_vert_map[v] = idx;
        return idx;
    };
    
    // 处理每个三角面
    for (int face_idx = 0; face_idx < n_faces; ++face_idx) {
        int i0 = faces_ptr[face_idx * 3 + 0];
        int i1 = faces_ptr[face_idx * 3 + 1];
        int i2 = faces_ptr[face_idx * 3 + 2];
        
        Vec3 v0 = vertices[i0];
        Vec3 v1 = vertices[i1];
        Vec3 v2 = vertices[i2];
        
        int s0 = signs[i0];
        int s1 = signs[i1];
        int s2 = signs[i2];
        
        // 情况1：所有顶点在正侧或平面上
        if (s0 >= 0 && s1 >= 0 && s2 >= 0) {
            int idx0 = add_pos_vert(v0);
            int idx1 = add_pos_vert(v1);
            int idx2 = add_pos_vert(v2);
            pos_faces.emplace_back(idx0, idx1, idx2);
        }
        // 情况2：所有顶点在负侧或平面上
        else if (s0 <= 0 && s1 <= 0 && s2 <= 0) {
            int idx0 = add_neg_vert(v0);
            int idx1 = add_neg_vert(v1);
            int idx2 = add_neg_vert(v2);
            neg_faces.emplace_back(idx0, idx1, idx2);
        }
        // 情况3：三角面跨越平面
        else {
            Vec3 verts[3] = {v0, v1, v2};
            int signs_arr[3] = {s0, s1, s2};
            double dists[3] = {distances[i0], distances[i1], distances[i2]};
            
            // 找交点
            std::vector<Vec3> intersections;
            
            for (int i = 0; i < 3; ++i) {
                int j = (i + 1) % 3;
                int si = signs_arr[i];
                int sj = signs_arr[j];
                
                // 边跨越平面
                if (si * sj < 0) {
                    Vec3 vi = verts[i];
                    Vec3 vj = verts[j];
                    double di = dists[i];
                    double dj = dists[j];
                    
                    // 计算交点
                    double t = di / (di - dj);
                    Vec3 intersection = vi + (vj - vi) * t;
                    // 强制交点在平面上
                    intersection[axis_idx] = position;
                    
                    intersections.push_back(intersection);
                }
            }
            
            if (intersections.size() != 2) {
                // 异常情况，跳过
                continue;
            }
            
            Vec3 p1 = intersections[0];
            Vec3 p2 = intersections[1];
            
            // 统计正侧和负侧的顶点
            std::vector<Vec3> pos_verts_tri;
            std::vector<Vec3> neg_verts_tri;
            
            for (int i = 0; i < 3; ++i) {
                if (signs_arr[i] > 0) {
                    pos_verts_tri.push_back(verts[i]);
                } else if (signs_arr[i] < 0) {
                    neg_verts_tri.push_back(verts[i]);
                }
            }
            
            if (pos_verts_tri.size() == 1) {
                // 1个顶点在正侧，2个在负侧
                // 正侧：1个三角形
                int idx0 = add_pos_vert(pos_verts_tri[0]);
                int idx1 = add_pos_vert(p1);
                int idx2 = add_pos_vert(p2);
                pos_faces.emplace_back(idx0, idx1, idx2);
                
                // 负侧：四边形 -> 2个三角形
                int idx_n0 = add_neg_vert(neg_verts_tri[0]);
                int idx_n1 = add_neg_vert(neg_verts_tri[1]);
                int idx_p1 = add_neg_vert(p1);
                int idx_p2 = add_neg_vert(p2);
                neg_faces.emplace_back(idx_n0, idx_n1, idx_p1);
                neg_faces.emplace_back(idx_n1, idx_p2, idx_p1);
            }
            else if (neg_verts_tri.size() == 1) {
                // 2个顶点在正侧，1个在负侧
                // 负侧：1个三角形
                int idx0 = add_neg_vert(neg_verts_tri[0]);
                int idx1 = add_neg_vert(p1);
                int idx2 = add_neg_vert(p2);
                neg_faces.emplace_back(idx0, idx1, idx2);
                
                // 正侧：四边形 -> 2个三角形
                int idx_p0 = add_pos_vert(pos_verts_tri[0]);
                int idx_p1_pos = add_pos_vert(pos_verts_tri[1]);
                int idx_p1 = add_pos_vert(p1);
                int idx_p2 = add_pos_vert(p2);
                pos_faces.emplace_back(idx_p0, idx_p1_pos, idx_p1);
                pos_faces.emplace_back(idx_p1_pos, idx_p2, idx_p1);
            }
        }
    }
    
    // 转换为 NumPy 数组
    // 正侧顶点
    py::array_t<double> pos_verts_np({(int)pos_verts.size(), 3});
    auto pos_verts_buf = pos_verts_np.request();
    double* pos_verts_ptr = static_cast<double*>(pos_verts_buf.ptr);
    for (size_t i = 0; i < pos_verts.size(); ++i) {
        pos_verts_ptr[i * 3 + 0] = pos_verts[i].x;
        pos_verts_ptr[i * 3 + 1] = pos_verts[i].y;
        pos_verts_ptr[i * 3 + 2] = pos_verts[i].z;
    }
    
    // 正侧面
    py::array_t<int> pos_faces_np({(int)pos_faces.size(), 3});
    auto pos_faces_buf = pos_faces_np.request();
    int* pos_faces_ptr = static_cast<int*>(pos_faces_buf.ptr);
    for (size_t i = 0; i < pos_faces.size(); ++i) {
        pos_faces_ptr[i * 3 + 0] = pos_faces[i].v0;
        pos_faces_ptr[i * 3 + 1] = pos_faces[i].v1;
        pos_faces_ptr[i * 3 + 2] = pos_faces[i].v2;
    }
    
    // 负侧顶点
    py::array_t<double> neg_verts_np({(int)neg_verts.size(), 3});
    auto neg_verts_buf = neg_verts_np.request();
    double* neg_verts_ptr = static_cast<double*>(neg_verts_buf.ptr);
    for (size_t i = 0; i < neg_verts.size(); ++i) {
        neg_verts_ptr[i * 3 + 0] = neg_verts[i].x;
        neg_verts_ptr[i * 3 + 1] = neg_verts[i].y;
        neg_verts_ptr[i * 3 + 2] = neg_verts[i].z;
    }
    
    // 负侧面
    py::array_t<int> neg_faces_np({(int)neg_faces.size(), 3});
    auto neg_faces_buf = neg_faces_np.request();
    int* neg_faces_ptr = static_cast<int*>(neg_faces_buf.ptr);
    for (size_t i = 0; i < neg_faces.size(); ++i) {
        neg_faces_ptr[i * 3 + 0] = neg_faces[i].v0;
        neg_faces_ptr[i * 3 + 1] = neg_faces[i].v1;
        neg_faces_ptr[i * 3 + 2] = neg_faces[i].v2;
    }
    
    return std::make_tuple(pos_verts_np, pos_faces_np, neg_verts_np, neg_faces_np);
}

// pybind11 模块定义
PYBIND11_MODULE(cpp_mesh_slicer, m) {
    m.doc() = "C++ 加速的网格分割模块";
    
    m.def("slice_mesh_precise", &slice_mesh_precise,
          "精准分割网格（C++ 加速版本）",
          py::arg("vertices"),
          py::arg("faces"),
          py::arg("axis_idx"),
          py::arg("position"));
}
