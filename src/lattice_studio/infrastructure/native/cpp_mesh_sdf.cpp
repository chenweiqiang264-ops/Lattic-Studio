#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

constexpr double EPS = 1e-12;
constexpr int LEAF_SIZE = 8;

struct Vec3 {
    double x;
    double y;
    double z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Triangle {
    Vec3 a;
    Vec3 b;
    Vec3 c;
    Vec3 centroid;
    Vec3 lo;
    Vec3 hi;
};

struct Node {
    Vec3 lo;
    Vec3 hi;
    int left = -1;
    int right = -1;
    int begin = 0;
    int end = 0;
};

inline Vec3 min_vec(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3 max_vec(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

inline double point_box_distance_sq(Vec3 p, const Node& node) {
    const double dx = std::max({node.lo.x - p.x, 0.0, p.x - node.hi.x});
    const double dy = std::max({node.lo.y - p.y, 0.0, p.y - node.hi.y});
    const double dz = std::max({node.lo.z - p.z, 0.0, p.z - node.hi.z});
    return dx * dx + dy * dy + dz * dz;
}

double point_triangle_distance_sq(Vec3 p, const Triangle& t) {
    const Vec3 ab = t.b - t.a;
    const Vec3 ac = t.c - t.a;
    const Vec3 ap = p - t.a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return dot(ap, ap);

    const Vec3 bp = p - t.b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return dot(bp, bp);

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const Vec3 q = t.a + ab * v;
        const Vec3 diff = p - q;
        return dot(diff, diff);
    }

    const Vec3 cp = p - t.c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return dot(cp, cp);

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const Vec3 q = t.a + ac * w;
        const Vec3 diff = p - q;
        return dot(diff, diff);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const Vec3 bc = t.c - t.b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3 q = t.b + bc * w;
        const Vec3 diff = p - q;
        return dot(diff, diff);
    }

    const Vec3 n = cross(ab, ac);
    const double n2 = dot(n, n);
    if (n2 <= EPS) {
        auto segment_distance_sq = [p](Vec3 a, Vec3 b) {
            const Vec3 edge = b - a;
            const double length_sq = dot(edge, edge);
            if (length_sq <= EPS) return dot(p - a, p - a);
            const double t = std::clamp(dot(p - a, edge) / length_sq, 0.0, 1.0);
            const Vec3 closest = a + edge * t;
            return dot(p - closest, p - closest);
        };
        return std::min({segment_distance_sq(t.a, t.b), segment_distance_sq(t.a, t.c), segment_distance_sq(t.b, t.c)});
    }
    const double distance_to_plane = dot(p - t.a, n) / std::sqrt(n2);
    return distance_to_plane * distance_to_plane;
}

class MeshSdf {
public:
    explicit MeshSdf(const double* vertices, const int32_t* faces, int n_vertices, int n_faces) {
        triangles_.reserve(static_cast<size_t>(n_faces));
        for (int i = 0; i < n_faces; ++i) {
            const int i0 = faces[i * 3 + 0];
            const int i1 = faces[i * 3 + 1];
            const int i2 = faces[i * 3 + 2];
            if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= n_vertices || i1 >= n_vertices || i2 >= n_vertices) {
                throw std::runtime_error("faces contain an out-of-range vertex index");
            }
            const Vec3 a{vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]};
            const Vec3 b{vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]};
            const Vec3 c{vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]};
            const Vec3 lo = min_vec(a, min_vec(b, c));
            const Vec3 hi = max_vec(a, max_vec(b, c));
            triangles_.push_back({a, b, c, (a + b + c) * (1.0 / 3.0), lo, hi});
        }
        order_.resize(triangles_.size());
        std::iota(order_.begin(), order_.end(), 0);
        if (!triangles_.empty()) build_node(0, static_cast<int>(order_.size()));
    }

    double unsigned_distance(Vec3 p) const {
        if (nodes_.empty()) return std::numeric_limits<double>::infinity();
        double best = std::numeric_limits<double>::infinity();
        std::vector<int> stack{0};
        while (!stack.empty()) {
            const int node_index = stack.back();
            stack.pop_back();
            const Node& node = nodes_[node_index];
            if (point_box_distance_sq(p, node) >= best) continue;
            if (node.left < 0) {
                for (int k = node.begin; k < node.end; ++k) {
                    best = std::min(best, point_triangle_distance_sq(p, triangles_[order_[k]]));
                }
            } else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }
        return std::sqrt(best);
    }

    bool inside(Vec3 p, double ray_length) const {
        // A small deterministic transverse offset avoids rays running exactly on
        // a mesh edge, which is a measure-zero but common voxel-grid case.
        const double eps = 1e-9;
        p.y += eps;
        p.z += eps * 0.61803398875;
        const Vec3 dir{1.0, 0.0, 0.0};
        int hits = 0;
        std::vector<int> stack{0};
        while (!stack.empty()) {
            const int node_index = stack.back();
            stack.pop_back();
            const Node& node = nodes_[node_index];
            if (p.y < node.lo.y - EPS || p.y > node.hi.y + EPS || p.z < node.lo.z - EPS || p.z > node.hi.z + EPS || node.hi.x <= p.x + EPS) continue;
            if (node.left < 0) {
                for (int k = node.begin; k < node.end; ++k) {
                    const Triangle& t = triangles_[order_[k]];
                    if (p.y < t.lo.y - EPS || p.y > t.hi.y + EPS || p.z < t.lo.z - EPS || p.z > t.hi.z + EPS || t.hi.x <= p.x + EPS) continue;
                    const Vec3 e1 = t.b - t.a;
                    const Vec3 e2 = t.c - t.a;
                    const Vec3 h = cross(dir, e2);
                    const double a = dot(e1, h);
                    if (std::abs(a) < EPS) continue;
                    const double f = 1.0 / a;
                    const Vec3 s = p - t.a;
                    const double u = f * dot(s, h);
                    if (u < -EPS || u > 1.0 + EPS) continue;
                    const Vec3 q = cross(s, e1);
                    const double v = f * dot(dir, q);
                    if (v < -EPS || u + v > 1.0 + EPS) continue;
                    const double distance = f * dot(e2, q);
                    if (distance > EPS && distance < ray_length) ++hits;
                }
            } else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }
        return (hits & 1) != 0;
    }

private:
    int build_node(int begin, int end) {
        Node node;
        node.begin = begin;
        node.end = end;
        node.lo = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        node.hi = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
        Vec3 centroid_lo = node.lo;
        Vec3 centroid_hi = node.hi;
        for (int k = begin; k < end; ++k) {
            const Triangle& t = triangles_[order_[k]];
            node.lo = min_vec(node.lo, t.lo);
            node.hi = max_vec(node.hi, t.hi);
            centroid_lo = min_vec(centroid_lo, t.centroid);
            centroid_hi = max_vec(centroid_hi, t.centroid);
        }
        const int index = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        if (end - begin > LEAF_SIZE) {
            const Vec3 span = centroid_hi - centroid_lo;
            int axis = 0;
            if (span.y > span.x && span.y >= span.z) axis = 1;
            else if (span.z > span.x && span.z > span.y) axis = 2;
            std::sort(order_.begin() + begin, order_.begin() + end, [&](int lhs, int rhs) {
                const Vec3 a = triangles_[lhs].centroid;
                const Vec3 b = triangles_[rhs].centroid;
                return (axis == 0 ? a.x : axis == 1 ? a.y : a.z) < (axis == 0 ? b.x : axis == 1 ? b.y : b.z);
            });
            const int middle = begin + (end - begin) / 2;
            const int left = build_node(begin, middle);
            const int right = build_node(middle, end);
            nodes_[index].left = left;
            nodes_[index].right = right;
        }
        return index;
    }

    std::vector<Triangle> triangles_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
};

} // namespace

py::array_t<float> compute_signed_distance(
    py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> faces,
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    int num_threads = 0
) {
    const auto vb = vertices.request();
    const auto fb = faces.request();
    const auto pb = points.request();
    if (vb.ndim != 2 || vb.shape[1] != 3) throw std::runtime_error("vertices must have shape (N, 3)");
    if (fb.ndim != 2 || fb.shape[1] != 3) throw std::runtime_error("faces must have shape (M, 3)");
    if (pb.ndim != 2 || pb.shape[1] != 3) throw std::runtime_error("points must have shape (K, 3)");
    if (fb.shape[0] == 0) throw std::runtime_error("faces must not be empty");

    const auto* v = static_cast<const double*>(vb.ptr);
    const auto* f = static_cast<const int32_t*>(fb.ptr);
    const auto* p = static_cast<const double*>(pb.ptr);
    MeshSdf mesh(v, f, static_cast<int>(vb.shape[0]), static_cast<int>(fb.shape[0]));
    py::array_t<float> result(pb.shape[0]);
    auto rb = result.request();
    auto* out = static_cast<float*>(rb.ptr);
    const double ray_length = 1.0e12;

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#endif
    for (py::ssize_t i = 0; i < pb.shape[0]; ++i) {
        const Vec3 point{p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
        const double distance = mesh.unsigned_distance(point);
        out[i] = static_cast<float>(mesh.inside(point, ray_length) ? -distance : distance);
    }
    return result;
}

PYBIND11_MODULE(cpp_mesh_sdf, m) {
    m.doc() = "C++ BVH accelerated signed distance field for a watertight triangle mesh";
    m.def("compute_signed_distance", &compute_signed_distance,
          py::arg("vertices"), py::arg("faces"), py::arg("points"), py::arg("num_threads") = 0,
          "Return negative-inside signed distances for query points.");
}
