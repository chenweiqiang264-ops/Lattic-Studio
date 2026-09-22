#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double EPS = 1e-12;

struct Tri2D {
    int i0;
    int i1;
    int i2;
    double minx;
    double maxx;
    double miny;
    double maxy;
};

inline bool point_in_tri_2d(
    double px, double py,
    const double* a,
    const double* b,
    const double* c,
    double& u,
    double& v,
    double& w
) {
    const double v0x = b[0] - a[0];
    const double v0y = b[1] - a[1];
    const double v1x = c[0] - a[0];
    const double v1y = c[1] - a[1];
    const double v2x = px - a[0];
    const double v2y = py - a[1];

    const double den = v0x * v1y - v1x * v0y;
    if (std::abs(den) < EPS) {
        return false;
    }

    const double inv = 1.0 / den;
    v = (v2x * v1y - v1x * v2y) * inv;
    w = (v0x * v2y - v2x * v0y) * inv;
    u = 1.0 - v - w;

    return (u >= -EPS && v >= -EPS && w >= -EPS);
}

} // namespace

py::array_t<uint8_t> compute_inside_mask(
    py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> faces,
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    int nx,
    int ny,
    int grid_size
) {
    auto vbuf = vertices.request();
    auto fbuf = faces.request();
    auto pbuf = points.request();

    if (vbuf.ndim != 2 || vbuf.shape[1] != 3) {
        throw std::runtime_error("vertices must be shape (N,3)");
    }
    if (fbuf.ndim != 2 || fbuf.shape[1] != 3) {
        throw std::runtime_error("faces must be shape (M,3)");
    }
    if (pbuf.ndim != 2 || pbuf.shape[1] != 3) {
        throw std::runtime_error("points must be shape (K,3)");
    }

    const auto* V = static_cast<const double*>(vbuf.ptr);
    const auto* F = static_cast<const int32_t*>(fbuf.ptr);
    const auto* P = static_cast<const double*>(pbuf.ptr);

    const py::ssize_t n_faces = fbuf.shape[0];
    const py::ssize_t n_pts = pbuf.shape[0];

    if (n_pts <= 0) {
        return py::array_t<uint8_t>(0);
    }
    if (nx <= 0 || ny <= 0) {
        throw std::runtime_error("nx and ny must be positive");
    }
    if (grid_size < 8) {
        grid_size = 8;
    }

    double pminx = P[0], pmaxx = P[0], pminy = P[1], pmaxy = P[1];
    for (py::ssize_t i = 1; i < n_pts; ++i) {
        const double x = P[i * 3 + 0];
        const double y = P[i * 3 + 1];
        pminx = std::min(pminx, x);
        pmaxx = std::max(pmaxx, x);
        pminy = std::min(pminy, y);
        pmaxy = std::max(pmaxy, y);
    }

    const double spanx = std::max(pmaxx - pminx, 1e-9);
    const double spany = std::max(pmaxy - pminy, 1e-9);

    std::vector<Tri2D> tris;
    tris.reserve(static_cast<size_t>(n_faces));

    for (py::ssize_t fi = 0; fi < n_faces; ++fi) {
        const int i0 = F[fi * 3 + 0];
        const int i1 = F[fi * 3 + 1];
        const int i2 = F[fi * 3 + 2];

        const double* a = &V[static_cast<py::ssize_t>(i0) * 3];
        const double* b = &V[static_cast<py::ssize_t>(i1) * 3];
        const double* c = &V[static_cast<py::ssize_t>(i2) * 3];

        const double minx = std::min({a[0], b[0], c[0]});
        const double maxx = std::max({a[0], b[0], c[0]});
        const double miny = std::min({a[1], b[1], c[1]});
        const double maxy = std::max({a[1], b[1], c[1]});

        if (maxx < pminx || minx > pmaxx || maxy < pminy || miny > pmaxy) {
            continue;
        }

        tris.push_back({i0, i1, i2, minx, maxx, miny, maxy});
    }

    const int gx = grid_size;
    const int gy = grid_size;
    std::vector<std::vector<int>> buckets(static_cast<size_t>(gx * gy));

    auto to_ix = [&](double x) {
        int ix = static_cast<int>(((x - pminx) / spanx) * gx);
        if (ix < 0) ix = 0;
        if (ix >= gx) ix = gx - 1;
        return ix;
    };
    auto to_iy = [&](double y) {
        int iy = static_cast<int>(((y - pminy) / spany) * gy);
        if (iy < 0) iy = 0;
        if (iy >= gy) iy = gy - 1;
        return iy;
    };

    for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
        const auto& t = tris[ti];
        const int ix0 = to_ix(t.minx);
        const int ix1 = to_ix(t.maxx);
        const int iy0 = to_iy(t.miny);
        const int iy1 = to_iy(t.maxy);
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                buckets[static_cast<size_t>(iy * gx + ix)].push_back(ti);
            }
        }
    }

    py::array_t<uint8_t> out(n_pts);
    auto obuf = out.request();
    auto* O = static_cast<uint8_t*>(obuf.ptr);

    std::vector<double> zhits;
    zhits.reserve(256);

    for (py::ssize_t pi = 0; pi < n_pts; ++pi) {
        const double px = P[pi * 3 + 0];
        const double py = P[pi * 3 + 1];
        const double pz = P[pi * 3 + 2];

        const int ix = to_ix(px);
        const int iy = to_iy(py);
        const auto& cand = buckets[static_cast<size_t>(iy * gx + ix)];

        zhits.clear();
        for (int tidx : cand) {
            const auto& t = tris[tidx];
            if (px < t.minx - EPS || px > t.maxx + EPS || py < t.miny - EPS || py > t.maxy + EPS) {
                continue;
            }

            const double* a = &V[static_cast<py::ssize_t>(t.i0) * 3];
            const double* b = &V[static_cast<py::ssize_t>(t.i1) * 3];
            const double* c = &V[static_cast<py::ssize_t>(t.i2) * 3];

            double u, v, w;
            if (!point_in_tri_2d(px, py, a, b, c, u, v, w)) {
                continue;
            }

            const double z = u * a[2] + v * b[2] + w * c[2];
            if (z > pz + EPS) {
                zhits.push_back(z);
            }
        }

        if (zhits.empty()) {
            O[pi] = 0;
            continue;
        }

        std::sort(zhits.begin(), zhits.end());
        int unique_hits = 1;
        for (size_t k = 1; k < zhits.size(); ++k) {
            if (std::abs(zhits[k] - zhits[k - 1]) > 1e-8) {
                ++unique_hits;
            }
        }

        O[pi] = static_cast<uint8_t>((unique_hits % 2) == 1 ? 1 : 0);
    }

    return out;
}

PYBIND11_MODULE(cpp_mask, m) {
    m.doc() = "C++ accelerated inside-mask computation for gyroid voxel points";
    m.def(
        "compute_inside_mask",
        &compute_inside_mask,
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("points"),
        py::arg("nx"),
        py::arg("ny"),
        py::arg("grid_size") = 96
    );
}
