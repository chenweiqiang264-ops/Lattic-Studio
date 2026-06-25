/**
 * C++ Periodic SDF Sampling Acceleration Module
 * Used to accelerate SDF sampling in periodic lattice filling
 * Bound to Python using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/**
 * Periodic SDF sampling (core acceleration function)
 * 
 * Map each point in large voxel grid to unit cell SDF for querying
 * 
 * @param unit_sdf: Unit cell SDF, shape (res, res, res)
 * @param target_resolution: Target resolution [nx, ny, nz]
 * @param target_bbox_min: Target bbox min [x, y, z]
 * @param spacing: Voxel spacing [dx, dy, dz]
 * @param unit_bbox_min: Unit cell bbox min [x, y, z]
 * @param cell_size: Cell size [sx, sy, sz]
 * @param unit_resolution: Unit cell SDF resolution
 * @return: Periodic lattice SDF, shape (nx, ny, nz)
 */
py::array_t<float> sample_periodic_sdf(
    py::array_t<float> unit_sdf,
    py::array_t<int> target_resolution,
    py::array_t<double> target_bbox_min,
    py::array_t<double> spacing,
    py::array_t<double> unit_bbox_min,
    py::array_t<double> cell_size,
    int unit_resolution
) {
    // Get input array buffers
    auto unit_sdf_buf = unit_sdf.request();
    auto target_res_buf = target_resolution.request();
    auto bbox_min_buf = target_bbox_min.request();
    auto spacing_buf = spacing.request();
    auto unit_bbox_buf = unit_bbox_min.request();
    auto cell_size_buf = cell_size.request();
    
    // Validate input
    if (unit_sdf_buf.ndim != 3) {
        throw std::runtime_error("unit_sdf must be 3D array");
    }
    if (target_res_buf.size != 3 || bbox_min_buf.size != 3 || 
        spacing_buf.size != 3 || unit_bbox_buf.size != 3 || cell_size_buf.size != 3) {
        throw std::runtime_error("All vector parameters must have size 3");
    }
    
    // Extract parameters
    float* unit_sdf_ptr = static_cast<float*>(unit_sdf_buf.ptr);
    int* target_res_ptr = static_cast<int*>(target_res_buf.ptr);
    double* bbox_min_ptr = static_cast<double*>(bbox_min_buf.ptr);
    double* spacing_ptr = static_cast<double*>(spacing_buf.ptr);
    double* unit_bbox_ptr = static_cast<double*>(unit_bbox_buf.ptr);
    double* cell_size_ptr = static_cast<double*>(cell_size_buf.ptr);
    
    int nx = target_res_ptr[0];
    int ny = target_res_ptr[1];
    int nz = target_res_ptr[2];
    
    double bbox_min_x = bbox_min_ptr[0];
    double bbox_min_y = bbox_min_ptr[1];
    double bbox_min_z = bbox_min_ptr[2];
    
    double spacing_x = spacing_ptr[0];
    double spacing_y = spacing_ptr[1];
    double spacing_z = spacing_ptr[2];
    
    double unit_bbox_x = unit_bbox_ptr[0];
    double unit_bbox_y = unit_bbox_ptr[1];
    double unit_bbox_z = unit_bbox_ptr[2];
    
    double cell_size_x = cell_size_ptr[0];
    double cell_size_y = cell_size_ptr[1];
    double cell_size_z = cell_size_ptr[2];
    
    // Create output array
    py::array_t<float> lattice_sdf({nx, ny, nz});
    auto lattice_buf = lattice_sdf.request();
    float* lattice_ptr = static_cast<float*>(lattice_buf.ptr);
    
    // Precompute constants
    double inv_cell_size_x = 1.0 / cell_size_x;
    double inv_cell_size_y = 1.0 / cell_size_y;
    double inv_cell_size_z = 1.0 / cell_size_z;
    double unit_res_scale = static_cast<double>(unit_resolution);
    
    // Parallel computation
    #ifdef _OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                // Compute world coordinates
                double world_x = bbox_min_x + i * spacing_x;
                double world_y = bbox_min_y + j * spacing_y;
                double world_z = bbox_min_z + k * spacing_z;
                
                // Map to unit cell (periodic)
                double local_x = world_x - unit_bbox_x;
                double local_y = world_y - unit_bbox_y;
                double local_z = world_z - unit_bbox_z;
                
                // Modulo (periodic)
                double local_mod_x = fmod(local_x, cell_size_x);
                double local_mod_y = fmod(local_y, cell_size_y);
                double local_mod_z = fmod(local_z, cell_size_z);
                
                // Handle negative modulo
                if (local_mod_x < 0) local_mod_x += cell_size_x;
                if (local_mod_y < 0) local_mod_y += cell_size_y;
                if (local_mod_z < 0) local_mod_z += cell_size_z;
                
                // Convert to unit SDF indices
                int unit_i = static_cast<int>(local_mod_x * inv_cell_size_x * unit_res_scale);
                int unit_j = static_cast<int>(local_mod_y * inv_cell_size_y * unit_res_scale);
                int unit_k = static_cast<int>(local_mod_z * inv_cell_size_z * unit_res_scale);
                
                // Boundary check
                if (unit_i >= unit_resolution) unit_i = unit_resolution - 1;
                if (unit_j >= unit_resolution) unit_j = unit_resolution - 1;
                if (unit_k >= unit_resolution) unit_k = unit_resolution - 1;
                if (unit_i < 0) unit_i = 0;
                if (unit_j < 0) unit_j = 0;
                if (unit_k < 0) unit_k = 0;
                
                // Query SDF (note: unit_sdf is C-order, index is [i][j][k])
                int unit_idx = unit_i * unit_resolution * unit_resolution + 
                               unit_j * unit_resolution + 
                               unit_k;
                
                // Write result
                int lattice_idx = i * ny * nz + j * nz + k;
                lattice_ptr[lattice_idx] = unit_sdf_ptr[unit_idx];
            }
        }
    }
    
    return lattice_sdf;
}

/**
 * Compute point to triangle distance (accurate)
 */
inline double point_to_triangle_distance(
    double px, double py, double pz,
    double v0x, double v0y, double v0z,
    double v1x, double v1y, double v1z,
    double v2x, double v2y, double v2z
) {
    // Vector from v0 to point
    double dx = px - v0x;
    double dy = py - v0y;
    double dz = pz - v0z;
    
    // Triangle edges
    double e0x = v1x - v0x;
    double e0y = v1y - v0y;
    double e0z = v1z - v0z;
    
    double e1x = v2x - v0x;
    double e1y = v2y - v0y;
    double e1z = v2z - v0z;
    
    // Dot products
    double a = e0x * e0x + e0y * e0y + e0z * e0z;
    double b = e0x * e1x + e0y * e1y + e0z * e1z;
    double c = e1x * e1x + e1y * e1y + e1z * e1z;
    double d = e0x * dx + e0y * dy + e0z * dz;
    double e = e1x * dx + e1y * dy + e1z * dz;
    
    double det = a * c - b * b;
    double s = b * e - c * d;
    double t = b * d - a * e;
    
    // Barycentric coordinates
    if (s + t <= det) {
        if (s < 0.0) {
            if (t < 0.0) {
                // Region 4
                s = 0.0;
                t = 0.0;
            } else {
                // Region 3
                s = 0.0;
                t = (e >= 0.0) ? 0.0 : ((-e >= c) ? 1.0 : -e / c);
            }
        } else if (t < 0.0) {
            // Region 5
            t = 0.0;
            s = (d >= 0.0) ? 0.0 : ((-d >= a) ? 1.0 : -d / a);
        } else {
            // Region 0 (inside triangle)
            double inv_det = 1.0 / det;
            s *= inv_det;
            t *= inv_det;
        }
    } else {
        if (s < 0.0) {
            // Region 2
            s = 0.0;
            t = 1.0;
        } else if (t < 0.0) {
            // Region 6
            s = 1.0;
            t = 0.0;
        } else {
            // Region 1
            double numer = c + e - b - d;
            if (numer <= 0.0) {
                s = 0.0;
            } else {
                double denom = a - 2.0 * b + c;
                s = (numer >= denom) ? 1.0 : numer / denom;
            }
            t = 1.0 - s;
        }
    }
    
    // Closest point on triangle
    double cpx = v0x + s * e0x + t * e1x;
    double cpy = v0y + s * e0y + t * e1y;
    double cpz = v0z + s * e0z + t * e1z;
    
    // Distance
    dx = px - cpx;
    dy = py - cpy;
    dz = pz - cpz;
    
    return sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Batch compute point-to-mesh distances (accurate version with OpenMP)
 * 
 * @param points: Query points (N, 3)
 * @param vertices: Mesh vertices (V, 3)
 * @param faces: Mesh faces (F, 3)
 * @return: Distance array (N,)
 */
py::array_t<double> compute_mesh_distance_accurate(
    py::array_t<double> points,
    py::array_t<double> vertices,
    py::array_t<int> faces
) {
    auto points_buf = points.request();
    auto verts_buf = vertices.request();
    auto faces_buf = faces.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("points must be (N, 3) array");
    }
    if (verts_buf.ndim != 2 || verts_buf.shape[1] != 3) {
        throw std::runtime_error("vertices must be (V, 3) array");
    }
    if (faces_buf.ndim != 2 || faces_buf.shape[1] != 3) {
        throw std::runtime_error("faces must be (F, 3) array");
    }
    
    size_t n_points = points_buf.shape[0];
    size_t n_faces = faces_buf.shape[0];
    
    double* points_ptr = static_cast<double*>(points_buf.ptr);
    double* verts_ptr = static_cast<double*>(verts_buf.ptr);
    int* faces_ptr = static_cast<int*>(faces_buf.ptr);
    
    // Create output array
    py::array_t<double> distances({(py::ssize_t)n_points});
    auto dist_buf = distances.request();
    double* dist_ptr = static_cast<double*>(dist_buf.ptr);
    
    // Parallel compute distance from each point to mesh
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1000)
    #endif
    for (size_t i = 0; i < n_points; ++i) {
        double px = points_ptr[i * 3 + 0];
        double py = points_ptr[i * 3 + 1];
        double pz = points_ptr[i * 3 + 2];
        
        double min_dist = 1e30;
        
        // Find minimum distance to all triangles
        for (size_t f = 0; f < n_faces; ++f) {
            int v0_idx = faces_ptr[f * 3 + 0];
            int v1_idx = faces_ptr[f * 3 + 1];
            int v2_idx = faces_ptr[f * 3 + 2];
            
            double v0x = verts_ptr[v0_idx * 3 + 0];
            double v0y = verts_ptr[v0_idx * 3 + 1];
            double v0z = verts_ptr[v0_idx * 3 + 2];
            
            double v1x = verts_ptr[v1_idx * 3 + 0];
            double v1y = verts_ptr[v1_idx * 3 + 1];
            double v1z = verts_ptr[v1_idx * 3 + 2];
            
            double v2x = verts_ptr[v2_idx * 3 + 0];
            double v2y = verts_ptr[v2_idx * 3 + 1];
            double v2z = verts_ptr[v2_idx * 3 + 2];
            
            double dist = point_to_triangle_distance(
                px, py, pz,
                v0x, v0y, v0z,
                v1x, v1y, v1z,
                v2x, v2y, v2z
            );
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        dist_ptr[i] = min_dist;
    }
    
    return distances;
}

/**
 * Batch compute point-to-mesh distances (for step 4)
 * 
 * This is a simplified version, should use more efficient algorithms (like BVH)
 * But for medium-scale meshes it's already much faster than Python
 * 
 * @param points: Query points (N, 3)
 * @param vertices: Mesh vertices (V, 3)
 * @param faces: Mesh faces (F, 3)
 * @return: Distance array (N,)
 */
py::array_t<double> compute_mesh_distance(
    py::array_t<double> points,
    py::array_t<double> vertices,
    py::array_t<int> faces
) {
    auto points_buf = points.request();
    auto verts_buf = vertices.request();
    auto faces_buf = faces.request();
    
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("points must be (N, 3) array");
    }
    if (verts_buf.ndim != 2 || verts_buf.shape[1] != 3) {
        throw std::runtime_error("vertices must be (V, 3) array");
    }
    if (faces_buf.ndim != 2 || faces_buf.shape[1] != 3) {
        throw std::runtime_error("faces must be (F, 3) array");
    }
    
    size_t n_points = points_buf.shape[0];
    size_t n_faces = faces_buf.shape[0];
    
    double* points_ptr = static_cast<double*>(points_buf.ptr);
    double* verts_ptr = static_cast<double*>(verts_buf.ptr);
    int* faces_ptr = static_cast<int*>(faces_buf.ptr);
    
    // Create output array
    py::array_t<double> distances({(py::ssize_t)n_points});
    auto dist_buf = distances.request();
    double* dist_ptr = static_cast<double*>(dist_buf.ptr);
    
    // Parallel compute distance from each point to mesh
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n_points; ++i) {
        double px = points_ptr[i * 3 + 0];
        double py = points_ptr[i * 3 + 1];
        double pz = points_ptr[i * 3 + 2];
        
        double min_dist = 1e30;
        
        // Iterate all triangles (simplified version, should use spatial index)
        for (size_t f = 0; f < n_faces; ++f) {
            int v0_idx = faces_ptr[f * 3 + 0];
            int v1_idx = faces_ptr[f * 3 + 1];
            int v2_idx = faces_ptr[f * 3 + 2];
            
            double v0x = verts_ptr[v0_idx * 3 + 0];
            double v0y = verts_ptr[v0_idx * 3 + 1];
            double v0z = verts_ptr[v0_idx * 3 + 2];
            
            double v1x = verts_ptr[v1_idx * 3 + 0];
            double v1y = verts_ptr[v1_idx * 3 + 1];
            double v1z = verts_ptr[v1_idx * 3 + 2];
            
            double v2x = verts_ptr[v2_idx * 3 + 0];
            double v2y = verts_ptr[v2_idx * 3 + 1];
            double v2z = verts_ptr[v2_idx * 3 + 2];
            
            // Simplified: only compute distance to triangle center (fast approximation)
            double cx = (v0x + v1x + v2x) / 3.0;
            double cy = (v0y + v1y + v2y) / 3.0;
            double cz = (v0z + v1z + v2z) / 3.0;
            
            double dx = px - cx;
            double dy = py - cy;
            double dz = pz - cz;
            
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        dist_ptr[i] = min_dist;
    }
    
    return distances;
}

// Python module binding
PYBIND11_MODULE(cpp_periodic_sdf, m) {
    m.doc() = "C++ periodic SDF sampling acceleration module";
    
    m.def("sample_periodic_sdf", &sample_periodic_sdf,
          "Periodic SDF sampling (core acceleration function)",
          py::arg("unit_sdf"),
          py::arg("target_resolution"),
          py::arg("target_bbox_min"),
          py::arg("spacing"),
          py::arg("unit_bbox_min"),
          py::arg("cell_size"),
          py::arg("unit_resolution"));
    
    m.def("compute_mesh_distance", &compute_mesh_distance,
          "Batch compute point-to-mesh distances (fast approximation)",
          py::arg("points"),
          py::arg("vertices"),
          py::arg("faces"));
    
    m.def("compute_mesh_distance_accurate", &compute_mesh_distance_accurate,
          "Batch compute point-to-mesh distances (accurate with OpenMP)",
          py::arg("points"),
          py::arg("vertices"),
          py::arg("faces"));
}
