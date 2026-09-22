# Compact GPU Extraction and Cache Display Proxies

Marching Cubes must identify vertices by grid edge before transferring output
to system memory. CUDA marks active local edges and emits final shared vertex
and face arrays; a bounded host edge-mask scan supplies compact indices and
global edge identities for chunk stitching. This removes duplicate vertex and
`int64` edge-ID transfers and the host `np.unique` sort. CUDA availability is
reported only after a real device allocation and copy probe succeeds.

Topology inspection counts connected components without materializing a mesh
copy for every component. Numerical-fragment cleanup uses face-component labels
and bounded signed-volume accumulation for watertight meshes, retaining the
existing tolerance and relative-volume criteria.

Large derived surface meshes remain authoritative for topology inspection and
STL export. Rendering uses a cached VTK topology-preserving display proxy whose
face budget combines an absolute floor with a minimum source-face ratio. This
prevents high-frequency TPMS surfaces from collapsing to a tiny fixed budget.
Section interaction updates polygon mapper clipping planes without rebuilding
the proxy; changing visibility or material reuses it. The proxy must never be
exported or written back into the implicit generation result.

Mesh actors, mappers, and their geometry inputs remain resident while callers
change material or visibility. A stable mesh cache key is therefore an
immutability contract: callers must provide a new key when geometry changes.
Camera rotation and panning may switch to a separately cached interaction-only
LOD built in a background worker, then restore the topology-preserving target
actor on release. Interaction LODs are display artifacts and must not be used
for topology checks, repair, simplification, or export. Inactive resident mesh
layers are bounded and evicted to avoid unbounded CPU and GPU memory growth.
