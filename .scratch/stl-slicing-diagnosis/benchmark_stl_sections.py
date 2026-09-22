from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np


HEADER_BYTES = 84
TRIANGLE_BYTES = 50
TRIANGLE_DTYPE = np.dtype(
    {
        "names": ["vertices"],
        "formats": [("<f4", (3, 3))],
        "offsets": [12],
        "itemsize": TRIANGLE_BYTES,
    }
)


def triangle_count(path: Path) -> int:
    with path.open("rb") as stream:
        stream.seek(80)
        count = struct.unpack("<I", stream.read(4))[0]
    if HEADER_BYTES + count * TRIANGLE_BYTES != path.stat().st_size:
        raise ValueError(f"Not a standard binary STL: {path}")
    return count


def model_z_bounds(records: np.memmap, chunk_size: int) -> tuple[float, float]:
    z_min = np.inf
    z_max = -np.inf
    for start in range(0, records.shape[0], chunk_size):
        z = records[start : start + chunk_size]["vertices"][:, :, 2]
        z_min = min(z_min, float(z.min()))
        z_max = max(z_max, float(z.max()))
    return z_min, z_max


def plane_segments(
    records: np.memmap, level: float, chunk_size: int
) -> tuple[np.ndarray, float]:
    pieces: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, records.shape[0], chunk_size):
        vertices = records[start : start + chunk_size]["vertices"].astype(
            np.float64, copy=False
        )
        z = vertices[:, :, 2]
        active = (z.min(axis=1) <= level) & (z.max(axis=1) > level)
        vertices = vertices[active]
        if vertices.size == 0:
            continue
        z = vertices[:, :, 2]
        intersections = []
        valid_edges = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            z0 = z[:, first]
            z1 = z[:, second]
            crossing = ((z0 <= level) & (z1 > level)) | (
                (z1 <= level) & (z0 > level)
            )
            denominator = z1 - z0
            weight = np.divide(
                level - z0,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator != 0,
            )
            point = vertices[:, first, :2] + weight[:, None] * (
                vertices[:, second, :2] - vertices[:, first, :2]
            )
            intersections.append(point)
            valid_edges.append(crossing)
        points = np.stack(intersections, axis=1)
        valid = np.stack(valid_edges, axis=1)
        exactly_two = valid.sum(axis=1) == 2
        if not np.any(exactly_two):
            continue
        points = points[exactly_two]
        valid = valid[exactly_two]
        selected = points[valid].reshape(-1, 2, 2)
        pieces.append(selected)
    segments = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2, 2))
    return segments, time.perf_counter() - started


def _find(parent: np.ndarray, value: int) -> int:
    root = value
    while parent[root] != root:
        root = int(parent[root])
    while parent[value] != value:
        next_value = int(parent[value])
        parent[value] = root
        value = next_value
    return root


def assemble(segments: np.ndarray, tolerance: float) -> dict[str, int | float]:
    started = time.perf_counter()
    segment_lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    keys = np.rint(segments.reshape(-1, 2) / tolerance).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    nonzero = edges[:, 0] != edges[:, 1]
    collapsed = int(np.count_nonzero(~nonzero))
    edges = edges[nonzero]
    canonical = np.sort(edges, axis=1)
    unique_edges, multiplicity = np.unique(canonical, axis=0, return_counts=True)
    duplicate_edges = int(np.count_nonzero(multiplicity > 1))
    degrees = np.bincount(unique_edges.ravel(), minlength=int(inverse.max()) + 1)
    parent = np.arange(degrees.size, dtype=np.int64)
    rank = np.zeros(degrees.size, dtype=np.uint8)
    for first, second in unique_edges:
        root_first = _find(parent, int(first))
        root_second = _find(parent, int(second))
        if root_first == root_second:
            continue
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1
    roots = {_find(parent, index) for index in np.flatnonzero(degrees)}
    edge_roots = np.fromiter(
        (_find(parent, int(first)) for first in unique_edges[:, 0]),
        dtype=np.int64,
        count=unique_edges.shape[0],
    )
    _, component_segment_counts = np.unique(edge_roots, return_counts=True)
    component_quantiles = np.quantile(
        component_segment_counts, [0.01, 0.1, 0.5, 0.9, 0.99]
    )
    elapsed = time.perf_counter() - started
    return {
        "vertices": int(np.count_nonzero(degrees)),
        "unique_segments": int(unique_edges.shape[0]),
        "collapsed_segments": collapsed,
        "duplicate_segments": duplicate_edges,
        "open_vertices": int(np.count_nonzero(degrees == 1)),
        "branch_vertices": int(np.count_nonzero(degrees > 2)),
        "contour_components": len(roots),
        "total_contour_length_mm": float(segment_lengths.sum()),
        "small_contours_le_8_segments": int(
            np.count_nonzero(component_segment_counts <= 8)
        ),
        "component_segment_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p10", "p50", "p90", "p99"),
                component_quantiles,
                strict=True,
            )
        },
        "assembly_seconds": elapsed,
    }


def benchmark(path: Path, relative_levels: list[float], chunk_size: int, tolerance: float):
    count = triangle_count(path)
    records = np.memmap(
        path,
        dtype=TRIANGLE_DTYPE,
        mode="r",
        offset=HEADER_BYTES,
        shape=(count,),
    )
    z_min, z_max = model_z_bounds(records, chunk_size)
    results = []
    for relative in relative_levels:
        level = z_min + relative * (z_max - z_min)
        segments, intersection_seconds = plane_segments(records, level, chunk_size)
        topology = assemble(segments, tolerance)
        results.append(
            {
                "relative_level": relative,
                "z_mm": level,
                "raw_segments": int(segments.shape[0]),
                "intersection_seconds": intersection_seconds,
                **topology,
            }
        )
    del records
    return {"path": str(path), "weld_tolerance_mm": tolerance, "levels": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--levels", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = []
    for path in args.paths:
        result = benchmark(path.resolve(), args.levels, args.chunk_size, args.tolerance)
        output.append(result)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
