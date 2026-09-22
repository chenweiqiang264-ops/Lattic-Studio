from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


STL_HEADER_BYTES = 84
STL_TRIANGLE_BYTES = 50
TRIANGLE_DTYPE = np.dtype(
    {
        "names": ["normal", "vertices", "attribute"],
        "formats": [("<f4", (3,)), ("<f4", (3, 3)), "<u2"],
        "offsets": [0, 12, 48],
        "itemsize": STL_TRIANGLE_BYTES,
    }
)


@dataclass(frozen=True)
class StlSummary:
    path: str
    file_bytes: int
    triangle_count: int
    read_seconds: float
    bounds_min: list[float]
    bounds_max: list[float]
    extents: list[float]
    area_sum: float
    zero_area_count: int
    tiny_area_count: int
    poor_quality_count: int
    very_poor_quality_count: int
    exact_horizontal_count: int
    near_horizontal_count: int
    tiny_z_span_count: int
    edge_length_quantiles_mm: dict[str, float]
    area_quantiles_mm2: dict[str, float]
    quality_quantiles: dict[str, float]
    z_span_quantiles_mm: dict[str, float]
    slice_levels_mm: list[float]
    slice_intersection_counts: list[int]
    slice_count_mean: float
    slice_count_max: int
    slice_count_sum: int


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._values = np.empty(capacity, dtype=np.float64)
        self._size = 0
        self._seen = 0

    def add(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        room = self._capacity - self._size
        if room > 0:
            take = min(room, flat.size)
            self._values[self._size : self._size + take] = flat[:take]
            self._size += take
            self._seen += take
            flat = flat[take:]
        if flat.size == 0:
            return
        # Vectorized priority sampling keeps memory bounded and is deterministic.
        positions = np.arange(self._seen + 1, self._seen + flat.size + 1)
        selected = self._rng.random(flat.size) < (self._capacity / positions)
        selected_values = flat[selected]
        if selected_values.size:
            slots = self._rng.integers(0, self._capacity, selected_values.size)
            self._values[slots] = selected_values
        self._seen += flat.size

    def quantiles(self) -> dict[str, float]:
        if self._size == 0:
            return {}
        quantile_values = np.quantile(
            self._values[: self._size], [0.01, 0.1, 0.5, 0.9, 0.99]
        )
        return {
            name: float(value)
            for name, value in zip(
                ("p01", "p10", "p50", "p90", "p99"), quantile_values, strict=True
            )
        }


def _triangle_count(path: Path) -> int:
    with path.open("rb") as stream:
        stream.seek(80)
        raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError(f"STL header is incomplete: {path}")
    count = struct.unpack("<I", raw)[0]
    expected = STL_HEADER_BYTES + count * STL_TRIANGLE_BYTES
    actual = path.stat().st_size
    if expected != actual:
        raise ValueError(
            f"Expected {expected:,} bytes for {count:,} binary triangles, got {actual:,}"
        )
    return count


def _chunks(records: np.memmap, chunk_triangles: int):
    for start in range(0, records.shape[0], chunk_triangles):
        yield records[start : start + chunk_triangles]["vertices"].astype(
            np.float64, copy=False
        )


def scan(path: Path, chunk_triangles: int, slice_levels: int) -> StlSummary:
    triangle_count = _triangle_count(path)
    records = np.memmap(
        path,
        dtype=TRIANGLE_DTYPE,
        mode="r",
        offset=STL_HEADER_BYTES,
        shape=(triangle_count,),
    )
    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    started = time.perf_counter()
    for vertices in _chunks(records, chunk_triangles):
        bounds_min = np.minimum(bounds_min, vertices.min(axis=(0, 1)))
        bounds_max = np.maximum(bounds_max, vertices.max(axis=(0, 1)))
    extents = bounds_max - bounds_min
    scale = max(float(np.max(extents)), 1.0)
    area_epsilon = (scale * scale) * 1e-14
    z_epsilon = scale * 1e-7
    levels = np.linspace(bounds_min[2], bounds_max[2], slice_levels + 2)[1:-1]
    slice_counts = np.zeros(levels.size, dtype=np.int64)

    area_sum = 0.0
    zero_area = 0
    tiny_area = 0
    poor_quality = 0
    very_poor_quality = 0
    exact_horizontal = 0
    near_horizontal = 0
    tiny_z_span = 0
    edge_sample = Reservoir(1_000_000, 101)
    area_sample = Reservoir(1_000_000, 102)
    quality_sample = Reservoir(1_000_000, 103)
    z_span_sample = Reservoir(1_000_000, 104)

    for vertices in _chunks(records, chunk_triangles):
        edges = np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 1],
                vertices[:, 0] - vertices[:, 2],
            ),
            axis=1,
        )
        edge_squared = np.einsum("...i,...i->...", edges, edges)
        edge_lengths = np.sqrt(edge_squared)
        double_area = np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        )
        areas = 0.5 * double_area
        # 1.0 is equilateral; values near zero are needle/sliver triangles.
        quality = np.divide(
            2.0 * math.sqrt(3.0) * double_area,
            edge_squared.sum(axis=1),
            out=np.zeros_like(double_area),
            where=edge_squared.sum(axis=1) > 0,
        )
        z_min = vertices[:, :, 2].min(axis=1)
        z_max = vertices[:, :, 2].max(axis=1)
        z_span = z_max - z_min

        area_sum += float(areas.sum())
        zero_area += int(np.count_nonzero(areas == 0.0))
        tiny_area += int(np.count_nonzero(areas <= area_epsilon))
        poor_quality += int(np.count_nonzero(quality < 0.05))
        very_poor_quality += int(np.count_nonzero(quality < 0.005))
        exact_horizontal += int(np.count_nonzero(z_span == 0.0))
        near_horizontal += int(np.count_nonzero(z_span <= z_epsilon))
        tiny_z_span += int(np.count_nonzero(z_span <= 1e-3))
        edge_sample.add(edge_lengths)
        area_sample.add(areas)
        quality_sample.add(quality)
        z_span_sample.add(z_span)

        # Convert each triangle's intersected-level interval to a difference array.
        # The half-open rule avoids counting a vertex touching two layers twice.
        first = np.searchsorted(levels, z_min, side="left")
        after_last = np.searchsorted(levels, z_max, side="left")
        changes = np.bincount(first, minlength=levels.size + 1).astype(np.int64)
        changes -= np.bincount(after_last, minlength=levels.size + 1)
        slice_counts += np.cumsum(changes[:-1])

    elapsed = time.perf_counter() - started
    del records
    return StlSummary(
        path=str(path),
        file_bytes=path.stat().st_size,
        triangle_count=triangle_count,
        read_seconds=elapsed,
        bounds_min=bounds_min.tolist(),
        bounds_max=bounds_max.tolist(),
        extents=extents.tolist(),
        area_sum=area_sum,
        zero_area_count=zero_area,
        tiny_area_count=tiny_area,
        poor_quality_count=poor_quality,
        very_poor_quality_count=very_poor_quality,
        exact_horizontal_count=exact_horizontal,
        near_horizontal_count=near_horizontal,
        tiny_z_span_count=tiny_z_span,
        edge_length_quantiles_mm=edge_sample.quantiles(),
        area_quantiles_mm2=area_sample.quantiles(),
        quality_quantiles=quality_sample.quantiles(),
        z_span_quantiles_mm=z_span_sample.quantiles(),
        slice_levels_mm=levels.tolist(),
        slice_intersection_counts=slice_counts.tolist(),
        slice_count_mean=float(slice_counts.mean()),
        slice_count_max=int(slice_counts.max()),
        slice_count_sum=int(slice_counts.sum()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--chunk-triangles", type=int, default=250_000)
    parser.add_argument("--slice-levels", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = []
    for path in args.paths:
        summary = scan(path.resolve(), args.chunk_triangles, args.slice_levels)
        summaries.append(asdict(summary))
        print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
