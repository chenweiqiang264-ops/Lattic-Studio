"""Mesh validation, repair, and conservative fragment cleanup."""

from __future__ import annotations

import numpy as np
import trimesh

from lattice_studio.domain.cell_map import CellMap
from lattice_studio.engine.implicit.field import SampledImplicitField
from lattice_studio.engine.contracts import (
    FieldFragmentCleanupReport,
    MeshQuality,
    NumericalFragmentCleanupReport,
)

def inspect_mesh(mesh: trimesh.Trimesh) -> MeshQuality:
    """Inspect a mesh without modifying it or assuming it is already valid."""

    try:
        finite = bool(np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all())
    except (TypeError, ValueError):
        finite = False
    non_empty = bool(len(mesh.vertices) > 0 and len(mesh.faces) > 0)
    if not finite or not non_empty:
        return MeshQuality(finite, non_empty, False, False, 0, "网格为空或包含非有限数据")
    try:
        watertight = bool(mesh.is_watertight)
        winding = bool(mesh.is_winding_consistent)
        components = int(mesh.body_count)
    except (TypeError, ValueError, RuntimeError) as exc:
        return MeshQuality(finite, non_empty, False, False, 0, str(exc))
    return MeshQuality(finite, non_empty, watertight, winding, components)


def safe_mesh_volume(mesh: trimesh.Trimesh) -> float | None:
    """Return a reliable signed volume, or ``None`` for invalid topology."""

    try:
        if not mesh.is_watertight:
            return None
        volume = float(mesh.volume)
    except (AttributeError, TypeError, ValueError, RuntimeError, ZeroDivisionError):
        return None
    return volume if np.isfinite(volume) else None


def remove_numerical_fragments(
    mesh: trimesh.Trimesh,
    tolerance_mm: float,
    relative_volume_limit: float = 1.0e-6,
    *,
    component_extent_limit_mm: float | None = None,
) -> tuple[trimesh.Trimesh, NumericalFragmentCleanupReport]:
    """Remove disconnected numerical or physically tiny components.

    A component is treated as a Marching Cubes numerical fragment only when its
    volume is no larger than ``tolerance_mm ** 3`` and no larger than the
    selected fraction of the largest closed component.  When
    ``component_extent_limit_mm`` is provided, a second conservative rule is
    enabled for STL post-processing: a component must also fit inside that
    physical size and remain below the selected relative volume (or surface
    area for an open component).  The largest component is always retained.
    Face count is reported but never used as a criterion.
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    tolerance = float(tolerance_mm)
    relative_limit = float(relative_volume_limit)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance_mm must be finite and positive")
    if relative_limit <= 0.0 or not np.isfinite(relative_limit):
        raise ValueError("relative_volume_limit must be finite and positive")
    if component_extent_limit_mm is not None:
        extent_limit = float(component_extent_limit_mm)
        if extent_limit <= 0.0 or not np.isfinite(extent_limit):
            raise ValueError(
                "component_extent_limit_mm must be finite and positive or None"
            )
    else:
        extent_limit = None

    volume_limit = tolerance**3
    try:
        component_count = int(mesh.body_count)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        component_count = len(mesh.split(only_watertight=False))
    if component_count <= 1:
        report = NumericalFragmentCleanupReport(
            0,
            0,
            0.0,
            component_count,
            volume_limit,
            relative_limit,
            "No disconnected numerical fragments were found.",
        )
        return mesh, report

    if mesh.is_watertight:
        labels = trimesh.graph.connected_component_labels(
            mesh.face_adjacency,
            node_count=len(mesh.faces),
        )
        component_count = int(labels.max()) + 1
        component_volumes = np.zeros(component_count, dtype=np.float64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        volume_batch_faces = 250_000
        component_mins = None
        component_maxs = None
        if extent_limit is not None:
            component_mins = np.full(
                (component_count, 3), np.inf, dtype=np.float64
            )
            component_maxs = np.full(
                (component_count, 3), -np.inf, dtype=np.float64
            )
        for start in range(0, len(faces), volume_batch_faces):
            end = min(start + volume_batch_faces, len(faces))
            batch_faces = faces[start:end]
            first = vertices[batch_faces[:, 0]]
            second = vertices[batch_faces[:, 1]]
            third = vertices[batch_faces[:, 2]]
            signed_volume = np.einsum(
                "ij,ij->i",
                first,
                np.cross(second, third),
                optimize=True,
            ) / 6.0
            component_volumes += np.bincount(
                labels[start:end],
                weights=signed_volume,
                minlength=component_count,
            )
            if component_mins is not None and component_maxs is not None:
                batch_points = vertices[batch_faces].reshape(-1, 3)
                batch_labels = np.repeat(labels[start:end], 3)
                for axis in range(3):
                    np.minimum.at(
                        component_mins[:, axis],
                        batch_labels,
                        batch_points[:, axis],
                    )
                    np.maximum.at(
                        component_maxs[:, axis],
                        batch_labels,
                        batch_points[:, axis],
                    )
        volumes = np.abs(component_volumes)
        main_index = int(np.argmax(volumes))
        main_volume = float(volumes[main_index])
        removed_mask = (
            (np.arange(component_count) != main_index)
            & (volumes <= volume_limit)
            & (volumes <= main_volume * relative_limit)
        )
        if (
            extent_limit is not None
            and component_mins is not None
            and component_maxs is not None
        ):
            extents = component_maxs - component_mins
            tiny_physical = np.isfinite(extents).all(axis=1) & (
                np.max(extents, axis=1) <= extent_limit
            )
            removed_mask |= (
                (np.arange(component_count) != main_index)
                & tiny_physical
                & (volumes <= main_volume * relative_limit)
            )
        removed_indices = np.flatnonzero(removed_mask)
        if not len(removed_indices):
            report = NumericalFragmentCleanupReport(
                0,
                0,
                0.0,
                component_count,
                volume_limit,
                relative_limit,
                "Disconnected components exceed the numerical-fragment limits; the mesh was unchanged.",
            )
            return mesh, report
        removed_faces = int(np.count_nonzero(removed_mask[labels]))
        cleaned = trimesh.Trimesh(
            vertices=vertices.copy(),
            faces=faces[~removed_mask[labels]],
            process=False,
        )
        cleaned = repair_mesh(cleaned, fill_holes=False)
        removed_volume = float(np.sum(volumes[removed_indices]))
        report = NumericalFragmentCleanupReport(
            len(removed_indices),
            removed_faces,
            removed_volume,
            component_count - len(removed_indices),
            volume_limit,
            relative_limit,
            (
                f"Removed {len(removed_indices)} closed numerical fragment(s), "
                f"{removed_faces} faces and {removed_volume:.6g} mm^3."
            ),
        )
        return cleaned, report

    components = list(mesh.split(only_watertight=False))

    volumes: list[float | None] = []
    for component in components:
        volume = safe_mesh_volume(component)
        volumes.append(abs(volume) if volume is not None else None)
    closed_indices = [
        index
        for index, volume in enumerate(volumes)
        if volume is not None and volume > 0.0
    ]
    if not closed_indices:
        report = NumericalFragmentCleanupReport(
            0,
            0,
            0.0,
            len(components),
            volume_limit,
            relative_limit,
            "No closed reference component was available; the mesh was unchanged.",
        )
        return mesh, report

    main_index = max(closed_indices, key=lambda index: float(volumes[index]))
    main_volume = float(volumes[main_index])
    removed_indices = {
        index
        for index, volume in enumerate(volumes)
        if index != main_index
        and volume is not None
        and volume <= volume_limit
        and volume <= main_volume * relative_limit
    }
    if extent_limit is not None:
        main_area = float(components[main_index].area)
        for index, component in enumerate(components):
            if index == main_index:
                continue
            extent = np.ptp(np.asarray(component.vertices, dtype=np.float64), axis=0)
            if (
                np.isfinite(extent).all()
                and float(np.max(extent)) <= extent_limit
                and float(component.area) <= main_area * relative_limit
            ):
                removed_indices.add(index)
    if not removed_indices:
        report = NumericalFragmentCleanupReport(
            0,
            0,
            0.0,
            len(components),
            volume_limit,
            relative_limit,
            "Disconnected components exceed the numerical-fragment limits; the mesh was unchanged.",
        )
        return mesh, report

    retained = [
        component
        for index, component in enumerate(components)
        if index not in removed_indices
    ]
    cleaned = retained[0].copy() if len(retained) == 1 else trimesh.util.concatenate(retained)
    cleaned = repair_mesh(cleaned, fill_holes=False)
    removed_faces = sum(len(components[index].faces) for index in removed_indices)
    removed_volume = sum(float(volumes[index]) for index in removed_indices)
    report = NumericalFragmentCleanupReport(
        len(removed_indices),
        removed_faces,
        removed_volume,
        len(retained),
        volume_limit,
        relative_limit,
        (
            f"Removed {len(removed_indices)} closed numerical fragment(s), "
            f"{removed_faces} faces and {removed_volume:.6g} mm^3."
        ),
    )
    return cleaned, report


def repair_mesh(mesh: trimesh.Trimesh, fill_holes: bool = True) -> trimesh.Trimesh:
    """Apply conservative, deterministic repairs and return a new mesh.

    This can repair duplicate vertices/faces, degenerate faces, unreferenced
    vertices, winding and simple boundary holes.  It deliberately does not
    invent geometry for large or ambiguous holes; callers must inspect the
    returned mesh again and decide whether the STL is suitable for SDF use.
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("cannot repair a mesh with non-finite vertex coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must have shape (M, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("mesh faces must use integer vertex indices")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("mesh faces reference vertices outside the vertex array")

    def filter_faces(
        candidate_vertices: np.ndarray,
        candidate_faces: np.ndarray,
    ) -> np.ndarray:
        """Remove repeated-index, near-degenerate, and duplicate triangles."""

        valid = np.logical_and.reduce(
            (
                candidate_faces[:, 0] != candidate_faces[:, 1],
                candidate_faces[:, 1] != candidate_faces[:, 2],
                candidate_faces[:, 0] != candidate_faces[:, 2],
            )
        )
        if np.any(valid):
            indices = np.flatnonzero(valid)
            valid[indices] &= trimesh.triangles.nondegenerate(
                candidate_vertices[candidate_faces[indices]]
            )
        kept = candidate_faces[valid]
        if not len(kept):
            return kept
        _, first_indices = np.unique(
            np.sort(kept, axis=1),
            axis=0,
            return_index=True,
        )
        return kept[np.sort(first_indices)]

    # ``Trimesh.update_faces`` asks its mutable cache whether it is empty.  A
    # malformed cache can recurse there indefinitely, so reconstruct a clean
    # mesh from arrays rather than mutating the input mesh in place.
    filtered_faces = filter_faces(vertices, faces)
    if len(filtered_faces):
        referenced = np.unique(filtered_faces.reshape(-1))
        compact_vertices = vertices[referenced]
        initial_remap = np.full(len(vertices), -1, dtype=np.int64)
        initial_remap[referenced] = np.arange(len(referenced), dtype=np.int64)
        compact_faces = initial_remap[filtered_faces]

        # Match Trimesh's default positional merge tolerance without touching
        # the original mesh cache.  Preserve the first source occurrence so
        # face ordering remains deterministic for diagnostics and export.
        digits = trimesh.util.decimal_to_digits(trimesh.constants.tol.merge)
        rounded_vertices = np.round(compact_vertices, decimals=digits)
        _, first_indices, inverse = np.unique(
            rounded_vertices,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        occurrence_order = np.argsort(first_indices)
        inverse_order = np.empty(len(occurrence_order), dtype=np.int64)
        inverse_order[occurrence_order] = np.arange(len(occurrence_order))
        merged_vertices = compact_vertices[first_indices[occurrence_order]]
        merged_faces = inverse_order[inverse][compact_faces]
        merged_faces = filter_faces(merged_vertices, merged_faces)
        if len(merged_faces):
            referenced = np.unique(merged_faces.reshape(-1))
            final_remap = np.full(len(merged_vertices), -1, dtype=np.int64)
            final_remap[referenced] = np.arange(len(referenced), dtype=np.int64)
            repaired = trimesh.Trimesh(
                vertices=merged_vertices[referenced],
                faces=final_remap[merged_faces],
                process=False,
            )
        else:
            repaired = trimesh.Trimesh(process=False)
    else:
        repaired = trimesh.Trimesh(process=False)

    if fill_holes and len(repaired.faces):
        trimesh.repair.fill_holes(repaired)
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fix_inversion(repaired)
    repaired.fix_normals()
    return repaired


def recommended_floating_component_extent_mm(
    cell_map: CellMap,
    tolerance_mm: float,
    minimum_feature_mm: float,
) -> float:
    """Return the physical size ceiling for automatic island cleanup.

    The ceiling is tied to the extraction scale, the smallest Cell Map period,
    and the smallest generated feature.  It is deliberately a size ceiling,
    not a face-count rule, so a real lattice region is not removed merely
    because it has fewer triangles.
    """

    if not isinstance(cell_map, CellMap):
        raise TypeError("cell_map must be a CellMap")
    tolerance = float(tolerance_mm)
    feature = float(minimum_feature_mm)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance_mm must be finite and positive")
    if feature <= 0.0 or not np.isfinite(feature):
        raise ValueError("minimum_feature_mm must be finite and positive")
    period = float(np.min(np.asarray(cell_map.spacing_mm, dtype=np.float64)))
    return max(4.0 * tolerance, 0.5 * period, 4.0 * feature)


def remove_small_negative_field_components(
    field: np.ndarray,
    spacing_mm: np.ndarray | tuple[float, float, float],
    tolerance_mm: float,
    relative_volume_limit: float = 0.02,
    *,
    component_extent_limit_mm: float | None = None,
) -> tuple[np.ndarray, FieldFragmentCleanupReport]:
    """Remove only physically small disconnected negative field components.

    The field uses the project convention that negative values are material.
    Components are classified on a 26-neighbour voxel graph and are removed
    only when they are clearly smaller than the main material component and
    fit inside the optional physical extent limit.  Boundary-touching
    components are retained because they may be valid clipped geometry.
    """

    values = np.asarray(field, dtype=np.float32)
    spacing = np.asarray(spacing_mm, dtype=np.float64).reshape(-1)
    tolerance = float(tolerance_mm)
    relative_limit = float(relative_volume_limit)
    if values.ndim != 3 or any(size < 2 for size in values.shape):
        raise ValueError("field must be a three-dimensional grid with at least 2 samples per axis")
    if not np.isfinite(values).all():
        raise ValueError("field contains non-finite values")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("spacing_mm must contain three finite positive values")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance_mm must be finite and positive")
    if relative_limit <= 0.0 or not np.isfinite(relative_limit):
        raise ValueError("relative_volume_limit must be finite and positive")
    if component_extent_limit_mm is not None:
        extent_limit = float(component_extent_limit_mm)
        if extent_limit <= 0.0 or not np.isfinite(extent_limit):
            raise ValueError(
                "component_extent_limit_mm must be finite and positive or None"
            )
    else:
        extent_limit = None

    from scipy import ndimage

    occupied = values < 0.0
    labels, component_count = ndimage.label(
        occupied,
        structure=ndimage.generate_binary_structure(3, 3),
        output=np.int32,
    )
    sizes = np.bincount(labels.ravel(), minlength=int(component_count) + 1)
    voxel_volume = float(np.prod(spacing))
    volume_limit = tolerance**3
    if component_count <= 1:
        return values, FieldFragmentCleanupReport(
            0,
            0,
            0.0,
            int(component_count),
            volume_limit,
            relative_limit,
            extent_limit,
            "No disconnected negative field components were found.",
        )

    main_label = int(np.argmax(sizes[1:]) + 1)
    main_volume = float(sizes[main_label]) * voxel_volume
    relative_threshold = main_volume * relative_limit
    removable = np.zeros(len(sizes), dtype=bool)
    slices = ndimage.find_objects(labels)
    for label in range(1, len(sizes)):
        if label == main_label or sizes[label] == 0:
            continue
        component_slice = slices[label - 1]
        if component_slice is None:
            continue
        if any(
            component_slice[axis].start == 0
            or component_slice[axis].stop == values.shape[axis]
            for axis in range(3)
        ):
            continue
        component_volume = float(sizes[label]) * voxel_volume
        if component_volume > relative_threshold:
            continue
        if component_volume <= volume_limit:
            removable[label] = True
            continue
        if extent_limit is None:
            continue
        extent = np.asarray(
            [component_slice[axis].stop - component_slice[axis].start for axis in range(3)],
            dtype=np.float64,
        ) * spacing
        if float(np.max(extent)) <= extent_limit:
            removable[label] = True

    removed_labels = np.flatnonzero(removable)
    if not len(removed_labels):
        return values, FieldFragmentCleanupReport(
            0,
            0,
            0.0,
            int(component_count),
            volume_limit,
            relative_limit,
            extent_limit,
            "Disconnected negative components exceed the physical cleanup limits; the field was unchanged.",
        )

    cleaned = values.copy()
    positive_epsilon = np.float32(
        max(float(np.max(np.abs(values), initial=0.0)), 1.0) * np.finfo(np.float32).eps * 8.0
    )
    removed_voxels = 0
    for start in range(0, values.shape[0], 32):
        stop = min(start + 32, values.shape[0])
        selected = removable[labels[start:stop]]
        removed_voxels += int(np.count_nonzero(selected))
        slab = cleaned[start:stop]
        slab[selected] = np.maximum(np.abs(slab[selected]), positive_epsilon)
    removed_volume = float(np.sum(sizes[removed_labels])) * voxel_volume
    return cleaned, FieldFragmentCleanupReport(
        len(removed_labels),
        removed_voxels,
        removed_volume,
        int(component_count - len(removed_labels)),
        volume_limit,
        relative_limit,
        extent_limit,
        (
            f"Removed {len(removed_labels)} small negative field component(s), "
            f"{removed_voxels} voxels and {removed_volume:.6g} mm^3."
        ),
    )


def _clean_display_field_components(
    field: SampledImplicitField,
    cell_map: CellMap,
    minimum_feature_mm: float,
) -> tuple[SampledImplicitField, FieldFragmentCleanupReport]:
    """Remove physically tiny display islands without changing the evaluator."""

    display_spacing = np.asarray(field.spacing, dtype=np.float64)
    minimum_period = float(np.min(np.asarray(cell_map.spacing_mm, dtype=np.float64)))
    extent_limit = max(
        4.0 * float(np.max(display_spacing)),
        0.5 * minimum_period,
        4.0 * float(minimum_feature_mm),
    )
    values, report = remove_small_negative_field_components(
        field.values,
        display_spacing,
        float(np.max(display_spacing)),
        relative_volume_limit=0.02,
        component_extent_limit_mm=extent_limit,
    )
    cleaned = SampledImplicitField(
        name=field.name,
        values=values,
        origin=field.origin,
        spacing=field.spacing,
        color=field.color,
        is_preview_only=field.is_preview_only,
        metallic=field.metallic,
        roughness=field.roughness,
    )
    return cleaned, report


