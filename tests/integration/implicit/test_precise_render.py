from __future__ import annotations

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.precise_render import (
    PreciseRenderCamera,
    PreciseRenderMaterial,
    PreciseRenderSettings,
    render_precise_implicit,
)


def _sphere_body(counter: dict[str, int]) -> ImplicitBody:
    def evaluate(points: np.ndarray, _stage_reporter=None) -> np.ndarray:
        counter["points"] += len(points)
        return np.linalg.norm(points, axis=1) - 1.0

    return ImplicitBody(
        name="unit sphere",
        bounds=np.array([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]),
        evaluate=evaluate,
    )


def test_precise_render_directly_evaluates_body_and_returns_requested_image() -> None:
    counter = {"points": 0}
    result = render_precise_implicit(
        _sphere_body(counter),
        camera=PreciseRenderCamera(
            position=(0.0, -4.0, 0.0),
            focal_point=(0.0, 0.0, 0.0),
            view_up=(0.0, 0.0, 1.0),
            parallel_projection=True,
            parallel_scale=1.35,
            view_angle_deg=30.0,
        ),
        settings=PreciseRenderSettings(
            width_px=96,
            height_px=72,
            minimum_feature_mm=0.25,
            tile_size_px=24,
        ),
        material=PreciseRenderMaterial(
            color=(0.62, 0.68, 0.74),
            metallic=0.0,
            roughness=0.55,
        ),
    )

    assert result.image_rgb.shape == (72, 96, 3)
    assert result.image_rgb.dtype == np.uint8
    assert counter["points"] > 0
    assert result.report.hit_pixels > 1200
    assert result.report.hit_pixels < 7000
    assert result.report.output_size_px == (96, 72)
    assert result.report.pixel_footprint_mm == 1.35 / 72
    assert result.report.surface_tolerance_mm < 0.02
    assert np.mean(result.image_rgb[36, 48]) < 245.0
    assert np.mean(result.image_rgb[0, 0]) > 245.0


def test_precise_render_uses_bounded_tiles_and_resolution_changes_sampling() -> None:
    calls: list[int] = []

    def evaluate(points: np.ndarray, _stage_reporter=None) -> np.ndarray:
        calls.append(len(points))
        return np.linalg.norm(points, axis=1) - 1.0

    body = ImplicitBody(
        name="sphere",
        bounds=np.array([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]),
        evaluate=evaluate,
    )
    common = dict(
        camera=PreciseRenderCamera(
            position=(0.0, -4.0, 0.0),
            focal_point=(0.0, 0.0, 0.0),
            view_up=(0.0, 0.0, 1.0),
            parallel_projection=True,
            parallel_scale=1.35,
            view_angle_deg=30.0,
        ),
        material=PreciseRenderMaterial(),
    )
    small = render_precise_implicit(
        body,
        settings=PreciseRenderSettings(
            width_px=80,
            height_px=60,
            minimum_feature_mm=0.25,
            tile_size_px=16,
        ),
        **common,
    )
    large = render_precise_implicit(
        body,
        settings=PreciseRenderSettings(
            width_px=160,
            height_px=120,
            minimum_feature_mm=0.25,
            tile_size_px=16,
        ),
        **common,
    )

    assert large.report.pixel_footprint_mm == small.report.pixel_footprint_mm / 2.0
    assert large.report.surface_tolerance_mm < small.report.surface_tolerance_mm
    assert large.report.max_resident_rays <= 16 * 16
    assert large.report.tile_count == 80
    assert max(calls) <= 16 * 16 * 16
