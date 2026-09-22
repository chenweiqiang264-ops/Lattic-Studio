"""Fast regression tests for the subprocess stress-test controller."""

from __future__ import annotations

from tests.stress import run_implicit_stress_matrix as matrix


def test_stress_profiles_cover_requested_high_risk_workflows() -> None:
    heavy = matrix.select_scenarios("heavy", None)

    assert "custom_large" in heavy
    assert "custom_reconstruct_chunked" in heavy
    assert "transition_custom_large" in heavy
    assert "shell_fusion_large" in heavy
    assert "memory_dense" in heavy
    assert "memory_chunked" in heavy


def test_explicit_stress_scenarios_are_deduplicated_in_order() -> None:
    selected = matrix.select_scenarios(
        "smoke",
        ["shell_fusion_smoke", "custom_smoke", "shell_fusion_smoke"],
    )

    assert selected == ("shell_fusion_smoke", "custom_smoke")


def test_windows_native_crash_codes_are_decoded() -> None:
    assert matrix.decode_exit_reason(-1073741819) == "access_violation"
    assert matrix.decode_exit_reason(-1073741571) == "stack_overflow"
    assert matrix.decode_exit_reason(1) == "exit_0x00000001"

