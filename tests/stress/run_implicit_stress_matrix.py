"""Run high-risk implicit workflows in monitored child processes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("implicit_pipeline_worker.py")
SMOKE_SCENARIOS = (
    "custom_smoke",
    "transition_smoke",
    "shell_fusion_smoke",
)
HEAVY_SCENARIOS = (
    "custom_large",
    "custom_reconstruct_chunked",
    "transition_custom_large",
    "shell_fusion_large",
    "memory_dense",
    "memory_chunked",
)


NATIVE_EXIT_CODES = {
    0xC0000005: "access_violation",
    0xC0000017: "no_memory",
    0xC0000094: "integer_divide_by_zero",
    0xC00000FD: "stack_overflow",
    0xC0000409: "stack_buffer_overrun",
}
ERROR_PATTERNS = (
    "QThread: Destroyed while thread is still running",
    "access violation",
    "RecursionError",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA out of memory",
    "bad allocation",
    "MemoryError",
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    status: str
    exit_code: int | None
    exit_reason: str | None
    elapsed_s: float
    peak_rss_mib: float
    peak_vram_mib: float | None
    timeout_s: float
    memory_limit_mib: float | None
    matched_errors: tuple[str, ...]
    worker_report: dict[str, object] | None
    stdout_path: str
    stderr_path: str


def decode_exit_reason(return_code: int | None) -> str | None:
    if return_code is None or return_code == 0:
        return None
    unsigned = return_code & 0xFFFFFFFF
    return NATIVE_EXIT_CODES.get(unsigned, f"exit_{unsigned:#010x}")


def select_scenarios(profile: str, requested: list[str] | None) -> tuple[str, ...]:
    available = SMOKE_SCENARIOS + HEAVY_SCENARIOS
    if requested:
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"unknown scenarios: {', '.join(unknown)}")
        return tuple(dict.fromkeys(requested))
    if profile == "smoke":
        return SMOKE_SCENARIOS
    if profile == "heavy":
        return HEAVY_SCENARIOS
    return available


def _process_tree_rss(process: psutil.Process) -> int:
    total = 0
    for member in (process, *process.children(recursive=True)):
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _gpu_memory_mib(pid: int) -> float | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    total = 0.0
    found = False
    for line in completed.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            row_pid, used = int(columns[0]), float(columns[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += used
            found = True
    return total if found else None


def _terminate_tree(process: psutil.Process) -> None:
    members = process.children(recursive=True)
    for member in reversed(members):
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        return
    _, alive = psutil.wait_procs([*members, process], timeout=3.0)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass


def run_scenario(
    scenario: str,
    output_dir: Path,
    *,
    timeout_s: float,
    memory_limit_mib: float | None,
    custom_cell: Path | None,
    domain_stl: Path | None,
    monitor_gpu: bool,
) -> ScenarioResult:
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    report_path = scenario_dir / "worker_report.json"
    stdout_path = scenario_dir / "stdout.log"
    stderr_path = scenario_dir / "stderr.log"
    command = [
        sys.executable,
        str(WORKER),
        "--scenario",
        scenario,
        "--output",
        str(report_path),
    ]
    if custom_cell is not None:
        command.extend(("--custom-cell", str(custom_cell)))
    if domain_stl is not None:
        command.extend(("--domain-stl", str(domain_stl)))

    started = time.perf_counter()
    peak_rss = 0
    peak_vram: float | None = None
    forced_status: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        process = psutil.Process(child.pid)
        next_gpu_sample = 0.0
        while child.poll() is None:
            elapsed = time.perf_counter() - started
            try:
                peak_rss = max(peak_rss, _process_tree_rss(process))
            except psutil.NoSuchProcess:
                pass
            if monitor_gpu and elapsed >= next_gpu_sample:
                current_vram = _gpu_memory_mib(child.pid)
                if current_vram is not None:
                    peak_vram = max(peak_vram or 0.0, current_vram)
                next_gpu_sample = elapsed + 1.0
            if memory_limit_mib is not None and peak_rss > memory_limit_mib * 1024**2:
                forced_status = "memory_limit"
                _terminate_tree(process)
                break
            if elapsed > timeout_s:
                forced_status = "timeout"
                _terminate_tree(process)
                break
            time.sleep(0.1)
        try:
            return_code = child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            forced_status = forced_status or "termination_timeout"
            _terminate_tree(process)
            return_code = child.wait(timeout=5.0)
    elapsed = time.perf_counter() - started
    worker_report = None
    if report_path.is_file():
        try:
            worker_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            worker_report = None
    combined_errors = (
        stdout_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + stderr_path.read_text(encoding="utf-8", errors="replace")
    )
    matched = tuple(pattern for pattern in ERROR_PATTERNS if pattern in combined_errors)
    if forced_status is not None:
        status = forced_status
    elif return_code != 0:
        status = "crashed" if decode_exit_reason(return_code) in NATIVE_EXIT_CODES.values() else "failed"
    elif worker_report is None:
        status = "missing_report"
    elif worker_report.get("status") != "passed":
        status = "failed"
    elif matched:
        status = "failed"
    else:
        status = "passed"
    return ScenarioResult(
        scenario=scenario,
        status=status,
        exit_code=return_code,
        exit_reason=decode_exit_reason(return_code),
        elapsed_s=elapsed,
        peak_rss_mib=peak_rss / 1024**2,
        peak_vram_mib=peak_vram,
        timeout_s=timeout_s,
        memory_limit_mib=memory_limit_mib,
        matched_errors=matched,
        worker_report=worker_report,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run crash- and memory-isolated implicit workflow stress tests."
    )
    parser.add_argument("--profile", choices=("smoke", "heavy", "all"), default="smoke")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--custom-cell", type=Path)
    parser.add_argument("--domain-stl", type=Path)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=0.0,
        help="terminate only the offending child above this RSS limit; 0 only monitors",
    )
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path, label in ((args.custom_cell, "custom cell"), (args.domain_stl, "domain STL")):
        if path is not None and not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    if args.memory_limit_gib < 0.0:
        parser.error("--memory-limit-gib cannot be negative")
    try:
        scenarios = select_scenarios(args.profile, args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "build" / "test-artifacts" / "stress" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_limit_mib = args.memory_limit_gib * 1024.0 or None
    results = []
    for scenario in scenarios:
        print(f"[stress] starting {scenario}", flush=True)
        result = run_scenario(
            scenario,
            output_dir,
            timeout_s=args.timeout_s,
            memory_limit_mib=memory_limit_mib,
            custom_cell=args.custom_cell,
            domain_stl=args.domain_stl,
            monitor_gpu=not args.no_gpu_monitor,
        )
        results.append(result)
        print(
            f"[stress] {scenario}: {result.status}; "
            f"{result.elapsed_s:.1f}s; RSS {result.peak_rss_mib:.0f} MiB",
            flush=True,
        )
    payload = {
        "profile": args.profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "custom_cell": str(args.custom_cell) if args.custom_cell else None,
        "domain_stl": str(args.domain_stl) if args.domain_stl else None,
        "results": [asdict(result) for result in results],
        "summary": {
            "passed": sum(result.status == "passed" for result in results),
            "failed": sum(result.status != "passed" for result in results),
            "total": len(results),
        },
    }
    report_path = output_dir / "matrix_report.json"
    _write_report(report_path, payload)
    print(f"[stress] report: {report_path}", flush=True)
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
