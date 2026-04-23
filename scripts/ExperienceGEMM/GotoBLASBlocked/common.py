#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Iterable


BENCHMARK_RE = re.compile(
    r"Mean:\s*(?P<mean>[0-9.]+)\s*s,\s*StdDev:\s*(?P<stddev>[0-9.]+)\s*s,\s*Reps:\s*(?P<reps>[0-9]+)"
)
RESULT_CHECK_RE = re.compile(r"Result check:\s*(?P<value>[-+0-9.eE]+)")
EFFECTIVE_CONFIG_RE = re.compile(
    r"Effective config:\s*impl=(?P<impl>\S+)"
    r"(?:\s+kernel=(?P<kernel>\S+)\s+mc=(?P<mc>[0-9]+)\s+nc=(?P<nc>[0-9]+)\s+kc=(?P<kc>[0-9]+))?"
)

PERF_EVENT_FIELDS = {
    "instructions": "Instructions",
    "cycles": "Cycles",
    "cache-misses": "CacheMisses",
    "cache-references": "CacheReferences",
    "L1-dcache-loads": "L1DLoads",
    "L1-dcache-load-misses": "L1DMisses",
    "dTLB-loads": "DTLBLoads",
    "dTLB-load-misses": "DTLBMisses",
    "context-switches": "ContextSwitches",
    "cpu-migrations": "CpuMigrations",
}
PERF_EVENTS = list(PERF_EVENT_FIELDS.keys())

RAW_FIELDS = [
    "Implementation",
    "RunType",
    "WorkloadFamily",
    "WorkloadId",
    "RequestedKernelShape",
    "KernelShape",
    "RequestedMc",
    "Mc",
    "RequestedNc",
    "Nc",
    "RequestedKc",
    "Kc",
    "Size",
    "Threads",
    "MeasurementKind",
    "SampleIndex",
    "Time_us",
    "StdDev_us",
    "GFLOPS",
    "Instructions",
    "Instr_StdDev",
    "Cycles",
    "Cycles_StdDev",
    "IPC",
    "IPC_StdDev",
    "CacheMisses",
    "CacheMisses_StdDev",
    "CacheReferences",
    "CacheReferences_StdDev",
    "L1DLoads",
    "L1DLoads_StdDev",
    "L1DMisses",
    "L1DMisses_StdDev",
    "DTLBLoads",
    "DTLBLoads_StdDev",
    "DTLBMisses",
    "DTLBMisses_StdDev",
    "ContextSwitches",
    "ContextSwitches_StdDev",
    "CpuMigrations",
    "CpuMigrations_StdDev",
    "Reps",
    "PinningPolicy",
    "Governor",
    "FreqPolicy",
    "Status",
    "ErrorMessage",
    "Command",
]

AGG_FIELDS = [
    "Implementation",
    "RunType",
    "WorkloadFamily",
    "WorkloadId",
    "KernelShape",
    "Mc",
    "Nc",
    "Kc",
    "Size",
    "Threads",
    "TimingSamples",
    "PerfSamples",
    "MedianTime_us",
    "BestTime_us",
    "TimeStdDev_us",
    "MedianGFLOPS",
    "BestGFLOPS",
    "Instructions",
    "Instr_StdDev",
    "Cycles",
    "Cycles_StdDev",
    "IPC",
    "IPC_StdDev",
    "CacheMisses",
    "CacheMisses_StdDev",
    "CacheReferences",
    "CacheReferences_StdDev",
    "L1DLoads",
    "L1DLoads_StdDev",
    "L1DMisses",
    "L1DMisses_StdDev",
    "DTLBLoads",
    "DTLBLoads_StdDev",
    "DTLBMisses",
    "DTLBMisses_StdDev",
    "ContextSwitches",
    "ContextSwitches_StdDev",
    "CpuMigrations",
    "CpuMigrations_StdDev",
    "ExpectedTimingSamples",
    "ExpectedPerfSamples",
    "AggregationStatus",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_csv_header(path: Path, fields: list[str]) -> None:
    ensure_parent(path)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def append_csv_rows(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    ensure_csv_header(path, fields)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def initialize_csv_output(path: Path, fields: list[str], append: bool) -> None:
    ensure_parent(path)
    if append:
        ensure_csv_header(path, fields)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def size_to_string(m: int, k: int, n: int) -> str:
    return f"{m}x{k}x{n}"


def parse_size(size_text: str) -> tuple[int, int, int]:
    m, k, n = size_text.split("x")
    return int(m), int(k), int(n)


def gflops(m: int, k: int, n: int, time_us: float) -> float:
    if time_us <= 0.0:
        return 0.0
    return (2.0 * m * k * n) / (time_us * 1e-6) / 1e9


def detect_governor(core_id: int) -> str:
    path = governor_path(core_id)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return "unknown"
    return "unknown"


def governor_path(core_id: int) -> Path:
    return Path(f"/sys/devices/system/cpu/cpu{core_id}/cpufreq/scaling_governor")


def set_governor(core_id: int, governor: str) -> tuple[bool, str]:
    path = governor_path(core_id)
    if path.exists():
        try:
            path.write_text(governor, encoding="utf-8")
            observed = detect_governor(core_id)
            if observed == governor:
                return True, f"sysfs:{governor}"
        except OSError as exc:
            last_error = str(exc)
        else:
            last_error = f"sysfs write did not stick (observed {detect_governor(core_id)})"
    else:
        last_error = "scaling_governor path not present"

    cpupower = shutil.which("cpupower")
    if cpupower is not None:
        proc = subprocess.run(
            [cpupower, "frequency-set", "-g", governor],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and detect_governor(core_id) == governor:
            return True, f"cpupower:{governor}"
        last_error = (proc.stderr or proc.stdout or last_error).strip()

    cpufreq_set = shutil.which("cpufreq-set")
    if cpufreq_set is not None:
        proc = subprocess.run(
            [cpufreq_set, "-g", governor],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and detect_governor(core_id) == governor:
            return True, f"cpufreq-set:{governor}"
        last_error = (proc.stderr or proc.stdout or last_error).strip()

    return False, last_error


def apply_frequency_policy(core_id: int, freq_policy: str, requested_governor: str) -> dict[str, str | bool]:
    original_governor = detect_governor(core_id)
    result: dict[str, str | bool] = {
        "original_governor": original_governor,
        "active_governor": original_governor,
        "freq_policy_label": freq_policy,
        "restore_needed": False,
        "error": "",
    }
    if freq_policy == "detect":
        return result
    if original_governor == requested_governor:
        result["freq_policy_label"] = f"{freq_policy}:already_{requested_governor}"
        return result
    if freq_policy == "require":
        result["error"] = (
            f"governor mismatch on cpu{core_id}: expected {requested_governor}, observed {original_governor}"
        )
        return result
    if freq_policy == "set_if_needed":
        ok, detail = set_governor(core_id, requested_governor)
        result["active_governor"] = detect_governor(core_id)
        if ok:
            result["freq_policy_label"] = f"set_if_needed:{detail}"
            result["restore_needed"] = result["active_governor"] != original_governor
            return result
        result["error"] = (
            f"failed to switch governor on cpu{core_id} from {original_governor} to {requested_governor}: {detail}"
        )
        return result
    result["error"] = f"unsupported freq_policy: {freq_policy}"
    return result


def restore_governor_if_needed(core_id: int, original_governor: str, restore_needed: bool) -> tuple[bool, str]:
    if not restore_needed or not original_governor or original_governor == "unknown":
        return True, "no_restore_needed"
    ok, detail = set_governor(core_id, original_governor)
    return ok, detail


def build_child_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["GOTO_NUM_THREADS"] = str(threads)
    env["BLIS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "true")
    env.setdefault("OMP_PLACES", "cores")
    return env


def build_benchmark_command(
    bench_bin: Path,
    m: int,
    k: int,
    n: int,
    reps: int,
    pin_core: int | None,
    use_taskset: bool,
) -> list[str]:
    cmd = [str(bench_bin), str(m), str(k), str(n), str(reps)]
    if use_taskset and pin_core is not None and shutil.which("taskset"):
        return ["taskset", "-c", str(pin_core), *cmd]
    return cmd


def command_string(cmd: list[str], env: dict[str, str]) -> str:
    keys = [
        "MATMUL_IMPL",
        "MATMUL_GOTO_KERNEL",
        "MATMUL_MC",
        "MATMUL_NC",
        "MATMUL_KC",
        "MATMUL_EMIT_EFFECTIVE_CONFIG",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
    ]
    prefix = [f"{key}={env[key]}" for key in keys if key in env]
    return " ".join([*prefix, *cmd])


def run_process(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def parse_benchmark_output(text: str) -> tuple[float, float, int, float | None]:
    match = BENCHMARK_RE.search(text)
    if match is None:
        raise ValueError("failed to parse benchmark output")
    mean_s = float(match.group("mean"))
    stddev_s = float(match.group("stddev"))
    reps = int(match.group("reps"))
    result_check_match = RESULT_CHECK_RE.search(text)
    result_check = None
    if result_check_match is not None:
        result_check = float(result_check_match.group("value"))
    return mean_s, stddev_s, reps, result_check


def parse_effective_config(text: str) -> dict[str, str]:
    match = EFFECTIVE_CONFIG_RE.search(text)
    if match is None:
        return {}
    kernel = match.group("kernel") or ""
    if kernel.startswith("avx2_"):
        kernel = kernel.removeprefix("avx2_")
    out = {
        "Implementation": match.group("impl") or "",
        "KernelShape": kernel,
        "Mc": match.group("mc") or "",
        "Nc": match.group("nc") or "",
        "Kc": match.group("kc") or "",
    }
    return out


def parse_perf_stat(stderr_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        value_text, _, event_name = parts[:3]
        event_name = event_name.removesuffix(":u")
        if event_name not in PERF_EVENT_FIELDS:
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        out[PERF_EVENT_FIELDS[event_name]] = value
    if "Cycles" in out and "Instructions" in out and out["Cycles"] > 0.0:
        out["IPC"] = out["Instructions"] / out["Cycles"]
    return out


def perf_command(base_cmd: list[str]) -> list[str]:
    return ["perf", "stat", "-x,", "-e", ",".join(PERF_EVENTS), *base_cmd]


def is_finite_result(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def measurement_status(proc: subprocess.CompletedProcess[str], parsed: tuple[float, float, int, float | None] | None) -> str:
    if proc.returncode != 0:
        return "failed"
    if parsed is None:
        return "failed"
    mean_s, _, _, result_check = parsed
    if not math.isfinite(mean_s) or mean_s <= 0.0:
        return "invalid"
    if not is_finite_result(result_check):
        return "invalid"
    return "ok"


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def median_or_blank(values: list[float]) -> str:
    return fmt_float(statistics.median(values)) if values else ""


def min_or_blank(values: list[float]) -> str:
    return fmt_float(min(values)) if values else ""


def stdev_or_blank(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    return fmt_float(statistics.stdev(values))
