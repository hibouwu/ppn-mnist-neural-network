#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    RAW_FIELDS,
    apply_frequency_policy,
    append_csv_rows,
    build_benchmark_command,
    build_child_env,
    command_string,
    fmt_float,
    gflops,
    initialize_csv_output,
    load_json,
    measurement_status,
    parse_benchmark_output,
    restore_governor_if_needed,
    run_process,
    size_to_string,
)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    default_config = script_dir / "config" / "single_thread_blocked_candidates.json"
    default_output = repo_root / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "raw" / "openblas_single_thread_baseline_raw.csv"
    parser = argparse.ArgumentParser(description="Run OpenBLAS single-thread baseline on the configured workloads.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--bench-bin", type=Path, default=None)
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_json(args.config)
    bench_bin = args.bench_bin or (Path(__file__).resolve().parent.parent.parent.parent / config["benchmark"]["binary"])
    protocol = config["execution_protocol"]
    workloads = [
        item
        for group_name in ("main", "sanity")
        for item in config["workloads"].get(group_name, [])
        if item.get("enabled", False)
    ]
    if args.limit_workloads is not None:
        workloads = workloads[:args.limit_workloads]

    if not bench_bin.exists():
        print(f"benchmark binary not found: {bench_bin}", file=sys.stderr)
        return 1

    initialize_csv_output(args.output, RAW_FIELDS, append=args.append)

    threads = int(protocol["threads"])
    pin_core = int(protocol["pin_core"])
    use_taskset = bool(protocol["use_taskset"])
    requested_governor = protocol.get("governor", "performance")
    freq_policy = protocol.get("freq_policy", "detect")
    freq_control = apply_frequency_policy(pin_core, freq_policy, requested_governor)
    if freq_control["error"]:
        print(str(freq_control["error"]), file=sys.stderr)
        return 1
    governor = str(freq_control["active_governor"])
    freq_policy_label = str(freq_control["freq_policy_label"])
    pinning_policy = f"taskset:{pin_core}" if use_taskset else "unbound"
    timing_reps = int(protocol["timing_reps"])
    benchmark_inner_reps = int(protocol.get("benchmark_inner_reps", 1))

    child_env = build_child_env(threads)
    child_env["MATMUL_IMPL"] = config["benchmark"]["baseline_impl"]

    try:
        for workload in workloads:
            m, k, n = workload["Size"]
            size_text = size_to_string(m, k, n)
            print(f"[openblas] workload={workload['WorkloadId']} size={size_text}")
            if args.dry_run:
                continue
            rows = []
            for sample_idx in range(1, timing_reps + 1):
                cmd = build_benchmark_command(bench_bin, m, k, n, benchmark_inner_reps, pin_core, use_taskset)
                proc = run_process(cmd, child_env)
                try:
                    parsed = parse_benchmark_output(proc.stdout + "\n" + proc.stderr)
                except ValueError:
                    parsed = None
                status = measurement_status(proc, parsed)
                row_out = {
                    "Implementation": config["benchmark"]["baseline_impl"],
                    "RunType": "openblas_baseline",
                    "WorkloadFamily": workload["WorkloadFamily"],
                    "WorkloadId": workload["WorkloadId"],
                    "RequestedKernelShape": "",
                    "KernelShape": "",
                    "RequestedMc": "",
                    "Mc": "",
                    "RequestedNc": "",
                    "Nc": "",
                    "RequestedKc": "",
                    "Kc": "",
                    "Size": size_text,
                    "Threads": str(threads),
                    "MeasurementKind": "timing",
                    "SampleIndex": str(sample_idx),
                    "Reps": str(benchmark_inner_reps),
                    "PinningPolicy": pinning_policy,
                    "Governor": governor,
                    "FreqPolicy": freq_policy_label,
                    "Status": status,
                    "Command": command_string(cmd, child_env),
                }
                if parsed is not None:
                    mean_s, stddev_s, _, _ = parsed
                    time_us = mean_s * 1e6
                    row_out["Time_us"] = fmt_float(time_us)
                    row_out["StdDev_us"] = fmt_float(stddev_s * 1e6)
                    row_out["GFLOPS"] = fmt_float(gflops(m, k, n, time_us))
                if status != "ok":
                    row_out["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
                    rows.append(row_out)
                    break
                rows.append(row_out)
            append_csv_rows(args.output, RAW_FIELDS, rows)
        return 0
    finally:
        restore_ok, restore_detail = restore_governor_if_needed(
            pin_core,
            str(freq_control["original_governor"]),
            bool(freq_control["restore_needed"]),
        )
        if not restore_ok:
            print(
                f"warning: failed to restore governor on cpu{pin_core}: {restore_detail}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
