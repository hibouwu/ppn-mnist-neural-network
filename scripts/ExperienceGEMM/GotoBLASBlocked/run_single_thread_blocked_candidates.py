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
    parse_effective_config,
    parse_perf_stat,
    perf_command,
    restore_governor_if_needed,
    run_process,
    size_to_string,
)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    default_config = script_dir / "config" / "single_thread_blocked_candidates.json"
    default_output = repo_root / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "raw" / "single_thread_blocked_candidates_raw.csv"
    parser = argparse.ArgumentParser(description="Run single-thread blocked candidate screening.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--bench-bin", type=Path, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def iter_workloads(config: dict, limit: int | None):
    workloads = [
        item
        for group_name in ("main", "sanity")
        for item in config["workloads"].get(group_name, [])
        if item.get("enabled", False)
    ]
    if limit is not None:
        workloads = workloads[:limit]
    return workloads


def main() -> int:
    args = build_parser().parse_args()
    config = load_json(args.config)
    bench_bin = args.bench_bin or (Path(__file__).resolve().parent.parent.parent.parent / config["benchmark"]["binary"])
    protocol = config["execution_protocol"]
    workloads = iter_workloads(config, args.limit_workloads)

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
    perf_reps = int(protocol["perf_reps"])
    benchmark_inner_reps = int(protocol.get("benchmark_inner_reps", 1))
    run_perf = bool(protocol.get("enable_perf", True)) and not args.skip_perf

    child_env = build_child_env(threads)
    child_env["MATMUL_IMPL"] = config["benchmark"]["custom_impl"]
    child_env["MATMUL_EMIT_EFFECTIVE_CONFIG"] = "1"

    try:
        candidate_counter = 0
        for kernel_group in config["kernel_rows"]:
            kernel_shape = kernel_group["KernelShape"]
            supported = bool(kernel_group.get("current_binary_supported", False))
            for row in kernel_group["rows"]:
                kc = int(row["Kc"])
                for mc in row["Mc"]:
                    for nc in row["Nc"]:
                        candidate_counter += 1
                        if args.limit_candidates is not None and candidate_counter > args.limit_candidates:
                            return 0
                        for workload in workloads:
                            m, k, n = workload["Size"]
                            size_text = size_to_string(m, k, n)
                            if not supported:
                                rows = [{
                                    "Implementation": config["benchmark"]["custom_impl"],
                                    "RunType": "custom",
                                    "WorkloadFamily": workload["WorkloadFamily"],
                                    "WorkloadId": workload["WorkloadId"],
                                    "RequestedKernelShape": kernel_shape,
                                    "KernelShape": "",
                                    "RequestedMc": str(mc),
                                    "Mc": "",
                                    "RequestedNc": str(nc),
                                    "Nc": "",
                                    "RequestedKc": str(kc),
                                    "Kc": "",
                                    "Size": size_text,
                                    "Threads": str(threads),
                                    "MeasurementKind": "meta",
                                    "SampleIndex": "0",
                                    "Reps": "0",
                                    "PinningPolicy": pinning_policy,
                                    "Governor": governor,
                                    "FreqPolicy": freq_policy_label,
                                    "Status": "unsupported_kernel_shape",
                                    "ErrorMessage": kernel_group.get("support_note", "kernel shape not runtime-switchable in current binary"),
                                }]
                                append_csv_rows(args.output, RAW_FIELDS, rows)
                                continue

                            child_env["MATMUL_MC"] = str(mc)
                            child_env["MATMUL_NC"] = str(nc)
                            child_env["MATMUL_KC"] = str(kc)
                            child_env["MATMUL_GOTO_KERNEL"] = kernel_shape

                            print(
                                f"[custom] workload={workload['WorkloadId']} kernel={kernel_shape} "
                                f"mc_nc_kc=({mc},{nc},{kc})"
                            )

                            if not args.dry_run:
                                timing_rows = []
                                candidate_failed = False
                                for sample_idx in range(1, timing_reps + 1):
                                    cmd = build_benchmark_command(
                                        bench_bin, m, k, n, benchmark_inner_reps, pin_core, use_taskset
                                    )
                                    proc = run_process(cmd, child_env)
                                    try:
                                        parsed = parse_benchmark_output(proc.stdout + "\n" + proc.stderr)
                                    except ValueError:
                                        parsed = None
                                    status = measurement_status(proc, parsed)
                                    row_out = {
                                        "Implementation": config["benchmark"]["custom_impl"],
                                        "RunType": "custom",
                                        "WorkloadFamily": workload["WorkloadFamily"],
                                        "WorkloadId": workload["WorkloadId"],
                                        "RequestedKernelShape": kernel_shape,
                                        "KernelShape": "",
                                        "RequestedMc": str(mc),
                                        "Mc": "",
                                        "RequestedNc": str(nc),
                                        "Nc": "",
                                        "RequestedKc": str(kc),
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
                                    effective = parse_effective_config(proc.stdout + "\n" + proc.stderr)
                                    row_out["KernelShape"] = effective.get("KernelShape", kernel_shape)
                                    row_out["Mc"] = effective.get("Mc", str(mc))
                                    row_out["Nc"] = effective.get("Nc", str(nc))
                                    row_out["Kc"] = effective.get("Kc", str(kc))
                                    if status != "ok":
                                        row_out["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
                                        timing_rows.append(row_out)
                                        candidate_failed = True
                                        break
                                    timing_rows.append(row_out)
                                append_csv_rows(args.output, RAW_FIELDS, timing_rows)
                                if candidate_failed:
                                    continue

                                if run_perf:
                                    perf_rows = []
                                    for sample_idx in range(1, perf_reps + 1):
                                        base_cmd = build_benchmark_command(
                                            bench_bin, m, k, n, benchmark_inner_reps, pin_core, use_taskset
                                        )
                                        cmd = perf_command(base_cmd)
                                        proc = run_process(cmd, child_env)
                                        try:
                                            parsed = parse_benchmark_output(proc.stdout + "\n" + proc.stderr)
                                        except ValueError:
                                            parsed = None
                                        status = measurement_status(proc, parsed)
                                        row_out = {
                                            "Implementation": config["benchmark"]["custom_impl"],
                                            "RunType": "custom",
                                            "WorkloadFamily": workload["WorkloadFamily"],
                                            "WorkloadId": workload["WorkloadId"],
                                            "RequestedKernelShape": kernel_shape,
                                            "KernelShape": "",
                                            "RequestedMc": str(mc),
                                            "Mc": "",
                                            "RequestedNc": str(nc),
                                            "Nc": "",
                                            "RequestedKc": str(kc),
                                            "Kc": "",
                                            "Size": size_text,
                                            "Threads": str(threads),
                                            "MeasurementKind": "perf",
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
                                        effective = parse_effective_config(proc.stdout + "\n" + proc.stderr)
                                        row_out["KernelShape"] = effective.get("KernelShape", kernel_shape)
                                        row_out["Mc"] = effective.get("Mc", str(mc))
                                        row_out["Nc"] = effective.get("Nc", str(nc))
                                        row_out["Kc"] = effective.get("Kc", str(kc))
                                        if status == "ok":
                                            perf_values = parse_perf_stat(proc.stderr)
                                            for key, value in perf_values.items():
                                                row_out[key] = fmt_float(value)
                                        else:
                                            row_out["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
                                        perf_rows.append(row_out)
                                    append_csv_rows(args.output, RAW_FIELDS, perf_rows)
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
