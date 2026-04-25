#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
COMMON_DIR = SCRIPT_DIR.parent / "GotoBLASBlocked"
sys.path.insert(0, str(COMMON_DIR))

from common import (  # noqa: E402
    command_string,
    fmt_float,
    gflops,
    measurement_status,
    parse_benchmark_output,
    parse_effective_config,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "thread_aware_mcnc"
)
DEFAULT_BENCHMARK = REPO_ROOT / "build" / "test_benchmark_large"

IMPLEMENTATION = "omp_gotoblas_avx512"
KERNEL_SHAPE = "avx512_8x32"
FIXED_KC = 160
MC_CANDIDATES = [8, 16, 24, 32, 48, 64, 72]
NC_CANDIDATES = [256, 320, 384, 448, 512, 640, 768]
THREADS = [1, 2, 4, 8]
DEFAULT_QUICK_REPS = 50
DEFAULT_FULL_REPS = 100

MINIMAL_WORKLOADS = [
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv2_dX_b32", 3200, 16, 150),
    ("fc_forward_mainstream_nn", "mlp_fc1_b32", 32, 784, 128),
    ("fc_head_small_n_nn", "mlp_fc2_b32", 32, 128, 10),
    ("fc_wide_output_nn", "mlp_fc1_hidden256_b32", 32, 784, 256),
]

FULL_WORKLOADS = [
    ("fc_forward_mainstream_nn", "mlp_fc1_b64", 64, 784, 128),
    ("fc_forward_mainstream_nn", "cnn_fc1_b32", 32, 400, 120),
    ("fc_head_small_n_nn", "mlp_fc2_b64", 64, 128, 10),
    ("fc_head_small_n_nn", "cnn_fc2_smallk_b32", 32, 84, 32),
    ("fc_wide_output_nn", "mlp_fc1_hidden256_b64", 64, 784, 256),
    ("fc_wide_output_nn", "cnn_fc1_wide_b32", 32, 400, 256),
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv1_dX_b32", 25088, 6, 25),
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv2_dX_b32", 3200, 16, 150),
]

RAW_FIELDS = [
    "Implementation",
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "RequestedKc",
    "RequestedMc",
    "RequestedNc",
    "Kc",
    "Mc",
    "Nc",
    "Reps",
    "SampleIndex",
    "Time_us",
    "GFLOPS",
    "Status",
    "ErrorMessage",
    "Command",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AVX-512 8x32 fixed-Kc thread-aware Mc/Nc tuning."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small smoke subset.")
    mode.add_argument("--full", action="store_true", help="Run the full Mc/Nc/thread/workload grid.")
    parser.add_argument("--benchmark-bin", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help=f"Benchmark inner repetitions per sample. Defaults to {DEFAULT_QUICK_REPS} for quick and {DEFAULT_FULL_REPS} for full.",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append to raw CSV instead of overwriting.")
    parser.add_argument("--workload-set", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--taskset-cpus", default="", help="Optional CPU list/range passed to taskset -c.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_grid(args: argparse.Namespace) -> tuple[list[int], list[int], list[int], list[tuple[str, str, int, int, int]]]:
    if args.quick or not args.full:
        workloads = MINIMAL_WORKLOADS[:2]
        return [1, 4], [8, 32], [384, 512], workloads
    workloads = FULL_WORKLOADS if args.workload_set == "full" else MINIMAL_WORKLOADS
    return THREADS, MC_CANDIDATES, NC_CANDIDATES, workloads


def init_csv(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()


def append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RAW_FIELDS).writerow(
            {field: row.get(field, "") for field in RAW_FIELDS}
        )


def build_env(threads: int, mc: int, nc: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "false",
        "OMP_PROC_BIND": "true",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MATMUL_IMPL": IMPLEMENTATION,
        "MATMUL_GOTO_KERNEL": KERNEL_SHAPE,
        "MATMUL_KC": str(FIXED_KC),
        "MATMUL_MC": str(mc),
        "MATMUL_NC": str(nc),
        "MATMUL_EMIT_EFFECTIVE_CONFIG": "1",
    })
    return env


def build_command(bench: Path, m: int, k: int, n: int, reps: int, taskset_cpus: str) -> list[str]:
    cmd = [str(bench), str(m), str(k), str(n), str(reps)]
    if taskset_cpus:
        return ["taskset", "-c", taskset_cpus, *cmd]
    return cmd


def run_case(
    args: argparse.Namespace,
    raw_path: Path,
    *,
    threads: int,
    mc: int,
    nc: int,
    reps: int,
    workload: tuple[str, str, int, int, int],
    sample_index: int,
) -> None:
    family, workload_id, m, k, n = workload
    env = build_env(threads, mc, nc)
    cmd = build_command(args.benchmark_bin, m, k, n, reps, args.taskset_cpus)
    base_row = {
        "Implementation": IMPLEMENTATION,
        "KernelShape": KERNEL_SHAPE,
        "WorkloadFamily": family,
        "WorkloadId": workload_id,
        "M": str(m),
        "K": str(k),
        "N": str(n),
        "Threads": str(threads),
        "RequestedKc": str(FIXED_KC),
        "RequestedMc": str(mc),
        "RequestedNc": str(nc),
        "Kc": str(FIXED_KC),
        "Mc": str(mc),
        "Nc": str(nc),
        "Reps": str(reps),
        "SampleIndex": str(sample_index),
        "Command": command_string(cmd, env),
    }
    if args.dry_run:
        append_row(raw_path, {**base_row, "Status": "dry_run"})
        return

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    text = proc.stdout + "\n" + proc.stderr
    try:
        parsed = parse_benchmark_output(text)
    except ValueError:
        parsed = None

    row = dict(base_row)
    effective = parse_effective_config(text)
    row["KernelShape"] = effective.get("KernelShape", row["KernelShape"]) or row["KernelShape"]
    row["Kc"] = effective.get("Kc", row["Kc"]) or row["Kc"]
    row["Mc"] = effective.get("Mc", row["Mc"]) or row["Mc"]
    row["Nc"] = effective.get("Nc", row["Nc"]) or row["Nc"]
    row["Status"] = measurement_status(proc, parsed)
    if parsed is not None:
        mean_s, _, _, _ = parsed
        time_us = mean_s * 1e6
        row["Time_us"] = fmt_float(time_us)
        row["GFLOPS"] = fmt_float(gflops(m, k, n, time_us))
    if row["Status"] != "ok":
        row["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
    append_row(raw_path, row)


def main() -> int:
    args = build_parser().parse_args()
    if not args.benchmark_bin.exists():
        print(f"benchmark binary not found: {args.benchmark_bin}", file=sys.stderr)
        return 1
    quick_mode = args.quick or not args.full
    reps = args.reps if args.reps is not None else (DEFAULT_QUICK_REPS if quick_mode else DEFAULT_FULL_REPS)
    if reps <= 0:
        print("--reps must be > 0", file=sys.stderr)
        return 1

    threads_list, mc_candidates, nc_candidates, workloads = selected_grid(args)
    samples = args.samples if args.samples is not None else (1 if quick_mode else 3)
    raw_path = args.output_dir / "raw_results.csv"
    init_csv(raw_path, args.append)
    total = len(threads_list) * len(mc_candidates) * len(nc_candidates) * len(workloads) * samples
    mode = "quick" if quick_mode else "full"
    print(
        f"[thread-aware-mcnc] mode={mode} kernel={KERNEL_SHAPE} kc={FIXED_KC} "
        f"threads={threads_list} mc={mc_candidates} nc={nc_candidates} "
        f"workloads={len(workloads)} samples={samples} reps={reps} total={total}"
    )

    index = 0
    for threads in threads_list:
        for mc in mc_candidates:
            for nc in nc_candidates:
                for workload in workloads:
                    for sample_index in range(1, samples + 1):
                        index += 1
                        print(
                            f"[thread-aware-mcnc] {index}/{total} "
                            f"T={threads} Mc={mc} Nc={nc} workload={workload[1]} sample={sample_index}",
                            flush=True,
                        )
                        run_case(
                            args,
                            raw_path,
                            threads=threads,
                            mc=mc,
                            nc=nc,
                            reps=reps,
                            workload=workload,
                            sample_index=sample_index,
                        )
    print(f"Wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
