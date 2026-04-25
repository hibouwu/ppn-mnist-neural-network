#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
COMMON_DIR = SCRIPT_DIR.parent / "GotoBLASBlocked"
sys.path.insert(0, str(COMMON_DIR))

from common import fmt_float, gflops, parse_benchmark_output  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "fixed_strong_scaling_comparison"
)
DEFAULT_BENCHMARK = REPO_ROOT / "build" / "test_benchmark_large"
DEFAULT_QUICK_REPS = 50
DEFAULT_FULL_REPS = 1000
THREADS = [1, 2, 4, 8]

MINIMAL_WORKLOADS = [
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv2_dX_b32", 3200, 16, 150),
    ("fc_forward_mainstream_nn", "mlp_fc1_b32", 32, 784, 128),
    ("fc_head_small_n_nn", "mlp_fc2_b32", 32, 128, 10),
    ("fc_wide_output_nn", "mlp_fc1_hidden256_b32", 32, 784, 256),
    ("square_matrix", "square_256", 256, 256, 256),
]

FULL_WORKLOADS = [
    # CNN Conv2 layer — dominates CNN FLOPs (63% combined across fwd/dX/dW)
    ("conv_fwd_medium_k_nn",              "cnn_conv2_fwd_b32",  3200, 150, 16),   # 23.3% CNN FLOPs
    ("conv_dx_extremely_skinny_k_nn",     "cnn_conv2_dx_b32",   3200,  16, 150),  # 20.0% CNN FLOPs
    ("conv_dw_transposed_very_skinny_m",  "cnn_conv2_dw_b32",     16, 3200, 150), # 20.0% CNN FLOPs
    # CNN Conv1 layer — large-M, very skinny K (21% combined across fwd/dW)
    ("conv_fwd_very_skinny_k_small_n_nn", "cnn_conv1_fwd_b32", 25088,  25,   6),  # 11.4% CNN FLOPs
    ("conv_dw_transposed_very_skinny_m",  "cnn_conv1_dw_b32",      6, 25088, 25), #  9.8% CNN FLOPs
    # CNN FC layers — mainstream dense GEMM
    ("fc_forward_mainstream_nn",          "cnn_fc1_b32",          32,  400, 120), #  4.7% CNN FLOPs
    # MLP FC1 — single dominant shape in MLP (36% MLP FLOPs)
    ("fc_forward_mainstream_nn",          "mlp_fc1_b32",          32,  784, 128), # 36.4% MLP FLOPs
    ("square_matrix", "square_256", 256, 256, 256),
    ("square_matrix", "square_512", 512, 512, 512),
]

# Workloads derived from actual MATMUL_TRACE_SHAPES=1 runs: CNN (LeNet-5) + MLP, batch=32, 1 epoch.
# All shapes are (M×K)@(K×N) non-transposed canonical form.
# Custom GotoBLAS impls benchmark non-transposed; transposed dW paths fall back to BLAS in production.
# Source: output/ExperienceGEMM/GotoBLASBlockedAVX512/training_shape_trace/
TRAINING_TRACE_WORKLOADS = [
    # CNN Conv2 layer — dominates CNN FLOPs (63% combined across fwd/dX/dW)
    ("conv_fwd_medium_k_nn",              "cnn_conv2_fwd_b32",  3200, 150, 16),   # 23.3% CNN FLOPs
    ("conv_dx_extremely_skinny_k_nn",     "cnn_conv2_dx_b32",   3200,  16, 150),  # 20.0% CNN FLOPs
    ("conv_dw_transposed_very_skinny_m",  "cnn_conv2_dw_b32",     16, 3200, 150), # 20.0% CNN FLOPs
    # CNN Conv1 layer — large-M, very skinny K (21% combined across fwd/dW)
    ("conv_fwd_very_skinny_k_small_n_nn", "cnn_conv1_fwd_b32", 25088,  25,   6),  # 11.4% CNN FLOPs
    ("conv_dw_transposed_very_skinny_m",  "cnn_conv1_dw_b32",      6, 25088, 25), #  9.8% CNN FLOPs
    # CNN FC layers — mainstream dense GEMM
    ("fc_forward_mainstream_nn",          "cnn_fc1_b32",          32,  400, 120), #  4.7% CNN FLOPs
    # MLP FC1 — single dominant shape in MLP (36% MLP FLOPs)
    ("fc_forward_mainstream_nn",          "mlp_fc1_b32",          32,  784, 128), # 36.4% MLP FLOPs
]

IMPLEMENTATIONS = [
    {
        "ImplementationLabel": "AVX2_GotoBLAS",
        "MatmulImpl": "omp_gotoblas_avx2",
        "KernelShape": "avx2_8x8",
        "Kc": "384",
        "Mc": "8",
        "Nc": "448",
    },
    {
        "ImplementationLabel": "AVX512_GotoBLAS",
        "MatmulImpl": "omp_gotoblas_avx512",
        "KernelShape": "avx512_8x32",
        "Kc": "160",
        "Mc": "8",
        "Nc": "448",
    },
    {
        "ImplementationLabel": "OpenBLAS",
        "MatmulImpl": "blas",
        "KernelShape": "n/a",
        "Kc": "n/a",
        "Mc": "n/a",
        "Nc": "n/a",
    },
]

RAW_FIELDS = [
    "ImplementationLabel",
    "MatmulImpl",
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

EFFECTIVE_RE = re.compile(
    r"Effective config:\s*impl=(?P<impl>\S+)"
    r"(?:\s+kernel=(?P<kernel>\S+)\s+mc=(?P<mc>[0-9]+)\s+nc=(?P<nc>[0-9]+)\s+kc=(?P<kc>[0-9]+))?"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-config strong-scaling comparison.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small smoke subset.")
    mode.add_argument("--full", action="store_true", help="Run the full comparison grid.")
    parser.add_argument("--benchmark-bin", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--workload-set", choices=("minimal", "full", "training-trace"), default="minimal")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--taskset-cpus", default="", help="Optional CPU list/range passed to taskset -c.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_string(cmd: list[str], env: dict[str, str]) -> str:
    keys = [
        "MATMUL_IMPL",
        "MATMUL_GOTO_KERNEL",
        "MATMUL_MC",
        "MATMUL_NC",
        "MATMUL_KC",
        "MATMUL_EMIT_EFFECTIVE_CONFIG",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    prefix = [f"{key}={env[key]}" for key in keys if key in env]
    return " ".join([*prefix, *cmd])


def parse_effective(text: str) -> dict[str, str]:
    match = EFFECTIVE_RE.search(text)
    if match is None:
        return {}
    return {
        "MatmulImpl": match.group("impl") or "",
        "KernelShape": match.group("kernel") or "",
        "Mc": match.group("mc") or "",
        "Nc": match.group("nc") or "",
        "Kc": match.group("kc") or "",
    }


def selected_workloads(args: argparse.Namespace) -> list[tuple[str, str, int, int, int]]:
    if args.quick or not args.full:
        return MINIMAL_WORKLOADS[:2]
    if args.workload_set == "full":
        return FULL_WORKLOADS
    if args.workload_set == "training-trace":
        return TRAINING_TRACE_WORKLOADS
    return MINIMAL_WORKLOADS


def selected_threads(args: argparse.Namespace) -> list[int]:
    if args.quick or not args.full:
        return [1, 2]
    return THREADS


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


def build_env(impl: dict[str, str], threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MATMUL_IMPL"] = impl["MatmulImpl"]
    env["MATMUL_EMIT_EFFECTIVE_CONFIG"] = "1"
    env["OMP_DYNAMIC"] = "false"
    env["MKL_NUM_THREADS"] = "1"
    if impl["ImplementationLabel"] == "OpenBLAS":
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
        env["GOTO_NUM_THREADS"] = "1"
        env["BLIS_NUM_THREADS"] = "1"
        for key in (
            "MATMUL_GOTO_KERNEL", "MATMUL_MC", "MATMUL_NC", "MATMUL_KC",
            "MATMUL_PACK_M", "MATMUL_PACK_N", "MATMUL_PACK_K",
            "MATMUL_EMIT_EFFECTIVE_CONFIG",
        ):
            env.pop(key, None)
        return env

    env["MATMUL_GOTO_KERNEL"] = impl["KernelShape"]
    env["MATMUL_KC"] = impl["Kc"]
    env["MATMUL_MC"] = impl["Mc"]
    env["MATMUL_NC"] = impl["Nc"]
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = "cores"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
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
    impl: dict[str, str],
    threads: int,
    workload: tuple[str, str, int, int, int],
    reps: int,
    sample_index: int,
) -> None:
    family, workload_id, m, k, n = workload
    env = build_env(impl, threads)
    cmd = build_command(args.benchmark_bin, m, k, n, reps, args.taskset_cpus)
    requested_kc = impl["Kc"]
    requested_mc = impl["Mc"]
    requested_nc = impl["Nc"]
    base_row = {
        "ImplementationLabel": impl["ImplementationLabel"],
        "MatmulImpl": impl["MatmulImpl"],
        "KernelShape": impl["KernelShape"],
        "WorkloadFamily": family,
        "WorkloadId": workload_id,
        "M": str(m),
        "K": str(k),
        "N": str(n),
        "Threads": str(threads),
        "RequestedKc": requested_kc,
        "RequestedMc": requested_mc,
        "RequestedNc": requested_nc,
        "Kc": requested_kc,
        "Mc": requested_mc,
        "Nc": requested_nc,
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
    effective = parse_effective(text)
    if effective.get("MatmulImpl"):
        row["MatmulImpl"] = effective["MatmulImpl"]
    if effective.get("KernelShape"):
        row["KernelShape"] = effective["KernelShape"]
    if effective.get("Kc"):
        row["Kc"] = effective["Kc"]
    if effective.get("Mc"):
        row["Mc"] = effective["Mc"]
    if effective.get("Nc"):
        row["Nc"] = effective["Nc"]

    if proc.returncode != 0 or parsed is None:
        row["Status"] = "error"
        row["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
        append_row(raw_path, row)
        return

    mean_s, _, _, result_check = parsed
    if mean_s <= 0.0 or not (result_check is None or result_check == result_check):
        row["Status"] = "error"
        row["ErrorMessage"] = "invalid benchmark result"
        append_row(raw_path, row)
        return

    if impl["ImplementationLabel"] != "OpenBLAS":
        mismatches = []
        if row["Kc"] != requested_kc:
            mismatches.append(f"Kc: requested={requested_kc} effective={row['Kc']}")
        if row["Mc"] != requested_mc:
            mismatches.append(f"Mc: requested={requested_mc} effective={row['Mc']}")
        if row["Nc"] != requested_nc:
            mismatches.append(f"Nc: requested={requested_nc} effective={row['Nc']}")
        if mismatches:
            row["Status"] = "error"
            row["ErrorMessage"] = "effective config mismatch: " + "; ".join(mismatches)
            append_row(raw_path, row)
            return

    time_us = mean_s * 1e6
    row["Status"] = "ok"
    row["Time_us"] = fmt_float(time_us)
    row["GFLOPS"] = fmt_float(gflops(m, k, n, time_us))
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

    workloads = selected_workloads(args)
    threads_list = selected_threads(args)
    samples = args.samples if args.samples is not None else (1 if quick_mode else 3)
    raw_path = args.output_dir / "raw_results.csv"
    init_csv(raw_path, args.append)
    total = len(IMPLEMENTATIONS) * len(threads_list) * len(workloads) * samples
    print(
        f"[fixed-scaling] mode={'quick' if quick_mode else 'full'} "
        f"implementations={len(IMPLEMENTATIONS)} threads={threads_list} "
        f"workloads={len(workloads)} samples={samples} reps={reps} total={total}"
    )

    index = 0
    for impl in IMPLEMENTATIONS:
        for threads in threads_list:
            for workload in workloads:
                for sample_index in range(1, samples + 1):
                    index += 1
                    print(
                        f"[fixed-scaling] {index}/{total} impl={impl['ImplementationLabel']} "
                        f"T={threads} workload={workload[1]} sample={sample_index}",
                        flush=True,
                    )
                    run_case(
                        args,
                        raw_path,
                        impl=impl,
                        threads=threads,
                        workload=workload,
                        reps=reps,
                        sample_index=sample_index,
                    )
    print(f"Wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
