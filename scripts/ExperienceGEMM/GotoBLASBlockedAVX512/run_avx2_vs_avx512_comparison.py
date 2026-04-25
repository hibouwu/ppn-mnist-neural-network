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
BENCH = REPO_ROOT / "build" / "test_benchmark_large"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "avx2_vs_avx512"
    / "raw_comparison.csv"
)
WORKLOADS = [
    ("fc_forward_mainstream_nn", "mlp_fc1_b32", 32, 784, 128),
    ("fc_head_small_n_nn", "mlp_fc2_b32", 32, 128, 10),
    ("fc_wide_output_nn", "mlp_fc1_hidden256_b32", 32, 784, 256),
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv2_dX_b32", 3200, 16, 150),
]
FIELDS = [
    "Implementation",
    "KernelShape",
    "Mc",
    "Nc",
    "Kc",
    "WorkloadFamily",
    "WorkloadId",
    "Size",
    "Threads",
    "Reps",
    "Status",
    "Command",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit or run a fixed AVX2 vs AVX-512 comparison protocol.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Only write commands; do not run benchmarks.")
    parser.add_argument("--avx2-kernel", default="avx2_8x8")
    parser.add_argument("--avx2-mc", default="8")
    parser.add_argument("--avx2-nc", default="448")
    parser.add_argument("--avx2-kc", default="384")
    parser.add_argument("--avx512-kernel", default="avx512_8x16")
    parser.add_argument("--avx512-mc", default="32")
    parser.add_argument("--avx512-nc", default="512")
    parser.add_argument("--avx512-kc", default="256")
    return parser


def command(env: dict[str, str], m: int, k: int, n: int, reps: int) -> list[str]:
    return [str(BENCH), str(m), str(k), str(n), str(reps)]


def env_for(args: argparse.Namespace, impl: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["GOTO_NUM_THREADS"] = str(args.threads)
    env["BLIS_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["MATMUL_IMPL"] = impl
    env["MATMUL_EMIT_EFFECTIVE_CONFIG"] = "1"
    if impl == "omp_gotoblas_avx2":
        env["MATMUL_GOTO_KERNEL"] = args.avx2_kernel
        env["MATMUL_MC"] = args.avx2_mc
        env["MATMUL_NC"] = args.avx2_nc
        env["MATMUL_KC"] = args.avx2_kc
    else:
        env["MATMUL_GOTO_KERNEL"] = args.avx512_kernel
        env["MATMUL_MC"] = args.avx512_mc
        env["MATMUL_NC"] = args.avx512_nc
        env["MATMUL_KC"] = args.avx512_kc
    return env


def command_string(env: dict[str, str], cmd: list[str]) -> str:
    keys = ["MATMUL_IMPL", "MATMUL_GOTO_KERNEL", "MATMUL_MC", "MATMUL_NC", "MATMUL_KC", "OMP_NUM_THREADS"]
    return " ".join([*(f"{key}={env[key]}" for key in keys), *cmd])


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run and not BENCH.exists():
        print(f"missing benchmark: {BENCH}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for impl in ("omp_gotoblas_avx2", "omp_gotoblas_avx512"):
            env = env_for(args, impl)
            for family, workload, m, k, n in WORKLOADS:
                cmd = command(env, m, k, n, args.reps)
                status = "dry_run"
                if not args.dry_run:
                    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
                    status = "ok" if proc.returncode == 0 else "failed"
                writer.writerow({
                    "Implementation": impl,
                    "KernelShape": env["MATMUL_GOTO_KERNEL"],
                    "Mc": env["MATMUL_MC"],
                    "Nc": env["MATMUL_NC"],
                    "Kc": env["MATMUL_KC"],
                    "WorkloadFamily": family,
                    "WorkloadId": workload,
                    "Size": f"{m}x{k}x{n}",
                    "Threads": str(args.threads),
                    "Reps": str(args.reps),
                    "Status": status,
                    "Command": command_string(env, cmd),
                })
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
