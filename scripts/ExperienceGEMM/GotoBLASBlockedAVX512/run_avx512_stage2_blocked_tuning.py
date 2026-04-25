#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
COMMON_DIR = SCRIPT_DIR.parent / "GotoBLASBlocked"
sys.path.insert(0, str(COMMON_DIR))

from common import (  # noqa: E402
    build_benchmark_command,
    command_string,
    fmt_float,
    gflops,
    measurement_status,
    parse_benchmark_output,
    parse_effective_config,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "stage2_blocked_tuning"
)
DEFAULT_BENCHMARK = REPO_ROOT / "build" / "test_benchmark_large"

RAW_FIELDS = [
    "Implementation",
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "RequestedMc",
    "RequestedNc",
    "RequestedKc",
    "Mc",
    "Nc",
    "Kc",
    "Reps",
    "SampleIndex",
    "Time_us",
    "GFLOPS",
    "Status",
    "ErrorMessage",
    "Command",
]

WORKLOADS = [
    {
        "WorkloadFamily": "conv_dx_extremely_skinny_k_nn",
        "WorkloadId": "cnn_conv2_dX_b32",
        "Size": (3200, 16, 150),
    },
    {
        "WorkloadFamily": "fc_forward_mainstream_nn",
        "WorkloadId": "mlp_fc1_b32",
        "Size": (32, 784, 128),
    },
    {
        "WorkloadFamily": "fc_head_small_n_nn",
        "WorkloadId": "mlp_fc2_b32",
        "Size": (32, 128, 10),
    },
    {
        "WorkloadFamily": "fc_wide_output_nn",
        "WorkloadId": "mlp_fc1_hidden256_b32",
        "Size": (32, 784, 256),
    },
]

TUNING_SPACE = [
    {
        "KernelShape": "avx512_8x32",
        "Role": "default_path_candidate",
        "Rows": [
            {"Kc": 96, "Mc": [32, 48, 64], "Nc": [384, 512, 640, 768, 1024]},
            {"Kc": 128, "Mc": [32, 48, 64], "Nc": [384, 512, 640, 768, 1024]},
            {"Kc": 160, "Mc": [32, 48, 64], "Nc": [384, 512, 640, 768, 1024]},
        ],
    },
    {
        "KernelShape": "avx512_16x16",
        "Role": "small_n_fallback_candidate",
        "Rows": [
            {"Kc": 160, "Mc": [32, 48, 64], "Nc": [256, 384, 512, 640, 768, 1024]},
            {"Kc": 192, "Mc": [32, 48, 64], "Nc": [256, 384, 512, 640, 768, 1024]},
            {"Kc": 256, "Mc": [32, 48, 64], "Nc": [256, 384, 512, 640, 768]},
        ],
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AVX-512 Stage 2 blocked-size tuning.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a tiny smoke subset.")
    mode.add_argument("--full", action="store_true", help="Run the full Stage 2 tuning set.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-bin", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--reps", type=int, default=None, help="Benchmark inner repetitions per sample.")
    parser.add_argument("--samples", type=int, default=None, help="Samples per candidate/workload.")
    parser.add_argument("--pin-core", type=int, default=0)
    parser.add_argument("--no-taskset", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print every workload/sample command as it runs.")
    return parser


def iter_candidates(quick: bool):
    for kernel in TUNING_SPACE:
        emitted_for_kernel = 0
        for row in kernel["Rows"]:
            kc = int(row["Kc"])
            for mc in row["Mc"]:
                for nc in row["Nc"]:
                    yield {
                        "KernelShape": kernel["KernelShape"],
                        "Role": kernel["Role"],
                        "Kc": kc,
                        "Mc": int(mc),
                        "Nc": int(nc),
                    }
                    emitted_for_kernel += 1
                    if quick and emitted_for_kernel >= 1:
                        break
                if quick and emitted_for_kernel >= 1:
                    break
            if quick and emitted_for_kernel >= 1:
                break


def init_csv(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()


def append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})


def child_env(candidate: dict[str, int | str]) -> dict[str, str]:
    env = os.environ.copy()
    env["MATMUL_IMPL"] = "omp_gotoblas_avx512"
    env["MATMUL_GOTO_KERNEL"] = str(candidate["KernelShape"])
    env["MATMUL_MC"] = str(candidate["Mc"])
    env["MATMUL_NC"] = str(candidate["Nc"])
    env["MATMUL_KC"] = str(candidate["Kc"])
    env["MATMUL_EMIT_EFFECTIVE_CONFIG"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "false"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("OMP_PROC_BIND", "true")
    env.setdefault("OMP_PLACES", "cores")
    return env


def main() -> int:
    args = build_parser().parse_args()
    quick = args.quick or not args.full
    reps = args.reps if args.reps is not None else (5 if quick else 10)
    samples = args.samples if args.samples is not None else (1 if quick else 3)
    bench_bin = args.benchmark_bin
    if not bench_bin.exists():
        print(f"benchmark binary not found: {bench_bin}", file=sys.stderr)
        return 1

    output_path = args.output_dir / "raw_results.csv"
    init_csv(output_path, append=args.append)
    use_taskset = not args.no_taskset and shutil.which("taskset") is not None
    pinning_policy = f"taskset:{args.pin_core}" if use_taskset else "unbound"

    candidates = list(iter_candidates(quick))
    print(
        f"[stage2] mode={'quick' if quick else 'full'} candidates={len(candidates)} "
        f"workloads={len(WORKLOADS)} samples={samples} reps={reps} "
        f"output={output_path}"
    )

    for candidate_index, candidate in enumerate(candidates, start=1):
        env = child_env(candidate)
        print(
            f"[stage2] candidate {candidate_index}/{len(candidates)} "
            f"kernel={candidate['KernelShape']} mc_nc_kc=({candidate['Mc']},{candidate['Nc']},{candidate['Kc']})"
        )
        for workload in WORKLOADS:
            m, k, n = workload["Size"]
            for sample_index in range(1, samples + 1):
                cmd = build_benchmark_command(
                    bench_bin, m, k, n, reps, args.pin_core, use_taskset
                )
                base_row = {
                    "Implementation": "omp_gotoblas_avx512",
                    "KernelShape": str(candidate["KernelShape"]),
                    "WorkloadFamily": workload["WorkloadFamily"],
                    "WorkloadId": workload["WorkloadId"],
                    "M": str(m),
                    "K": str(k),
                    "N": str(n),
                    "Threads": "1",
                    "RequestedMc": str(candidate["Mc"]),
                    "RequestedNc": str(candidate["Nc"]),
                    "RequestedKc": str(candidate["Kc"]),
                    "Mc": str(candidate["Mc"]),
                    "Nc": str(candidate["Nc"]),
                    "Kc": str(candidate["Kc"]),
                    "Reps": str(reps),
                    "SampleIndex": str(sample_index),
                    "Command": command_string(cmd, env),
                }
                if args.verbose:
                    print(
                        f"[stage2]   workload={workload['WorkloadId']} "
                        f"sample={sample_index}/{samples} pinning={pinning_policy}"
                    )
                if args.dry_run:
                    row = {**base_row, "Status": "dry_run"}
                    append_row(output_path, row)
                    continue
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
                text = proc.stdout + "\n" + proc.stderr
                try:
                    parsed = parse_benchmark_output(text)
                except ValueError:
                    parsed = None
                effective = parse_effective_config(text)
                row = dict(base_row)
                row["Status"] = measurement_status(proc, parsed)
                row["KernelShape"] = effective.get("KernelShape", row["KernelShape"]) or row["KernelShape"]
                row["Mc"] = effective.get("Mc", row["Mc"]) or row["Mc"]
                row["Nc"] = effective.get("Nc", row["Nc"]) or row["Nc"]
                row["Kc"] = effective.get("Kc", row["Kc"]) or row["Kc"]
                if parsed is not None:
                    mean_s, _, _, _ = parsed
                    time_us = mean_s * 1e6
                    row["Time_us"] = fmt_float(time_us)
                    row["GFLOPS"] = fmt_float(gflops(m, k, n, time_us))
                if row["Status"] != "ok":
                    row["ErrorMessage"] = (proc.stderr or proc.stdout).strip()[:500]
                append_row(output_path, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
