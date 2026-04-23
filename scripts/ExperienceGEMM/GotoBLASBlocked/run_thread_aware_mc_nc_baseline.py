#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from common import parse_perf_stat, perf_command


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "build" / "test_benchmark_large"
OUT_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "thread_aware_mc_nc_baseline"
RAW_CSV = OUT_DIR / "raw.csv"
SUMMARY_MD = OUT_DIR / "summary.md"

THREADS = [1, 2, 4]
MC_CANDIDATES = [4, 8, 16, 24, 32, 48, 64, 96]
NC_CANDIDATES = [192, 256, 320, 384, 448, 512, 640, 768]
FIXED_KC = 384
FIXED_KERNEL = "avx2_8x8"
BENCH_REPS = 300
PERF_REPS = 3

WORKLOADS = [
    ("fc_forward_mainstream_nn", "mlp_fc1_b64", 64, 784, 128),
    ("fc_forward_mainstream_nn", "cnn_fc1_b32", 32, 400, 120),
    ("fc_head_small_n_nn", "mlp_fc2_b64", 64, 128, 10),
    ("fc_head_small_n_nn", "cnn_fc2_smallk_b32", 32, 84, 32),
    ("fc_wide_output_nn", "mlp_fc1_hidden256_b64", 64, 784, 256),
    ("fc_wide_output_nn", "cnn_fc1_wide_b32", 32, 400, 256),
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv1_dX_b32", 25088, 6, 25),
    ("conv_dx_extremely_skinny_k_nn", "cnn_conv2_dX_b32", 3200, 16, 150),
]

MEAN_RE = re.compile(r"Mean:\s*([0-9.]+)\s*s")
EFFECTIVE_RE = re.compile(
    r"Effective config: .*kernel=(?P<kernel>\S+) mc=(?P<mc>\d+) nc=(?P<nc>\d+) kc=(?P<kc>\d+)"
)

PERF_FIELDS = [
    "Instructions",
    "Cycles",
    "IPC",
    "CacheMisses",
    "CacheReferences",
    "L1DMisses",
    "DTLBMisses",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline-only thread-aware Mc/Nc tuning with optional perf counters."
    )
    parser.add_argument("--skip-perf", action="store_true", help="Skip perf stat collection.")
    parser.add_argument("--perf-reps", type=int, default=PERF_REPS, help="perf stat repetitions per point.")
    parser.add_argument("--limit-workloads", type=int, default=None, help="Optional workload prefix limit.")
    parser.add_argument("--limit-mc", type=int, default=None, help="Optional Mc candidate prefix limit.")
    parser.add_argument("--limit-nc", type=int, default=None, help="Optional Nc candidate prefix limit.")
    parser.add_argument("--threads", nargs="*", type=int, default=None, help="Optional thread list override.")
    return parser


def gflops(m: int, k: int, n: int, seconds: float) -> float:
    return (2.0 * m * k * n) / seconds / 1e9


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def mean_from_output(text: str) -> float | None:
    match = MEAN_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def effective_from_output(text: str) -> dict[str, str]:
    match = EFFECTIVE_RE.search(text)
    if match is None:
        return {}
    return {
        "KernelShape": match.group("kernel"),
        "Mc": match.group("mc"),
        "Nc": match.group("nc"),
        "Kc": match.group("kc"),
    }


def build_env(threads: int, mc: int, nc: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "MATMUL_IMPL": "omp_gotoblas_avx2",
        "MATMUL_MC": str(mc),
        "MATMUL_NC": str(nc),
        "MATMUL_KC": str(FIXED_KC),
        "MATMUL_GOTO_KERNEL": FIXED_KERNEL,
        "MATMUL_EMIT_EFFECTIVE_CONFIG": "1",
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    return env


def run_one(
    threads: int,
    mc: int,
    nc: int,
    workload: tuple[str, str, int, int, int],
    *,
    collect_perf: bool,
    perf_reps: int,
) -> dict[str, str]:
    family, workload_id, m, k, n = workload
    env = build_env(threads, mc, nc)
    cmd = [str(BENCH), str(m), str(k), str(n), str(BENCH_REPS)]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    output = proc.stdout + "\n" + proc.stderr

    row: dict[str, str] = {
        "T": str(threads),
        "RequestedMc": str(mc),
        "RequestedNc": str(nc),
        "RequestedKc": str(FIXED_KC),
        "RequestedKernelShape": FIXED_KERNEL,
        "WorkloadFamily": family,
        "WorkloadId": workload_id,
        "M": str(m),
        "K": str(k),
        "N": str(n),
        "Reps": str(BENCH_REPS),
        "Status": "failed",
        "Mean_s": "",
        "GFLOPS": "",
        "PerfStatus": "skipped" if not collect_perf else "failed",
        "PerfReps": str(perf_reps if collect_perf else 0),
        "PerfMean_s": "",
        "PerfGFLOPS": "",
        "KernelShape": "",
        "Mc": "",
        "Nc": "",
        "Kc": "",
        "ErrorMessage": "",
        "PerfErrorMessage": "",
    }
    for field in PERF_FIELDS:
        row[field] = ""

    effective = effective_from_output(output)
    if effective:
        row.update(effective)

    mean_s = mean_from_output(output)
    if proc.returncode != 0 or mean_s is None:
        row["ErrorMessage"] = output.strip()[-500:]
        return row

    row["Status"] = "ok"
    row["Mean_s"] = f"{mean_s:.12f}"
    row["GFLOPS"] = f"{gflops(m, k, n, mean_s):.6f}"

    if not collect_perf:
        return row

    perf_times: list[float] = []
    perf_gflops: list[float] = []
    perf_values_by_field: dict[str, list[float]] = {field: [] for field in PERF_FIELDS}
    for _ in range(perf_reps):
        perf_proc = subprocess.run(perf_command(cmd), env=env, text=True, capture_output=True, check=False)
        perf_output = perf_proc.stdout + "\n" + perf_proc.stderr
        perf_mean_s = mean_from_output(perf_output)
        if perf_proc.returncode != 0 or perf_mean_s is None:
            row["PerfStatus"] = "failed"
            row["PerfErrorMessage"] = perf_output.strip()[-500:]
            return row
        perf_times.append(perf_mean_s)
        perf_gflops.append(gflops(m, k, n, perf_mean_s))
        perf_values = parse_perf_stat(perf_proc.stderr)
        for field in PERF_FIELDS:
            value = perf_values.get(field)
            if value is not None:
                perf_values_by_field[field].append(value)

    row["PerfStatus"] = "ok"
    row["PerfMean_s"] = f"{statistics.median(perf_times):.12f}"
    row["PerfGFLOPS"] = f"{statistics.median(perf_gflops):.6f}"
    for field, values in perf_values_by_field.items():
        if values:
            row[field] = f"{statistics.median(values):.6f}"
    return row


def perf_cache_text(subset: list[dict[str, str]], collect_perf: bool) -> str:
    if not collect_perf:
        return "n/a"
    if not all(row["PerfStatus"] == "ok" and row["CacheMisses"] for row in subset):
        return "n/a"
    values = [float(row["CacheMisses"]) for row in subset]
    if any(value <= 0.0 for value in values):
        return "n/a"
    return f"{geomean(values):.6f}"


def write_summary(
    rows: list[dict[str, str]],
    *,
    threads_list: list[int],
    mc_candidates: list[int],
    nc_candidates: list[int],
    workloads: list[tuple[str, str, int, int, int]],
    collect_perf: bool,
) -> None:
    winners = []
    for threads in threads_list:
        scored = []
        for mc in mc_candidates:
            for nc in nc_candidates:
                subset = [
                    r for r in rows
                    if r["T"] == str(threads)
                    and r["RequestedMc"] == str(mc)
                    and r["RequestedNc"] == str(nc)
                    and r["Status"] == "ok"
                ]
                if len(subset) != len(workloads):
                    continue
                score = geomean([float(r["GFLOPS"]) for r in subset])
                time_score = geomean([1e6 * float(r["Mean_s"]) for r in subset])
                scored.append((score, mc, nc, time_score, perf_cache_text(subset, collect_perf)))
        if not scored:
            winners.append((threads, None, None, None, None, "n/a"))
            continue
        score, mc, nc, time_score, cache_score = max(scored, key=lambda item: item[0])
        winners.append((threads, mc, nc, score, time_score, cache_score))

    families = sorted({workload[0] for workload in workloads})
    family_winners = []
    for family in families:
        expected = [workload for workload in workloads if workload[0] == family]
        for threads in threads_list:
            scored = []
            for mc in mc_candidates:
                for nc in nc_candidates:
                    subset = [
                        r for r in rows
                        if r["T"] == str(threads)
                        and r["RequestedMc"] == str(mc)
                        and r["RequestedNc"] == str(nc)
                        and r["WorkloadFamily"] == family
                        and r["Status"] == "ok"
                    ]
                    if len(subset) != len(expected):
                        continue
                    score = geomean([float(r["GFLOPS"]) for r in subset])
                    time_score = geomean([1e6 * float(r["Mean_s"]) for r in subset])
                    scored.append((score, mc, nc, time_score, perf_cache_text(subset, collect_perf)))
            if not scored:
                family_winners.append((family, threads, None, None, None, None, "n/a"))
                continue
            score, mc, nc, time_score, cache_score = max(scored, key=lambda item: item[0])
            family_winners.append((family, threads, mc, nc, score, time_score, cache_score))

    lines = [
        "# Thread-Aware Mc/Nc Baseline Tuning",
        "",
        "Scope: baseline-only `omp_gotoblas_avx2`; no experimental parallel scheme.",
        f"Fixed kernel shape: `{FIXED_KERNEL}`.",
        f"Fixed Kc: `{FIXED_KC}`.",
        f"Mc candidates: `{mc_candidates}`.",
        f"Nc candidates: `{nc_candidates}`.",
        f"Threads: `{threads_list}`.",
        f"Workloads: `{len(workloads)}` representative NN GEMM shapes across `{len(families)}` families.",
        "Winner metric: geomean GFLOPS across selected workloads.",
        f"Perf collection: `{'enabled' if collect_perf else 'disabled'}`.",
        "",
        "## Overall Winners",
        "",
        "| T | Best Mc | Best Nc | Geomean GFLOPS | Geomean Time_us | Geomean CacheMisses |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for threads, mc, nc, score, time_score, cache_score in winners:
        if mc is None:
            lines.append(f"| {threads} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {threads} | {mc} | {nc} | {score:.6f} | {time_score:.6f} | {cache_score} |")

    lines.extend([
        "",
        "## Family Winners",
        "",
        "| WorkloadFamily | T | Best Mc | Best Nc | Geomean GFLOPS | Geomean Time_us | Geomean CacheMisses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, threads, mc, nc, score, time_score, cache_score in family_winners:
        if mc is None:
            lines.append(f"| {family} | {threads} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {family} | {threads} | {mc} | {nc} | {score:.6f} | {time_score:.6f} | {cache_score} |")

    lines.extend([
        "",
        "## Workload Set",
        "",
        "| WorkloadFamily | WorkloadId | M | K | N |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for family, workload_id, m, k, n in workloads:
        lines.append(f"| {family} | {workload_id} | {m} | {k} | {n} |")

    lines.extend([
        "",
        "This is an empirical table for the current platform, current baseline implementation, fixed kernel shape, fixed Kc, and the selected representative workload set. It is not a global optimality proof.",
    ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if not BENCH.exists():
        raise SystemExit(f"benchmark binary not found: {BENCH}")
    if args.perf_reps <= 0:
        raise SystemExit("--perf-reps must be > 0")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    collect_perf = (not args.skip_perf) and shutil.which("perf") is not None
    threads_list = args.threads if args.threads else THREADS
    mc_candidates = MC_CANDIDATES[:args.limit_mc] if args.limit_mc is not None else MC_CANDIDATES
    nc_candidates = NC_CANDIDATES[:args.limit_nc] if args.limit_nc is not None else NC_CANDIDATES
    workloads = WORKLOADS[:args.limit_workloads] if args.limit_workloads is not None else WORKLOADS

    rows = []
    for threads in threads_list:
        for mc in mc_candidates:
            for nc in nc_candidates:
                for workload in workloads:
                    print(f"T={threads} Mc={mc} Nc={nc} workload={workload[1]}", flush=True)
                    rows.append(
                        run_one(
                            threads,
                            mc,
                            nc,
                            workload,
                            collect_perf=collect_perf,
                            perf_reps=args.perf_reps,
                        )
                    )

    fieldnames = list(rows[0].keys())
    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(
        rows,
        threads_list=threads_list,
        mc_candidates=mc_candidates,
        nc_candidates=nc_candidates,
        workloads=workloads,
        collect_perf=collect_perf,
    )
    print(f"Wrote {RAW_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
