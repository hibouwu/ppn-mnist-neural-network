#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_ROOT = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "stage2_blocked_tuning"
DEFAULT_INPUT = DEFAULT_ROOT / "raw_results.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "summary"

AGG_FIELDS = [
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "Mc",
    "Nc",
    "Kc",
    "Samples",
    "MedianTime_us",
    "MedianGFLOPS",
    "MeanGFLOPS",
    "BestGFLOPS",
    "RelativeToBestPct",
    "Status",
]

WINNER_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "BestKernelShape",
    "BestMc",
    "BestNc",
    "BestKc",
    "BestMedianGFLOPS",
    "BestMedianTime_us",
]

KERNEL_FIELDS = [
    "KernelShape",
    "WorkloadCount",
    "GeomeanGFLOPS",
    "MeanRelativeToBestPct",
    "MinRelativeToBestPct",
    "BestWorkloads",
]

PARAMETER_AGG_FIELDS = [
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Kc",
    "Mc",
    "Nc",
    "MedianGFLOPS",
    "MeanGFLOPS",
    "NumSamples",
    "RelativeToKernelWorkloadBestPct",
    "RelativeToGlobalWorkloadBestPct",
]

PARAMETER_BEST_PER_WORKLOAD_FIELDS = [
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "BestKc",
    "BestMc",
    "BestNc",
    "BestMedianGFLOPS",
    "BestRelativeToGlobalWorkloadBestPct",
]

PARAMETER_BEST_GLOBAL_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "BestKernelShape",
    "BestKc",
    "BestMc",
    "BestNc",
    "BestMedianGFLOPS",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AVX-512 Stage 2 blocked-size tuning.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def geomean(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("Status") != "ok":
            continue
        key = (
            row["KernelShape"],
            row["WorkloadFamily"],
            row["WorkloadId"],
            row["M"],
            row["K"],
            row["N"],
            row["Threads"],
            row["Mc"],
            row["Nc"],
            row["Kc"],
        )
        grouped[key].append(row)

    out: list[dict[str, str]] = []
    for key, group in grouped.items():
        gflops = [as_float(row["GFLOPS"]) for row in group]
        gflops = [value for value in gflops if value is not None]
        times = [as_float(row["Time_us"]) for row in group]
        times = [value for value in times if value is not None]
        out.append({
            "KernelShape": key[0],
            "WorkloadFamily": key[1],
            "WorkloadId": key[2],
            "M": key[3],
            "K": key[4],
            "N": key[5],
            "Threads": key[6],
            "Mc": key[7],
            "Nc": key[8],
            "Kc": key[9],
            "Samples": str(len(group)),
            "MedianTime_us": fmt(statistics.median(times) if times else None),
            "MedianGFLOPS": fmt(statistics.median(gflops) if gflops else None),
            "MeanGFLOPS": fmt(statistics.mean(gflops) if gflops else None),
            "BestGFLOPS": fmt(max(gflops) if gflops else None),
            "RelativeToBestPct": "",
            "Status": "ok",
        })
    return sorted(out, key=lambda row: (
        row["WorkloadFamily"],
        row["WorkloadId"],
        row["KernelShape"],
        int(row["Kc"]),
        int(row["Mc"]),
        int(row["Nc"]),
    ))


def add_relative_and_winners(aggregates: list[dict[str, str]]) -> list[dict[str, str]]:
    by_workload: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregates:
        by_workload[(row["WorkloadFamily"], row["WorkloadId"])].append(row)

    winners: list[dict[str, str]] = []
    for workload_key, rows in by_workload.items():
        best = max(rows, key=lambda row: float(row["MedianGFLOPS"] or "0"))
        best_gflops = float(best["MedianGFLOPS"])
        for row in rows:
            value = float(row["MedianGFLOPS"]) if row["MedianGFLOPS"] else 0.0
            row["RelativeToBestPct"] = fmt(value / best_gflops * 100.0 if best_gflops > 0 else None)
        winners.append({
            "WorkloadFamily": workload_key[0],
            "WorkloadId": workload_key[1],
            "BestKernelShape": best["KernelShape"],
            "BestMc": best["Mc"],
            "BestNc": best["Nc"],
            "BestKc": best["Kc"],
            "BestMedianGFLOPS": best["MedianGFLOPS"],
            "BestMedianTime_us": best["MedianTime_us"],
        })
    return sorted(winners, key=lambda row: (row["WorkloadFamily"], row["WorkloadId"]))


def build_kernel_summary(aggregates: list[dict[str, str]], winners: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_kernel_workload: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in aggregates:
        key = (row["KernelShape"], row["WorkloadFamily"], row["WorkloadId"])
        current = best_by_kernel_workload.get(key)
        if current is None or float(row["MedianGFLOPS"] or "0") > float(current["MedianGFLOPS"] or "0"):
            best_by_kernel_workload[key] = row

    winner_shapes = defaultdict(int)
    for winner in winners:
        winner_shapes[winner["BestKernelShape"]] += 1

    by_kernel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (kernel, _, _), row in best_by_kernel_workload.items():
        by_kernel[kernel].append(row)

    out: list[dict[str, str]] = []
    for kernel, rows in sorted(by_kernel.items()):
        gf = [float(row["MedianGFLOPS"]) for row in rows if row["MedianGFLOPS"]]
        rel = [float(row["RelativeToBestPct"]) for row in rows if row["RelativeToBestPct"]]
        out.append({
            "KernelShape": kernel,
            "WorkloadCount": str(len(rows)),
            "GeomeanGFLOPS": fmt(geomean(gf)),
            "MeanRelativeToBestPct": fmt(statistics.mean(rel) if rel else None),
            "MinRelativeToBestPct": fmt(min(rel) if rel else None),
            "BestWorkloads": str(winner_shapes[kernel]),
        })
    return sorted(out, key=lambda row: float(row["GeomeanGFLOPS"] or "0"), reverse=True)


def build_parameter_outputs(
    aggregates: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    by_kernel_workload: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    by_workload: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregates:
        by_kernel_workload[(row["KernelShape"], row["WorkloadFamily"], row["WorkloadId"])].append(row)
        by_workload[(row["WorkloadFamily"], row["WorkloadId"])].append(row)

    kernel_workload_best: dict[tuple[str, str, str], dict[str, str]] = {}
    for key, rows in by_kernel_workload.items():
        kernel_workload_best[key] = max(rows, key=lambda row: float(row["MedianGFLOPS"] or "0"))

    global_workload_best: dict[tuple[str, str], dict[str, str]] = {}
    for key, rows in by_workload.items():
        global_workload_best[key] = max(rows, key=lambda row: float(row["MedianGFLOPS"] or "0"))

    parameter_aggregates: list[dict[str, str]] = []
    for row in aggregates:
        kernel_key = (row["KernelShape"], row["WorkloadFamily"], row["WorkloadId"])
        workload_key = (row["WorkloadFamily"], row["WorkloadId"])
        median = float(row["MedianGFLOPS"] or "0")
        kernel_best = float(kernel_workload_best[kernel_key]["MedianGFLOPS"] or "0")
        global_best = float(global_workload_best[workload_key]["MedianGFLOPS"] or "0")
        parameter_aggregates.append({
            "KernelShape": row["KernelShape"],
            "WorkloadFamily": row["WorkloadFamily"],
            "WorkloadId": row["WorkloadId"],
            "M": row["M"],
            "K": row["K"],
            "N": row["N"],
            "Kc": row["Kc"],
            "Mc": row["Mc"],
            "Nc": row["Nc"],
            "MedianGFLOPS": row["MedianGFLOPS"],
            "MeanGFLOPS": row["MeanGFLOPS"],
            "NumSamples": row["Samples"],
            "RelativeToKernelWorkloadBestPct": fmt(median / kernel_best * 100.0 if kernel_best > 0 else None),
            "RelativeToGlobalWorkloadBestPct": fmt(median / global_best * 100.0 if global_best > 0 else None),
        })

    best_per_workload: list[dict[str, str]] = []
    for key, best in sorted(kernel_workload_best.items()):
        best_per_workload.append({
            "KernelShape": key[0],
            "WorkloadFamily": key[1],
            "WorkloadId": key[2],
            "BestKc": best["Kc"],
            "BestMc": best["Mc"],
            "BestNc": best["Nc"],
            "BestMedianGFLOPS": best["MedianGFLOPS"],
            "BestRelativeToGlobalWorkloadBestPct": best["RelativeToBestPct"],
        })

    best_global: list[dict[str, str]] = []
    for key, best in sorted(global_workload_best.items()):
        best_global.append({
            "WorkloadFamily": key[0],
            "WorkloadId": key[1],
            "BestKernelShape": best["KernelShape"],
            "BestKc": best["Kc"],
            "BestMc": best["Mc"],
            "BestNc": best["Nc"],
            "BestMedianGFLOPS": best["MedianGFLOPS"],
        })

    return parameter_aggregates, best_per_workload, best_global


def write_markdown(
    path: Path,
    aggregates: list[dict[str, str]],
    winners: list[dict[str, str]],
    kernel_summary: list[dict[str, str]],
) -> None:
    best_overall = kernel_summary[0] if kernel_summary else None
    small_n_winner = next((row for row in winners if row["WorkloadId"] == "mlp_fc2_b32"), None)
    default_path_ok = best_overall is not None and best_overall["KernelShape"] == "avx512_8x32"
    small_n_ok = small_n_winner is not None and small_n_winner["BestKernelShape"] == "avx512_16x16"

    lines = [
        "# AVX-512 Stage 2 Blocked-Size Tuning Summary",
        "",
        "This summary covers only AVX-512 Stage 2 candidates. No AVX2 comparison has been run here.",
        "",
        f"- Aggregated candidates: {len(aggregates)}",
        "- Candidate scope: `avx512_8x32` default path and `avx512_16x16` small-N fallback only.",
        "- No stricter pruning is applied by this summarizer; it reports whatever candidate rows were present in the raw input.",
        "- Parameter-level CSV outputs retain one row per `KernelShape + workload + Kc + Mc + Nc` combination for direct blocked-size tuning plots.",
        f"- Best overall candidate by cross-workload geomean GFLOPS: {best_overall['KernelShape'] if best_overall else 'n/a'}",
        f"- `avx512_8x32` remains best default-path candidate by geomean: {'yes' if default_path_ok else 'no'}",
        f"- `avx512_16x16` remains best small-N fallback for `mlp_fc2_b32`: {'yes' if small_n_ok else 'no'}",
        "- Absolute GFLOPS are reported in the CSV outputs; relative percentages are not used as a substitute for absolute throughput.",
        "- No AVX2 baseline comparison yet.",
        "",
        "## Best Candidate Per Workload",
        "",
        "| WorkloadFamily | WorkloadId | KernelShape | Mc | Nc | Kc | MedianGFLOPS | MedianTime_us |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in winners:
        lines.append(
            f"| {row['WorkloadFamily']} | {row['WorkloadId']} | {row['BestKernelShape']} | "
            f"{row['BestMc']} | {row['BestNc']} | {row['BestKc']} | "
            f"{row['BestMedianGFLOPS']} | {row['BestMedianTime_us']} |"
        )

    lines.extend([
        "",
        "## Kernel Summary",
        "",
        "| KernelShape | WorkloadCount | GeomeanGFLOPS | MeanRelativeToBestPct | MinRelativeToBestPct | BestWorkloads |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in kernel_summary:
        lines.append(
            f"| {row['KernelShape']} | {row['WorkloadCount']} | {row['GeomeanGFLOPS']} | "
            f"{row['MeanRelativeToBestPct']} | {row['MinRelativeToBestPct']} | {row['BestWorkloads']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    aggregates = aggregate(rows)
    winners = add_relative_and_winners(aggregates)
    kernel_summary = build_kernel_summary(aggregates, winners)
    parameter_aggregates, parameter_best_per_workload, parameter_best_global = build_parameter_outputs(aggregates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_aggregates.csv", AGG_FIELDS, aggregates)
    write_csv(args.output_dir / "workload_winners.csv", WINNER_FIELDS, winners)
    write_csv(args.output_dir / "kernel_summary.csv", KERNEL_FIELDS, kernel_summary)
    write_csv(args.output_dir / "parameter_aggregates.csv", PARAMETER_AGG_FIELDS, parameter_aggregates)
    write_csv(
        args.output_dir / "parameter_best_per_workload.csv",
        PARAMETER_BEST_PER_WORKLOAD_FIELDS,
        parameter_best_per_workload,
    )
    write_csv(args.output_dir / "parameter_best_global.csv", PARAMETER_BEST_GLOBAL_FIELDS, parameter_best_global)
    write_markdown(args.output_dir / "overall_summary.md", aggregates, winners, kernel_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
