#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_ROOT = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "fixed_strong_scaling_comparison"
)
DEFAULT_INPUT = DEFAULT_ROOT / "raw_results.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "summary"

AGG_FIELDS = [
    "ImplementationLabel",
    "MatmulImpl",
    "KernelShape",
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "Kc",
    "Mc",
    "Nc",
    "MedianTimeUs",
    "MedianGFLOPS",
    "MeanGFLOPS",
    "MinGFLOPS",
    "MaxGFLOPS",
    "NumSamples",
]

SPEEDUP_FIELDS = [
    *AGG_FIELDS,
    "SpeedupVsT1",
    "ParallelEfficiency",
]

RELATIVE_FIELDS = [
    *SPEEDUP_FIELDS,
    "RelativeToOpenBLASPct",
    "RelativeToAVX2Pct",
    "RelativeToBestPct",
    "BestImplementationLabel",
]

OVERALL_FIELDS = [
    "ImplementationLabel",
    "Threads",
    "GeomeanGFLOPS",
    "WorkloadCount",
    "SpeedupVsT1",
    "ParallelEfficiency",
    "RelativeToOpenBLASPct",
    "RelativeToAVX2Pct",
    "RelativeToBestPct",
    "BestImplementationLabel",
]

FAMILY_FIELDS = [
    "ImplementationLabel",
    "Threads",
    "WorkloadFamily",
    "GeomeanGFLOPS",
    "WorkloadCount",
]

WINNER_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "BestImplementationLabel",
    "BestMedianGFLOPS",
]

IMPL_ORDER = ["AVX2_GotoBLAS", "AVX512_GotoBLAS", "OpenBLAS"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize fixed strong-scaling comparison.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def as_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
            row["ImplementationLabel"],
            row["MatmulImpl"],
            row["KernelShape"],
            row["WorkloadFamily"],
            row["WorkloadId"],
            row["M"],
            row["K"],
            row["N"],
            row["Threads"],
            row["Kc"],
            row["Mc"],
            row["Nc"],
        )
        grouped[key].append(row)

    out: list[dict[str, str]] = []
    for key, group in grouped.items():
        times = [as_float(row.get("Time_us", "")) for row in group]
        times = [value for value in times if value is not None]
        gflops = [as_float(row.get("GFLOPS", "")) for row in group]
        gflops = [value for value in gflops if value is not None]
        out.append({
            "ImplementationLabel": key[0],
            "MatmulImpl": key[1],
            "KernelShape": key[2],
            "WorkloadFamily": key[3],
            "WorkloadId": key[4],
            "M": key[5],
            "K": key[6],
            "N": key[7],
            "Threads": key[8],
            "Kc": key[9],
            "Mc": key[10],
            "Nc": key[11],
            "MedianTimeUs": fmt(statistics.median(times) if times else None),
            "MedianGFLOPS": fmt(statistics.median(gflops) if gflops else None),
            "MeanGFLOPS": fmt(statistics.mean(gflops) if gflops else None),
            "MinGFLOPS": fmt(min(gflops) if gflops else None),
            "MaxGFLOPS": fmt(max(gflops) if gflops else None),
            "NumSamples": str(len(group)),
        })
    return sorted(out, key=lambda row: (
        row["WorkloadFamily"],
        row["WorkloadId"],
        int(row["Threads"]),
        IMPL_ORDER.index(row["ImplementationLabel"]) if row["ImplementationLabel"] in IMPL_ORDER else 99,
    ))


def add_speedup(aggregates: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline: dict[tuple[str, str, str, str, str, str], float] = {}
    for row in aggregates:
        if row["Threads"] != "1":
            continue
        key = (
            row["ImplementationLabel"],
            row["WorkloadFamily"],
            row["WorkloadId"],
            row["M"],
            row["K"],
            row["N"],
        )
        value = as_float(row["MedianGFLOPS"])
        if value is not None and value > 0.0:
            baseline[key] = value

    out: list[dict[str, str]] = []
    for row in aggregates:
        key = (
            row["ImplementationLabel"],
            row["WorkloadFamily"],
            row["WorkloadId"],
            row["M"],
            row["K"],
            row["N"],
        )
        value = as_float(row["MedianGFLOPS"])
        base = baseline.get(key)
        speedup = value / base if value is not None and base else None
        threads = int(row["Threads"])
        out.append({
            **row,
            "SpeedupVsT1": fmt(speedup),
            "ParallelEfficiency": fmt(speedup / threads if speedup is not None and threads > 0 else None),
        })
    return out


def add_relative(speedup_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_workload_thread: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in speedup_rows:
        key = (row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"], row["Threads"])
        by_workload_thread[key].append(row)

    relative: list[dict[str, str]] = []
    winners: list[dict[str, str]] = []
    for key, rows in by_workload_thread.items():
        value_by_impl = {
            row["ImplementationLabel"]: as_float(row["MedianGFLOPS"])
            for row in rows
            if as_float(row["MedianGFLOPS"]) is not None
        }
        best_label = ""
        best_value = None
        if value_by_impl:
            best_label, best_value = max(value_by_impl.items(), key=lambda item: item[1])
        openblas = value_by_impl.get("OpenBLAS")
        avx2 = value_by_impl.get("AVX2_GotoBLAS")
        for row in rows:
            value = as_float(row["MedianGFLOPS"])
            relative.append({
                **row,
                "RelativeToOpenBLASPct": fmt(value / openblas * 100.0 if value is not None and openblas else None),
                "RelativeToAVX2Pct": fmt(value / avx2 * 100.0 if value is not None and avx2 else None),
                "RelativeToBestPct": fmt(value / best_value * 100.0 if value is not None and best_value else None),
                "BestImplementationLabel": best_label,
            })
        winners.append({
            "WorkloadFamily": key[0],
            "WorkloadId": key[1],
            "M": key[2],
            "K": key[3],
            "N": key[4],
            "Threads": key[5],
            "BestImplementationLabel": best_label,
            "BestMedianGFLOPS": fmt(best_value),
        })
    winners.sort(key=lambda row: (row["WorkloadFamily"], row["WorkloadId"], int(row["Threads"])))
    return relative, winners


def build_overall(relative_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in relative_rows:
        grouped[(row["ImplementationLabel"], row["Threads"])].append(row)

    raw: list[dict[str, str]] = []
    for key, group in grouped.items():
        values = [as_float(row["MedianGFLOPS"]) for row in group]
        values = [value for value in values if value is not None]
        raw.append({
            "ImplementationLabel": key[0],
            "Threads": key[1],
            "GeomeanGFLOPS": fmt(geomean(values)),
            "WorkloadCount": str(len(values)),
        })

    baseline_by_impl = {
        row["ImplementationLabel"]: as_float(row["GeomeanGFLOPS"])
        for row in raw
        if row["Threads"] == "1"
    }
    by_thread: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_thread[row["Threads"]].append(row)

    out: list[dict[str, str]] = []
    for row in raw:
        value = as_float(row["GeomeanGFLOPS"])
        base = baseline_by_impl.get(row["ImplementationLabel"])
        speedup = value / base if value is not None and base else None
        threads = int(row["Threads"])
        peers = by_thread[row["Threads"]]
        peer_values = {
            peer["ImplementationLabel"]: as_float(peer["GeomeanGFLOPS"])
            for peer in peers
            if as_float(peer["GeomeanGFLOPS"]) is not None
        }
        best_label = ""
        best_value = None
        if peer_values:
            best_label, best_value = max(peer_values.items(), key=lambda item: item[1])
        openblas = peer_values.get("OpenBLAS")
        avx2 = peer_values.get("AVX2_GotoBLAS")
        out.append({
            **row,
            "SpeedupVsT1": fmt(speedup),
            "ParallelEfficiency": fmt(speedup / threads if speedup is not None and threads > 0 else None),
            "RelativeToOpenBLASPct": fmt(value / openblas * 100.0 if value is not None and openblas else None),
            "RelativeToAVX2Pct": fmt(value / avx2 * 100.0 if value is not None and avx2 else None),
            "RelativeToBestPct": fmt(value / best_value * 100.0 if value is not None and best_value else None),
            "BestImplementationLabel": best_label,
        })
    return sorted(out, key=lambda row: (
        int(row["Threads"]),
        IMPL_ORDER.index(row["ImplementationLabel"]) if row["ImplementationLabel"] in IMPL_ORDER else 99,
    ))


def build_family(relative_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in relative_rows:
        grouped[(row["ImplementationLabel"], row["Threads"], row["WorkloadFamily"])].append(row)
    out: list[dict[str, str]] = []
    for key, group in grouped.items():
        values = [as_float(row["MedianGFLOPS"]) for row in group]
        values = [value for value in values if value is not None]
        out.append({
            "ImplementationLabel": key[0],
            "Threads": key[1],
            "WorkloadFamily": key[2],
            "GeomeanGFLOPS": fmt(geomean(values)),
            "WorkloadCount": str(len(values)),
        })
    return sorted(out, key=lambda row: (
        row["WorkloadFamily"],
        int(row["Threads"]),
        IMPL_ORDER.index(row["ImplementationLabel"]) if row["ImplementationLabel"] in IMPL_ORDER else 99,
    ))


def markdown_table(lines: list[str], fields: list[str], rows: list[dict[str, str]]) -> None:
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")


def write_summary_md(
    path: Path,
    aggregates: list[dict[str, str]],
    relative_rows: list[dict[str, str]],
    overall: list[dict[str, str]],
    winners: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
) -> None:
    status_counts = Counter(row.get("Status", "") for row in raw_rows)
    workload_keys = sorted({
        (row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"])
        for row in aggregates
    })
    missing_warnings: list[str] = []
    expected_impls = set(IMPL_ORDER)
    by_workload_thread: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    for row in relative_rows:
        key = (row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"], row["Threads"])
        by_workload_thread[key].add(row["ImplementationLabel"])
    for key, seen in sorted(by_workload_thread.items()):
        missing = sorted(expected_impls - seen)
        if missing:
            missing_warnings.append(f"{key}: missing {missing}")

    lines = [
        "# Fixed-Config Strong-Scaling Comparison Summary",
        "",
        "This is a fixed-config strong-scaling comparison. No universal superiority claim is made.",
        "OpenBLAS uses its own internal threading through `OPENBLAS_NUM_THREADS`; AVX2 and AVX-512 use the custom OpenMP GotoBLAS paths.",
        "",
        "## Fixed Configs",
        "",
        "| ImplementationLabel | MATMUL_IMPL | KernelShape | Kc | Mc | Nc | Threading |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
        "| AVX2_GotoBLAS | omp_gotoblas_avx2 | avx2_8x8 | 384 | 8 | 448 | OMP_NUM_THREADS |",
        "| AVX512_GotoBLAS | omp_gotoblas_avx512 | avx512_8x32 | 160 | 8 | 448 | OMP_NUM_THREADS |",
        "| OpenBLAS | blas | n/a | n/a | n/a | n/a | OPENBLAS_NUM_THREADS |",
        "",
        "## Workload Set",
        "",
        "| WorkloadFamily | WorkloadId | M | K | N |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family, workload_id, m, k, n in workload_keys:
        lines.append(f"| {family} | {workload_id} | {m} | {k} | {n} |")

    lines.extend(["", "## Overall Geomean Scaling", ""])
    markdown_table(
        lines,
        [
            "ImplementationLabel",
            "Threads",
            "GeomeanGFLOPS",
            "SpeedupVsT1",
            "ParallelEfficiency",
            "RelativeToOpenBLASPct",
            "RelativeToAVX2Pct",
            "RelativeToBestPct",
        ],
        overall,
    )

    lines.extend(["", "## Per-Workload Winners", ""])
    markdown_table(lines, WINNER_FIELDS, winners)

    relative_short = [
        {
            "ImplementationLabel": row["ImplementationLabel"],
            "WorkloadId": row["WorkloadId"],
            "Threads": row["Threads"],
            "MedianGFLOPS": row["MedianGFLOPS"],
            "SpeedupVsT1": row["SpeedupVsT1"],
            "ParallelEfficiency": row["ParallelEfficiency"],
            "RelativeToOpenBLASPct": row["RelativeToOpenBLASPct"],
            "RelativeToAVX2Pct": row["RelativeToAVX2Pct"],
        }
        for row in relative_rows
    ]
    lines.extend(["", "## Workload-Level Speedup And Relative Ratios", ""])
    markdown_table(
        lines,
        [
            "ImplementationLabel",
            "WorkloadId",
            "Threads",
            "MedianGFLOPS",
            "SpeedupVsT1",
            "ParallelEfficiency",
            "RelativeToOpenBLASPct",
            "RelativeToAVX2Pct",
        ],
        relative_short,
    )

    lines.extend([
        "",
        "## Warnings",
        "",
        f"- Raw status counts: `{dict(status_counts)}`.",
    ])
    if missing_warnings:
        lines.append("- Missing relative baselines:")
        for warning in missing_warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- No missing implementation baselines in the aggregated rows.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    raw_rows = read_rows(args.input)
    aggregates = aggregate(raw_rows)
    speedup_rows = add_speedup(aggregates)
    relative_rows, winners = add_relative(speedup_rows)
    overall = build_overall(relative_rows)
    family = build_family(relative_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregates.csv", AGG_FIELDS, aggregates)
    write_csv(args.output_dir / "speedup_by_workload.csv", SPEEDUP_FIELDS, speedup_rows)
    write_csv(args.output_dir / "relative_by_workload_thread.csv", RELATIVE_FIELDS, relative_rows)
    write_csv(args.output_dir / "overall_geomean_by_thread.csv", OVERALL_FIELDS, overall)
    write_csv(args.output_dir / "family_geomean_by_thread.csv", FAMILY_FIELDS, family)
    write_csv(args.output_dir / "winners_by_workload_thread.csv", WINNER_FIELDS, winners)
    write_summary_md(args.output_dir / "overall_summary.md", aggregates, relative_rows, overall, winners, raw_rows)
    print(f"Wrote summary under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
