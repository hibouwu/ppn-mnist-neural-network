#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import AGG_FIELDS, RAW_FIELDS, fmt_float, load_json, median_or_blank, min_or_blank, parse_size, stdev_or_blank


GROUP_KEYS = [
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
]

ROW_WINNER_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "KernelShape",
    "Kc",
    "WinningMc",
    "WinningNc",
    "MedianTime_us",
    "BestTime_us",
    "MedianGFLOPS",
    "BestGFLOPS",
    "TimingSamples",
    "PerfSamples",
]

CROSS_ROW_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "BestCustomKernelShape",
    "BestCustomMc",
    "BestCustomNc",
    "BestCustomKc",
    "BestCustomMedianTime_us",
    "BestCustomBestTime_us",
    "BestCustomMedianGFLOPS",
    "OpenBLASMedianTime_us",
    "OpenBLASBestTime_us",
    "OpenBLASMedianGFLOPS",
    "MedianTimeVsOpenBLAS_pct",
    "MedianGFLOPSVsOpenBLAS_pct",
]


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    default_config = script_dir / "config" / "single_thread_blocked_candidates.json"
    raw_dir = repo_root / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "raw"
    summary_dir = repo_root / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "summary"
    parser = argparse.ArgumentParser(description="Summarize single-thread blocked candidate results.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--raw-dir", type=Path, default=raw_dir)
    parser.add_argument("--summary-dir", type=Path, default=summary_dir)
    parser.add_argument(
        "--summary-title",
        type=str,
        default="Single-Thread Blocked Candidate Summary",
        help="Markdown summary title.",
    )
    return parser


def read_raw_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            missing = [field for field in RAW_FIELDS if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"missing fields in {path}: {', '.join(missing)}")
            rows.extend(reader)
    return rows


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    return float(value)


def aggregate_groups(rows: list[dict[str, str]], expected_timing: int, expected_perf: int) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_KEYS)
        grouped[key].append(row)

    out: list[dict[str, str]] = []
    perf_metric_fields = [
        "Instructions",
        "Cycles",
        "IPC",
        "CacheMisses",
        "CacheReferences",
        "L1DLoads",
        "L1DMisses",
        "DTLBLoads",
        "DTLBMisses",
        "ContextSwitches",
        "CpuMigrations",
    ]
    for key, group_rows in grouped.items():
        base = {field: value for field, value in zip(GROUP_KEYS, key)}
        timing_rows = [row for row in group_rows if row["MeasurementKind"] == "timing" and row["Status"] == "ok"]
        perf_rows = [row for row in group_rows if row["MeasurementKind"] == "perf" and row["Status"] == "ok"]
        unsupported = any(row["Status"] == "unsupported_kernel_shape" for row in group_rows)
        failed = any(row["Status"] == "failed" for row in group_rows)
        invalid = any(row["Status"] == "invalid" for row in group_rows)

        timing_times = [as_float(row, "Time_us") for row in timing_rows]
        timing_times = [value for value in timing_times if value is not None]
        timing_gflops = [as_float(row, "GFLOPS") for row in timing_rows]
        timing_gflops = [value for value in timing_gflops if value is not None]

        agg = {field: "" for field in AGG_FIELDS}
        agg.update(base)
        agg["TimingSamples"] = str(len(timing_rows))
        agg["PerfSamples"] = str(len(perf_rows))
        agg["ExpectedTimingSamples"] = str(expected_timing)
        agg["ExpectedPerfSamples"] = str(expected_perf if base["RunType"] == "custom" else 0)
        agg["MedianTime_us"] = median_or_blank(timing_times)
        agg["BestTime_us"] = min_or_blank(timing_times)
        agg["TimeStdDev_us"] = stdev_or_blank(timing_times)
        agg["MedianGFLOPS"] = median_or_blank(timing_gflops)
        agg["BestGFLOPS"] = fmt_float(max(timing_gflops)) if timing_gflops else ""

        for metric in perf_metric_fields:
            values = [as_float(row, metric) for row in perf_rows]
            values = [value for value in values if value is not None]
            agg[metric] = median_or_blank(values)
            std_key = "Instr_StdDev" if metric == "Instructions" else f"{metric}_StdDev"
            agg[std_key] = stdev_or_blank(values)

        if unsupported and not timing_rows:
            agg["AggregationStatus"] = "unsupported_kernel_shape"
        elif invalid:
            agg["AggregationStatus"] = "invalid"
        elif failed and not timing_rows:
            agg["AggregationStatus"] = "failed"
        elif not timing_rows:
            agg["AggregationStatus"] = "no_valid_timing"
        elif base["RunType"] == "openblas_baseline":
            agg["AggregationStatus"] = "ok" if len(timing_rows) == expected_timing else "timing_incomplete"
        else:
            if len(timing_rows) == expected_timing and len(perf_rows) == expected_perf:
                agg["AggregationStatus"] = "ok"
            elif len(timing_rows) == expected_timing:
                agg["AggregationStatus"] = "perf_incomplete"
            else:
                agg["AggregationStatus"] = "timing_incomplete"
        out.append(agg)
    return out


def float_or_inf(value: str) -> float:
    return float(value) if value else float("inf")


def build_row_winners(aggregates: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregates:
        if row["RunType"] != "custom" or row["AggregationStatus"] != "ok":
            continue
        key = (row["WorkloadFamily"], row["WorkloadId"], row["KernelShape"], row["Kc"])
        grouped[key].append(row)

    winners: list[dict[str, str]] = []
    for key, rows in grouped.items():
        winner = min(rows, key=lambda row: (float_or_inf(row["MedianTime_us"]), float_or_inf(row["BestTime_us"])))
        winners.append({
            "WorkloadFamily": key[0],
            "WorkloadId": key[1],
            "KernelShape": key[2],
            "Kc": key[3],
            "WinningMc": winner["Mc"],
            "WinningNc": winner["Nc"],
            "MedianTime_us": winner["MedianTime_us"],
            "BestTime_us": winner["BestTime_us"],
            "MedianGFLOPS": winner["MedianGFLOPS"],
            "BestGFLOPS": winner["BestGFLOPS"],
            "TimingSamples": winner["TimingSamples"],
            "PerfSamples": winner["PerfSamples"],
        })
    return sorted(winners, key=lambda row: (row["WorkloadFamily"], row["WorkloadId"], row["KernelShape"], int(row["Kc"])))


def build_cross_row_summary(
    aggregates: list[dict[str, str]],
    row_winners: list[dict[str, str]],
) -> list[dict[str, str]]:
    best_custom_by_workload: dict[tuple[str, str], dict[str, str]] = {}
    for row in row_winners:
        key = (row["WorkloadFamily"], row["WorkloadId"])
        if key not in best_custom_by_workload:
            best_custom_by_workload[key] = row
            continue
        current = best_custom_by_workload[key]
        if (float_or_inf(row["MedianTime_us"]), float_or_inf(row["BestTime_us"])) < (
            float_or_inf(current["MedianTime_us"]),
            float_or_inf(current["BestTime_us"]),
        ):
            best_custom_by_workload[key] = row

    baseline_by_workload: dict[tuple[str, str], dict[str, str]] = {}
    for row in aggregates:
        if row["RunType"] == "openblas_baseline" and row["AggregationStatus"] == "ok":
            baseline_by_workload[(row["WorkloadFamily"], row["WorkloadId"])] = row

    out: list[dict[str, str]] = []
    for key, custom in sorted(best_custom_by_workload.items()):
        baseline = baseline_by_workload.get(key)
        custom_median = float(custom["MedianTime_us"])
        custom_gflops = float(custom["MedianGFLOPS"])
        baseline_median = float(baseline["MedianTime_us"]) if baseline and baseline["MedianTime_us"] else None
        baseline_gflops = float(baseline["MedianGFLOPS"]) if baseline and baseline["MedianGFLOPS"] else None
        time_pct = ""
        gflops_pct = ""
        if baseline_median and baseline_median > 0.0:
            time_pct = fmt_float((custom_median / baseline_median - 1.0) * 100.0)
        if baseline_gflops and baseline_gflops > 0.0:
            gflops_pct = fmt_float((custom_gflops / baseline_gflops - 1.0) * 100.0)
        out.append({
            "WorkloadFamily": key[0],
            "WorkloadId": key[1],
            "BestCustomKernelShape": custom["KernelShape"],
            "BestCustomMc": custom["WinningMc"],
            "BestCustomNc": custom["WinningNc"],
            "BestCustomKc": custom["Kc"],
            "BestCustomMedianTime_us": custom["MedianTime_us"],
            "BestCustomBestTime_us": custom["BestTime_us"],
            "BestCustomMedianGFLOPS": custom["MedianGFLOPS"],
            "OpenBLASMedianTime_us": baseline["MedianTime_us"] if baseline else "",
            "OpenBLASBestTime_us": baseline["BestTime_us"] if baseline else "",
            "OpenBLASMedianGFLOPS": baseline["MedianGFLOPS"] if baseline else "",
            "MedianTimeVsOpenBLAS_pct": time_pct,
            "MedianGFLOPSVsOpenBLAS_pct": gflops_pct,
        })
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    row_winners: list[dict[str, str]],
    cross_rows: list[dict[str, str]],
    aggregates: list[dict[str, str]],
    title: str,
) -> None:
    unsupported = [row for row in aggregates if row["AggregationStatus"] == "unsupported_kernel_shape"]
    lines = [
        f"# {title}",
        "",
        f"- Row winners: {len(row_winners)}",
        f"- Cross-row summaries: {len(cross_rows)}",
        f"- Unsupported kernel-shape groups under current binary: {len(unsupported)}",
        "",
        "## Cross-Row Best Custom vs OpenBLAS",
        "",
        "| WorkloadFamily | WorkloadId | BestCustomKernelShape | Mc | Nc | Kc | BestCustomMedianTime_us | OpenBLASMedianTime_us | MedianGFLOPSVsOpenBLAS_pct |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cross_rows:
        lines.append(
            f"| {row['WorkloadFamily']} | {row['WorkloadId']} | {row['BestCustomKernelShape']} | "
            f"{row['BestCustomMc']} | {row['BestCustomNc']} | {row['BestCustomKc']} | "
            f"{row['BestCustomMedianTime_us']} | {row['OpenBLASMedianTime_us']} | {row['MedianGFLOPSVsOpenBLAS_pct']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    config = load_json(args.config)
    protocol = config["execution_protocol"]
    expected_timing = int(protocol["timing_reps"])
    expected_perf = int(protocol["perf_reps"])
    rows = read_raw_rows(args.raw_dir)
    aggregates = aggregate_groups(rows, expected_timing, expected_perf)
    row_winners = build_row_winners(aggregates)
    cross_rows = build_cross_row_summary(aggregates, row_winners)

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.summary_dir / "candidate_aggregates.csv", AGG_FIELDS, aggregates)
    write_csv(args.summary_dir / "row_winners.csv", ROW_WINNER_FIELDS, row_winners)
    write_csv(args.summary_dir / "cross_row_summary.csv", CROSS_ROW_FIELDS, cross_rows)
    write_markdown(args.summary_dir / "summary.md", row_winners, cross_rows, aggregates, args.summary_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
