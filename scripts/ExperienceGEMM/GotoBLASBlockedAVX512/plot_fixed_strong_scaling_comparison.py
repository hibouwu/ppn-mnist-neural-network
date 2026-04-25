#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_SUMMARY_DIR = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "fixed_strong_scaling_comparison"
    / "summary"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "plots"
IMPL_ORDER = ["AVX2_GotoBLAS", "AVX512_GotoBLAS", "OpenBLAS"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot fixed strong-scaling comparison summaries.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def plot_lines(
    rows: list[dict[str, str]],
    *,
    y_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    implementations: list[str] | None = None,
    ideal_speedup: bool = False,
) -> None:
    implementations = implementations or IMPL_ORDER
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    all_threads = sorted({int(row["Threads"]) for row in rows})
    for impl in implementations:
        impl_rows = sorted(
            [row for row in rows if row["ImplementationLabel"] == impl],
            key=lambda row: int(row["Threads"]),
        )
        xs = [int(row["Threads"]) for row in impl_rows if as_float(row.get(y_field, "")) is not None]
        ys = [as_float(row[y_field]) for row in impl_rows if as_float(row.get(y_field, "")) is not None]
        if xs and ys:
            ax.plot(xs, ys, marker="o", label=impl)
    if ideal_speedup and all_threads:
        ax.plot(all_threads, all_threads, linestyle="--", label="ideal")
    ax.set_xlabel("Threads")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(all_threads)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_winner_heatmap(rows: list[dict[str, str]], output_path: Path, dpi: int) -> None:
    workloads = sorted({(row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"]) for row in rows})
    threads = sorted({int(row["Threads"]) for row in rows})
    value_by_key = {
        (row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"], int(row["Threads"])): row["BestImplementationLabel"]
        for row in rows
    }
    label_to_value = {label: index for index, label in enumerate(IMPL_ORDER)}
    matrix: list[list[float]] = []
    annotations: list[list[str]] = []
    for workload in workloads:
        mrow: list[float] = []
        arow: list[str] = []
        for thread in threads:
            label = value_by_key.get((*workload, thread), "")
            mrow.append(float(label_to_value[label]) if label in label_to_value else math.nan)
            arow.append(label)
        matrix.append(mrow)
        annotations.append(arow)

    fig, ax = plt.subplots(
        figsize=(max(6.5, 1.2 * len(threads) + 4.5), max(4.8, 0.7 * len(workloads) + 2.0)),
        constrained_layout=True,
    )
    ax.imshow(matrix, aspect="auto", vmin=0, vmax=max(1, len(IMPL_ORDER) - 1))
    ax.set_xticks(range(len(threads)))
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels([f"{family}\n{workload_id}\n{m}x{k}x{n}" for family, workload_id, m, k, n in workloads])
    ax.set_xlabel("Threads")
    ax.set_title("Fixed-Config Strong-Scaling Comparison: Workload Winners")
    for i, row in enumerate(annotations):
        for j, label in enumerate(row):
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_family_geomean(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    families = sorted({row["WorkloadFamily"] for row in rows})
    for family in families:
        family_rows = [row for row in rows if row["WorkloadFamily"] == family]
        path = output_dir / f"family_geomean_gflops_by_threads_{family}.png"
        plot_lines(
            family_rows,
            y_field="GeomeanGFLOPS",
            title=f"Fixed-Config Strong-Scaling Comparison: {family}",
            ylabel="Family Geomean GFLOPS",
            output_path=path,
            dpi=dpi,
        )
        outputs.append(path)
    return outputs


def plot_individual_workloads(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    workloads = sorted({(row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"]) for row in rows})
    for family, workload_id, m, k, n in workloads:
        workload_rows = [
            row for row in rows
            if row["WorkloadFamily"] == family and row["WorkloadId"] == workload_id and row["M"] == m and row["K"] == k and row["N"] == n
        ]
        path = output_dir / f"workload_{family}_{workload_id}_{m}x{k}x{n}.png"
        plot_lines(
            workload_rows,
            y_field="MedianGFLOPS",
            title=f"Workload: {workload_id} ({m}x{k}x{n})",
            ylabel="Median GFLOPS",
            output_path=path,
            dpi=dpi,
        )
        outputs.append(path)
    return outputs


def main() -> int:
    args = build_parser().parse_args()
    overall = read_csv(args.summary_dir / "overall_geomean_by_thread.csv")
    winners = read_csv(args.summary_dir / "winners_by_workload_thread.csv")
    family = read_csv(args.summary_dir / "family_geomean_by_thread.csv")
    relative = read_csv(args.summary_dir / "relative_by_workload_thread.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        args.output_dir / "overall_geomean_gflops_by_threads.png",
        args.output_dir / "overall_speedup_by_threads.png",
        args.output_dir / "overall_efficiency_by_threads.png",
        args.output_dir / "relative_to_openblas_by_threads.png",
        args.output_dir / "relative_to_avx2_by_threads.png",
        args.output_dir / "workload_winners_heatmap.png",
    ]
    plot_lines(
        overall,
        y_field="GeomeanGFLOPS",
        title="Fixed-Config Strong-Scaling Comparison: Overall Geomean GFLOPS",
        ylabel="Geomean GFLOPS",
        output_path=outputs[0],
        dpi=args.dpi,
    )
    plot_lines(
        overall,
        y_field="SpeedupVsT1",
        title="Fixed-Config Strong-Scaling Comparison: Overall Speedup",
        ylabel="Speedup vs T=1",
        output_path=outputs[1],
        dpi=args.dpi,
        ideal_speedup=True,
    )
    plot_lines(
        overall,
        y_field="ParallelEfficiency",
        title="Fixed-Config Strong-Scaling Comparison: Overall Parallel Efficiency",
        ylabel="Parallel Efficiency",
        output_path=outputs[2],
        dpi=args.dpi,
    )
    plot_lines(
        overall,
        y_field="RelativeToOpenBLASPct",
        title="Fixed-Config Strong-Scaling Comparison: Relative To OpenBLAS",
        ylabel="Relative to OpenBLAS (%)",
        output_path=outputs[3],
        dpi=args.dpi,
        implementations=["AVX2_GotoBLAS", "AVX512_GotoBLAS"],
    )
    plot_lines(
        overall,
        y_field="RelativeToAVX2Pct",
        title="Fixed-Config Strong-Scaling Comparison: Relative To AVX2 GotoBLAS",
        ylabel="Relative to AVX2 GotoBLAS (%)",
        output_path=outputs[4],
        dpi=args.dpi,
        implementations=["AVX512_GotoBLAS", "OpenBLAS"],
    )
    plot_winner_heatmap(winners, outputs[5], args.dpi)
    outputs.extend(plot_family_geomean(family, args.output_dir, args.dpi))
    outputs.extend(plot_individual_workloads(relative, args.output_dir, args.dpi))
    print(f"Wrote {len(outputs)} plots under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
