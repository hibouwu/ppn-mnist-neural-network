#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "thread_aware_mc_nc_baseline"
DEFAULT_RAW_CSV = DEFAULT_EXPERIMENT_DIR / "raw.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "plots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Mc/Nc heatmaps for the thread-aware baseline GotoBLAS experiment."
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def value_text(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.1f}"


def plot_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    height = max(5.0, 0.8 * len(row_labels) + 2.0)
    width = max(7.0, 0.8 * len(col_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="cividis")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Nc")
    ax.set_ylabel("Mc")
    ax.set_title(title)

    values = [value for row in matrix for value in row if not math.isnan(value)]
    data_min = min(values) if values else 0.0
    data_max = max(values) if values else 1.0
    norm = Normalize(vmin=data_min, vmax=data_max)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                ax.text(j, i, "NA", ha="center", va="center", color="black", fontsize=8)
                continue
            color = "white" if norm(value) < 0.38 else "black"
            ax.text(j, i, value_text(value), ha="center", va="center", color=color, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def aggregate_metric(rows: list[dict[str, str]], metric: str) -> float | None:
    if metric == "gflops":
        values = [float(row["GFLOPS"]) for row in rows if row["GFLOPS"]]
    elif metric == "time_us":
        values = [1e6 * float(row["Mean_s"]) for row in rows if row["Mean_s"]]
    elif metric == "cache_misses":
        if not all(row.get("PerfStatus") == "ok" and row.get("CacheMisses") for row in rows):
            return None
        values = [float(row["CacheMisses"]) for row in rows]
    elif metric == "cache_misses_per_gflops":
        if not all(row.get("PerfStatus") == "ok" and row.get("CacheMisses") and row.get("GFLOPS") for row in rows):
            return None
        values = [float(row["CacheMisses"]) / float(row["GFLOPS"]) for row in rows if float(row["GFLOPS"]) > 0.0]
    else:
        raise ValueError(f"unsupported metric: {metric}")

    if not values or any(value <= 0.0 for value in values):
        return None
    return geomean(values)


def build_score_maps(
    rows: list[dict[str, str]],
    metric: str,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[str],
    dict[int, dict[tuple[int, int], float]],
    dict[tuple[str, int], dict[tuple[int, int], float]],
]:
    ok_rows = [row for row in rows if row["Status"] == "ok"]
    threads = sorted({int(row["T"]) for row in ok_rows})
    mcs = sorted({int(row["Mc"]) for row in ok_rows})
    ncs = sorted({int(row["Nc"]) for row in ok_rows})
    families = sorted({row["WorkloadFamily"] for row in ok_rows})
    workload_count = len({(row["WorkloadFamily"], row["WorkloadId"]) for row in ok_rows})

    overall_scores: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    family_scores: dict[tuple[str, int], dict[tuple[int, int], float]] = defaultdict(dict)

    by_t_mc_nc: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    by_family_t_mc_nc: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    family_workloads: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for row in ok_rows:
        t = int(row["T"])
        mc = int(row["Mc"])
        nc = int(row["Nc"])
        family = row["WorkloadFamily"]
        workload = (family, row["WorkloadId"])
        by_t_mc_nc[(t, mc, nc)].append(row)
        by_family_t_mc_nc[(family, t, mc, nc)].append(row)
        family_workloads[family].add(workload)

    for t in threads:
        for mc in mcs:
            for nc in ncs:
                subset = by_t_mc_nc.get((t, mc, nc), [])
                if len(subset) == workload_count:
                    value = aggregate_metric(subset, metric)
                    if value is not None:
                        overall_scores[t][(mc, nc)] = value

    for family in families:
        expected = len(family_workloads[family])
        for t in threads:
            for mc in mcs:
                for nc in ncs:
                    subset = by_family_t_mc_nc.get((family, t, mc, nc), [])
                    if len(subset) == expected:
                        value = aggregate_metric(subset, metric)
                        if value is not None:
                            family_scores[(family, t)][(mc, nc)] = value

    return threads, mcs, ncs, families, overall_scores, family_scores


def matrix_for_grid(
    mcs: list[int],
    ncs: list[int],
    scores: dict[tuple[int, int], float],
) -> list[list[float]]:
    out: list[list[float]] = []
    for mc in mcs:
        row = []
        for nc in ncs:
            row.append(scores.get((mc, nc), math.nan))
        out.append(row)
    return out


def emit_metric_heatmaps(
    rows: list[dict[str, str]],
    *,
    metric: str,
    metric_label: str,
    filename_prefix: str,
    args: argparse.Namespace,
) -> None:
    threads, mcs, ncs, families, overall_scores, family_scores = build_score_maps(rows, metric)

    for t in threads:
        matrix = matrix_for_grid(mcs, ncs, overall_scores[t])
        plot_heatmap(
            matrix,
            [str(mc) for mc in mcs],
            [str(nc) for nc in ncs],
            title=f"Thread-Aware Baseline Mc/Nc Heatmap ({metric_label}, T={t})",
            colorbar_label=metric_label,
            output_path=args.output_dir / f"{filename_prefix}_overall_T{t}.png",
            dpi=args.dpi,
        )

    for family in families:
        safe_family = family.replace("/", "_")
        for t in threads:
            matrix = matrix_for_grid(mcs, ncs, family_scores[(family, t)])
            plot_heatmap(
                matrix,
                [str(mc) for mc in mcs],
                [str(nc) for nc in ncs],
                title=f"{family} Mc/Nc Heatmap ({metric_label}, T={t})",
                colorbar_label=metric_label,
                output_path=args.output_dir / f"{filename_prefix}_{safe_family}_T{t}.png",
                dpi=args.dpi,
            )


def main() -> None:
    args = build_parser().parse_args()
    rows = read_csv(args.raw_csv)

    emit_metric_heatmaps(
        rows,
        metric="gflops",
        metric_label="Geomean GFLOPS",
        filename_prefix="gflops",
        args=args,
    )
    emit_metric_heatmaps(
        rows,
        metric="time_us",
        metric_label="Geomean Time_us",
        filename_prefix="time_us",
        args=args,
    )
    emit_metric_heatmaps(
        rows,
        metric="cache_misses",
        metric_label="Geomean CacheMisses",
        filename_prefix="cache_misses",
        args=args,
    )
    emit_metric_heatmaps(
        rows,
        metric="cache_misses_per_gflops",
        metric_label="Geomean CacheMisses/GFLOPS",
        filename_prefix="cache_misses_per_gflops",
        args=args,
    )

    print(f"Wrote heatmaps under {args.output_dir}")


if __name__ == "__main__":
    main()
