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
DEFAULT_SUMMARY_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "summary"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "plots"
DEFAULT_SHAPE_ORDER = ["8x8", "12x8", "13x8", "4x16", "5x16", "6x16"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the kernel-shape selection heatmap for the single-thread GotoBLAS screening results."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shape-order",
        nargs="+",
        default=DEFAULT_SHAPE_ORDER,
        help="Kernel-shape order used on the x-axis.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Kernel Shape Relative To Best Custom Shape Per Workload",
        help="Heatmap title.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="relative_to_best_custom_heatmap.png",
        help="Output PNG filename under --output-dir.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG export DPI.")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def as_float(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    return float(value)


def workload_sort_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["WorkloadFamily"], row["WorkloadId"])


def workload_label(row: dict[str, str]) -> str:
    size = row.get("Size", "").strip()
    family = row.get("WorkloadFamily", "").strip()
    workload = row.get("WorkloadId", "").strip()
    if size:
        return f"{family}\n{workload}\n{size}"
    return f"{family}\n{workload}"


def best_custom_by_shape(
    candidate_rows: list[dict[str, str]],
    shape_order: list[str],
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    filtered = [
        row for row in candidate_rows
        if row["RunType"] == "custom"
        and row["AggregationStatus"] == "ok"
        and row["KernelShape"] in shape_order
    ]
    workloads = sorted(
        {(row["WorkloadFamily"], row["WorkloadId"]) for row in filtered},
        key=lambda item: item,
    )
    best: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for workload_family, workload_id in workloads:
        rows = [
            row for row in filtered
            if row["WorkloadFamily"] == workload_family and row["WorkloadId"] == workload_id
        ]
        for shape in shape_order:
            shape_rows = [row for row in rows if row["KernelShape"] == shape]
            if not shape_rows:
                continue
            winner = min(
                shape_rows,
                key=lambda row: (
                    float(row["MedianTime_us"]),
                    float(row["BestTime_us"]),
                ),
            )
            best[(workload_family, workload_id)][shape] = float(winner["MedianGFLOPS"])
    workload_labels = []
    for workload_family, workload_id in workloads:
        sample = next(
            row for row in filtered
            if row["WorkloadFamily"] == workload_family and row["WorkloadId"] == workload_id
        )
        workload_labels.append(workload_label(sample))
    return workload_labels, best


def build_matrix(
    workload_keys: list[tuple[str, str]],
    shape_order: list[str],
    values: dict[tuple[str, str], dict[str, float]],
    transform,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for workload_key in workload_keys:
        row_out: list[float] = []
        per_shape = values.get(workload_key, {})
        for shape in shape_order:
            value = per_shape.get(shape)
            row_out.append(transform(workload_key, shape, value))
        matrix.append(row_out)
    return matrix


def value_text(value: float, percent: bool = False) -> str:
    if math.isnan(value):
        return ""
    if percent:
        return f"{value:.1f}%"
    return f"{value:.1f}"


def plot_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    cmap: str,
    percent: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    height = max(6.0, 0.6 * len(row_labels))
    width = max(8.0, 1.2 * len(col_labels) + 4.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    if vmin is None or vmax is None:
        values = [value for row in matrix for value in row if not math.isnan(value)]
        data_min = min(values) if values else 0.0
        data_max = max(values) if values else 1.0
    else:
        data_min = vmin
        data_max = vmax
    norm = Normalize(vmin=data_min, vmax=data_max)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                ax.text(j, i, "NA", ha="center", va="center", color="black", fontsize=8)
                continue
            color = "white" if norm(value) < 0.38 else "black"
            ax.text(
                j,
                i,
                value_text(value, percent=percent),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    candidate_path = args.summary_dir / "candidate_aggregates.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing summary file: {candidate_path}")

    candidate_rows = read_csv(candidate_path)
    workload_labels, best_by_shape = best_custom_by_shape(candidate_rows, args.shape_order)
    workload_keys = sorted(best_by_shape.keys(), key=lambda item: item)

    relative_to_best_matrix = build_matrix(
        workload_keys,
        args.shape_order,
        best_by_shape,
        lambda wk, _shape, value: (
            math.nan
            if value is None
            else value / max(best_by_shape[wk].values()) * 100.0
        ),
    )

    plot_heatmap(
        relative_to_best_matrix,
        workload_labels,
        args.shape_order,
        title=args.title,
        colorbar_label="GFLOPS vs best custom (%)",
        output_path=args.output_dir / args.output_filename,
        dpi=args.dpi,
        cmap="cividis",
        percent=True,
        vmin=45.0,
        vmax=100.0,
    )

    print(f"Wrote selection heatmap to {args.output_dir / args.output_filename}")


if __name__ == "__main__":
    main()
