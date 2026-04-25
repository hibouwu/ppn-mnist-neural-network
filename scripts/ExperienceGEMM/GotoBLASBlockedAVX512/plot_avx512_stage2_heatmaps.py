#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
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
    / "stage2_blocked_tuning"
    / "summary"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "plots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AVX-512 Stage 2 blocked tuning summaries.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def workload_label(row: dict[str, str]) -> str:
    return f"{row['WorkloadFamily']}\n{row['WorkloadId']}\n{row['M']}x{row['K']}x{row['N']}"


def best_per_kernel_workload(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["KernelShape"], row["WorkloadFamily"], row["WorkloadId"])
        current = best.get(key)
        if current is None or float(row["MedianGFLOPS"] or "0") > float(current["MedianGFLOPS"] or "0"):
            best[key] = row
    return list(best.values())


def draw_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    percent: bool,
) -> None:
    height = max(5.0, 0.9 * len(row_labels) + 1.5)
    width = max(7.0, 1.7 * len(col_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    values = [value for row in matrix for value in row if not math.isnan(value)]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    denom = vmax - vmin if vmax > vmin else 1.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                text = "NA"
                color = "black"
            else:
                text = f"{value:.1f}%" if percent else f"{value:.2f}"
                color = "white" if ((value - vmin) / denom) < 0.38 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def draw_parameter_heatmap(
    matrix: list[list[float]],
    annotations: list[list[str]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    height = max(4.6, 0.8 * len(row_labels) + 1.8)
    width = max(6.4, 1.55 * len(col_labels) + 2.8)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    values = [value for row in matrix for value in row if not math.isnan(value)]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Best median GFLOPS")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    denom = vmax - vmin if vmax > vmin else 1.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text = annotations[i][j]
            color = "black"
            if not math.isnan(value):
                color = "white" if ((value - vmin) / denom) < 0.42 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_relative_and_absolute(summary_dir: Path, output_dir: Path, dpi: int) -> None:
    aggregates = read_csv(summary_dir / "candidate_aggregates.csv")
    best_rows = best_per_kernel_workload(aggregates)
    kernels = sorted({row["KernelShape"] for row in best_rows})
    workloads = sorted({(row["WorkloadFamily"], row["WorkloadId"]) for row in best_rows})

    row_by_key = {
        (row["WorkloadFamily"], row["WorkloadId"], row["KernelShape"]): row
        for row in best_rows
    }
    sample_by_workload = {
        (row["WorkloadFamily"], row["WorkloadId"]): row
        for row in best_rows
    }
    labels = [workload_label(sample_by_workload[key]) for key in workloads]

    relative_matrix: list[list[float]] = []
    absolute_matrix: list[list[float]] = []
    for workload in workloads:
        rel_row: list[float] = []
        abs_row: list[float] = []
        for kernel in kernels:
            row = row_by_key.get((workload[0], workload[1], kernel))
            rel_row.append(float(row["RelativeToBestPct"]) if row and row["RelativeToBestPct"] else math.nan)
            abs_row.append(float(row["MedianGFLOPS"]) if row and row["MedianGFLOPS"] else math.nan)
        relative_matrix.append(rel_row)
        absolute_matrix.append(abs_row)

    draw_heatmap(
        relative_matrix,
        labels,
        kernels,
        "AVX-512 Stage 2 Best-Per-Kernel Summary: Relative To Workload Best",
        "Median GFLOPS vs best (%)",
        output_dir / "stage2_relative_to_best_heatmap.png",
        dpi,
        percent=True,
    )
    draw_heatmap(
        absolute_matrix,
        labels,
        kernels,
        "AVX-512 Stage 2 Best-Per-Kernel Summary: Absolute Median GFLOPS",
        "Median GFLOPS",
        output_dir / "stage2_absolute_gflops_heatmap.png",
        dpi,
        percent=False,
    )


def plot_workload_winner_bars(summary_dir: Path, output_dir: Path, dpi: int) -> None:
    winners = read_csv(summary_dir / "workload_winners.csv")
    labels = [f"{row['WorkloadId']}\n{row['BestKernelShape']}" for row in winners]
    values = [float(row["BestMedianGFLOPS"]) for row in winners]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(labels)), 5.0), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median GFLOPS")
    ax.set_title("AVX-512 Stage 2 Best-Per-Kernel Summary: Best Candidate Absolute GFLOPS")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stage2_best_candidate_absolute_gflops.png", dpi=dpi)
    plt.close(fig)


def plot_parameter_level(summary_dir: Path, output_dir: Path, dpi: int) -> None:
    rows = read_csv(summary_dir / "parameter_aggregates.csv")
    best_rows = read_csv(summary_dir / "parameter_best_per_workload.csv")
    parameter_dir = output_dir / "parameter_level"
    kernels = [kernel for kernel in ("avx512_8x32", "avx512_16x16") if any(row["KernelShape"] == kernel for row in rows)]

    by_kernel_workload: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_kernel_workload[(row["KernelShape"], row["WorkloadFamily"], row["WorkloadId"])].append(row)

    for kernel in kernels:
        for (shape, family, workload), group in sorted(by_kernel_workload.items()):
            if shape != kernel:
                continue
            safe_kernel = sanitize_filename(shape)
            safe_workload = sanitize_filename(family)

            kcs = sorted({int(row["Kc"]) for row in group})
            mcs = sorted({int(row["Mc"]) for row in group})
            ncs = sorted({int(row["Nc"]) for row in group})

            kc_mc_best: dict[tuple[int, int], dict[str, str]] = {}
            for row in group:
                key = (int(row["Kc"]), int(row["Mc"]))
                current = kc_mc_best.get(key)
                if current is None or float(row["MedianGFLOPS"] or "0") > float(current["MedianGFLOPS"] or "0"):
                    kc_mc_best[key] = row

            kc_mc_matrix: list[list[float]] = []
            kc_mc_annotations: list[list[str]] = []
            for mc in mcs:
                values: list[float] = []
                notes: list[str] = []
                for kc in kcs:
                    best = kc_mc_best.get((kc, mc))
                    if best is None:
                        values.append(math.nan)
                        notes.append("NA")
                    else:
                        values.append(float(best["MedianGFLOPS"]))
                        notes.append(
                            f"{float(best['MedianGFLOPS']):.2f}\n"
                            f"best Nc={best['Nc']}\n"
                            f"{float(best['RelativeToKernelWorkloadBestPct']):.1f}%"
                        )
                kc_mc_matrix.append(values)
                kc_mc_annotations.append(notes)

            draw_parameter_heatmap(
                kc_mc_matrix,
                kc_mc_annotations,
                [str(value) for value in mcs],
                [str(value) for value in kcs],
                f"{shape} {family}: Kc x Mc, Best Nc",
                "Kc",
                "Mc",
                parameter_dir / f"{safe_kernel}__{safe_workload}__kc_mc_bestnc_heatmap.png",
                dpi,
            )

            kc_nc_best: dict[tuple[int, int], dict[str, str]] = {}
            for row in group:
                key = (int(row["Kc"]), int(row["Nc"]))
                current = kc_nc_best.get(key)
                if current is None or float(row["MedianGFLOPS"] or "0") > float(current["MedianGFLOPS"] or "0"):
                    kc_nc_best[key] = row

            kc_nc_matrix: list[list[float]] = []
            kc_nc_annotations: list[list[str]] = []
            for nc in ncs:
                values = []
                notes = []
                for kc in kcs:
                    best = kc_nc_best.get((kc, nc))
                    if best is None:
                        values.append(math.nan)
                        notes.append("NA")
                    else:
                        values.append(float(best["MedianGFLOPS"]))
                        notes.append(
                            f"{float(best['MedianGFLOPS']):.2f}\n"
                            f"best Mc={best['Mc']}\n"
                            f"{float(best['RelativeToKernelWorkloadBestPct']):.1f}%"
                        )
                kc_nc_matrix.append(values)
                kc_nc_annotations.append(notes)

            draw_parameter_heatmap(
                kc_nc_matrix,
                kc_nc_annotations,
                [str(value) for value in ncs],
                [str(value) for value in kcs],
                f"{shape} {family}: Kc x Nc, Best Mc",
                "Kc",
                "Nc",
                parameter_dir / f"{safe_kernel}__{safe_workload}__kc_nc_bestmc_heatmap.png",
                dpi,
            )

    by_kernel_best: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in best_rows:
        by_kernel_best[row["KernelShape"]].append(row)

    for kernel in kernels:
        lines = [
            f"# {kernel} Parameter Best Table",
            "",
            "| WorkloadFamily | WorkloadId | BestKc | BestMc | BestNc | BestMedianGFLOPS | BestRelativeToGlobalWorkloadBestPct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(by_kernel_best[kernel], key=lambda item: (item["WorkloadFamily"], item["WorkloadId"])):
            lines.append(
                f"| {row['WorkloadFamily']} | {row['WorkloadId']} | {row['BestKc']} | "
                f"{row['BestMc']} | {row['BestNc']} | {row['BestMedianGFLOPS']} | "
                f"{row['BestRelativeToGlobalWorkloadBestPct']} |"
            )
        table_path = parameter_dir / f"{sanitize_filename(kernel)}_parameter_best_table.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    plot_relative_and_absolute(args.summary_dir, args.output_dir, args.dpi)
    plot_workload_winner_bars(args.summary_dir, args.output_dir, args.dpi)
    plot_parameter_level(args.summary_dir, args.output_dir, args.dpi)
    print(f"Wrote Stage 2 plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
