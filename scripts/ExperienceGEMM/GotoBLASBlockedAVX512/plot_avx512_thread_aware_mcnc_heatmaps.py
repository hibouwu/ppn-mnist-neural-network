#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
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
    / "thread_aware_mcnc"
    / "summary"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "plots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AVX-512 thread-aware Mc/Nc heatmaps.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def draw_heatmap(
    matrix: list[list[float]],
    mcs: list[int],
    ncs: list[int],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    *,
    percent: bool = False,
) -> None:
    height = max(4.8, 0.68 * len(mcs) + 2.0)
    width = max(7.0, 0.85 * len(ncs) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    values = [value for row in matrix for value in row if not math.isnan(value)]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(range(len(ncs)))
    ax.set_xticklabels([str(nc) for nc in ncs], rotation=25, ha="right")
    ax.set_yticks(range(len(mcs)))
    ax.set_yticklabels([str(mc) for mc in mcs])
    ax.set_xlabel("Nc")
    ax.set_ylabel("Mc")
    ax.set_title(title)

    denom = vmax - vmin if vmax > vmin else 1.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                ax.text(j, i, "", ha="center", va="center", fontsize=8)
                continue
            text = f"{value:.1f}%" if percent else f"{value:.1f}"
            color = "white" if ((value - vmin) / denom) < 0.38 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def matrix_for_rows(rows: list[dict[str, str]]) -> tuple[list[int], list[int], list[list[float]]]:
    mcs = sorted({int(row["Mc"]) for row in rows})
    ncs = sorted({int(row["Nc"]) for row in rows})
    value_by_key = {
        (int(row["Mc"]), int(row["Nc"])): float(row["GeomeanGFLOPS"])
        for row in rows
        if row.get("GeomeanGFLOPS")
    }
    matrix: list[list[float]] = []
    for mc in mcs:
        matrix.append([value_by_key.get((mc, nc), math.nan) for nc in ncs])
    return mcs, ncs, matrix


def relative_matrix(matrix: list[list[float]]) -> list[list[float]]:
    values = [value for row in matrix for value in row if not math.isnan(value)]
    best = max(values) if values else math.nan
    if not math.isfinite(best) or best <= 0.0:
        return [[math.nan for _ in row] for row in matrix]
    return [
        [value / best * 100.0 if not math.isnan(value) else math.nan for value in row]
        for row in matrix
    ]


def plot_family_heatmaps(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    families = sorted({row["WorkloadFamily"] for row in rows})
    threads = sorted({int(row["Threads"]) for row in rows})
    for family in families:
        for threads_value in threads:
            subset = [
                row for row in rows
                if row["WorkloadFamily"] == family and int(row["Threads"]) == threads_value
            ]
            if not subset:
                continue
            mcs, ncs, matrix = matrix_for_rows(subset)
            path = output_dir / f"family_{sanitize(family)}_T{threads_value}_mcnc_heatmap.png"
            draw_heatmap(
                matrix,
                mcs,
                ncs,
                f"{family} Mc/Nc Heatmap (Geomean GFLOPS, T={threads_value})",
                "Geomean GFLOPS",
                path,
                dpi,
            )
            outputs.append(path)
    return outputs


def plot_overall_heatmaps(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    threads = sorted({int(row["Threads"]) for row in rows})
    for threads_value in threads:
        subset = [row for row in rows if int(row["Threads"]) == threads_value]
        if not subset:
            continue
        mcs, ncs, matrix = matrix_for_rows(subset)
        absolute_path = output_dir / f"overall_T{threads_value}_mcnc_heatmap.png"
        draw_heatmap(
            matrix,
            mcs,
            ncs,
            f"AVX-512 8x32 Overall Mc/Nc Heatmap (Geomean GFLOPS, T={threads_value})",
            "Geomean GFLOPS",
            absolute_path,
            dpi,
        )
        outputs.append(absolute_path)

        relative_path = output_dir / f"overall_T{threads_value}_relative_to_best_heatmap.png"
        draw_heatmap(
            relative_matrix(matrix),
            mcs,
            ncs,
            f"AVX-512 8x32 Overall Mc/Nc Heatmap (Relative To Best, T={threads_value})",
            "Geomean GFLOPS vs best (%)",
            relative_path,
            dpi,
            percent=True,
        )
        outputs.append(relative_path)
    return outputs


def main() -> int:
    args = build_parser().parse_args()
    family_rows = read_csv(args.summary_dir / "family_geomean_by_candidate.csv")
    overall_rows = read_csv(args.summary_dir / "overall_geomean_by_candidate.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_family_heatmaps(family_rows, args.output_dir, args.dpi))
    outputs.extend(plot_overall_heatmaps(overall_rows, args.output_dir, args.dpi))
    print(f"Wrote {len(outputs)} heatmaps under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
