#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
AVX2_PLOTTER = SCRIPT_DIR.parent / "GotoBLASBlocked" / "plot_single_thread_blocked_heatmaps.py"
PRIMARY_SHAPE_ORDER = [
    "avx512_4x16",
    "avx512_8x16",
    "avx512_14x16",
    "avx512_16x16",
    "avx512_18x16",
    "avx512_20x16",
    "avx512_6x32",
    "avx512_8x32",
    "avx512_10x32",
    "avx512_12x32",
]
WITH_STRESS_SHAPE_ORDER = [*PRIMARY_SHAPE_ORDER, "avx512_4x32"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AVX-512 GotoBLAS candidate heatmaps.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-stress", action="store_true", help="Include avx512_4x32 in the plot.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def stage_root(stage: str) -> Path:
    name = "stage1_kernel_screening" if stage == "stage1" else "stage2_blocked_candidates"
    return REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / name


def main() -> int:
    args = build_parser().parse_args()
    root = stage_root(args.stage)
    shape_order = WITH_STRESS_SHAPE_ORDER if args.include_stress else PRIMARY_SHAPE_ORDER
    title = (
        "AVX-512 Stage 1 Kernel Shape Relative To Best Primary Shape"
        if args.stage == "stage1"
        else "AVX-512 Stage 2 Blocked Candidate Relative To Best Primary Shape"
    )
    cmd = [
        sys.executable,
        str(AVX2_PLOTTER),
        "--summary-dir",
        str(args.summary_dir or (root / "summary")),
        "--output-dir",
        str(args.output_dir or (root / "summary" / "plots")),
        "--shape-order",
        *shape_order,
        "--title",
        title,
        "--output-filename",
        "avx512_relative_to_best_primary_heatmap.png",
        "--dpi",
        str(args.dpi),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
