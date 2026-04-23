#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import plot_single_thread_blocked_heatmaps as round1_plot


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "single_thread_blocked_candidates_round2_refinement.json"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "round2_refinement" / "summary"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "plots"
DEFAULT_SHAPE_ORDER = ["8x8", "4x16"]


def load_plot_metadata(config_path: Path) -> tuple[str, str]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    title = str(data.get("plot_title", "Round 2 Refinement: Kernel Shape Relative To Best Custom Shape Per Workload"))
    filename = str(data.get("plot_filename", "round2_refinement_relative_to_best_custom_heatmap.png"))
    return title, filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the second-round refinement heatmap. "
            "This wrapper reuses the round-1 plotting script but writes into the "
            "round2 refinement plot directory with a round2-specific title and filename."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shape-order",
        nargs="+",
        default=DEFAULT_SHAPE_ORDER,
        help="Kernel-shape order for the round2 refinement heatmap. Defaults to the two retained round2 shapes.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    title, filename = load_plot_metadata(args.config)
    argv = [
        str(Path(round1_plot.__file__).resolve()),
        "--summary-dir", str(args.summary_dir),
        "--output-dir", str(args.output_dir),
        "--shape-order", *args.shape_order,
        "--title", title,
        "--output-filename", filename,
        "--dpi", str(args.dpi),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        round1_plot.main()
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
