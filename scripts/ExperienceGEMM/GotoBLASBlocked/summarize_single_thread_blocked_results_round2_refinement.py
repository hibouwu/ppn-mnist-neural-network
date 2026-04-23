#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import summarize_single_thread_blocked_results as round1_summary


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "single_thread_blocked_candidates_round2_refinement.json"
DEFAULT_RAW_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "round2_refinement" / "raw"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "round2_refinement" / "summary"


def load_summary_title(config_path: Path) -> str:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return str(data.get("summary_title", "Single-Thread Blocked Round 2 Refinement Summary"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the second-round single-thread blocked-size refinement. "
            "Grouping remains by runtime-effective KernelShape/Mc/Nc/Kc."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--summary-title", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_title = args.summary_title or load_summary_title(args.config)
    argv = [
        str(Path(round1_summary.__file__).resolve()),
        "--config", str(args.config),
        "--raw-dir", str(args.raw_dir),
        "--summary-dir", str(args.summary_dir),
        "--summary-title", summary_title,
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        return round1_summary.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
