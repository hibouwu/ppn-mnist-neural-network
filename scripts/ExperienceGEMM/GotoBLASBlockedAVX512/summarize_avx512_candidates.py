#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
AVX2_SUMMARIZER = SCRIPT_DIR.parent / "GotoBLASBlocked" / "summarize_single_thread_blocked_results.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AVX-512 GotoBLAS candidate results.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--summary-title", type=str, default=None)
    return parser


def defaults(stage: str) -> tuple[Path, Path, Path, str]:
    if stage == "stage1":
        root = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "stage1_kernel_screening"
        return (
            SCRIPT_DIR / "config" / "avx512_stage1_kernel_screening.json",
            root / "raw",
            root / "summary",
            "AVX-512 Stage 1 Kernel-Shape Screening Summary",
        )
    root = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "stage2_blocked_candidates"
    return (
        SCRIPT_DIR / "config" / "avx512_stage2_blocked_candidates.json",
        root / "raw",
        root / "summary",
        "AVX-512 Stage 2 Blocked-Size Candidate Summary",
    )


def main() -> int:
    args = build_parser().parse_args()
    config, raw_dir, summary_dir, title = defaults(args.stage)
    cmd = [
        sys.executable,
        str(AVX2_SUMMARIZER),
        "--config",
        str(args.config or config),
        "--raw-dir",
        str(args.raw_dir or raw_dir),
        "--summary-dir",
        str(args.summary_dir or summary_dir),
        "--summary-title",
        args.summary_title or title,
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
