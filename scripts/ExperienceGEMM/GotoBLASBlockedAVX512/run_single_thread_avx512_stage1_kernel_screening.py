#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
AVX2_RUNNER = SCRIPT_DIR.parent / "GotoBLASBlocked" / "run_single_thread_blocked_candidates.py"
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "avx512_stage1_kernel_screening.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "stage1_kernel_screening"
    / "raw"
    / "avx512_stage1_kernel_screening_raw.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AVX-512 stage-1 kernel-shape screening.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bench-bin", type=Path, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cmd = [sys.executable, str(AVX2_RUNNER), "--config", str(args.config), "--output", str(args.output)]
    if args.bench_bin is not None:
        cmd += ["--bench-bin", str(args.bench_bin)]
    if args.limit_candidates is not None:
        cmd += ["--limit-candidates", str(args.limit_candidates)]
    if args.limit_workloads is not None:
        cmd += ["--limit-workloads", str(args.limit_workloads)]
    if args.skip_perf:
        cmd.append("--skip-perf")
    if args.append:
        cmd.append("--append")
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
