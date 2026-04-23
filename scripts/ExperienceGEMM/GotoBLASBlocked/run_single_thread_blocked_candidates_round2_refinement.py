#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_single_thread_blocked_candidates as round1_runner


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "single_thread_blocked_candidates_round2_refinement.json"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlocked" / "round2_refinement" / "raw" / "single_thread_blocked_candidates_round2_refinement_raw.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the second-round single-thread blocked-size refinement. "
            "This wrapper reuses the round-1 runner, but points it at the round2 "
            "local-refinement candidate grid and round2 output directory."
        )
    )
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
    argv = [
        str(Path(round1_runner.__file__).resolve()),
        "--config", str(args.config),
        "--output", str(args.output),
    ]
    if args.bench_bin is not None:
        argv.extend(["--bench-bin", str(args.bench_bin)])
    if args.limit_candidates is not None:
        argv.extend(["--limit-candidates", str(args.limit_candidates)])
    if args.limit_workloads is not None:
        argv.extend(["--limit-workloads", str(args.limit_workloads)])
    if args.skip_perf:
        argv.append("--skip-perf")
    if args.append:
        argv.append("--append")
    if args.dry_run:
        argv.append("--dry-run")

    old_argv = sys.argv
    try:
        sys.argv = argv
        return round1_runner.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
