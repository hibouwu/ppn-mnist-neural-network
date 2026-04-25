#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DRIVER = REPO_ROOT / "build" / "test_gemm_gotoblas_driver"


def main() -> int:
    if not DRIVER.exists():
        print(f"missing driver: {DRIVER}", file=sys.stderr)
        print("Build it first: cmake --build build --target test_gemm_gotoblas_driver", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["TEST_GEMM_GOTOBLAS_IMPL"] = "avx512"
    proc = subprocess.run([str(DRIVER)], cwd=REPO_ROOT, env=env, text=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
