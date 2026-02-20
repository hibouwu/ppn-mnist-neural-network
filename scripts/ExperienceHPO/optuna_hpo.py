#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

import optuna


# ============================================================
# Helpers
# ============================================================

def read_best_test_acc(metrics_csv: Path) -> float:
    """
    Return the BEST test_acc observed across epochs.
    This is usually more stable than "last epoch" because test acc can oscillate.

    Expected CSV header:
      epoch,train_loss,train_acc,test_loss,test_acc
    """
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")

    best = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["test_acc"])
            best = v if best is None else max(best, v)

    if best is None:
        raise RuntimeError(f"metrics.csv is empty: {metrics_csv}")
    return float(best)


def read_last_test_acc(metrics_csv: Path) -> float:
    """Return last epoch test_acc (sometimes useful for strict comparisons)."""
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")

    last = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row

    if last is None:
        raise RuntimeError(f"metrics.csv is empty: {metrics_csv}")
    return float(last["test_acc"])


def run_cmd_capture(cmd, stdout_path: Path, timeout_s: int) -> int:
    """
    Run command, redirect stdout+stderr to stdout_path.
    Returns process return code. Timeout returns 124.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out:
        p = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        try:
            p.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            return 124
        return p.returncode


def build_repro_cmd(binary: str, args_list) -> str:
    """
    Build a shell one-liner. (No quoting; keep values simple.)
    """
    return " ".join([binary] + args_list)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Program / data
    ap.add_argument("--binary", default="./build/ppn_train", help="Path to ppn_train binary")
    ap.add_argument("--data_dir", default="mnist", help="MNIST directory (idx ubyte files)")

    # HPO controls
    ap.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    ap.add_argument("--epochs", type=int, default=8, help="Epochs per trial (keep small for HPO)")
    ap.add_argument("--seed", type=int, default=42, help="Seed passed to program")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel trials (start with 1)")
    ap.add_argument("--timeout", type=int, default=0, help="Per-trial timeout (seconds), 0=none")

    # Study persistence (optional)
    ap.add_argument("--study_name", default="ppn_cnn_hpo")
    ap.add_argument("--storage", default="", help="Optuna storage URL, e.g. sqlite:///hpo.db (empty=in-memory)")

    # Logs
    ap.add_argument("--root_out", default="hpo_runs", help="Root output dir for logs/results")

    # Metric choice
    ap.add_argument(
        "--metric",
        choices=["best", "last"],
        default="best",
        help="Optimize best test_acc over epochs (best) or last-epoch test_acc (last).",
    )

    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    # Create run directory
    root_out = Path(args.root_out)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = run_dir / "trials.csv"
    best_txt = run_dir / "best.txt"
    meta_json = run_dir / "run_meta.json"

    with meta_json.open("w") as f:
        json.dump(
            {
                "run_id": run_id,
                "binary": str(binary),
                "data_dir": args.data_dir,
                "trials": args.trials,
                "epochs": args.epochs,
                "seed": args.seed,
                "n_jobs": args.n_jobs,
                "timeout": args.timeout,
                "study_name": args.study_name,
                "storage": args.storage,
                "metric": args.metric,
            },
            f,
            indent=2,
        )

    # Create Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(study_name=args.study_name, direction="maximize")

    # Write CSV header once
    if not trials_csv.exists():
        with trials_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "trial",
                    "score",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "seed",
                    "out_dir",
                ]
            )

    # Objective function
    def objective(trial: optuna.Trial) -> float:
        # Effective hyperparameters for your current CNN implementation:
        lr = trial.suggest_float("learning_rate", 1e-3, 2e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])

        out_dir = run_dir / f"trial_{trial.number:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = out_dir / "stdout.log"

        cmd_args = [
            "--model", "cnn",
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--seed", str(args.seed),
            "--data_dir", args.data_dir,
            "--out_dir", str(out_dir),
        ]
        cmd = [str(binary)] + cmd_args

        rc = run_cmd_capture(cmd, stdout_log, args.timeout)
        if rc != 0:
            # prune failed runs (bad params, crash, timeout, etc.)
            raise optuna.exceptions.TrialPruned()

        metrics_csv = out_dir / "metrics.csv"
        if args.metric == "best":
            score = read_best_test_acc(metrics_csv)
        else:
            score = read_last_test_acc(metrics_csv)

        with trials_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([trial.number, score, lr, batch, args.epochs, args.seed, str(out_dir)])

        return score

    # Run optimization
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    # Save best
    best = study.best_trial
    best_params = best.params
    best_out_dir = run_dir / f"trial_{best.number:04d}"

    repro_args = [
        "--model", "cnn",
        "--epochs", str(args.epochs),
        "--batch_size", str(best_params["batch_size"]),
        "--learning_rate", str(best_params["learning_rate"]),
        "--seed", str(args.seed),
        "--data_dir", args.data_dir,
        "--out_dir", str(best_out_dir),
    ]
    repro_cmd = build_repro_cmd(str(binary), repro_args)

    with best_txt.open("w") as f:
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best score ({args.metric} test_acc): {best.value}\n\n")
        f.write("Best params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nReproduce command:\n")
        f.write(repro_cmd + "\n")

    print("[OK] HPO finished.")
    print(f"[OK] Run dir   : {run_dir}")
    print(f"[OK] trials.csv: {trials_csv}")
    print(f"[OK] best.txt  : {best_txt}")


if __name__ == "__main__":
    main()