#!/usr/bin/env python3
import argparse
import os
import re
import csv
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import optuna


def read_last_test_acc(metrics_csv: Path) -> float:
    # metrics.csv header: epoch,train_loss,train_acc,test_loss,test_acc
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    last = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if last is None:
        raise RuntimeError(f"metrics.csv empty: {metrics_csv}")
    return float(last["test_acc"])


def run_cmd(cmd, cwd=None, env=None, stdout_path: Path = None) -> int:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w") as out:
            p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
            return p.wait()
    else:
        p = subprocess.run(cmd, cwd=cwd, env=env)
        return p.returncode


def build_repro_cmd(binary: str, args_list) -> str:
    # Create a reproducible one-liner command
    parts = [binary] + args_list
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/ppn_train", help="Path to ppn_train binary")
    ap.add_argument("--data_dir", default="mnist", help="MNIST directory (must contain idx ubyte files)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=0, help="Per-trial timeout seconds (0=none)")
    ap.add_argument("--study_name", default="ppn_hpo")
    ap.add_argument("--storage", default="", help="Optuna storage URL, e.g. sqlite:///hpo.db (empty = in-memory)")
    ap.add_argument("--seed", type=int, default=0, help="Seed passed to program (0=random)")
    ap.add_argument("--epochs", type=int, default=3, help="Epochs per trial (keep small for HPO)")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel trials (be careful with CPU)")
    ap.add_argument("--root_out", default="hpo_runs", help="Root dir to save logs/results")
    args = ap.parse_args()

    root_out = Path(args.root_out)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = run_dir / "trials.csv"
    best_txt = run_dir / "best.txt"
    meta_json = run_dir / "run_meta.json"

    with meta_json.open("w") as f:
        json.dump({
            "run_id": run_id,
            "binary": args.binary,
            "data_dir": args.data_dir,
            "trials": args.trials,
            "epochs": args.epochs,
            "seed": args.seed,
        }, f, indent=2)

    # Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=args.storage,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(study_name=args.study_name, direction="maximize")

    # Prepare CSV header
    if not trials_csv.exists():
        with trials_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial", "value_test_acc",
                "learning_rate", "batch_size", "hidden_sizes",
                "activation", "init",
                "seed", "epochs",
                "out_dir"
            ])

    def objective(trial: optuna.Trial) -> float:
        # ---- Search space (可按你们组讨论再扩展) ----
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # hidden sizes: 用字符串传给 --hidden_sizes
        hidden_sizes = trial.suggest_categorical(
            "hidden_sizes",
            ["128", "256", "256,128", "512,256", "256,128,64"]
        )

        activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
        init = trial.suggest_categorical("init", ["he", "xavier"])

        # ---- Per-trial output dir ----
        out_dir = run_dir / f"trial_{trial.number:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = out_dir / "stdout.log"

        cmd_args = [
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--hidden_sizes", hidden_sizes,
            "--activation", activation,
            "--init", init,
            "--seed", str(args.seed),
            "--data_dir", args.data_dir,
            "--out_dir", str(out_dir)   # 依赖你在 main.cpp 加的参数
        ]

        cmd = [args.binary] + cmd_args

        # Run training
        env = os.environ.copy()

        try:
            if args.timeout and args.timeout > 0:
                # subprocess with timeout
                with stdout_log.open("w") as out:
                    p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
                    p.wait(timeout=args.timeout)
                    rc = p.returncode
            else:
                rc = run_cmd(cmd, env=env, stdout_path=stdout_log)
        except subprocess.TimeoutExpired:
            # kill process on timeout
            try:
                p.kill()
            except Exception:
                pass
            raise optuna.exceptions.TrialPruned()

        if rc != 0:
            # non-zero exit -> prune (or raise)
            raise optuna.exceptions.TrialPruned()

        # Parse metric from metrics.csv
        metrics_csv = out_dir / "metrics.csv"
        test_acc = read_last_test_acc(metrics_csv)

        # Log one line to global CSV
        with trials_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                trial.number, test_acc,
                lr, batch, hidden_sizes,
                activation, init,
                args.seed, args.epochs,
                str(out_dir)
            ])

        return test_acc

    # Run HPO
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    # Save best
    best = study.best_trial
    best_params = best.params
    best_out_dir = run_dir / f"trial_{best.number:04d}"

    repro_args = [
        "--epochs", str(args.epochs),
        "--batch_size", str(best_params["batch_size"]),
        "--learning_rate", str(best_params["learning_rate"]),
        "--hidden_sizes", str(best_params["hidden_sizes"]),
        "--activation", str(best_params["activation"]),
        "--init", str(best_params["init"]),
        "--seed", str(args.seed),
        "--data_dir", args.data_dir,
        "--out_dir", str(best_out_dir)
    ]
    repro_cmd = build_repro_cmd(args.binary, repro_args)

    with best_txt.open("w") as f:
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best test_acc: {best.value}\n")
        f.write("Best params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nReproduce command:\n")
        f.write(repro_cmd + "\n")

    print(f"[OK] HPO finished. Results in: {run_dir}")
    print(f"[OK] trials.csv: {trials_csv}")
    print(f"[OK] best.txt: {best_txt}")


if __name__ == "__main__":
    main()
