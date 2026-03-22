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


def read_best_test_acc(metrics_csv: Path) -> float:
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
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        try:
            for line in p.stdout:
                print(line, end="")      # 终端显示
                out.write(line)          # 写日志文件

            p.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)

        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            return 124

        return p.returncode


def build_repro_cmd(binary: str, args_list) -> str:
    return " ".join([binary] + args_list)


def main():
    ap = argparse.ArgumentParser(description="Enhanced CNN HPO for PPN.")

    ap.add_argument("--binary", default="./build/ppn_train", help="Path to ppn_train")
    ap.add_argument("--data_dir", default="mnist", help="Dataset directory")
    ap.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    ap.add_argument("--epochs", type=int, default=8, help="Epochs per trial")
    ap.add_argument("--seed", type=int, default=42, help="Training seed")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna jobs")
    ap.add_argument("--timeout", type=int, default=0, help="Per-trial timeout in seconds")
    ap.add_argument("--study_name", default="ppn_cnn_hpo")
    ap.add_argument("--storage", default="", help="Optuna storage, e.g. sqlite:///hpo.db")
    ap.add_argument("--root_out", default="output/ExperienceHPO", help="Root output directory")
    ap.add_argument(
        "--metric",
        choices=["best", "last"],
        default="best",
        help="Use best test_acc over epochs or last test_acc"
    )

    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

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
                "model": "cnn",
            },
            f,
            indent=2,
        )

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
        )

    if not trials_csv.exists():
        with trials_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial",
                "score",
                "learning_rate",
                "batch_size",
                "activation",
                "init",
                "optimizer",
                "momentum",
                "nesterov",
                "weight_decay",
                "beta1",
                "beta2",
                "eps",
                "epochs",
                "seed",
                "metric_mode",
                "out_dir",
            ])

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 2e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])

        activation = trial.suggest_categorical(
            "activation",
            ["relu", "leaky_relu", "gelu", "tanh"]
        )

        init = trial.suggest_categorical(
            "init",
            ["he", "xavier"]
        )

        optimizer = trial.suggest_categorical(
            "optimizer",
            ["sgd", "momentum_sgd", "adamw"]
        )

        print(f"\n========== Trial {trial.number} ==========")
        print(f"lr={lr}, batch={batch}, activation={activation}, init={init}, optimizer={optimizer}")

        momentum = ""
        nesterov = ""
        weight_decay = ""
        beta1 = ""
        beta2 = ""
        eps = ""

        out_dir = run_dir / f"trial_{trial.number:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = out_dir / "stdout.log"

        cmd_args = [
            "--model", "cnn",
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--activation", activation,
            "--init", init,
            "--optimizer", optimizer,
            "--seed", str(args.seed),
            "--data_dir", args.data_dir,
            "--out_dir", str(out_dir),
        ]

        if optimizer == "sgd":
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            cmd_args += [
                "--weight_decay", str(weight_decay),
            ]

        elif optimizer == "momentum_sgd":
            momentum = trial.suggest_float("momentum", 0.80, 0.99)
            nesterov = trial.suggest_categorical("nesterov", [0, 1])
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            cmd_args += [
                "--momentum", str(momentum),
                "--nesterov", str(nesterov),
                "--weight_decay", str(weight_decay),
            ]

        elif optimizer == "adamw":
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            beta1 = trial.suggest_float("beta1", 0.85, 0.95)
            beta2 = trial.suggest_float("beta2", 0.99, 0.9999)
            eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
            cmd_args += [
                "--weight_decay", str(weight_decay),
                "--beta1", str(beta1),
                "--beta2", str(beta2),
                "--eps", str(eps),
            ]

        cmd = [str(binary)] + cmd_args
        rc = run_cmd_capture(cmd, stdout_log, args.timeout)

        if rc != 0:
            raise optuna.exceptions.TrialPruned()

        metrics_csv = out_dir / "metrics.csv"
        score = read_best_test_acc(metrics_csv) if args.metric == "best" else read_last_test_acc(metrics_csv)

        with trials_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                trial.number,
                score,
                lr,
                batch,
                activation,
                init,
                optimizer,
                momentum,
                nesterov,
                weight_decay,
                beta1,
                beta2,
                eps,
                args.epochs,
                args.seed,
                args.metric,
                str(out_dir),
            ])

        return score

    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    best = study.best_trial
    best_params = best.params
    best_out_dir = run_dir / f"trial_{best.number:04d}"

    repro_args = [
        "--model", "cnn",
        "--epochs", str(args.epochs),
        "--batch_size", str(best_params["batch_size"]),
        "--learning_rate", str(best_params["learning_rate"]),
        "--activation", str(best_params["activation"]),
        "--init", str(best_params["init"]),
        "--optimizer", str(best_params["optimizer"]),
        "--seed", str(args.seed),
        "--data_dir", args.data_dir,
        "--out_dir", str(best_out_dir),
    ]

    if best_params["optimizer"] == "sgd":
        if "weight_decay" in best_params:
            repro_args += ["--weight_decay", str(best_params["weight_decay"])]

    elif best_params["optimizer"] == "momentum_sgd":
        if "momentum" in best_params:
            repro_args += ["--momentum", str(best_params["momentum"])]
        if "nesterov" in best_params:
            repro_args += ["--nesterov", str(best_params["nesterov"])]
        if "weight_decay" in best_params:
            repro_args += ["--weight_decay", str(best_params["weight_decay"])]

    elif best_params["optimizer"] == "adamw":
        if "weight_decay" in best_params:
            repro_args += ["--weight_decay", str(best_params["weight_decay"])]
        if "beta1" in best_params:
            repro_args += ["--beta1", str(best_params["beta1"])]
        if "beta2" in best_params:
            repro_args += ["--beta2", str(best_params["beta2"])]
        if "eps" in best_params:
            repro_args += ["--eps", str(best_params["eps"])]

    repro_cmd = build_repro_cmd(str(binary), repro_args)

    with best_txt.open("w") as f:
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best score ({args.metric} test_acc): {best.value}\n\n")
        f.write("Best params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nReproduce command:\n")
        f.write(repro_cmd + "\n")
        f.write("\nSuggested final confirmation command (20 epochs):\n")
        final_cmd = repro_cmd.replace(f"--epochs {args.epochs}", "--epochs 20")
        f.write(final_cmd + "\n")

    print("[OK] Enhanced CNN HPO finished.")
    print(f"[OK] Run dir    : {run_dir}")
    print(f"[OK] trials.csv : {trials_csv}")
    print(f"[OK] best.txt   : {best_txt}")


if __name__ == "__main__":
    main()