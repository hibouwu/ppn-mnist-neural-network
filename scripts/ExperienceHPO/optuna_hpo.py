#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

MPI_ENV_KEYS = [
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_NODE_RANK",
    "OMPI_UNIVERSE_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "PMIX_RANK",
    "PMIX_NAMESPACE",
    "PMIX_SERVER_URI2",
    "MPI_LOCALNRANKS",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
]


def read_best_test_acc(metrics_csv: Path) -> float:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")

    best = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = float(row["test_acc"])
            best = value if best is None else max(best, value)

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


def run_cmd_capture(cmd, stdout_path: Path, timeout_s: int, env=None) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env if env is not None else os.environ.copy(),
        )

        try:
            for line in process.stdout:
                print(line, end="")
                out.write(line)

            process.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            return 124

        return process.returncode


def detect_mpi_context():
    rank_keys = [
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    ]
    size_keys = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ]

    rank = 0
    world_size = 1

    for key in rank_keys:
        value = os.environ.get(key)
        if value is not None:
            rank = int(value)
            break

    for key in size_keys:
        value = os.environ.get(key)
        if value is not None:
            world_size = int(value)
            break

    return rank, world_size


def sanitize_study_name(study_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in study_name)


def resolve_run_id(
    root_out: Path,
    requested_run_id: str,
    study_name: str,
    mpi_rank: int,
    shared_across_ranks: bool,
) -> str:
    if requested_run_id:
        return requested_run_id

    if not shared_across_ranks:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    root_out.mkdir(parents=True, exist_ok=True)
    coord_file = root_out / f".{sanitize_study_name(study_name)}.run_id"

    if mpi_rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        coord_file.write_text(run_id + "\n", encoding="utf-8")
        return run_id

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if coord_file.exists():
            run_id = coord_file.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
        time.sleep(0.1)

    raise RuntimeError(f"Timed out waiting for shared run_id file: {coord_file}")


def build_training_cmd(binary: Path, base_args, mpiexec: str, mpi_world_size: int):
    if mpi_world_size <= 1:
        return [str(binary)] + list(base_args)
    return [mpiexec, "-n", str(mpi_world_size), str(binary)] + list(base_args)


def build_repro_cmd(binary: Path, args_list, mpiexec: str, mpi_world_size: int) -> str:
    return " ".join(build_training_cmd(binary, args_list, mpiexec, mpi_world_size))


def build_child_env(clear_mpi_env: bool):
    env = os.environ.copy()
    if clear_mpi_env:
        for key in MPI_ENV_KEYS:
            env.pop(key, None)
    return env


def count_finished_trials(study: optuna.Study) -> int:
    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    return sum(1 for trial in study.get_trials(deepcopy=False) if trial.state in finished_states)


def wait_for_study_completion(
    study: optuna.Study,
    expected_trials: int,
    poll_interval_s: float = 1.0,
    timeout_s: int = 300,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if count_finished_trials(study) >= expected_trials:
            return
        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for {expected_trials} finished trials in study '{study.study_name}'."
    )


def wait_for_file(path: Path, timeout_s: float = 60.0, poll_interval_s: float = 0.1):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for file: {path}")


def classify_storage(storage_value: str) -> str:
    if not storage_value:
        return "none"
    if storage_value.startswith("sqlite:///"):
        return "sqlite"
    if storage_value.startswith("journal://"):
        return "journal"
    return "other"


def resolve_storage_config(args, run_dir: Path, using_mpi_optuna_workers: bool):
    storage_kind = classify_storage(args.storage)
    journal_path = None
    init_kind = "direct"
    storage_for_meta = args.storage

    if using_mpi_optuna_workers and storage_kind == "none":
        journal_path = run_dir / "optuna_journal.log"
        storage_kind = "journal"
        init_kind = "journal_auto"
        storage_for_meta = f"journal://{journal_path}"
    elif storage_kind == "journal":
        journal_path = Path(args.storage[len("journal://"):])
        init_kind = "journal_explicit"

    return {
        "storage_kind": storage_kind,
        "storage_for_meta": storage_for_meta,
        "journal_path": journal_path,
        "init_kind": init_kind,
    }


def make_storage_object(storage_cfg):
    if storage_cfg["storage_kind"] == "journal":
        return JournalStorage(JournalFileBackend(str(storage_cfg["journal_path"])))
    if storage_cfg["storage_kind"] in {"sqlite", "other"}:
        return storage_cfg["storage_for_meta"]
    return None


def create_or_load_study(
    study_name: str,
    sampler,
    storage_cfg,
    using_mpi_optuna_workers: bool,
    mpi_rank: int,
    run_dir: Path,
):
    ready_file = run_dir / f".{sanitize_study_name(study_name)}.study_ready"
    storage_obj = make_storage_object(storage_cfg)
    common_kwargs = {
        "study_name": study_name,
        "direction": "maximize",
        "sampler": sampler,
    }

    if storage_obj is None:
        return optuna.create_study(**common_kwargs)

    if not using_mpi_optuna_workers:
        return optuna.create_study(
            storage=storage_obj,
            load_if_exists=True,
            **common_kwargs,
        )

    if storage_cfg["storage_kind"] == "sqlite":
        if mpi_rank == 0:
            study = optuna.create_study(
                storage=storage_obj,
                load_if_exists=True,
                **common_kwargs,
            )
            ready_file.write_text("ready\n", encoding="utf-8")
            return study

        wait_for_file(ready_file)
        return optuna.load_study(
            study_name=study_name,
            storage=storage_obj,
            sampler=sampler,
        )

    return optuna.create_study(
        storage=storage_obj,
        load_if_exists=True,
        **common_kwargs,
    )


def export_trials_csv(study: optuna.Study, trials_csv: Path):
    trials_csv.parent.mkdir(parents=True, exist_ok=True)

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "state",
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
            "mpi_world_size",
            "out_dir",
        ])

        for trial in sorted(study.get_trials(deepcopy=False), key=lambda item: item.number):
            params = trial.params
            attrs = trial.user_attrs
            writer.writerow([
                trial.number,
                trial.state.name,
                trial.value if trial.value is not None else "",
                params.get("learning_rate", ""),
                params.get("batch_size", ""),
                params.get("activation", ""),
                params.get("init", ""),
                params.get("optimizer", ""),
                params.get("momentum", ""),
                params.get("nesterov", ""),
                params.get("weight_decay", ""),
                params.get("beta1", ""),
                params.get("beta2", ""),
                params.get("eps", ""),
                attrs.get("epochs", ""),
                attrs.get("seed", ""),
                attrs.get("metric_mode", ""),
                attrs.get("mpi_world_size", ""),
                attrs.get("out_dir", ""),
            ])


def main():
    ap = argparse.ArgumentParser(description="Enhanced CNN HPO for PPN with Optuna + MPI support.")

    ap.add_argument("--binary", default="./build/ppn_train", help="Path to ppn_train")
    ap.add_argument("--data_dir", default="mnist", help="Dataset directory")
    ap.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    ap.add_argument("--epochs", type=int, default=8, help="Epochs per trial")
    ap.add_argument("--seed", type=int, default=42, help="Training seed")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna jobs inside one Python process")
    ap.add_argument("--timeout", type=int, default=0, help="Per-trial timeout in seconds")
    ap.add_argument("--study_name", default="ppn_cnn_hpo")
    ap.add_argument("--storage", default="", help="Optuna storage, e.g. sqlite:///hpo.db")
    ap.add_argument("--root_out", default="output/ExperienceHPO", help="Root output directory")
    ap.add_argument("--run_id", default="", help="Reuse an existing run directory name")
    ap.add_argument("--mpiexec", default="mpiexec", help="MPI launcher path for distributed training trials")
    ap.add_argument("--mpi_world_size", type=int, default=1, help="MPI world size for each training trial")
    ap.add_argument(
        "--mpi_optuna_workers",
        action="store_true",
        help="When this script itself is launched under mpiexec, let each MPI rank act as one Optuna worker via shared storage; mutually exclusive with --mpi_world_size > 1",
    )
    ap.add_argument(
        "--metric",
        choices=["best", "last"],
        default="best",
        help="Use best test_acc over epochs or last test_acc",
    )

    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    if args.mpi_world_size <= 0:
        raise ValueError("--mpi_world_size must be >= 1.")

    mpi_rank, mpi_size = detect_mpi_context()
    using_mpi_optuna_workers = bool(args.mpi_optuna_workers and mpi_size > 1)

    if using_mpi_optuna_workers and args.mpi_world_size > 1:
        raise ValueError(
            "--mpi_optuna_workers and --mpi_world_size > 1 cannot be combined. "
            "Choose one mode: MPI ranks as Optuna workers, or MPI-distributed training inside each trial."
        )

    if using_mpi_optuna_workers and not args.storage:
        pass

    root_out = Path(args.root_out)
    run_id = resolve_run_id(
        root_out=root_out,
        requested_run_id=args.run_id,
        study_name=args.study_name,
        mpi_rank=mpi_rank,
        shared_across_ranks=using_mpi_optuna_workers,
    )
    run_dir = root_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = run_dir / "trials.csv"
    best_txt = run_dir / "best.txt"
    meta_json = run_dir / "run_meta.json"

    storage_cfg = resolve_storage_config(args, run_dir, using_mpi_optuna_workers)

    meta_payload = {
        "run_id": run_id,
        "binary": str(binary),
        "data_dir": args.data_dir,
        "trials": args.trials,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "timeout": args.timeout,
        "study_name": args.study_name,
        "storage": storage_cfg["storage_for_meta"],
        "storage_kind": storage_cfg["storage_kind"],
        "metric": args.metric,
        "model": "cnn",
        "mpiexec": args.mpiexec,
        "mpi_world_size": args.mpi_world_size,
        "mpi_optuna_workers": using_mpi_optuna_workers,
        "mpi_worker_rank": mpi_rank,
        "mpi_worker_world_size": mpi_size,
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    sampler = optuna.samplers.TPESampler(seed=args.seed + mpi_rank)
    if mpi_rank == 0 and using_mpi_optuna_workers:
        if storage_cfg["storage_kind"] == "journal":
            print(f"[INFO] MPI worker mode enabled. Using JournalStorage at {storage_cfg['journal_path']}")
        elif storage_cfg["storage_kind"] == "sqlite":
            print("[INFO] MPI worker mode with SQLite detected. Rank 0 will initialize the Optuna study before other ranks load it.")

    study = create_or_load_study(
        study_name=args.study_name,
        sampler=sampler,
        storage_cfg=storage_cfg,
        using_mpi_optuna_workers=using_mpi_optuna_workers,
        mpi_rank=mpi_rank,
        run_dir=run_dir,
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 2e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu", "tanh"])
        init = trial.suggest_categorical("init", ["he", "xavier"])
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "momentum_sgd", "adamw"])

        print(f"\n========== Trial {trial.number} / worker rank {mpi_rank} ==========")
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
            cmd_args += ["--weight_decay", str(weight_decay)]
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

        trial.set_user_attr("epochs", args.epochs)
        trial.set_user_attr("seed", args.seed)
        trial.set_user_attr("metric_mode", args.metric)
        trial.set_user_attr("mpi_world_size", args.mpi_world_size)
        trial.set_user_attr("out_dir", str(out_dir))

        cmd = build_training_cmd(binary, cmd_args, args.mpiexec, args.mpi_world_size)
        child_env = build_child_env(clear_mpi_env=using_mpi_optuna_workers)
        rc = run_cmd_capture(cmd, stdout_log, args.timeout, env=child_env)

        if rc != 0:
            raise optuna.exceptions.TrialPruned()

        metrics_csv = out_dir / "metrics.csv"
        score = read_best_test_acc(metrics_csv) if args.metric == "best" else read_last_test_acc(metrics_csv)
        return score

    def stop_on_finished_trial_budget(study_: optuna.Study, _trial: optuna.Trial):
        if count_finished_trials(study_) >= args.trials:
            study_.stop()

    optimize_kwargs = {
        "callbacks": [stop_on_finished_trial_budget],
    }
    if using_mpi_optuna_workers:
        optimize_kwargs["n_trials"] = None
        optimize_kwargs["n_jobs"] = 1
    else:
        optimize_kwargs["n_trials"] = args.trials
        optimize_kwargs["n_jobs"] = args.n_jobs

    study.optimize(objective, **optimize_kwargs)

    if using_mpi_optuna_workers and mpi_rank == 0:
        wait_for_study_completion(study, args.trials)

    should_finalize = (not using_mpi_optuna_workers) or mpi_rank == 0
    if not should_finalize:
        print(f"[OK] Worker rank {mpi_rank} finished.")
        return

    export_trials_csv(study, trials_csv)

    completed_trials = [
        trial for trial in study.get_trials(deepcopy=False) if trial.state == TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("No completed Optuna trials were produced. Check stdout.log files under the run directory.")

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

    repro_cmd = build_repro_cmd(binary, repro_args, args.mpiexec, args.mpi_world_size)

    with best_txt.open("w", encoding="utf-8") as f:
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best score ({args.metric} test_acc): {best.value}\n\n")
        f.write("Best params:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
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
