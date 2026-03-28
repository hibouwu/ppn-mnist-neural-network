#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class QualificationValidationError(RuntimeError):
    def __init__(self, reason: str, detail: Optional[Dict[str, object]] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage C qualification for overlap_bucketed against bucketed baseline.")
    parser.add_argument("--binary", required=True, help="Path to ppn_train binary.")
    parser.add_argument("--out-root", required=True, help="Root output directory for the qualification group.")
    parser.add_argument("--world-size", required=True, type=int, help="MPI world size.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of preregistered repeats.")
    parser.add_argument("--mpiexec", default="/usr/lib64/mpich/bin/mpiexec", help="MPI launcher path.")
    parser.add_argument("--comparison-group-id", default="", help="Optional comparison group id.")
    parser.add_argument(
        "--qualification-metric",
        default="avg_step_time_ms",
        help="Pre-registered Stage C performance metric: sync_wait_s or avg_step_time_ms.")
    parser.add_argument(
        "ppn_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to ppn_train. Use '--' before them.")
    return parser.parse_args()


def normalize_ppn_args(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return list(args)


def require_cli_value(args: List[str], flag: str) -> str:
    for idx, value in enumerate(args):
        if value == flag and idx + 1 < len(args):
            return args[idx + 1]
    raise SystemExit(f"Missing required forwarded CLI flag {flag} in ppn_args.")


def replace_or_append_flag(args: List[str], flag: str, value: str) -> List[str]:
    out = list(args)
    for idx, item in enumerate(out):
        if item == flag:
            if idx + 1 >= len(out):
                raise SystemExit(f"Flag {flag} missing value.")
            out[idx + 1] = value
            return out
    out.extend([flag, value])
    return out


def git_commit_hash(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True)
    return result.stdout.strip()


def read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, got {len(rows)}.")
    return rows[0]


def read_metrics_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_trace_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_group_manifest(out_root: Path) -> Dict[str, object]:
    path = out_root / "qualification_group_manifest.json"
    if not path.exists():
        raise QualificationValidationError("Fail-GroupManifestMissing", {"path": str(path)})
    return read_json(path)


def load_run_manifest(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "qualification_run_manifest.json"
    if not path.exists():
        raise QualificationValidationError("Fail-RunManifestMissing", {"path": str(path)})
    return read_json(path)


def expect_path_within_run_dir(path_value: object, run_dir: Path, field: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise QualificationValidationError("Fail-ManifestBinding", {"field": field, "reason": "missing_path"})
    resolved = Path(path_value).resolve()
    run_root = run_dir.resolve()
    if resolved != run_root and run_root not in resolved.parents:
        raise QualificationValidationError(
            "Fail-ManifestBinding",
            {"field": field, "path": str(resolved), "run_dir": str(run_root)})


def validate_group_manifest_structure(group_manifest: Dict[str, object],
                                      expected_world_size: int,
                                      expected_comparison_group_id: str,
                                      preregistered_runs: List[Dict[str, object]]) -> None:
    if group_manifest.get("comparison_group_id") != expected_comparison_group_id:
        raise QualificationValidationError(
            "Fail-ManifestBinding",
            {"field": "comparison_group_id", "expected": expected_comparison_group_id,
             "observed": group_manifest.get("comparison_group_id")})
    if int(group_manifest.get("world_size", -1)) != expected_world_size:
        raise QualificationValidationError(
            "Fail-ManifestBinding",
            {"field": "world_size", "expected": expected_world_size,
             "observed": group_manifest.get("world_size")})

    manifest_runs = group_manifest.get("runs")
    if not isinstance(manifest_runs, list):
        raise QualificationValidationError("Fail-RunSetMismatch", {"reason": "group_runs_missing"})

    expected_by_id = {str(run["run_id"]): run for run in preregistered_runs}
    observed_ids: Set[str] = set()
    if len(manifest_runs) != len(preregistered_runs):
        raise QualificationValidationError(
            "Fail-RunSetMismatch",
            {"expected_count": len(preregistered_runs), "observed_count": len(manifest_runs)})

    for run in manifest_runs:
        run_id = str(run.get("run_id", ""))
        if run_id not in expected_by_id:
            raise QualificationValidationError(
                "Fail-RunSetMismatch",
                {"reason": "unexpected_run_id", "run_id": run_id})
        expected = expected_by_id[run_id]
        observed_ids.add(run_id)
        for field in ("repeat_index", "mode", "out_dir", "stdout_path"):
            if run.get(field) != expected.get(field):
                raise QualificationValidationError(
                    "Fail-ManifestBinding",
                    {"run_id": run_id, "field": field,
                     "expected": expected.get(field), "observed": run.get(field)})

    if observed_ids != set(expected_by_id.keys()):
        raise QualificationValidationError(
            "Fail-RunSetMismatch",
            {"reason": "missing_preregistered_runs",
             "missing": sorted(set(expected_by_id.keys()) - observed_ids)})


def validate_run_manifest_binding(run_manifest: Dict[str, object],
                                  expected_run_entry: Dict[str, object],
                                  expected_comparison_group_id: str,
                                  expected_world_size: int,
                                  expected_seed: str,
                                  expected_bucket_size: str,
                                  expected_commit_hash: str,
                                  run_dir: Path) -> None:
    for field, expected in (
        ("comparison_group_id", expected_comparison_group_id),
        ("run_id", str(expected_run_entry["run_id"])),
        ("mode", str(expected_run_entry["mode"])),
        ("world_size", expected_world_size),
        ("seed", expected_seed),
        ("bucket_size", expected_bucket_size),
        ("commit_hash", expected_commit_hash),
        ("status", "completed"),
    ):
        observed = run_manifest.get(field)
        if observed != expected:
            raise QualificationValidationError(
                "Fail-ManifestBinding",
                {"run_id": expected_run_entry["run_id"], "field": field,
                 "expected": expected, "observed": observed})

    for field in (
        "stdout_path",
        "timing_path",
        "profile_path",
        "trace_path",
        "parameter_snapshot_index_path",
        "parameter_layout_path",
    ):
        expect_path_within_run_dir(run_manifest.get(field), run_dir, field)

    trace_paths = run_manifest.get("trace_paths")
    if not isinstance(trace_paths, dict) or len(trace_paths) != expected_world_size:
        raise QualificationValidationError(
            "Fail-ManifestBinding",
            {"run_id": expected_run_entry["run_id"], "field": "trace_paths"})
    for rank in range(expected_world_size):
        key = f"rank_{rank:03d}"
        if key not in trace_paths:
            raise QualificationValidationError(
                "Fail-ManifestBinding",
                {"run_id": expected_run_entry["run_id"], "field": "trace_paths", "missing_rank": key})
        expect_path_within_run_dir(trace_paths[key], run_dir, f"trace_paths.{key}")


def preregister_group_manifest(out_root: Path,
                               comparison_group_id: str,
                               repeats: int,
                               world_size: int,
                               binary: str,
                               ppn_args: List[str],
                               qualification_metric: str) -> Dict[str, object]:
    runs = []
    for repeat in range(1, repeats + 1):
        for mode in ("bucketed", "overlap_bucketed"):
            run_id = f"{mode}_r{repeat:02d}"
            out_dir = out_root / "runs" / run_id
            runs.append({
                "run_id": run_id,
                "repeat_index": repeat,
                "mode": mode,
                "out_dir": str(out_dir),
                "stdout_path": str(out_dir / "stdout.log"),
            })
    manifest = {
        "comparison_group_id": comparison_group_id,
        "world_size": world_size,
        "repeats": repeats,
        "binary": binary,
        "ppn_args": ppn_args,
        "qualification_metric": qualification_metric,
        "runs": runs,
    }
    manifest_path = out_root / "qualification_group_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def execute_run(repo_root: Path,
                mpiexec: str,
                world_size: int,
                binary: str,
                run_entry: Dict[str, object],
                ppn_args: List[str]) -> Tuple[int, List[str]]:
    run_id = str(run_entry["run_id"])
    mode = str(run_entry["mode"])
    out_dir = Path(str(run_entry["out_dir"]))
    stdout_path = Path(str(run_entry["stdout_path"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [mpiexec, "-n", str(world_size), binary]
    cmd.extend(replace_or_append_flag(ppn_args, "--grad_sync_mode", mode))
    cmd = replace_or_append_flag(cmd, "--out_dir", str(out_dir))
    cmd = replace_or_append_flag(cmd, "--qualification_artifacts", "1")

    with stdout_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True)
    return proc.returncode, cmd


def build_run_manifest(repo_root: Path,
                       comparison_group_id: str,
                       commit_hash: str,
                       world_size: int,
                       run_entry: Dict[str, object],
                       cli: List[str],
                       bucket_size: str,
                       seed: str) -> Dict[str, object]:
    out_dir = Path(str(run_entry["out_dir"]))
    profile_row = read_single_row_csv(out_dir / "profile_run_summary.csv")
    trace_paths = {
        f"rank_{rank:03d}": str(out_dir / "qualification" / f"rank_{rank:03d}" / "sync_trace.csv")
        for rank in range(world_size)
    }
    manifest = {
        "run_id": str(run_entry["run_id"]),
        "comparison_group_id": comparison_group_id,
        "mode": str(run_entry["mode"]),
        "world_size": world_size,
        "seed": seed,
        "bucket_size": bucket_size,
        "commit_hash": commit_hash,
        "cli": cli,
        "stdout_path": str(out_dir / "stdout.log"),
        "timing_path": str(out_dir / "metrics.csv"),
        "profile_path": str(out_dir / "profile_run_summary.csv"),
        "trace_path": trace_paths["rank_000"],
        "trace_paths": trace_paths,
        "parameter_snapshot_index_path": str(out_dir / "qualification" / "parameter_snapshot_index.csv"),
        "parameter_layout_path": str(out_dir / "qualification" / "parameter_layout.csv"),
        "parameter_consistency_check_path": "",
        "mode_metadata": {
            "label": profile_row["grad_sync_mode_label"],
            "semantics": profile_row["grad_sync_semantics"],
            "correctness_only": int(profile_row["grad_sync_correctness_only"]),
        },
    }
    manifest_path = out_dir / "qualification_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_snapshot_index(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def extract_step_set(rows: List[Dict[str, str]]) -> Set[Tuple[int, int]]:
    return {(int(row["epoch"]), int(row["step"])) for row in rows}


def validate_parameter_layouts(baseline_dir: Path, overlap_dir: Path) -> Dict[str, object]:
    baseline_layout = read_csv_rows(baseline_dir / "qualification" / "parameter_layout.csv")
    overlap_layout = read_csv_rows(overlap_dir / "qualification" / "parameter_layout.csv")
    if len(baseline_layout) != len(overlap_layout):
        return {
            "passed": False,
            "reason": "Fail-ParameterLayoutMismatch",
            "detail": {"baseline_count": len(baseline_layout), "overlap_count": len(overlap_layout)},
        }
    for idx, (base_row, over_row) in enumerate(zip(baseline_layout, overlap_layout)):
        for field in ("param_idx", "rows", "cols", "numel"):
            if base_row[field] != over_row[field]:
                return {
                    "passed": False,
                    "reason": "Fail-ParameterLayoutMismatch",
                    "detail": {"row_index": idx, "field": field,
                               "baseline": base_row[field], "overlap": over_row[field]},
                }
    return {"passed": True, "reason": "", "detail": {}}


def validate_snapshot_index_metadata(baseline_index: List[Dict[str, str]],
                                     overlap_index: List[Dict[str, str]]) -> Dict[str, object]:
    if len(baseline_index) != len(overlap_index):
        return {
            "passed": False,
            "reason": "Fail-SnapshotIndexMismatch",
            "detail": {"baseline_count": len(baseline_index), "overlap_count": len(overlap_index)},
        }
    for idx, (base_row, over_row) in enumerate(zip(baseline_index, overlap_index)):
        for field in ("epoch", "step"):
            if base_row[field] != over_row[field]:
                return {
                    "passed": False,
                    "reason": "Fail-SnapshotIndexMismatch",
                    "detail": {"row_index": idx, "field": field,
                               "baseline": base_row[field], "overlap": over_row[field]},
                }
        for field in ("total_values", "total_bytes"):
            if base_row[field] != over_row[field]:
                return {
                    "passed": False,
                    "reason": "Fail-SnapshotMetadataMismatch",
                    "detail": {"row_index": idx, "field": field,
                               "baseline": base_row[field], "overlap": over_row[field]},
                }
    return {"passed": True, "reason": "", "detail": {}}


def compare_parameter_snapshots(baseline_dir: Path, overlap_dir: Path, result_path: Path) -> Dict[str, object]:
    baseline_index = load_snapshot_index(baseline_dir / "qualification" / "parameter_snapshot_index.csv")
    overlap_index = load_snapshot_index(overlap_dir / "qualification" / "parameter_snapshot_index.csv")
    result = {
        "passed": True,
        "reason": "",
        "mismatched_step": None,
        "compared_steps": 0,
        "baseline_step_set": sorted(extract_step_set(baseline_index)),
        "overlap_step_set": sorted(extract_step_set(overlap_index)),
    }

    layout_validation = validate_parameter_layouts(baseline_dir, overlap_dir)
    if not layout_validation["passed"]:
        result["passed"] = False
        result["reason"] = layout_validation["reason"]
        result["detail"] = layout_validation["detail"]
    else:
        index_validation = validate_snapshot_index_metadata(baseline_index, overlap_index)
        if not index_validation["passed"]:
            result["passed"] = False
            result["reason"] = index_validation["reason"]
            result["detail"] = index_validation["detail"]

    if result["passed"] and len(baseline_index) != len(overlap_index):
        result["passed"] = False
        result["reason"] = "snapshot_count_mismatch"
    elif result["passed"]:
        for base_row, over_row in zip(baseline_index, overlap_index):
            base_key = (base_row["epoch"], base_row["step"])
            over_key = (over_row["epoch"], over_row["step"])
            if base_key != over_key:
                result["passed"] = False
                result["reason"] = "snapshot_step_index_mismatch"
                result["mismatched_step"] = {"baseline": base_key, "overlap": over_key}
                break

            base_path = baseline_dir / "qualification" / base_row["relative_path"]
            over_path = overlap_dir / "qualification" / over_row["relative_path"]
            if base_path.read_bytes() != over_path.read_bytes():
                result["passed"] = False
                result["reason"] = "parameter_bytes_mismatch"
                result["mismatched_step"] = {"epoch": base_row["epoch"], "step": base_row["step"]}
                break
            result["compared_steps"] += 1

    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def group_trace_rows_by_step(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int], List[Dict[str, str]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
    for row in rows:
        key = (int(row["epoch"]), int(row["step"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def validate_trace_step_coverage(overlap_dir: Path,
                                 world_size: int,
                                 expected_step_set: Set[Tuple[int, int]]) -> Dict[str, object]:
    per_rank_step_sets: Dict[int, Set[Tuple[int, int]]] = {}
    for rank in range(world_size):
        trace_path = overlap_dir / "qualification" / f"rank_{rank:03d}" / "sync_trace.csv"
        rows = read_trace_rows(trace_path)
        per_rank_step_sets[rank] = extract_step_set(rows)

    reference = per_rank_step_sets[0] if per_rank_step_sets else set()
    for rank, step_set in per_rank_step_sets.items():
        if step_set != reference:
            return {
                "passed": False,
                "reason": "Fail-RankTraceStepMismatch",
                "detail": {"rank": rank, "reference": sorted(reference), "observed": sorted(step_set)},
                "per_rank_step_sets": {f"rank_{r:03d}": sorted(v) for r, v in per_rank_step_sets.items()},
            }
    if reference != expected_step_set:
        return {
            "passed": False,
            "reason": "Fail-TraceStepSetMismatch",
            "detail": {"expected": sorted(expected_step_set), "observed": sorted(reference)},
            "per_rank_step_sets": {f"rank_{r:03d}": sorted(v) for r, v in per_rank_step_sets.items()},
        }
    return {
        "passed": True,
        "reason": "",
        "detail": {},
        "per_rank_step_sets": {f"rank_{r:03d}": sorted(v) for r, v in per_rank_step_sets.items()},
    }


def validate_trace_row(rank: int, row: Dict[str, str]) -> Optional[Dict[str, object]]:
    channel = row["channel"]
    reason = row["reason"]
    lifecycle = row["request_lifecycle"]
    expected_count = int(row["expected_count"])
    ready_count = int(row["ready_count"])
    seconds_since_backward_complete = float(row["seconds_since_backward_complete"])

    # Simulated/synthetic/reconstructed-style rows must never present as real communication.
    if lifecycle == "simulated_only" and channel == "real_comm":
        return {
            "rank": rank,
            "epoch": int(row["epoch"]),
            "step": int(row["step"]),
            "bucket_idx": int(row["bucket_idx"]),
            "reason": "real_comm_with_simulated_only_lifecycle",
        }

    if reason == "structural_zero_expected_launch" and expected_count != 0:
        return {
            "rank": rank,
            "epoch": int(row["epoch"]),
            "step": int(row["step"]),
            "bucket_idx": int(row["bucket_idx"]),
            "reason": "structural_zero_expected_launch_with_nonzero_expected_count",
        }

    if reason == "real_early_launch":
        if expected_count <= 0:
            return {
                "rank": rank,
                "epoch": int(row["epoch"]),
                "step": int(row["step"]),
                "bucket_idx": int(row["bucket_idx"]),
                "reason": "real_early_launch_with_nonpositive_expected_count",
            }
        if ready_count != expected_count:
            return {
                "rank": rank,
                "epoch": int(row["epoch"]),
                "step": int(row["step"]),
                "bucket_idx": int(row["bucket_idx"]),
                "reason": "real_early_launch_without_pack_safe_ready_count",
            }

    if reason == "tail_flush_launch" and seconds_since_backward_complete < 0.0:
        return {
            "rank": rank,
            "epoch": int(row["epoch"]),
            "step": int(row["step"]),
            "bucket_idx": int(row["bucket_idx"]),
            "reason": "tail_flush_launch_before_backward_complete",
        }

    return None


def analyze_overlap_trace(overlap_dir: Path, world_size: int) -> Dict[str, object]:
    per_rank_rows = {}
    for rank in range(world_size):
        trace_path = overlap_dir / "qualification" / f"rank_{rank:03d}" / "sync_trace.csv"
        per_rank_rows[rank] = read_trace_rows(trace_path)

    legal_real_early_launch_count = 0
    illegal_events = []
    rank_sequence_mismatch = []
    prefix_order_mismatch = []

    grouped_by_rank = {rank: group_trace_rows_by_step(rows) for rank, rows in per_rank_rows.items()}
    all_steps = sorted({step for grouped in grouped_by_rank.values() for step in grouped.keys()})

    for rank, rows in per_rank_rows.items():
        for row in rows:
            invalid = validate_trace_row(rank, row)
            if invalid is not None:
                illegal_events.append(invalid)
                continue
            if row["channel"] == "real_comm" and row["reason"] == "real_early_launch":
                legal = (
                    float(row["seconds_since_backward_complete"]) < 0.0 and
                    int(row["expected_count"]) > 0 and
                    int(row["ready_count"]) == int(row["expected_count"]) and
                    row["request_lifecycle"] != "simulated_only"
                )
                if legal:
                    legal_real_early_launch_count += 1
                else:
                    illegal_events.append({
                        "rank": rank,
                        "epoch": int(row["epoch"]),
                        "step": int(row["step"]),
                        "bucket_idx": int(row["bucket_idx"]),
                    })

    for step in all_steps:
        sequences = []
        for rank in range(world_size):
            rows = grouped_by_rank.get(rank, {}).get(step, [])
            seq = [int(row["bucket_idx"]) for row in rows if row["channel"] == "real_comm"]
            sequences.append(seq)
            expected_prefix = list(range(len(seq)))
            if seq != expected_prefix:
                prefix_order_mismatch.append({
                    "rank": rank,
                    "epoch": step[0],
                    "step": step[1],
                    "sequence": seq,
                    "expected_prefix": expected_prefix,
                })
        for seq in sequences[1:]:
            if seq != sequences[0]:
                rank_sequence_mismatch.append({
                    "epoch": step[0],
                    "step": step[1],
                    "reference": sequences[0],
                    "observed": seq,
                })

    return {
        "legal_real_early_launch_count": legal_real_early_launch_count,
        "illegal_events": illegal_events,
        "rank_sequence_mismatch": rank_sequence_mismatch,
        "prefix_order_mismatch": prefix_order_mismatch,
        "trace_legal": not illegal_events,
        "rank_order_consistent": not rank_sequence_mismatch and not prefix_order_mismatch,
    }


def summarize_performance(run_dir: Path) -> Dict[str, float]:
    run_summary = read_single_row_csv(run_dir / "profile_run_summary.csv")
    metrics_rows = read_metrics_rows(run_dir / "metrics.csv")
    avg_step_time_ms = 0.0
    if metrics_rows:
        avg_step_time_ms = sum(float(row["avg_step_time_ms"]) for row in metrics_rows) / len(metrics_rows)
    return {
        "sync_wait_s": float(run_summary["sync_wait_s"]),
        "sync_total_s": float(run_summary["sync_total_s"]),
        "avg_step_time_ms": avg_step_time_ms,
    }


def update_manifest_with_consistency(manifest_path: Path, consistency_path: Path) -> None:
    data = json.loads(manifest_path.read_text())
    data["parameter_consistency_check_path"] = str(consistency_path)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def evaluate_group(out_root: Path,
                   comparison_group_id: str,
                   world_size: int,
                   preregistered_runs: List[Dict[str, object]]) -> Dict[str, object]:
    paired_results = []
    try:
        group_manifest = load_group_manifest(out_root)
        validate_group_manifest_structure(
            group_manifest,
            expected_world_size=world_size,
            expected_comparison_group_id=comparison_group_id,
            preregistered_runs=preregistered_runs)
        qualification_metric = group_manifest.get("qualification_metric")
        if qualification_metric is None:
            raise QualificationValidationError("Fail-QualificationMetricMissing", {})
        if qualification_metric not in ("sync_wait_s", "avg_step_time_ms"):
            raise QualificationValidationError(
                "Fail-QualificationMetricInvalid",
                {"observed": qualification_metric})

        manifest_runs = group_manifest["runs"]
        run_manifests = []
        expected_commit_hash: Optional[str] = None
        expected_seed: Optional[str] = None
        expected_bucket_size: Optional[str] = None
        for expected_run in manifest_runs:
            run_dir = Path(str(expected_run["out_dir"]))
            run_manifest = load_run_manifest(run_dir)
            if expected_commit_hash is None:
                expected_commit_hash = str(run_manifest.get("commit_hash", ""))
            if expected_seed is None:
                expected_seed = str(run_manifest.get("seed", ""))
            if expected_bucket_size is None:
                expected_bucket_size = str(run_manifest.get("bucket_size", ""))
            validate_run_manifest_binding(
                run_manifest,
                expected_run_entry=expected_run,
                expected_comparison_group_id=comparison_group_id,
                expected_world_size=world_size,
                expected_seed=expected_seed,
                expected_bucket_size=expected_bucket_size,
                expected_commit_hash=expected_commit_hash,
                run_dir=run_dir)
            run_manifests.append(run_manifest)

        baseline_by_repeat = {
            int(run["repeat_index"]): run for run in manifest_runs if run["mode"] == "bucketed"}
        overlap_by_repeat = {
            int(run["repeat_index"]): run for run in manifest_runs if run["mode"] == "overlap_bucketed"}
        if set(baseline_by_repeat.keys()) != set(overlap_by_repeat.keys()):
            raise QualificationValidationError(
                "Fail-RunSetMismatch",
                {"baseline_repeats": sorted(baseline_by_repeat.keys()),
                 "overlap_repeats": sorted(overlap_by_repeat.keys())})

        for repeat in sorted(baseline_by_repeat.keys()):
            baseline_run = baseline_by_repeat[repeat]
            overlap_run = overlap_by_repeat[repeat]
            baseline_dir = Path(str(baseline_run["out_dir"]))
            overlap_dir = Path(str(overlap_run["out_dir"]))
            consistency_path = out_root / f"comparison_repeat_{repeat:02d}.json"
            consistency = compare_parameter_snapshots(baseline_dir, overlap_dir, consistency_path)
            update_manifest_with_consistency(baseline_dir / "qualification_run_manifest.json", consistency_path)
            update_manifest_with_consistency(overlap_dir / "qualification_run_manifest.json", consistency_path)

            expected_overlap_steps = {
                tuple(step) for step in consistency["overlap_step_set"]
            }
            trace_coverage = validate_trace_step_coverage(overlap_dir, world_size, expected_overlap_steps)
            trace_analysis = analyze_overlap_trace(overlap_dir, world_size) if trace_coverage["passed"] else {
                "legal_real_early_launch_count": 0,
                "illegal_events": [],
                "rank_sequence_mismatch": [],
                "prefix_order_mismatch": [],
                "trace_legal": False,
                "rank_order_consistent": False,
            }
            baseline_perf = summarize_performance(baseline_dir)
            overlap_perf = summarize_performance(overlap_dir)
            paired = {
                "repeat_index": repeat,
                "parameter_consistency": consistency,
                "trace_coverage": trace_coverage,
                "trace_analysis": trace_analysis,
                "baseline_performance": baseline_perf,
                "overlap_performance": overlap_perf,
                "sync_wait_improved": overlap_perf["sync_wait_s"] < baseline_perf["sync_wait_s"],
                "sync_total_improved": overlap_perf["sync_total_s"] < baseline_perf["sync_total_s"],
                "avg_step_improved": overlap_perf["avg_step_time_ms"] < baseline_perf["avg_step_time_ms"],
            }
            paired_results.append(paired)

        correctness_ok = all(item["parameter_consistency"]["passed"] for item in paired_results)
        trace_coverage_ok = all(item["trace_coverage"]["passed"] for item in paired_results)
        trace_legal = all(item["trace_analysis"]["trace_legal"] for item in paired_results)
        rank_order_consistent = all(item["trace_analysis"]["rank_order_consistent"] for item in paired_results)
        any_illegal_trace = any(item["trace_analysis"]["illegal_events"] for item in paired_results)
        any_rank_mismatch = any(
            item["trace_analysis"]["rank_sequence_mismatch"] or item["trace_analysis"]["prefix_order_mismatch"]
            for item in paired_results)
        legal_early_launch_present = all(
            item["trace_analysis"]["legal_real_early_launch_count"] > 0 for item in paired_results)
        sync_wait_directions = [item["sync_wait_improved"] for item in paired_results]
        avg_step_directions = [item["avg_step_improved"] for item in paired_results]
        if qualification_metric == "sync_wait_s":
            stable_benefit = all(sync_wait_directions)
        else:
            stable_benefit = all(avg_step_directions)

        if not correctness_ok:
            public_result = "Fail"
            internal_reason = "Fail-Correctness"
        elif not trace_coverage_ok:
            public_result = "Fail"
            internal_reason = "Fail-TraceCoverage"
        elif not trace_legal:
            public_result = "Fail"
            internal_reason = "Fail-TraceLegality"
        elif not rank_order_consistent:
            public_result = "Fail"
            internal_reason = "Fail-RankOrder"
        elif not legal_early_launch_present:
            public_result = "Partial"
            internal_reason = "Partial-CorrectnessOnly"
        elif not stable_benefit:
            public_result = "Partial"
            internal_reason = "Partial-NoStableBenefit"
        else:
            public_result = "Pass"
            internal_reason = "Pass"

        nondeterministic_failure = (
            not correctness_ok or
            not trace_coverage_ok or
            any_illegal_trace or
            any_rank_mismatch or
            len(set(sync_wait_directions)) > 1 and len(set(avg_step_directions)) > 1
        )
    except QualificationValidationError as exc:
        public_result = "Fail"
        internal_reason = exc.reason
        correctness_ok = False
        trace_coverage_ok = False
        trace_legal = False
        rank_order_consistent = False
        legal_early_launch_present = False
        stable_benefit = False
        nondeterministic_failure = True
        paired_results.append({"validation_error": exc.detail})

    if public_result == "Pass":
        public_statement = (
            "Stage C has passed on the minimum qualification configuration (world_size=2)."
            if world_size == 2
            else "Stage C passed on the evaluated qualification configuration."
        )
    else:
        public_statement = "overlap_bucketed remains correctness-only pending further qualification."

    summary = {
        "comparison_group_id": comparison_group_id,
        "world_size": world_size,
        "qualification_metric": group_manifest.get("qualification_metric")
        if 'group_manifest' in locals() else None,
        "public_result": public_result,
        "internal_reason": internal_reason,
        "public_statement": public_statement,
        "correctness_ok": correctness_ok,
        "trace_coverage_ok": trace_coverage_ok,
        "trace_legal": trace_legal,
        "rank_order_consistent": rank_order_consistent,
        "legal_real_early_launch_present": legal_early_launch_present,
        "stable_benefit": stable_benefit,
        "nondeterministic_failure": nondeterministic_failure,
        "paired_results": paired_results,
    }
    summary_path = out_root / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_root).resolve()
    ppn_args = normalize_ppn_args(args.ppn_args)
    if args.repeats < 3:
        raise SystemExit("Stage C requires --repeats >= 3.")
    if args.world_size < 2:
        raise SystemExit("Stage C minimum qualification requires --world-size >= 2.")

    seed = require_cli_value(ppn_args, "--seed")
    bucket_size = require_cli_value(ppn_args, "--bucket_size_bytes")

    comparison_group_id = args.comparison_group_id or out_root.name
    commit_hash = git_commit_hash(repo_root)
    preregistered = preregister_group_manifest(
        out_root=out_root,
        comparison_group_id=comparison_group_id,
        repeats=args.repeats,
        world_size=args.world_size,
        binary=args.binary,
        ppn_args=ppn_args,
        qualification_metric=args.qualification_metric)

    run_entries = preregistered["runs"]
    for run_entry in run_entries:
        returncode, cli = execute_run(
            repo_root=repo_root,
            mpiexec=args.mpiexec,
            world_size=args.world_size,
            binary=args.binary,
            run_entry=run_entry,
            ppn_args=ppn_args)
        manifest_path = Path(str(run_entry["out_dir"])) / "qualification_run_manifest.json"
        if returncode != 0:
            run_manifest = {
                "run_id": str(run_entry["run_id"]),
                "comparison_group_id": comparison_group_id,
                "mode": str(run_entry["mode"]),
                "world_size": args.world_size,
                "seed": seed,
                "bucket_size": bucket_size,
                "commit_hash": commit_hash,
                "cli": cli,
                "stdout_path": str(Path(str(run_entry["out_dir"])) / "stdout.log"),
                "return_code": returncode,
                "status": "failed_before_artifacts",
            }
            manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        else:
            run_manifest = build_run_manifest(
                repo_root=repo_root,
                comparison_group_id=comparison_group_id,
                commit_hash=commit_hash,
                world_size=args.world_size,
                run_entry=run_entry,
                cli=cli,
                bucket_size=bucket_size,
                seed=seed)
            run_manifest["return_code"] = returncode
            run_manifest["status"] = "completed"
            manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        if returncode != 0:
            summary = {
                "comparison_group_id": comparison_group_id,
                "public_result": "Fail",
                "internal_reason": "Fail-RunExecution",
                "failed_run_id": run_entry["run_id"],
                "return_code": returncode,
            }
            (out_root / "qualification_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return returncode

    summary = evaluate_group(
        out_root=out_root,
        comparison_group_id=comparison_group_id,
        world_size=args.world_size,
        preregistered_runs=run_entries)

    print(json.dumps({
        "comparison_group_id": summary["comparison_group_id"],
        "public_result": summary["public_result"],
        "internal_reason": summary["internal_reason"],
        "public_statement": summary["public_statement"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
