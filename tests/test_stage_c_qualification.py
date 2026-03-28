import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.Performance.stage_c_qualification import evaluate_group


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class StageCQualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="stagec_gate_", dir="/tmp")
        self.root = Path(self.tmp.name)
        self.world_size = 2
        self.group_id = "group_ws2"
        self.commit_hash = "abc123"
        self.seed = "42"
        self.bucket_size = "1048576"

    def tearDown(self):
        self.tmp.cleanup()

    def _make_run(self, repeat, mode):
        run_id = f"{mode}_r{repeat:02d}"
        out_dir = self.root / "runs" / run_id
        qdir = out_dir / "qualification"
        snapshot_dir = qdir / "parameter_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "profile_run_summary.csv",
            ["run_id", "grad_sync_mode_label", "grad_sync_semantics",
             "grad_sync_correctness_only", "total_epochs", "fwd_bwd_s",
             "sync_total_s", "sync_wait_s", "sync_effective_overlap",
             "opt_s", "profiled_total_us"],
            [[run_id, mode, mode, 1 if mode == "overlap_bucketed" else 0,
              1, 1.0, 8.0 if mode == "overlap_bucketed" else 10.0,
              3.0 if mode == "overlap_bucketed" else 5.0, 0, 1.0, 100]],
        )
        write_csv(out_dir / "metrics.csv", ["avg_step_time_ms"], [[90 if mode == "overlap_bucketed" else 100]])
        write_csv(qdir / "parameter_layout.csv", ["param_idx", "rows", "cols", "numel"], [[0, 1, 2, 2]])
        write_csv(
            qdir / "parameter_snapshot_index.csv",
            ["epoch", "step", "relative_path", "total_values", "total_bytes"],
            [[1, 1, "parameter_snapshots/s1.bin", 2, 16]],
        )
        (snapshot_dir / "s1.bin").write_bytes(b"0123456789ABCDEF")
        trace_paths = {}
        for rank in range(self.world_size):
            trace_path = qdir / f"rank_{rank:03d}" / "sync_trace.csv"
            write_csv(
                trace_path,
                ["run_id", "epoch", "step", "bucket_idx", "expected_count", "ready_count",
                 "outstanding_requests", "seconds_since_begin_step",
                 "seconds_since_backward_complete", "channel", "reason", "request_lifecycle"],
                ([[run_id, 1, 1, 0, 1, 1, 1, 0.1, -0.01, "real_comm", "real_early_launch", "in_flight"]]
                 if mode == "overlap_bucketed" else []),
            )
            trace_paths[f"rank_{rank:03d}"] = str(trace_path)

        run_manifest = {
            "run_id": run_id,
            "comparison_group_id": self.group_id,
            "mode": mode,
            "world_size": self.world_size,
            "seed": self.seed,
            "bucket_size": self.bucket_size,
            "commit_hash": self.commit_hash,
            "cli": [],
            "stdout_path": str(out_dir / "stdout.log"),
            "timing_path": str(out_dir / "metrics.csv"),
            "profile_path": str(out_dir / "profile_run_summary.csv"),
            "trace_path": trace_paths["rank_000"],
            "trace_paths": trace_paths,
            "parameter_snapshot_index_path": str(qdir / "parameter_snapshot_index.csv"),
            "parameter_layout_path": str(qdir / "parameter_layout.csv"),
            "parameter_consistency_check_path": "",
            "mode_metadata": {"label": mode, "semantics": mode, "correctness_only": int(mode == "overlap_bucketed")},
            "status": "completed",
        }
        (out_dir / "qualification_run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
        return {
            "run_id": run_id,
            "repeat_index": repeat,
            "mode": mode,
            "out_dir": str(out_dir),
            "stdout_path": str(out_dir / "stdout.log"),
        }

    def _write_group_manifest(self, runs):
        manifest = {
            "comparison_group_id": self.group_id,
            "world_size": self.world_size,
            "repeats": 3,
            "binary": "./build-mpi/ppn_train",
            "ppn_args": ["--seed", self.seed, "--bucket_size_bytes", self.bucket_size],
            "qualification_metric": "avg_step_time_ms",
            "runs": runs,
        }
        (self.root / "qualification_group_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _build_valid_fixture(self):
        runs = []
        for repeat in range(1, 4):
            runs.append(self._make_run(repeat, "bucketed"))
            runs.append(self._make_run(repeat, "overlap_bucketed"))
        self._write_group_manifest(runs)
        return runs

    def test_manifest_binding_error_fails(self):
        runs = self._build_valid_fixture()
        bad_manifest = json.loads((self.root / "runs" / "bucketed_r01" / "qualification_run_manifest.json").read_text())
        bad_manifest["mode"] = "overlap_bucketed"
        (self.root / "runs" / "bucketed_r01" / "qualification_run_manifest.json").write_text(json.dumps(bad_manifest, indent=2))
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-ManifestBinding")

    def test_trace_missing_step_fails(self):
        runs = self._build_valid_fixture()
        for mode in ("bucketed_r01", "overlap_bucketed_r01"):
            qdir = self.root / "runs" / mode / "qualification"
            write_csv(
                qdir / "parameter_snapshot_index.csv",
                ["epoch", "step", "relative_path", "total_values", "total_bytes"],
                [[1, 1, "parameter_snapshots/s1.bin", 2, 16],
                 [1, 2, "parameter_snapshots/s2.bin", 2, 16]],
            )
            (qdir / "parameter_snapshots" / "s2.bin").write_bytes(b"FEDCBA9876543210")
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-TraceCoverage")

    def test_rank_trace_step_set_mismatch_fails(self):
        runs = self._build_valid_fixture()
        trace_path = self.root / "runs" / "overlap_bucketed_r01" / "qualification" / "rank_001" / "sync_trace.csv"
        write_csv(
            trace_path,
            ["run_id", "epoch", "step", "bucket_idx", "expected_count", "ready_count",
             "outstanding_requests", "seconds_since_begin_step",
             "seconds_since_backward_complete", "channel", "reason", "request_lifecycle"],
            [],
        )
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-TraceCoverage")

    def test_parameter_layout_or_index_mismatch_fails(self):
        runs = self._build_valid_fixture()
        write_csv(
            self.root / "runs" / "overlap_bucketed_r01" / "qualification" / "parameter_layout.csv",
            ["param_idx", "rows", "cols", "numel"],
            [[0, 1, 3, 3]],
        )
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-Correctness")
        self.assertEqual(summary["paired_results"][0]["parameter_consistency"]["reason"], "Fail-ParameterLayoutMismatch")

    def test_qualification_metric_missing_fails(self):
        runs = self._build_valid_fixture()
        group_manifest_path = self.root / "qualification_group_manifest.json"
        group_manifest = json.loads(group_manifest_path.read_text())
        del group_manifest["qualification_metric"]
        group_manifest_path.write_text(json.dumps(group_manifest, indent=2))
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-QualificationMetricMissing")

    def test_qualification_metric_invalid_fails(self):
        runs = self._build_valid_fixture()
        group_manifest_path = self.root / "qualification_group_manifest.json"
        group_manifest = json.loads(group_manifest_path.read_text())
        group_manifest["qualification_metric"] = "sync_total_s"
        group_manifest_path.write_text(json.dumps(group_manifest, indent=2))
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-QualificationMetricInvalid")

    def test_avg_step_metric_blocks_pass_even_if_sync_wait_improves(self):
        runs = self._build_valid_fixture()
        for repeat in range(1, 4):
            overlap_metrics = self.root / "runs" / f"overlap_bucketed_r{repeat:02d}" / "metrics.csv"
            write_csv(overlap_metrics, ["avg_step_time_ms"], [[110]])
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertNotEqual(summary["public_result"], "Pass")
        self.assertEqual(summary["internal_reason"], "Partial-NoStableBenefit")

    def test_real_comm_with_simulated_only_lifecycle_fails_trace_legality(self):
        runs = self._build_valid_fixture()
        trace_path = self.root / "runs" / "overlap_bucketed_r01" / "qualification" / "rank_000" / "sync_trace.csv"
        write_csv(
            trace_path,
            ["run_id", "epoch", "step", "bucket_idx", "expected_count", "ready_count",
             "outstanding_requests", "seconds_since_begin_step",
             "seconds_since_backward_complete", "channel", "reason", "request_lifecycle"],
            [["overlap_bucketed_r01", 1, 1, 0, 1, 1, 0, 0.1, -0.01, "real_comm", "real_early_launch", "simulated_only"]],
        )
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-TraceLegality")

    def test_structural_zero_expected_launch_with_nonzero_expected_count_fails(self):
        runs = self._build_valid_fixture()
        trace_path = self.root / "runs" / "overlap_bucketed_r01" / "qualification" / "rank_000" / "sync_trace.csv"
        write_csv(
            trace_path,
            ["run_id", "epoch", "step", "bucket_idx", "expected_count", "ready_count",
             "outstanding_requests", "seconds_since_begin_step",
             "seconds_since_backward_complete", "channel", "reason", "request_lifecycle"],
            [["overlap_bucketed_r01", 1, 1, 0, 1, 0, 0, 0.1, 0.02, "real_comm", "structural_zero_expected_launch", "completed"]],
        )
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-TraceLegality")

    def test_tail_flush_before_backward_complete_fails(self):
        runs = self._build_valid_fixture()
        trace_path = self.root / "runs" / "overlap_bucketed_r01" / "qualification" / "rank_000" / "sync_trace.csv"
        write_csv(
            trace_path,
            ["run_id", "epoch", "step", "bucket_idx", "expected_count", "ready_count",
             "outstanding_requests", "seconds_since_begin_step",
             "seconds_since_backward_complete", "channel", "reason", "request_lifecycle"],
            [["overlap_bucketed_r01", 1, 1, 0, 1, 1, 0, 0.1, -0.02, "real_comm", "tail_flush_launch", "completed"]],
        )
        summary = evaluate_group(self.root, self.group_id, self.world_size, runs)
        self.assertEqual(summary["public_result"], "Fail")
        self.assertEqual(summary["internal_reason"], "Fail-TraceLegality")


if __name__ == "__main__":
    unittest.main()
