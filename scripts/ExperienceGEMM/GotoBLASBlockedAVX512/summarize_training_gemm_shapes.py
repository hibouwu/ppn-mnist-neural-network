#!/usr/bin/env python3
"""Summarize MATMUL_TRACE_SHAPES CSV files into shape, family, and workload tables."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "training_shape_trace"
    / "matmul_shapes.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "ExperienceGEMM"
    / "GotoBLASBlockedAVX512"
    / "training_shape_trace"
    / "summary"
)


def classify_family(m: int, k: int, n: int, trans_a: bool, trans_b: bool) -> str:
    if trans_a and not trans_b:
        if m <= 32 and k >= 1000:
            return "conv_dw_transposed_very_skinny_m_nn"
        if m <= 32 and k >= 100:
            return "conv_dw_transposed_large_m_nn"
        if m <= 200 and k >= 50:
            return "fc_weight_grad_transposed_nn"
        return "transposed_weight_grad_nn"
    if not trans_a and trans_b:
        if k <= 50 and n <= 32:
            return "conv_fwd_very_skinny_k_small_n_nn"
        if k <= 200 and n <= 32:
            return "conv_fwd_medium_k_nn"
        if n <= 32:
            return "fc_head_small_n_nn"
        return "fc_forward_mainstream_nn"
    if not trans_a and not trans_b:
        if k <= 32 and m >= 500:
            return "conv_dx_extremely_skinny_k_nn"
        if n <= 32 and k >= 50:
            return "fc_head_small_n_nn"
        if k >= 300:
            return "fc_forward_mainstream_nn"
        if k >= 50:
            return "fc_cnn_flatten_medium_nn"
        return "unknown_training_gemm"
    return "unknown_training_gemm"


def make_workload_id(m: int, k: int, n: int, trans_a: bool, trans_b: bool) -> str:
    suffix = ""
    if trans_a:
        suffix += "_tA"
    if trans_b:
        suffix += "_tB"
    return f"m{m}_k{k}_n{n}{suffix}_b32"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize training GEMM shape traces.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input matmul_shapes.csv (from MATMUL_TRACE_SHAPES=1)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-flops-share-pct", type=float, default=0.5,
                        help="Minimum FLOPs share %% to include in recommended_workloads")
    return parser


def fmt(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.6f}"


def main() -> int:
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", flush=True)
        print("Run training with MATMUL_TRACE_SHAPES=1 MATMUL_TRACE_SHAPES_FILE=<path>")
        return 1

    # Read raw trace
    raw_rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8")))
    if not raw_rows:
        print("Empty trace file.")
        return 1

    # Aggregate by (M, K, N, transA, transB, impl)
    counts: dict[tuple[int, int, int, bool, bool, str], int] = defaultdict(int)
    for row in raw_rows:
        key = (
            int(row["M"]), int(row["K"]), int(row["N"]),
            row["transA"] == "1", row["transB"] == "1",
            row.get("impl", "blas"),
        )
        counts[key] += 1

    total_flops = sum(2 * m * k * n * v for (m, k, n, _, _, _), v in counts.items())

    # Build shape_summary
    shape_rows = []
    for (m, k, n, ta, tb, impl), count in sorted(
        counts.items(), key=lambda x: -2 * x[0][0] * x[0][1] * x[0][2] * x[1]
    ):
        flops = 2 * m * k * n * count
        family = classify_family(m, k, n, ta, tb)
        shape_rows.append({
            "M": m, "K": k, "N": n,
            "transA": int(ta), "transB": int(tb),
            "impl": impl,
            "Count": count,
            "TotalFLOPs": flops,
            "AvgFLOPsPerCall": 2 * m * k * n,
            "ShareOfCallsPct": fmt(100.0 * count / len(raw_rows)),
            "ShareOfFLOPsPct": fmt(100.0 * flops / total_flops) if total_flops > 0 else "",
            "RepresentativeFamily": family,
            "Notes": "",
        })

    # Build family_summary
    family_agg: dict[str, dict] = defaultdict(lambda: {"Count": 0, "TotalFLOPs": 0, "Shapes": []})
    for row in shape_rows:
        fam = row["RepresentativeFamily"]
        family_agg[fam]["Count"] += row["Count"]
        family_agg[fam]["TotalFLOPs"] += row["TotalFLOPs"]
        family_agg[fam]["Shapes"].append(f"{row['M']}x{row['K']}x{row['N']}")

    family_rows = []
    for fam, agg in sorted(family_agg.items(), key=lambda x: -x[1]["TotalFLOPs"]):
        flops = agg["TotalFLOPs"]
        family_rows.append({
            "Family": fam,
            "Count": agg["Count"],
            "TotalFLOPs": flops,
            "ShareOfCallsPct": fmt(100.0 * agg["Count"] / len(raw_rows)),
            "ShareOfFLOPsPct": fmt(100.0 * flops / total_flops) if total_flops > 0 else "",
            "RepresentativeShapes": "|".join(sorted(set(agg["Shapes"]))),
        })

    # Build recommended_workloads (transA=false, transB=false normalized shapes only, by FLOPs share)
    recommended: list[dict] = []
    seen_shapes: set[tuple[int, int, int]] = set()
    for row in shape_rows:
        m, k, n = row["M"], row["K"], row["N"]
        ta, tb = row["transA"], row["transB"]
        share = float(row["ShareOfFLOPsPct"]) if row["ShareOfFLOPsPct"] else 0.0

        # Include shapes with significant FLOPs share
        if share < args.min_flops_share_pct:
            continue

        # Normalize transposed variants: canonical benchmark shape is (M×K)@(K×N)
        # as returned by matmul_into M,K,N (already canonical regardless of transA/transB)
        shape_key = (m, k, n)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)

        family = row["RepresentativeFamily"]
        workload_id = make_workload_id(m, k, n, ta == 1, tb == 1)
        reason = f"FLOPs share {share:.1f}%; family={family}"

        recommended.append({
            "WorkloadFamily": family,
            "WorkloadId": workload_id,
            "M": m, "K": k, "N": n,
            "transA": ta, "transB": tb,
            "Source": "training_trace_cnn_mlp_mnist_b32_1epoch",
            "Reason": reason,
            "Count": row["Count"],
            "ShareOfFLOPsPct": row["ShareOfFLOPsPct"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    write_csv(
        args.output_dir / "shape_summary.csv",
        ["M", "K", "N", "transA", "transB", "impl", "Count", "TotalFLOPs",
         "AvgFLOPsPerCall", "ShareOfCallsPct", "ShareOfFLOPsPct", "RepresentativeFamily", "Notes"],
        shape_rows,
    )
    write_csv(
        args.output_dir / "family_summary.csv",
        ["Family", "Count", "TotalFLOPs", "ShareOfCallsPct", "ShareOfFLOPsPct", "RepresentativeShapes"],
        family_rows,
    )
    write_csv(
        args.output_dir / "recommended_workloads.csv",
        ["WorkloadFamily", "WorkloadId", "M", "K", "N", "transA", "transB",
         "Source", "Reason", "Count", "ShareOfFLOPsPct"],
        recommended,
    )

    md_lines = [
        "# Training GEMM Shape Summary",
        "",
        f"Source: `{args.input}`",
        f"Total trace calls: {len(raw_rows):,}",
        f"Total FLOPs (estimated): {total_flops/1e9:.3f} GFlops",
        "",
        "## Shape Summary (by FLOPs descending)",
        "",
        "| M | K | N | tA | tB | Count | TotalGFlops | FLOPs% | Family |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in shape_rows:
        gf = row["TotalFLOPs"] / 1e9
        md_lines.append(
            f"| {row['M']} | {row['K']} | {row['N']} | {row['transA']} | {row['transB']} "
            f"| {row['Count']} | {gf:.3f} | {row['ShareOfFLOPsPct']}% | {row['RepresentativeFamily']} |"
        )

    md_lines.extend([
        "",
        "## Family Summary",
        "",
        "| Family | Count | TotalGFlops | FLOPs% | Representative Shapes |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in family_rows:
        gf = row["TotalFLOPs"] / 1e9
        md_lines.append(
            f"| {row['Family']} | {row['Count']} | {gf:.3f} "
            f"| {row['ShareOfFLOPsPct']}% | {row['RepresentativeShapes']} |"
        )

    md_lines.extend([
        "",
        "## Recommended Benchmark Workloads",
        "",
        "Only shapes with FLOPs share ≥ threshold are listed.",
        "Shapes are the canonical (M×K)@(K×N) form after applying transpose flags.",
        "",
        "| WorkloadFamily | WorkloadId | M | K | N | Source | FLOPs% |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in recommended:
        md_lines.append(
            f"| {row['WorkloadFamily']} | {row['WorkloadId']} | {row['M']} | {row['K']} | {row['N']} "
            f"| {row['Source']} | {row['ShareOfFLOPsPct']}% |"
        )

    (args.output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote summary under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
