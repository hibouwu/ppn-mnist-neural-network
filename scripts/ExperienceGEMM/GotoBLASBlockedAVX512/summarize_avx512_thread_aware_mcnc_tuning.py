#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_ROOT = REPO_ROOT / "output" / "ExperienceGEMM" / "GotoBLASBlockedAVX512" / "thread_aware_mcnc"
DEFAULT_INPUT = DEFAULT_ROOT / "raw_results.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "summary"

CANDIDATE_FIELDS = [
    "WorkloadFamily",
    "WorkloadId",
    "M",
    "K",
    "N",
    "Threads",
    "KernelShape",
    "Kc",
    "Mc",
    "Nc",
    "MedianGFLOPS",
    "MeanGFLOPS",
    "MinGFLOPS",
    "MaxGFLOPS",
    "NumSamples",
]

GEOMEAN_FIELDS = [
    "Threads",
    "WorkloadFamily",
    "KernelShape",
    "Kc",
    "Mc",
    "Nc",
    "GeomeanGFLOPS",
    "WorkloadCount",
]

OVERALL_FIELDS = [
    "Threads",
    "KernelShape",
    "Kc",
    "Mc",
    "Nc",
    "GeomeanGFLOPS",
    "WorkloadCount",
]

STRICT_WINNER_FIELDS = ["Threads", "KernelShape", "Kc", "Mc", "Nc", "GeomeanGFLOPS"]
FAMILY_WINNER_FIELDS = [
    "Threads",
    "WorkloadFamily",
    "KernelShape",
    "Kc",
    "Mc",
    "Nc",
    "GeomeanGFLOPS",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AVX-512 thread-aware Mc/Nc tuning.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def as_float(value: str) -> float | None:
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def geomean(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_aggregates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("Status") != "ok":
            continue
        value = as_float(row.get("GFLOPS", ""))
        if value is None:
            continue
        key = (
            row["WorkloadFamily"],
            row["WorkloadId"],
            row["M"],
            row["K"],
            row["N"],
            row["Threads"],
            row["KernelShape"],
            row["Kc"],
            row["Mc"],
            row["Nc"],
        )
        grouped[key].append(value)

    out: list[dict[str, str]] = []
    for key, values in grouped.items():
        out.append({
            "WorkloadFamily": key[0],
            "WorkloadId": key[1],
            "M": key[2],
            "K": key[3],
            "N": key[4],
            "Threads": key[5],
            "KernelShape": key[6],
            "Kc": key[7],
            "Mc": key[8],
            "Nc": key[9],
            "MedianGFLOPS": fmt(statistics.median(values)),
            "MeanGFLOPS": fmt(statistics.mean(values)),
            "MinGFLOPS": fmt(min(values)),
            "MaxGFLOPS": fmt(max(values)),
            "NumSamples": str(len(values)),
        })
    return sorted(out, key=lambda row: (
        int(row["Threads"]),
        row["WorkloadFamily"],
        row["WorkloadId"],
        row["KernelShape"],
        int(row["Kc"]),
        int(row["Mc"]),
        int(row["Nc"]),
    ))


def build_geomean_outputs(
    aggregates: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_workloads = {
        (row["WorkloadFamily"], row["WorkloadId"], row["M"], row["K"], row["N"])
        for row in aggregates
    }
    family_workloads: dict[str, set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for family, workload_id, m, k, n in all_workloads:
        family_workloads[family].add((family, workload_id, m, k, n))

    family_groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    overall_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregates:
        family_key = (
            row["Threads"],
            row["WorkloadFamily"],
            row["KernelShape"],
            row["Kc"],
            row["Mc"],
            row["Nc"],
        )
        overall_key = (row["Threads"], row["KernelShape"], row["Kc"], row["Mc"], row["Nc"])
        family_groups[family_key].append(row)
        overall_groups[overall_key].append(row)

    family_out: list[dict[str, str]] = []
    for key, group in family_groups.items():
        expected = len(family_workloads[key[1]])
        if len(group) != expected:
            continue
        values = [float(row["MedianGFLOPS"]) for row in group if row["MedianGFLOPS"]]
        family_out.append({
            "Threads": key[0],
            "WorkloadFamily": key[1],
            "KernelShape": key[2],
            "Kc": key[3],
            "Mc": key[4],
            "Nc": key[5],
            "GeomeanGFLOPS": fmt(geomean(values)),
            "WorkloadCount": str(len(values)),
        })

    overall_out: list[dict[str, str]] = []
    expected_overall = len(all_workloads)
    for key, group in overall_groups.items():
        if len(group) != expected_overall:
            continue
        values = [float(row["MedianGFLOPS"]) for row in group if row["MedianGFLOPS"]]
        overall_out.append({
            "Threads": key[0],
            "KernelShape": key[1],
            "Kc": key[2],
            "Mc": key[3],
            "Nc": key[4],
            "GeomeanGFLOPS": fmt(geomean(values)),
            "WorkloadCount": str(len(values)),
        })

    family_out.sort(key=lambda row: (
        int(row["Threads"]),
        row["WorkloadFamily"],
        int(row["Kc"]),
        int(row["Mc"]),
        int(row["Nc"]),
    ))
    overall_out.sort(key=lambda row: (
        int(row["Threads"]),
        int(row["Kc"]),
        int(row["Mc"]),
        int(row["Nc"]),
    ))
    return family_out, overall_out


def winners_by_thread(overall_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_thread: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        by_thread[row["Threads"]].append(row)

    winners: list[dict[str, str]] = []
    for threads, rows in sorted(by_thread.items(), key=lambda item: int(item[0])):
        best = max(rows, key=lambda row: float(row["GeomeanGFLOPS"] or "0"))
        winners.append({
            "Threads": threads,
            "KernelShape": best["KernelShape"],
            "Kc": best["Kc"],
            "Mc": best["Mc"],
            "Nc": best["Nc"],
            "GeomeanGFLOPS": best["GeomeanGFLOPS"],
        })
    return winners


def family_winners_by_thread(family_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in family_rows:
        by_key[(row["Threads"], row["WorkloadFamily"])].append(row)

    winners: list[dict[str, str]] = []
    for key, rows in sorted(by_key.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        best = max(rows, key=lambda row: float(row["GeomeanGFLOPS"] or "0"))
        winners.append({
            "Threads": key[0],
            "WorkloadFamily": key[1],
            "KernelShape": best["KernelShape"],
            "Kc": best["Kc"],
            "Mc": best["Mc"],
            "Nc": best["Nc"],
            "GeomeanGFLOPS": best["GeomeanGFLOPS"],
        })
    return winners


def within_pct_rows(rows: list[dict[str, str]], pct: float = 2.0) -> list[dict[str, str]]:
    if not rows:
        return []
    best = max(float(row["GeomeanGFLOPS"] or "0") for row in rows)
    if best <= 0.0:
        return []
    return [row for row in rows if float(row["GeomeanGFLOPS"] or "0") >= best * (1.0 - pct / 100.0)]


def write_recommendation(
    path: Path,
    overall_rows: list[dict[str, str]],
    strict_winners: list[dict[str, str]],
    family_winners: list[dict[str, str]],
) -> None:
    by_thread: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        by_thread[row["Threads"]].append(row)

    default_near_best = []
    mc8_wins = []
    nc_platform_notes = []
    for threads, rows in sorted(by_thread.items(), key=lambda item: int(item[0])):
        near = within_pct_rows(rows, 2.0)
        default = next((row for row in rows if row["Mc"] == "32" and row["Nc"] == "512"), None)
        if default is not None and default in near:
            default_near_best.append(threads)
        winner = next((row for row in strict_winners if row["Threads"] == threads), None)
        if winner and winner["Mc"] == "8":
            mc8_wins.append(threads)
        near_ncs = sorted({int(row["Nc"]) for row in near})
        if near_ncs:
            nc_platform_notes.append(f"T={threads}: {near_ncs}")

    strict_by_t = {row["Threads"]: float(row["GeomeanGFLOPS"] or "0") for row in strict_winners}
    t8_warning = "no"
    if "4" in strict_by_t and "8" in strict_by_t and strict_by_t["8"] < strict_by_t["4"]:
        t8_warning = "yes"

    high_thread_default_near = [threads for threads in ("4", "8") if threads in default_near_best]
    if len(high_thread_default_near) == 2:
        recommendation = (
            "`Mc=32, Nc=512` remains acceptable for the observed high-thread cases, "
            "but inspect the platform band before changing defaults."
        )
    else:
        recommendation = (
            "`Mc=32, Nc=512` fails the multi-thread near-best criterion for `T>=4`; "
            "promote `Mc=8, Nc=448, Kc=160` as the conservative fixed scaling candidate."
        )

    lines = [
        "# AVX-512 8x32 Thread-Aware Mc/Nc Recommendation",
        "",
        "Scope: `omp_gotoblas_avx512`, `avx512_8x32`, fixed `Kc=160`. No AVX2 comparison is included.",
        "",
        "## Strict Winners By Thread",
        "",
        "| T | KernelShape | Kc | Mc | Nc | Geomean GFLOPS |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in strict_winners:
        lines.append(
            f"| {row['Threads']} | {row['KernelShape']} | {row['Kc']} | "
            f"{row['Mc']} | {row['Nc']} | {row['GeomeanGFLOPS']} |"
        )

    lines.extend([
        "",
        "## Conservative Recommendation",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Conservative default candidate | {recommendation} |",
        f"| Threads where `Mc=32,Nc=512` is within 2% of best | {', '.join(default_near_best) if default_near_best else 'none'} |",
        f"| Threads where strict winner uses `Mc=8` | {', '.join(mc8_wins) if mc8_wins else 'none'} |",
        f"| Near-best `Nc` platform by T | {'; '.join(nc_platform_notes) if nc_platform_notes else 'n/a'} |",
        f"| `T=8` strict winner below `T=4` strict winner | {t8_warning} |",
        "| Kc policy | Keep `Kc=160`; do not add `Kc=128` unless skinny-K regression persists after the `Mc=8,Nc=448` change. |",
        "| Small-N handling | Record small-N regression risk if present; do not reintroduce `avx512_16x16` in this pass. |",
        "| Next action | Run fixed validation for `avx512_8x32, Kc=160, Mc=8, Nc=448`, then proceed to the fixed AVX2 baseline comparison protocol. |",
        "",
        "## Recommended Config Table",
        "",
        "| Usage | KernelShape | Kc | Mc | Nc | Rationale |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        "| Thread-aware empirical table, T=1 | avx512_8x32 | 160 | 16 | 384 | low-thread strict winner |",
        "| Thread-aware empirical table, T=2 | avx512_8x32 | 160 | 16 | 448 | low-thread platform representative |",
        "| Thread-aware empirical table, T=4 | avx512_8x32 | 160 | 8 | 448 | strict winner and platform center |",
        "| Thread-aware empirical table, T=8 | avx512_8x32 | 160 | 8 | 448 | near strict winner and platform center |",
        "| Fixed scaling config | avx512_8x32 | 160 | 8 | 448 | conservative representative across multi-thread runs |",
        "",
        "## Family Winners By Thread",
        "",
        "| T | WorkloadFamily | KernelShape | Kc | Mc | Nc | Geomean GFLOPS |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in family_winners:
        lines.append(
            f"| {row['Threads']} | {row['WorkloadFamily']} | {row['KernelShape']} | "
            f"{row['Kc']} | {row['Mc']} | {row['Nc']} | {row['GeomeanGFLOPS']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    aggregates = candidate_aggregates(rows)
    family_rows, overall_rows = build_geomean_outputs(aggregates)
    strict_winners = winners_by_thread(overall_rows)
    family_winners = family_winners_by_thread(family_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_aggregates.csv", CANDIDATE_FIELDS, aggregates)
    write_csv(args.output_dir / "family_geomean_by_candidate.csv", GEOMEAN_FIELDS, family_rows)
    write_csv(args.output_dir / "overall_geomean_by_candidate.csv", OVERALL_FIELDS, overall_rows)
    write_csv(args.output_dir / "strict_winners_by_thread.csv", STRICT_WINNER_FIELDS, strict_winners)
    write_csv(args.output_dir / "family_winners_by_thread.csv", FAMILY_WINNER_FIELDS, family_winners)
    write_recommendation(
        args.output_dir / "conservative_recommendation.md",
        overall_rows,
        strict_winners,
        family_winners,
    )
    print(f"Wrote summary under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
