#!/usr/bin/env python3
"""
Emit Tables A/B/C for oak vs random benchmark comparison (fixed layout).

  Table A — μ, σ across env rows + P(Δ>0) (only table with P(Δ>0)).
  Table B — W, L, T counts.
  Table C — mean oak, mean random, mean Δ.

Truth runs: benchmark.json with schema oak.benchmark.truth_first.v8 (tasks[].oak|random.reward.sum.mean).

Partial runs: a metrics/ directory of per-seed JSON files; we average rewardSum per (envId, policy),
  then one row per env where both policies exist.

Usage:
  ./scripts/oak_benchmark_tables.py
  ./scripts/oak_benchmark_tables.py --partial .oak-benchmarks/20260416-070404-96998/metrics
  ./scripts/oak_benchmark_tables.py --gemma .oak-benchmarks/gemma4/benchmark.json --gpt ".oak-benchmarks/gpt 5.4/benchmark.json"
  ./scripts/oak_benchmark_tables.py --only HalfCheetah-v4,Walker2d-v4 --partial .oak-benchmarks/<run>/metrics
  ./scripts/oak_benchmark_tables.py --partial .oak-benchmarks/<run>/metrics --no-align-to-partial  # full n in truth JSONs

Run from repo root (or pass absolute paths).

With --partial, gemma/gpt are restricted to the same envIds as paired metrics in that directory (same n) unless --no-align-to-partial.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

EPS = 1e-9


def classify(delta: float) -> str:
    if delta > EPS:
        return "W"
    if delta < -EPS:
        return "L"
    return "T"


def rows_from_truth(path: Path) -> list[tuple[str, float, float, float, str]]:
    with path.open() as f:
        data = json.load(f)
    rows: list[tuple[str, float, float, float, str]] = []
    for t in data["tasks"]:
        eid = t["envId"]
        o = t["oak"]["reward"]["sum"]["mean"]
        r = t["random"]["reward"]["sum"]["mean"]
        dlt = o - r
        rows.append((eid, o, r, dlt, classify(dlt)))
    return rows


def rows_from_metrics(mdir: Path) -> list[tuple[str, float, float, float, str]]:
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"oak": [], "random": []})
    for fp in sorted(mdir.glob("*.json")):
        with fp.open() as f:
            j = json.load(f)
        pol = j.get("policy")
        if pol not in ("oak", "random"):
            continue
        eid = j.get("envId")
        rs = j.get("rewardSum")
        if eid is None or rs is None:
            continue
        agg[eid][pol].append(rs)
    rows: list[tuple[str, float, float, float, str]] = []
    for eid in sorted(agg.keys()):
        dd = agg[eid]
        if not dd["oak"] or not dd["random"]:
            continue
        mo = mean(dd["oak"])
        mr = mean(dd["random"])
        dlt = mo - mr
        rows.append((eid, mo, mr, dlt, classify(dlt)))
    return rows


def filter_env_rows(
    rows: list[tuple[str, float, float, float, str]], env_ids: set[str] | None
) -> list[tuple[str, float, float, float, str]]:
    if not env_ids:
        return rows
    return [x for x in rows if x[0] in env_ids]


def stats(rows: list[tuple[str, float, float, float, str]]) -> dict:
    o = [x[1] for x in rows]
    r = [x[2] for x in rows]
    d = [x[3] for x in rows]
    n = len(rows)
    w = sum(1 for x in rows if x[4] == "W")
    l = sum(1 for x in rows if x[4] == "L")
    t = sum(1 for x in rows if x[4] == "T")
    p_pos = (w / n) if n else float("nan")
    return {
        "n": n,
        "mu_oak": mean(o) if n else float("nan"),
        "sd_oak": pstdev(o) if n > 1 else 0.0,
        "mu_rand": mean(r) if n else float("nan"),
        "sd_rand": pstdev(r) if n > 1 else 0.0,
        "mu_delta": mean(d) if n else float("nan"),
        "sd_delta": pstdev(d) if n > 1 else 0.0,
        "wins": w,
        "losses": l,
        "ties": t,
        "p_delta_gt_0": p_pos,
    }


def fmt_sc(x: float) -> str:
    if x != x:  # nan
        return "—"
    return f"{x:.3e}"


def fmt_p(x: float) -> str:
    if x != x:
        return "—"
    return f"{x:.3f}"


def print_tables(
    runs: list[tuple[str, list[tuple[str, float, float, float, str]]]],
    env_filter_note: str | None = None,
    partial_comparison_note: str | None = None,
) -> None:
    print("DEFINITIONS")
    if partial_comparison_note:
        print(f"  {partial_comparison_note}")
    if env_filter_note:
        print(f"  Env filter: {env_filter_note}")
    print("  Row = one env. Δ = mean oak total reward − mean random total reward (means over seeds in partial metrics).")
    print(f"  W / L / T: count envs with Δ > {EPS:g} / Δ < -{EPS:g} / |Δ| ≤ {EPS:g}.")
    print("  P(Δ>0) = W/n (reported in Table A only). μ, σ: population std of per-env means across rows.")
    print()

    print("TABLE A — Summary (mean reward, variance, P(Δ>0))")
    print("| Run | n | μ_oak | σ_oak | μ_rand | σ_rand | μ_Δ | σ_Δ | P(Δ>0) |")
    print("|-----|---|-------|-------|--------|--------|-----|-----|--------|")
    for label, rows in runs:
        s = stats(rows)
        print(
            f"| {label} | {s['n']} | {fmt_sc(s['mu_oak'])} | {fmt_sc(s['sd_oak'])} | "
            f"{fmt_sc(s['mu_rand'])} | {fmt_sc(s['sd_rand'])} | {fmt_sc(s['mu_delta'])} | {fmt_sc(s['sd_delta'])} | {fmt_p(s['p_delta_gt_0'])} |"
        )

    print()
    print("TABLE B — Wins / losses / ties")
    print("| Run | n | W | L | T |")
    print("|-----|---|---|---|---|")
    for label, rows in runs:
        s = stats(rows)
        n = s["n"]
        if n == 0:
            print(f"| {label} | 0 | — | — | — |")
            continue
        print(f"| {label} | {n} | {s['wins']} | {s['losses']} | {s['ties']} |")

    print()
    print("TABLE C — Mean total reward")
    print("| Run | n | mean oak reward | mean random reward | mean Δ |")
    print("|-----|---|-----------------|-------------------|--------|")
    for label, rows in runs:
        s = stats(rows)
        print(
            f"| {label} | {s['n']} | {fmt_sc(s['mu_oak'])} | {fmt_sc(s['mu_rand'])} | {fmt_sc(s['mu_delta'])} |"
        )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Benchmark Tables A/B/C (oak vs random).")
    p.add_argument(
        "--gemma",
        type=Path,
        default=root / ".oak-benchmarks/gemma4/benchmark.json",
        help="Path to gemma4 benchmark.json",
    )
    p.add_argument(
        "--gpt",
        type=Path,
        default=root / ".oak-benchmarks/gpt 5.4/benchmark.json",
        help="Path to gpt 5.4 benchmark.json",
    )
    p.add_argument(
        "--partial",
        type=Path,
        default=None,
        metavar="METRICS_DIR",
        help="Optional metrics/ directory (per-seed JSON) for a partial run.",
    )
    p.add_argument(
        "--partial-label",
        default=None,
        help="Label for partial run (default: parent of METRICS_DIR).",
    )
    p.add_argument(
        "--only",
        default=None,
        metavar="ENV_IDS",
        help="Comma-separated envId list (e.g. HalfCheetah-v4,Walker2d-v4). Restricts all inputs to these envs.",
    )
    p.add_argument(
        "--no-align-to-partial",
        action="store_true",
        help="With --partial, do not restrict gemma/gpt to partial envIds (truth n may differ from partial). Default is aligned.",
    )
    args = p.parse_args()

    env_ids: set[str] | None = None
    env_note: str | None = None
    if args.only:
        env_ids = {s.strip() for s in args.only.split(",") if s.strip()}
        env_note = ", ".join(sorted(env_ids))

    partial_dir_ok = args.partial is not None and args.partial.is_dir()
    align_to_partial = partial_dir_ok and not args.no_align_to_partial

    partial_comparison_note: str | None = None
    align_ids: set[str] | None = None
    if align_to_partial:
        partial_for_align = filter_env_rows(rows_from_metrics(args.partial), env_ids)
        align_ids = {r[0] for r in partial_for_align}
        partial_comparison_note = (
            "Partial comparison: gemma/gpt rows restricted to envIds with paired oak+random metrics in --partial "
            f"(n = {len(align_ids)} envs)."
        )
    elif partial_dir_ok and args.no_align_to_partial:
        partial_comparison_note = (
            "Unaligned (--no-align-to-partial): gemma/gpt use every task in benchmark.json; n can differ from partial."
        )

    def truth_filter(rows: list[tuple[str, float, float, float, str]]) -> list[tuple[str, float, float, float, str]]:
        out = filter_env_rows(rows, env_ids)
        if align_ids is not None:
            out = filter_env_rows(out, align_ids)
        return out

    runs: list[tuple[str, list[tuple[str, float, float, float, str]]]] = []

    if args.gemma.is_file():
        runs.append(("gemma4", truth_filter(rows_from_truth(args.gemma))))
    else:
        print(f"(skip) not found: {args.gemma}", flush=True)

    if args.gpt.is_file():
        runs.append(("gpt 5.4", truth_filter(rows_from_truth(args.gpt))))
    else:
        print(f"(skip) not found: {args.gpt}", flush=True)

    if args.partial is not None:
        mdir = args.partial
        if mdir.is_dir():
            label = args.partial_label or f"{mdir.parent.name}/metrics"
            runs.append((label, filter_env_rows(rows_from_metrics(mdir), env_ids)))
        else:
            print(f"(skip) not a directory: {mdir}", flush=True)

    if not runs:
        print("No inputs found. Provide --gemma/--gpt files and/or --partial METRICS_DIR.", flush=True)
        raise SystemExit(1)

    print_tables(runs, env_filter_note=env_note, partial_comparison_note=partial_comparison_note)


if __name__ == "__main__":
    main()
