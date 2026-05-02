#!/usr/bin/env python3
"""Gate an OaK candidate root against a keeper benchmark root.

Uniform 10-seed scoring (no privileged seeds). A candidate passes iff:

  1. cross-seed-min:    min(cand reward) >= min(base reward) - eps
  2. cross-seed-mean:   mean(cand reward) >= mean(base reward) - tol
                        (tol = max(--mean-abs, --mean-rel * |base mean|))
  3. late-stability:    for each seed with sufficient heartbeats, the rolling
                        25k-step reward rate over the last 50k steps must be
                        >= 0.5 * the peak rolling 25k rate over the run
  4. nf-ceiling:        per-phase cumulative non-finite count <= 2x base
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as stats
from pathlib import Path


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def metric(record: dict, name: str) -> float:
    if name == "rewardSum":
        value = record.get("rewardSum")
    elif name == "nonFiniteTotal":
        phases = record.get("health", {}).get("nonFiniteByPhase", {})
        value = sum(v for v in phases.values() if isinstance(v, (int, float))) if isinstance(phases, dict) else 0
    else:
        value = record.get("health", {}).get("envInfo", {}).get(name, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def nf_by_phase(record: dict) -> dict[str, int]:
    phases = record.get("health", {}).get("nonFiniteByPhase", {})
    if not isinstance(phases, dict):
        return {}
    return {k: int(v) for k, v in phases.items() if isinstance(v, (int, float))}


def collect_final(root: Path, policy: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in glob.glob(str(root / "**" / "*.json"), recursive=True):
        item = load(path)
        if not item or item.get("command") != "benchmark" or item.get("policy") != policy:
            continue
        if item.get("stopReason") != "measured_steps":
            continue
        seed = int(item.get("seed", -1))
        step = int(item.get("measuredStepCount", 0))
        if seed >= 0 and step >= int(out.get(seed, {}).get("measuredStepCount", -1)):
            out[seed] = item
    return out


def collect_heartbeats(root: Path, policy: str) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for path in glob.glob(str(root / "**" / "*.heartbeat-*.json"), recursive=True):
        item = load(path)
        if not item or item.get("policy") != policy:
            continue
        seed = int(item.get("seed", -1))
        step = int(item.get("measuredStepCount", 0))
        rsum = item.get("rewardSum", 0.0)
        if seed < 0 or not isinstance(rsum, (int, float)):
            continue
        out.setdefault(seed, []).append((step, float(rsum)))
    for seed in out:
        out[seed].sort()
    return out


def rolling_rate(series: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    """For each (step, rewardSum), find the closest earlier point >= window steps back
    and return (step, (rsum - rsum_back) / (step - step_back))."""
    out: list[tuple[int, float]] = []
    for i, (step, rsum) in enumerate(series):
        target = step - window
        j = i - 1
        while j >= 0 and series[j][0] > target:
            j -= 1
        if j < 0 or series[j][0] >= step:
            continue
        dt = step - series[j][0]
        out.append((step, (rsum - series[j][1]) / dt))
    return out


def fmt(x: float, precision: int = 3) -> str:
    text = f"{x:.{precision}f}"
    return f"{x:.2e}" if len(text) > 11 else text


def check_late_stability(
    base_hb: list[tuple[int, float]],
    cand_hb: list[tuple[int, float]],
    window: int,
    last_window: int,
    floor_ratio: float,
) -> tuple[str, float, float]:
    """Return (verdict, peak_rate, last_rate) for the candidate."""
    if not cand_hb or len(cand_hb) < 2:
        return ("no-data", 0.0, 0.0)
    rates = rolling_rate(cand_hb, window)
    if not rates:
        return ("no-data", 0.0, 0.0)
    peak = max(r for _, r in rates)
    final_step = cand_hb[-1][0]
    tail = [r for s, r in rates if s >= final_step - last_window]
    if not tail:
        return ("no-data", peak, 0.0)
    last = stats.mean(tail)
    if peak <= 0.0:
        return ("ok", peak, last)
    if last < floor_ratio * peak:
        return ("late-collapse", peak, last)
    return ("ok", peak, last)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", default="oak")
    parser.add_argument("--min-eps", type=float, default=10.0,
                        help="absolute slack on cross-seed-min check (default 10)")
    parser.add_argument("--mean-abs", type=float, default=10.0,
                        help="absolute slack on cross-seed-mean check (default 10)")
    parser.add_argument("--mean-rel", type=float, default=0.05,
                        help="relative slack on cross-seed-mean check (default 0.05 = 5%%)")
    parser.add_argument("--late-window", type=int, default=25000,
                        help="rolling reward-rate window size (default 25000)")
    parser.add_argument("--late-tail", type=int, default=50000,
                        help="how many trailing steps of rates to average for last-rate (default 50000)")
    parser.add_argument("--late-floor", type=float, default=0.5,
                        help="last-rate must be >= floor * peak-rate (default 0.5)")
    parser.add_argument("--nf-mult", type=float, default=2.0,
                        help="per-phase NF must be <= mult * base (default 2.0)")
    parser.add_argument("--allow-nf-regression", action="store_true")
    parser.add_argument("--allow-late-collapse", action="store_true",
                        help="skip late-stability check (use when keeper itself collapses)")
    args = parser.parse_args()

    base = collect_final(args.baseline, args.policy)
    cand = collect_final(args.candidate, args.policy)
    base_hb = collect_heartbeats(args.baseline, args.policy)
    cand_hb = collect_heartbeats(args.candidate, args.policy)

    seeds = sorted(set(base) & set(cand))
    if not seeds:
        raise SystemExit("no common final benchmark files found")

    failures: list[str] = []
    rows: list[str] = ["seed     baseR     candR        dR  bNF  cNF   peakR    lastR  late"]
    base_rewards: list[float] = []
    cand_rewards: list[float] = []
    for seed in seeds:
        br = metric(base[seed], "rewardSum")
        cr = metric(cand[seed], "rewardSum")
        bnf_by = nf_by_phase(base[seed])
        cnf_by = nf_by_phase(cand[seed])
        bnf = sum(bnf_by.values())
        cnf = sum(cnf_by.values())
        base_rewards.append(br)
        cand_rewards.append(cr)
        late_verdict, peak, last = check_late_stability(
            base_hb.get(seed, []),
            cand_hb.get(seed, []),
            args.late_window, args.late_tail, args.late_floor,
        )
        if not args.allow_nf_regression:
            for phase, cn in cnf_by.items():
                bn = bnf_by.get(phase, 0)
                ceiling = max(int(args.nf_mult * bn), 100)
                if cn > ceiling:
                    failures.append(f"seed {seed} NF[{phase}] {bn}->{cn} > {ceiling}")
        if not args.allow_late_collapse and late_verdict == "late-collapse":
            failures.append(
                f"seed {seed} late-collapse: last {fmt(last)} < {args.late_floor:.2f} * peak {fmt(peak)}")
        rows.append(
            f"{seed:4d} {fmt(br):>9} {fmt(cr):>9} {fmt(cr - br):>9}"
            f" {bnf:4d} {cnf:4d}  {fmt(peak):>6} {fmt(last):>6}  {late_verdict}")

    base_min, cand_min = min(base_rewards), min(cand_rewards)
    base_mean, cand_mean = stats.mean(base_rewards), stats.mean(cand_rewards)
    if cand_min < base_min - args.min_eps:
        failures.append(
            f"cross-seed-min regressed: base {fmt(base_min)} -> cand {fmt(cand_min)} (eps {args.min_eps})")
    mean_tol = max(args.mean_abs, args.mean_rel * abs(base_mean))
    if cand_mean < base_mean - mean_tol:
        failures.append(
            f"cross-seed-mean regressed: base {fmt(base_mean)} -> cand {fmt(cand_mean)} (tol {fmt(mean_tol)})")

    improved = sum(1 for b, c in zip(base_rewards, cand_rewards) if c > b)
    rows.append(
        f"  -- min {fmt(base_min):>5}->{fmt(cand_min):>5}  mean {fmt(base_mean):>5}->{fmt(cand_mean):>5}"
        f"  improved {improved}/{len(seeds)}")
    print("\n".join(rows))
    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
