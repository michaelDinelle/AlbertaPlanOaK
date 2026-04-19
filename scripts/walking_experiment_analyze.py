#!/usr/bin/env python3
"""Analyze the walking-experiment principled measurement.

Reads heartbeats + final from .oak-benchmarks/walking-experiment/seed{0,1,2}-{oak,random}/
and emits:
  1. Per-policy trajectory: rR vs step (mean ± std across seeds)
  2. Per-seed trajectories
  3. Final-step paired comparison (oak rR − random rR per seed) with t-test
  4. Numerical stability check (NF, weight maxes)
  5. Learning evidence: monotonic improvement check on oak's rR trajectory
"""
import json
import glob
import os
import sys
import math
from pathlib import Path

ROOT = Path("/Users/studio/OaKMeta/.oak-benchmarks/walking-experiment-1")
SEEDS = [0, 1, 2]
POLICIES = ["oak", "random"]

def load_heartbeats(seed: int, policy: str):
    """Returns sorted list of (step, rewardSum, health-dict) tuples."""
    pattern = ROOT / f"seed{seed}-{policy}" / "**" / "*.json"
    files = sorted(glob.glob(str(pattern), recursive=True))
    out = []
    for f in files:
        if "heartbeat" in os.path.basename(f).lower() or os.path.basename(f).startswith("heartbeat-"):
            d = json.load(open(f))
            n  = d.get("measuredStepCount", 0)
            rs = d.get("rewardSum", 0.0)
            h  = d.get("health", {}) or {}
            out.append((n, rs, h))
        else:
            # final snapshot
            d = json.load(open(f))
            n  = d.get("measuredStepCount", 0)
            rs = d.get("rewardSum", 0.0)
            h  = d.get("health", {}) or {}
            out.append((n, rs, h))
    out.sort(key=lambda t: t[0])
    return out

def mean_std(xs):
    if not xs: return (float('nan'), float('nan'))
    n = len(xs)
    m = sum(xs) / n
    if n < 2: return (m, 0.0)
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (m, math.sqrt(v))

def paired_t_test(diffs):
    """One-sample t-test on diffs (oak - random per seed) vs 0."""
    n = len(diffs)
    if n < 2: return (float('nan'), float('nan'))
    m, s = mean_std(diffs)
    se = s / math.sqrt(n)
    if se == 0: return (float('inf') if m != 0 else 0.0, 0.0)
    t = m / se
    # rough p-value approximation (two-sided, df=n-1)
    return (t, m)

def check_monotonic_improvement(trajectory):
    """Check if rR per-window is improving over the trajectory.
    Splits trajectory into halves, compares mean rR.
    Returns (early_mean, late_mean, late > early)."""
    if len(trajectory) < 4: return (float('nan'), float('nan'), False)
    # Compute per-window inst_rR from rewardSum deltas
    inst_rRs = []
    prev_step, prev_rs = 0, 0.0
    for (n, rs, _) in trajectory:
        dn = n - prev_step
        if dn > 0:
            inst_rRs.append((rs - prev_rs) / dn)
        prev_step, prev_rs = n, rs
    half = len(inst_rRs) // 2
    early = inst_rRs[:half]
    late  = inst_rRs[half:]
    em = sum(early) / max(len(early), 1)
    lm = sum(late) / max(len(late), 1)
    return (em, lm, lm > em)

def main():
    print(f"=== HalfCheetah-v4 walking experiment: 3 seeds × 1M steps × {{oak, random}} ===\n")

    by_policy = {p: {s: load_heartbeats(s, p) for s in SEEDS} for p in POLICIES}

    # Per-seed final values
    print("Per-seed FINAL values (1M steps):")
    print(f"  {'seed':>4s} | {'policy':>7s} | {'rewardSum':>10s} | {'rR':>8s} | {'NF':>6s} | {'modErr':>10s} | {'pEff':>5s} | {'aΔ':>9s}")
    finals = {p: {} for p in POLICIES}
    for p in POLICIES:
        for s in SEEDS:
            traj = by_policy[p][s]
            if not traj:
                print(f"  {s:>4d} | {p:>7s} | (no data yet)")
                continue
            n, rs, h = traj[-1]
            rR = rs / max(n, 1)
            nf = int(h.get('nonFiniteRawTotal', 0) or 0)
            modErr = h.get('modelErrEMA', 0.0) or 0.0
            pEff = h.get('pEffectiveRank', 0.0) or 0.0
            aD = h.get('actionDeltaL2Mean', 0.0) or 0.0
            print(f"  {s:>4d} | {p:>7s} | {rs:>10.0f} | {rR:>+8.4f} | {nf:>6d} | {modErr:>10.3g} | {pEff:>5.2f} | {aD:>9.2e}")
            finals[p][s] = rR

    # Mean ± std across seeds
    print("\nMean ± std across seeds (final rR):")
    for p in POLICIES:
        rRs = list(finals[p].values())
        m, s = mean_std(rRs)
        print(f"  {p:>7s}:  rR = {m:+.4f} ± {s:.4f}  (n={len(rRs)})")

    # Paired comparison
    print("\nPaired oak−random per seed (positive = oak better):")
    diffs = []
    for s in SEEDS:
        if s in finals.get('oak', {}) and s in finals.get('random', {}):
            d = finals['oak'][s] - finals['random'][s]
            diffs.append(d)
            print(f"  seed {s}:  Δ = {d:+.4f}")
    t, m = paired_t_test(diffs)
    print(f"  paired mean Δ = {m:+.4f},  t-stat = {t:+.3f}  (df = {len(diffs)-1})")

    # Monotonic improvement check
    print("\nMonotonic learning check (mean inst_rR over last vs first half of trajectory):")
    for p in POLICIES:
        print(f"  {p}:")
        for s in SEEDS:
            traj = by_policy[p][s]
            em, lm, improving = check_monotonic_improvement(traj)
            symbol = "↗" if improving else "·"
            delta = lm - em
            print(f"    seed {s}: early {em:+.4f} → late {lm:+.4f}  Δ={delta:+.4f}  {symbol}")

    # Trajectory dump for one seed
    print("\nSeed-0 oak trajectory (every heartbeat):")
    print(f"  {'step':>8s} | {'rewSum':>10s} | {'inst_rR':>9s} | {'modErr':>10s} | {'pEff':>5s} | {'encMx':>8s} | {'valMx':>8s} | {'NF':>4s}")
    if 0 in by_policy['oak']:
        traj = by_policy['oak'][0]
        prev_step, prev_rs = 0, 0.0
        for (n, rs, h) in traj:
            dn = n - prev_step
            inst = (rs - prev_rs) / max(dn, 1)
            print(f"  {n:>8d} | {rs:>10.0f} | {inst:>+9.4f} | "
                  f"{(h.get('modelErrEMA',0) or 0):>10.3g} | "
                  f"{(h.get('pEffectiveRank',0) or 0):>5.2f} | "
                  f"{(h.get('encoderMax',0) or 0):>8.3g} | "
                  f"{(h.get('valueWMax',0) or 0):>8.3g} | "
                  f"{int(h.get('nonFiniteRawTotal',0) or 0):>4d}")
            prev_step, prev_rs = n, rs

if __name__ == "__main__":
    main()
