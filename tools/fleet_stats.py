#!/usr/bin/env python3
"""Population statistics for OaK fleet runs.

Reads a fleet run root (meta.json, fleet.jsonl, deaths.jsonl, aggregate.jsonl)
and reports, per policy:
  - Kaplan-Meier survival (time-to-first-nonfinite) + RMST
  - divergence rate with Wilson 95% CI (overall and per detector)
  - IQM delivered reward rate (rewardSum/steps) with bootstrap 95% CI,
    over ALL lanes (headline) and survivors-only (diagnostic)
  - plasticity: per-lane zenith step/rate, decay ratio (end/zenith), quartiles
  - throughput: aggregate steps/s, total lane-steps

The statistic functions (wilson, km_curve, rmst, iqm, bootstrap_ci,
paired_bootstrap) are imported by fleet_gate.py — keep them dependency-light.

Usage: fleet_stats.py <run-root> [--policy oak] [--window STEPS] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ENTROPY_COLLAPSE_EPS = 1e-6


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion: (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimator. times: event-or-censor step per lane;
    events: 1 if death observed, 0 if censored (survived to horizon).
    Returns (event_times, S(t) just after each event time)."""
    order = np.argsort(times)
    t_sorted, e_sorted = times[order], events[order]
    uniq = np.unique(t_sorted[e_sorted == 1])
    n = len(times)
    surv, s, at_risk_idx = [], 1.0, 0
    for t in uniq:
        at_risk = n - np.searchsorted(t_sorted, t, side="left")
        deaths = int(((t_sorted == t) & (e_sorted == 1)).sum())
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
        surv.append(s)
    return uniq, np.array(surv)


def rmst(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    """Restricted mean survival time: area under KM curve up to horizon."""
    ts, surv = km_curve(times, events)
    area, prev_t, prev_s = 0.0, 0.0, 1.0
    for t, s in zip(ts, surv):
        t = min(float(t), float(horizon))
        area += prev_s * (t - prev_t)
        prev_t, prev_s = t, s
        if prev_t >= horizon:
            break
    if prev_t < horizon:
        area += prev_s * (horizon - prev_t)
    return area


def iqm(values: np.ndarray) -> float:
    """Interquartile mean (mean of middle 50%, rliable-style)."""
    if len(values) == 0:
        return float("nan")
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    lo, hi = n // 4, n - n // 4
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())


def bootstrap_ci(
    values: np.ndarray, stat, n_boot: int = 10_000, seed: int = 0,
    lo_q: float = 2.5, hi_q: float = 97.5,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    draws = np.array([stat(v[row]) for row in idx])
    return float(np.percentile(draws, lo_q)), float(np.percentile(draws, hi_q))


def paired_bootstrap(
    keeper: np.ndarray, candidate: np.ndarray, stat, n_boot: int = 10_000, seed: int = 0
) -> np.ndarray:
    """Bootstrap distribution of stat(candidate*) - stat(keeper*) where both
    sides are resampled with the SAME seed indices (paired by lane)."""
    assert len(keeper) == len(candidate)
    rng = np.random.default_rng(seed)
    n = len(keeper)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.array([stat(candidate[row]) - stat(keeper[row]) for row in idx])


def rolling_rate(series: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    """For each (step, rewardSum), find the closest earlier point >= window steps
    back and return (step, (rsum - rsum_back) / (step - step_back))."""
    out: list[tuple[int, float]] = []
    for i, (step, rsum) in enumerate(series):
        target = step - window
        j = i - 1
        while j >= 0 and series[j][0] > target:
            j -= 1
        if j < 0 or series[j][0] >= step:
            continue
        out.append((step, (rsum - series[j][1]) / (step - series[j][0])))
    return out


# ------------------------------------------------------------------- loading


def load_run(root: Path) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    lanes: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with open(root / "fleet.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lanes[(row["policy"], row["seed"])].append(row)
    for rows in lanes.values():
        rows.sort(key=lambda r: r["step"])
    deaths = []
    deaths_path = root / "deaths.jsonl"
    if deaths_path.exists():
        with open(deaths_path) as f:
            deaths = [json.loads(line) for line in f]
    return {"meta": meta, "lanes": lanes, "deaths": deaths}


def finals(run: dict, policy: str) -> dict[int, dict]:
    out = {}
    for (pol, seed), rows in run["lanes"].items():
        if pol != policy:
            continue
        fin = [r for r in rows if r["kind"] == "final"]
        out[seed] = fin[-1] if fin else rows[-1]
    return out


def death_times(fin: dict[int, dict], horizon: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(times, events, seeds) for time-to-first-nonfinite, censored at horizon."""
    seeds = sorted(fin)
    times = np.array(
        [fin[s].get("firstNonFiniteStep") or horizon for s in seeds], dtype=float)
    events = np.array(
        [1 if fin[s].get("firstNonFiniteStep") is not None else 0 for s in seeds])
    return times, events, seeds


def entropy_collapse_step(rows: list[dict]) -> int | None:
    """First heartbeat step where actorSupportEntropy < eps for 2 consecutive
    heartbeats (the FC-STOMP-goes-quiet fixed point)."""
    prev_low, prev_step = False, None
    for r in rows:
        ent = (r.get("health") or {}).get("actorSupportEntropy")
        low = ent is not None and ent < ENTROPY_COLLAPSE_EPS
        if low and prev_low:
            return prev_step
        prev_low, prev_step = low, r["step"]
    return None


def plasticity(rows: list[dict], window: int) -> dict | None:
    """Zenith and decay statistics from the heartbeat rewardSum series."""
    series = [(r["step"], r["rewardSum"]) for r in rows]
    rates = rolling_rate(series, window)
    if not rates:
        return None
    zenith_step, zenith_rate = max(rates, key=lambda t: t[1])
    end_step, end_rate = rates[-1]
    decay = end_rate / zenith_rate if zenith_rate > 0 else float("nan")
    return {
        "zenithStep": zenith_step, "zenithRate": zenith_rate,
        "endRate": end_rate, "decayRatio": decay,
    }


# ----------------------------------------------------------------- reporting


def quartiles(v: list[float]) -> str:
    if not v:
        return "n/a"
    q = np.percentile(np.array(v, dtype=float), [25, 50, 75])
    return f"{q[0]:.4g} / {q[1]:.4g} / {q[2]:.4g}"


def analyze_policy(run: dict, policy: str, window: int) -> dict:
    meta = run["meta"]
    horizon = meta["steps"]
    fin = finals(run, policy)
    n = len(fin)
    if n == 0:
        return {}
    seeds = sorted(fin)
    delivered = np.array(
        [fin[s]["rewardSum"] / max(fin[s]["step"], 1) for s in seeds])
    times, events, _ = death_times(fin, horizon)
    dead = int(events.sum())
    p, lo, hi = wilson(dead, n)
    by_detector: dict[str, int] = defaultdict(int)
    for s in seeds:
        det = fin[s].get("firstNonFiniteDetector")
        if det:
            by_detector[det] += 1
    surv_mask = events == 0
    out: dict = {
        "policy": policy, "lanes": n, "horizon": horizon,
        "divergence": {"dead": dead, "rate": p, "wilsonLo": lo, "wilsonHi": hi,
                       "byDetector": dict(sorted(by_detector.items()))},
        "rmst": rmst(times, events, horizon),
        "iqmDeliveredAll": iqm(delivered),
        "iqmDeliveredSurvivors": iqm(delivered[surv_mask]) if surv_mask.any() else None,
    }
    if n >= 4:
        ci = bootstrap_ci(delivered, iqm, seed=1)
        out["iqmDeliveredAllCI95"] = list(ci)
    ts, surv = km_curve(times, events)
    out["survival"] = {
        str(int(t)): float(s) for t, s in zip(ts, surv)
    }
    if policy == "oak":
        plast, collapses = [], 0
        for s in seeds:
            rows = run["lanes"][(policy, s)]
            pl = plasticity(rows, window)
            if pl:
                plast.append(pl)
            if entropy_collapse_step(rows) is not None:
                collapses += 1
        if plast:
            out["plasticity"] = {
                "zenithStepQ": quartiles([p_["zenithStep"] for p_ in plast]),
                "zenithRateQ": quartiles([p_["zenithRate"] for p_ in plast]),
                "decayRatioQ": quartiles(
                    [p_["decayRatio"] for p_ in plast if math.isfinite(p_["decayRatio"])]),
            }
        cp, clo, chi = wilson(collapses, n)
        out["entropyCollapse"] = {"count": collapses, "rate": cp,
                                  "wilsonLo": clo, "wilsonHi": chi}
    # throughput from the lane rows (per-lane sps is lane-steps/wall; the run is
    # lockstep so aggregate = B * per-lane)
    sps = [fin[s].get("stepsPerSecond") for s in seeds if fin[s].get("stepsPerSecond")]
    if sps:
        out["aggStepsPerSecond"] = float(np.median(sps)) * n
        out["laneStepsTotal"] = int(sum(fin[s]["step"] for s in seeds))
    return out


def render_markdown(results: list[dict], meta: dict) -> str:
    lines = [
        f"### Fleet report — {meta['envId']} B={meta['batch']} × {meta['steps']} steps "
        f"(ablation={meta.get('ablationLabel', 'none')})",
        "",
        "| policy | lanes | diverged (Wilson 95%) | RMST/horizon | IQM delivered (all) | IQM (survivors) | agg steps/s |",
        "|---|---:|---|---:|---|---|---:|",
    ]
    for r in results:
        if not r:
            continue
        d = r["divergence"]
        ci = r.get("iqmDeliveredAllCI95")
        ci_s = f" [{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
        surv = r.get("iqmDeliveredSurvivors")
        lines.append(
            f"| {r['policy']} | {r['lanes']} | {d['dead']}/{r['lanes']} = {d['rate']:.1%} "
            f"[{d['wilsonLo']:.1%}, {d['wilsonHi']:.1%}] | {r['rmst'] / r['horizon']:.3f} "
            f"| {r['iqmDeliveredAll']:.5g}{ci_s} | "
            f"{'n/a' if surv is None else f'{surv:.5g}'} | "
            f"{r.get('aggStepsPerSecond', 0):,.0f} |")
    for r in results:
        if r and r.get("plasticity"):
            p_ = r["plasticity"]
            ec = r["entropyCollapse"]
            lines += [
                "",
                f"**{r['policy']} plasticity** (quartiles 25/50/75): zenith step {p_['zenithStepQ']}; "
                f"zenith rate {p_['zenithRateQ']}; decay ratio {p_['decayRatioQ']}. "
                f"Entropy collapse: {ec['count']}/{r['lanes']} = {ec['rate']:.1%} "
                f"[{ec['wilsonLo']:.1%}, {ec['wilsonHi']:.1%}].",
            ]
        if r and r["divergence"]["byDetector"]:
            dets = ", ".join(f"{k}: {v}" for k, v in r["divergence"]["byDetector"].items())
            lines.append(f"**{r['policy']} death detectors**: {dets}.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--policy", default=None, help="restrict to one policy")
    ap.add_argument("--window", type=int, default=None,
                    help="rolling-rate window for plasticity (default: 2 heartbeats)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write summary.json here (default: <root>/summary.json)")
    args = ap.parse_args()
    run = load_run(args.root)
    meta = run["meta"]
    window = args.window or max(meta["steps"] // 5, 1)
    policies = [args.policy] if args.policy else meta.get("policies", ["oak"])
    results = [analyze_policy(run, p, window) for p in policies]
    print(render_markdown(results, meta))
    summary = {"meta": meta, "results": [r for r in results if r]}
    out = args.json or (args.root / "summary.json")
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
