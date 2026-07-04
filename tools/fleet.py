#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ENTROPY_COLLAPSE_EPS = 1e-6
NF_PHASES = ("encode", "learn", "plan", "tdPsi")

SOTA_EVAL = {"TD7": 17.4, "TD-MPC2": 15.1, "MR.Q": 13.3, "TD3": 10.1, "SAC": 9.9}
SOTA_DELIVERED_EST = {"SAC (delivered, est.)": 6.0}
SOTA_FRAMING_NOTE = (
    "eval = final-policy evaluation on free 1000-step episodes (published); "
    "delivered = online reward over the entire stream, flailing included (ours). "
    "Per-FLOP claims require both sides instrumented (AGENTS.md rules).")

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)

def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    t_sorted, e_sorted = times[order], events[order]
    uniq = np.unique(t_sorted[e_sorted == 1])
    n = len(times)
    surv, s = [], 1.0
    for t in uniq:
        at_risk = n - np.searchsorted(t_sorted, t, side="left")
        deaths = int(((t_sorted == t) & (e_sorted == 1)).sum())
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
        surv.append(s)
    return uniq, np.array(surv)

def rmst(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
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
    assert len(keeper) == len(candidate)
    rng = np.random.default_rng(seed)
    n = len(keeper)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.array([stat(candidate[row]) - stat(keeper[row]) for row in idx])

def unpaired_bootstrap(keeper, candidate, stat, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    k, c = np.asarray(keeper, dtype=float), np.asarray(candidate, dtype=float)
    ik = rng.integers(0, len(k), size=(n_boot, len(k)))
    ic = rng.integers(0, len(c), size=(n_boot, len(c)))
    return np.array([stat(c[rc]) - stat(k[rk]) for rk, rc in zip(ik, ic)])

def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    if lam < 0.3:
        return d, 1.0
    j = np.arange(1, 101)
    p = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * (lam * j) ** 2))
    return d, float(min(max(p, 0.0), 1.0))

def rolling_rate(series: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
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
    seeds = sorted(fin)
    times = np.array(
        [fin[s].get("firstNonFiniteStep") or horizon for s in seeds], dtype=float)
    events = np.array(
        [1 if fin[s].get("firstNonFiniteStep") is not None else 0 for s in seeds])
    return times, events, seeds

def delivered(fin: dict[int, dict], seeds: list[int]) -> np.ndarray:
    return np.array([fin[s]["rewardSum"] / max(fin[s]["step"], 1) for s in seeds])

def recent_rates(run: dict, policy: str, fin: dict[int, dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for s in sorted(fin):
        rows = run["lanes"][(policy, s)]
        if len(rows) >= 2 and rows[-1]["step"] > rows[-2]["step"]:
            out[s] = (rows[-1]["rewardSum"] - rows[-2]["rewardSum"]) / (
                rows[-1]["step"] - rows[-2]["step"])
        else:
            out[s] = fin[s]["rewardSum"] / max(fin[s]["step"], 1)
    return out

def sota_gap_line(best_rate: float, iqm_all: float) -> str:
    td7 = SOTA_EVAL["TD7"]
    sac_del = SOTA_DELIVERED_EST["SAC (delivered, est.)"]
    def gap(x, ref):
        if x <= 0:
            return "n/a (rate <= 0)"
        return f"{ref / x:,.0f}x"
    return (f"**SOTA gap** — best lane {best_rate:+.3f}/step, population IQM {iqm_all:+.4f}/step. "
            f"vs TD7 {td7}/step (eval): gap {gap(best_rate, td7)} at best lane. "
            f"vs SAC ~{sac_del}/step (delivered, est.): gap {gap(best_rate, sac_del)} at best lane. "
            f"({SOTA_FRAMING_NOTE})")

def entropy_collapse_step(rows: list[dict]) -> int | None:
    prev_low, prev_step = False, None
    for r in rows:
        ent = (r.get("health") or {}).get("actorSupportEntropy")
        low = ent is not None and ent < ENTROPY_COLLAPSE_EPS
        if low and prev_low:
            return prev_step
        prev_low, prev_step = low, r["step"]
    return None

def plasticity(rows: list[dict], window: int) -> dict | None:
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
    rates = delivered(fin, seeds)
    times, events, _ = death_times(fin, horizon)
    dead = int(events.sum())
    p, lo, hi = wilson(dead, n)
    by_detector: dict[str, int] = defaultdict(int)
    for s in seeds:
        det = fin[s].get("firstNonFiniteDetector")
        if det:
            by_detector[det] += 1
    surv_mask = events == 0
    recents = recent_rates(run, policy, fin)
    recent_arr = np.array([recents[s] for s in seeds])
    best_idx = int(np.argmax(rates))
    out: dict = {
        "policy": policy, "lanes": n, "horizon": horizon,
        "divergence": {"dead": dead, "rate": p, "wilsonLo": lo, "wilsonHi": hi,
                       "byDetector": dict(sorted(by_detector.items()))},
        "rmst": rmst(times, events, horizon),
        "iqmDeliveredAll": iqm(rates),
        "iqmDeliveredSurvivors": iqm(rates[surv_mask]) if surv_mask.any() else None,
        "bestLaneSeed": seeds[best_idx],
        "bestLaneRate": float(rates[best_idx]),
        "bestLaneRewardSum": float(fin[seeds[best_idx]]["rewardSum"]),
        "positiveLanes": int((rates > 0).sum()),
        "positiveFraction": float((rates > 0).mean()),
        "iqmRecent": iqm(recent_arr),
        "recentMax": float(recent_arr.max()),
        "recentRates": recents,
        "deliveredBySeed": {s: float(r) for s, r in zip(seeds, rates)},
    }
    # Completeness: a run is valid only if it reached its declared horizon. A silently-killed run
    # (laptop sleep, OOM, harness stop) leaves lanes short of horizon — flag it loudly so no one
    # gates or reports a partial run as if it finished.
    max_step = max(int(f["step"]) for f in fin.values())
    has_final = any(r["kind"] == "final" for s in seeds for r in run["lanes"][(policy, s)])
    out["complete"] = bool(has_final and max_step >= horizon)
    out["maxStep"] = max_step
    # Population progress toward the goal (the WHOLE population at SOTA, staying there forever).
    # Report the distribution and the fraction of lanes clearing each reward-rate tier — never a
    # single hand-picked lane. Tiers: >0 walking at all, >1 the milestone, >5 ~SAC-delivered, >10 SOTA.
    _pct = np.percentile(rates, [10, 25, 50, 75, 90, 99])
    out["rateDistribution"] = {k: float(v) for k, v in
                               zip(("p10", "p25", "p50", "p75", "p90", "p99"), _pct)}
    out["tierFractions"] = {"gt0": float((rates > 0).mean()), "gt1": float((rates > 1).mean()),
                            "gt5": float((rates > 5).mean()), "gt10": float((rates > 10).mean())}
    flops = run["meta"].get("flopsPerLaneStep")
    if flops:
        out["rewardPerGFLOP"] = out["iqmDeliveredAll"] / (flops / 1e9)
        out["bestRewardPerGFLOP"] = out["bestLaneRate"] / (flops / 1e9)
    if n >= 4:
        out["iqmDeliveredAllCI95"] = list(bootstrap_ci(rates, iqm, seed=1))
    ts, surv = km_curve(times, events)
    out["survival"] = {str(int(t)): float(s) for t, s in zip(ts, surv)}
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
            # Ignition metrics (Mark III audit law: Tier-M IQM alone is blind to
            # walking-mode kills — the tail is a first-class gate axis).
            # igniter := lane whose peak windowed rate exceeded 0.1/step;
            # sustained := igniter still holding >= half its zenith at the end.
            _ign = [p_ for p_ in plast if p_["zenithRate"] > 0.1]
            _sus = [p_ for p_ in _ign
                    if math.isfinite(p_["decayRatio"]) and p_["decayRatio"] >= 0.5]
            out["igniters"] = len(_ign)
            out["ignitersSustained"] = len(_sus)
            out["igniterSustainFraction"] = (len(_sus) / len(_ign)) if _ign else None
            # "Staying power": fraction of lanes retaining >= half their zenith rate at the end
            # (the opposite of post-zenith decay/crash). A continuous online agent must STAY at rate.
            _dec = [p_["decayRatio"] for p_ in plast if math.isfinite(p_["decayRatio"])]
            out["sustainingFraction"] = (
                float(np.mean([d >= 0.5 for d in _dec])) if _dec else None)
        cp, clo, chi = wilson(collapses, n)
        out["entropyCollapse"] = {"count": collapses, "rate": cp,
                                  "wilsonLo": clo, "wilsonHi": chi}
    sps = [fin[s].get("stepsPerSecond") for s in seeds if fin[s].get("stepsPerSecond")]
    if sps:
        out["aggStepsPerSecond"] = float(np.median(sps)) * n
        out["laneStepsTotal"] = int(sum(fin[s]["step"] for s in seeds))
    return out

def render_markdown(results: list[dict], meta: dict) -> str:
    banner = []
    for r in results:
        if r and not r.get("complete", True):
            banner += [
                f"> ⚠️ **INCOMPLETE RUN — {r['policy']} reached step {r.get('maxStep', 0):,} of "
                f"{meta['steps']:,} ({r.get('maxStep', 0) / max(meta['steps'], 1):.0%}); it was "
                f"silently killed (sleep / OOM / harness stop). Do NOT gate or trust as final.**", ""]
    lines = [
        f"### Fleet report — {meta['envId']} B={meta['batch']} × {meta['steps']} steps "
        f"(ablation={meta.get('ablationLabel', 'none')})",
        "",
        *banner,
        "**Delivered reward (the headline)**",
        "",
        "| policy | IQM rate (all lanes) | best lane | positive lanes | recent-window IQM | recent max"
        + (" | reward/GFLOP |" if any(r.get("rewardPerGFLOP") is not None for r in results if r) else " |"),
        "|---|---|---|---:|---|---"
        + ("|---|" if any(r.get("rewardPerGFLOP") is not None for r in results if r) else "|"),
    ]
    has_flop = any(r.get("rewardPerGFLOP") is not None for r in results if r)
    for r in results:
        if not r:
            continue
        ci = r.get("iqmDeliveredAllCI95")
        ci_s = f" [{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
        row = (f"| {r['policy']} | **{r['iqmDeliveredAll']:.5g}**{ci_s} "
               f"| seed {r['bestLaneSeed']}: {r['bestLaneRate']:+.4f}/step "
               f"({r['bestLaneRewardSum']:+,.0f} total) "
               f"| {r['positiveLanes']}/{r['lanes']} = {r['positiveFraction']:.1%} "
               f"| {r['iqmRecent']:+.5g} | {r['recentMax']:+.4f} ")
        if has_flop:
            rg = r.get("rewardPerGFLOP")
            row += f"| {rg:+.4g} |" if rg is not None else "| n/a |"
        else:
            row += "|"
        lines.append(row)
    oak = next((r for r in results if r and r["policy"] == "oak"), None)
    if oak:
        lines += ["", sota_gap_line(oak["bestLaneRate"], oak["iqmDeliveredAll"])]
    lines += [
        "",
        "**Population progress (the whole population is the deliverable — never a single lane)**",
        "",
        "| policy | complete | rate P10/P50/P90 | %>0 | %>1 | %>5 | %>SOTA(10) | sustaining |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if not r:
            continue
        rd, tf = r.get("rateDistribution", {}), r.get("tierFractions", {})
        sust = r.get("sustainingFraction")
        comp = ("yes" if r.get("complete", True)
                else f"**NO {r.get('maxStep', 0) / max(meta['steps'], 1):.0%}**")
        lines.append(
            f"| {r['policy']} | {comp} "
            f"| {rd.get('p10', 0):+.4f} / {rd.get('p50', 0):+.4f} / {rd.get('p90', 0):+.4f} "
            f"| {tf.get('gt0', 0):.1%} | {tf.get('gt1', 0):.1%} | {tf.get('gt5', 0):.1%} "
            f"| {tf.get('gt10', 0):.1%} | {'n/a' if sust is None else f'{sust:.1%}'} |")
    lines += [
        "",
        "**Survival & robustness**",
        "",
        "| policy | lanes | diverged (Wilson 95%) | RMST/horizon | IQM (survivors) |",
        "|---|---:|---|---:|---|",
    ]
    for r in results:
        if not r:
            continue
        d = r["divergence"]
        surv = r.get("iqmDeliveredSurvivors")
        lines.append(
            f"| {r['policy']} | {r['lanes']} | {d['dead']}/{r['lanes']} = {d['rate']:.1%} "
            f"[{d['wilsonLo']:.1%}, {d['wilsonHi']:.1%}] | {r['rmst'] / r['horizon']:.3f} "
            f"| {'n/a' if surv is None else f'{surv:.5g}'} |")
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
        if r:
            lines.append(
                f"**{r['policy']} throughput**: {r.get('aggStepsPerSecond', 0):,.0f} agg steps/s"
                + (f"; {run_flops_str(r)}" if r.get("rewardPerGFLOP") is not None else "") + ".")
    return "\n".join(lines)

def run_flops_str(r: dict) -> str:
    return f"best-lane reward/GFLOP {r['bestRewardPerGFLOP']:+.4g} (OaK measured; SOTA est. only)"

def render_leaderboard(run: dict, result: dict, top: int) -> str:
    policy = result["policy"]
    fin = finals(run, policy)
    delivered = result["deliveredBySeed"]
    recents = result["recentRates"]
    order = sorted(delivered, key=lambda s: delivered[s], reverse=True)[:top]
    lines = [
        "", f"**{policy} leaderboard — top {len(order)} lanes by delivered rate**", "",
        "| seed | rate/step | total reward | recent rate | zenith step | status |",
        "|---:|---|---|---|---:|---|",
    ]
    window = max(result["horizon"] // 5, 1)
    for s in order:
        rows = run["lanes"][(policy, s)]
        pl = plasticity(rows, window)
        zen = f"{pl['zenithStep']:,}" if pl else "n/a"
        rec, life = recents[s], delivered[s]
        nf = fin[s].get("firstNonFiniteStep") is not None
        if rec > max(life, 0) * 1.1 and rec > 0:
            status = "climbing"
        elif pl and pl["zenithRate"] > 0 and rec < 0.5 * pl["zenithRate"]:
            status = "decaying"
        else:
            status = "steady"
        if nf:
            status += " (NF)"
        lines.append(
            f"| {s} | {life:+.4f} | {fin[s]['rewardSum']:+,.0f} | {rec:+.4f} | {zen} | {status} |")
    return "\n".join(lines)

def cmd_report(args) -> int:
    run = load_run(args.root)
    meta = run["meta"]
    window = args.window or max(meta["steps"] // 5, 1)
    policies = [args.policy] if args.policy else meta.get("policies", ["oak"])
    results = [analyze_policy(run, p, window) for p in policies]
    print(render_markdown(results, meta))
    if args.leaderboard:
        for r in results:
            if r:
                print(render_leaderboard(run, r, args.top))
    summary = {"meta": meta, "results": [
        {k: v for k, v in r.items() if k not in ("recentRates", "deliveredBySeed")}
        for r in results if r]}
    out = args.json or (args.root / "summary.json")
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0

def gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, label,
                  keep_run=None, cand_run=None):
    delta = args.delta if args.delta is not None else 0.64 / np.sqrt(n)
    verdicts: list[tuple[str, bool, str]] = []

    k_times, k_events, _ = death_times(keep_fin, horizon)
    c_times, c_events, _ = death_times(cand_fin, horizon)
    dp = paired_bootstrap(k_events.astype(float), c_events.astype(float),
                          np.mean, n_boot=args.boot, seed=11)
    dp95 = float(np.percentile(dp, 95))
    p_keep, p_cand = k_events.mean(), c_events.mean()
    verdicts.append((
        "divergence", dp95 <= delta,
        f"p keeper={p_keep:.3f} cand={p_cand:.3f} Δp95={dp95:+.4f} δ={delta:.4f}"))

    k_del, c_del = delivered(keep_fin, keep_seeds), delivered(cand_fin, cand_seeds)
    di = paired_bootstrap(k_del, c_del, iqm, n_boot=args.boot, seed=12)
    di5 = float(np.percentile(di, 5))
    iqm_keep = iqm(k_del)
    margin = args.rate_rel * abs(iqm_keep)
    improved = di5 > 0
    verdicts.append((
        "iqmDelivered", di5 >= -margin,
        f"IQM keeper={iqm_keep:.5g} cand={iqm(c_del):.5g} ΔIQM5={di5:+.5g} "
        f"margin={-margin:.5g}{' IMPROVEMENT' if improved else ''}"))

    rng = np.random.default_rng(13)
    idx = rng.integers(0, n, size=(args.boot, n))
    dr = np.array([
        rmst(c_times[row], c_events[row], horizon) - rmst(k_times[row], k_events[row], horizon)
        for row in idx
    ])
    dr5 = float(np.percentile(dr, 5))
    rmst_margin = args.rmst_rel * horizon
    verdicts.append((
        "rmstSurvival", dr5 >= -rmst_margin,
        f"RMST keeper={rmst(k_times, k_events, horizon):,.0f} "
        f"cand={rmst(c_times, c_events, horizon):,.0f} ΔRMST5={dr5:+,.0f} "
        f"margin={-rmst_margin:,.0f}"))

    # NF criterion: per-lane PREVALENCE non-inferiority per phase (the fraction of lanes with any
    # non-finite in that phase), mirroring the divergence criterion (paired bootstrap, Δ95 ≤ δ) and
    # consistent with the equivalence gate's nf_prevalence. Replaces a fleet-sum-of-totals test, which
    # a few already-diverged lanes inflate by spraying non-finites after death — double-counting lanes
    # the divergence criterion already counts. Population NF health is how MANY lanes go non-finite.
    def nf_prevalence_vec(fin, seeds, phase):
        return np.array([
            1.0 if ((fin[s].get("health") or {}).get("nonFiniteByPhase") or {}).get(phase, 0) > 0
            else 0.0
            for s in seeds])
    nf_over, nf_parts = [], []
    for i, ph in enumerate(NF_PHASES):
        kp = nf_prevalence_vec(keep_fin, keep_seeds, ph)
        cp = nf_prevalence_vec(cand_fin, cand_seeds, ph)
        dnf95 = float(np.percentile(
            paired_bootstrap(kp, cp, np.mean, n_boot=args.boot, seed=20 + i), 95))
        if dnf95 > delta:
            nf_over.append(ph)
        nf_parts.append(f"{ph} {kp.mean():.3f}->{cp.mean():.3f} Δ95={dnf95:+.4f}")
    verdicts.append((
        "nfPrevalence", not nf_over,
        "; ".join(nf_parts) + (f" OVER={nf_over}" if nf_over else "") + f" δ={delta:.4f}"))

    # Ignition / walking-mode tail (Mark III audit law: Tier-M IQM alone approved a
    # mode-kill — the tail is a first-class gate axis). Always reported; BLOCKING
    # only when explicit bars are passed (--min-igniters/--min-best/--min-positive).
    def _ign(run_, seeds_):
        run_lanes = run_["lanes"]
        pol = args.policy
        window = max(horizon // 5, 1)
        ign = sus = 0
        for s_ in seeds_:
            pl = plasticity(run_lanes[(pol, s_)], window)
            if pl and pl["zenithRate"] > 0.1:
                ign += 1
                if math.isfinite(pl["decayRatio"]) and pl["decayRatio"] >= 0.5:
                    sus += 1
        return ign, sus
    try:
        if keep_run is None or cand_run is None:
            raise ValueError("run data not provided")
        k_ign, k_sus = _ign(keep_run, keep_seeds)
        c_ign, c_sus = _ign(cand_run, cand_seeds)
        c_best = float(np.max(c_del))
        c_pos = float((c_del > 0).mean())
        bars_given = any(v is not None for v in
                         (args.min_igniters, args.min_best, args.min_positive))
        ign_ok = True
        if args.min_igniters is not None:
            ign_ok &= c_ign >= args.min_igniters
        if args.min_best is not None:
            ign_ok &= c_best >= args.min_best
        if args.min_positive is not None:
            ign_ok &= c_pos >= args.min_positive
        detail = (f"igniters keeper={k_ign}(sus {k_sus}) cand={c_ign}(sus {c_sus}); "
                  f"best={c_best:+.4f} pos={c_pos:.1%}")
        if bars_given:
            detail += (f" bars[ign>={args.min_igniters} best>={args.min_best} "
                       f"pos>={args.min_positive}]")
            verdicts.append(("ignitionTail", ign_ok, detail))
        else:
            print(f"  [info] ignitionTail  {detail}")
    except Exception as e:  # tail metrics must never break the core gate
        print(f"  [info] ignitionTail  (unavailable: {e})")

    hdr = f"fleet gate: N={n} horizon={horizon} policy={args.policy} boot={args.boot}"
    if label:
        hdr += f" [{label}]"
    print(hdr)
    ok = True
    for name, passed, detail in verdicts:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:14s} {detail}")
    return ok

def equivalence_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args) -> int:
    k_t, k_e, _ = death_times(keep_fin, horizon)
    c_t, c_e, _ = death_times(cand_fin, horizon)
    k_del, c_del = delivered(keep_fin, keep_seeds), delivered(cand_fin, cand_seeds)
    print(f"fleet gate --equivalence: N={n} horizon={horizon} policy={args.policy} "
          f"boot={args.boot} (unpaired TOST)")

    def tost(name: str, draws: np.ndarray, band: float, fmt: str = "{:+.4f}") -> str:
        lo, hi = np.percentile(draws, [5, 95])
        point = float(np.median(draws))
        if -band <= lo and hi <= band:
            verdict = "EQUIVALENT"
        elif hi < -band or lo > band:
            verdict = "DIFFERENT"
        else:
            verdict = "INCONCLUSIVE"
        half = (hi - lo) / 2
        need = (half / band) ** 2 if band > 0 else float("inf")
        extra = f" needs ~{need:.1f}x N" if verdict == "INCONCLUSIVE" and need > 1 else ""
        print(f"  [{verdict:12s}] {name:14s} Δ={fmt.format(point)} "
              f"CI90=[{fmt.format(lo)}, {fmt.format(hi)}] band=±{fmt.format(band).lstrip('+')}{extra}")
        return verdict

    verdicts = []
    dp = unpaired_bootstrap(k_e.astype(float), c_e.astype(float), np.mean,
                            n_boot=args.boot, seed=11)

    band_p = args.delta if args.delta is not None else 0.03
    verdicts.append(tost("divergence", dp, band_p))
    di = unpaired_bootstrap(k_del, c_del, iqm, n_boot=args.boot, seed=12)
    verdicts.append(tost("iqmDelivered", di, args.rate_rel * abs(iqm(k_del)), "{:+.5g}"))
    rng = np.random.default_rng(13)
    ik = rng.integers(0, n, size=(args.boot, n))
    ic = rng.integers(0, n, size=(args.boot, n))
    dr = np.array([
        rmst(c_t[rc], c_e[rc], horizon) - rmst(k_t[rk], k_e[rk], horizon)
        for rk, rc in zip(ik, ic)])
    verdicts.append(tost("rmstSurvival", dr, args.rmst_rel * horizon, "{:+,.0f}"))

    def nf_prevalence(fin, seeds, phase):
        return np.array([
            1.0 if ((fin[s].get("health") or {}).get("nonFiniteByPhase") or {}).get(phase, 0) > 0
            else 0.0
            for s in seeds])
    nf_details = []
    for i, ph in enumerate(NF_PHASES):
        kp = nf_prevalence(keep_fin, keep_seeds, ph)
        cp = nf_prevalence(cand_fin, cand_seeds, ph)
        d_nf = unpaired_bootstrap(kp, cp, np.mean, n_boot=args.boot, seed=20 + i)
        lo, hi = np.percentile(d_nf, [5, 95])
        if -band_p <= lo and hi <= band_p:
            v = "EQUIVALENT"
        elif hi < -band_p or lo > band_p:
            v = "DIFFERENT"
        else:
            v = "INCONCLUSIVE"
        verdicts.append(v)
        nf_details.append(f"{ph} Δ={float(np.median(d_nf)):+.4f}[{lo:+.4f},{hi:+.4f}]({v[0]})")
    print(f"  [{'mixed' if len(set(verdicts[-4:])) > 1 else verdicts[-1]:12s}] "
          f"nfPrevalence   {' '.join(nf_details)} band=±{band_p:.4f}")

    d_del, p_del = ks_test(k_del, c_del)
    d_srv, p_srv = ks_test(k_t[k_e == 1], c_t[c_e == 1]) if k_e.sum() and c_e.sum() else (0.0, 1.0)
    ks_ok = p_del >= 0.01 and p_srv >= 0.01
    print(f"  [{'EQUIVALENT' if ks_ok else 'DIFFERENT':12s}] ksDistribution deliveredRate "
          f"D={d_del:.4f} p={p_del:.4f}; deathTimes D={d_srv:.4f} p={p_srv:.4f}")
    verdicts.append("EQUIVALENT" if ks_ok else "DIFFERENT")
    if all(v == "EQUIVALENT" for v in verdicts):
        print("EQUIVALENT")
        return 0
    if "DIFFERENT" in verdicts:
        print("DIFFERENT")
        return 1
    print("INCONCLUSIVE — increase N or horizon and rerun")
    return 3

def cmd_gate(args) -> int:
    keep_run, cand_run = load_run(args.keeper), load_run(args.candidate)
    keep_fin, cand_fin = (finals(r, args.policy) for r in (keep_run, cand_run))
    keep_seeds, cand_seeds = sorted(keep_fin), sorted(cand_fin)
    if args.equivalence:
        if len(keep_seeds) != len(cand_seeds):
            print(f"GATE ERROR: population sizes differ "
                  f"({len(keep_seeds)} vs {len(cand_seeds)}).")
            return 2
    elif keep_seeds != cand_seeds:
        print(f"GATE ERROR: seed sets differ (keeper {len(keep_seeds)}, "
              f"candidate {len(cand_seeds)}); paired comparison requires identical seeds.")
        return 2
    kh, ch = keep_run["meta"]["steps"], cand_run["meta"]["steps"]
    if kh != ch:
        print(f"GATE ERROR: horizons differ (keeper {kh}, candidate {ch}).")
        return 2
    n, horizon = len(keep_seeds), kh
    # Refuse to gate a silently-killed run: a partial run (laptop sleep / OOM / harness stop) that did
    # not reach its declared horizon must never be PROMOTEd or REJECTed as if it finished.
    for label, fin in (("keeper", keep_fin), ("candidate", cand_fin)):
        ms = max((int(f["step"]) for f in fin.values()), default=0)
        if ms < horizon:
            print(f"GATE ERROR: {label} run is INCOMPLETE — reached step {ms:,} of {horizon:,} "
                  f"({ms / max(horizon, 1):.0%}); it was silently killed (sleep / OOM / harness stop). "
                  f"Re-run to completion before gating.")
            return 2
    if args.equivalence:
        return equivalence_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args)
    ok = gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, None,
                       keep_run=keep_run, cand_run=cand_run)
    print("PROMOTE" if ok else "REJECT")
    return 0 if ok else 1

def cmd_sota(args) -> int:
    roots = []
    for d in sorted(args.dir.iterdir()):
        if not (d / "meta.json").exists() or not (d / "fleet.jsonl").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except json.JSONDecodeError:
            continue
        if "oak" not in meta.get("policies", []):
            continue
        if meta.get("batch", 0) < args.min_lanes or meta.get("steps", 0) < args.min_steps:
            continue
        roots.append((meta.get("startedAt", ""), d, meta))
    if not roots:
        print(f"(no fleet roots with oak policy under {args.dir})")
        return 1
    print("### OaK reward scoreboard — delivered rate across fleet runs")
    print()
    print("| run | date | B × steps | IQM delivered (all) | best lane | positive | "
          "recent IQM | gap to TD7 (eval) | gap to SAC (delivered est.) |")
    print("|---|---|---|---|---|---:|---|---|---|")
    td7 = SOTA_EVAL["TD7"]
    sac = SOTA_DELIVERED_EST["SAC (delivered, est.)"]
    for started, d, meta in roots:
        try:
            run = load_run(d)
            r = analyze_policy(run, "oak", max(meta["steps"] // 5, 1))
        except Exception as e:
            print(f"| {d.name} | {started[:10]} | (unreadable: {e}) | | | | | | |")
            continue
        if not r:
            continue
        best = r["bestLaneRate"]
        g7 = f"{td7 / best:,.0f}x" if best > 0 else "n/a"
        gs = f"{sac / best:,.0f}x" if best > 0 else "n/a"
        print(f"| {d.name} | {started[:10]} | {meta['batch']}×{meta['steps']:,} "
              f"| {r['iqmDeliveredAll']:+.5g} | seed {r['bestLaneSeed']}: {best:+.4f} "
              f"| {r['positiveFraction']:.1%} | {r['iqmRecent']:+.5g} | {g7} | {gs} |")
    print()
    print(f"Reference lines: " + ", ".join(f"{k} {v}/step" for k, v in SOTA_EVAL.items())
          + " (eval); " + ", ".join(f"{k} ~{v}/step" for k, v in SOTA_DELIVERED_EST.items()) + ".")
    print(f"({SOTA_FRAMING_NOTE})")
    return 0

def latest_rows(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    path = root / "aggregate.jsonl"
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["policy"]] = row
    return rows

def render_watch(rows: dict[str, dict]) -> str:
    if not rows:
        return "(no aggregate.jsonl rows yet)"
    has_best = any("bestLaneSeed" in r for r in rows.values())
    hdr = (f"{'policy':8s} {'step':>10s} {'alive':>11s} "
           f"{'P10':>10s} {'P50':>10s} {'P90':>10s}")
    if has_best:
        hdr += f" {'best lane':>16s} {'recentP50':>10s}"
    hdr += f" {'agg sps':>10s} {'wall':>8s}"
    out = [hdr]
    for pol, r in sorted(rows.items()):
        total = r["alive"] + r["dead"]
        line = (f"{pol:8s} {r['step']:>10,d} {r['alive']:>6d}/{total:<4d} "
                f"{r['deliveredRateP10']:>10.4g} {r['deliveredRateP50']:>10.4g} "
                f"{r['deliveredRateP90']:>10.4g}")
        if has_best:
            if "bestLaneSeed" in r:
                line += (f" {('s' + str(r['bestLaneSeed']) + ':' + format(r['bestLaneDeliveredRate'], '+.3f')):>16s}"
                         f" {r.get('recentRateP50', float('nan')):>10.4g}")
            else:
                line += f" {'-':>16s} {'-':>10s}"
        line += f" {r['aggStepsPerSecond']:>10,.0f} {r['wallSeconds']:>7.0f}s"
        out.append(line)
    return "\n".join(out)

def cmd_watch(args) -> int:
    while True:
        print(render_watch(latest_rows(args.root)), flush=True)
        if not args.follow:
            return 0
        time.sleep(args.interval)
        print()

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("report", help="population report -> markdown + summary.json")
    rp.add_argument("root", type=Path)
    rp.add_argument("--policy", default=None, help="restrict to one policy")
    rp.add_argument("--window", type=int, default=None,
                    help="rolling-rate window for plasticity (default: 2 heartbeats)")
    rp.add_argument("--json", type=Path, default=None,
                    help="summary.json path (default: <root>/summary.json)")
    rp.add_argument("--leaderboard", action="store_true",
                    help="append top-N lanes by delivered rate")
    rp.add_argument("--top", type=int, default=10)

    sp = sub.add_parser("sota", help="reward scoreboard across all fleet roots vs SOTA lines")
    sp.add_argument("--dir", type=Path, default=Path(".oak-benchmarks"))
    sp.add_argument("--min-lanes", type=int, default=512,
                    help="hide validation/debug roots below this population")
    sp.add_argument("--min-steps", type=int, default=50_000)

    gp = sub.add_parser("gate", help="promotion law / instrument-swap equivalence")
    gp.add_argument("keeper", type=Path)
    gp.add_argument("candidate", type=Path)
    gp.add_argument("--policy", default="oak")
    gp.add_argument("--delta", type=float, default=None,
                    help="divergence margin (paired default 0.64/sqrt(N); "
                         "equivalence band default 0.03)")
    gp.add_argument("--rate-rel", type=float, default=0.05)
    gp.add_argument("--rmst-rel", type=float, default=0.05)
    gp.add_argument("--boot", type=int, default=10_000)
    gp.add_argument("--min-igniters", type=int, default=None,
                    help="blocking ignition bar: candidate lanes with peak windowed rate > 0.1")
    gp.add_argument("--min-best", type=float, default=None,
                    help="blocking bar: candidate best-lane delivered rate")
    gp.add_argument("--min-positive", type=float, default=None,
                    help="blocking bar: candidate positive-lane fraction (0..1)")
    gp.add_argument("--equivalence", action="store_true",
                    help="unpaired TOST instrument-swap gate (see module docstring)")

    wp = sub.add_parser("watch", help="live aggregate dashboard")
    wp.add_argument("root", type=Path)
    wp.add_argument("--follow", action="store_true")
    wp.add_argument("--interval", type=float, default=5.0)

    args = ap.parse_args()
    return {"report": cmd_report, "gate": cmd_gate, "watch": cmd_watch,
            "sota": cmd_sota}[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
