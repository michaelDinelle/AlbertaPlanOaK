#!/usr/bin/env python3
"""Population science for OaK fleet runs: report, gate, watch.

  fleet.py report <run-root>             KM survival + RMST, divergence Wilson
                                         CIs, IQM delivered rate + bootstrap CI,
                                         plasticity quartiles -> markdown +
                                         summary.json
  fleet.py gate <keeper> <candidate>     the promotion law: 4 paired
                                         non-inferiority criteria (PROMOTE /
                                         REJECT); --equivalence switches to the
                                         unpaired TOST instrument-swap gate
                                         (EQUIVALENT / DIFFERENT / INCONCLUSIVE)
  fleet.py watch <run-root> [--follow]   live aggregate.jsonl dashboard

Statistics notes (standing law, see docs/roadmap.md):
- candidate gating is PAIRED by seed (variance reduction is real there);
- instrument swaps (e.g. physics backends) use UNPAIRED bootstrap + TOST with
  fixed practical-irrelevance bands, and any equivalence margin must first be
  validated against a measured same-physics null control;
- NF activity is compared by per-lane PREVALENCE, never by fleet sums (sums
  are heavy-tailed and dominated by a few diverged lanes).
"""

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
    surv, s = [], 1.0
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


def unpaired_bootstrap(keeper, candidate, stat, n_boot=10_000, seed=0):
    """Bootstrap distribution of stat(candidate*) - stat(keeper*) with each side
    resampled INDEPENDENTLY. Correct for physics-swap / disjoint-population
    comparisons where lanes are not paired twins (pairing by seed is meaningless
    across physics backends: different RNG streams, different trajectories)."""
    rng = np.random.default_rng(seed)
    k, c = np.asarray(keeper, dtype=float), np.asarray(candidate, dtype=float)
    ik = rng.integers(0, len(k), size=(n_boot, len(k)))
    ic = rng.integers(0, len(c), size=(n_boot, len(c)))
    return np.array([stat(c[rc]) - stat(k[rk]) for rk, rc in zip(ik, ic)])


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value (numpy only)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    if lam < 0.3:  # asymptotic series diverges as lam -> 0; p is 1 here
        return d, 1.0
    j = np.arange(1, 101)
    p = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * (lam * j) ** 2))
    return d, float(min(max(p, 0.0), 1.0))


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


def delivered(fin: dict[int, dict], seeds: list[int]) -> np.ndarray:
    return np.array([fin[s]["rewardSum"] / max(fin[s]["step"], 1) for s in seeds])


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


# -------------------------------------------------------------------- report


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
    out: dict = {
        "policy": policy, "lanes": n, "horizon": horizon,
        "divergence": {"dead": dead, "rate": p, "wilsonLo": lo, "wilsonHi": hi,
                       "byDetector": dict(sorted(by_detector.items()))},
        "rmst": rmst(times, events, horizon),
        "iqmDeliveredAll": iqm(rates),
        "iqmDeliveredSurvivors": iqm(rates[surv_mask]) if surv_mask.any() else None,
    }
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
        cp, clo, chi = wilson(collapses, n)
        out["entropyCollapse"] = {"count": collapses, "rate": cp,
                                  "wilsonLo": clo, "wilsonHi": chi}
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


def cmd_report(args) -> int:
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


# ---------------------------------------------------------------------- gate


def nf_totals(fin: dict[int, dict]) -> dict[str, int]:
    tot = dict.fromkeys(NF_PHASES, 0)
    for row in fin.values():
        nf = (row.get("health") or {}).get("nonFiniteByPhase") or {}
        for ph in NF_PHASES:
            tot[ph] += int(nf.get(ph, 0))
    return tot


def gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, label):
    delta = args.delta if args.delta is not None else 0.64 / np.sqrt(n)
    verdicts: list[tuple[str, bool, str]] = []

    # 1. divergence non-inferiority (paired)
    k_times, k_events, _ = death_times(keep_fin, horizon)
    c_times, c_events, _ = death_times(cand_fin, horizon)
    dp = paired_bootstrap(k_events.astype(float), c_events.astype(float),
                          np.mean, n_boot=args.boot, seed=11)
    dp95 = float(np.percentile(dp, 95))
    p_keep, p_cand = k_events.mean(), c_events.mean()
    verdicts.append((
        "divergence", dp95 <= delta,
        f"p keeper={p_keep:.3f} cand={p_cand:.3f} Δp95={dp95:+.4f} δ={delta:.4f}"))

    # 2. IQM delivered-rate non-inferiority (paired, all lanes)
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

    # 3. survival non-inferiority via RMST (paired)
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

    # 4. no new NF phases + ceiling
    k_nf, c_nf = nf_totals(keep_fin), nf_totals(cand_fin)
    new_phases = [ph for ph in NF_PHASES if c_nf[ph] > 0 and k_nf[ph] == 0]
    over = [ph for ph in NF_PHASES
            if c_nf[ph] > max(2 * k_nf[ph], 100 * n / 10)]
    verdicts.append((
        "nfPhases", not new_phases and not over,
        f"keeper={k_nf} cand={c_nf}"
        + (f" NEW={new_phases}" if new_phases else "")
        + (f" OVER={over}" if over else "")))

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
    """TOST-style two-sided equivalence with three-way verdicts per criterion:
    EQUIVALENT   — 90% CI of the unpaired bootstrap difference lies inside the band
    DIFFERENT    — 90% CI lies entirely outside the band
    INCONCLUSIVE — CI spans a band edge: underpowered at this N/horizon; the
                   printed factor estimates how much more N is needed.
    Overall: EQUIVALENT iff all numeric criteria EQUIVALENT and KS p >= 0.01;
    DIFFERENT if any criterion DIFFERENT or KS fails; else INCONCLUSIVE (exit 3)."""
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
    # Fixed practical-irrelevance band for instrument swaps (NOT N-dependent:
    # the band encodes what difference we don't care about; N buys resolution).
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

    # NF equivalence by per-lane PREVALENCE (fraction of lanes with any NF in
    # the phase), not raw fleet sums: sums are heavy-tailed (dominated by a few
    # diverged lanes; the same-physics null showed 1.67x total swings) and are
    # not a calibrated population statistic. Same TOST band as divergence.
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
    if args.equivalence:
        return equivalence_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args)
    ok = gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, None)
    print("PROMOTE" if ok else "REJECT")
    return 0 if ok else 1


# --------------------------------------------------------------------- watch


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
                continue  # partial last line of a live run
            rows[row["policy"]] = row
    return rows


def render_watch(rows: dict[str, dict]) -> str:
    if not rows:
        return "(no aggregate.jsonl rows yet)"
    out = [f"{'policy':8s} {'step':>10s} {'alive':>11s} "
           f"{'P10':>10s} {'P50':>10s} {'P90':>10s} {'agg sps':>10s} {'wall':>8s}"]
    for pol, r in sorted(rows.items()):
        total = r["alive"] + r["dead"]
        out.append(
            f"{pol:8s} {r['step']:>10,d} {r['alive']:>6d}/{total:<4d} "
            f"{r['deliveredRateP10']:>10.4g} {r['deliveredRateP50']:>10.4g} "
            f"{r['deliveredRateP90']:>10.4g} {r['aggStepsPerSecond']:>10,.0f} "
            f"{r['wallSeconds']:>7.0f}s")
    return "\n".join(out)


def cmd_watch(args) -> int:
    while True:
        print(render_watch(latest_rows(args.root)), flush=True)
        if not args.follow:
            return 0
        time.sleep(args.interval)
        print()


# ---------------------------------------------------------------------- main


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
    gp.add_argument("--equivalence", action="store_true",
                    help="unpaired TOST instrument-swap gate (see module docstring)")

    wp = sub.add_parser("watch", help="live aggregate dashboard")
    wp.add_argument("root", type=Path)
    wp.add_argument("--follow", action="store_true")
    wp.add_argument("--interval", type=float, default=5.0)

    args = ap.parse_args()
    return {"report": cmd_report, "gate": cmd_gate, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
