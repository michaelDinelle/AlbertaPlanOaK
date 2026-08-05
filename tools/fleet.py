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

# summary.json schema. v2 (2026-08-04) is the terminology migration: a fleet lane
# IS a seed (one agent, one stream), "delivered rate" IS the average reward rate,
# "zenith/decay" IS peak/retention, and an "igniter" IS a seed that crossed the
# solved threshold. Same numbers, same gate decisions — only the names moved.
# Readers accept v1 (no summarySchemaVersion) via frontend/server/summaryAdapter.ts.
SUMMARY_SCHEMA_VERSION = 2
# A seed counts as SOLVED when its peak windowed reward rate exceeded this.
# Structural, not tuned: it is the same 0.1/step bar the gate's solved-seed
# (was: igniter) criterion has always used — published in summary.json so no
# reader has to hard-code it.
SOLVED_THRESHOLD = 0.1

SOTA_EVAL = {"TD7": 17.4, "TD-MPC2": 15.1, "MR.Q": 13.3, "TD3": 10.1, "SAC": 9.9}
SOTA_DELIVERED_EST = {"SAC (delivered, est.)": 6.0}
SOTA_FRAMING_NOTE = (
    "eval = final-policy evaluation on free 1000-step episodes (published); "
    "delivered = online reward over the entire stream, flailing included (ours). "
    "Per-FLOP claims require both sides instrumented (AGENTS.md rules).")


def json_safe(o):
    """Replace non-finite floats with `null` so the artifact is REAL JSON.

    RFC 8259 / ECMA-404 have no NaN or Infinity; `json.dumps` emits them bare as a
    Python extension, and every strict parser (JS `JSON.parse`, Go, Rust, jq)
    rejects the whole document. v2 persists per-seed vectors, and `retentionBySeed`
    carries `nan` for any seed whose peak windowed rate was <= 0 — so ONE such seed
    used to make the entire summary unreadable and silently blanked the frontend's
    population panel (`readJsonFile` in frontend/server/runIndex.ts returns null on
    a parse error). `null` is the honest encoding of "undefined for this seed", and
    it is what every reader already renders as "—".
    """
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path: Path, payload) -> None:
    """The ONLY sanctioned artifact writer: sanitize, then refuse to emit non-JSON.

    `allow_nan=False` is a permanent tripwire — if a non-finite ever escapes
    `json_safe` (a new numpy scalar type, say), this raises instead of writing a
    file that no reader can parse.
    """
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True,
                               allow_nan=False))

# --- Phase 8: per-organ reward/GFLOP instrument ----------------------------
# organCostVersion 1 mirror of the host-side static FLOP model
# (Sources/Main.swift, "organFlops / flopsPerLaneStep"). MUST stay term-for-term
# identical to the Swift derivation: same guards, same fma-units-then-x2
# convention. Legacy roots (no organFlops in meta.json) are recomputed with
# this mirror and flagged "recomputed-v1".
ORGAN_COST_VERSION = 1
ORGAN_COST_NOTE = (
    "static trip-count model (fma=2 flops), ablation-aware; over-attributes "
    "~3-5x vs silicon — consume organ values as RELATIVE SHARES or per-organ "
    "deltas across ablations, never absolute FLOPs (design "
    "docs/research/allocation-and-instrument-design.md §1). Cross-version "
    "absolute comparisons are refused.")
# Env-window physics estimates (mirror of the Swift envFlops table).
ENV_FLOPS = {
    "HalfCheetah-v4": 20_000, "FourRooms-cont-v0": 40, "DeepSea-cont-v0": 20,
    "KeyCorridor-cont-v0": 60, "Acrobot-cont-v0": 200, "MountainCar-cont-v0": 40,
    "CartPole-cont-v0": 60, "Pendulum-cont-v0": 50, "Chain-cont-v0": 10,
    "LargeDeepSea-cont-v0": 25, "MultiKeyCorridor-cont-v0": 60,
    "MemoryChain-cont-v0": 15,
    "ProceduralMaze-cont-v0": 50, "HighDimMaze-cont-v0": 55,
    "POMaze-cont-v0": 55, "DelayedChain-cont-v0": 8,
}

def organ_flops_v1(ablations) -> dict[str, int]:
    """organCostVersion 1: FC-STOMP organ -> flops/lane-step (agent only).
    fma-units accumulation x2 at the end; guards identical to the Swift model.
    Config: universal I/O contract O=17, D=12 -> inputW=99 -> R=256, K=32,
    A=64, DL=6, traceLevels=4, Rn=R (OAK_SUTTON_MULT=1)."""
    abl = set(ablations or [])
    f_obs, f_d = 17, 12
    traces = int(math.log2(max(f_obs, 1))) + 1
    input_w = traces * f_obs + f_d + 2
    f_r = 32
    while f_r < 2 * input_w:
        f_r *= 2
    f_k, f_a, f_dl = 32, 64, max(f_d // 2, 1)
    f_tl = max(traces - 1, 0)
    f_rn = f_r
    fma = {
        "FeatureConstruction": 16 + 3 * 32,
        "GVFPrediction": 2 * f_a + 6 * f_a,
        "RewardRespectingSubtasks": 2 * f_k * f_a + f_k * f_a,
        "OptionPolicies": f_k * f_dl,
        "OptionModels": 0, "Planning": 0,
        "OptionKeyboard": f_k + f_d * f_k + f_k,
        "SearchControl": f_a * f_k + f_k + f_k // 2,
        "GenerateAndTest": 0,
        "ActionEmission": 5 * f_d + 3 * f_k * f_dl + f_k,
        "misc": 4_000,
    }
    if "feature-construction" not in abl:
        fma["FeatureConstruction"] += f_obs * (2 + f_tl) + 32 * 7 * 5
    if "gvf-prediction" not in abl:
        fma["GVFPrediction"] += 2 * f_a
    if "rr-subtasks" not in abl:
        fma["RewardRespectingSubtasks"] += 2 * f_k * f_a + f_k
    if "option-policies" not in abl:
        fma["OptionPolicies"] += 2 * f_k * f_a * f_dl
        if "generate-and-test" not in abl:
            fma["OptionPolicies"] += f_dl * f_k + f_dl * f_a
    if "option-models" not in abl:
        fma["OptionModels"] += 4 * f_k * f_a + 4 * f_k * f_a + 2 * f_k * f_a + 16 * f_k
    if "planning" not in abl:
        fma["Planning"] += 14 * f_a * 32 + 4 * f_k * f_a + 2 * f_k * 32
    if "generate-and-test" not in abl:
        fma["GenerateAndTest"] += f_rn // 2 + f_r + f_k
    return {k: 2 * v for k, v in fma.items()}

# Design §2 Tier-0: per-organ online yield proxies already flowing through
# fleet.jsonl health (kernelRho lives at row top level). Ranking signals only
# — they never convict and are never written into learning targets.
TIER0_SIGNALS = {
    "FeatureConstruction": ["oakFeatureReinitCountFinal"],
    "GVFPrediction": ["mainTd", "kernelRho"],
    # optionDurationMean was removed 2026-08-05: the kernel wrote a literal
    # 0.0f into it (a ranking signal that was identically zero).
    "RewardRespectingSubtasks": ["terminationRate", "selectedContinuation"],
    "OptionPolicies": ["actorAdvantageL1", "actorUpdateNormMean",
                       "actorSupportEntropy", "actionDeltaL2Mean"],
    "OptionModels": ["modelPriorityMax", "importanceEss"],
    "Planning": ["planTd"],
    "OptionKeyboard": ["keyboardL1", "keyboardEffectiveL1",
                       "selectedKeyboardWeight"],
    "SearchControl": ["searchControlUseMean"],
    "GenerateAndTest": ["generateTestEligibleCount"],
    "ActionEmission": ["actionDeltaL2Mean", "actionDeltaPeak"],
}

def env_flops_map(meta: dict) -> dict[str, int]:
    """envId -> per-env flopsPerLaneStep from meta.envSchedule (mixed runs)."""
    return {e["envId"]: e["flopsPerLaneStep"]
            for e in (meta.get("envSchedule") or []) if e.get("flopsPerLaneStep")}

def tier0_yields(run: dict, policy: str = "oak") -> dict[str, dict[str, dict]]:
    """Per-organ Tier-0 yield trend: per heartbeat, the cross-lane median of
    each mapped signal; reported as q25/q50/q75 over heartbeats + first/last
    (the trend). Organs whose signals are absent in this root are omitted."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    for (pol, _seed), rows in run["lanes"].items():
        if pol != policy:
            continue
        for r in rows:
            if r.get("kind") in ("hb", "final"):
                by_step[r["step"]].append(r)
    def sig_val(r, key):
        v = r.get(key)
        if not isinstance(v, (int, float)):
            v = (r.get("health") or {}).get(key)
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None
    out: dict[str, dict[str, dict]] = {}
    for organ, keys in TIER0_SIGNALS.items():
        for key in keys:
            medians = []
            for step in sorted(by_step):
                vals = [v for v in (sig_val(r, key) for r in by_step[step])
                        if v is not None]
                if vals:
                    medians.append((step, float(np.median(vals))))
            if not medians:
                continue
            arr = np.array([m for _, m in medians])
            q = np.percentile(arr, [25, 50, 75])
            out.setdefault(organ, {})[key] = {
                "heartbeatMedianQ25": float(q[0]), "heartbeatMedianQ50": float(q[1]),
                "heartbeatMedianQ75": float(q[2]),
                "firstHeartbeatMedian": float(arr[0]), "lastHeartbeatMedian": float(arr[-1]),
                "heartbeats": len(medians),
            }
    return out

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
    policies_seen: set[str] = set()
    with open(root / "fleet.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "hb":
                continue  # pre-tape archives; the tape is the only series source
            lanes[(row["policy"], row["seed"])].append(row)
            policies_seen.add(row["policy"])
    # METRICS LAW (2026-08-05): fleet.jsonl carries only the final certificate
    # rows; every heartbeat row comes from the continuous metrics tape. No
    # legacy path — a root without a tape is not a valid instrument.
    import metrics_tape
    for pol in policies_seen:
        if metrics_tape.find_tape(root, pol) is None:
            raise SystemExit(
                f"LOAD ERROR: {root} has no metrics.{pol}.oakt tape. Pre-tape "
                f"roots are archives, not instruments — re-run on the current "
                f"binary (forward-only law; the agent stream is bit-identical).")
        for row in metrics_tape.synth_hb_rows(root, pol):
            lanes[(row["policy"], row["seed"])].append(row)
    for rows in lanes.values():
        rows.sort(key=lambda r: r["step"])
    deaths = []
    deaths_path = root / "deaths.jsonl"
    if deaths_path.exists():
        with open(deaths_path) as f:
            deaths = [json.loads(line) for line in f]
    return {"meta": meta, "lanes": lanes, "deaths": deaths, "root": root}

def finals(run: dict, policy: str) -> dict[int, dict]:
    out = {}
    for (pol, seed), rows in run["lanes"].items():
        if pol != policy:
            continue
        fin = [r for r in rows if r["kind"] == "final"]
        out[seed] = fin[-1] if fin else rows[-1]
    return out

def lane_env_map(run: dict, policy: str) -> dict[int, str]:
    """seed -> envId. Mixed-env fleets stamp envId on every fleet.jsonl row and carry
    meta.envSchedule; legacy single-env roots predate per-row envId — fall back to
    meta.envId (every lane in those roots runs the one env)."""
    default = run["meta"].get("envId")
    out: dict[int, str] = {}
    for (pol, seed), rows in run["lanes"].items():
        if pol != policy:
            continue
        out[seed] = next((r.get("envId") for r in rows if r.get("envId")), default)
    return out

def env_slices(run: dict, policies: list[str]) -> dict[str, list[int]]:
    """envId -> sorted seeds, unioned across the reported policies (env is a property
    of the lane/seed, not the policy). Single-env roots return a one-entry map."""
    slices: dict[str, set[int]] = defaultdict(set)
    for p in policies:
        for s, e in lane_env_map(run, p).items():
            slices[e].add(s)
    return {e: sorted(ss) for e, ss in sorted(slices.items())}

def slice_run(run: dict, seeds: list[int]) -> dict:
    """A run view restricted to the given lane seeds — analyze/render over it is
    exactly the single-env computation over that env's lanes only."""
    keep = set(seeds)
    return {
        "meta": run["meta"],
        "lanes": {k: v for k, v in run["lanes"].items() if k[1] in keep},
        "deaths": [d for d in run["deaths"] if d.get("seed") in keep],
    }

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
    return (f"**SOTA gap** — best seed {best_rate:+.3f}/step, population IQM {iqm_all:+.4f}/step. "
            f"vs TD7 {td7}/step (eval): gap {gap(best_rate, td7)} at best seed. "
            f"vs SAC ~{sac_del}/step (delivered, est.): gap {gap(best_rate, sac_del)} at best seed. "
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
    """Per-seed peak & retention over the windowed reward rate.
    `retention` = final ÷ peak (was decayRatio): 1.0 means the seed still
    delivers everything it ever delivered."""
    series = [(r["step"], r["rewardSum"]) for r in rows]
    rates = rolling_rate(series, window)
    if not rates:
        return None
    peak_step, peak_rate = max(rates, key=lambda t: t[1])
    end_step, end_rate = rates[-1]
    retention = end_rate / peak_rate if peak_rate > 0 else float("nan")
    return {
        "peakStep": peak_step, "peakRewardRate": peak_rate,
        "endRewardRate": end_rate, "retention": retention,
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
        "policy": policy, "seeds": n, "horizon": horizon,
        "divergence": {"diverged": dead, "rate": p, "wilsonLo": lo, "wilsonHi": hi,
                       "byDetector": dict(sorted(by_detector.items()))},
        "rmst": rmst(times, events, horizon),
        "rewardRateIQM": iqm(rates),
        "rewardRateIQMNonDiverged": iqm(rates[surv_mask]) if surv_mask.any() else None,
        "bestSeed": seeds[best_idx],
        "bestSeedRewardRate": float(rates[best_idx]),
        "bestSeedTotalReward": float(fin[seeds[best_idx]]["rewardSum"]),
        "positiveSeeds": int((rates > 0).sum()),
        "positiveFraction": float((rates > 0).mean()),
        "recentRewardRateIQM": iqm(recent_arr),
        "recentRewardRateMax": float(recent_arr.max()),
        # Per-seed vectors are PERSISTED in summary.json (v2): the population is
        # the deliverable, and a reader that cannot see the per-seed spread has to
        # re-parse fleet.jsonl to draw it.
        "recentRewardRateBySeed": recents,
        "rewardRateBySeed": {s: float(r) for s, r in zip(seeds, rates)},
    }
    # Completeness: a run is valid only if it reached its declared horizon. A silently-killed run
    # (laptop sleep, OOM, harness stop) leaves lanes short of horizon — flag it loudly so no one
    # gates or reports a partial run as if it finished.
    max_step = max(int(f["step"]) for f in fin.values())
    has_final = any(r["kind"] == "final" for s in seeds for r in run["lanes"][(policy, s)])
    out["complete"] = bool(has_final and max_step >= horizon)
    out["maxStep"] = max_step
    # Population progress toward the goal (the WHOLE population at SOTA, staying there forever).
    # Report the distribution and the fraction of seeds clearing each reward-rate threshold — never
    # a single hand-picked seed. Thresholds: >0 walking at all, >1 the milestone, >5 ~SAC-delivered,
    # >10 SOTA.
    _pct = np.percentile(rates, [10, 25, 50, 75, 90, 99])
    out["rewardRatePercentiles"] = {k: float(v) for k, v in
                                    zip(("p10", "p25", "p50", "p75", "p90", "p99"), _pct)}
    out["thresholdFractions"] = {"gt0": float((rates > 0).mean()), "gt1": float((rates > 1).mean()),
                                 "gt5": float((rates > 5).mean()), "gt10": float((rates > 10).mean())}
    flops = run["meta"].get("flopsPerLaneStep")
    if flops:
        out["rewardPerGFLOP"] = out["rewardRateIQM"] / (flops / 1e9)
        out["bestSeedRewardPerGFLOP"] = out["bestSeedRewardRate"] / (flops / 1e9)
    # `n` is THIS view's seed count — for a per-env slice that is the slice's own
    # population, never the pooled fleet, so a 64-seed env slice still publishes
    # its own CI.
    if n >= 4:
        out["rewardRateIQMCI95"] = list(bootstrap_ci(rates, iqm, seed=1))
    ts, surv = km_curve(times, events)
    out["survival"] = {str(int(t)): float(s) for t, s in zip(ts, surv)}
    if policy == "oak":
        plast, collapses = [], 0
        retention_by_seed: dict[int, float] = {}
        for s in seeds:
            rows = run["lanes"][(policy, s)]
            pl = plasticity(rows, window)
            if pl:
                plast.append(pl)
                retention_by_seed[s] = float(pl["retention"])
            if entropy_collapse_step(rows) is not None:
                collapses += 1
        if plast:
            out["peak"] = {
                "peakStepQ": quartiles([p_["peakStep"] for p_ in plast]),
                "peakRewardRateQ": quartiles([p_["peakRewardRate"] for p_ in plast]),
                "retentionQ": quartiles(
                    [p_["retention"] for p_ in plast if math.isfinite(p_["retention"])]),
            }
            out["retentionBySeed"] = retention_by_seed
            # Solved-threshold metrics (Mark III audit law: Tier-M IQM alone is blind
            # to walking-mode kills — the tail is a first-class gate axis).
            # solved   := seed whose peak windowed rate exceeded SOLVED_THRESHOLD;
            # retained := solved seed still holding >= half its peak at the end.
            _ign = [p_ for p_ in plast if p_["peakRewardRate"] > SOLVED_THRESHOLD]
            _sus = [p_ for p_ in _ign
                    if math.isfinite(p_["retention"]) and p_["retention"] >= 0.5]
            out["solvedSeeds"] = len(_ign)
            out["solvedSeedsRetained"] = len(_sus)
            out["solvedRetentionFraction"] = (len(_sus) / len(_ign)) if _ign else None
            # "Staying power": fraction of seeds retaining >= half their peak rate at the end
            # (the opposite of post-peak decay/crash). A continuous online agent must STAY at rate.
            _dec = [p_["retention"] for p_ in plast if math.isfinite(p_["retention"])]
            out["retentionFraction"] = (
                float(np.mean([d >= 0.5 for d in _dec])) if _dec else None)
        cp, clo, chi = wilson(collapses, n)
        out["entropyCollapse"] = {"count": collapses, "rate": cp,
                                  "wilsonLo": clo, "wilsonHi": chi}
    sps = [fin[s].get("stepsPerSecond") for s in seeds if fin[s].get("stepsPerSecond")]
    if sps:
        out["aggStepsPerSecond"] = float(np.median(sps)) * n
        out["seedStepsTotal"] = int(sum(fin[s]["step"] for s in seeds))
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
        "**Average reward rate (the headline)**",
        "",
        "| policy | IQM reward rate ± 95% CI (all seeds) | best seed | positive seeds "
        "| recent-window IQM | recent max"
        + (" | reward/GFLOP |" if any(r.get("rewardPerGFLOP") is not None for r in results if r) else " |"),
        "|---|---|---|---:|---|---"
        + ("|---|" if any(r.get("rewardPerGFLOP") is not None for r in results if r) else "|"),
    ]
    has_flop = any(r.get("rewardPerGFLOP") is not None for r in results if r)
    for r in results:
        if not r:
            continue
        ci = r.get("rewardRateIQMCI95")
        ci_s = f" [{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
        row = (f"| {r['policy']} | **{r['rewardRateIQM']:.5g}**{ci_s} "
               f"| seed {r['bestSeed']}: {r['bestSeedRewardRate']:+.4f}/step "
               f"({r['bestSeedTotalReward']:+,.0f} total) "
               f"| {r['positiveSeeds']}/{r['seeds']} = {r['positiveFraction']:.1%} "
               f"| {r['recentRewardRateIQM']:+.5g} | {r['recentRewardRateMax']:+.4f} ")
        if has_flop:
            rg = r.get("rewardPerGFLOP")
            row += f"| {rg:+.4g} |" if rg is not None else "| n/a |"
        else:
            row += "|"
        lines.append(row)
    oak = next((r for r in results if r and r["policy"] == "oak"), None)
    if oak:
        lines += ["", sota_gap_line(oak["bestSeedRewardRate"], oak["rewardRateIQM"])]
    lines += [
        "",
        "**Population progress (the whole population is the deliverable — never a single seed)**",
        "",
        "| policy | complete | reward rate P10/P50/P90 | %>0 | %>1 | %>5 | %>SOTA(10) | retention |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if not r:
            continue
        rd, tf = r.get("rewardRatePercentiles", {}), r.get("thresholdFractions", {})
        sust = r.get("retentionFraction")
        comp = ("yes" if r.get("complete", True)
                else f"**NO {r.get('maxStep', 0) / max(meta['steps'], 1):.0%}**")
        lines.append(
            f"| {r['policy']} | {comp} "
            f"| {rd.get('p10', 0):+.4f} / {rd.get('p50', 0):+.4f} / {rd.get('p90', 0):+.4f} "
            f"| {tf.get('gt0', 0):.1%} | {tf.get('gt1', 0):.1%} | {tf.get('gt5', 0):.1%} "
            f"| {tf.get('gt10', 0):.1%} | {'n/a' if sust is None else f'{sust:.1%}'} |")
    lines += [
        "",
        "**Survival & robustness** (RMST = restricted mean survival time)",
        "",
        "| policy | seeds | numerically diverged (Wilson 95%) | RMST/horizon | IQM (non-diverged) |",
        "|---|---:|---|---:|---|",
    ]
    for r in results:
        if not r:
            continue
        d = r["divergence"]
        surv = r.get("rewardRateIQMNonDiverged")
        lines.append(
            f"| {r['policy']} | {r['seeds']} | {d['diverged']}/{r['seeds']} = {d['rate']:.1%} "
            f"[{d['wilsonLo']:.1%}, {d['wilsonHi']:.1%}] | {r['rmst'] / r['horizon']:.3f} "
            f"| {'n/a' if surv is None else f'{surv:.5g}'} |")
    for r in results:
        if r and r.get("peak"):
            p_ = r["peak"]
            ec = r["entropyCollapse"]
            lines += [
                "",
                f"**{r['policy']} peak & retention** (quartiles 25/50/75): peak step {p_['peakStepQ']}; "
                f"peak reward rate {p_['peakRewardRateQ']}; retention {p_['retentionQ']}. "
                f"Entropy collapse: {ec['count']}/{r['seeds']} = {ec['rate']:.1%} "
                f"[{ec['wilsonLo']:.1%}, {ec['wilsonHi']:.1%}].",
            ]
        if r and r["divergence"]["byDetector"]:
            dets = ", ".join(f"{k}: {v}" for k, v in r["divergence"]["byDetector"].items())
            lines.append(f"**{r['policy']} divergence detectors**: {dets}.")
        if r:
            lines.append(
                f"**{r['policy']} throughput**: {r.get('aggStepsPerSecond', 0):,.0f} agg steps/s"
                + (f"; {run_flops_str(r)}" if r.get("rewardPerGFLOP") is not None else "") + ".")
    return "\n".join(lines)

def run_flops_str(r: dict) -> str:
    return f"best-seed reward/GFLOP {r['bestSeedRewardPerGFLOP']:+.4g} (OaK measured; SOTA est. only)"

def render_leaderboard(run: dict, result: dict, top: int) -> str:
    policy = result["policy"]
    fin = finals(run, policy)
    delivered = result["rewardRateBySeed"]
    recents = result["recentRewardRateBySeed"]
    order = sorted(delivered, key=lambda s: delivered[s], reverse=True)[:top]
    lines = [
        "", f"**{policy} leaderboard — top {len(order)} seeds by average reward rate**", "",
        "| seed | reward rate/step | total reward | recent rate | peak step | status |",
        "|---:|---|---|---|---:|---|",
    ]
    window = max(result["horizon"] // 5, 1)
    for s in order:
        rows = run["lanes"][(policy, s)]
        pl = plasticity(rows, window)
        zen = f"{pl['peakStep']:,}" if pl else "n/a"
        rec, life = recents[s], delivered[s]
        nf = fin[s].get("firstNonFiniteStep") is not None
        if rec > max(life, 0) * 1.1 and rec > 0:
            status = "climbing"
        elif pl and pl["peakRewardRate"] > 0 and rec < 0.5 * pl["peakRewardRate"]:
            status = "decaying"
        else:
            status = "steady"
        if nf:
            status += " (NF)"
        lines.append(
            f"| {s} | {life:+.4f} | {fin[s]['rewardSum']:+,.0f} | {rec:+.4f} | {zen} | {status} |")
    return "\n".join(lines)

def build_organ_report(run: dict, slice_results: dict[str, list[dict]],
                       results: list[dict], slices: dict[str, list[int]]) -> dict:
    """<run-root>/organ-report.json — the Phase-8 purge artifact: per-organ
    FLOP share (meta.organFlops when present, else the recomputed v1 mirror),
    per-organ Tier-0 yield trends, and the per-env reward/GFLOP breakdown, so
    a purge can read 'organ X: Y% of agent FLOPs, Z measured yield'."""
    meta = run["meta"]
    ablations = meta.get("ablation") or []
    if meta.get("organFlops"):
        organ_flops = {k: int(v) for k, v in meta["organFlops"].items()}
        version = int(meta.get("organCostVersion", 0))
        source = "meta.json"
    else:
        organ_flops = organ_flops_v1(ablations)
        version = ORGAN_COST_VERSION
        source = "recomputed-v1 (legacy root predates organFlops)"
    agent_flops = sum(organ_flops.values())
    yields = tier0_yields(run)
    organs = {}
    for organ in sorted(organ_flops):
        fl = organ_flops[organ]
        entry = {
            "flopsPerLaneStep": fl,
            "share": fl / agent_flops if agent_flops else 0.0,
        }
        if organ in yields:
            entry["tier0Yield"] = yields[organ]
        organs[organ] = entry
    envs: dict[str, dict] = {}
    env_flops = env_flops_map(meta)
    res_by_env = slice_results if slice_results else {meta.get("envId"): results}
    for env_id, sres in res_by_env.items():
        r = next((x for x in sres if x and x.get("policy") == "oak"), None) or \
            next((x for x in sres if x), None)
        if not r:
            continue
        fl = env_flops.get(env_id) or meta.get("flopsPerLaneStep")
        env_ph = ENV_FLOPS.get(env_id)
        v1_fl = (agent_flops + env_ph) if env_ph is not None else None
        envs[env_id] = {
            "seeds": r["seeds"],
            "flopsPerLaneStep": fl,
            "envFlopsPerLaneStep": env_ph,
            "agentFlopsPerLaneStep": agent_flops,
            # v1-consistent lane cost: this report's organ model + the env
            # physics estimate. For legacy roots (v0 meta) `flopsPerLaneStep`
            # above embeds the pre-organ agent model — use the v1 pair for any
            # cross-root ratio work (the caveat's version rule, applied).
            "v1FlopsPerLaneStep": v1_fl,
            "rewardRateIQM": r["rewardRateIQM"],
            "positiveFraction": r["positiveFraction"],
            "retentionFraction": r.get("retentionFraction"),
            "divergenceRate": r["divergence"]["rate"],
            "rewardPerGFLOP": (r["rewardRateIQM"] / (fl / 1e9)) if fl else None,
            "v1RewardPerGFLOP": (r["rewardRateIQM"] / (v1_fl / 1e9)) if v1_fl else None,
        }
    report = {
        "organCostVersion": version,
        "organCostSource": source,
        "caveat": ORGAN_COST_NOTE,
        "ablation": ablations,
        "ablationLabel": meta.get("ablationLabel", "none"),
        "agentFlopsPerLaneStep": agent_flops,
        "envSeeds": {e: len(s) for e, s in slices.items()},
        "organs": organs,
        "envs": envs,
    }
    ledger = Path(run["root"]).parent / "ORGANS.json"
    if ledger.exists():
        # The cross-run conviction ledger (Tier-1 ablation-matrix verdicts):
        # attached verbatim so this report is the single purge brief.
        try:
            report["organsLedger"] = json.loads(ledger.read_text())
        except json.JSONDecodeError:
            report["organsLedger"] = {"error": "unparseable ORGANS.json"}
    return report

def render_organ_section(report: dict) -> str:
    """Compact printed form of the organ report (full artifact: organ-report.json)."""
    lines = [
        "",
        "**Organ yield × GFLOP** (organCostVersion %d, %s)"
        % (report["organCostVersion"], report["organCostSource"]),
        "",
        "> Shares are RELATIVE (static trip-count model, ~3-5x over-attribution "
        "vs silicon); per-organ reward conviction comes from the paired "
        "ablation matrix (ORGANS.json), not from this table. Tier-0 yields "
        "rank hypotheses — they never convict.",
        "",
        "| organ | FLOP share | flops/lane-step | tier-0 yield (heartbeat-median q50, first→last) |",
        "|---|---:|---:|---|",
    ]
    for organ, e in sorted(report["organs"].items(),
                           key=lambda kv: -kv[1]["flopsPerLaneStep"]):
        y = e.get("tier0Yield") or {}
        ys = "; ".join(
            f"{k} {v['heartbeatMedianQ50']:.3g} ({v['firstHeartbeatMedian']:.3g}→"
            f"{v['lastHeartbeatMedian']:.3g})" for k, v in sorted(y.items())) or "—"
        lines.append("| %s | %.1f%% | %s | %s |"
                     % (organ, 100.0 * e["share"], f"{e['flopsPerLaneStep']:,}", ys))
    if report.get("envs"):
        lines += [
            "",
            "**Per-env reward/GFLOP** (per-env lane cost = agent + env physics; "
            "IQM average reward rate over all seeds of the slice)",
            "",
            "| env | seeds | IQM reward rate | %>0 | retention | divergence | reward/GFLOP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for env_id, e in sorted(report["envs"].items()):
            sust = e["retentionFraction"]
            rpg = e["rewardPerGFLOP"]
            lines.append(
                "| %s | %d | %+.4f | %.1f%% | %s | %.1f%% | %s |"
                % (env_id, e["seeds"], e["rewardRateIQM"],
                   100.0 * e["positiveFraction"],
                   "n/a" if sust is None else f"{sust:.1%}",
                   100.0 * e["divergenceRate"],
                   "n/a" if rpg is None else f"{rpg:+.4g}"))
    return "\n".join(lines)

def cmd_report(args) -> int:
    run = load_run(args.root)
    meta = run["meta"]
    window = args.window or max(meta["steps"] // 5, 1)
    policies = [args.policy] if args.policy else meta.get("policies", ["oak"])
    results = [analyze_policy(run, p, window) for p in policies]
    # Mixed-env fleets (meta.envSchedule + per-lane envId): reward scales differ
    # across envs, so lanes are NEVER pooled into one IQM — print the full
    # population report once per env slice, then the legacy overall block with a
    # warning. Single-env roots take the old path untouched (byte-identical output).
    slices = env_slices(run, policies)
    mixed = len(slices) > 1
    slice_results: dict[str, list[dict]] = {}
    if mixed:
        env_flops = env_flops_map(meta)
        for env_id, seeds in slices.items():
            view = slice_run(run, seeds)
            # Per-env slice view: the env's own lane cost (env physics differs
            # per env; the top-level flopsPerLaneStep is the primary env's) and
            # lane count, so rewardPerGFLOP and the header are per-env honest.
            view["meta"] = {
                **meta, "envId": env_id, "batch": len(seeds),
                "flopsPerLaneStep": env_flops.get(
                    env_id, meta.get("flopsPerLaneStep")),
            }
            sres = [analyze_policy(view, p, window) for p in policies]
            slice_results[env_id] = sres
            print(render_markdown(sres, view["meta"]))
            if args.leaderboard:
                for r in sres:
                    if r:
                        print(render_leaderboard(view, r, args.top))
            print()
        print("> ⚠️ MIXED-ENV RUN: overall block mixes envs; read slices "
              "(reward scales differ across envs).")
        print()
    print(render_markdown(results, meta))
    if args.leaderboard:
        for r in results:
            if r:
                print(render_leaderboard(run, r, args.top))
    # v2 persists the per-seed vectors (rewardRateBySeed / recentRewardRateBySeed /
    # retentionBySeed) instead of stripping them: the population IS the deliverable,
    # and a reader that only gets aggregates has to re-parse fleet.jsonl to draw the
    # spread. Nothing is dropped on the way out any more.
    summary = {
        "summarySchemaVersion": SUMMARY_SCHEMA_VERSION,
        "solvedThreshold": SOLVED_THRESHOLD,
        "meta": meta,
        "results": [r for r in results if r],
    }
    if mixed:
        summary["envSlices"] = {env: [r for r in sres if r]
                                for env, sres in slice_results.items()}
    out = args.json or (args.root / "summary.json")
    write_json(out, summary)
    # Phase 8: per-organ reward/GFLOP artifact (the purge brief).
    organ_report = build_organ_report(run, slice_results, results, slices)
    write_json(args.root / "organ-report.json", organ_report)
    print(render_organ_section(organ_report))
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
        "rewardRateIQM (was iqmDelivered)", di5 >= -margin,
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
    # mode-kill — the tail is a first-class BLOCKING gate axis). Four criteria,
    # paired by seed with the same machinery as the core gate:
    #   solvedSeeds   — per-seed solved indicator, non-inferiority at δ
    #   retainedSeeds — per-seed retained-solved indicator, non-inferiority at δ
    #   bestLaneClass — paired top-8-seed mean reward rate, 5% relative band
    #   positiveFrac  — per-seed rate>0 indicator, non-inferiority at δ
    # Explicit absolute bars (--min-igniters/--min-best/--min-positive) ADD on top.
    def _ign_vec(run_, seeds_):
        run_lanes = run_["lanes"]
        pol = args.policy
        window = max(horizon // 5, 1)
        ign = np.zeros(len(seeds_))
        sus = np.zeros(len(seeds_))
        for j, s_ in enumerate(seeds_):
            pl = plasticity(run_lanes[(pol, s_)], window)
            if pl and pl["peakRewardRate"] > SOLVED_THRESHOLD:
                ign[j] = 1.0
                if math.isfinite(pl["retention"]) and pl["retention"] >= 0.5:
                    sus[j] = 1.0
        return ign, sus
    try:
        if keep_run is None or cand_run is None:
            raise ValueError("run data not provided")
        k_ign_v, k_sus_v = _ign_vec(keep_run, keep_seeds)
        c_ign_v, c_sus_v = _ign_vec(cand_run, cand_seeds)
        c_best = float(np.max(c_del))
        c_pos = float((c_del > 0).mean())
        for name, kv, cv, seed_ in (("solvedSeeds (was igniters)", k_ign_v, c_ign_v, 30),
                                    ("retainedSeeds (was sustainers)", k_sus_v, c_sus_v, 31)):
            d95 = float(np.percentile(
                paired_bootstrap(cv, kv, np.mean, n_boot=args.boot, seed=seed_), 95))
            verdicts.append((name, d95 <= delta,
                             f"keeper={int(kv.sum())} cand={int(cv.sum())} "
                             f"Δ95={d95:+.4f} δ={delta:.4f}"))
        # bestLaneClass REMOVED as a promotion criterion (user law 2026-07-20):
        # "rockets are an instability, all agents should converge to the same
        # high reward rate." The gate's population axes (IQM delivered rate,
        # igniters, sustainers, positiveFrac, divergence, RMST, NF prevalence)
        # are the promotion law; the top-8 best-lane mean is NOT a blocking
        # criterion — it preserved the rocket mode the user wants eliminated.
        k_pos_v = (k_del > 0).astype(float)
        c_pos_v = (c_del > 0).astype(float)
        d_pos95 = float(np.percentile(
            paired_bootstrap(c_pos_v, k_pos_v, np.mean, n_boot=args.boot, seed=33), 95))
        verdicts.append(("positiveSeedFraction (was positiveFrac)", d_pos95 <= delta,
                         f"keeper={k_pos_v.mean():.1%} cand={c_pos:.1%} "
                         f"Δ95={d_pos95:+.4f} δ={delta:.4f}"))
        ign_ok = True
        bars_given = any(v is not None for v in
                         (args.min_igniters, args.min_best, args.min_positive))
        if args.min_igniters is not None:
            ign_ok &= int(c_ign_v.sum()) >= args.min_igniters
        if args.min_best is not None:
            ign_ok &= c_best >= args.min_best
        if args.min_positive is not None:
            ign_ok &= c_pos >= args.min_positive
        if bars_given:
            verdicts.append(("solvedBars (was ignitionBars)", ign_ok,
                             f"best={c_best:+.4f} pos={c_pos:.1%} bars[solved>={args.min_igniters} "
                             f"best>={args.min_best} pos>={args.min_positive}]"))
    except Exception as e:  # tail metrics must never break the core gate
        print(f"  [info] solvedTail  (unavailable: {e})")

    hdr = f"fleet gate: N={n} seeds horizon={horizon} policy={args.policy} boot={args.boot}"
    if label:
        hdr += f" [{label}]"
    print(hdr)
    ok = True
    # Criterion labels carry their pre-v2 name in parens; the tests, thresholds and
    # PASS/FAIL semantics behind them are unchanged.
    width = max((len(name) for name, _, _ in verdicts), default=14)
    for name, passed, detail in verdicts:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:{width}s} {detail}")
    if keep_run is not None and cand_run is not None and label is None:
        print_per_env_delta(keep_run, cand_run, keep_fin, cand_fin, keep_seeds, args)
    return ok

def print_per_env_delta(keep_run, cand_run, keep_fin, cand_fin, seeds, args) -> None:
    """DIAGNOSTIC per-env ΔIQM. Never blocking — it prints, it does not vote.

    The pooled IQM is dominated by whichever env brings the most seeds and the
    largest reward scale: under the default schedule HalfCheetah is 1024 of 1408
    lanes with rewards ~100x the gridworlds', so an env can collapse without moving
    the headline. That blindness is real, and a `perEnvNoRegression` criterion did
    run against it in July 2026 (see .oak-benchmarks/boost-rhat-tierL-20260725.log:
    "2/16 envs regressed: HighDimMaze-cont-v0 -94.20%, POMaze-cont-v0 -91.55%").
    It was never committed — it lived only in an uncommitted working copy and was
    lost — so this restores its INFORMATION, not its verdict.

    Deliberately not revived as law: its recorded form rejected stepcap1M-20260724
    for "Acrobot-cont-v0 -2.0%, Chain-cont-v0 -0.0%" while every other criterion
    PASSED with an IQM IMPROVEMENT. A -0.0% point difference on a 64-seed slice is
    noise, and it printed no interval, so it was an unpaired point test with a
    ~zero threshold. Restoring that verbatim would reject almost everything; picking
    a new threshold here would be inventing promotion law by guess. For a per-env
    PROMOTE/REJECT use the real instrument that already exists:
    `fleet.py gate --env-slice <envId>`, which runs EVERY criterion paired on that
    env's own lanes with the same non-inferiority margins.
    """
    envs = lane_env_map(keep_run, args.policy)
    by_env: dict[str, list[int]] = defaultdict(list)
    for s in seeds:
        e = envs.get(s)
        if e is not None:
            by_env[e].append(s)
    if len(by_env) < 2:
        return  # single-env root: the headline IQM already IS the per-env number
    rows = []
    for i, (env_id, e_seeds) in enumerate(sorted(by_env.items())):
        k = delivered(keep_fin, e_seeds)
        c = delivered(cand_fin, e_seeds)
        k_iqm, c_iqm = iqm(k), iqm(c)
        d5 = float(np.percentile(
            paired_bootstrap(k, c, iqm, n_boot=args.boot, seed=40 + i), 5))
        rel = (c_iqm - k_iqm) / abs(k_iqm) if k_iqm else float("nan")
        rows.append((rel, env_id, len(e_seeds), k_iqm, c_iqm, d5))
    rows.sort(key=lambda r: (math.isnan(r[0]), r[0]))  # worst first, undefined last
    print(f"  [info] perEnvDelta  (diagnostic, NOT a criterion — "
          f"per-env verdicts: gate --env-slice <envId>)")
    for rel, env_id, m, k_iqm, c_iqm, d5 in rows:
        rel_s = "  n/a" if math.isnan(rel) else f"{rel:+.1%}"
        print(f"           {env_id:26s} n={m:<5d} IQM {k_iqm:+.5g} -> {c_iqm:+.5g} "
              f"({rel_s})  ΔIQM5={d5:+.5g}")

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
    verdicts.append(tost("rewardRateIQM", di, args.rate_rel * abs(iqm(k_del)), "{:+.5g}"))
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
    keep_envs = lane_env_map(keep_run, args.policy)
    cand_envs = lane_env_map(cand_run, args.policy)
    if args.env_slice is not None:
        # Restrict BOTH roots to lanes on this env, paired by seed. The 8-criterion
        # gate below then runs unchanged on the slices.
        keep_fin = {s: f for s, f in keep_fin.items() if keep_envs.get(s) == args.env_slice}
        cand_fin = {s: f for s, f in cand_fin.items() if cand_envs.get(s) == args.env_slice}
        if not keep_fin or not cand_fin:
            print(f"GATE ERROR: --env-slice {args.env_slice}: empty slice "
                  f"(keeper {len(keep_fin)} lanes, candidate {len(cand_fin)} lanes).")
            return 2
        k_only = sorted(set(keep_fin) - set(cand_fin))
        c_only = sorted(set(cand_fin) - set(keep_fin))
        if k_only or c_only:
            print(f"GATE ERROR: --env-slice {args.env_slice}: slice seed sets differ — "
                  f"seeds only in keeper: {k_only[:8]}{'...' if len(k_only) > 8 else ''}; "
                  f"only in candidate: {c_only[:8]}{'...' if len(c_only) > 8 else ''}.")
            return 2
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
    if args.env_slice is None:
        # Same seed on different envs is not a paired comparison — reward scales differ.
        mixed = [(s, keep_envs.get(s), cand_envs.get(s)) for s in keep_seeds
                 if s in cand_envs and keep_envs.get(s) != cand_envs.get(s)]
        if mixed:
            ex = ", ".join(f"seed {s}: keeper={ke} candidate={ce}" for s, ke, ce in mixed[:5])
            print(f"GATE ERROR: env mismatch on {len(mixed)} paired seed(s) — {ex}"
                  f"{'...' if len(mixed) > 5 else ''}. Lanes on different envs are not "
                  f"comparable; re-gate per env with --env-slice <envId>.")
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
    label = f"env={args.env_slice}" if args.env_slice is not None else None
    ok = gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, label,
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
    print("### OaK reward scoreboard — average reward rate across fleet runs")
    print()
    print("| run | date | B × steps | IQM reward rate (all seeds) | best seed | positive | "
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
        best = r["bestSeedRewardRate"]
        g7 = f"{td7 / best:,.0f}x" if best > 0 else "n/a"
        gs = f"{sac / best:,.0f}x" if best > 0 else "n/a"
        print(f"| {d.name} | {started[:10]} | {meta['batch']}×{meta['steps']:,} "
              f"| {r['rewardRateIQM']:+.5g} | seed {r['bestSeed']}: {best:+.4f} "
              f"| {r['positiveFraction']:.1%} | {r['recentRewardRateIQM']:+.5g} | {g7} | {gs} |")
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
    hdr = (f"{'policy':8s} {'step':>10s} {'nonDiv':>11s} "
           f"{'P10':>10s} {'P50':>10s} {'P90':>10s}")
    if has_best:
        hdr += f" {'best seed':>16s} {'recentP50':>10s}"
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
                    help="rolling-rate window for peak/retention (default: 2 heartbeats)")
    rp.add_argument("--json", type=Path, default=None,
                    help="summary.json path (default: <root>/summary.json)")
    rp.add_argument("--leaderboard", action="store_true",
                    help="append top-N seeds by average reward rate")
    rp.add_argument("--top", type=int, default=10)

    sp = sub.add_parser("sota", help="reward scoreboard across all fleet roots vs SOTA lines")
    sp.add_argument("--dir", type=Path, default=Path(".oak-benchmarks"))
    sp.add_argument("--min-lanes", type=int, default=512,
                    help="hide validation/debug roots below this seed count")
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
                    help="blocking solved bar: candidate seeds with peak windowed rate > 0.1")
    gp.add_argument("--min-best", type=float, default=None,
                    help="blocking bar: candidate best-seed average reward rate")
    gp.add_argument("--min-positive", type=float, default=None,
                    help="blocking bar: candidate positive-seed fraction (0..1)")
    gp.add_argument("--equivalence", action="store_true",
                    help="unpaired TOST instrument-swap gate (see module docstring)")
    gp.add_argument("--env-slice", default=None, metavar="ENV_ID",
                    help="mixed-env fleets: restrict BOTH roots to seeds whose envId "
                         "matches, paired by seed on the slice (hard error on empty or "
                         "asymmetric slices), then run the gate unchanged")

    wp = sub.add_parser("watch", help="live aggregate dashboard")
    wp.add_argument("root", type=Path)
    wp.add_argument("--follow", action="store_true")
    wp.add_argument("--interval", type=float, default=5.0)

    lp2 = sub.add_parser("lanes", help="per-seed forensics: pack health diff at a step")
    lp2.add_argument("root", type=Path)
    lp2.add_argument("--at", type=int, required=True, help="heartbeat step to slice")
    lp2.add_argument("--policy", default="oak")

    fp = sub.add_parser("floors", help="extract per-task normalization floors -> FLOORS.json")
    fp.add_argument("root", type=Path, help="a zero/random-policy fleet root (run report first)")

    args = ap.parse_args()
    return {"report": cmd_report, "gate": cmd_gate, "watch": cmd_watch, "lanes": cmd_lanes,
            "sota": cmd_sota, "floors": cmd_floors}[args.cmd](args)


def cmd_floors(args) -> int:
    """Per-task normalization floors from a zero/random-policy sweep.

    Reads the root's summary.json (v2) and writes <bench-root>/FLOORS.json:
      { floorsSchemaVersion: 1, sourceRun, steps,
        floors: { <envId>: { zero: rewardRateIQM, random: rewardRateIQM,
                             floor: max(zero, random) } } }
    `floor` is the normalization zero-point (the better trivial policy — the
    convention rliable-style normalized scores expect).
    """
    root: Path = args.root
    sp = root / "summary.json"
    if not sp.exists():
        print(f"FLOORS ERROR: {sp} missing — run `fleet.py report {root}` first")
        return 1
    summary = json.loads(sp.read_text())
    if summary.get("summarySchemaVersion", 1) < 2:
        print("FLOORS ERROR: summary is pre-v2 — regenerate with the current fleet.py report")
        return 1
    slices = summary.get("envSlices") or {}
    floors: dict[str, dict] = {}
    for env_id, results in sorted(slices.items()):
        by_policy = {r.get("policy"): r.get("rewardRateIQM") for r in results}
        zero, random_ = by_policy.get("zero"), by_policy.get("random")
        if zero is None and random_ is None:
            continue
        candidates = [v for v in (zero, random_) if v is not None]
        floors[env_id] = {"zero": zero, "random": random_, "floor": max(candidates)}
    if not floors:
        print("FLOORS ERROR: no zero/random envSlices in this summary — was the run "
              "launched with OAK_BENCH_POLICIES=zero,random ?")
        return 1
    meta = summary.get("meta") or {}
    out = {
        "floorsSchemaVersion": 1,
        "sourceRun": root.name,
        "steps": meta.get("steps"),
        "floors": floors,
    }
    dest = root.parent / "FLOORS.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest} — {len(floors)} tasks (floor = max(zero, random) reward rate IQM)")
    for env_id, f in sorted(floors.items()):
        print(f"  {env_id:28s} zero {f['zero']!s:>12} random {f['random']!s:>12} -> floor {f['floor']}")
    return 0


def cmd_lanes(args) -> int:
    """Per-seed forensics: crashed / near-zero / median / leader health diff."""
    import statistics as _st
    root = Path(args.root)
    rows = [json.loads(l) for l in open(root / "fleet.jsonl")
            if f'"step": {args.at}' in l or f'"step":{args.at}' in l]
    lanes = [r for r in rows if r.get("kind") == "hb" and r.get("policy", "oak") == args.policy]
    if not lanes:
        print(f"no heartbeat rows at step {args.at}"); return 1
    def rate(r): return r["rewardSum"] / max(r["step"], 1)
    packs = {
        "crashed(<-0.1)": [r for r in lanes if rate(r) < -0.1],
        "negative": [r for r in lanes if -0.1 <= rate(r) < 0],
        "low(0-0.15)": [r for r in lanes if 0 <= rate(r) < 0.15],
        "mid(0.15-0.5)": [r for r in lanes if 0.15 <= rate(r) < 0.5],
        "leaders(>0.5)": [r for r in lanes if rate(r) >= 0.5],
    }
    keys = ["actorSupportEntropy", "terminationRate",
            "keyboardL1", "oakFeatureReinitCountFinal", "actionDeltaL2Mean"]
    print(f"pack | n | rate | " + " | ".join(keys) + " | pMax | rho")
    for name, g in packs.items():
        if not g: continue
        def h(k):
            vals = [(r["health"].get(k) if isinstance(r["health"].get(k), (int, float)) else 0) or 0 for r in g]
            return _st.median(vals)
        print(f"{name} | {len(g)} | {_st.median(rate(r) for r in g):+.3f} | "
              + " | ".join(f"{h(k):.3g}" for k in keys)
              + f" | {_st.median(r['paramAbsMax'] for r in g):.3g}"
              + f" | {_st.median(r['kernelRho'] for r in g):+.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
