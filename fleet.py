#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from arch.bootstrap import MovingBlockBootstrap
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from rliable import library as rly
from scipy.optimize import isotonic_regression
from statsmodels.stats.proportion import proportion_confint

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
BASELINE_STORE = BENCH / "baselines"
ENVIRONMENT_SOURCES = ("Sources/EnvCore.metal", "Sources/Env.metal", "Sources/Baseline.metal",
                       "Sources/envs", "Sources/OaK/Resources")


def environment_digest() -> str:
    digest = hashlib.sha256()
    for entry in ENVIRONMENT_SOURCES:
        base = REPO / entry
        paths = sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else [base]
        for path in paths:
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def baseline_identity(meta: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({key: meta[key] for key in ("envSchedule", "seeds", "steps", "batch")},
                             sort_keys=True).encode())
    digest.update(environment_digest().encode())
    return digest.hexdigest()[:40]


def stored_baseline(meta: dict, policy: str) -> Path | None:
    root = BASELINE_STORE / baseline_identity(meta) / policy
    if not (root / ".done").is_file():
        return None
    run = load(root)
    if run["meta"]["policy"] != policy or not same_run_shape({"meta": meta}, run):
        raise SystemExit(f"{root}: stored baseline does not match the schedule it is filed under")
    if int(run["tape"]["ends"][-1]) < run["meta"]["steps"]:
        raise SystemExit(f"{root}: stored baseline is incomplete")
    return root


def link_baseline(candidate_root: Path, policy: str, stored: Path) -> None:
    link = candidate_root / "baselines" / policy
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(os.path.relpath(stored, link.parent), target_is_directory=True)

BOOT_REPS = 50_000
BOOT_BASE_SEED = 20260817
GATE_REPS = 10_000
PLASTICITY_FIELDS = {
    "oak": ["dormantFraction", "utilityDormantFraction", "featureBirths", "weightAbsMean",
            "weightAbsMeanValue", "erank", "logAlphaValueP10", "logAlphaValueP50",
            "logAlphaValueP90", "logAlphaOptionValueP50", "logAlphaModelP50"],
    "ppo": ["dormantFraction", "dormantFractionCritic", "weightAbsMean", "erank",
            "logStepP10", "logStepP50", "logStepP90"],
    "random": [],
}

FILE_MAGIC = 0x544B414F


def read_tape(path: Path) -> dict:
    data = path.read_bytes()
    magic, lanes, json_len = struct.unpack_from("<3I", data, 0)
    if magic != FILE_MAGIC:
        raise SystemExit(f"{path}: not an OAKT tape")
    fields = json.loads(data[12:12 + json_len])["fields"]
    n_f = len(fields)
    off = 12 + json_len
    rows = []
    while off + 16 <= len(data):
        ss, sc, ws, cl = struct.unpack_from("<IIfI", data, off)
        if off + 16 + cl > len(data):
            break
        block = zlib.decompress(data[off + 16:off + 16 + cl], -15)
        off += 16 + cl
        if len(block) != n_f * lanes * 4:
            raise SystemExit(f"{path}: tape row block size {len(block)} != {n_f * lanes * 4}")
        rows.append((ss + sc, ws, np.frombuffer(block, np.float32).reshape(n_f, lanes)))
    if not rows:
        raise SystemExit(f"{path}: no complete tape rows yet")
    ends, wall, blocks = zip(*rows)
    return {"fields": fields, "lanes": lanes, "ends": np.asarray(ends, np.int64),
            "wall": np.asarray(wall, np.float64), "data": np.stack(blocks)}


def load(root: Path) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    if "policy" not in meta:
        raise SystemExit(f"{root}: meta.json has no policy")
    tp = root / "tape.bin"
    if not tp.exists():
        raise SystemExit(f"{root}: no tape.bin")
    tape = read_tape(tp)
    ds = tape["data"][-1, tape["fields"].index("deathStep"), :]
    db = ds.view(np.uint32)
    deaths = {int(lane): int(db[lane]) for lane in np.nonzero(np.isfinite(ds))[0]}
    return {"meta": meta, "tape": tape, "deaths": deaths, "root": root}


def series(run: dict, field: str) -> np.ndarray:
    return run["tape"]["data"][:, run["tape"]["fields"].index(field), :]


def last_finite(mat: np.ndarray) -> np.ndarray:
    W, B = mat.shape
    mask = np.isfinite(mat)
    idx = W - 1 - np.argmax(mask[::-1], axis=0)
    v = mat[idx, np.arange(B)]
    return np.where(mask.any(axis=0), v, np.nan)


def peak_finite(mat: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(np.where(np.isfinite(mat), mat, -np.inf), axis=0)
    peak[~np.isfinite(peak)] = np.nan
    return peak


def env_slices(run: dict) -> dict[str, np.ndarray]:
    seed_lane = {s: i for i, s in enumerate(run["meta"]["seeds"])}
    out: dict[str, list[int]] = {}
    for e in run["meta"]["envSchedule"]:
        for s in range(e["seedStart"], e["seedEnd"] + 1):
            if s in seed_lane:
                out.setdefault(e["envId"], []).append(seed_lane[s])
    return {k: np.array(v) for k, v in sorted(out.items())}


def window_ctx(run: dict) -> tuple[np.ndarray, np.ndarray]:
    ends = run["tape"]["ends"]
    win = max(run["meta"]["steps"] // 5, 1)
    j = np.searchsorted(ends, ends - win, side="right") - 1
    return ends, j


def windowed_rates(run: dict, lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends, jr = window_ctx(run)
    i = np.nonzero(jr >= 0)[0]
    j = jr[i]
    rs = series(run, "rewardSum")[:, lanes]
    rates = (rs[i] - rs[j]) / (ends[i] - ends[j])[:, None]
    return peak_finite(rates), last_finite(rates)


def retention_null(run: dict, lanes: np.ndarray,
                   boot: int = 200) -> tuple[np.ndarray, np.ndarray]:
    ends, jr = window_ctx(run)
    rs_all = series(run, "rewardSum")[:, lanes]
    n = len(lanes)
    p5, spread = (np.full(n, np.nan) for _ in range(2))
    if not (jr >= 0).any() or rs_all.shape[0] < 8:
        return p5, spread
    dt = np.diff(ends).astype(float)
    wi = np.nonzero(jr >= 0)[0]
    wj = jr[wi]
    for li in range(n):
        rs = rs_all[:, li]
        fin = np.isfinite(rs)
        m = int(np.argmax(~fin)) if (~fin).any() else len(rs)
        if m < 8:
            continue
        dr = np.diff(rs[:m]) / dt[:m - 1]
        fit = isotonic_regression(dr).x
        res = dr - fit
        k = len(res)
        blk = max(1, int(round(math.sqrt(k))))
        mbb = MovingBlockBootstrap(blk, res, seed=101 + int(lanes[li]))
        res_star = np.stack([d[0] for d, _ in mbb.bootstrap(boot)])
        synth = np.cumsum((fit + res_star) * dt[:k], axis=1) + rs[0]
        synth = np.concatenate([np.full((boot, 1), rs[0]), synth], axis=1)
        mask = wi < m
        vi, vj = wi[mask], wj[mask]
        if vi.size == 0:
            continue
        r = (synth[:, vi] - synth[:, vj]) / (ends[vi] - ends[vj])
        pk = r.max(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = np.where(pk > 0, r[:, -1] / pk, np.nan)
        ok = np.isfinite(ret)
        if ok.sum() >= boot // 2:
            p5[li] = np.percentile(ret[ok], 5)
        spread[li] = float(np.percentile(pk, 95) - np.percentile(pk, 50))
    return p5, spread


def death_vectors(run: dict, lanes: np.ndarray,
                  censor: int) -> tuple[np.ndarray, np.ndarray]:
    died = np.array([1.0 if int(lane) in run["deaths"] else 0.0 for lane in lanes])
    dtimes = np.array(
        [run["deaths"].get(int(lane), censor) for lane in lanes], dtype=float)
    return died, dtimes


def lane_vectors(run: dict) -> dict[str, np.ndarray]:
    max_step = int(run["tape"]["ends"][-1])
    lanes = np.arange(run["tape"]["lanes"])
    delivered = last_finite(series(run, "rewardSum")) / max_step
    blind = ~np.isfinite(delivered)
    worst = min(0.0, float(np.min(delivered[~blind]))) if (~blind).any() else 0.0
    delivered = np.where(blind, worst, delivered)
    died, dtimes = death_vectors(run, lanes, run["meta"]["steps"])
    peak, end_rate = windowed_rates(run, lanes)
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = np.where(peak > 0, end_rate / peak, np.nan)
    null_p5, peak_spread = retention_null(run, lanes)
    learned = (np.isfinite(peak) & np.isfinite(peak_spread)
               & (peak > 0) & (peak > peak_spread))
    decayed = (learned & np.isfinite(retention) & np.isfinite(null_p5)
               & (retention < null_p5))
    return {"delivered": delivered, "died": died, "dtimes": dtimes,
            "positive": (delivered > 0).astype(float),
            "learned": learned.astype(float),
            "retained": (learned & ~decayed).astype(float)}


def take(vectors: dict[str, np.ndarray], lanes: np.ndarray) -> dict[str, np.ndarray]:
    return {name: v[lanes] for name, v in vectors.items()}


def iqm(v: np.ndarray) -> float:
    a = np.asarray(v, float).ravel()
    lower = int(0.25 * a.shape[0])
    upper = a.shape[0] - lower
    trimmed = np.partition(a, (lower, upper - 1))
    return float(trimmed[lower:upper].mean())


def impute_worst(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    if fin.all() or not fin.any():
        return v
    return np.where(fin, v, float(v[fin].min()))


def runs_matrix(cols: list) -> np.ndarray:
    return np.column_stack([impute_worst(c) for c in cols])


def survival_fit(times: np.ndarray, events: np.ndarray,
                 horizon: int) -> KaplanMeierFitter:
    return KaplanMeierFitter().fit(np.minimum(times, horizon), event_observed=events)


def rmst(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    if not events.any():
        return float(horizon)
    return float(restricted_mean_survival_time(
        survival_fit(times, events, horizon), t=float(horizon)))


def mean_fraction(v: np.ndarray, horizon: int) -> float:
    return float(np.mean(v))


def delivered_iqm(v: np.ndarray, horizon: int) -> float:
    return iqm(v)


def paired(best: dict, cand: dict, keys: tuple[str, ...], stat,
           horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(best[keys[0]])
    idx = rng.integers(0, n, size=(GATE_REPS, n))
    return np.array([stat(*[cand[key][row] for key in keys], horizon)
                     - stat(*[best[key][row] for key in keys], horizon) for row in idx])


def fl(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def fl_list(v) -> list:
    return [fl(x) for x in v]


def diag_iqm(v: np.ndarray) -> float | None:
    fin = v[np.isfinite(v)]
    return fl(iqm(fin)) if fin.size else None


def agg_iqm(scores: np.ndarray) -> np.ndarray:
    return np.array([iqm(scores)])


def agg_prob_improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pairs = x.shape[0] * y.shape[0]
    per_task = np.empty(x.shape[1])
    for task in range(x.shape[1]):
        ranked = np.sort(y[:, task])
        beaten = np.searchsorted(ranked, x[:, task], "left").sum()
        matched = np.searchsorted(ranked, x[:, task], "right").sum() - beaten
        per_task[task] = (beaten + 0.5 * matched) / pairs
    return np.array([per_task.mean()])


def ci_pair(ci: np.ndarray) -> list:
    return [fl(float(ci[0][0])), fl(float(ci[1][0]))]


def interval_estimate(scores, func, seed: int) -> tuple[float | None, list]:
    np.random.seed(seed)
    point, ci = rly.get_interval_estimates({"scores": scores}, func, reps=BOOT_REPS)
    return fl(float(point["scores"][0])), ci_pair(ci["scores"])


def rate_stats(rates: np.ndarray, seed: int) -> dict:
    point, ci = interval_estimate(runs_matrix([rates]), agg_iqm, seed)
    return {"iqm": point, "ci95": ci}


def divergence_block(run: dict, lanes: np.ndarray, max_step: int) -> dict:
    died, dtimes = death_vectors(run, lanes, max_step)
    diverged = int(died.sum())
    lo, hi = proportion_confint(diverged, len(lanes), method="wilson")
    km = survival_fit(dtimes, died, max_step)
    alive, alive_var = restricted_mean_survival_time(km, t=float(max_step), return_variance=True)
    half = 1.96 * math.sqrt(alive_var) if alive_var >= 0 else float("nan")
    sf = km.survival_function_.iloc[:, 0]
    sf_ci = km.confidence_interval_survival_function_
    return {"diverged": diverged, "rate": fl(died.mean()),
            "rateCi95": [fl(float(lo)), fl(float(hi))],
            "expectedAliveSteps": fl(float(alive)),
            "expectedAliveCi95": [fl(float(alive) - half), fl(float(alive) + half)],
            "km": {"steps": [int(t) for t in sf.index],
                   "survival": fl_list(sf.to_numpy()),
                   "ci95": [fl_list(sf_ci.iloc[:, 0].to_numpy()),
                            fl_list(sf_ci.iloc[:, 1].to_numpy())]}}


def full_window_rates(run: dict) -> np.ndarray:
    ends, j = window_ctx(run)
    rs = series(run, "rewardSum")
    jc = np.maximum(j, 0)
    start = np.where(j >= 0, ends[jc], 0.0)
    base = np.where(j[:, None] >= 0, rs[jc], 0.0)
    with np.errstate(invalid="ignore"):
        return (rs - base) / (ends - start)[:, None]


def final_window_rates(run: dict) -> np.ndarray:
    return last_finite(full_window_rates(run))


def same_run_shape(run: dict, other: dict) -> bool:
    keys = ("envSchedule", "seeds", "steps", "batch")
    return all(run["meta"][key] == other["meta"][key] for key in keys)


def baseline_runs(run: dict) -> dict[str, dict]:
    out = {}
    for policy in ("ppo", "random"):
        root = run["root"] / "baselines" / policy
        if not (root / ".done").exists():
            raise SystemExit(f"{run['root']}: missing complete {policy} baseline")
        base = load(root)
        if base["meta"]["policy"] != policy:
            raise SystemExit(f"{root}: policy is {base['meta']['policy']}, expected {policy}")
        if not same_run_shape(run, base):
            raise SystemExit(f"{root}: run shape differs from {run['root']}")
        if not (root / "summary.json").exists():
            raise SystemExit(f"{root}: missing summary.json")
        out[policy] = base
    return out


def vs_baseline(seed_base: int, run: dict, base: dict, slices: dict) -> dict:
    own_final = final_window_rates(run)
    base_final = final_window_rates(base)
    base_slices = env_slices(base)
    env_ids = sorted(slices)
    per_env = {}
    own_runs, base_runs = [], []
    for e, env in enumerate(env_ids):
        own = own_final[slices[env]]
        other = base_final[base_slices[env]]
        p, ci = interval_estimate(
            [runs_matrix([own]), runs_matrix([other])], agg_prob_improvement, seed_base + e)
        per_env[env] = {"p": p, "ci95": ci}
        own_runs.append(own)
        base_runs.append(other)
    mean_p, mean_ci = interval_estimate(
        [runs_matrix(own_runs), runs_matrix(base_runs)], agg_prob_improvement,
        seed_base + len(env_ids))
    return {"probImprovement": {"mean": mean_p, "ci95": mean_ci, "envs": per_env}}


def cmd_report(args) -> int:
    run = load(args.root)
    slices = env_slices(run)
    per_env_counts = {len(lanes) for lanes in slices.values()}
    if len(per_env_counts) != 1:
        raise SystemExit(f"{args.root}: uneven env slices {sorted(per_env_counts)}")
    meta = run["meta"]
    horizon = meta["steps"]
    ends = run["tape"]["ends"]
    max_step = int(ends[-1])
    if max_step < horizon:
        raise SystemExit(f"{args.root}: incomplete at {max_step:,} of {horizon:,} steps")
    n_lanes = run["tape"]["lanes"]
    started = datetime.strptime(meta["startedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    generated_at = (started + timedelta(seconds=float(run["tape"]["wall"][-1]))).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    rates = full_window_rates(run)
    final_rate = last_finite(rates)
    peak_rate = peak_finite(rates)
    env_ids = sorted(slices)
    plast_fields = PLASTICITY_FIELDS[meta["policy"]]
    missing = [f for f in plast_fields if f not in run["tape"]["fields"]]
    if missing:
        raise SystemExit(f"{args.root}: tape lacks plasticity fields {missing}")
    plast_rows = np.unique(np.linspace(0, len(ends) - 1, 100).astype(int))
    plast_series = {f: series(run, f) for f in plast_fields}
    envs_out = {}
    for e, env in enumerate(env_ids):
        lanes = slices[env]
        envs_out[env] = {
            "finalRate": rate_stats(final_rate[lanes], BOOT_BASE_SEED + 101 + e * 10),
            "forgetting": rate_stats(
                peak_rate[lanes] - final_rate[lanes], BOOT_BASE_SEED + 102 + e * 10),
            "divergence": divergence_block(run, lanes, max_step),
            "curves": {"iqm": fl_list(iqm(impute_worst(row)) for row in rates[:, lanes])},
        }
        if plast_fields:
            envs_out[env]["plasticity"] = {
                f: {"final": diag_iqm(plast_series[f][-1, lanes]),
                    "iqm": [diag_iqm(plast_series[f][r][lanes]) for r in plast_rows]}
                for f in plast_fields
            }
    summary = {
        "generatedAt": generated_at,
        "horizon": horizon,
        "maxStep": max_step,
        "complete": True,
        "seedsPerEnv": per_env_counts.pop(),
        "bootstrap": {"reps": BOOT_REPS, "baseSeed": BOOT_BASE_SEED},
        "curveSteps": [int(s) for s in ends],
        "curveSeconds": fl_list(run["tape"]["wall"]),
        "plasticitySteps": [int(ends[r]) for r in plast_rows] if plast_fields else [],
        "envs": envs_out,
        "aggregate": {"divergence": divergence_block(run, np.arange(n_lanes), max_step)},
    }
    if meta["policy"] == "oak":
        bases = baseline_runs(run)
        summary["vsBaselines"] = {
            policy: vs_baseline(
                BOOT_BASE_SEED + 3000 + i * 1000, run, bases[policy], slices)
            for i, policy in enumerate(("ppo", "random"))
        }
    out = args.root / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(out)
    return 0


GATE_ENV_CHECKS = ("divergence", "rewardRateIQM", "rmstSurvival")
GATE_FLEET_CHECKS = ("divergence", "rewardRateIQM", "learnedSeeds", "retainedSeeds",
                     "positiveSeedFraction")
WIN_CHECK = "rewardRateIQM"
GATE_CHECKS = {
    "divergence": (11, ("died",), mean_fraction, -1.0, "deaths avoided"),
    "rewardRateIQM": (12, ("delivered",), delivered_iqm, 1.0, "delivered IQM"),
    "rmstSurvival": (13, ("dtimes", "died"), rmst, 1.0, "RMST steps"),
    "learnedSeeds": (30, ("learned",), mean_fraction, 1.0, "learned fraction"),
    "retainedSeeds": (31, ("retained",), mean_fraction, 1.0, "retained fraction"),
    "positiveSeedFraction": (33, ("positive",), mean_fraction, 1.0, "positive fraction"),
}


def boot_seed(tag: str, offset: int) -> int:
    return (BOOT_BASE_SEED + offset
            + int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little"))


def slice_checks(tag: str, k: dict, c: dict, horizon: int,
                 categories: tuple[str, ...]) -> list[tuple[str, str, str, bool]]:
    out = []
    for name in categories:
        offset, keys, stat, sign, label = GATE_CHECKS[name]
        d = sign * paired(k, c, keys, stat, horizon, boot_seed(tag, offset))
        lo, hi = float(np.percentile(d, 5)), float(np.percentile(d, 95))
        out.append((name, "PASS" if hi >= 0.0 else "FAIL",
                    f"{label} [{lo + 0.0:+.6g}, {hi + 0.0:+.6g}]",
                    name == WIN_CHECK and lo > 0.0))
    return out


def throughput_check(best: dict, candidate: dict) -> tuple[str, str, str, bool]:
    ks = float(best["meta"]["stepsPerSecond"])
    cs = float(candidate["meta"]["stepsPerSecond"])
    kh = best["meta"]["hardwareKey"]
    ch = candidate["meta"]["hardwareKey"]
    if kh != ch:
        return ("throughput", "FAIL", f"hardware differs: best {kh}, candidate {ch}", False)
    delta = cs - ks
    return ("throughput", "PASS" if delta >= 0.0 else "FAIL",
            f"steps/s candidate {cs:,.0f} best {ks:,.0f} d {delta:+,.0f} on {ch}", False)


def _fail(msg: str) -> int:
    print(msg)
    return 2


def gate(best: dict, candidate: dict) -> int:
    for label, run in (("best", best), ("candidate", candidate)):
        if run["meta"]["policy"] != "oak":
            return _fail(f"GATE ERROR: {label} policy is {run['meta']['policy']}")
        if int(run["tape"]["ends"][-1]) < run["meta"]["steps"]:
            return _fail(f"GATE ERROR: {label} incomplete")
    if not same_run_shape(best, candidate):
        return _fail("GATE ERROR: environment, seed, horizon, or lane schedule differs")
    if best["meta"]["ablationLabel"] != candidate["meta"]["ablationLabel"]:
        return _fail("GATE ERROR: ablation differs")
    if best["meta"]["preset"] != candidate["meta"]["preset"]:
        return _fail("GATE ERROR: geometry preset differs")
    if not (candidate["root"] / "summary.json").exists():
        return _fail("GATE ERROR: candidate summary missing")
    baseline_runs(candidate)
    slices = env_slices(best)
    env_ids = sorted(slices)
    print(f"  environments ({len(env_ids)}): {', '.join(env_ids)}")
    horizon = best["meta"]["steps"]
    best_vectors = lane_vectors(best)
    candidate_vectors = lane_vectors(candidate)
    rows, wins, regressions = [], [], []

    def collect(scope: str, checks: list[tuple[str, str, str, bool]]) -> None:
        for name, status, detail, win in checks:
            rows.append((status, scope, name, detail, win))
            if status == "FAIL":
                regressions.append(f"{scope} {name}")
            if win:
                wins.append(f"{scope} {name} [{detail}]")

    for env in env_ids:
        lanes = slices[env]
        collect(env, slice_checks(
            env, take(best_vectors, lanes), take(candidate_vectors, lanes),
            horizon, GATE_ENV_CHECKS))
    all_lanes = np.concatenate([slices[env] for env in env_ids])
    collect("fleet", slice_checks(
        "fleet", take(best_vectors, all_lanes), take(candidate_vectors, all_lanes),
        horizon, GATE_FLEET_CHECKS))
    collect("fleet", [throughput_check(best, candidate)])
    scope_width = max(len(row[1]) for row in rows)
    check_width = max(len(row[2]) for row in rows)
    for status, scope, name, detail, win in rows:
        print(f"  {status:<7} {scope:<{scope_width}}  {name:<{check_width}}  {detail}"
              + ("  WIN" if win else ""))
    if regressions:
        print("REJECT: regression in " + ", ".join(regressions))
        return 1
    if not wins:
        print("REJECT: no strict reward win")
        return 1
    print("PROMOTE: strict reward win " + ", ".join(wins))
    return 0


def best_hash() -> str:
    try:
        name = (REPO / "best").read_text().strip()
    except OSError as error:
        raise SystemExit("best pointer missing") from error
    if not name:
        raise SystemExit("best pointer empty")
    return name


def best_run() -> dict:
    name = best_hash()
    root = BENCH / name
    if not root.is_dir():
        raise SystemExit(f"best artifact missing: {root}")
    run = load(root)
    if run["meta"]["policy"] != "oak":
        raise SystemExit(f"best policy is {run['meta']['policy']}")
    if int(run["tape"]["ends"][-1]) < run["meta"]["steps"]:
        raise SystemExit(f"best incomplete: {root}")
    return run


def write_best(name: str) -> None:
    (REPO / "best").write_text(name + "\n")


def cmd_gate(args) -> int:
    return gate(best_run(), load(args.candidate))


def env_schedule(meta: dict) -> str:
    return ",".join(
        f"{entry['envId']}:{entry['seedEnd'] - entry['seedStart'] + 1}"
        for entry in meta["envSchedule"])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_identity() -> dict:
    executable = REPO / ".build/release/OaK"
    if not executable.is_file():
        raise SystemExit(f"release executable missing: {executable}")
    metallib = REPO / ".build/OakKernel.metallib"
    if not metallib.is_file():
        raise SystemExit(f"Metal library missing: {metallib}; run ./setup.sh")
    metal_sources = sorted((REPO / "Sources").rglob("*.metal"))
    stale = next((path for path in metal_sources
                  if path.stat().st_mtime_ns > metallib.stat().st_mtime_ns), None)
    if stale is not None:
        raise SystemExit(f"Metal library is stale for {stale}; run ./setup.sh")
    paths = [REPO / "fleet.py", REPO / "Package.swift", REPO / "requirements.txt",
             REPO / "setup.sh"]
    resolved = REPO / "Package.resolved"
    if resolved.exists():
        paths.append(resolved)
    paths.extend(sorted(path for path in (REPO / "Sources").rglob("*") if path.is_file()))
    source = hashlib.sha256()
    for path in paths:
        source.update(str(path.relative_to(REPO)).encode())
        source.update(b"\0")
        source.update(path.read_bytes())
        source.update(b"\0")
    source_digest = source.hexdigest()
    binary = file_sha256(executable)
    metal = file_sha256(metallib)
    content = hashlib.sha256()
    content.update(source_digest.encode())
    content.update(binary.encode())
    content.update(metal.encode())
    return {
        "id": content.hexdigest()[:40],
        "source": source_digest,
        "binary": binary,
        "metal": metal,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
    }


def benchmark_env(meta: dict, policy: str, root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("OAK_ABLATE", None)
    env.update(
        OAK_BENCH_POLICY=policy,
        OAK_BENCH_STEPS=str(meta["steps"]),
        OAK_BENCH_SEED_COUNT=str(meta["batch"]),
        OAK_BENCH_SEEDS=",".join(map(str, meta["seeds"])),
        OAK_BENCH_ENVS=env_schedule(meta),
        OAK_FLEET_TIMING="1",
        OAK_METRICS_STORAGE=str(root),
    )
    if policy == "oak":
        env["OAK_ABLATE"] = meta["ablationLabel"]
    return env


def record_policy(candidate: dict, policy: str, identity: dict) -> dict:
    if policy != "oak":
        stored = stored_baseline(candidate["meta"], policy)
        if stored is not None:
            link_baseline(candidate["root"], policy, stored)
            print(f"[fleet] {policy} reused from {stored.relative_to(BENCH)}", flush=True)
            return load(stored)
        root = BASELINE_STORE / baseline_identity(candidate["meta"]) / policy
        root.mkdir(parents=True)
    else:
        root = candidate["root"]
    if file_sha256(REPO / ".build/release/OaK") != identity["binary"]:
        raise SystemExit("release executable changed during the science run")
    output = None
    log = None
    if policy != "oak":
        log = open(root.parent / f"{policy}.log", "w")
        output = log
    try:
        rc = subprocess.run(
            ["caffeinate", "-dimsu", str(REPO / ".build/release/OaK"), "benchmark"],
            env=benchmark_env(candidate["meta"], policy, root), cwd=REPO,
            stdout=output, stderr=subprocess.STDOUT if output else None).returncode
    finally:
        if log is not None:
            log.close()
    if rc != 0:
        raise SystemExit(f"{policy} benchmark exit={rc}")
    run = load(root)
    if run["meta"]["policy"] != policy:
        raise SystemExit(f"{root}: policy is {run['meta']['policy']}, expected {policy}")
    if not same_run_shape(candidate, run):
        raise SystemExit(f"{root}: run shape differs from {candidate['root']}")
    if int(run["tape"]["ends"][-1]) < run["meta"]["steps"]:
        raise SystemExit(f"{root}: incomplete")
    if policy != "oak":
        cmd_report(argparse.Namespace(root=root))
        (root / ".done").touch()
        link_baseline(candidate["root"], policy, root)
    print(f"[fleet] {policy} complete", flush=True)
    return run


def cmd_run(args) -> int:
    best_run()
    if os.environ.get("OAK_BENCH_POLICY", "oak") != "oak":
        raise SystemExit("science runs use the oak policy")
    if args.preset:
        subprocess.run([str(REPO / "setup.sh"), "--preset", args.preset], cwd=REPO, check=True)
    subprocess.run(["swift", "build", "-c", "release"], cwd=REPO, check=True)
    identity = artifact_identity()
    name = identity["id"]
    if name == best_hash():
        raise SystemExit(f"{name} is already the best")
    root = BENCH / name
    if root.exists():
        raise SystemExit(f"artifact already exists: {root}")
    BENCH.mkdir(exist_ok=True)
    root.mkdir()
    (root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n")
    log = BENCH / f"{name}.log"
    with open(log, "x") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "job", name]
            + (["--steps", str(args.steps)] if args.steps else []),
            stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=REPO)
    print(f"DETACHED PID {process.pid}\n  {log}\n  {root}/.done")
    return 0


def cmd_job(args) -> int:
    root = BENCH / args.name
    identity = json.loads((root / "identity.json").read_text())
    if identity != artifact_identity() or identity["id"] != args.name:
        raise SystemExit("science identity changed before the run started")
    best = best_run()
    meta = dict(best["meta"])
    if args.steps:
        meta["steps"] = int(args.steps)
    candidate = {"root": root, "meta": meta}
    for policy in ("ppo", "random"):
        stored = stored_baseline(meta, policy)
        if stored is not None:
            link_baseline(root, policy, stored)
    record_policy(candidate, "oak", identity)
    record_policy(candidate, "ppo", identity)
    record_policy(candidate, "random", identity)
    cmd_report(argparse.Namespace(root=root))
    completed = load(root)
    if (meta["steps"], completed["meta"]["preset"]) != (best["meta"]["steps"], best["meta"]["preset"]):
        print(f"[fleet] {completed['meta']['preset']} at {meta['steps']:,} steps is not the best's"
              f" {best['meta']['preset']} at {best['meta']['steps']:,}: recorded without a gate", flush=True)
        (root / ".done").touch()
        print("[fleet] DONE", flush=True)
        return 0
    result = gate(best, completed)
    if result == 0:
        write_best(args.name)
        print(f"[fleet] best {args.name}", flush=True)
    elif result == 1:
        print(f"[fleet] best stays {best_hash()}", flush=True)
    else:
        raise SystemExit("gate could not compare the completed run")
    (root / ".done").touch()
    print("[fleet] DONE", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--steps", type=int)
    rp.add_argument("--preset")
    jp = sub.add_parser("job")
    jp.add_argument("name")
    jp.add_argument("--steps", type=int)
    pp = sub.add_parser("report")
    pp.add_argument("root", type=Path)
    gp = sub.add_parser("gate")
    gp.add_argument("candidate", type=Path)
    args = ap.parse_args()
    return {"run": cmd_run, "job": cmd_job, "report": cmd_report,
            "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
