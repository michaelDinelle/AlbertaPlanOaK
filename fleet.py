#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
FLOOR_CACHE: dict[str, dict] = {}

FILE_MAGIC = 0x544B414F
SEG_MAGIC = 0x54474553

def read_tape(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, lanes, json_len = struct.unpack_from("<4I", data, 0)
    if magic != FILE_MAGIC or version != 1:
        raise SystemExit(f"{path}: not an OAKT v1 tape")
    fields = json.loads(data[16:16 + json_len])["fields"]
    n_f = len(fields)
    row_bytes = 8 + n_f * lanes * 4
    starts, counts, mats = [], [], []
    off = 16 + json_len
    while off + 20 <= len(data):
        smagic, n_win, raw_bytes, comp_bytes, crc = struct.unpack_from("<5I", data, off)
        comp = data[off + 20: off + 20 + comp_bytes]
        if (smagic != SEG_MAGIC or off + 20 + comp_bytes > len(data)
                or zlib.crc32(comp) != crc):
            break
        payload = zlib.decompress(comp, wbits=-15)
        if len(payload) != raw_bytes or raw_bytes != n_win * row_bytes:
            break
        for p in range(0, raw_bytes, row_bytes):
            s, c = struct.unpack_from("<2I", payload, p)
            starts.append(s)
            counts.append(c)
            mats.append(np.frombuffer(payload, np.float32, n_f * lanes, p + 8)
                        .reshape(n_f, lanes))
        off += 20 + comp_bytes
    if not mats:
        raise SystemExit(f"{path}: no complete tape windows yet")
    return {"fields": fields, "lanes": lanes,
            "ends": np.array(starts, np.int64) + np.array(counts, np.int64),
            "data": np.stack(mats)}

def load(root: Path, policy: str) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    tp = root / f"metrics.{policy}.oakt"
    if not tp.exists():
        raise SystemExit(f"{root}: no metrics.{policy}.oakt (pre-tape roots are "
                         f"archives — re-run on the current binary)")
    tape = read_tape(tp)
    deaths = {}
    dp = root / "deaths.jsonl"
    if dp.exists():
        for line in dp.read_text().splitlines():
            d = json.loads(line)
            if d.get("policy", "oak") == policy:
                deaths.setdefault(d["lane"], d)
    agg = None
    ap = root / "aggregate.jsonl"
    if ap.exists():
        for line in ap.read_text().splitlines():
            r = json.loads(line)
            if r.get("policy", "oak") == policy:
                agg = r
    seeds = meta.get("seeds") or list(range(tape["lanes"]))
    return {"meta": meta, "tape": tape, "deaths": deaths, "agg": agg,
            "seeds": seeds, "root": root}

def series(run: dict, field: str) -> np.ndarray:
    return run["tape"]["data"][:, run["tape"]["fields"].index(field), :]

def last_finite(mat: np.ndarray) -> np.ndarray:
    W, B = mat.shape
    mask = np.isfinite(mat)
    idx = W - 1 - np.argmax(mask[::-1], axis=0)
    v = mat[idx, np.arange(B)]
    return np.where(mask.any(axis=0), v, np.nan)

def floors() -> dict:
    if "floors" not in FLOOR_CACHE:
        fp = BENCH / "FLOORS.json"
        FLOOR_CACHE["floors"] = (json.loads(fp.read_text()).get("floors", {})
                                 if fp.exists() else {})
    return FLOOR_CACHE["floors"]

def lane_env_ids(run: dict) -> list[str]:
    n = run["tape"]["lanes"]
    sched = run["meta"].get("envSchedule") or []
    if not sched:
        return [run["meta"].get("envId", "")] * n
    seed_lane = {s: i for i, s in enumerate(run["seeds"])}
    out = [""] * n
    for e in sched:
        for s in range(e["seedStart"], e["seedEnd"] + 1):
            if s in seed_lane:
                out[seed_lane[s]] = e["envId"]
    return out

def lane_floor_vec(run: dict, lanes: np.ndarray) -> np.ndarray:
    fl = floors()
    envs = lane_env_ids(run)
    return np.array([fl.get(envs[l], {}).get("floor", 0.0) for l in lanes], float)

def isotonic_fit(y: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    wts: list[int] = []
    for x in y:
        v, w = float(x), 1
        while vals and vals[-1] > v:
            pv, pw = vals.pop(), wts.pop()
            v = (v * w + pv * pw) / (w + pw)
            w += pw
        vals.append(v)
        wts.append(w)
    out = np.empty(len(y))
    pos = 0
    for v, w in zip(vals, wts):
        out[pos:pos + w] = v
        pos += w
    return out

def window_ctx(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ends = run["tape"]["ends"]
    win = max(run["meta"]["steps"] // 5, 1)
    j = np.searchsorted(ends, ends - win, side="right") - 1
    i = np.nonzero(j >= 0)[0]
    return ends, i, j[i]

def windowed_rates(run: dict, lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends, i, j = window_ctx(run)
    rs = series(run, "rewardSum")[:, lanes]
    if i.size == 0:
        n = len(lanes)
        return np.full(n, np.nan), np.full(n, np.nan)
    rates = (rs[i] - rs[j]) / (ends[i] - ends[j])[:, None]
    peak = np.nanmax(np.where(np.isfinite(rates), rates, -np.inf), axis=0)
    peak[~np.isfinite(peak)] = np.nan
    fin_mask = np.isfinite(rates)
    li = rates.shape[0] - 1 - np.argmax(fin_mask[::-1], axis=0)
    end = np.where(fin_mask.any(axis=0), rates[li, np.arange(rates.shape[1])], np.nan)
    return peak, end

def retention_null(run: dict, lanes: np.ndarray,
                   boot: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ends, wi, wj = window_ctx(run)
    rs_all = series(run, "rewardSum")[:, lanes]
    n = len(lanes)
    p5, p50, spread = (np.full(n, np.nan) for _ in range(3))
    if wi.size == 0 or rs_all.shape[0] < 8:
        return p5, p50, spread
    dt = np.diff(ends).astype(float)
    for li in range(n):
        rs = rs_all[:, li]
        fin = np.isfinite(rs)
        m = int(np.argmax(~fin)) if (~fin).any() else len(rs)
        if m < 8:
            continue
        dr = np.diff(rs[:m]) / dt[:m - 1]
        fit = isotonic_fit(dr)
        res = dr - fit
        k = len(res)
        blk = max(1, int(round(math.sqrt(k))))
        nblk = -(-k // blk)
        rng = np.random.default_rng(101 + int(lanes[li]))
        starts = rng.integers(0, k - blk + 1, size=(boot, nblk))
        idx = (starts[:, :, None] + np.arange(blk)).reshape(boot, nblk * blk)[:, :k]
        synth = np.cumsum((fit + res[idx]) * dt[:k], axis=1) + rs[0]
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
            p50[li] = np.percentile(ret[ok], 50)
        spread[li] = float(np.percentile(pk, 95) - np.percentile(pk, 50))
    return p5, p50, spread

def analyze(run: dict, lanes: np.ndarray) -> dict:
    meta, horizon = run["meta"], run["meta"]["steps"]
    ends = run["tape"]["ends"]
    max_step = int(ends[-1])
    rs = series(run, "rewardSum")[:, lanes]
    delivered = last_finite(rs) / max_step
    delivered = np.nan_to_num(delivered, nan=0.0)
    died = np.array([1.0 if l in run["deaths"] else 0.0 for l in lanes])
    dtimes = np.array([run["deaths"][l]["step"] if l in run["deaths"] else horizon
                       for l in lanes], dtype=float)
    peak, end_rate = windowed_rates(run, lanes)
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = np.where(peak > 0, end_rate / peak, np.nan)
    null_p5, null_p50, peak_spread = retention_null(run, lanes)
    floor = lane_floor_vec(run, lanes)
    solved = (np.isfinite(peak) & (peak > floor)
              & np.isfinite(peak_spread) & (peak - floor > peak_spread))
    decayed = (solved & np.isfinite(retention) & np.isfinite(null_p5)
               & (retention < null_p5))
    retained = solved & ~decayed
    ret_margin = retention - null_p50
    if rs.shape[0] >= 2:
        recent = (rs[-1] - rs[-2]) / float(ends[-1] - ends[-2])
        recent = np.where(np.isfinite(recent), recent, delivered)
    else:
        recent = delivered
    seeds = [run["seeds"][l] for l in lanes]
    pct = np.percentile(delivered, [10, 25, 50, 75, 90])
    best = int(np.argmax(delivered))
    out = {
        "seeds": len(lanes), "horizon": horizon, "maxStep": max_step,
        "complete": max_step >= horizon,
        "rewardRateIQM": iqm(delivered),
        "recentRewardRateIQM": iqm(recent),
        "recentRewardRateMax": float(np.max(recent)),
        "bestSeed": seeds[best], "bestSeedRewardRate": float(delivered[best]),
        "positiveSeeds": int((delivered > 0).sum()),
        "positiveFraction": float((delivered > 0).mean()),
        "rewardRatePercentiles": dict(zip(("p10", "p25", "p50", "p75", "p90"),
                                          map(float, pct))),
        "thresholdFractions": {f"gt{t}": float((delivered > t).mean())
                               for t in (0, 1, 5, 10)},
        "divergence": {"diverged": int(died.sum()), "rate": float(died.mean())},
        "rmst": rmst(dtimes, died, horizon),
        "solvedSeeds": int(solved.sum()),
        "solvedSeedsRetained": int(retained.sum()),
        "decayedSeeds": int(decayed.sum()),
        "decayFraction": float(decayed[solved].mean()) if solved.any() else None,
        "retentionMarginP50": (float(np.nanmedian(ret_margin))
                               if np.isfinite(ret_margin).any() else None),
        "retentionFraction": (float(retained.sum() / solved.sum())
                              if solved.any() else None),
        "rewardRateBySeed": {str(s): float(r) for s, r in zip(seeds, delivered)},
        "retentionBySeed": {str(s): (float(r) if math.isfinite(r) else None)
                            for s, r in zip(seeds, retention)},
        "_vectors": {"delivered": delivered, "died": died, "dtimes": dtimes,
                     "solved": solved.astype(float), "retained": retained.astype(float),
                     "decayed": decayed.astype(float), "retMargin": ret_margin},
    }
    if run["agg"]:
        out["aggStepsPerSecond"] = run["agg"].get("aggStepsPerSecond")
    return out

def env_slices(run: dict) -> dict[str, np.ndarray]:
    sched = run["meta"].get("envSchedule") or []
    if not sched:
        return {run["meta"].get("envId", "env"): np.arange(run["tape"]["lanes"])}
    seed_lane = {s: i for i, s in enumerate(run["seeds"])}
    out: dict[str, list[int]] = {}
    for e in sched:
        for s in range(e["seedStart"], e["seedEnd"] + 1):
            if s in seed_lane:
                out.setdefault(e["envId"], []).append(seed_lane[s])
    return {k: np.array(v) for k, v in sorted(out.items())}

def iqm(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, float))
    n = len(v)
    lo, hi = n // 4, n - n // 4
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())

def rmst(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    order = np.argsort(times)
    t_s, e_s = times[order], events[order]
    area, prev_t, s = 0.0, 0.0, 1.0
    n = len(times)
    for t in np.unique(t_s[e_s == 1]):
        t = min(float(t), float(horizon))
        area += s * (t - prev_t)
        at_risk = n - np.searchsorted(t_s, t, side="left")
        deaths = int(((t_s == t) & (e_s == 1)).sum())
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
        prev_t = t
        if prev_t >= horizon:
            break
    if prev_t < horizon:
        area += s * (horizon - prev_t)
    return area

def paired(keeper: np.ndarray, cand: np.ndarray, stat, boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keeper), size=(boot, len(keeper)))
    return np.array([stat(cand[row]) - stat(keeper[row]) for row in idx])

def render(name: str, results: dict[str, dict], meta: dict) -> str:
    lines = []
    for env, r in results.items():
        if not r["complete"]:
            lines.append(f"> ⚠️ INCOMPLETE — step {r['maxStep']:,} of {r['horizon']:,} "
                         f"({r['maxStep'] / r['horizon']:.0%}). Never gate on this.")
    lines += [
        f"### {name} — B={meta['batch']} × {meta['steps']:,} steps "
        f"(ablation={meta.get('ablationLabel', 'none')})", "",
        "| env | seeds | IQM rate | P10/P50/P90 | best | %>0 | %>1 | %>SOTA(10) "
        "| recent IQM | solved | retained | decayed | retMargin | dead | RMST/h | SPS |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for env, r in results.items():
        p = r["rewardRatePercentiles"]
        tf = r["thresholdFractions"]
        sps = r.get("aggStepsPerSecond")
        lines.append(
            f"| {env} | {r['seeds']} | **{r['rewardRateIQM']:+.5g}** "
            f"| {p['p10']:+.4f}/{p['p50']:+.4f}/{p['p90']:+.4f} "
            f"| s{r['bestSeed']}:{r['bestSeedRewardRate']:+.4f} "
            f"| {tf['gt0']:.1%} | {tf['gt1']:.1%} | {tf['gt10']:.1%} "
            f"| {r['recentRewardRateIQM']:+.5g} "
            f"| {r['solvedSeeds']} | {r['solvedSeedsRetained']} "
            f"| {r['decayedSeeds']} "
            f"| {'n/a' if r['retentionMarginP50'] is None else format(r['retentionMarginP50'], '+.3f')} "
            f"| {r['divergence']['diverged']} | {r['rmst'] / r['horizon']:.3f} "
            f"| {'' if sps is None else format(sps, ',.0f')} |")
    return "\n".join(lines)

def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o

def cmd_report(args) -> int:
    run = load(args.root, args.policy)
    slices = env_slices(run)
    results = {env: analyze(run, lanes) for env, lanes in slices.items()}
    if len(slices) > 1:
        results["ALL (mixed scales — read slices)"] = analyze(
            run, np.arange(run["tape"]["lanes"]))
    print(render(args.root.name, results, run["meta"]))
    overall = results.get("ALL (mixed scales — read slices)") or next(iter(results.values()))
    summary = {
        "summarySchemaVersion": 3,
        "solvedLaw": "peak > floor + nullPeakSpread(p95-p50)",
        "retainedLaw": "solved & retention >= own stationary-null p5",
        "meta": run["meta"],
        "results": [{"policy": args.policy, **json_safe(overall)}],
        "envSlices": {env: [{"policy": args.policy, **json_safe(r)}]
                      for env, r in results.items() if not env.startswith("ALL")},
    }
    (args.root / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0

def slice_verdicts(k: dict, c: dict, n: int, horizon: int,
                   boot: int) -> tuple[list[tuple[str, bool, str]], list[str]]:
    delta = 0.64 / math.sqrt(n)
    verdicts: list[tuple[str, bool, str]] = []

    d95 = float(np.percentile(paired(k["died"], c["died"], np.mean, boot, 11), 95))
    verdicts.append(("divergence", d95 <= delta,
                     f"p {k['died'].mean():.3f}->{c['died'].mean():.3f} "
                     f"Δ95={d95:+.4f} δ={delta:.4f}"))

    di5 = float(np.percentile(paired(k["delivered"], c["delivered"], iqm, boot, 12), 5))
    ik = iqm(k["delivered"])
    margin = 0.05 * abs(ik)
    verdicts.append(("rewardRateIQM", di5 >= -margin,
                     f"IQM {ik:.5g}->{iqm(c['delivered']):.5g} ΔIQM5={di5:+.5g} "
                     f"margin={-margin:.5g}{' IMPROVEMENT' if di5 > 0 else ''}"))

    rng = np.random.default_rng(13)
    idx = rng.integers(0, n, size=(boot, n))
    dr = np.array([rmst(c["dtimes"][r_], c["died"][r_], horizon)
                   - rmst(k["dtimes"][r_], k["died"][r_], horizon) for r_ in idx])
    dr5 = float(np.percentile(dr, 5))
    verdicts.append(("rmstSurvival", dr5 >= -0.05 * horizon,
                     f"ΔRMST5={dr5:+,.0f} margin={-0.05 * horizon:,.0f}"))

    for name, key, seed in (("solvedSeeds", "solved", 30), ("retainedSeeds", "retained", 31)):
        d = float(np.percentile(paired(c[key], k[key], np.mean, boot, seed), 95))
        verdicts.append((name, d <= delta,
                         f"keeper={int(k[key].sum())} cand={int(c[key].sum())} "
                         f"Δ95={d:+.4f} δ={delta:.4f}"))

    kpos, cpos = (k["delivered"] > 0).astype(float), (c["delivered"] > 0).astype(float)
    d = float(np.percentile(paired(cpos, kpos, np.mean, boot, 33), 95))
    verdicts.append(("positiveSeedFraction", d <= delta,
                     f"{kpos.mean():.1%}->{cpos.mean():.1%} Δ95={d:+.4f} δ={delta:.4f}"))

    dsup = float(np.percentile(paired(k["decayed"], c["decayed"], np.mean, boot, 40), 95))
    sup_lines = [f"decayed {k['decayed'].mean():.3f}->{c['decayed'].mean():.3f} "
                 f"Δ95={dsup:+.4f}{' IMPROVEMENT' if dsup < 0 else ''}"]
    mfin = np.isfinite(k["retMargin"]) & np.isfinite(c["retMargin"])
    if mfin.any():
        m5 = float(np.percentile(
            paired(k["retMargin"][mfin], c["retMargin"][mfin], np.mean, boot, 41), 5))
        sup_lines.append(f"retMargin Δ5={m5:+.4f}"
                         f"{' IMPROVEMENT' if m5 > 0 else ''}")
    return verdicts, sup_lines

def cmd_gate(args) -> int:
    kh = keeper_hash()
    if kh is None:
        print("GATE ERROR: no keeper file beside the benchmark directory.")
        return 2
    if not (BENCH / kh).is_dir():
        print(f"GATE ERROR: keeper data {BENCH / kh} absent locally.")
        return 2
    kr, cr = load(BENCH / kh, args.policy), load(args.candidate, args.policy)
    for label, r in (("keeper", kr), ("candidate", cr)):
        if int(r["tape"]["ends"][-1]) < r["meta"]["steps"]:
            print(f"GATE ERROR: {label} run is INCOMPLETE "
                  f"({int(r['tape']['ends'][-1]):,} of {r['meta']['steps']:,}).")
            return 2
    if kr["meta"]["steps"] != cr["meta"]["steps"]:
        print("GATE ERROR: horizons differ.")
        return 2
    if (kr["meta"].get("ablationLabel") or "none") != (cr["meta"].get("ablationLabel") or "none"):
        print(f"GATE ERROR: ablation differs (keeper "
              f"{kr['meta'].get('ablationLabel')} vs {cr['meta'].get('ablationLabel')}).")
        return 2
    k_sl, c_sl = env_slices(kr), env_slices(cr)
    if args.env_slice:
        if args.env_slice not in k_sl or args.env_slice not in c_sl:
            print(f"GATE ERROR: --env-slice {args.env_slice} absent from a root.")
            return 2
        k_lanes, c_lanes = k_sl[args.env_slice], c_sl[args.env_slice]
    else:
        k_lanes = np.arange(kr["tape"]["lanes"])
        c_lanes = np.arange(cr["tape"]["lanes"])
    k_seeds = [kr["seeds"][l] for l in k_lanes]
    c_seeds = [cr["seeds"][l] for l in c_lanes]
    if k_seeds != c_seeds:
        print(f"GATE ERROR: seed sets differ ({len(k_seeds)} vs {len(c_seeds)}); "
              f"the paired gate requires identical seeds.")
        return 2
    n, horizon, boot = len(k_seeds), kr["meta"]["steps"], args.boot
    if args.env_slice or len(k_sl) <= 1:
        k = analyze(kr, k_lanes)["_vectors"]
        c = analyze(cr, c_lanes)["_vectors"]
        verdicts, sup_lines = slice_verdicts(k, c, n, horizon, boot)
        print(f"fleet gate: N={n} horizon={horizon:,} policy={args.policy} boot={boot}"
              + (f" [env={args.env_slice}]" if args.env_slice else ""))
        ok = True
        width = max(len(v[0]) for v in verdicts)
        for name, passed, detail in verdicts:
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:{width}s} {detail}")
        print(f"  [info] retention superiority: {'; '.join(sup_lines)}")
        print("PROMOTE" if ok else "REJECT")
        return 0 if ok else 1

    print(f"fleet gate: horizon={horizon:,} policy={args.policy} boot={boot} "
          f"verdict=per-env ({len(k_sl)} slices, all must pass)")
    ok = True
    for env in k_sl:
        if env not in c_sl or [kr["seeds"][l] for l in k_sl[env]] !=\
                              [cr["seeds"][l] for l in c_sl[env]]:
            print(f"  [FAIL] {env:28s} seed sets differ between roots")
            ok = False
            continue
        ek = analyze(kr, k_sl[env])["_vectors"]
        ec = analyze(cr, c_sl[env])["_vectors"]
        verdicts, sup_lines = slice_verdicts(ek, ec, len(k_sl[env]), horizon, boot)
        fleet_only = ("solvedSeeds", "retainedSeeds", "positiveSeedFraction")
        verdicts = [v for v in verdicts if v[0] not in fleet_only]
        failed = [name for name, passed, _ in verdicts if not passed]
        ok &= not failed
        ki, ci = iqm(ek["delivered"]), iqm(ec["delivered"])
        rel = f"{(ci - ki) / abs(ki):+.1%}" if ki else "n/a"
        print(f"  [{'FAIL' if failed else 'PASS'}] {env:28s} n={len(k_sl[env]):<4d} "
              f"IQM {ki:+.5g} -> {ci:+.5g} ({rel})"
              + (f" failed={','.join(failed)}" if failed else ""))
        for name, passed, detail in verdicts:
            if not passed:
                print(f"         {name}: {detail}")
        for line in sup_lines:
            if "IMPROVEMENT" in line:
                print(f"         superiority: {line}")
    kf = analyze(kr, np.arange(kr["tape"]["lanes"]))["_vectors"]
    cf = analyze(cr, np.arange(cr["tape"]["lanes"]))["_vectors"]
    fleet_v, fleet_sup = slice_verdicts(kf, cf, kr["tape"]["lanes"], horizon, boot)
    for name, passed, detail in fleet_v:
        if name in ("divergence", "solvedSeeds", "retainedSeeds",
                    "positiveSeedFraction"):
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] fleet.{name} {detail}")
    print(f"  [info] fleet retention superiority: {'; '.join(fleet_sup)}")
    print("PROMOTE" if ok else "REJECT")
    return 0 if ok else 1

def keeper_hash() -> str | None:
    try:
        text = (REPO / "keeper").read_text().strip()
        return text or None
    except OSError:
        return None

def write_keeper(name: str) -> None:
    (REPO / "keeper").write_text(name + "\n")

def cmd_run(args) -> int:
    name = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    root = BENCH / name
    log = BENCH / f"{name}.log"
    kh = keeper_hash()
    if kh == name and root.exists():
        print(f"REFUSED: {name} is the keeper — its data is not overwritten; "
              f"delete .oak-benchmarks/{name} by hand if a re-run is truly meant.")
        return 2
    keeper = BENCH / kh if kh and kh != name and (BENCH / kh).is_dir() else None
    BENCH.mkdir(exist_ok=True)
    if keeper and (keeper / "meta.json").exists():
        kabl = json.loads((keeper / "meta.json").read_text()).get("ablationLabel") or "none"
        if kabl != os.environ.get("OAK_ABLATE", "none"):
            print(f"CONFIG ERROR: OAK_ABLATE={os.environ.get('OAK_ABLATE', 'none')} "
                  f"!= keeper ablation '{kabl}' — the gate would compare different configs.")
            return 2
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    log.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "job",
           str(args.steps), name]
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True,
                             cwd=REPO)
    print(f"DETACHED PID {p.pid} (new session — survives harness compaction/idle)")
    print(f"  log : {log}")
    print(f"  done: {root}/.done")
    return 0

def cmd_job(args) -> int:
    root = BENCH / args.name
    env = dict(os.environ,
               OAK_BENCH_POLICIES=os.environ.get("OAK_BENCH_POLICIES", "oak"),
               OAK_BENCH_STEPS=str(args.steps),
               OAK_BENCH_SEED_COUNT=os.environ.get("OAK_BENCH_SEED_COUNT", "1408"),
               OAK_RECORD_ACTIONS=os.environ.get("OAK_RECORD_ACTIONS", "1"),
               OAK_RECORD_FORMAT=os.environ.get("OAK_RECORD_FORMAT", "4"),
               OAK_FLEET_TIMING="1", OAK_METRICS_STORAGE=str(root))
    rc = subprocess.run(["caffeinate", "-dimsu", str(REPO / ".build/release/OaK"),
                         "benchmark"], env=env, cwd=REPO).returncode
    print(f"[fleet] benchmark exit={rc}", flush=True)
    me = [sys.executable, str(Path(__file__).resolve())]
    subprocess.run(me + ["report", str(root)], cwd=REPO)
    prior = keeper_hash()
    if prior and prior != args.name and (BENCH / prior).is_dir():
        if subprocess.run(me + ["gate", str(root)], cwd=REPO).returncode == 0:
            write_keeper(args.name)
            print(f"[fleet] keeper {args.name}", flush=True)
        else:
            print(f"[fleet] keeper stays {prior}", flush=True)
    elif rc == 0:
        write_keeper(args.name)
        print(f"[fleet] keeper {args.name}", flush=True)
    (root / ".done").touch()
    print("[fleet] DONE — .done written.", flush=True)
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="detached benchmark -> report -> gate -> .done")
    rp.add_argument("steps", type=int)

    jp = sub.add_parser("job")
    jp.add_argument("steps", type=int)
    jp.add_argument("name")

    pp = sub.add_parser("report", help="population view from the tape (in-flight OK)")
    pp.add_argument("root", type=Path)
    pp.add_argument("--policy", default="oak")

    gp = sub.add_parser("gate", help="paired promotion law: PROMOTE/REJECT")
    gp.add_argument("candidate", type=Path)
    gp.add_argument("--policy", default="oak")
    gp.add_argument("--boot", type=int, default=10_000)
    gp.add_argument("--env-slice", default=None, metavar="ENV_ID")

    args = ap.parse_args()
    return {"run": cmd_run, "job": cmd_job, "report": cmd_report,
            "gate": cmd_gate}[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
