#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
        raise SystemExit(f"{root}: no metrics.{policy}.oakt")
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
    return {"meta": meta, "tape": tape, "deaths": deaths, "agg": agg,
            "seeds": meta["seeds"], "root": root}


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


def env_slices(run: dict) -> dict[str, np.ndarray]:
    seed_lane = {s: i for i, s in enumerate(run["seeds"])}
    out: dict[str, list[int]] = {}
    for e in run["meta"]["envSchedule"]:
        for s in range(e["seedStart"], e["seedEnd"] + 1):
            if s in seed_lane:
                out.setdefault(e["envId"], []).append(seed_lane[s])
    return {k: np.array(v) for k, v in sorted(out.items())}


def lane_floor_vec(run: dict, lanes: np.ndarray) -> np.ndarray:
    fl = floors()
    ids = [""] * run["tape"]["lanes"]
    for env, sl in env_slices(run).items():
        for l in sl:
            ids[int(l)] = env
    return np.array([fl.get(ids[int(l)], {}).get("floor", 0.0) for l in lanes], float)


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
    overall = (analyze(run, np.arange(run["tape"]["lanes"])) if len(results) > 1
               else next(iter(results.values())))
    summary = {
        "summarySchemaVersion": 3,
        "solvedLaw": "peak > floor + nullPeakSpread(p95-p50)",
        "retainedLaw": "solved & retention >= own stationary-null p5",
        "meta": run["meta"],
        "results": [{"policy": args.policy, **json_safe(overall)}],
        "envSlices": {env: [{"policy": args.policy, **json_safe(r)}]
                      for env, r in results.items()},
    }
    out = args.root / "summary.json"
    out.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    print(out)
    return 0


def slice_verdicts(k: dict, c: dict, n: int, horizon: int, boot: int) -> list[tuple[str, bool]]:
    delta = 0.64 / math.sqrt(n)
    d95 = float(np.percentile(paired(k["died"], c["died"], np.mean, boot, 11), 95))
    di5 = float(np.percentile(paired(k["delivered"], c["delivered"], iqm, boot, 12), 5))
    rng = np.random.default_rng(13)
    idx = rng.integers(0, n, size=(boot, n))
    dr5 = float(np.percentile(
        [rmst(c["dtimes"][r], c["died"][r], horizon)
         - rmst(k["dtimes"][r], k["died"][r], horizon) for r in idx], 5))
    kpos = (k["delivered"] > 0).astype(float)
    cpos = (c["delivered"] > 0).astype(float)
    out = [
        ("divergence", d95 <= delta),
        ("rewardRateIQM", di5 >= -0.05 * abs(iqm(k["delivered"]))),
        ("rmstSurvival", dr5 >= -0.05 * horizon),
    ]
    for name, key, seed in (("solvedSeeds", "solved", 30), ("retainedSeeds", "retained", 31)):
        d = float(np.percentile(paired(c[key], k[key], np.mean, boot, seed), 95))
        out.append((name, d <= delta))
    d = float(np.percentile(paired(cpos, kpos, np.mean, boot, 33), 95))
    out.append(("positiveSeedFraction", d <= delta))
    return out


def _fail(msg: str) -> int:
    print(msg)
    return 2


def cmd_gate(args) -> int:
    kh = keeper_hash()
    if kh is None or not (BENCH / kh).is_dir():
        return _fail("GATE ERROR: keeper missing")
    kr, cr = load(BENCH / kh, args.policy), load(args.candidate, args.policy)
    for label, r in (("keeper", kr), ("candidate", cr)):
        if int(r["tape"]["ends"][-1]) < r["meta"]["steps"]:
            return _fail(f"GATE ERROR: {label} incomplete")
    if kr["meta"]["steps"] != cr["meta"]["steps"]:
        return _fail("GATE ERROR: horizons differ")
    if (kr["meta"].get("ablationLabel") or "none") != (cr["meta"].get("ablationLabel") or "none"):
        return _fail("GATE ERROR: ablation differs")
    k_sl, c_sl = env_slices(kr), env_slices(cr)
    if args.env_slice:
        if args.env_slice not in k_sl or args.env_slice not in c_sl:
            return _fail(f"GATE ERROR: missing slice {args.env_slice}")
        k_lanes, c_lanes = k_sl[args.env_slice], c_sl[args.env_slice]
    else:
        k_lanes = np.arange(kr["tape"]["lanes"])
        c_lanes = np.arange(cr["tape"]["lanes"])
    if [kr["seeds"][l] for l in k_lanes] != [cr["seeds"][l] for l in c_lanes]:
        return _fail("GATE ERROR: seed sets differ")
    n, horizon, boot = len(k_lanes), kr["meta"]["steps"], args.boot

    def apply(tag: str, verdicts: list[tuple[str, bool]]) -> bool:
        ok = True
        for name, passed in verdicts:
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {tag}{name}")
        return ok

    if args.env_slice or len(k_sl) <= 1:
        ok = apply("", slice_verdicts(analyze(kr, k_lanes)["_vectors"],
                                      analyze(cr, c_lanes)["_vectors"],
                                      n, horizon, boot))
    else:
        ok = True
        skip = {"solvedSeeds", "retainedSeeds", "positiveSeedFraction"}
        for env in k_sl:
            if env not in c_sl or [kr["seeds"][l] for l in k_sl[env]] != \
                                  [cr["seeds"][l] for l in c_sl[env]]:
                print(f"  [FAIL] {env} seeds")
                ok = False
                continue
            v = [x for x in slice_verdicts(analyze(kr, k_sl[env])["_vectors"],
                                           analyze(cr, c_sl[env])["_vectors"],
                                           len(k_sl[env]), horizon, boot)
                 if x[0] not in skip]
            ok &= apply(f"{env} ", v)
        fleet = {"divergence"} | skip
        fv = slice_verdicts(analyze(kr, np.arange(kr["tape"]["lanes"]))["_vectors"],
                            analyze(cr, np.arange(cr["tape"]["lanes"]))["_vectors"],
                            kr["tape"]["lanes"], horizon, boot)
        ok &= apply("fleet.", [x for x in fv if x[0] in fleet])
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


def keeper_meta() -> dict:
    kh = keeper_hash()
    p = BENCH / kh / "meta.json" if kh else None
    return json.loads(p.read_text()) if p and p.exists() else {}


def cmd_run(args) -> int:
    name = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    root, log = BENCH / name, BENCH / f"{name}.log"
    kh = keeper_hash()
    if kh == name and root.exists():
        print(f"REFUSED: {name} is the keeper")
        return 2
    km = keeper_meta()
    if km and (km.get("ablationLabel") or "none") != os.environ.get("OAK_ABLATE", "none"):
        print("CONFIG ERROR: OAK_ABLATE != keeper ablation")
        return 2
    BENCH.mkdir(exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    log.unlink(missing_ok=True)
    with open(log, "w") as lf:
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "job", str(args.steps), name],
            stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=REPO)
    print(f"DETACHED PID {p.pid}\n  {log}\n  {root}/.done")
    return 0


def cmd_job(args) -> int:
    root = BENCH / args.name
    km = keeper_meta()
    envs = os.environ.get("OAK_BENCH_ENVS") or ",".join(
        f"{e['envId']}:{e['seedEnd'] - e['seedStart'] + 1}"
        for e in km.get("envSchedule") or ())
    if not envs:
        raise SystemExit("OAK_BENCH_ENVS required")
    env = dict(os.environ,
               OAK_BENCH_STEPS=str(args.steps),
               OAK_BENCH_SEED_COUNT=os.environ.get(
                   "OAK_BENCH_SEED_COUNT", str(km.get("batch", 1024))),
               OAK_BENCH_ENVS=envs,
               OAK_RECORD_ACTIONS=os.environ.get("OAK_RECORD_ACTIONS", "1"),
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
    print("[fleet] DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("steps", type=int)
    jp = sub.add_parser("job")
    jp.add_argument("steps", type=int)
    jp.add_argument("name")
    pp = sub.add_parser("report")
    pp.add_argument("root", type=Path)
    pp.add_argument("--policy", default="oak")
    gp = sub.add_parser("gate")
    gp.add_argument("candidate", type=Path)
    gp.add_argument("--policy", default="oak")
    gp.add_argument("--boot", type=int, default=10_000)
    gp.add_argument("--env-slice", default=None)
    args = ap.parse_args()
    return {"run": cmd_run, "job": cmd_job, "report": cmd_report,
            "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
