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
SOLVED_THRESHOLD = 0.1
RETAINED_FRACTION = 0.5
NF_PHASES = ("nfEncode", "nfLearn", "nfPlan", "nfTdPsi")

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

def windowed_rates(run: dict, lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = run["tape"]["ends"]
    rs = series(run, "rewardSum")[:, lanes]
    win = max(run["meta"]["steps"] // 5, 1)
    j = np.searchsorted(ends, ends - win, side="right") - 1
    valid = j >= 0
    if not valid.any():
        n = len(lanes)
        return np.full(n, np.nan), np.full(n, np.nan)
    i = np.nonzero(valid)[0]
    rates = (rs[i] - rs[j[i]]) / (ends[i] - ends[j[i]])[:, None]
    peak = np.nanmax(np.where(np.isfinite(rates), rates, -np.inf), axis=0)
    peak[~np.isfinite(peak)] = np.nan
    fin_mask = np.isfinite(rates)
    li = rates.shape[0] - 1 - np.argmax(fin_mask[::-1], axis=0)
    end = np.where(fin_mask.any(axis=0), rates[li, np.arange(rates.shape[1])], np.nan)
    return peak, end

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
    solved = np.isfinite(peak) & (peak > SOLVED_THRESHOLD)
    retained = solved & np.isfinite(retention) & (retention >= RETAINED_FRACTION)
    if rs.shape[0] >= 2:
        recent = (rs[-1] - rs[-2]) / float(ends[-1] - ends[-2])
        recent = np.where(np.isfinite(recent), recent, delivered)
    else:
        recent = delivered
    nf = {ph: np.nan_to_num(last_finite(series(run, ph)[:, lanes])) > 0
          for ph in NF_PHASES if ph in run["tape"]["fields"]}
    seeds = [run["seeds"][l] for l in lanes]
    pct = np.percentile(delivered, [10, 25, 50, 75, 90])
    best = int(np.argmax(delivered))
    ret_ok = np.isfinite(retention)
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
        "retentionFraction": float((retention[ret_ok] >= RETAINED_FRACTION).mean())
                             if ret_ok.any() else None,
        "rewardRateBySeed": {str(s): float(r) for s, r in zip(seeds, delivered)},
        "retentionBySeed": {str(s): (float(r) if math.isfinite(r) else None)
                            for s, r in zip(seeds, retention)},
        "_vectors": {"delivered": delivered, "died": died, "dtimes": dtimes,
                     "solved": solved.astype(float), "retained": retained.astype(float),
                     "nf": nf},
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
        "| recent IQM | solved | retained | dead | RMST/h | SPS |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        "summarySchemaVersion": 2, "solvedThreshold": SOLVED_THRESHOLD,
        "meta": run["meta"],
        "results": [{"policy": args.policy, **json_safe(overall)}],
        "envSlices": {env: [{"policy": args.policy, **json_safe(r)}]
                      for env, r in results.items() if not env.startswith("ALL")},
    }
    (args.root / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0

def cmd_gate(args) -> int:
    kr, cr = load(args.keeper, args.policy), load(args.candidate, args.policy)
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
    k, c = analyze(kr, k_lanes)["_vectors"], analyze(cr, c_lanes)["_vectors"]
    n, horizon, boot = len(k_seeds), kr["meta"]["steps"], args.boot
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

    nf_over, nf_parts = [], []
    for i, ph in enumerate(NF_PHASES):
        if ph not in k["nf"] or ph not in c["nf"]:
            continue
        kp, cp = k["nf"][ph].astype(float), c["nf"][ph].astype(float)
        d = float(np.percentile(paired(kp, cp, np.mean, boot, 20 + i), 95))
        if d > delta:
            nf_over.append(ph)
        nf_parts.append(f"{ph[2:].lower()} {kp.mean():.3f}->{cp.mean():.3f} Δ95={d:+.4f}")
    verdicts.append(("nfPrevalence", not nf_over,
                     "; ".join(nf_parts) + (f" OVER={nf_over}" if nf_over else "")))

    for name, key, seed in (("solvedSeeds", "solved", 30), ("retainedSeeds", "retained", 31)):
        d = float(np.percentile(paired(c[key], k[key], np.mean, boot, seed), 95))
        verdicts.append((name, d <= delta,
                         f"keeper={int(k[key].sum())} cand={int(c[key].sum())} "
                         f"Δ95={d:+.4f} δ={delta:.4f}"))

    kpos, cpos = (k["delivered"] > 0).astype(float), (c["delivered"] > 0).astype(float)
    d = float(np.percentile(paired(cpos, kpos, np.mean, boot, 33), 95))
    verdicts.append(("positiveSeedFraction", d <= delta,
                     f"{kpos.mean():.1%}->{cpos.mean():.1%} Δ95={d:+.4f} δ={delta:.4f}"))

    print(f"fleet gate: N={n} horizon={horizon:,} policy={args.policy} boot={boot}"
          + (f" [env={args.env_slice}]" if args.env_slice else ""))
    ok = True
    width = max(len(v[0]) for v in verdicts)
    for name, passed, detail in verdicts:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:{width}s} {detail}")
    if not args.env_slice and len(k_sl) > 1:
        print("  [info] per-env ΔIQM (diagnostic — verdicts via gate --env-slice):")
        for env in k_sl:
            if env not in c_sl or [kr["seeds"][l] for l in k_sl[env]] !=\
                                  [cr["seeds"][l] for l in c_sl[env]]:
                continue
            kd = analyze(kr, k_sl[env])["_vectors"]["delivered"]
            cd = analyze(cr, c_sl[env])["_vectors"]["delivered"]
            ki, ci = iqm(kd), iqm(cd)
            rel = f"{(ci - ki) / abs(ki):+.1%}" if ki else "n/a"
            print(f"         {env:28s} n={len(kd):<5d} IQM {ki:+.5g} -> {ci:+.5g} ({rel})")
    print("PROMOTE" if ok else "REJECT")
    return 0 if ok else 1

def cmd_run(args) -> int:
    root = BENCH / args.name
    log = BENCH / f"{args.name}.log"
    BENCH.mkdir(exist_ok=True)
    if args.keeper and (args.keeper / "meta.json").exists():
        kabl = json.loads((args.keeper / "meta.json").read_text()).get("ablationLabel") or "none"
        if kabl != os.environ.get("OAK_ABLATE", "none"):
            print(f"CONFIG ERROR: OAK_ABLATE={os.environ.get('OAK_ABLATE', 'none')} "
                  f"!= keeper ablation '{kabl}' — the gate would compare different configs.")
            return 2
    if args.role == "candidate":
        rp = BENCH / "ROLES.json"
        roles = json.loads(rp.read_text()) if rp.exists() else {}
        roles["candidate"] = args.name
        roles["rolesSchemaVersion"] = 2
        rp.write_text(json.dumps(roles, indent=1))
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    log.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "job",
           str(args.steps), args.name, str(args.keeper or "")]
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
    if args.keeper and args.keeper != "None":
        subprocess.run(me + ["gate", args.keeper, str(root)], cwd=REPO)
    (root / ".done").touch()
    print("[fleet] DONE — .done written.", flush=True)
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="detached benchmark -> report -> gate -> .done")
    rp.add_argument("steps", type=int)
    rp.add_argument("name")
    rp.add_argument("keeper", type=Path, nargs="?", default=None)
    rp.add_argument("--role", default="candidate", choices=("candidate", "probe"))

    jp = sub.add_parser("job")
    jp.add_argument("steps", type=int)
    jp.add_argument("name")
    jp.add_argument("keeper", nargs="?", default="")

    pp = sub.add_parser("report", help="population view from the tape (in-flight OK)")
    pp.add_argument("root", type=Path)
    pp.add_argument("--policy", default="oak")

    gp = sub.add_parser("gate", help="paired promotion law: PROMOTE/REJECT")
    gp.add_argument("keeper", type=Path)
    gp.add_argument("candidate", type=Path)
    gp.add_argument("--policy", default="oak")
    gp.add_argument("--boot", type=int, default=10_000)
    gp.add_argument("--env-slice", default=None, metavar="ENV_ID")

    args = ap.parse_args()
    return {"run": cmd_run, "job": cmd_job, "report": cmd_report,
            "gate": cmd_gate}[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
