#!/usr/bin/env python3
import filecmp, hashlib, json, math, os, subprocess, sys
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
OAK = REPO / ".build/release/OaK"
STEPS = 1_000_000
ENV_SOURCES = ("Sources/EnvCore.metal", "Sources/Env.metal", "Sources/Baseline.metal", "Sources/envs", "Sources/OaK/Resources")
CANDIDATE_SOURCES = ("Sources", "Package.swift", "setup.sh", "fleet.py", ".build/release/OaK", ".build/OakKernel.metallib")

def files(entries):
    return [p for e in entries for base in [REPO / e]
            for p in (sorted(q for q in base.rglob("*") if q.is_file()) if base.is_dir() else [base])]

def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(REPO)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()

def meta_of(root): return json.loads((root / "meta.json").read_text())

def env_lanes(meta):
    lanes, at = {}, 0
    for e in meta["envSchedule"]:
        n = e["seedEnd"] - e["seedStart"] + 1
        lanes[e["envId"]], at = range(at, at + n), at + n
    return lanes

def lane_means(root):
    meta = meta_of(root)
    col = array("f", (root / "series/rewardSum.f32").read_bytes())
    return [v / meta["steps"] for v in col[-len(meta["seeds"]):]]

def env_mean(means, lanes): return sum(means[l] for l in lanes) / len(lanes)

def wins(a, b, lanes): return sum(0.5 if a[l] == b[l] else a[l] > b[l] for l in lanes) / len(lanes)

def record(policy, root):
    if (root / "meta.json").is_file() and "finishedAt" in meta_of(root): print(f"{policy}: reusing {root.relative_to(BENCH)}")
    elif root.exists(): raise SystemExit(f"{root}: incomplete recording; delete it to re-record")
    else: subprocess.run(["caffeinate", "-i", OAK, policy, str(STEPS), root], cwd=REPO, check=True)

def main():
    subprocess.run([REPO / "setup.sh"], cwd=REPO, check=True)
    best_id = (REPO / "best").read_text().strip()
    cand_id = digest(files(CANDIDATE_SOURCES))[:40]
    if cand_id == best_id: raise SystemExit(f"{cand_id} is already the best")
    cand = BENCH / cand_id
    record("oak", cand)
    meta = meta_of(cand)
    base_id = hashlib.sha256(json.dumps({k: meta[k] for k in ("envSchedule", "seeds", "steps", "batch")}, sort_keys=True).encode()
                             + digest(files(ENV_SOURCES)).encode()).hexdigest()[:40]
    runs = {"oak": cand}
    for policy in ("ppo", "random"):
        runs[policy] = BENCH / "baselines" / base_id / policy
        record(policy, runs[policy])
        link = cand / "baselines" / policy
        link.parent.mkdir(exist_ok=True)
        if not link.is_symlink(): link.symlink_to(os.path.relpath(runs[policy], link.parent))
    means = {p: lane_means(r) for p, r in runs.items()}
    lanes = env_lanes(meta)
    print("| Environment | OaK | PPO | Random | OaK vs PPO wins |\n|---|---:|---:|---:|---:|")
    for env, ls in lanes.items():
        print(f"| {env} | {env_mean(means['oak'], ls):.6g} | {env_mean(means['ppo'], ls):.6g} | {env_mean(means['random'], ls):.6g} | {wins(means['oak'], means['ppo'], ls):.2%} |")
    if not best_id:
        (REPO / "best").write_text(cand_id + "\n"); return print(f"PROMOTE: bootstrap best {cand_id}")
    best = BENCH / best_id
    bmeta = meta_of(best)
    if any(meta[k] != bmeta[k] for k in ("envSchedule", "seeds", "steps", "batch", "preset")): raise SystemExit("candidate schedule or preset differs from the best")
    identical = all(filecmp.cmp(cand / "rewards" / f"{e}.f32", best / "rewards" / f"{e}.f32", shallow=False) for e in lanes)
    print(f"rewards bitwise identical: {'yes' if identical else 'no'}")
    bmeans = lane_means(best)
    worse = [e for e, ls in lanes.items() if not wins(means["oak"], bmeans, ls) >= 0.5 - 1.645 * math.sqrt(0.25 / len(ls))]
    same_hw = meta["hardwareKey"] == bmeta["hardwareKey"]
    print(f"steps/s {meta['stepsPerSecond']:,.0f} vs best {bmeta['stepsPerSecond']:,.0f} ({'same' if same_hw else 'different'} hardware)")
    if not worse and same_hw and meta["stepsPerSecond"] >= bmeta["stepsPerSecond"]:
        (REPO / "best").write_text(cand_id + "\n"); return print(f"PROMOTE {cand_id}")
    print(f"REJECT: best stays {best_id}" + (f"; regressed in {', '.join(worse)}" if worse else ""))
    sys.exit(1)

if __name__ == "__main__": main()
