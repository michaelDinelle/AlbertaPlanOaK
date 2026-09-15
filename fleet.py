#!/usr/bin/env python3
import filecmp
import hashlib
import json
import os
import subprocess
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
SCHEDULE = ("envSchedule", "seeds", "steps", "batch")
ENV_SOURCES = (
    "Sources/EnvCore.metal", "Sources/Env.metal", "Sources/Random.metal",
    "Sources/PPO.metal", "Sources/Fleet.metal", "Sources/envs", "Sources/OaK/Resources",
)

def digest(entries):
    result = hashlib.sha256()
    for entry in entries:
        path = REPO / entry
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for path in files:
            result.update(str(path.relative_to(REPO)).encode() + b"\0" + path.read_bytes() + b"\0")
    return result.hexdigest()

def read(root, expected=None, keys=SCHEDULE):
    meta = json.loads((root / "meta.json").read_text())
    ends = array("I", (root / "ends.u32").read_bytes())
    size = (root / "series/rewardSum.f32").stat().st_size
    if "finishedAt" not in meta or not ends or ends[-1] != meta["steps"] or size != 4 * len(ends) * meta["batch"]:
        raise SystemExit(f"Incomplete recording: {root}")
    for entry in meta["envSchedule"]:
        lanes = entry["seedEnd"] - entry["seedStart"] + 1
        if (root / "rewards" / f"{entry['envId']}.f32").stat().st_size != 4 * meta["steps"] * lanes:
            raise SystemExit(f"Incomplete rewards: {root}/{entry['envId']}")
    if expected and any(meta[key] != expected[key] for key in keys):
        raise SystemExit(f"Recording does not match the best: {root}")
    return meta

best_root = BENCH / (REPO / "best").read_text().strip()
best = read(best_root)
if best["steps"] != 1_000_000 or "OAK_MEGA_STAGE_MASK" in os.environ:
    raise SystemExit("A fleet comparison requires one million steps and no timing instruments")
subprocess.run([REPO / "setup.sh", "--preset", best["preset"]], cwd=REPO, check=True)
schedule = json.dumps({key: best[key] for key in SCHEDULE}, sort_keys=True).encode()
baseline_id = hashlib.sha256(schedule + digest(ENV_SOURCES).encode()).hexdigest()[:40]
baselines = BENCH / "baselines" / baseline_id
for policy in ("ppo", "random"):
    read(baselines / policy, best)
candidate = BENCH / digest((
    "Sources", "Package.swift", "setup.sh", "fleet.py", ".build/release/OaK", ".build/OakKernel.metallib",
))[:40]
if not candidate.exists():
    print(f"Recording {candidate.name}: {best['batch']} lanes × {best['steps']:,} steps", flush=True)
    subprocess.run(["caffeinate", "-i", REPO / ".build/release/OaK", "oak", str(best["steps"]), candidate], check=True, cwd=REPO)
meta = read(candidate, best, SCHEDULE + ("preset",))
(candidate / "baselines").mkdir(exist_ok=True)
for policy in ("ppo", "random"):
    link = candidate / "baselines" / policy
    if not link.is_symlink():
        link.symlink_to(os.path.relpath(baselines / policy, link.parent))
    if link.resolve() != (baselines / policy).resolve():
        raise SystemExit(f"Incorrect baseline link: {link}")
different = []
for entry in best["envSchedule"]:
    file = Path("rewards") / f"{entry['envId']}.f32"
    if not filecmp.cmp(candidate / file, best_root / file, shallow=False):
        different.append(entry["envId"])
print(f"Changed reward streams: {different}")
print(f"Steps/s: {meta['stepsPerSecond']:,.0f}; best: {best['stepsPerSecond']:,.0f}")
if different or not meta["stepsPerSecond"] >= best["stepsPerSecond"]:
    raise SystemExit(f"REJECT: best stays {best_root.name}")
(REPO / "best").write_text(candidate.name + "\n")
print(f"PROMOTE: {candidate.name}")
