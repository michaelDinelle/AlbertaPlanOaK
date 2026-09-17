#!/usr/bin/env python3
import filecmp
import hashlib
import json
import os
import subprocess
import sys
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

def read(root, expected=None):
    meta = json.loads((root / "meta.json").read_text())
    ends = array("I", (root / "ends.u32").read_bytes())
    size = (root / "series/rewardSum.f32").stat().st_size
    if "finishedAt" not in meta or not ends or ends[-1] != meta["steps"] or size != 4 * len(ends) * meta["batch"]:
        raise SystemExit(f"Incomplete recording: {root}")
    for entry in meta["envSchedule"]:
        lanes = entry["seedEnd"] - entry["seedStart"] + 1
        if (root / "rewards" / f"{entry['envId']}.f32").stat().st_size != 4 * meta["steps"] * lanes:
            raise SystemExit(f"Incomplete rewards: {root}/{entry['envId']}")
    if expected and any(meta[key] != expected[key] for key in SCHEDULE):
        raise SystemExit(f"Recording does not match the reference: {root}")
    return meta

def record(policy, root, steps):
    print(f"Recording {policy} {root.name}: {schedule['batch']} lanes x {steps:,} steps", flush=True)
    subprocess.run(["caffeinate", "-i", REPO / ".build/release/OaK", policy, str(steps), root],
                   check=True, cwd=REPO)

if "OAK_MEGA_STAGE_MASK" in os.environ:
    raise SystemExit("A recording requires no timing instruments")
subprocess.run([REPO / "setup.sh"] + sys.argv[1:], cwd=REPO, check=True)
geometry = (REPO / ".build/OakKernel.geometry").read_text().strip()
schedule = json.loads((BENCH / "schedule.json").read_text())
steps = schedule["steps"]

baselines = BENCH / "baselines" / hashlib.sha256(
    json.dumps({k: schedule[k] for k in SCHEDULE}, sort_keys=True).encode() + digest(ENV_SOURCES).encode()
).hexdigest()[:40]
for policy in ("ppo", "random"):
    if not (baselines / policy / "meta.json").exists():
        record(policy, baselines / policy, steps)
    read(baselines / policy, schedule)

candidate = BENCH / digest((
    "Sources", "Package.swift", "setup.sh", "run.py", ".build/release/OaK", ".build/OakKernel.metallib",
))[:40]
if not candidate.exists():
    record("oak", candidate, steps)
meta = read(candidate, schedule)
(candidate / "baselines").mkdir(exist_ok=True)
for policy in ("ppo", "random"):
    link = candidate / "baselines" / policy
    if not link.is_symlink():
        link.symlink_to(os.path.relpath(baselines / policy, link.parent))
    if link.resolve() != (baselines / policy).resolve():
        raise SystemExit(f"Incorrect baseline link: {link}")

reference = BENCH / "best" / geometry
if not reference.exists():
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(candidate.name + "\n")
    print(f"Steps/s: {meta['stepsPerSecond']:,.0f}")
    raise SystemExit(f"ADOPT {geometry}: {candidate.name}")

best_root = BENCH / reference.read_text().strip()
best = read(best_root, schedule)
different = [entry["envId"] for entry in schedule["envSchedule"]
             if not filecmp.cmp(candidate / "rewards" / f"{entry['envId']}.f32",
                                best_root / "rewards" / f"{entry['envId']}.f32", shallow=False)]
print(f"Changed reward streams: {different}")
print(f"Steps/s: {meta['stepsPerSecond']:,.0f}; best: {best['stepsPerSecond']:,.0f}")
if different or not meta["stepsPerSecond"] >= best["stepsPerSecond"]:
    raise SystemExit(f"REJECT {geometry}: best stays {best_root.name}")
reference.write_text(candidate.name + "\n")
print(f"PROMOTE {geometry}: {candidate.name}")
