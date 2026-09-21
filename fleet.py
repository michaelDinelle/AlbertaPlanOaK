#!/usr/bin/env python3
import filecmp
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
BEST = REPO / "best"
ENGINE = REPO / ".build/oak"
STEPS = 1_000_000
TERMINAL = 100_000
MARGIN = 0.01
ALPHA = 0.05
BOOTSTRAP = 10_000
SEED_POWER = 7.85
SEED_EFFECT = 0.10
SEED_FLOOR, SEED_CAP = 16, 64
PROBE_STEPS = 20_000
PROBE_BLOCKS = 3
POLICIES = ("random", "ppo", "oracle")
POLICY_SOURCES = {"random": ("Sources/random.c",), "ppo": ("Sources/ppo.c",), "oracle": ()}
ENV_LAYER = ("Sources/env.h", "Sources/rng.h", "Sources/portable.h")
SNAPSHOT = ("Sources", "catalog.py", "setup.sh")
ENV_ROW = re.compile(r'\s*X\(\s*\w+\s*,\s*(\w+)\s*,.*?"([^"]+)"')


def digest(entries):
    result = hashlib.sha256()
    for entry in sorted(entries):
        path = REPO / entry
        files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")) if path.is_dir() else [path]
        for file in files:
            result.update(str(file.relative_to(REPO)).encode() + b"\0" + file.read_bytes() + b"\0")
    return result.hexdigest()


def env_digests():
    table = json.loads(subprocess.run([ENGINE, "envs"], capture_output=True, text=True, check=True).stdout)
    rows = {}
    for line in (REPO / "Sources/env.c").read_text().splitlines():
        row = ENV_ROW.match(line)
        if row:
            rows[row.group(2)] = (row.group(1), line.strip())
    digests = {}
    for entry in table:
        name = entry["envId"]
        if name not in rows:
            raise SystemExit(f"{name} has no row in the environment table")
        family, row = rows[name]
        hook = re.compile(rf"\b{family}_step\(")
        owners = [p for p in sorted((REPO / "Sources/envs").glob("*.c")) if hook.search(p.read_text())]
        if len(owners) != 1:
            raise SystemExit(f"{name} draws its behaviour from {len(owners)} environment files")
        digests[name] = hashlib.sha256((digest(ENV_LAYER + (str(owners[0].relative_to(REPO)),)) + row).encode()).hexdigest()
    return digests


class Recording:
    def __init__(self, root):
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text())
        if "finishedAt" not in self.meta:
            raise SystemExit(f"Incomplete recording: {root}")
        self.steps = self.meta["steps"]
        self.lanes = self.meta["batch"]
        self.ends = array("I", (root / "ends.u32").read_bytes())
        sums = array("f", (root / "series/rewardSum.f32").read_bytes())
        if not self.ends or self.ends[-1] != self.steps or len(sums) != len(self.ends) * self.lanes:
            raise SystemExit(f"Incomplete recording: {root}")
        self.hash = array("Q", (root / "hash.u64").read_bytes())
        self.health = array("I", (root / "health.u32").read_bytes())
        if len(self.hash) != self.lanes or len(self.health) != self.lanes:
            raise SystemExit(f"Incomplete recording: {root}")
        start = max(i for i, end in enumerate(self.ends) if end <= self.steps - TERMINAL)
        span = float(self.ends[-1] - self.ends[start])
        last, first = len(self.ends) - 1, start
        self.score = [(sums[last * self.lanes + lane] - sums[first * self.lanes + lane]) / span for lane in range(self.lanes)]
        self.lane = {}
        for entry in self.meta["envSchedule"]:
            for lane, seed in enumerate(range(entry["seedStart"], entry["seedEnd"] + 1)):
                self.lane[(entry["envId"], seed)] = self.lane_base(entry) + lane
        self.envs = [entry["envId"] for entry in self.meta["envSchedule"]]
        self.seeds = {entry["envId"]: list(range(entry["seedStart"], entry["seedEnd"] + 1)) for entry in self.meta["envSchedule"]}

    def lane_base(self, entry):
        return self.meta["seeds"].index(entry["seedStart"])

    def scores(self, env):
        return [self.score[self.lane[(env, seed)]] for seed in self.seeds[env]]

    def sources(self):
        return json.loads((self.root / "sources.json").read_text())


def snapshot(target):
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for entry in SNAPSHOT:
        source = REPO / entry
        if source.is_dir():
            shutil.copytree(source, target / entry, ignore=shutil.ignore_patterns(".*"))
        else:
            shutil.copy2(source, target / entry)


def probe(engine, tag, index):
    root = BENCH / f"probe-{tag}-{index}"
    shutil.rmtree(root, ignore_errors=True)
    out = subprocess.run(["caffeinate", "-i", str(engine), "record", "oak", str(PROBE_STEPS), str(root)],
                         capture_output=True, text=True, check=True, cwd=REPO).stdout
    shutil.rmtree(root, ignore_errors=True)
    return float(out.rsplit(":", 1)[1].split()[0])


def reference_engine(reference, preset):
    source = reference.root / "source"
    if not source.is_dir():
        raise SystemExit(f"Recorded reference carries no source snapshot and cannot be timed: {reference.root.name}")
    tree = REPO / ".build/reference"
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(source, tree)
    subprocess.run([tree / "setup.sh", "--preset", preset], cwd=tree, check=True, capture_output=True)
    return tree / ".build/oak"


def paired_speed(reference, preset):
    other = reference_engine(reference, preset)
    if filecmp.cmp(ENGINE, other, shallow=False):
        print(f"binary identical to recorded {reference.root.name[:9]}; no probe", flush=True)
        return 1.0
    print(f"timing against recorded {reference.root.name[:9]}", flush=True)
    ratios = []
    for block in range(PROBE_BLOCKS):
        a = probe(ENGINE, "candidate", 2 * block)
        b = probe(other, "reference", 2 * block)
        c = probe(other, "reference", 2 * block + 1)
        d = probe(ENGINE, "candidate", 2 * block + 1)
        ratios.append((a + d) / (b + c))
        print(f"speed block {block + 1}: candidate {(a + d) / 2:,.0f}  reference {(b + c) / 2:,.0f}  ratio {ratios[-1]:.4f}", flush=True)
    return quantile(sorted(ratios), 0.5)


def record(policy, root, sources):
    print(f"Recording {policy} into {root.name}", flush=True)
    staged = root.with_name(root.name + ".source")
    snapshot(staged)
    subprocess.run(["caffeinate", "-i", ENGINE, "record", policy, str(STEPS), root], check=True, cwd=REPO)
    (root / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    staged.rename(root / "source")
    return Recording(root)


def baseline(policy, digests):
    policy_digest = digest(POLICY_SOURCES[policy])
    sources = {"policy": policy, "envs": digests, "policyDigest": policy_digest}
    identity = hashlib.sha256(json.dumps({"sources": sources, "steps": STEPS}, sort_keys=True).encode()).hexdigest()[:40]
    root = BENCH / f"{policy}-{identity}"
    if (root / "meta.json").exists():
        print(f"Reusing {policy}: {root.name}", flush=True)
        rec = Recording(root)
    else:
        rec = record(policy, root, sources)
    if rec.meta["policy"] != policy or rec.steps != STEPS or rec.sources() != sources:
        raise SystemExit(f"Recorded {policy} baseline does not match its identity: {root.name}")
    return {env: rec for env in rec.envs}


def mean(xs):
    return sum(xs) / len(xs)


def quantile(sorted_xs, q):
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def scale(xs):
    s = sorted(xs)
    iqr = quantile(s, 0.75) - quantile(s, 0.25)
    if iqr > 0:
        return iqr
    med = quantile(s, 0.5)
    mad = quantile(sorted(abs(x - med) for x in xs), 0.5) * 1.4826
    return mad if mad > 0 else 1.0


def iqm(xs):
    s = sorted(xs)
    n = len(s)
    lo, hi = n // 4, n - n // 4
    return mean(s[lo:hi])


def bootstrap(groups, statistic):
    rng = random.Random(0)
    reps = []
    for _ in range(BOOTSTRAP):
        pooled = []
        for group in groups:
            pooled.extend(rng.choices(group, k=len(group)))
        reps.append(statistic(pooled))
    reps.sort()
    return quantile(reps, 0.025), quantile(reps, 0.975)


def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def wilcoxon_less(diffs):
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    center = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    return normal_cdf((positive - center) / math.sqrt(var))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rejected = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > ALPHA / (len(pvalues) - rank):
            break
        rejected[i] = True
    return rejected


def wins(a, b):
    return mean([1.0 if x > y else 0.5 if x == y else 0.0 for x, y in zip(a, b)])


def cv_and_rho(a, b):
    ma, mb = mean(a), mean(b)
    va = mean([(x - ma) ** 2 for x in a])
    vb = mean([(y - mb) ** 2 for y in b])
    cov = mean([(x - ma) * (y - mb) for x, y in zip(a, b)])
    cv = math.sqrt(vb) / abs(mb) if mb != 0 else float("inf")
    rho = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    return cv, rho


def seeds_needed(cv, rho):
    if not math.isfinite(cv):
        return SEED_CAP
    n = 2 * (1 - rho) * (cv / SEED_EFFECT) ** 2 * SEED_POWER
    return max(SEED_FLOOR, min(SEED_CAP, math.ceil(n)))


def cv_of(xs):
    m = mean(xs)
    spread = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return spread / abs(m) if m != 0 else float("inf")


def report(candidate, baselines, best):
    cols = ["env", "oak", "random", "oracle", "score", "ppo", "win>ppo"]
    if best is not None:
        cols += ["best", "d", "win>best", "guard p", "cv", "rho", "seeds"]
    else:
        cols += ["cv", "seeds"]
    rows = []
    d_groups, diff_groups, pvalues, win_best = [], [], [], []
    for env in candidate.envs:
        oak = candidate.scores(env)
        rnd = baselines["random"][env].scores(env)
        ppo = baselines["ppo"][env].scores(env)
        oracle = baselines["oracle"][env].scores(env)
        row = [env, mean(oak), mean(rnd), mean(oracle), (mean(oak) - mean(rnd)) / (mean(oracle) - mean(rnd)), mean(ppo), wins(oak, ppo)]
        if best is not None:
            ref = best.scores(env)
            s = scale(ref)
            d = [(x - y) / s for x, y in zip(oak, ref)]
            diffs = [x - y for x, y in zip(oak, ref)]
            p = wilcoxon_less(diffs)
            cv, rho = cv_and_rho(oak, ref)
            row += [mean(ref), iqm(d), wins(oak, ref), p, cv, rho, seeds_needed(cv, rho)]
            d_groups.append(d)
            diff_groups.append(diffs)
            pvalues.append(p)
            win_best.append(wins(oak, ref))
        else:
            row += [cv_of(oak), seeds_needed(cv_of(oak), 0.0)]
        rows.append(row)
    widths = [max(len(c), max(len(fmt(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print(" | ".join(fmt(v).rjust(w) if i else fmt(v).ljust(w) for i, (v, w) in enumerate(zip(r, widths))))
    dead = sum(1 for h in candidate.health if h != 0xffffffff)
    print(f"dead lanes: {dead} of {candidate.lanes}")
    if best is None:
        return None
    suite = iqm([x for g in d_groups for x in g])
    lo, hi = bootstrap(d_groups, iqm)
    poi = mean(win_best)
    plo, phi = bootstrap([[1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in g] for g in diff_groups], mean)
    guard = holm(pvalues)
    fired = [env for env, rej in zip(candidate.envs, guard) if rej]
    identical = all(candidate.hash[candidate.lane[key]] == best.hash[best.lane[key]] for key in candidate.lane)
    print(f"suite IQM d: {suite:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   P(improve): {poi:.3f} [{plo:.3f}, {phi:.3f}]")
    print(f"regression guard (Holm {ALPHA}): {'fired on ' + ', '.join(fired) if fired else 'silent'}")
    print(f"reward streams: {'identical' if identical else 'different'}")
    return {"lo": lo, "hi": hi, "guard": bool(fired), "identical": identical}


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.4f}"
    return str(v)


def main():
    verb = sys.argv[1] if len(sys.argv) > 1 else "judge"
    if verb not in ("judge", "adopt"):
        raise SystemExit("usage: ./fleet.py [adopt]")
    best_id = BEST.read_text().strip() if BEST.exists() else ""
    if best_id:
        preset = json.loads((BENCH / best_id / "meta.json").read_text())["preset"]
    elif (REPO / ".build/preset").exists() and (REPO / ".build/preset").read_text().strip() in ("base", "pro", "max", "ultra"):
        preset = (REPO / ".build/preset").read_text().strip()
    else:
        raise SystemExit("No best and no built preset: ./setup.sh --preset base|pro|max|ultra first")
    subprocess.run([REPO / "setup.sh", "--preset", preset], cwd=REPO, check=True)
    tree = hashlib.sha256((digest(("Sources", "setup.sh")) + preset).encode()).hexdigest()[:40]
    digests = env_digests()
    if not best_id:
        raise SystemExit("No recorded reference with baseline links; nothing will be recorded")
    reference = Recording(BENCH / best_id)
    best = reference if verb == "judge" else None
    if best is not None:
        table = json.loads(subprocess.run([ENGINE, "envs"], capture_output=True, text=True, check=True).stdout)
        schedule = {row["envId"]: (row["seedStart"], row["seedEnd"]) for row in table}
        expected = {entry["envId"]: (entry["seedStart"], entry["seedEnd"]) for entry in best.meta["envSchedule"]}
        if schedule != expected or best.steps != STEPS:
            raise SystemExit("Schedule differs from the recorded best; environments, seeds, and horizon must match before recording")
        reference_sources = best.sources()["envs"]
        changed = sorted(env for env in reference_sources.keys() | digests.keys() if reference_sources.get(env) != digests.get(env))
        if changed:
            print(f"Environment sources changed since the recorded best ({len(changed)} environments); comparing on the same seeds and horizon: {', '.join(changed)}", flush=True)
    baselines = {policy: baseline(policy, digests) for policy in POLICIES}
    BENCH.mkdir(exist_ok=True)
    candidate_root = BENCH / tree
    sources = {"policy": "oak", "preset": preset, "envs": digests, "policyDigest": digest(("Sources",))}
    if (candidate_root / "meta.json").exists():
        (candidate_root / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
        snapshot(candidate_root / "source")
        candidate = Recording(candidate_root)
    else:
        candidate = record("oak", candidate_root, sources)
    for policy, found in baselines.items():
        if set(found) != set(candidate.envs) or any(found[env].steps != candidate.steps or found[env].seeds[env] != candidate.seeds[env] for env in candidate.envs):
            raise SystemExit(f"Recorded {policy} baselines do not match the candidate schedule")
    (candidate_root / "baselines.json").write_text(json.dumps(
        {policy: {env: rec.root.name for env, rec in found.items()} for policy, found in baselines.items()},
        indent=1, sort_keys=True) + "\n")
    if best is not None and (set(best.envs) != set(candidate.envs) or best.seeds != candidate.seeds or best.steps != candidate.steps):
        print(f"best envs: {best.envs}\ncandidate envs: {candidate.envs}")
        raise SystemExit("Schedule differs from the best: ./fleet.py adopt if this change is intended")
    verdict = report(candidate, baselines, best)
    sps = candidate.meta["stepsPerSecond"]
    if best is None:
        print(f"Steps/s: {sps:,.0f}")
        if best_id and verb == "adopt":
            previous = Recording(BENCH / best_id)
            common = [key for key in candidate.lane if key in previous.lane]
            same = sum(1 for key in common if candidate.hash[candidate.lane[key]] == previous.hash[previous.lane[key]])
            print(f"reward streams identical to the previous best on {same} of {len(common)} shared lanes")
        BEST.write_text(tree + "\n")
        print(f"ADOPT: best is {tree}")
        return
    reward_better = verdict["lo"] > 0 and not verdict["guard"]
    reward_not_worse = verdict["identical"] or (verdict["lo"] >= -MARGIN and not verdict["guard"])
    if not reward_not_worse:
        raise SystemExit(f"REJECT: best stays {best_id}")
    ratio = paired_speed(best, preset)
    print(f"Steps/s: {sps:,.0f} recorded; paired speed against best: {ratio:.4f}x")
    sps_better = ratio >= 1 + MARGIN
    sps_not_worse = ratio >= 1 - MARGIN
    if (reward_better and sps_not_worse) or sps_better:
        BEST.write_text(tree + "\n")
        print(f"PROMOTE: {tree}")
    else:
        raise SystemExit(f"REJECT: best stays {best_id}")


if __name__ == "__main__":
    main()
