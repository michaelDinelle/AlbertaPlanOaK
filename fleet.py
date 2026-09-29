#!/usr/bin/env python3
import bisect
import fcntl
import filecmp
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from array import array
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
BASELINES = BENCH / "baselines"
SPEED = BENCH / "speed"
VERDICTS = BENCH / "verdicts.log"
RUNNER_LOCK = BENCH / "runner.lock"
RUNNER_LOG = BENCH / "runner.log"
TRIES = BENCH / "tries"
CHAMPION = TRIES / "champion"
QUEUE = REPO / ".build/queue"
BEST = REPO / "best"
ANCHOR = "232cf8f183dddf4c46025163f06757a3703b4af4"
RECORDED_ENGINE = "oak"
STEPS = 1_000_000
TERMINAL = 100_000
MARGIN = 0.01
BUDGET = 0.5
ALPHA = 0.05
PROBE_STEPS = 20_000
PROBE_BLOCKS = 3
CONFIRM_SEEDS = 3
SELECTED_SEEDS = 1 + CONFIRM_SEEDS
ENVCHECK_STEPS = 200_000
REWARD_ANCHOR = BENCH / "reward-anchor"
CHECKS = BENCH / "checks"
POLICIES = ("random", "constant", "ppo", "oracle")
POLICY_SOURCES = {"random": ("Sources/random.c",), "constant": ("Sources/constant.c",), "ppo": ("Sources/ppo.c",), "oracle": ()}
ENV_LAYER = ("Sources/env.h", "Sources/rng.h", "Sources/portable.h")
SNAPSHOT = ("Sources", "catalog.py", "setup.sh")
ENV_ROW = re.compile(r'\s*X\(\s*\w+\s*,\s*(\w+)\s*,.*?"([^"]+)"')
VERBS = ("judge", "adopt", "probe", "trace", "try", "check", "invariance")
TIMING = ("probe", "trace", "try", "check", "invariance")
TRACE_SECONDS = 10
ACTIVE = []


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def digest(entries, root=REPO):
    result = hashlib.sha256()
    for entry in sorted(entries):
        path = root / entry
        files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")) if path.is_dir() else [path]
        for file in files:
            result.update(str(file.relative_to(root)).encode() + b"\0" + file.read_bytes() + b"\0")
    return result.hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_digests(table, root=REPO):
    rows = {}
    for line in (root / "Sources/env.c").read_text().splitlines():
        row = ENV_ROW.match(line)
        if row:
            rows[row.group(2)] = (row.group(1), line.strip())
    texts = {p: p.read_text() for p in sorted((root / "Sources/envs").glob("*.c"))}
    layers, digests = {}, {}
    for entry in table:
        name = entry["envId"]
        if name not in rows:
            raise SystemExit(f"{name} has no row in the environment table")
        family, row = rows[name]
        if family not in layers:
            hook = re.compile(rf"\b{family}_step\(")
            owners = [p for p, text in texts.items() if hook.search(text)]
            if len(owners) != 1:
                raise SystemExit(f"{name} draws its behaviour from {len(owners)} environment files")
            layers[family] = digest(ENV_LAYER + (str(owners[0].relative_to(root)),), root)
        digests[name] = hashlib.sha256((layers[family] + row).encode()).hexdigest()
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


def complete(root):
    meta = root / "meta.json"
    return meta.exists() and "finishedAt" in json.loads(meta.read_text())


def snapshot(target, root=REPO):
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for entry in SNAPSHOT:
        source = root / entry
        if source.is_dir():
            shutil.copytree(source, target / entry, ignore=shutil.ignore_patterns(".*"))
        else:
            shutil.copy2(source, target / entry)


def processes():
    out = subprocess.run(["ps", "-ax", "-o", "pid=,stat=,comm="], capture_output=True, text=True, check=True).stdout
    for fields in (line.split(None, 2) for line in out.splitlines()):
        if len(fields) == 3:
            yield int(fields[0]), fields[1], Path(fields[2].strip()).name


def stopped(pid):
    return any(p == pid and stat.startswith("T") for p, stat, _ in processes())


@contextmanager
def quiet():
    if ACTIVE and (ACTIVE[-1] / "cancel").exists():
        raise SystemExit("CANCELLED")
    paused = []
    for pid, stat, name in processes():
        if name == "oak" and not stat.startswith("T"):
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
            except ProcessLookupError:
                pass
    try:
        yield
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def probe(engine, tag, index):
    root = BENCH / f"probe-{tag}-{index}"
    shutil.rmtree(root, ignore_errors=True)
    with quiet():
        subprocess.run(["caffeinate", "-i", str(engine), "record", "oak", str(PROBE_STEPS), str(root)],
                       capture_output=True, check=True, cwd=REPO)
    streams = array("Q", (root / "hash.u64").read_bytes())
    speed = json.loads((root / "meta.json").read_text())["stepsPerSecond"]
    shutil.rmtree(root, ignore_errors=True)
    return speed, streams


def leaves(export):
    frames, backtraces, binaries, counts = {}, {}, {}, {}
    for _, element in ElementTree.iterparse(io.StringIO(export), events=("end",)):
        if element.tag == "binary" and "id" in element.attrib:
            binaries[element.attrib["id"]] = element.attrib
        elif element.tag == "frame" and "id" in element.attrib:
            binary = element.find("binary")
            frames[element.attrib["id"]] = (element.attrib.get("name", "?"), element.attrib.get("addr", "0x0"),
                                            None if binary is None else binary.attrib.get("id", binary.attrib.get("ref")))
        elif element.tag == "backtrace" and "id" in element.attrib:
            first = element.find("frame")
            backtraces[element.attrib["id"]] = first.attrib.get("id", first.attrib.get("ref"))
        elif element.tag == "row":
            backtrace = element.find("backtrace")
            if backtrace is not None:
                leaf = backtraces[backtrace.attrib.get("id", backtrace.attrib.get("ref"))]
                counts[leaf] = counts.get(leaf, 0) + 1
            element.clear()
    return [(frames[leaf], binaries.get(frames[leaf][2], {}), count) for leaf, count in counts.items()]


def trace(engine):
    root, report = BENCH / "probe-trace", BENCH / "trace.trace"
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(report, ignore_errors=True)
    with quiet():
        process = subprocess.Popen([str(engine), "record", "oak", str(PROBE_STEPS), str(root)], cwd=REPO, stdout=subprocess.DEVNULL)
        while not (root / "meta.json").exists():
            time.sleep(0.1)
        subprocess.run(["xcrun", "xctrace", "record", "--template", "Time Profiler", "--attach", str(process.pid),
                        "--time-limit", f"{TRACE_SECONDS}s", "--output", str(report)], check=True, capture_output=True)
        process.wait()
    shutil.rmtree(root, ignore_errors=True)
    export = subprocess.run(["xcrun", "xctrace", "export", "--input", str(report), "--xpath",
                             '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'], capture_output=True, text=True, check=True).stdout
    shutil.rmtree(report, ignore_errors=True)
    symbols = sorted((int(address, 16), name) for address, kind, name in
                     (line.split() for line in subprocess.run(["nm", "-n", str(engine)], capture_output=True, text=True, check=True).stdout.splitlines() if line.count(" ") == 2)
                     if kind in "tT")
    starts = [address for address, _ in symbols]
    functions, places, total = {}, {}, 0
    for (name, address, _), binary, count in leaves(export):
        total += count
        functions[name] = functions.get(name, 0) + count
        if binary.get("name") == "oak" and "load-addr" in binary:
            static = (0x100000000 + int(address, 16) - int(binary["load-addr"], 16)) & ~3
            index = bisect.bisect_right(starts, static) - 1
            place = f"{symbols[index][1]}+0x{static - starts[index]:x}"
            places[place] = places.get(place, 0) + count
    print(f"{total} samples", flush=True)
    for name, count in sorted(functions.items(), key=lambda item: -item[1])[:20]:
        print(f"  {100 * count / total:6.2f}%  {name}", flush=True)
    for place, count in sorted(places.items(), key=lambda item: -item[1])[:40]:
        print(f"  {100 * count / total:6.2f}%  {place}", flush=True)


def toolchain():
    driver = subprocess.run(["clang", "-###", "-x", "c", "-c", "/dev/null", "-o", "/dev/null"], capture_output=True, text=True, check=True).stderr
    sysroot = re.search(r'"-isysroot" "([^"]+)"', driver)[1]
    return driver + str(Path(sysroot).resolve()) + subprocess.run(["ld", "-v"], capture_output=True, text=True, check=True).stderr


def reference_engine(reference):
    source = reference.root / "source"
    if not source.is_dir():
        raise SystemExit(f"Recorded reference carries no source snapshot and cannot be timed: {reference.root.name}")
    key = digest([p.name for p in source.iterdir() if not p.name.startswith(".")], source) + toolchain()
    tree = REPO / ".build" / ("reference-" + hashlib.sha256(key.encode()).hexdigest()[:16])
    if not tree.is_dir():
        staged = tree.with_name(tree.name + ".partial")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(source, staged)
        subprocess.run([staged / "setup.sh"], cwd=staged, check=True, capture_output=True)
        staged.rename(tree)
    return tree / ".build/oak"


def paired_speed(engine, other, reference, compare):
    if filecmp.cmp(engine, other, shallow=False):
        print(f"binary identical to recorded {reference.root.name[:9]}; no probe", flush=True)
        return 1.0, 0
    print(f"timing against recorded {reference.root.name[:9]}", flush=True)
    ratios, diverged = [], 0
    for block in range(PROBE_BLOCKS):
        a, streams = probe(engine, "candidate", 2 * block)
        b, expected = probe(other, "reference", 2 * block)
        if compare and block == 0:
            diverged = sum(1 for x, y in zip(streams, expected) if x != y)
            if diverged:
                print(f"reward streams diverge from recorded {reference.root.name[:9]} on {diverged} of {len(streams)} lanes within {PROBE_STEPS} steps", flush=True)
        c, _ = probe(other, "reference", 2 * block + 1)
        d, _ = probe(engine, "candidate", 2 * block + 1)
        ratios.append((a + d) / (b + c))
        print(f"speed block {block + 1}: candidate {(a + d) / 2:,.0f}  reference {(b + c) / 2:,.0f}  ratio {ratios[-1]:.4f}", flush=True)
    return quantile(sorted(ratios), 0.5), diverged


def speed_stage(engine, timing_engine, timing_reference, compare):
    key = file_digest(engine) + file_digest(timing_engine)
    path = SPEED / (hashlib.sha256(key.encode()).hexdigest()[:40] + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        print(f"speed stage already measured at {result['at']} for this binary pair", flush=True)
        if compare and result["diverged"]:
            print(f"reward streams diverge from recorded {timing_reference.root.name[:9]} on {result['diverged']} lanes within {PROBE_STEPS} steps", flush=True)
    else:
        ratio, diverged = paired_speed(engine, timing_engine, timing_reference, compare)
        result = {"at": stamp(), "ratio": ratio, "diverged": diverged}
        SPEED.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"paired speed against recorded {timing_reference.root.name[:9]}: {result['ratio']:.4f}x", flush=True)
    return result["ratio"]


def supervise(command):
    process = subprocess.Popen(command, cwd=REPO, start_new_session=True)
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)])
    (ACTIVE[-1] / "pid").write_text(f"{process.pid}\n")
    while process.poll() is None:
        job = pending(TIMING)
        if job is None:
            time.sleep(2)
            continue
        while process.poll() is None and not stopped(process.pid):
            try:
                os.kill(process.pid, signal.SIGTSTP)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        print(f"{stamp()} paused for probe {job.name}", flush=True)
        run(job)
        print(f"{stamp()} resumed", flush=True)
        try:
            os.kill(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    awake.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def record(engine, policy, root, sources, tree, envs=()):
    print(f"Recording {policy} into {root.name}", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    staged = root.with_name(root.name + ".source")
    snapshot(staged, tree)
    supervise([str(engine), "record", policy, str(STEPS), str(root), *envs])
    (root / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    staged.rename(root / "source")
    keep_engine(engine, root)
    return Recording(root)


def baseline_identity(policy, env, env_digest, seeds, policy_digest):
    return hashlib.sha256(json.dumps({"policy": policy, "env": env, "digest": env_digest, "seeds": seeds, "steps": STEPS, "policyDigest": policy_digest}, sort_keys=True).encode()).hexdigest()[:40]


def baseline(policy, digests, schedule, engine, tree):
    policy_digest = digest(POLICY_SOURCES[policy], tree)
    index = BASELINES / policy
    found, missing = {}, []
    for env, seeds in schedule.items():
        identity = baseline_identity(policy, env, digests[env], seeds, policy_digest)
        link = index / identity
        if link.exists():
            found[env] = BENCH / link.read_text().strip()
        else:
            missing.append((env, identity))
    if missing:
        names = [env for env, _ in missing]
        root = BENCH / f"{policy}-{hashlib.sha256(json.dumps(sorted(identity for _, identity in missing)).encode()).hexdigest()[:40]}"
        record(engine, policy, root, {"policy": policy, "envs": {env: digests[env] for env in names}, "policyDigest": policy_digest}, tree, names)
        index.mkdir(parents=True, exist_ok=True)
        for env, identity in missing:
            (index / identity).write_text(root.name + "\n")
            found[env] = root
    else:
        print(f"Reusing {policy} for all {len(found)} environments", flush=True)
    loaded, result = {}, {}
    for env, root in found.items():
        if root not in loaded:
            loaded[root] = Recording(root)
        rec = loaded[root]
        sources = rec.sources()
        if env not in rec.seeds or rec.meta["policy"] != policy or rec.steps != STEPS or sources["policyDigest"] != policy_digest or sources["envs"].get(env) != digests[env] or [rec.seeds[env][0], rec.seeds[env][-1]] != schedule[env]:
            raise SystemExit(f"Recorded {policy} baseline for {env} does not match its identity: {root.name}")
        result[env] = rec
    return result


def mean(xs):
    return sum(xs) / len(xs)


def quantile(sorted_xs, q):
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def iqm(xs):
    s = sorted(xs)
    n = len(s)
    lo, hi = n // 4, n - n // 4
    return mean(s[lo:hi])


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


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.4f}"
    return str(v)


def sign_tail(wins, losses):
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0


def table_print(cols, rows):
    widths = [max(len(c), max(len(fmt(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print(" | ".join(fmt(v).rjust(w) if i else fmt(v).ljust(w) for i, (v, w) in enumerate(zip(r, widths))))


def lanes_of(recordings):
    found = {}
    for rec in recordings:
        for key, lane in rec.lane.items():
            found[key] = (rec.score[lane], rec.hash[lane], rec.health[lane])
    return found


class Baselines:
    def __init__(self, recording):
        self.links = json.loads((recording.root / "baselines.json").read_text())
        missing = [policy for policy in POLICIES if policy not in self.links]
        if missing:
            raise SystemExit(f"Recording {recording.root.name[:9]} links no {', '.join(missing)} baselines")
        self.loaded = {}

    def mean(self, policy, env):
        name = self.links[policy][env]
        if name not in self.loaded:
            self.loaded[name] = Recording(BENCH / name)
        return mean(self.loaded[name].scores(env))

    def span(self, env):
        floor = max(self.mean("random", env), self.mean("constant", env))
        return floor, self.mean("oracle", env) - floor

    def norm(self, env, value):
        floor, width = self.span(env)
        return (value - floor) / width if width > 0 else math.nan


def finite(values):
    return [v for v in values if math.isfinite(v)]


def average(values):
    kept = finite(values)
    return mean(kept) if kept else math.nan


def compare(table, best, candidates, references, seeds, title):
    print(f"\n{title}", flush=True)
    base = Baselines(best)
    cand, ref = lanes_of(candidates), lanes_of(references)
    category = {row["envId"]: row["category"] for row in table}
    rows = []
    for env in candidates[0].envs:
        keys = [(env, seed) for seed in seeds[env]]
        absent = [key[1] for key in keys if key not in cand or key not in ref]
        if absent:
            raise SystemExit(f"{title}: {env} lacks lanes for seeds {absent}")
        c, r = mean([cand[k][0] for k in keys]), mean([ref[k][0] for k in keys])
        same = all(cand[k][1] == ref[k][1] for k in keys)
        rows.append({"env": env, "category": category[env], "cand": base.norm(env, c), "ref": base.norm(env, r),
                     "ppo": base.norm(env, base.mean("ppo", env)),
                     "sign": 0 if same or c == r else (1 if c > r else -1), "same": same, "dead": any(cand[k][2] != 0xffffffff for k in keys)})
    rows.sort(key=lambda r: (r["category"], r["env"]))
    table_print(["env", "category", "oak", "ref", "ppo", "d"],
                [[r["env"], r["category"], r["cand"], r["ref"], r["ppo"], "=" if r["same"] else r["cand"] - r["ref"]] for r in rows])
    groups = [("suite", rows)] + [(name, [r for r in rows if r["category"] == name]) for name in sorted(set(category.values()))]
    better, worse, summary = [], [], []
    for name, members in groups:
        up, down = sum(1 for r in members if r["sign"] > 0), sum(1 for r in members if r["sign"] < 0)
        better.append(sign_tail(up, down))
        worse.append(sign_tail(down, up))
        summary.append([name, len(members), average([r["cand"] for r in members]), average([r["ref"] for r in members]), up, down, better[-1], worse[-1]])
    print()
    table_print(["group", "n", "oak", "ref", "wins", "losses", "p better", "p worse"], summary)
    stats = {}
    for key in ("cand", "ref"):
        values = sorted(finite([r[key] for r in rows]))
        stats[key] = (mean(values), iqm(values), quantile(values, 0.5))
        print(f"{'oak' if key == 'cand' else 'ref'}: mean {stats[key][0]:+.4f}  IQM {stats[key][1]:+.4f}  median {stats[key][2]:+.4f}", flush=True)
    for rec in candidates:
        sps = rec.meta["stepsPerSecond"]
        print(f"{rec.root.name[:12]}: dead lanes {sum(1 for h in rec.health if h != 0xffffffff)} of {rec.lanes}   {sps:,.0f} steps/s   {rec.lanes * rec.steps / sps:.1f} s", flush=True)
    if all(r["same"] for r in rows):
        verdict = "IDENTICAL"
    elif any(holm(worse)):
        verdict = "WORSE"
    elif any(holm(better)):
        verdict = "BETTER"
    else:
        verdict = "NO SIGNAL"
    print(f"verdict: {verdict}", flush=True)
    return {"verdict": verdict, "closer": all(c > r for c, r in zip(stats["cand"], stats["ref"])), "wins": summary[0][4], "losses": summary[0][5], "rows": rows}


def guard(candidate, reference, seeds, title):
    pvalues = []
    for env in candidate.envs:
        diffs = [candidate.score[candidate.lane[(env, seed)]] - reference.score[reference.lane[(env, seed)]] for seed in seeds[env]]
        pvalues.append(wilcoxon_less(diffs))
    fired = [env for env, rejected in zip(candidate.envs, holm(pvalues)) if rejected]
    print(f"regression guard against {title} (Holm {ALPHA}): {'fired on ' + ', '.join(fired) if fired else 'silent'}", flush=True)
    return fired


def atlas(recording, rows):
    raw = array("I", (recording.root / "instruments.u32").read_bytes())
    if len(raw) != 4 * recording.lanes:
        raise SystemExit(f"Recording {recording.root.name[:9]} has instruments for {len(raw) // 4} of {recording.lanes} lanes")
    standing = {r["env"]: r for r in rows}
    found = {}
    for env in recording.envs:
        lanes = [recording.lane[(env, seed)] for seed in recording.seeds[env]]
        firsts = [raw[4 * lane + 3] for lane in lanes if raw[4 * lane + 3]]
        entry = {"episodesPerMillion": mean([raw[4 * lane] for lane in lanes]) * 1_000_000 / recording.steps,
                 "positive": sum(raw[4 * lane + 1] for lane in lanes), "negative": sum(raw[4 * lane + 2] for lane in lanes),
                 "firstPositive": min(firsts) if firsts else None, "labels": []}
        row = standing.get(env)
        floored = bool(row) and math.isfinite(row["cand"]) and row["cand"] <= 0.02
        if floored and entry["positive"] == 0:
            entry["labels"].append("never rewarded")
        if floored and entry["episodesPerMillion"] < 100:
            entry["labels"].append("starved")
        if floored:
            entry["labels"].append("at the floor")
        if row and math.isfinite(row["cand"]) and math.isfinite(row["ppo"]) and row["cand"] < row["ppo"]:
            entry["labels"].append("below PPO")
        if row:
            entry["score"] = row["cand"]
        found[env] = entry
    (recording.root / "atlas.json").write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    counts = {}
    for entry in found.values():
        for label in entry["labels"]:
            counts[label] = counts.get(label, 0) + 1
    print("atlas: " + ", ".join(f"{label} {count}" for label, count in sorted(counts.items())), flush=True)
    return found


def keep_engine(engine, root):
    staged = root / f".{RECORDED_ENGINE}.partial"
    shutil.copy2(engine, staged)
    staged.rename(root / RECORDED_ENGINE)


def attempted(engine, root, first, count, tree=None, policy="oak"):
    if complete(root):
        print(f"already tried: {root.name[:9]} (seeds {first}..{first + count - 1})", flush=True)
    else:
        shutil.rmtree(root, ignore_errors=True)
        staged = root.with_name(root.name + ".source")
        if tree is not None:
            snapshot(staged, tree)
        with quiet():
            subprocess.run(["caffeinate", "-i", str(engine), "try", policy, str(STEPS), str(first), str(count), str(root)], check=True, cwd=REPO)
        if tree is not None:
            staged.rename(root / "source")
            keep_engine(engine, root)
    return Recording(root)


def attempt(engine, tree, table, best):
    TRIES.mkdir(parents=True, exist_ok=True)
    key = file_digest(engine)[:40]
    label = f"try {key[:9]} against best {best.root.name[:9]}"
    screen = attempted(engine, TRIES / key, 0, 1, tree)
    s = compare(table, best, [screen], [best], {env: best.seeds[env][:1] for env in best.envs}, "screen: seed 0 against the best")
    atlas(screen, s["rows"])
    if s["verdict"] == "IDENTICAL":
        return f"IDENTICAL: {label}"
    if s["verdict"] == "WORSE" or (s["verdict"] == "NO SIGNAL" and not s["closer"]):
        return f"{s['verdict']}: {label} on seed 0; suite {s['wins']} wins, {s['losses']} losses"
    fresh = {env: best.seeds[env][1:1 + CONFIRM_SEEDS] for env in best.envs}
    wide = attempted(engine, TRIES / f"{key}+1+{CONFIRM_SEEDS}", 1, CONFIRM_SEEDS)
    c = compare(table, best, [wide], [best], fresh, f"confirmation: seeds 1..{CONFIRM_SEEDS} against the best")
    crowned, against = c["verdict"] == "BETTER", ""
    champion = CHAMPION.read_text().strip() if CHAMPION.exists() else ""
    if crowned and champion:
        h = compare(table, best, [wide], [Recording(TRIES / f"{champion}+1+{CONFIRM_SEEDS}")], fresh, f"against champion {champion[:9]}'s confirmation run")
        against = f"; against champion {champion[:9]}: {h['verdict']}"
        crowned = h["verdict"] != "WORSE"
    if crowned:
        CHAMPION.write_text(key + "\n")
        return f"BETTER, confirmed: {label}; seeds 1-{CONFIRM_SEEDS} suite {c['wins']} wins, {c['losses']} losses{against}"
    return f"{s['verdict']} on seed 0, {c['verdict']} on seeds 1-{CONFIRM_SEEDS}: {label}; suite {c['wins']} wins, {c['losses']} losses{against}"


def invariance(engine, tree, table, best):
    key = file_digest(engine)[:40]
    screen = attempted(engine, TRIES / key, 0, 1, tree)
    first = {env: best.seeds[env][:1] for env in best.envs}
    CHECKS.mkdir(parents=True, exist_ok=True)
    found = []
    for variant in ("oak-permuted", "oak-rescaled"):
        probe_run = attempted(engine, CHECKS / f"{key}-{variant}", 0, 1, policy=variant)
        v = compare(table, best, [probe_run], [screen], first, f"{variant} against the plain try")
        found.append(f"{variant[4:]} {'INVARIANT' if v['verdict'] in ('NO SIGNAL', 'IDENTICAL') else 'VARIES (' + v['verdict'] + ')'}")
    return "invariance: " + "; ".join(found)


def fuzz_runs(engine):
    return [json.loads(line) for line in subprocess.run([str(engine), "fuzz"], stdout=subprocess.PIPE, text=True, check=True, cwd=REPO).stdout.splitlines() if line.strip()]


def failing(runs):
    return [f for f in runs if f["nonfinite"] or f["invalidControls"]]


def check(engine, tree_id):
    CHECKS.mkdir(parents=True, exist_ok=True)
    with quiet():
        worlds = [json.loads(line) for line in subprocess.run([str(engine), "envcheck", str(ENVCHECK_STEPS)], stdout=subprocess.PIPE, text=True, check=True, cwd=REPO).stdout.splitlines() if line.strip()]
        fuzzed = fuzz_runs(engine)
    flagged = []
    for w in worlds:
        floor = max(w["random"]["reward"], w["constant"]["reward"])
        flags = []
        if not w["deterministic"]:
            flags.append("oracle not deterministic")
        if not all(w[p]["healthy"] for p in ("random", "constant", "oracle")):
            flags.append("non-finite reward or observation")
        if w["oracle"]["reward"] <= floor:
            flags.append("oracle at or below the floor")
        w["flags"] = flags
        if flags:
            flagged.append(w)
    table_print(["env", "random", "constant", "oracle", "oracle eps", "random eps", "flags"],
                [[w["env"], w["random"]["reward"], w["constant"]["reward"], w["oracle"]["reward"], w["oracle"]["episodes"], w["random"]["episodes"], "; ".join(w["flags"])] for w in worlds])
    failures = failing(fuzzed)
    print()
    if failures:
        table_print(["env", "obs", "actions", "box", "scenario", "nonfinite", "invalid"], [[f["env"], f["obs"], f["actions"], f["box"], f["scenario"], f["nonfinite"], f["invalidControls"]] for f in failures])
    (CHECKS / f"{tree_id}.json").write_text(json.dumps({"worlds": worlds, "fuzz": fuzzed}, indent=1) + "\n")
    return f"check: {len(worlds)} worlds, {len(flagged)} flagged; fuzz: {len(failures)} of {len(fuzzed)} runs failed"


def learner_bytes(root):
    return sum((root / "Sources" / name).stat().st_size for name in ("oak.c", "geometry.h"))


def anchor_ratio(recording):
    path = recording.root / "speed.json"
    if not path.exists():
        raise SystemExit(f"Recording {recording.root.name[:9]} carries no ratio to the budget anchor; adopt it first")
    return json.loads(path.read_text())["anchorRatio"]


def standing(table, recording):
    base = Baselines(recording)
    category = {row["envId"]: row["category"] for row in table}
    rows = [{"env": env, "category": category[env], "cand": base.norm(env, mean(recording.scores(env))), "ref": math.nan,
             "ppo": base.norm(env, base.mean("ppo", env))} for env in recording.envs]
    groups = [(name, [r for r in rows if r["category"] == name]) for name in sorted(set(category.values()))] + [("suite", rows)]
    table_print(["group", "n", "oak", "ppo"], [[name, len(members)] + [average([r[key] for r in members]) for key in ("cand", "ppo")] for name, members in groups])
    return rows


def execute(tree, verb):
    best_id = BEST.read_text().strip() if BEST.exists() else ""
    if not best_id:
        raise SystemExit("No recorded best to judge against")
    reference = Recording(BENCH / best_id)
    subprocess.run([tree / "setup.sh"], cwd=tree, check=True)
    engine = tree / ".build/oak"
    tree_id = hashlib.sha256(digest(("Sources", "setup.sh"), tree).encode()).hexdigest()[:40]
    if verb == "check":
        return check(engine, tree_id)
    table = json.loads(subprocess.run([engine, "envs"], capture_output=True, text=True, check=True).stdout)
    schedule = {row["envId"]: [row["seedStart"], row["seedEnd"]] for row in table}
    digests = env_digests(table, tree)
    judged = verb != "adopt"
    if judged:
        expected = {entry["envId"]: [entry["seedStart"], entry["seedEnd"]] for entry in reference.meta["envSchedule"]}
        if schedule != expected or reference.steps != STEPS:
            raise SystemExit("Schedule differs from the recorded best; environments, seeds, and horizon must match: ./fleet.py adopt if this change is intended")
        reference_sources = reference.sources()["envs"]
        changed = sorted(env for env in reference_sources.keys() | digests.keys() if reference_sources.get(env) != digests.get(env))
        if changed and verb in ("try", "invariance"):
            raise SystemExit(f"REFUSE: environment sources changed since the recorded best ({', '.join(changed)}); a {verb} compares agents only")
        if changed:
            print(f"Environment sources changed since the recorded best ({len(changed)} environments); comparing on the same seeds and horizon: {', '.join(changed)}", flush=True)
    if verb == "trace":
        trace(engine)
        return f"trace: sampled {TRACE_SECONDS} s of the working tree's engine"
    if verb == "try":
        return attempt(engine, tree, table, reference)
    if verb == "invariance":
        return invariance(engine, tree, table, reference)
    other = reference_engine(reference)
    if judged and filecmp.cmp(engine, other, shallow=False):
        if verb == "probe":
            return f"probe: binary identical to recorded {best_id[:9]}"
        raise SystemExit(f"REJECT: binary identical to recorded {best_id[:9]}; nothing to judge")
    if judged:
        chain = anchor_ratio(reference)
        ratio = speed_stage(engine, other, reference, True)
        budget = ratio * chain
    else:
        anchor = Recording(BENCH / ANCHOR)
        ratio = None
        budget = speed_stage(engine, other if best_id == ANCHOR else reference_engine(anchor), anchor, False)
    if verb == "probe":
        return f"probe: {ratio:.4f}x against recorded {best_id[:9]}; {budget:.4f}x the budget anchor"
    if budget < BUDGET:
        raise SystemExit(f"{'REJECT' if judged else 'REFUSE'}: {budget:.4f}x the budget anchor's throughput, below the budget of {BUDGET}; best stays {best_id}")
    if judged:
        with quiet():
            fuzz_best, fuzz_candidate = len(failing(fuzz_runs(other))), len(failing(fuzz_runs(engine)))
        print(f"fuzz: {fuzz_candidate} failing runs against the best's {fuzz_best}", flush=True)
        if fuzz_candidate > fuzz_best:
            raise SystemExit(f"REJECT: fuzz failures rise from {fuzz_best} to {fuzz_candidate}; best stays {best_id}")
    baselines = {policy: baseline(policy, digests, schedule, engine, tree) for policy in POLICIES}
    candidate_root = BENCH / tree_id
    sources = {"policy": "oak", "envs": digests, "policyDigest": digest(("Sources",), tree)}
    if complete(candidate_root):
        (candidate_root / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
        snapshot(candidate_root / "source", tree)
        candidate = Recording(candidate_root)
    else:
        candidate = record(engine, "oak", candidate_root, sources, tree)
    (candidate_root / "speed.json").write_text(json.dumps({"anchorRatio": budget, "bestRatio": ratio}, indent=1) + "\n")
    key = file_digest(engine)[:40]
    for tried in (TRIES / key, TRIES / f"{key}+1+{CONFIRM_SEEDS}"):
        if complete(tried):
            run = Recording(tried)
            drifted = sorted({env for (env, seed), lane in run.lane.items() if run.hash[lane] != candidate.hash[candidate.lane[(env, seed)]]})
            if drifted:
                raise SystemExit(f"ERROR: the try of this binary differs from its fleet lanes on {len(drifted)} environments: {', '.join(drifted)}")
    for policy, found in baselines.items():
        if set(found) != set(candidate.envs) or any(found[env].steps != candidate.steps or found[env].seeds[env] != candidate.seeds[env] for env in candidate.envs):
            raise SystemExit(f"Recorded {policy} baselines do not match the candidate schedule")
    (candidate_root / "baselines.json").write_text(json.dumps(
        {policy: {env: rec.root.name for env, rec in found.items()} for policy, found in baselines.items()},
        indent=1, sort_keys=True) + "\n")
    sps = candidate.meta["stepsPerSecond"]
    if not judged:
        atlas(candidate, standing(table, candidate))
        print(f"Steps/s: {sps:,.0f}; {budget:.4f}x the budget anchor")
        common = [key for key in candidate.lane if key in reference.lane]
        same = sum(1 for key in common if candidate.hash[candidate.lane[key]] == reference.hash[reference.lane[key]])
        print(f"reward streams identical to the previous best on {same} of {len(common)} shared lanes")
        BEST.write_text(tree_id + "\n")
        REWARD_ANCHOR.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"ADOPT: best is {tree_id}"
    if set(reference.envs) != set(candidate.envs) or reference.seeds != candidate.seeds or reference.steps != candidate.steps:
        raise SystemExit("Schedule differs from the best: ./fleet.py adopt if this change is intended")
    judged_key = file_digest(engine)[:40]
    if all(candidate.hash[candidate.lane[key]] == reference.hash[reference.lane[key]] for key in reference.lane):
        print(f"Steps/s: {sps:,.0f} recorded; paired speed against best: {ratio:.4f}x; every lane's reward stream is identical to the best's", flush=True)
        if ratio >= 1 - MARGIN:
            atlas(candidate, standing(table, candidate))
            BEST.write_text(tree_id + "\n")
            CHAMPION.unlink(missing_ok=True)
            return f"PROMOTE on identity: {tree_id}"
        raise SystemExit(f"REJECT: identical reward streams but {ratio:.4f}x the best's speed; best stays {best_id}")
    selected = {env: reference.seeds[env][SELECTED_SEEDS:] for env in reference.envs}
    j = compare(table, reference, [candidate], [reference], selected, f"judge: seeds {SELECTED_SEEDS}..{len(reference.seeds[reference.envs[0]]) - 1} against the best")
    atlas(candidate, j["rows"])
    fired = guard(candidate, reference, selected, "the best")
    reward_better = j["verdict"] == "BETTER" and not fired
    not_worse = j["verdict"] != "WORSE" and not fired
    anchor_id = REWARD_ANCHOR.read_text().strip()
    if not_worse and not reward_better and anchor_id != best_id:
        anchor_run = Recording(BENCH / anchor_id)
        a = compare(table, reference, [candidate], [anchor_run], selected, f"against the reward anchor {anchor_id[:9]}")
        not_worse = a["verdict"] != "WORSE" and not guard(candidate, anchor_run, selected, "the reward anchor")
    print(f"Steps/s: {sps:,.0f} recorded; paired speed against best: {ratio:.4f}x; {budget:.4f}x the budget anchor")
    if reward_better:
        BEST.write_text(tree_id + "\n")
        REWARD_ANCHOR.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"PROMOTE on reward: {tree_id}"
    if not_worse and ratio >= 1 + MARGIN:
        BEST.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"PROMOTE on speed: {tree_id}"
    if not_worse and ratio >= 1 - MARGIN and learner_bytes(tree) < learner_bytes(reference.root / "source"):
        BEST.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"PROMOTE on simplicity: {tree_id}; learner source falls from {learner_bytes(reference.root / 'source'):,} to {learner_bytes(tree):,} bytes"
    if not_worse and fuzz_candidate < fuzz_best:
        BEST.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"PROMOTE on repair: {tree_id}; fuzz failures fall from {fuzz_best} to {fuzz_candidate}"
    if CHAMPION.exists() and CHAMPION.read_text().strip() == judged_key:
        CHAMPION.unlink()
        raise SystemExit(f"REJECT: best stays {best_id}; champion {judged_key[:9]} dethroned")
    raise SystemExit(f"REJECT: best stays {best_id}")


@contextmanager
def logging_to(path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def jobs():
    if not QUEUE.is_dir():
        return []
    return sorted(p for p in QUEUE.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "job.json").exists())


def pending(verbs):
    for job in jobs():
        if not (job / "started").exists() and json.loads((job / "job.json").read_text())["verb"] in verbs:
            return job
    return None


def run(job):
    verb = json.loads((job / "job.json").read_text())["verb"]
    (job / "started").write_text(stamp() + "\n")
    ACTIVE.append(job)
    with logging_to(job / "log"):
        print(f"{stamp()} {verb} {job.name}", flush=True)
        try:
            if (job / "cancel").exists():
                raise SystemExit("CANCELLED")
            verdict = execute(job / "tree", verb)
        except SystemExit as stop:
            verdict = str(stop)
        except Exception as error:
            traceback.print_exc()
            verdict = f"ERROR: {error}"
        print(f"{stamp()} {verdict}", flush=True)
    ACTIVE.pop()
    (job / "pid").unlink(missing_ok=True)
    (job / "verdict").write_text(verdict + "\n")
    with open(VERDICTS, "a") as log:
        log.write(f"{stamp()} {verb} {job.name}: {verdict}\n")
    subprocess.run(["osascript", "-e", f"display notification {json.dumps(verdict[:240])} with title {json.dumps('OaK ' + verb)}"], capture_output=True)
    shutil.rmtree(job / "tree", ignore_errors=True)


def requeue():
    for job in jobs():
        if (job / "started").exists() and not (job / "verdict").exists():
            pid = job / "pid"
            if pid.exists():
                orphan = int(pid.read_text())
                if any(p == orphan and name == "oak" for p, _, name in processes()):
                    os.kill(orphan, signal.SIGKILL)
                pid.unlink()
            (job / "started").unlink()
            print(f"{stamp()} requeued {job.name}", flush=True)


def locked(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def serve():
    if os.getpriority(os.PRIO_PROCESS, 0) > 0:
        print(f"{stamp()} refusing to run at reduced priority (nice {os.getpriority(os.PRIO_PROCESS, 0)})", flush=True)
        return
    BENCH.mkdir(exist_ok=True)
    lock = open(RUNNER_LOCK, "w")
    if not locked(lock):
        return
    sys.stdout.reconfigure(line_buffering=True)
    requeue()
    while True:
        job = pending(TIMING) or pending(("judge", "adopt"))
        if job is not None:
            run(job)
            continue
        fcntl.flock(lock, fcntl.LOCK_UN)
        if pending(VERBS) is None or not locked(lock):
            return


def wake():
    if os.getpriority(os.PRIO_PROCESS, 0) > 0:
        return False
    BENCH.mkdir(exist_ok=True)
    with open(RUNNER_LOG, "a") as log:
        subprocess.Popen([sys.executable, str(REPO / "fleet.py"), "serve"], cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return True


def submit(verb):
    QUEUE.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{verb}"
    staged = QUEUE / f".{name}"
    snapshot(staged / "tree")
    (staged / "job.json").write_text(json.dumps({"verb": verb, "submitted": stamp()}) + "\n")
    staged.rename(QUEUE / name)
    return QUEUE / name


def follow(job):
    log, offset, checked = job / "log", 0, time.monotonic()
    while True:
        done = (job / "verdict").exists()
        if log.exists():
            with open(log) as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            print(chunk, end="", flush=True)
        if done:
            return
        if time.monotonic() - checked > 10:
            checked = time.monotonic()
            with open(RUNNER_LOCK, "a") as lock:
                if locked(lock):
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    wake()
        time.sleep(0.5)


def main():
    verb = sys.argv[1] if len(sys.argv) > 1 else "judge"
    if verb == "serve":
        serve()
        return
    if verb not in VERBS:
        raise SystemExit(f"usage: ./fleet.py [{'|'.join(VERBS)}]")
    job = submit(verb)
    if not wake():
        print(f"queued {job.name}, but this shell runs at reduced priority (nice {os.getpriority(os.PRIO_PROCESS, 0)}), so no runner was started; run ./fleet.py from a normal-priority shell", flush=True)
    if verb in TIMING:
        try:
            follow(job)
        except KeyboardInterrupt:
            (job / "cancel").touch()
            print(f"\ncancelling {job.name} at its next timed run")
    else:
        print(f"queued {verb} as {job.name}; its log is {(job / 'log').relative_to(REPO)} and the verdict lands in {VERDICTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
