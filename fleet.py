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
ANCHOR = "4262868f4bce5ea25ac4905e60742f75d11be371"
RECORDED_ENGINE = "oak"
STEPS = 1_000_000
TERMINAL = 100_000
MARGIN = 0.01
ALPHA = 0.05
BOOTSTRAP = 10_000
POWER = 0.8
PROBE_STEPS = 20_000
PROBE_BLOCKS = 3
TIEBREAK_SEEDS = 4
POLICIES = ("random", "ppo", "oracle")
POLICY_SOURCES = {"random": ("Sources/random.c",), "ppo": ("Sources/ppo.c",), "oracle": ()}
ENV_LAYER = ("Sources/env.h", "Sources/rng.h", "Sources/portable.h")
SNAPSHOT = ("Sources", "catalog.py", "setup.sh")
ENV_ROW = re.compile(r'\s*X\(\s*\w+\s*,\s*(\w+)\s*,.*?"([^"]+)"')
VERBS = ("judge", "adopt", "probe", "trace", "try")
TIMING = ("probe", "trace", "try")
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
        out = subprocess.run(["caffeinate", "-i", str(engine), "record", "oak", str(PROBE_STEPS), str(root)],
                             capture_output=True, text=True, check=True, cwd=REPO).stdout
    streams = array("Q", (root / "hash.u64").read_bytes())
    shutil.rmtree(root, ignore_errors=True)
    return float(out.rsplit(":", 1)[1].split()[0]), streams


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


def rho_of(a, b):
    ma, mb = mean(a), mean(b)
    va = mean([(x - ma) ** 2 for x in a])
    vb = mean([(y - mb) ** 2 for y in b])
    cov = mean([(x - ma) * (y - mb) for x, y in zip(a, b)])
    return min(1.0, cov / math.sqrt(va * vb)) if va > 0 and vb > 0 else 0.0


def mdr(rho, tests, n):
    normal = statistics.NormalDist()
    return (normal.inv_cdf(1 - ALPHA / tests) + normal.inv_cdf(POWER)) * math.sqrt(2 * (1 - rho)) / (2 * normal.inv_cdf(0.75) * math.sqrt(n))


def report(candidate, baselines, best):
    cols = ["env", "oak", "random", "oracle", "score", "ppo", "win>ppo"]
    if best is not None:
        cols += ["best", "d", "win>best", "guard p", "rho", "mdr"]
    else:
        cols += ["mdr"]
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
            rho = rho_of(oak, ref)
            row += [mean(ref), iqm(d), wins(oak, ref), p, rho, mdr(rho, len(candidate.envs), len(oak))]
            d_groups.append(d)
            diff_groups.append(diffs)
            pvalues.append(p)
            win_best.append(wins(oak, ref))
        else:
            row += [mdr(0.0, len(candidate.envs), len(oak))]
        rows.append(row)
    table_print(cols, rows)
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


def sign_tail(wins, losses):
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0


def table_print(cols, rows):
    widths = [max(len(c), max(len(fmt(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print(" | ".join(fmt(v).rjust(w) if i else fmt(v).ljust(w) for i, (v, w) in enumerate(zip(r, widths))))


def keep_engine(engine, root):
    staged = root / f".{RECORDED_ENGINE}.partial"
    shutil.copy2(engine, staged)
    staged.rename(root / RECORDED_ENGINE)


def attempted(engine, root, first, count, tree=None):
    if complete(root):
        print(f"already tried: {root.name[:9]} (seeds {first}..{first + count - 1})", flush=True)
    else:
        shutil.rmtree(root, ignore_errors=True)
        staged = root.with_name(root.name + ".source")
        if tree is not None:
            snapshot(staged, tree)
        with quiet():
            subprocess.run(["caffeinate", "-i", str(engine), "try", "oak", str(STEPS), str(first), str(count), str(root)], check=True, cwd=REPO)
        if tree is not None:
            staged.rename(root / "source")
            keep_engine(engine, root)
    return Recording(root)


def compare(table, best, candidates, references):
    links = json.loads((best.root / "baselines.json").read_text())
    loaded = {}

    def linked(policy, env):
        name = links[policy][env]
        if name not in loaded:
            loaded[name] = Recording(BENCH / name)
        return loaded[name]

    def lanes(recordings):
        found = {}
        for rec in recordings:
            for key, lane in rec.lane.items():
                found[key] = (rec.score[lane], rec.hash[lane], rec.health[lane])
        return found

    cand, ref = lanes(candidates), lanes(references)
    category = {row["envId"]: row["category"] for row in table}
    rows = []
    for env in candidates[0].envs:
        keys = sorted(key for key in cand if key[0] == env)
        low, high = mean(linked("random", env).scores(env)), mean(linked("oracle", env).scores(env))
        ppo = linked("ppo", env)

        def norm(value):
            return (value - low) / (high - low)

        same = all(cand[key][1] == ref[key][1] for key in keys)
        row = {"env": env, "category": category[env], "cand": mean([norm(cand[key][0]) for key in keys]), "ref": mean([norm(ref[key][0]) for key in keys]),
               "best": mean([norm(best.score[best.lane[key]]) for key in keys]), "ppo": mean([norm(ppo.score[ppo.lane[key]]) for key in keys]),
               "same": same, "dead": any(cand[key][2] != 0xffffffff for key in keys)}
        row["d"] = 0.0 if same else row["cand"] - row["ref"]
        rows.append(row)
    rows.sort(key=lambda r: (r["category"], r["env"]))
    table_print(["env", "category", "oak", "ref", "best", "ppo", "d"],
                [[r["env"], r["category"], r["cand"], r["ref"], r["best"], r["ppo"], "=" if r["same"] else r["d"]] for r in rows])
    groups = [("suite", rows)] + [(name, [r for r in rows if r["category"] == name]) for name in sorted(set(category.values()))]
    better, worse, summary = [], [], []
    for name, members in groups:
        wins, losses = sum(1 for r in members if r["d"] > 0), sum(1 for r in members if r["d"] < 0)
        better.append(sign_tail(wins, losses))
        worse.append(sign_tail(losses, wins))
        summary.append([name, len(members)] + [mean([r[k] for r in members]) for k in ("cand", "ref", "best", "ppo")] + [wins, losses, better[-1], worse[-1]])
    print()
    table_print(["group", "n", "oak", "ref", "best", "ppo", "wins", "losses", "p better", "p worse"], summary)
    stats = {}
    for key, name in (("cand", "oak"), ("ref", "ref")):
        values = [r[key] for r in rows]
        stats[key] = (mean(values), iqm(values), quantile(sorted(values), 0.5))
        print(f"{name}: mean {stats[key][0]:+.4f}  IQM {stats[key][1]:+.4f}  median {stats[key][2]:+.4f}  below Random {sum(1 for v in values if v < 0)}  above PPO {sum(1 for r in rows if r[key] > r['ppo'])}")
    for rec in candidates:
        sps = rec.meta["stepsPerSecond"]
        cores = min(os.cpu_count() or 1, rec.lanes)
        print(f"{rec.root.name[:12]}: dead lanes {sum(1 for h in rec.health if h != 0xffffffff)} of {rec.lanes}   {sps:,.0f} steps/s   {cores / sps * 1e6:.3f} us per step per core   {rec.lanes * rec.steps / sps:.1f} s")
    if all(r["same"] for r in rows):
        verdict = "IDENTICAL"
    elif any(holm(worse)):
        verdict = "WORSE"
    elif any(holm(better)):
        verdict = "BETTER"
    else:
        verdict = "NO SIGNAL"
    closer = all(c > r for c, r in zip(stats["cand"], stats["ref"]))
    return verdict, closer, summary[0]


def attempt(engine, tree, table, best):
    TRIES.mkdir(parents=True, exist_ok=True)
    key = file_digest(engine)[:40]
    candidate = attempted(engine, TRIES / key, 0, 1, tree)
    champion = CHAMPION.read_text().strip() if CHAMPION.exists() else ""
    reference = Recording(TRIES / champion) if champion else best
    label = f"champion {champion[:9]}" if champion else f"best {best.root.name[:9]}"
    verdict, closer, suite = compare(table, best, [candidate], [reference])
    if verdict == "NO SIGNAL" and closer:
        print(f"\ntie-break: mean, IQM and median all improved; adding seeds 1..{TIEBREAK_SEEDS - 1} of every block", flush=True)
        wide = attempted(engine, TRIES / f"{key}+1+{TIEBREAK_SEEDS - 1}", 1, TIEBREAK_SEEDS - 1)
        if champion:
            referenceWide = attempted(TRIES / champion / "oak", TRIES / f"{champion}+1+{TIEBREAK_SEEDS - 1}", 1, TIEBREAK_SEEDS - 1)
            references = [reference, referenceWide]
        else:
            references = [best]
        verdict, closer, suite = compare(table, best, [candidate, wide], references)
        verdict = f"{verdict} on {TIEBREAK_SEEDS} seeds"
    if verdict.startswith(("IDENTICAL", "BETTER")):
        CHAMPION.write_text(key + "\n")
    return f"{verdict}: try {key[:9]} against {label}; suite {suite[6]} wins, {suite[7]} losses"


def execute(tree, verb):
    best_id = BEST.read_text().strip() if BEST.exists() else ""
    if not best_id:
        raise SystemExit("No recorded best to judge against")
    reference = Recording(BENCH / best_id)
    subprocess.run([tree / "setup.sh"], cwd=tree, check=True)
    engine = tree / ".build/oak"
    tree_id = hashlib.sha256(digest(("Sources", "setup.sh"), tree).encode()).hexdigest()[:40]
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
        if changed and verb == "try":
            raise SystemExit(f"REFUSE: environment sources changed since the recorded best ({', '.join(changed)}); a try compares agents only")
        if changed:
            print(f"Environment sources changed since the recorded best ({len(changed)} environments); comparing on the same seeds and horizon: {', '.join(changed)}", flush=True)
    if verb == "trace":
        trace(engine)
        return f"trace: sampled {TRACE_SECONDS} s of the working tree's engine"
    if verb == "try":
        return attempt(engine, tree, table, reference)
    other = reference_engine(reference)
    if judged and filecmp.cmp(engine, other, shallow=False):
        if verb == "probe":
            return f"probe: binary identical to recorded {best_id[:9]}"
        raise SystemExit(f"REJECT: binary identical to recorded {best_id[:9]}; nothing to judge")
    if judged:
        timing = (other, reference)
    else:
        anchor = Recording(BENCH / ANCHOR)
        timing = (other if best_id == ANCHOR else reference_engine(anchor), anchor)
    ratio = speed_stage(engine, timing[0], timing[1], judged)
    if verb == "probe":
        return f"probe: {ratio:.4f}x against recorded {best_id[:9]}"
    if ratio < 1 - MARGIN:
        raise SystemExit(f"{'REJECT' if judged else 'REFUSE'}: paired speed {ratio:.4f}x against recorded {timing[1].root.name[:9]}; best stays {best_id}")
    baselines = {policy: baseline(policy, digests, schedule, engine, tree) for policy in POLICIES}
    candidate_root = BENCH / tree_id
    sources = {"policy": "oak", "envs": digests, "policyDigest": digest(("Sources",), tree)}
    if complete(candidate_root):
        (candidate_root / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
        snapshot(candidate_root / "source", tree)
        candidate = Recording(candidate_root)
    else:
        candidate = record(engine, "oak", candidate_root, sources, tree)
    tried = TRIES / file_digest(engine)[:40]
    if complete(tried):
        attempted = Recording(tried)
        drifted = [env for env in attempted.envs if attempted.hash[attempted.lane[(env, attempted.seeds[env][0])]] != candidate.hash[candidate.lane[(env, attempted.seeds[env][0])]]]
        if drifted:
            raise SystemExit(f"ERROR: the try of this binary differs from its fleet lanes on {len(drifted)} environments: {', '.join(drifted)}")
    for policy, found in baselines.items():
        if set(found) != set(candidate.envs) or any(found[env].steps != candidate.steps or found[env].seeds[env] != candidate.seeds[env] for env in candidate.envs):
            raise SystemExit(f"Recorded {policy} baselines do not match the candidate schedule")
    (candidate_root / "baselines.json").write_text(json.dumps(
        {policy: {env: rec.root.name for env, rec in found.items()} for policy, found in baselines.items()},
        indent=1, sort_keys=True) + "\n")
    if judged and (set(reference.envs) != set(candidate.envs) or reference.seeds != candidate.seeds or reference.steps != candidate.steps):
        raise SystemExit("Schedule differs from the best: ./fleet.py adopt if this change is intended")
    verdict = report(candidate, baselines, reference if judged else None)
    sps = candidate.meta["stepsPerSecond"]
    if not judged:
        print(f"Steps/s: {sps:,.0f}")
        common = [key for key in candidate.lane if key in reference.lane]
        same = sum(1 for key in common if candidate.hash[candidate.lane[key]] == reference.hash[reference.lane[key]])
        print(f"reward streams identical to the previous best on {same} of {len(common)} shared lanes")
        BEST.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"ADOPT: best is {tree_id}"
    print(f"Steps/s: {sps:,.0f} recorded; paired speed against best: {ratio:.4f}x")
    reward_better = verdict["lo"] > 0 and not verdict["guard"]
    reward_not_worse = verdict["identical"] or (verdict["lo"] >= -MARGIN and not verdict["guard"])
    if reward_not_worse and ((reward_better and ratio >= 1 - MARGIN) or ratio >= 1 + MARGIN):
        BEST.write_text(tree_id + "\n")
        CHAMPION.unlink(missing_ok=True)
        return f"PROMOTE: {tree_id}"
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
        raise SystemExit("usage: ./fleet.py [adopt|probe|trace|try]")
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
