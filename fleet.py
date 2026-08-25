#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
from rliable import metrics as rl_metrics
from scipy.optimize import isotonic_regression
from statsmodels.stats.proportion import proportion_confint

REPO = Path(__file__).resolve().parent
BENCH = REPO / ".oak-benchmarks"
BASELINES = Path(os.environ.get("OAK_BASELINES_DIR") or REPO / ".oak-baselines")
SCALE_PATH = REPO / "scale-v1.json"

BOOT_REPS = 50_000
CURVE_REPS = 2000
BOOT_BASE_SEED = 20260817
BOOT_METHOD = ("rliable stratified percentile bootstrap over runs; "
               "independent x/y resampling for probImprovement; "
               "Wilson interval for divergence rate; Greenwood variance for RMST")
TAU_GRID = [round(-0.25 + 0.025 * k, 6) for k in range(61)]
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
    if len(data) < 12:
        raise SystemExit(f"{path}: not an OAKT tape")
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
    mats = np.stack(blocks)
    return {"fields": fields, "lanes": lanes, "ends": np.asarray(ends, np.int64),
            "wall": np.asarray(wall, np.float64), "data": mats, "bits": mats.view(np.uint32)}


def load(root: Path) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    if "policy" not in meta:
        raise SystemExit(f"{root}: meta.json has no policy")
    tp = root / "tape.bin"
    if not tp.exists():
        raise SystemExit(f"{root}: no tape.bin")
    tape = read_tape(tp)
    di = tape["fields"].index("deathStep")
    ds, db = tape["data"][-1, di, :], tape["bits"][-1, di, :]
    deaths = {int(l): int(db[l]) for l in np.nonzero(np.isfinite(ds))[0]}
    return {"meta": meta, "tape": tape, "deaths": deaths,
            "seeds": meta["seeds"], "root": root}


def series(run: dict, field: str) -> np.ndarray:
    return run["tape"]["data"][:, run["tape"]["fields"].index(field), :]


def last_finite(mat: np.ndarray) -> np.ndarray:
    W, B = mat.shape
    mask = np.isfinite(mat)
    idx = W - 1 - np.argmax(mask[::-1], axis=0)
    v = mat[idx, np.arange(B)]
    return np.where(mask.any(axis=0), v, np.nan)


def env_slices(run: dict) -> dict[str, np.ndarray]:
    seed_lane = {s: i for i, s in enumerate(run["seeds"])}
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
    ends, jr = window_ctx(run)
    rs_all = series(run, "rewardSum")[:, lanes]
    n = len(lanes)
    p5, p50, spread = (np.full(n, np.nan) for _ in range(3))
    if not (jr >= 0).any() or rs_all.shape[0] < 8:
        return p5, p50, spread
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
            p50[li] = np.percentile(ret[ok], 50)
        spread[li] = float(np.percentile(pk, 95) - np.percentile(pk, 50))
    return p5, p50, spread


def analyze(run: dict, lanes: np.ndarray) -> dict:
    meta, horizon = run["meta"], run["meta"]["steps"]
    ends = run["tape"]["ends"]
    max_step = int(ends[-1])
    rs = series(run, "rewardSum")[:, lanes]
    delivered = last_finite(rs) / max_step
    blind = ~np.isfinite(delivered)
    worst = min(0.0, float(np.min(delivered[~blind]))) if (~blind).any() else 0.0
    delivered = np.where(blind, worst, delivered)
    died = np.array([1.0 if int(l) in run["deaths"] else 0.0 for l in lanes])
    dtimes = np.array([run["deaths"].get(int(l), horizon) for l in lanes], dtype=float)
    peak, end_rate = windowed_rates(run, lanes)
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = np.where(peak > 0, end_rate / peak, np.nan)
    null_p5, null_p50, peak_spread = retention_null(run, lanes)
    learned = (np.isfinite(peak) & np.isfinite(peak_spread)
              & (peak > 0) & (peak > peak_spread))
    decayed = (learned & np.isfinite(retention) & np.isfinite(null_p5)
               & (retention < null_p5))
    retained = learned & ~decayed
    ret_margin = retention - null_p50
    return {
        "seeds": len(lanes), "horizon": horizon, "maxStep": max_step,
        "complete": max_step >= horizon,
        "rewardRateIQM": iqm(delivered),
        "positiveFraction": float((delivered > 0).mean()),
        "divergence": {"diverged": int(died.sum())},
        "learnedSeeds": int(learned.sum()),
        "retentionFraction": (float(retained.sum() / learned.sum())
                              if learned.any() else None),
        "_vectors": {"delivered": delivered, "died": died, "dtimes": dtimes,
                     "learned": learned.astype(float), "retained": retained.astype(float),
                     "decayed": decayed.astype(float), "retMargin": ret_margin},
    }


def iqm(v: np.ndarray) -> float:
    return float(rl_metrics.aggregate_iqm(np.asarray(v, float)))


def impute_worst(v: np.ndarray, high_is_worst: bool = False) -> np.ndarray:
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    if fin.all() or not fin.any():
        return v
    worst = float(v[fin].max()) if high_is_worst else float(v[fin].min())
    return np.where(fin, v, worst)


def rmst(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    if not events.any():
        return float(horizon)
    km = KaplanMeierFitter().fit(np.minimum(times, horizon), event_observed=events)
    return float(restricted_mean_survival_time(km, t=float(horizon)))


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


U32 = 0xFFFFFFFF
PHILOX_M0, PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
PHILOX_W0, PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


def philox4x32(key0: int, key1: int, c0: int, c1: int, c2: int, c3: int) -> tuple[int, int, int, int]:
    for r in range(10):
        if r > 0:
            key0 = (key0 + PHILOX_W0) & U32
            key1 = (key1 + PHILOX_W1) & U32
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ key0, p1 & U32, (p0 >> 32) ^ c3 ^ key1, p0 & U32
    return c0, c1, c2, c3


def procedural_maze_walls(seed: int) -> list[int]:
    walls = [U32, U32, 0x1FFFF]
    for j in range(5):
        for i in range(5):
            idx = 2 * i + 9 * (2 * j)
            walls[idx >> 5] &= ~(1 << (idx & 31)) & U32
    visited, stack = 1, [0]
    counter, word_index = 0, 0
    block = philox4x32(seed, 0, 0, 0, 0, 0)
    while stack:
        ci, cj = stack[-1] % 5, stack[-1] // 5
        neighbors = [ni + 5 * nj
                     for ni, nj in ((ci + 1, cj), (ci, cj + 1), (ci - 1, cj), (ci, cj - 1))
                     if 0 <= ni < 5 and 0 <= nj < 5 and not visited >> (ni + 5 * nj) & 1]
        if not neighbors:
            stack.pop()
            continue
        if word_index == 4:
            counter += 1
            block = philox4x32(seed, 0, counter, 0, 0, 0)
            word_index = 0
        nxt = neighbors[block[word_index] % len(neighbors)]
        word_index += 1
        idx = (ci + nxt % 5) + 9 * (cj + nxt // 5)
        walls[idx >> 5] &= ~(1 << (idx & 31)) & U32
        visited |= 1 << nxt
        stack.append(nxt)
    return walls


def mean_pair_distance(cells: list[tuple[int, int]]) -> float:
    cell_set = set(cells)
    total = 0
    for src in cells:
        dist = {src: 0}
        frontier = [src]
        depth = 0
        while frontier:
            depth += 1
            nxt = []
            for x, y in frontier:
                for cand in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                    if cand in cell_set and cand not in dist:
                        dist[cand] = depth
                        nxt.append(cand)
            frontier = nxt
        if len(dist) != len(cells):
            raise SystemExit(f"disconnected grid from {src}: {len(dist)} of {len(cells)} reached")
        total += sum(dist.values())
    return total / (len(cells) * (len(cells) - 1))


def four_rooms_ceiling() -> tuple[float, str]:
    def wall(x: int, y: int) -> bool:
        return (x <= 0 or x >= 10 or y <= 0 or y >= 10
                or (x == 5 and y not in (2, 8)) or (y == 5 and x not in (2, 8)))

    cells = [(x, y) for x in range(1, 10) for y in range(1, 10) if not wall(x, y)]
    if len(cells) != 68:
        raise SystemExit(f"FourRooms free-cell count {len(cells)} != 68")
    mean_d = mean_pair_distance(cells)
    return 1.0 / mean_d, (
        f"1/E[BFS d(u,v)] = 1/{mean_d:.6f} over ordered pairs of the 68 free cells; "
        "agent sits at the reached goal and the next goal is uniform over the other 67 cells, "
        "so episode starts are uniform over ordered pairs")


def key_corridor_ceiling() -> tuple[float, str]:
    def wall(x: int, y: int) -> bool:
        return x <= 0 or x >= 8 or y <= 0 or y >= 6 or (x == 4 and y != 3)

    left = [(1 + i % 3, 1 + i // 3) for i in range(15)]
    right = [(5 + i % 3, 1 + i // 3) for i in range(15)]
    total = 0
    for key in left:
        for goal in right:
            for agent in left:
                start = (agent, False, False)
                seen = {start}
                frontier = [start]
                depth, reached = 0, None
                while reached is None:
                    if not frontier:
                        raise SystemExit(f"KeyCorridor unreachable goal {goal} from {agent} key {key}")
                    depth += 1
                    nxt = []
                    for (x, y), has_key, door_open in frontier:
                        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            px, py, pk, pd = x + dx, y + dy, has_key, door_open
                            if wall(px, py):
                                px, py = x, y
                            elif (px, py) == (4, 3) and not door_open:
                                if has_key:
                                    pd = True
                                else:
                                    px, py = x, y
                            if not pk and (px, py) == key:
                                pk = True
                            if (px, py) == goal:
                                reached = depth
                                break
                            state = ((px, py), pk, pd)
                            if state not in seen:
                                seen.add(state)
                                nxt.append(state)
                        if reached is not None:
                            break
                    frontier = nxt
                total += reached
    mean_d = total / (15 * 15 * 15)
    return 1.0 / mean_d, (
        f"1/E[episode] = 1/{mean_d:.6f}; exact BFS over (x,y,hasKey,doorUnlocked) "
        "for all 15^3 spawn triples: agent+key uniform in the left 3x5 room, "
        "goal uniform in the right 3x5 room, door at (4,3)")


def multi_key_corridor_ceiling() -> tuple[float, str]:
    def wall(x: int, y: int) -> bool:
        return x <= 0 or x >= 16 or y <= 0 or y >= 6 or (x in (4, 8, 12) and y != 3)

    def room(x0: int) -> list[tuple[int, int]]:
        return [(x0 + i % 3, 1 + i // 3) for i in range(15)]

    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def revisit_cost(cell: tuple[int, int]) -> int:
        x, y = cell
        blocked = any(wall(x + dx, y + dy) or (x + dx, y + dy) in ((4, 3), (8, 3), (12, 3))
                      for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)))
        return 1 if blocked else 2

    doors = ((4, 3), (8, 3), (12, 3))
    seg_agent_key = sum(manhattan(a, k) if a != k else revisit_cost(k)
                        for a in room(1) for k in room(1)) / 225
    seg_key1_door = sum(manhattan(k, doors[0]) for k in room(1)) / 15
    seg_mid = [sum(manhattan(doors[d], k) + manhattan(k, doors[d + 1]) for k in room(x0)) / 15
               for d, x0 in ((0, 5), (1, 9))]
    seg_goal = sum(manhattan(doors[2], g) for g in room(13)) / 15
    mean_d = seg_agent_key + seg_key1_door + seg_mid[0] + seg_mid[1] + seg_goal
    return 1.0 / mean_d, (
        f"1/E[episode] = 1/{mean_d:.6f}; forced serial route key1-door1-key2-door2-key3-door3-goal "
        "with Manhattan segments (rooms are open 3x5 boxes), agent/keys/goal uniform per room; "
        "spawning on key1 costs 1 extra step beside a wall or locked door, else 2")


def high_dim_maze_ceiling() -> tuple[float, str]:
    def wall(x: int, y: int) -> bool:
        return (x <= 0 or x >= 10 or y <= 0 or y >= 10
                or (x == 3 and y != 9) or (x == 7 and y != 1))

    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def slide(x: int, y: int, d: int) -> tuple[int, int]:
        dx, dy = moves[d]
        return (x, y) if wall(x + dx, y + dy) else (x + dx, y + dy)

    cells = [(x, y) for x in range(1, 10) for y in range(1, 10) if not wall(x, y)]
    if len(cells) != 65:
        raise SystemExit(f"HighDimMaze free-cell count {len(cells)} != 65")
    total = 0
    for src in cells:
        dist = {(src, d): 0 for d in range(4)}
        best = {src: 0}
        frontier = list(dist)
        depth = 0
        while frontier:
            depth += 1
            nxt = []
            for (x, y), d in frontier:
                steps = [(slide(x, y, a), a) for a in range(4)]
                dash1 = slide(x, y, d)
                if dash1 != (x, y):
                    steps.append((slide(*dash1, d), d))
                for pos, nd in steps:
                    state = (pos, nd)
                    if state not in dist:
                        dist[state] = depth
                        nxt.append(state)
                        if pos not in best:
                            best[pos] = depth
            frontier = nxt
        if len(best) != len(cells):
            raise SystemExit(f"HighDimMaze disconnected from {src}")
        total += sum(best.values())
    mean_d = total / (len(cells) * (len(cells) - 1))
    return 1.0 / mean_d, (
        f"1/E[BFS d(u,v)] = 1/{mean_d:.6f} over ordered pairs of the 65 free cells, "
        "states (x,y,lastDir) with dash edges, initial lastDir chosen freely per episode: "
        "an upper bound since the carried-over lastDir may not be the best one")


def procedural_maze_ceiling(seeds: list[int]) -> tuple[float, str]:
    values = []
    for seed in seeds:
        walls = procedural_maze_walls(seed)

        def is_wall(x: int, y: int) -> bool:
            if x <= 0 or x >= 10 or y <= 0 or y >= 10:
                return True
            idx = (x - 1) + 9 * (y - 1)
            return walls[idx >> 5] >> (idx & 31) & 1 != 0

        cells = [(x, y) for x in range(1, 10) for y in range(1, 10) if not is_wall(x, y)]
        if len(cells) != 49:
            raise SystemExit(f"ProceduralMaze seed {seed}: {len(cells)} free cells != 49")
        values.append(1.0 / mean_pair_distance(cells))
    return iqm(np.array(values)), (
        f"IQM over the {len(seeds)} keeper-slice seeds of per-seed 1/E[BFS d(u,v)] "
        "on the exact philox+DFS maze (49 free cells each), spawn and goal uniform over free cells")


def closed_form_ceilings() -> dict[str, tuple[float, str]]:
    return {
        "Chain-cont-v0": (1.0 / 15.0,
                          "press the advancing action every step; reward fires when pos reaches 15, "
                          "a 15-step cycle"),
        "DelayedChain-cont-v0": (1.0 / 31.0,
                                 "press the advancing action every step; reward fires when pos reaches 31, "
                                 "a 31-step cycle"),
        "MemoryChain-cont-v0": (1.0 / 32.0,
                                "31 advancing steps to pos 31 plus the cue-answer step, "
                                "a 32-step cycle at one reward"),
        "DeepSea-cont-v0": ((1.0 - 3.5 * 0.00125) / 7.0,
                            "dive right all 7 rows for the +1; the -0.00125 action-bit cost hits on "
                            "E=3.5 of 7 diagonal cells over the per-seed philox map"),
        "LargeDeepSea-cont-v0": ((1.0 - 7.5 * 0.000625) / 15.0,
                                 "dive right all 15 rows for the +1; the -0.000625 action-bit cost hits on "
                                 "E=7.5 of 15 diagonal cells over the per-seed philox map"),
        "CartPole-cont-v0": (1.0,
                             "reward is 1.0 every step for every policy; degenerate env, "
                             "floor forced to analytic 0.0"),
        "Pendulum-cont-v0": (0.0,
                             "cost theta^2 + 0.1 thetaDot^2 + 0.001 torque^2 is nonnegative and the "
                             "upright rest state is a zero-cost fixed point"),
    }


def hash_env_sources(h) -> None:
    for rel in ("EnvCore.metal", "Env.metal", "Fleet.metal", "Baseline.metal"):
        h.update((REPO / "Sources" / rel).read_bytes())
    for p in sorted((REPO / "Sources" / "envs").glob("*.metal")):
        h.update(p.read_bytes())


def env_sources_hash() -> str:
    h = hashlib.sha256()
    hash_env_sources(h)
    return h.hexdigest()


def delivered_rates(run: dict, lanes: np.ndarray) -> np.ndarray:
    return last_finite(series(run, "rewardSum")[:, lanes]) / int(run["tape"]["ends"][-1])


def delivered_iqm(run: dict, lanes: np.ndarray) -> float:
    rates = delivered_rates(run, lanes)
    if not np.isfinite(rates).all():
        raise SystemExit(f"{run['root']}: blind lanes in delivered rates")
    return iqm(rates)


def load_scale() -> dict:
    if not SCALE_PATH.exists():
        raise SystemExit(f"{SCALE_PATH} missing; run ./fleet.py scale")
    table = json.loads(SCALE_PATH.read_text())
    if table["envSourcesHash"] != env_sources_hash():
        raise SystemExit(f"{SCALE_PATH} stale: env sources changed; run ./fleet.py scale")
    return table


def cmd_scale(args) -> int:
    kh = keeper_hash()
    if kh is None or not (BENCH / kh).is_dir():
        raise SystemExit("scale needs a crowned keeper")
    keeper = load(BENCH / kh)
    runs = {}
    for policy in ("random", "ppo"):
        name = baseline_pointer(policy)
        if name is None:
            raise SystemExit(f"scale needs the {policy} baseline; run ./fleet.py baselines")
        runs[policy] = load(BASELINES / name)
        if (runs[policy]["meta"]["envSchedule"] != keeper["meta"]["envSchedule"]
                or runs[policy]["seeds"] != keeper["seeds"]):
            raise SystemExit(f"{policy} baseline schedule differs from keeper")
    slices = env_slices(keeper)
    catalog_count = len(keeper["meta"]["envCatalog"])
    if len(slices) != catalog_count:
        raise SystemExit(
            f"keeper schedule has {len(slices)} envs, catalog declares {catalog_count}")
    proc_seeds = [keeper["seeds"][l] for l in slices["ProceduralMaze-cont-v0"]]
    analytic = closed_form_ceilings()
    enumerated = {
        "FourRooms-cont-v0": four_rooms_ceiling,
        "KeyCorridor-cont-v0": key_corridor_ceiling,
        "MultiKeyCorridor-cont-v0": multi_key_corridor_ceiling,
        "HighDimMaze-cont-v0": high_dim_maze_ceiling,
        "ProceduralMaze-cont-v0": lambda: procedural_maze_ceiling(proc_seeds),
    }
    envs = {}
    for env, lanes in slices.items():
        floor, floor_source = delivered_iqm(runs["random"], lanes), "random-iqm"
        ppo_measured = delivered_iqm(runs["ppo"], lanes)
        keeper_measured = delivered_iqm(keeper, lanes)
        if env in analytic:
            ceiling, derivation = analytic[env]
            source = "closed-form"
        elif env in enumerated:
            ceiling, derivation = enumerated[env]()
            source = "enumerated"
        else:
            ceiling, source = ppo_measured, "ppo-iqm"
            derivation = "lifetime delivered-rate IQM of the crowned ppo baseline on this env's lanes"
        if env == "CartPole-cont-v0":
            floor, floor_source = 0.0, "analytic"
        if source != "ppo-iqm":
            for label, measured in (("ppo", ppo_measured), ("keeper", keeper_measured)):
                if ceiling < measured:
                    raise SystemExit(
                        f"{env}: ceiling {ceiling:.6g} below measured {label} IQM {measured:.6g}")
        if ceiling - floor <= 1e-9 * max(1.0, abs(ceiling), abs(floor)):
            raise SystemExit(f"{env}: ceiling {ceiling:.6g} does not clear floor {floor:.6g}")
        envs[env] = {"floor": floor, "floorSource": floor_source,
                     "ceiling": ceiling, "ceilingSource": source, "derivation": derivation}
        print(f"{env:26} floor {floor:+.6f} ({floor_source})  "
              f"ceiling {ceiling:+.6f} ({source})  ppo {ppo_measured:+.6f}  "
              f"keeper {keeper_measured:+.6f}", flush=True)
    table = {"scale": "v1",
             "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "sourceRunNames": {p: runs[p]["root"].name for p in ("ppo", "random")},
             "envSourcesHash": env_sources_hash(),
             "envs": envs}
    SCALE_PATH.write_text(json.dumps(table, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(SCALE_PATH)
    return 0


def fl(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def fl_list(v) -> list:
    return [fl(x) for x in v]


def row_iqm(mat: np.ndarray) -> np.ndarray:
    return np.array([iqm(impute_worst(row)) for row in mat])


def diag_iqm(v: np.ndarray) -> float | None:
    fin = v[np.isfinite(v)]
    return fl(iqm(fin)) if fin.size else None


def agg_iqm(scores: np.ndarray) -> np.ndarray:
    return np.array([iqm(scores)])


def agg_gap(scores: np.ndarray) -> np.ndarray:
    return np.array([rl_metrics.aggregate_optimality_gap(scores)])


def agg_prob_improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([rl_metrics.probability_of_improvement(x, y)])


def ci_pair(ci: np.ndarray) -> list:
    return [fl(float(ci[0][0])), fl(float(ci[1][0]))]


def interval_estimate(scores: np.ndarray, func, seed: int) -> tuple[float, list]:
    np.random.seed(seed)
    point, ci = rly.get_interval_estimates({"s": scores}, func, reps=BOOT_REPS)
    return float(point["s"][0]), ci_pair(ci["s"])


def rate_stats(rates: np.ndarray, floor: float, span: float, seed: int) -> dict:
    point, ci = interval_estimate(impute_worst(rates)[:, None], agg_iqm, seed)
    return {"iqm": fl(point), "ci95": ci,
            "iqmZ": fl((point - floor) / span),
            "ci95Z": [None if v is None else (v - floor) / span for v in ci]}


def divergence_block(run: dict, lanes: np.ndarray, max_step: int) -> dict:
    died = np.array([1.0 if int(l) in run["deaths"] else 0.0 for l in lanes])
    dtimes = np.minimum(
        np.array([run["deaths"].get(int(l), max_step) for l in lanes], dtype=float), max_step)
    diverged = int(died.sum())
    lo, hi = proportion_confint(diverged, len(lanes), method="wilson")
    km = KaplanMeierFitter().fit(dtimes, event_observed=died)
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


def runs_matrix(cols: list, high_is_worst: bool = False) -> np.ndarray:
    return np.column_stack([impute_worst(c, high_is_worst) for c in cols])


def prob_improvement(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, list]:
    np.random.seed(seed)
    point, ci = rly.get_interval_estimates(
        {"x,y": [x, y]}, agg_prob_improvement, reps=BOOT_REPS)
    return fl(float(point["x,y"][0])), ci_pair(ci["x,y"])


def final_z_by_env(run: dict, slices: dict, scale_envs: dict) -> dict[str, np.ndarray]:
    rates = full_window_rates(run)
    fin_mask = np.isfinite(rates)
    li = rates.shape[0] - 1 - np.argmax(fin_mask[::-1], axis=0)
    end = np.where(fin_mask.any(axis=0), rates[li, np.arange(rates.shape[1])], np.nan)
    out = {}
    for env, lanes in slices.items():
        floor, ceiling = scale_envs[env]["floor"], scale_envs[env]["ceiling"]
        out[env] = (end[lanes] - floor) / (ceiling - floor)
    return out


def vs_baseline(policy: str, seed_base: int, run: dict, slices: dict, scale_envs: dict) -> dict | None:
    name = baseline_pointer(policy)
    if name is None:
        return None
    base = load(BASELINES / name)
    if (base["meta"]["envSchedule"] != run["meta"]["envSchedule"]
            or base["seeds"] != run["seeds"]):
        raise SystemExit(f"{policy} baseline schedule differs from {run['root']}")
    own_z = final_z_by_env(run, slices, scale_envs)
    base_z = final_z_by_env(base, slices, scale_envs)
    env_ids = sorted(slices)
    per_env = {}
    for e, env in enumerate(env_ids):
        p, ci = prob_improvement(runs_matrix([own_z[env]]), runs_matrix([base_z[env]]),
                                 seed_base + e)
        per_env[env] = {"p": p, "ci95": ci}
    mean_p, mean_ci = prob_improvement(runs_matrix([own_z[e] for e in env_ids]),
                                       runs_matrix([base_z[e] for e in env_ids]),
                                       seed_base + len(env_ids))
    return {"run": name,
            "probImprovement": {"mean": mean_p, "ci95": mean_ci, "envs": per_env}}


def cmd_report(args) -> int:
    run = load(args.root)
    scale_table = load_scale()
    scale_envs = scale_table["envs"]
    slices = env_slices(run)
    if set(slices) != set(scale_envs):
        raise SystemExit(f"{args.root}: env set differs from {SCALE_PATH}")
    per_env_counts = {len(lanes) for lanes in slices.values()}
    if len(per_env_counts) != 1:
        raise SystemExit(f"{args.root}: uneven env slices {sorted(per_env_counts)}")
    meta = run["meta"]
    flops_per_step = meta["flopsPerStep"]
    env_flops_per_step = meta["envFlopsPerStep"]
    horizon = meta["steps"]
    ends = run["tape"]["ends"]
    max_step = int(ends[-1])
    n_lanes = run["tape"]["lanes"]
    started = datetime.strptime(meta["startedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    generated_at = (started + timedelta(seconds=float(run["tape"]["wall"][-1]))).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    delivered = last_finite(series(run, "rewardSum")) / max_step
    rates = full_window_rates(run)
    fin_mask = np.isfinite(rates)
    last_row = rates.shape[0] - 1 - np.argmax(fin_mask[::-1], axis=0)
    final_rate = np.where(fin_mask.any(axis=0), rates[last_row, np.arange(n_lanes)], np.nan)
    with np.errstate(invalid="ignore"):
        peak_rate = np.nanmax(np.where(fin_mask, rates, -np.inf), axis=0)
    peak_rate[~np.isfinite(peak_rate)] = np.nan
    env_ids = sorted(slices)
    plast_fields = PLASTICITY_FIELDS[meta["policy"]]
    missing = [f for f in plast_fields if f not in run["tape"]["fields"]]
    if missing:
        raise SystemExit(f"{args.root}: tape lacks plasticity fields {missing}")
    plast_rows = np.unique(np.linspace(0, len(ends) - 1, 100).astype(int))
    plast_series = {f: series(run, f) for f in plast_fields}
    floor_lane = np.empty(n_lanes)
    span_lane = np.empty(n_lanes)
    for env, lanes in slices.items():
        floor_lane[lanes] = scale_envs[env]["floor"]
        span_lane[lanes] = scale_envs[env]["ceiling"] - scale_envs[env]["floor"]
    z_rates = (rates - floor_lane[None, :]) / span_lane[None, :]
    z_life = (delivered - floor_lane) / span_lane
    z_final = (final_rate - floor_lane) / span_lane
    phi = (peak_rate - final_rate) / span_lane
    envs_out = {}
    for e, env in enumerate(env_ids):
        lanes = slices[env]
        floor, ceiling = scale_envs[env]["floor"], scale_envs[env]["ceiling"]
        span = ceiling - floor
        envs_out[env] = {
            "floor": fl(floor), "ceiling": fl(ceiling),
            "finalRate": rate_stats(final_rate[lanes], floor, span,
                                    BOOT_BASE_SEED + 101 + e * 10),
            "divergence": divergence_block(run, lanes, max_step),
            "curves": {"iqmZ": fl_list(row_iqm(z_rates[:, lanes]))},
        }
        if plast_fields:
            envs_out[env]["plasticity"] = {
                f: {"final": diag_iqm(plast_series[f][-1, lanes]),
                    "iqm": [diag_iqm(plast_series[f][r][lanes]) for r in plast_rows]}
                for f in plast_fields
            }
    z_final_runs = runs_matrix([z_final[slices[env]] for env in env_ids])
    z_life_runs = runs_matrix([z_life[slices[env]] for env in env_ids])
    phi_runs = runs_matrix([phi[slices[env]] for env in env_ids], high_is_worst=True)
    final_z_iqm, final_z_ci = interval_estimate(z_final_runs, agg_iqm, BOOT_BASE_SEED + 1)
    life_z_iqm, life_z_ci = interval_estimate(z_life_runs, agg_iqm, BOOT_BASE_SEED + 2)
    phi_iqm, phi_agg_ci = interval_estimate(phi_runs, agg_iqm, BOOT_BASE_SEED + 3)
    gap_mean, gap_ci = interval_estimate(z_final_runs, agg_gap, BOOT_BASE_SEED + 1)
    np.random.seed(BOOT_BASE_SEED + 4)
    profile, profile_ci = rly.create_performance_profile(
        {"s": z_final_runs}, TAU_GRID, reps=CURVE_REPS)
    z_curve = row_iqm(z_rates)
    cum_flops = [flops_per_step * int(s) for s in ends]
    aggregate = {
        "lifetimeZ": {"iqm": fl(life_z_iqm), "ci95": life_z_ci},
        "finalZ": {"iqm": fl(final_z_iqm), "ci95": final_z_ci},
        "optimalityGap": {"mean": fl(gap_mean), "ci95": gap_ci},
        "forgetting": {"phiIQM": fl(phi_iqm), "ci95": phi_agg_ci},
        "iprZ90": fl(np.percentile(z_final_runs, 95) - np.percentile(z_final_runs, 5))
                  if np.isfinite(z_final_runs).all() else None,
        "divergence": divergence_block(run, np.arange(n_lanes), max_step),
        "profile": {"tau": TAU_GRID, "fraction": fl_list(profile["s"]),
                    "ci95": [fl_list(profile_ci["s"][0]), fl_list(profile_ci["s"][1])]},
        "frontier": {"cumFlopsPerLane": cum_flops, "zIQM": fl_list(z_curve)},
    }
    compute = {"flopsPerStep": flops_per_step, "envFlopsPerStep": env_flops_per_step}
    if "gpuSeconds" in meta:
        compute["gpuSeconds"] = meta["gpuSeconds"]
    summary = {
        "scale": scale_table["scale"],
        "generatedAt": generated_at,
        "horizon": horizon,
        "maxStep": max_step,
        "complete": max_step >= horizon,
        "seedsPerEnv": per_env_counts.pop(),
        "bootstrap": {"reps": BOOT_REPS, "curveReps": CURVE_REPS,
                      "baseSeed": BOOT_BASE_SEED, "method": BOOT_METHOD},
        "curveSteps": [int(s) for s in ends],
        "plasticitySteps": [int(ends[r]) for r in plast_rows] if plast_fields else [],
        "compute": compute,
        "envs": envs_out,
        "aggregate": aggregate,
    }
    if meta["policy"] == "oak":
        vs = {p: vs_baseline(p, BOOT_BASE_SEED + 3000 + b * 1000, run, slices, scale_envs)
              for b, p in enumerate(("ppo", "random"))}
        if all(v is not None for v in vs.values()):
            summary["vsBaselines"] = vs
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
    for name, key, seed in (("learnedSeeds", "learned", 30), ("retainedSeeds", "retained", 31)):
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
    kr, cr = load(BENCH / kh), load(args.candidate)
    for label, r in (("keeper", kr), ("candidate", cr)):
        if r["meta"]["policy"] != "oak":
            return _fail(f"GATE ERROR: {label} policy is {r['meta']['policy']}")
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
        skip = {"learnedSeeds", "retainedSeeds", "positiveSeedFraction"}
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


def keeper_envs(km: dict) -> str:
    return ",".join(
        f"{e['envId']}:{e['seedEnd'] - e['seedStart'] + 1}"
        for e in km.get("envSchedule") or ())


def baseline_fingerprint(policy: str, envs: str, steps: int, batch: int, seeds: list[int]) -> str:
    h = hashlib.sha256()
    hash_env_sources(h)
    h.update(f"{policy}\n{envs}\n{steps}\n{batch}\n".encode())
    h.update(",".join(map(str, seeds)).encode())
    return h.hexdigest()


def baseline_pointer(policy: str) -> str | None:
    try:
        text = (BASELINES / policy).read_text().strip()
        return text or None
    except OSError:
        return None


def baseline_name(policy: str, km: dict) -> str:
    fp = baseline_fingerprint(policy, keeper_envs(km), km["steps"], km["batch"], km["seeds"])
    return f"baseline-{policy}-{fp[:12]}"


def cmd_baselines(args) -> int:
    km = keeper_meta()
    if not km:
        raise SystemExit("baselines need a crowned keeper for the schedule")
    envs = keeper_envs(km)
    BASELINES.mkdir(exist_ok=True)
    me = [sys.executable, str(Path(__file__).resolve())]
    for policy in ("random", "ppo"):
        name = baseline_name(policy, km)
        root = BASELINES / name
        if baseline_pointer(policy) == name and (root / ".done").exists():
            print(f"[baselines] {policy} current: {name}", flush=True)
            continue
        shutil.rmtree(root, ignore_errors=True)
        env = dict(os.environ,
                   OAK_BENCH_POLICY=policy,
                   OAK_BENCH_STEPS=str(km["steps"]),
                   OAK_BENCH_SEED_COUNT=str(km["batch"]),
                   OAK_BENCH_ENVS=envs,
                   OAK_FLEET_TIMING="1", OAK_METRICS_STORAGE=str(root))
        log = BASELINES / f"{name}.log"
        log.unlink(missing_ok=True)
        with open(log, "w") as lf:
            rc = subprocess.run(["caffeinate", "-dimsu", str(REPO / ".build/release/OaK"),
                                 "benchmark"], env=env, cwd=REPO,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"[baselines] {policy} benchmark exit={rc}; see {log}")
        subprocess.run(me + ["report", str(root)], cwd=REPO, check=True)
        (BASELINES / policy).write_text(name + "\n")
        (root / ".done").touch()
        print(f"[baselines] {policy} crowned: {name}", flush=True)
    return 0


def cmd_run(args) -> int:
    if os.environ.get("OAK_BENCH_POLICY", "oak") != "oak":
        print("CONFIG ERROR: science runs are oak; use ./fleet.py baselines")
        return 2
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
    km_now = keeper_meta()
    if km_now:
        for policy in ("random", "ppo"):
            if baseline_pointer(policy) != baseline_name(policy, km_now):
                print(f"[fleet] baseline {policy} stale; run ./fleet.py baselines", flush=True)
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
    gp = sub.add_parser("gate")
    gp.add_argument("candidate", type=Path)
    gp.add_argument("--boot", type=int, default=10_000)
    gp.add_argument("--env-slice", default=None)
    sub.add_parser("baselines")
    sub.add_parser("scale")
    args = ap.parse_args()
    return {"run": cmd_run, "job": cmd_job, "report": cmd_report,
            "gate": cmd_gate, "baselines": cmd_baselines, "scale": cmd_scale}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
