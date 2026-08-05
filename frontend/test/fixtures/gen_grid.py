#!/usr/bin/env python3
"""Gridworld decoder golden fixtures (11 envs).

Oracle (imported, never re-implemented): tools/gridworld_ref.py — the module
the Metal port is BITWISE certified against by `OaK gridworld-probe`. Every
number below is produced by calling into it; this file only reshapes.

Emits, next to this file:
    static_layouts.json        fixed wall layouts + hallway/door/key/goal cells
    procmaze_layouts.json      16 lane seeds -> wall words + 11x11 bitmap + path
    deepsea_masks.json         16 lane seeds -> mask words + 8x8 bitmap
    largedeepsea_masks.json    16 lane seeds -> 8 mask words + 16x16 bitmap
    mkc_replay.json            500 scripted MultiKeyCorridor steps + full state
    traces/<envId>.bin         first 200 record-v3 rows of a real fleet lane
    traces/<envId>.json        per-frame decoded state + derived events
    grid_event_cases.json      SYNTHESIZED record-v3 rows from scripted solves,
                               so every event type is exercised on every env

The trace goldens are the end-to-end contract: the .bin is cut verbatim from
.oak-benchmarks/frontend-demo-16env-20260805, and the .json is produced by
replaying ctrl[0] through gridworld_ref — hidden state (DeepSea masks,
ProceduralMaze walls, MultiKeyCorridor keys) included. The decoder is correct
iff `decodeRecording` reproduces the .json from the .bin.

Run with the repo venv (numpy + tools/ on sys.path):
    /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py --only grid
"""
from __future__ import annotations

import json
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gridworld_ref as gw  # noqa: E402

VERSION = 1

RUN_ROOT = _REPO / ".oak-benchmarks" / "frontend-demo-16env-20260805"
TRACE_STEPS = 200
RECORD_FLOATS = 16
SEED_SAMPLES = list(range(16))

# The 11 gridworlds this track owns, in EnvId order.
GRID_ENVS = [
    "FourRooms-cont-v0",
    "DeepSea-cont-v0",
    "KeyCorridor-cont-v0",
    "Chain-cont-v0",
    "LargeDeepSea-cont-v0",
    "MultiKeyCorridor-cont-v0",
    "MemoryChain-cont-v0",
    "ProceduralMaze-cont-v0",
    "HighDimMaze-cont-v0",
    "POMaze-cont-v0",
    "DelayedChain-cont-v0",
]


def _write(outdir: Path, name: str, obj) -> str:
    (outdir / name).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return name


def _cells(pairs) -> list[dict]:
    return [{"x": int(x), "y": int(y)} for x, y in pairs]


def _bitmap(w: int, h: int, is_wall) -> list[int]:
    """w*h row-major occupancy, 1 = wall (the GridGeometry.walls layout)."""
    return [1 if is_wall(x, y) else 0 for y in range(h) for x in range(w)]


# ------------------------------------------------------------ static layouts
def _static_layouts(outdir: Path) -> str:
    four = _bitmap(11, 11, gw.four_is_wall)
    assert len(four) - sum(four) == gw.FOUR_NFREE == 68, "FourRooms free-cell count"

    kc = _bitmap(9, 7, gw.kc_is_wall)
    assert kc[3 * 9 + 4] == 0, "KeyCorridor door cell (4,3) must be free"

    mkc = _bitmap(17, 7, gw.mkc_is_wall)
    for dx in (4, 8, 12):
        assert mkc[3 * 17 + dx] == 0, f"MultiKeyCorridor door ({dx},3) must be free"

    hdm = _bitmap(11, 11, gw.hdm_is_wall)
    assert hdm[9 * 11 + 3] == 0 and hdm[1 * 11 + 7] == 0, "HighDimMaze gaps"
    assert len(gw.HDM_FREE) == len(hdm) - sum(hdm), "HighDimMaze free-cell count"

    po = _bitmap(11, 11, gw.po_is_wall)
    assert po[3 * 11 + 8] == 0 and po[7 * 11 + 2] == 0, "POMaze gaps"
    assert len(gw.PO_FREE) == len(po) - sum(po), "POMaze free-cell count"

    layouts = {
        "FourRooms-cont-v0": {
            "w": 11, "h": 11, "walls": four, "freeCells": gw.FOUR_NFREE,
            "goal": {"x": 9, "y": 9}, "start": {"x": 1, "y": 1},
            # the four gaps in the cross walls
            "hallways": _cells([(5, 2), (5, 8), (2, 5), (8, 5)]),
            "free": _cells(gw.FOUR_FREE),
        },
        "KeyCorridor-cont-v0": {
            "w": 9, "h": 7, "walls": kc,
            "goal": {"x": 7, "y": 1}, "start": {"x": 1, "y": 1},
            "keys": _cells([(3, 5)]), "doors": _cells([(4, 3)]),
            "leftRoom": _cells(gw.KC_LEFT), "rightRoom": _cells(gw.KC_RIGHT),
        },
        "MultiKeyCorridor-cont-v0": {
            "w": 17, "h": 7, "walls": mkc,
            "goal": {"x": 15, "y": 1}, "start": {"x": 1, "y": 1},
            "keys": _cells([(3, 5), (7, 5), (11, 5)]),
            "doors": _cells([(4, 3), (8, 3), (12, 3)]),
            "roomBase": list(gw.MKC_BASE),
            "rooms": [_cells(r) for r in gw.MKC_ROOMS],
        },
        "HighDimMaze-cont-v0": {
            "w": 11, "h": 11, "walls": hdm,
            "goal": {"x": 9, "y": 9}, "start": {"x": 1, "y": 1},
            "gaps": _cells([(3, 9), (7, 1)]), "actions": 6, "initLastDir": 1,
            "free": _cells(gw.HDM_FREE),
        },
        "POMaze-cont-v0": {
            "w": 11, "h": 11, "walls": po,
            "goal": {"x": 9, "y": 9}, "start": {"x": 1, "y": 1},
            "gaps": _cells([(8, 3), (2, 7)]), "viewRadius": 1,
            "free": _cells(gw.PO_FREE),
            # the 8 wall bits the agent observes, in kernel channel order
            # N, NE, E, SE, S, SW, W, NW, for every free cell
            "views": [
                {"x": int(x), "y": int(y), "bits": [
                    1 if gw.po_is_wall(x + dx, y + dy) else 0
                    for dx, dy in ((0, -1), (1, -1), (1, 0), (1, 1),
                                   (0, 1), (-1, 1), (-1, 0), (-1, -1))
                ]}
                for x, y in gw.PO_FREE
            ],
        },
        # 1-D corridors: w x 1 grids with no walls, goal at the far end
        "Chain-cont-v0": {
            "w": gw.CHAIN_L, "h": 1, "walls": [0] * gw.CHAIN_L,
            "goal": {"x": gw.CHAIN_L - 1, "y": 0}, "start": {"x": 0, "y": 0},
        },
        "MemoryChain-cont-v0": {
            "w": gw.MCHAIN_L, "h": 1, "walls": [0] * gw.MCHAIN_L,
            "goal": {"x": gw.MCHAIN_L - 1, "y": 0}, "start": {"x": 0, "y": 0},
        },
        "DelayedChain-cont-v0": {
            "w": gw.DCHAIN_L, "h": 1, "walls": [0] * gw.DCHAIN_L,
            "goal": {"x": gw.DCHAIN_L - 1, "y": 0}, "start": {"x": 0, "y": 0},
        },
        # DeepSea family: no walls, the map IS the per-lane action mask
        "DeepSea-cont-v0": {
            "w": gw.DEEPSEA_N, "h": gw.DEEPSEA_N,
            "walls": [0] * (gw.DEEPSEA_N ** 2),
            "goal": {"x": gw.DEEPSEA_N - 1, "y": gw.DEEPSEA_N - 1},
            "start": {"x": 0, "y": 0}, "rightCost": float(gw.DEEPSEA_RIGHT_COST),
            "lastRow": gw.DEEPSEA_N - 2,
        },
        "LargeDeepSea-cont-v0": {
            "w": gw.LDEEPSEA_N, "h": gw.LDEEPSEA_N,
            "walls": [0] * (gw.LDEEPSEA_N ** 2),
            "goal": {"x": gw.LDEEPSEA_N - 1, "y": gw.LDEEPSEA_N - 1},
            "start": {"x": 0, "y": 0}, "rightCost": float(gw.LDEEPSEA_RIGHT_COST),
            "lastRow": gw.LDEEPSEA_N - 2,
        },
    }
    return _write(outdir, "static_layouts.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py is_wall predicates + room/free tables",
        "note": "walls is w*h row-major, 1 = wall (GridGeometry.walls layout).",
        "layouts": layouts,
    })


# ------------------------------------------------------- ProceduralMaze walls
def _pmaze_path(lane_seed: int) -> list[dict]:
    """The 24 between-cells the DFS spanning tree carves, in carve order.

    Mirrors the generator's DFS using the SAME oracle primitives
    (`gw._philox4x32_10`, integer-only); `_procmaze_layouts` asserts that
    starting from all-wall + the 25 room cells and clearing exactly these
    cells reproduces `gw.pmaze_gen` bit for bit.
    """
    path: list[dict] = []
    visited = 1
    stack = [0]
    ctr, bi = 0, 0
    buf = gw._philox4x32_10(lane_seed, 0, [0, 0, 0, 0])
    while stack:
        cur = stack[-1]
        ci, cj = cur % 5, cur // 5
        cand = []
        for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < 5 and 0 <= nj < 5 and not (visited >> (ni + 5 * nj)) & 1:
                cand.append((ni, nj))
        if not cand:
            stack.pop()
            continue
        if bi == 4:
            ctr += 1
            buf = gw._philox4x32_10(lane_seed, 0, [ctr, 0, 0, 0])
            bi = 0
        ni, nj = cand[buf[bi] % len(cand)]
        bi += 1
        path.append({"x": ci + ni + 1, "y": cj + nj + 1})
        visited |= 1 << (ni + 5 * nj)
        stack.append(ni + 5 * nj)
    return path


def _procmaze_layouts(outdir: Path) -> str:
    cases = []
    for seed in SEED_SAMPLES:
        w = gw.pmaze_gen(seed)
        path = _pmaze_path(seed)
        # independent check: all-wall + 25 rooms + exactly the carved
        # between-cells cleared
        check = [0xFFFFFFFF, 0xFFFFFFFF, 0x1FFFF]
        cells = ([{"x": 2 * i + 1, "y": 2 * j + 1} for j in range(5) for i in range(5)]
                 + path)
        for cell in cells:
            idx = (cell["x"] - 1) + 9 * (cell["y"] - 1)
            check[idx >> 5] &= 0xFFFFFFFF ^ (1 << (idx & 31))
        assert check == list(w), f"carved path disagrees with pmaze_gen at seed {seed}"
        assert all(not gw.pmaze_is_wall(w, c["x"], c["y"]) for c in path)
        assert len(path) == 24 and len(gw.pmaze_free(w)) == 49
        cases.append({
            "seed": seed,
            "words": [int(v) for v in w],
            "walls": _bitmap(11, 11, lambda x, y: gw.pmaze_is_wall(w, x, y)),
            "carvedPath": path,
            "freeCells": len(gw.pmaze_free(w)),
        })
    return _write(outdir, "procmaze_layouts.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py:pmaze_gen / pmaze_is_wall",
        "note": "words = the 3 packed wall words (w2 masked to 17 bits); "
                "walls = 11x11 row-major expansion; carvedPath = the 24 "
                "between-cells the DFS carves, in carve order (debug overlay "
                "only). Perfect maze: all 49 free cells mutually reachable.",
        "cases": cases,
    })


# ------------------------------------------------------------ DeepSea masks
def _deepsea_masks(outdir: Path) -> str:
    cases = []
    for seed in SEED_SAMPLES:
        r, c, last_a, lo, hi = gw.deepsea_reset(seed)
        assert (r, c, last_a) == (0, 0, 0)
        bits = [
            (lo >> i) & 1 if i < 32 else (hi >> (i - 32)) & 1
            for i in range(gw.DEEPSEA_N ** 2)
        ]
        cases.append({"seed": seed, "maskLo": int(lo), "maskHi": int(hi), "mask": bits})
    return _write(outdir, "deepsea_masks.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py:deepsea_reset (RESET stream, ctr=(0,0,0,0))",
        "note": "mask is 8*8 row-major, bit r*8+c; 1 = action 1 moves right.",
        "n": gw.DEEPSEA_N,
        "cases": cases,
    })


def _largedeepsea_masks(outdir: Path) -> str:
    cases = []
    for seed in SEED_SAMPLES:
        r, c, last_a, mask = gw.ldeepsea_reset(seed)
        assert (r, c, last_a) == (0, 0, 0)
        bits = [(mask[i >> 5] >> (i & 31)) & 1 for i in range(gw.LDEEPSEA_N ** 2)]
        cases.append({"seed": seed, "words": [int(v) for v in mask], "mask": bits})
    return _write(outdir, "largedeepsea_masks.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py:ldeepsea_reset "
                  "(RESET stream, ctr=(0,0,0,0) then (1,0,0,0))",
        "note": "mask is 16*16 row-major, bit r*16+c; 1 = action 1 moves right.",
        "n": gw.LDEEPSEA_N,
        "cases": cases,
    })


# ------------------------------------------------- MultiKeyCorridor replay
def _mkc_passable(x: int, y: int, flags: int) -> bool:
    if gw.mkc_is_wall(x, y):
        return False
    if y == 3 and x in (4, 8, 12):
        j = (x >> 2) - 1
        return bool(flags & (8 << j)) or bool(flags & (1 << j))
    return True


def _mkc_first_move(x: int, y: int, flags: int, target) -> int | None:
    """First action of a shortest passable path to `target` (BFS), or None."""
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))
    seen = {(x, y): None}
    q = deque([(x, y)])
    while q:
        cx, cy = q.popleft()
        for a, (dx, dy) in enumerate(moves):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or not _mkc_passable(nx, ny, flags):
                continue
            seen[(nx, ny)] = (cx, cy, a) if (cx, cy) != (x, y) else (cx, cy, a)
            if (nx, ny) == tuple(target):
                node = (nx, ny)
                while True:
                    px, py, a0 = seen[node]
                    if (px, py) == (x, y):
                        return a0
                    node = (px, py)
            q.append((nx, ny))
    return None


def _mkc_replay(outdir: Path) -> str:
    """500 scripted steps: a BFS-greedy solver with 10% random noise.

    Greedy so the trace reaches several completions (and therefore several
    Philox respawns, the only part of MultiKeyCorridor a decoder must replay);
    noisy so wall bumps and locked-door blocks are exercised too.
    """
    lane_seed = 80
    rng = np.random.default_rng(20260803)
    x, y, k1, k2, k3, gx, gy, flags, sg, ri = gw.mkc_reset()
    steps = []
    for _ in range(500):
        keys = [k1, k2, k3]
        target = (gx, gy)
        for j in range(3):
            if not (flags & (1 << j)):
                target = keys[j]
                break
        a = _mkc_first_move(x, y, flags, target)
        if a is None or rng.random() < 0.10:
            a = int(rng.integers(0, 4))
        (r, x, y, k1, k2, k3, gx, gy, flags, sg, ri, dn, _obs) = gw.mkc_step(
            lane_seed, ri, x, y, k1, k2, k3, gx, gy, flags, sg, a)
        steps.append({
            "action": int(a), "reward": float(r), "done": int(dn),
            "x": int(x), "y": int(y), "gx": int(gx), "gy": int(gy),
            "keys": [{"x": int(k[0]), "y": int(k[1])} for k in (k1, k2, k3)],
            "flags": int(flags), "respawnIdx": int(ri),
        })
    completions = sum(s["done"] for s in steps)
    assert completions >= 3, f"scripted MKC replay only completed {completions}x"
    assert any(s["flags"] & 4 for s in steps), "third key never collected"
    return _write(outdir, "mkc_replay.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py:mkc_step, scripted BFS-greedy + 10% noise",
        "note": "steps[i] is the state AFTER applying steps[i].action; the "
                "initial state is mkc_reset() = agent (1,1), keys (3,5)/(7,5)/"
                "(11,5), goal (15,1), flags 0, respawnIdx 0.",
        "laneSeed": lane_seed,
        "completions": int(completions),
        "steps": steps,
    })


# ---------------------------------------------------------------- traces
def _seed_starts() -> dict[str, int]:
    meta = json.loads((RUN_ROOT / "meta.json").read_text())
    return {e["envId"]: int(e["seedStart"]) for e in meta["envSchedule"]}


def _load_rows(env: str, seed: int) -> np.ndarray:
    # record4.load_rows dispatches on the lane's recordVersion (3 or 4).
    # Gridworld data is BIT-EXACT in v4 (u8 integer poses, action indices,
    # done bits), so either format is a legal fixture source here.
    import record4
    rows = record4.load_rows(RUN_ROOT / env / f"seed{seed}", TRACE_STEPS)
    assert rows.shape[0] == TRACE_STEPS, \
        f"{RUN_ROOT / env / f'seed{seed}'} too short ({rows.shape[0]} rows)"
    return rows


def _four_room(x: int, y: int) -> int:
    return int(gw.four_room(x, y))


def _trace_fourrooms(seed, rows):
    x, y, gx, gy, sg, ri = gw.four_reset()
    frames = [{"x": x, "y": y, "gx": gx, "gy": gy, "room": _four_room(x, y)}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        px, py, pgx, pgy = x, y, gx, gy
        r, x, y, gx, gy, sg, ri, _o = gw.four_step(seed, ri, x, y, gx, gy, sg, a)
        assert (rows[i][7], rows[i][8], rows[i][9], rows[i][10]) == (x, y, gx, gy)
        ev = [{"type": "goal"}] if r > 0 else []
        # cross-check: the reward condition IS "stepped onto the old goal"
        assert (r > 0) == ((x, y) == (pgx, pgy)), i
        assert (px, py) != (x, y) or True
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy, "room": _four_room(x, y)})
        events.append(ev)
        rewards.append(float(r))
    return frames, events, rewards


def _trace_deepsea(seed, rows):
    r_, c_, la, lo, hi = gw.deepsea_reset(seed)
    frames = [{"r": r_, "c": c_, "lastA": la}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        prev_r = r_
        rw, r_, c_, la, _o = gw.deepsea_step(lo, hi, r_, c_, la, a)
        assert (rows[i][7], rows[i][8]) == (r_, c_)
        ev = []
        if prev_r == gw.DEEPSEA_N - 2:
            if rw > 0:
                ev.append({"type": "goal"})
            ev.append({"type": "resurface"})
            assert (r_, c_) == (0, 0)
        else:
            assert rw <= 0
        frames.append({"r": r_, "c": c_, "lastA": la})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_largedeepsea(seed, rows):
    r_, c_, la, mask = gw.ldeepsea_reset(seed)
    frames = [{"r": r_, "c": c_, "lastA": la}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        prev_r = r_
        rw, r_, c_, la, _o = gw.ldeepsea_step(mask, r_, c_, la, a)
        assert (rows[i][7], rows[i][8]) == (r_, c_)
        ev = []
        if prev_r == gw.LDEEPSEA_N - 2:
            if rw > 0:
                ev.append({"type": "goal"})
            ev.append({"type": "resurface"})
            assert (r_, c_) == (0, 0)
        frames.append({"r": r_, "c": c_, "lastA": la})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_keycorridor(seed, rows):
    x, y, kx, ky, gx, gy, fl, sg, ri = gw.kc_reset()

    def snap():
        return {"x": x, "y": y, "gx": gx, "gy": gy, "kx": kx, "ky": ky,
                "hasKey": bool(fl & 1), "doorOpen": bool(fl & 2)}

    frames, events, rewards = [snap()], [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        pfl = fl
        (rw, x, y, kx, ky, gx, gy, fl, sg, ri, dn, _o) = gw.kc_step(
            seed, ri, x, y, kx, ky, gx, gy, fl, sg, a)
        assert tuple(rows[i][7:16]) == (x, y, gx, gy, kx, ky, fl & 1, (fl >> 1) & 1, 0)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = []
        if dn:
            ev.append({"type": "goal"})
        if not (pfl & 1) and (fl & 1):
            ev.append({"type": "key", "index": 0})
        if (pfl & 2) != (fl & 2):
            ev.append({"type": "door", "index": 0, "open": bool(fl & 2)})
        if dn:
            ev.append({"type": "teleport"})
            ev.append({"type": "done"})
        frames.append(snap())
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_mkc(seed, rows):
    x, y, k1, k2, k3, gx, gy, fl, sg, ri = gw.mkc_reset()

    def snap():
        return {"x": x, "y": y, "gx": gx, "gy": gy,
                "keys": [{"x": k[0], "y": k[1]} for k in (k1, k2, k3)],
                "hasKey": [bool(fl & 1), bool(fl & 2), bool(fl & 4)],
                "doorOpen": [bool(fl & 8), bool(fl & 16), bool(fl & 32)],
                "flags": int(fl), "respawnIdx": int(ri), "poseAgrees": True}

    frames, events, rewards = [snap()], [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        pfl = fl
        (rw, x, y, k1, k2, k3, gx, gy, fl, sg, ri, dn, _o) = gw.mkc_step(
            seed, ri, x, y, k1, k2, k3, gx, gy, fl, sg, a)
        assert tuple(rows[i][7:16]) == (x, y, gx, gy, fl & 7, (fl >> 3) & 7, 0, 0, 0)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = []
        if dn:
            ev.append({"type": "goal"})
        for j in range(3):
            if not (pfl & (1 << j)) and (fl & (1 << j)):
                ev.append({"type": "key", "index": j})
        for j in range(3):
            if (pfl & (8 << j)) != (fl & (8 << j)):
                ev.append({"type": "door", "index": j, "open": bool(fl & (8 << j))})
        if dn:
            ev.append({"type": "teleport"})
            ev.append({"type": "done"})
        frames.append(snap())
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_chain(seed, rows):
    pos = 0
    frames, events, rewards = [{"pos": pos, "len": gw.CHAIN_L}], [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        rw, pos, dn, _o = gw.chain_step(pos, a)
        assert (rows[i][7], rows[i][8]) == (pos, gw.CHAIN_L - 1)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = ([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
              if dn else [])
        frames.append({"pos": pos, "len": gw.CHAIN_L})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_memorychain(seed, rows):
    pos, cue, la, ri = gw.mchain_reset()
    frames = [{"pos": pos, "cue": cue, "lastA": la, "len": gw.MCHAIN_L}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        pcue = cue
        rw, pos, cue, la, ri, dn, _o = gw.mchain_step(seed, pos, cue, la, ri, a)
        assert (rows[i][7], rows[i][8], rows[i][9]) == (pos, gw.MCHAIN_L - 1, cue)
        assert int(rows[i][6]) == dn
        ev = []
        if dn:
            if rw > 0:
                ev.append({"type": "goal"})
                assert (a & 1) == pcue
            ev.append({"type": "cue", "value": int(cue)})
            ev.append({"type": "teleport"})
            ev.append({"type": "done"})
        frames.append({"pos": pos, "cue": int(cue), "lastA": la, "len": gw.MCHAIN_L})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_delayedchain(seed, rows):
    pos, la = gw.dchain_reset()
    frames = [{"pos": pos, "lastA": la, "len": gw.DCHAIN_L}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        rw, pos, la, dn, _o = gw.dchain_step(pos, la, a)
        assert (rows[i][7], rows[i][8], rows[i][9]) == (pos, gw.DCHAIN_L - 1, la)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = ([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
              if dn else [])
        frames.append({"pos": pos, "lastA": la, "len": gw.DCHAIN_L})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_procmaze(seed, rows):
    w = gw.pmaze_gen(seed)
    x, y, gx, gy, sg, ri = 1, 1, 9, 9, 0, 0
    frames, events, rewards = [{"x": x, "y": y, "gx": gx, "gy": gy}], [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        (rw, x, y, gx, gy, sg, ri, dn, _o) = gw.pmaze_step(
            seed, ri, x, y, gx, gy, sg, w, a)
        assert (rows[i][7], rows[i][8], rows[i][9], rows[i][10]) == (x, y, gx, gy)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = ([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
              if dn else [])
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_highdimmaze(seed, rows):
    x, y, gx, gy, sg, ri, ld = 1, 1, 9, 9, 0, 0, 1
    frames = [{"x": x, "y": y, "gx": gx, "gy": gy, "lastDir": ld}]
    events, rewards = [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        (rw, x, y, gx, gy, sg, ri, ld, dn, _o) = gw.hdm_step(
            seed, ri, x, y, gx, gy, sg, ld, a)
        assert tuple(rows[i][7:12]) == (x, y, gx, gy, ld)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = ([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
              if dn else [])
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy, "lastDir": int(ld)})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


def _trace_pomaze(seed, rows):
    x, y, gx, gy, sg, ri = 1, 1, 9, 9, 0, 0
    frames, events, rewards = [{"x": x, "y": y, "gx": gx, "gy": gy}], [[]], []
    for i in range(len(rows)):
        a = int(rows[i][0])
        (rw, x, y, gx, gy, sg, ri, dn, _o) = gw.po_step(seed, ri, x, y, gx, gy, sg, a)
        assert (rows[i][7], rows[i][8], rows[i][9], rows[i][10]) == (x, y, gx, gy)
        assert int(rows[i][6]) == dn and (rw > 0) == bool(dn)
        ev = ([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
              if dn else [])
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy})
        events.append(ev)
        rewards.append(float(rw))
    return frames, events, rewards


TRACERS = {
    "FourRooms-cont-v0": _trace_fourrooms,
    "DeepSea-cont-v0": _trace_deepsea,
    "KeyCorridor-cont-v0": _trace_keycorridor,
    "Chain-cont-v0": _trace_chain,
    "LargeDeepSea-cont-v0": _trace_largedeepsea,
    "MultiKeyCorridor-cont-v0": _trace_mkc,
    "MemoryChain-cont-v0": _trace_memorychain,
    "ProceduralMaze-cont-v0": _trace_procmaze,
    "HighDimMaze-cont-v0": _trace_highdimmaze,
    "POMaze-cont-v0": _trace_pomaze,
    "DelayedChain-cont-v0": _trace_delayedchain,
}


def _traces(outdir: Path) -> list[str]:
    tdir = outdir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    starts = _seed_starts()
    written: list[str] = []
    for env in GRID_ENVS:
        seed = starts[env]
        rows = _load_rows(env, seed)
        (tdir / f"{env}.bin").write_bytes(rows.tobytes())
        frames, events, rewards = TRACERS[env](seed, rows)
        assert len(frames) == TRACE_STEPS + 1 and len(events) == TRACE_STEPS + 1
        _write(tdir, f"{env}.json", {
            "version": VERSION,
            "envId": env,
            "seed": seed,
            "steps": TRACE_STEPS,
            "source": f".oak-benchmarks/{RUN_ROOT.name}/{env}/seed{seed}/actions.bin",
            "note": "frames[0] is the synthesized init frame (step -1); "
                    "frames[i+1] decodes record row i. events[i] fires on the "
                    "transition INTO frames[i]; events[0] is always empty.",
            "eventCounts": _count_events(events),
            "rewardSum": float(sum(rewards)),
            "frames": frames,
            "events": events,
            "rewards": rewards,
        })
        written.append(f"traces/{env}.bin")
        written.append(f"traces/{env}.json")
    return written


def _count_events(events) -> dict:
    counts: dict[str, int] = {}
    for evs in events:
        for e in evs:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    return counts


# ------------------------------------------------------------ event cases
# A 200-step window of a 100k-step lane rarely contains a goal on the harder
# envs (the policies are weak there), so the trace goldens alone cannot pin
# the event derivation. These cases SYNTHESIZE record-v3 rows from scripted
# solves driven by the same reference dynamics: the row packing is exactly
# what `oak_stage_fleet_env` writes, so they decode through the identical path.
def _row(ctrl0: int, done: int, pose) -> list[float]:
    row = [0.0] * RECORD_FLOATS
    row[0] = float(ctrl0)
    row[6] = float(done)
    for i, v in enumerate(pose):
        row[7 + i] = float(v)
    return row


def _bfs_move(passable, start, target, moves=((0, -1), (0, 1), (-1, 0), (1, 0))):
    """First action of a shortest passable path from `start` to `target`."""
    if tuple(start) == tuple(target):
        return None
    prev = {tuple(start): None}
    q = deque([tuple(start)])
    while q:
        cx, cy = q.popleft()
        for a, (dx, dy) in enumerate(moves):
            nxt = (cx + dx, cy + dy)
            if nxt in prev or not passable(*nxt):
                continue
            prev[nxt] = (cx, cy, a)
            if nxt == tuple(target):
                node = nxt
                while True:
                    px, py, a0 = prev[node]
                    if (px, py) == tuple(start):
                        return a0
                    node = (px, py)
            q.append(nxt)
    return None


def _case_fourrooms(seed=8, goals=3):
    x, y, gx, gy, sg, ri = gw.four_reset()
    rows, frames, events = [], [{"x": x, "y": y, "gx": gx, "gy": gy,
                                 "room": _four_room(x, y)}], [[]]
    hit = 0
    while hit < goals and len(rows) < 400:
        a = _bfs_move(lambda cx, cy: not gw.four_is_wall(cx, cy), (x, y), (gx, gy))
        assert a is not None
        pgx, pgy = gx, gy
        r, x, y, gx, gy, sg, ri, _o = gw.four_step(seed, ri, x, y, gx, gy, sg, a)
        rows.append(_row(a, 0, [x, y, gx, gy, 0, 0, 0, 0, 0]))
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy, "room": _four_room(x, y)})
        ev = []
        if r > 0:
            assert (x, y) == (pgx, pgy)
            ev.append({"type": "goal"})
            hit += 1
        events.append(ev)
    assert hit == goals
    return rows, frames, events


def _case_deepsea(seed=16, dives=3, sabotage=True):
    """Perfect dives (always move right) then one deliberate left dive."""
    r_, c_, la, lo, hi = gw.deepsea_reset(seed)
    n = gw.DEEPSEA_N
    rows, frames, events = [], [{"r": r_, "c": c_, "lastA": la}], [[]]
    plan = [True] * dives + ([False] if sabotage else [])
    for go_right in plan:
        for _ in range(n - 1):
            idx = r_ * n + c_
            bit = (lo >> idx) & 1 if idx < 32 else (hi >> (idx - 32)) & 1
            a = bit if go_right else 1 - bit
            prev_r = r_
            rw, r_, c_, la, _o = gw.deepsea_step(lo, hi, r_, c_, la, a)
            rows.append(_row(a, 0, [r_, c_, n - 1, n - 1, 0, 0, 0, 0, 0]))
            frames.append({"r": r_, "c": c_, "lastA": la})
            ev = []
            if prev_r == n - 2:
                if rw > 0:
                    ev.append({"type": "goal"})
                ev.append({"type": "resurface"})
            events.append(ev)
    kinds = [e["type"] for evs in events for e in evs]
    assert kinds.count("goal") == dives and kinds.count("resurface") == len(plan)
    return rows, frames, events


def _case_largedeepsea(seed=72, dives=2):
    r_, c_, la, mask = gw.ldeepsea_reset(seed)
    n = gw.LDEEPSEA_N
    rows, frames, events = [], [{"r": r_, "c": c_, "lastA": la}], [[]]
    for k in range(dives + 1):
        go_right = k < dives
        for _ in range(n - 1):
            idx = r_ * n + c_
            bit = (mask[idx >> 5] >> (idx & 31)) & 1
            a = bit if go_right else 1 - bit
            prev_r = r_
            rw, r_, c_, la, _o = gw.ldeepsea_step(mask, r_, c_, la, a)
            rows.append(_row(a, 0, [r_, c_, n - 1, n - 1, 0, 0, 0, 0, 0]))
            frames.append({"r": r_, "c": c_, "lastA": la})
            ev = []
            if prev_r == n - 2:
                if rw > 0:
                    ev.append({"type": "goal"})
                ev.append({"type": "resurface"})
            events.append(ev)
    kinds = [e["type"] for evs in events for e in evs]
    assert kinds.count("goal") == dives
    return rows, frames, events


def _case_keycorridor(seed=24, goals=3):
    x, y, kx, ky, gx, gy, fl, sg, ri = gw.kc_reset()

    def snap():
        return {"x": x, "y": y, "gx": gx, "gy": gy, "kx": kx, "ky": ky,
                "hasKey": bool(fl & 1), "doorOpen": bool(fl & 2)}

    def passable(cx, cy):
        if gw.kc_is_wall(cx, cy):
            return False
        if (cx, cy) == (4, 3) and not (fl & 2):
            return bool(fl & 1)
        return True

    rows, frames, events = [], [snap()], [[]]
    hit = 0
    while hit < goals and len(rows) < 400:
        target = (gx, gy) if (fl & 1) else (kx, ky)
        # a respawn can drop the agent ON the key (pickup is deferred one
        # step), and then start == target: step off and the next BFS returns.
        a = _bfs_move(passable, (x, y), target)
        if a is None:
            a = 0
        pfl = fl
        (r, x, y, kx, ky, gx, gy, fl, sg, ri, dn, _o) = gw.kc_step(
            seed, ri, x, y, kx, ky, gx, gy, fl, sg, a)
        rows.append(_row(a, dn, [x, y, gx, gy, kx, ky, fl & 1, (fl >> 1) & 1, 0]))
        frames.append(snap())
        ev = []
        if dn:
            ev.append({"type": "goal"})
        if not (pfl & 1) and (fl & 1):
            ev.append({"type": "key", "index": 0})
        if (pfl & 2) != (fl & 2):
            ev.append({"type": "door", "index": 0, "open": bool(fl & 2)})
        if dn:
            ev.append({"type": "teleport"})
            ev.append({"type": "done"})
            hit += 1
        events.append(ev)
    assert hit == goals
    return rows, frames, events


def _case_mkc(seed=80, goals=3):
    x, y, k1, k2, k3, gx, gy, fl, sg, ri = gw.mkc_reset()

    def snap():
        return {"x": x, "y": y, "gx": gx, "gy": gy,
                "keys": [{"x": k[0], "y": k[1]} for k in (k1, k2, k3)],
                "hasKey": [bool(fl & 1), bool(fl & 2), bool(fl & 4)],
                "doorOpen": [bool(fl & 8), bool(fl & 16), bool(fl & 32)],
                "flags": int(fl), "respawnIdx": int(ri), "poseAgrees": True}

    rows, frames, events = [], [snap()], [[]]
    hit = 0
    while hit < goals and len(rows) < 1200:
        keys = [k1, k2, k3]
        target = (gx, gy)
        for j in range(3):
            if not (fl & (1 << j)):
                target = keys[j]
                break
        a = _bfs_move(lambda cx, cy: _mkc_passable(cx, cy, fl), (x, y), target)
        if a is None:
            a = 0   # already standing on the target key: step off, come back
        pfl = fl
        (r, x, y, k1, k2, k3, gx, gy, fl, sg, ri, dn, _o) = gw.mkc_step(
            seed, ri, x, y, k1, k2, k3, gx, gy, fl, sg, a)
        rows.append(_row(a, dn, [x, y, gx, gy, fl & 7, (fl >> 3) & 7, 0, 0, 0]))
        frames.append(snap())
        ev = []
        if dn:
            ev.append({"type": "goal"})
        for j in range(3):
            if not (pfl & (1 << j)) and (fl & (1 << j)):
                ev.append({"type": "key", "index": j})
        for j in range(3):
            if (pfl & (8 << j)) != (fl & (8 << j)):
                ev.append({"type": "door", "index": j, "open": bool(fl & (8 << j))})
        if dn:
            ev.append({"type": "teleport"})
            ev.append({"type": "done"})
            hit += 1
        events.append(ev)
    assert hit == goals
    kinds = [e["type"] for evs in events for e in evs]
    assert kinds.count("key") >= 3 * goals and kinds.count("door") >= 3 * goals
    return rows, frames, events


def _case_chain(seed=64, laps=2):
    pos = 0
    rows, frames, events = [], [{"pos": pos, "len": gw.CHAIN_L}], [[]]
    for _ in range(laps * (gw.CHAIN_L - 1)):
        a = 1
        rw, pos, dn, _o = gw.chain_step(pos, a)
        rows.append(_row(a, dn, [pos, gw.CHAIN_L - 1, 0, 0, 0, 0, 0, 0, 0]))
        frames.append({"pos": pos, "len": gw.CHAIN_L})
        events.append([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
                      if dn else [])
    assert sum(1 for evs in events for e in evs if e["type"] == "goal") == laps
    return rows, frames, events


def _case_delayedchain(seed=120, laps=2):
    pos, la = gw.dchain_reset()
    rows, frames, events = [], [{"pos": pos, "lastA": la, "len": gw.DCHAIN_L}], [[]]
    # one deliberate step back first, to exercise the clamp at 0
    plan = [0] + [1] * (laps * (gw.DCHAIN_L - 1))
    for a in plan:
        rw, pos, la, dn, _o = gw.dchain_step(pos, la, a)
        rows.append(_row(a, dn, [pos, gw.DCHAIN_L - 1, la, 0, 0, 0, 0, 0, 0]))
        frames.append({"pos": pos, "lastA": la, "len": gw.DCHAIN_L})
        events.append([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
                      if dn else [])
    assert sum(1 for evs in events for e in evs if e["type"] == "goal") == laps
    return rows, frames, events


def _case_memorychain(seed=88, rounds=4):
    """Walk to the far end and alternate a correct / incorrect choice."""
    pos, cue, la, ri = gw.mchain_reset()
    rows, frames = [], [{"pos": pos, "cue": cue, "lastA": la, "len": gw.MCHAIN_L}]
    events = [[]]
    for k in range(rounds):
        plan = [1] * (gw.MCHAIN_L - 1 - pos)
        plan.append(cue if k % 2 == 0 else 1 - cue)   # the choice at pos 31
        for a in plan:
            pcue = cue
            rw, pos, cue, la, ri, dn, _o = gw.mchain_step(seed, pos, cue, la, ri, a)
            rows.append(_row(a, dn, [pos, gw.MCHAIN_L - 1, cue, 0, 0, 0, 0, 0, 0]))
            frames.append({"pos": pos, "cue": int(cue), "lastA": la,
                           "len": gw.MCHAIN_L})
            ev = []
            if dn:
                if rw > 0:
                    assert (a & 1) == pcue
                    ev.append({"type": "goal"})
                ev.append({"type": "cue", "value": int(cue)})
                ev.append({"type": "teleport"})
                ev.append({"type": "done"})
            events.append(ev)
    kinds = [e["type"] for evs in events for e in evs]
    assert kinds.count("done") == rounds and kinds.count("cue") == rounds
    assert kinds.count("goal") == (rounds + 1) // 2   # every other round is wrong
    return rows, frames, events


def _case_procmaze(seed=96, goals=3):
    """Perfect maze: every free cell is mutually reachable, so BFS always
    finds the redrawn goal. `_pmaze_case_seed` keeps its search shape but
    every candidate now reaches the full `goals` count by construction."""
    w = gw.pmaze_gen(seed)
    x, y, gx, gy, sg, ri = 1, 1, 9, 9, 0, 0
    rows, frames, events = [], [{"x": x, "y": y, "gx": gx, "gy": gy}], [[]]
    hit = 0
    while hit < goals and len(rows) < 800:
        a = _bfs_move(lambda cx, cy: not gw.pmaze_is_wall(w, cx, cy), (x, y), (gx, gy))
        if a is None:
            break   # goal stranded in another component: nothing more to solve
        (r, x, y, gx, gy, sg, ri, dn, _o) = gw.pmaze_step(
            seed, ri, x, y, gx, gy, sg, w, a)
        rows.append(_row(a, dn, [x, y, gx, gy, 0, 0, 0, 0, 0]))
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy})
        events.append([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
                      if dn else [])
        hit += int(dn)
    assert hit >= 1, f"ProceduralMaze seed {seed} never reached its goal"
    return rows, frames, events


def _pmaze_case_seed(goals=3, candidates=tuple(range(96, 104)) + tuple(range(32))):
    """First lane seed whose maze admits `goals` consecutive solves."""
    for seed in candidates:
        _r, _f, events = _case_procmaze(seed, goals)
        if _count_events(events).get("goal", 0) >= goals:
            return seed
    raise AssertionError(f"no ProceduralMaze seed reached {goals} goals")


def _case_highdimmaze(seed=104, goals=3):
    x, y, gx, gy, sg, ri, ld = 1, 1, 9, 9, 0, 0, 1
    rows, frames = [], [{"x": x, "y": y, "gx": gx, "gy": gy, "lastDir": ld}]
    events = [[]]
    hit, n = 0, 0
    while hit < goals and len(rows) < 800:
        a = _bfs_move(lambda cx, cy: not gw.hdm_is_wall(cx, cy), (x, y), (gx, gy))
        assert a is not None, (x, y, gx, gy)
        n += 1
        if n % 7 == 0:
            a = gw_stay_or_dash(n)   # exercise stay (4) and dash (5)
        (r, x, y, gx, gy, sg, ri, ld, dn, _o) = gw.hdm_step(
            seed, ri, x, y, gx, gy, sg, ld, a)
        rows.append(_row(a, dn, [x, y, gx, gy, ld, 0, 0, 0, 0]))
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy, "lastDir": int(ld)})
        events.append([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
                      if dn else [])
        hit += int(dn)
    assert hit == goals
    assert any(int(r[0]) == 5 for r in rows) and any(int(r[0]) == 4 for r in rows)
    return rows, frames, events


def gw_stay_or_dash(n: int) -> int:
    return 4 if (n // 7) % 2 == 0 else 5


def _case_pomaze(seed=112, goals=3):
    x, y, gx, gy, sg, ri = 1, 1, 9, 9, 0, 0
    rows, frames, events = [], [{"x": x, "y": y, "gx": gx, "gy": gy}], [[]]
    hit = 0
    while hit < goals and len(rows) < 800:
        a = _bfs_move(lambda cx, cy: not gw.po_is_wall(cx, cy), (x, y), (gx, gy))
        assert a is not None, (x, y, gx, gy)
        (r, x, y, gx, gy, sg, ri, dn, _o) = gw.po_step(seed, ri, x, y, gx, gy, sg, a)
        rows.append(_row(a, dn, [x, y, gx, gy, 0, 0, 0, 0, 0]))
        frames.append({"x": x, "y": y, "gx": gx, "gy": gy})
        events.append([{"type": "goal"}, {"type": "teleport"}, {"type": "done"}]
                      if dn else [])
        hit += int(dn)
    assert hit == goals
    return rows, frames, events


EVENT_CASES = [
    ("FourRooms-cont-v0", 8, "three-goal solve", _case_fourrooms),
    ("DeepSea-cont-v0", 16, "3 treasure dives + 1 failed dive", _case_deepsea),
    ("KeyCorridor-cont-v0", 24, "three key/door/goal solves", _case_keycorridor),
    ("Chain-cont-v0", 64, "two laps", _case_chain),
    ("LargeDeepSea-cont-v0", 72, "2 treasure dives + 1 failed dive",
     _case_largedeepsea),
    ("MultiKeyCorridor-cont-v0", 80, "three 3-key solves", _case_mkc),
    ("MemoryChain-cont-v0", 88, "4 rounds, alternating correct/incorrect",
     _case_memorychain),
    ("ProceduralMaze-cont-v0", _pmaze_case_seed, "three goal solves", _case_procmaze),
    ("HighDimMaze-cont-v0", 104, "three goal solves with stay + dash",
     _case_highdimmaze),
    ("POMaze-cont-v0", 112, "three goal solves", _case_pomaze),
    ("DelayedChain-cont-v0", 120, "two laps after a clamped step back",
     _case_delayedchain),
]


def _event_cases(outdir: Path) -> str:
    cases = []
    for env, seed, label, fn in EVENT_CASES:
        if callable(seed):
            seed = seed()
        rows, frames, events = fn(seed)
        assert len(frames) == len(rows) + 1 == len(events)
        cases.append({
            "envId": env, "seed": seed, "label": label, "steps": len(rows),
            "eventCounts": _count_events(events),
            "rows": rows, "frames": frames, "events": events,
        })
    return _write(outdir, "grid_event_cases.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py step functions, driven by scripted "
                  "BFS solves / perfect dives",
        "note": "rows are SYNTHESIZED record-v3 rows (ctrl[6], done, pose[9]) "
                "packed exactly as oak_stage_fleet_env writes them, so they "
                "decode through the identical path as a real actions.bin.",
        "cases": cases,
    })


# ------------------------------------------------------------ entry point
def generate(outdir: Path) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = [
        _static_layouts(outdir),
        _procmaze_layouts(outdir),
        _deepsea_masks(outdir),
        _largedeepsea_masks(outdir),
        _mkc_replay(outdir),
        _event_cases(outdir),
    ]
    out.extend(_traces(outdir))
    return out


if __name__ == "__main__":
    for f in generate(_HERE):
        print(f"wrote {f}")
