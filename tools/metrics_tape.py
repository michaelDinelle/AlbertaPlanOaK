"""OAKT v1 — the continuous metrics tape reader (METRICS LAW 2026-08-05).

One file per run root per policy (`metrics.<policy>.oakt`), written by
Sources/Main.swift `MetricsTapeWriter`: one row per lane per command-buffer
window (~1024 steps), segments raw-DEFLATE-compressed at heartbeat cadence.
Self-describing — the header carries the field table, so this reader never
hard-codes the schema. Torn tails drop segment-by-segment via crc, like
record v4 chunks.

    tape = read_tape(root / "metrics.oak.oakt")
    tape["fields"]                  # list of field names
    tape["stepStart"], tape["stepCount"]   # (W,) window boundaries
    tape["data"]                    # (W, F, B) float32
    series(tape, "rewardSum")       # (W, B)
    at_step(tape, "rewardSum", 100000)  # (B,) value at the window ENDING there
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

FILE_MAGIC = 0x544B414F
SEG_MAGIC = 0x54474553


def read_tape(path: Path | str) -> dict:
    data = Path(path).read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path}: too short for an OAKT header")
    magic, version, lane_count, json_len = struct.unpack_from("<4I", data, 0)
    if magic != FILE_MAGIC or version != 1:
        raise ValueError(f"{path}: not an OAKT v1 tape (magic {magic:#x} v{version})")
    fields = json.loads(data[16:16 + json_len])["fields"]
    n_f = len(fields)
    row_bytes = 8 + n_f * lane_count * 4
    starts, counts, mats = [], [], []
    off = 16 + json_len
    while off + 20 <= len(data):
        smagic, n_win, raw_bytes, comp_bytes, crc = struct.unpack_from("<5I", data, off)
        if smagic != SEG_MAGIC or off + 20 + comp_bytes > len(data):
            break
        comp = data[off + 20: off + 20 + comp_bytes]
        if zlib.crc32(comp) != crc:
            break
        payload = zlib.decompress(comp, wbits=-15)
        if len(payload) != raw_bytes or raw_bytes != n_win * row_bytes:
            break
        p = 0
        for _ in range(n_win):
            s, c = struct.unpack_from("<2I", payload, p)
            starts.append(s)
            counts.append(c)
            mats.append(np.frombuffer(payload, dtype=np.float32,
                                      count=n_f * lane_count, offset=p + 8)
                        .reshape(n_f, lane_count))
            p += row_bytes
        off += 20 + comp_bytes
    return {
        "fields": fields,
        "laneCount": lane_count,
        "stepStart": np.array(starts, dtype=np.int64),
        "stepCount": np.array(counts, dtype=np.int64),
        "data": np.stack(mats) if mats else np.zeros((0, n_f, lane_count), np.float32),
    }


def series(tape: dict, field: str) -> np.ndarray:
    """(W, B) series of one field across all windows."""
    return tape["data"][:, tape["fields"].index(field), :]


def steps_end(tape: dict) -> np.ndarray:
    """(W,) the step each window ENDS on (stepStart + stepCount)."""
    return tape["stepStart"] + tape["stepCount"]


def at_step(tape: dict, field: str, step: int) -> np.ndarray:
    """(B,) the field at the window ending exactly at `step` (heartbeats land
    on window boundaries by construction). Raises if no such window."""
    ends = steps_end(tape)
    idx = np.nonzero(ends == step)[0]
    if idx.size == 0:
        raise KeyError(f"no tape window ends at step {step} (ends: {ends[:5]}...{ends[-3:]})")
    return series(tape, field)[idx[0]]


def find_tape(run_root: Path | str, policy: str = "oak") -> Path | None:
    p = Path(run_root) / f"metrics.{policy}.oakt"
    return p if p.exists() else None


def _lane_env_map(meta: dict) -> dict[int, str]:
    sched = meta.get("envSchedule") or []
    if not sched:
        return {}
    out = {}
    for e in sched:
        for s in range(e["seedStart"], e["seedEnd"] + 1):
            out[s] = e["envId"]
    return out


def synth_hb_rows(root: Path | str, policy: str = "oak") -> list[dict]:
    """Reconstruct fleet.jsonl-shaped `hb` rows from the continuous tape.

    Tape-era roots (METRICS LAW 2026-08-05) write only `final` rows to
    fleet.jsonl; every heartbeat consumer (fleet.py report/lanes, table.py)
    gets its rows from here instead — synthesized at the same heartbeat
    steps aggregate.jsonl records, decision-identical with pre-tape roots.
    Returns [] when the root has no tape (old roots: fleet.jsonl has real
    hb rows) or no aggregate heartbeats yet.
    """
    root = Path(root)
    tp = find_tape(root, policy)
    if tp is None:
        return []
    meta = json.loads((root / "meta.json").read_text())
    steps_total = meta.get("steps")
    agg_path = root / "aggregate.jsonl"
    if not agg_path.exists():
        return []
    hb_rows = [json.loads(l) for l in open(agg_path) if l.strip()]
    hb = {(r["step"], r.get("policy", "oak")): r for r in hb_rows}
    hb_steps = sorted(s for (s, pol) in hb if pol == policy and s != steps_total)
    if not hb_steps:
        return []
    tape = read_tape(tp)
    ends = steps_end(tape)
    fields = tape["fields"]
    fi = {f: i for i, f in enumerate(fields)}
    envs = _lane_env_map(meta)
    env_default = meta.get("envId")
    seeds = meta.get("seeds") or list(range(tape["laneCount"]))
    deaths = []
    dp = root / "deaths.jsonl"
    if dp.exists():
        deaths = [json.loads(l) for l in open(dp) if l.strip()]
    death_by_lane: dict[int, dict] = {}
    for d in deaths:
        if d.get("policy", "oak") == policy:
            death_by_lane.setdefault(d["lane"], d)
    ablation = meta.get("ablationLabel") or meta.get("ablation") or "none"
    B = tape["laneCount"]
    rows: list[dict] = []
    import numpy as _np
    for s in hb_steps:
        idx = _np.nonzero(ends == s)[0]
        if idx.size == 0:
            continue
        w = tape["data"][idx[0]]
        wall = hb.get((s, policy), {}).get("wallSeconds")
        def g(name, lane):
            v = float(w[fi[name], lane])
            return v if math.isfinite(v) else None
        for lane in range(B):
            death = death_by_lane.get(lane)
            dead = death is not None and death["step"] <= s
            pam = float(w[fi["paramAbsMax"], lane]) if "paramAbsMax" in fi else -1.0
            row = {
                "v": 1, "kind": "hb", "envId": envs.get(seeds[lane], env_default),
                "policy": policy, "ablationLabel": ablation,
                "lane": lane, "seed": seeds[lane], "step": s,
                "rewardSum": g("rewardSum", lane),
                "doneCount": int(w[fi["doneCount"], lane]),
                "firstNonFiniteStep": death["step"] if dead else None,
                "firstNonFiniteDetector": death["detector"] if dead else None,
                "kernelRho": g("kernelRho", lane),
                "paramAbsMax": pam if math.isfinite(pam) and pam >= 0 else None,
                "health": {
                    "rewardRate": g("rewardRate", lane),
                    "actionDeltaL2Mean": (g("deltaSum", lane) or 0.0) / max(s, 1),
                    "actionDeltaPeak": g("deltaPeak", lane),
                    "mainTd": g("mainTd", lane),
                    "actorSupportEntropy": g("actorSupportEntropy", lane),
                    "terminationRate": g("terminationRate", lane),
                    "keyboardEffectiveL1": g("keyboardEffectiveL1", lane),
                    "keyboardL1": g("keyboardL1", lane),
                    "actorAdvantageL1": g("actorAdvantageL1", lane),
                    "modelPriorityMax": g("modelPriorityMax", lane),
                    "importanceEss": g("importanceEss", lane),
                    "actorUpdateNormMean": g("actorUpdateNormMean", lane),
                    "searchControlUseMean": g("searchControlUseMean", lane),
                    "selectedOptionPriority": g("selectedOptionPriority", lane),
                    "selectedContinuation": g("selectedContinuation", lane),
                    "generateTestEligibleCount": g("generateTestEligibleCount", lane),
                    "oakFeatureReinitCountFinal": int(w[fi["featureReinitCount"], lane]),
                    "oakOptionReinitCountFinal": int(w[fi["optionReinitCount"], lane]),
                    "nonFiniteByPhase": {
                        "encode": int(w[fi["nfEncode"], lane]),
                        "learn": int(w[fi["nfLearn"], lane]),
                        "plan": int(w[fi["nfPlan"], lane]),
                        "tdPsi": int(w[fi["nfTdPsi"], lane]),
                    },
                },
            }
            if wall and wall > 0:
                row["wallSeconds"] = wall
                row["stepsPerSecond"] = s / wall
            rows.append(row)
    return rows
