#!/usr/bin/env python3
"""sim-core golden fixtures: Philox4x32-10, u01 edges, and cheetah FK.

Oracles:
    tools/gridworld_ref.py : _philox4x32_10, _u01     (the bitwise contract, imported)
    _fk_oracle (below)     : the FK contract — the verbatim reference formerly in
                             tools/replay_html.py (retired with the old 2D viewer),
                             still hand-verified against HAND_ZERO / HAND_BENT

Emits, next to this file:
    philox_vectors.json   64 key/counter cases across streams 0-3 + per-word u01
    u01_edges.json        u01 / u01open boundary values
    fk_cheetah.json       qpos=0 and the bent pose (both hand-verified in the
                          replay_html selftest) + 50 random qpos rows

All floats are written as exact doubles: np.float32 -> float is lossless, so a
JS reader gets the identical value back and can compare with Object.is after
Math.fround.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TOOLS = _HERE.parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gridworld_ref as gw  # noqa: E402

# ---------------------------------------------------------------- FK oracle
# Verbatim reference from the retired tools/replay_html.py (planar x-z, hinges
# about +y; R_y(t): (x, z) -> (x cos t + z sin t, -x sin t + z cos t); angles
# accumulate down the chain; toe tips are the XML foot-geom capsules extended
# to their lower cap). HAND_ZERO / HAND_BENT below are its independent check.
import math

TOE_B = (0.055, -0.188)
TOE_F = (0.085, -0.128)


def _rot(t, x, z):
    c, s = math.cos(t), math.sin(t)
    return (x * c + z * s, -x * s + z * c)


def _fk_add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def _fk_oracle(qpos):
    q0, q1, q2, q3, q4, q5, q6, q7, q8 = (float(v) for v in qpos)
    t1 = q2
    t_bt = t1 + q3
    t_bs = t_bt + q4
    t_bf = t_bs + q5
    t_ft = t1 + q6
    t_fs = t_ft + q7
    t_ff = t_fs + q8
    o1 = (q0, 0.7 + q1)
    bt = _fk_add(o1, _rot(t1, -0.5, 0.0))
    ft = _fk_add(o1, _rot(t1, 0.5, 0.0))
    head = _fk_add(o1, _rot(t1, 0.6, 0.1))
    bs = _fk_add(bt, _rot(t_bt, 0.16, -0.25))
    bf = _fk_add(bs, _rot(t_bs, -0.28, -0.14))
    toe_b = _fk_add(bf, _rot(t_bf, *TOE_B))
    fs = _fk_add(ft, _rot(t_ft, -0.14, -0.24))
    ff = _fk_add(fs, _rot(t_fs, 0.13, -0.18))
    toe_f = _fk_add(ff, _rot(t_ff, *TOE_F))
    return dict(o1=o1, bt=bt, ft=ft, head=head, bs=bs, bf=bf, toe_b=toe_b,
                fs=fs, ff=ff, toe_f=toe_f)

VERSION = 1

STREAMS = [
    (0, "RESET"),
    (1, "CTRL"),
    (2, "ACT"),
    (3, "GRIDWORLD"),
]

# The 10 FK point names, in the CHEETAH_FK_POINTS order (src/sim/types.ts).
FK_POINTS = ["o1", "bt", "ft", "head", "bs", "bf", "toe_b", "fs", "ff", "toe_f"]

# replay_html selftest hand values (L534-556) — the independent check on the port.
HAND_ZERO = {
    "o1": (0.0, 0.7), "bt": (-0.5, 0.7), "ft": (0.5, 0.7), "head": (0.6, 0.8),
    "bs": (-0.34, 0.45), "bf": (-0.62, 0.31), "toe_b": (-0.565, 0.122),
    "fs": (0.36, 0.46), "ff": (0.49, 0.28), "toe_f": (0.575, 0.152),
}
BENT_QPOS = [1.5, 0.05, 0.1, 0.5, -0.8, 0.3, -0.4, 0.9, 0.2]
HAND_BENT = {
    "o1": (1.5, 0.75),
    "bt": (1.0024979173609871, 0.7999167083234141),
    "ft": (1.9975020826390129, 0.7000832916765859),
    "head": (2.1069858408314983, 0.7896003665397057),
    "bs": (0.9933909973977767, 0.5032400088527889),
    "bf": (0.7467860619135376, 0.3104032753323979),
    "toe_b": (0.7827426086742173, 0.1178516543443535),
    "fs": (1.9346798237601495, 0.4294297053538529),
    "ff": (1.9403378084873013, 0.2074657731287562),
    "toe_f": (1.9077362991466714, 0.0573120466058596),
}


def _u01(x: int) -> float:
    return float(gw._u01(x))


def _u01open(x: int) -> float:
    """hc_u01open: (float32(x >> 8) + 1) * 2^-24 (OakPhysics.metal L1269)."""
    return float((np.float32(x >> 8) + np.float32(1.0)) * np.float32(2.0 ** -24))


def _write(outdir: Path, name: str, obj) -> str:
    (outdir / name).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return name


# ------------------------------------------------------------------ philox
def _philox_cases() -> list[dict]:
    """64 cases: 16 per stream. Each stream gets the canonical OaK counter shape
    (idx, block, 0, 0) plus full 4-word counters that exercise c2/c3 mixing."""
    rng = np.random.default_rng(20260803)
    cases: list[dict] = []
    for stream, sname in STREAMS:
        shaped = [
            (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 5, 3),
            (7, 42, 0), (7, 42, 11), (0xFFFFFFFF, 0, 0), (0xFFFFFFFF, 123, 7),
        ]
        for seed, idx, block in shaped:
            cases.append({
                "stream": stream, "streamName": sname,
                "k0": int(seed), "k1": int(stream),
                "ctr": [int(idx), int(block), 0, 0],
                "shape": "oak",
            })
        for _ in range(8):
            k0 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            ctr = [int(v) for v in rng.integers(0, 1 << 32, size=4, dtype=np.uint64)]
            cases.append({
                "stream": stream, "streamName": sname,
                "k0": k0, "k1": int(stream), "ctr": ctr, "shape": "raw",
            })
    assert len(cases) == 64, len(cases)
    for c in cases:
        out = gw._philox4x32_10(c["k0"], c["k1"], c["ctr"])
        c["out"] = [int(w) for w in out]
        c["u01"] = [_u01(int(w)) for w in out]
    return cases


def _philox_vectors(outdir: Path) -> str:
    return _write(outdir, "philox_vectors.json", {
        "version": VERSION,
        "source": "tools/gridworld_ref.py:_philox4x32_10 (mirrors OakPhysics.metal hc_philox4x32)",
        "note": "key=(k0,k1) with k1=stream; ctr words are uint32; out/u01 are per-word.",
        "streams": {name: s for s, name in STREAMS},
        "cases": _philox_cases(),
    })


# ------------------------------------------------------------------ u01 edges
def _u01_edges(outdir: Path) -> str:
    xs = [
        0x00000000, 0x00000001, 0x0000007F, 0x000000FF, 0x00000100, 0x00000101,
        0x000001FF, 0x00000200, 0x0000FFFF, 0x00010000, 0x00FFFFFF, 0x01000000,
        0x7FFFFF00, 0x7FFFFFFF, 0x80000000, 0x80000001, 0x800000FF,
        0xABCDEF01, 0xDEADBEEF, 0xFFFFFEFF, 0xFFFFFF00, 0xFFFFFF01, 0xFFFFFFFE,
        0xFFFFFFFF,
    ]
    rng = np.random.default_rng(4489)
    xs += [int(v) for v in rng.integers(0, 1 << 32, size=16, dtype=np.uint64)]
    cases = [{"x": int(x), "u01": _u01(int(x)), "u01open": _u01open(int(x))} for x in xs]
    # structural facts the JS side must also satisfy
    assert cases[0]["u01"] == 0.0
    assert all(0.0 <= c["u01"] < 1.0 for c in cases)
    assert all(0.0 < c["u01open"] <= 1.0 for c in cases)
    return _write(outdir, "u01_edges.json", {
        "version": VERSION,
        "source": "OakPhysics.metal hc_u01/hc_u01open; tools/gridworld_ref.py:_u01",
        "note": "u01(x)=float32(x>>8)*2^-24 ; u01open(x)=(float32(x>>8)+1)*2^-24",
        "cases": cases,
    })


# ------------------------------------------------------------------ cheetah FK
def _fk_flat(qpos) -> list[float]:
    pts = _fk_oracle(qpos)
    flat: list[float] = []
    for name in FK_POINTS:
        p = pts[name]
        flat.append(float(p[0]))
        flat.append(float(p[1]))
    return flat


def _hand_flat(hand: dict) -> list[float]:
    flat: list[float] = []
    for name in FK_POINTS:
        flat.append(float(hand[name][0]))
        flat.append(float(hand[name][1]))
    return flat


def _fk_cheetah(outdir: Path) -> str:
    cases: list[dict] = []

    zero = [0.0] * 9
    zero_pts = _fk_flat(zero)
    zero_hand = _hand_flat(HAND_ZERO)
    err = max(abs(a - b) for a, b in zip(zero_pts, zero_hand))
    assert err <= 1e-12, f"qpos=0 FK disagrees with the selftest hand values: {err}"
    cases.append({"label": "zero", "qpos": zero, "pts": zero_pts, "hand": zero_hand})

    bent_pts = _fk_flat(BENT_QPOS)
    bent_hand = _hand_flat(HAND_BENT)
    err = max(abs(a - b) for a, b in zip(bent_pts, bent_hand))
    assert err <= 1e-9, f"bent FK disagrees with the selftest hand values: {err}"
    cases.append({"label": "bent", "qpos": list(BENT_QPOS), "pts": bent_pts,
                  "hand": bent_hand})

    # 50 random poses: rootx spans the treadmill, rootz the hop range, and every
    # hinge is drawn past +-pi so the port's angle accumulation is exercised
    # through the wrap region (FK sees only sin/cos, so no wrapping is applied).
    rng = np.random.default_rng(11173)
    for i in range(50):
        q = [
            float(rng.uniform(-60.0, 60.0)),
            float(rng.uniform(-0.6, 0.9)),
            *[float(rng.uniform(-2.0 * np.pi, 2.0 * np.pi)) for _ in range(7)],
        ]
        cases.append({"label": f"rand{i:02d}", "qpos": q, "pts": _fk_flat(q),
                      "hand": None})

    return _write(outdir, "fk_cheetah.json", {
        "version": VERSION,
        "source": "gen_core._fk_oracle (verbatim from the retired tools/replay_html.py fk)",
        "note": "pts = 20 doubles, [x,z] per point in `points` order; "
                "hand = the replay_html selftest hand values where they exist.",
        "points": FK_POINTS,
        "constants": {
            "torsoHalf": 0.5, "head": [0.6, 0.1], "bs": [0.16, -0.25],
            "bf": [-0.28, -0.14], "toeB": list(TOE_B), "fs": [-0.14, -0.24],
            "ff": [0.13, -0.18], "toeF": list(TOE_F),
            "rootZ0": 0.7, "radius": 0.046, "headRadius": 0.15,
        },
        "cases": cases,
    })


# ------------------------------------------------------------------ entry point
def generate(outdir: Path) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        _philox_vectors(outdir),
        _u01_edges(outdir),
        _fk_cheetah(outdir),
    ]


if __name__ == "__main__":
    for f in generate(_HERE):
        print(f"wrote {f}")
