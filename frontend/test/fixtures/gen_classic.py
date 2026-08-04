#!/usr/bin/env python3
"""Classic-control + HalfCheetah decoder goldens for the OaK frontend sim-core.

Owns (test/fixtures/):
    init_states.json          frame -1 state per env x 8 seeds (MERGED: other
                              generators' env keys are preserved)
    traces/<envId>.bin        first 200 record-v3 rows of a real recording
    traces/<envId>.json       the expected decode of those rows

Envs covered here: HalfCheetah-v4, Acrobot-cont-v0, MountainCar-cont-v0,
CartPole-cont-v0, Pendulum-cont-v0.  Gridworld init states / traces belong to
gen_grid.py; this generator never writes their keys.

Oracles (imported, never re-implemented):
    tools/gridworld_ref.py    _philox4x32_10, _u01   (the bitwise RNG contract)
    tools/physics_parity.py   classic_step, classic_init, _state_diff
                              (the fp64 classic-control dynamics)
    mujoco                    fp64 HalfCheetah, to validate the reset port

Two independent validations run at generation time and their results are
recorded in the fixtures:

  * HalfCheetah reset: seed fp64 MuJoCo with the derived (qpos, qvel), replay
    the recorded ctrl, compare against the recorded pose.  ~1e-8 at step 0 and
    ~5e-7 at step 30 is pure fp32-vs-fp64 drift, which proves the Philox draw
    order AND the qvel Box-Muller port are right.
  * Classic control: F1-style ONE-STEP parity.  Every recorded transition is
    re-stepped in fp64 from the recorded pre-state (the init state for row 0)
    and compared to the recorded post-state.  This pins the pose packing
    (which pose slot is which state variable) and the fleet init.

Both need mujoco (physics_parity imports it at module load).  Without it the
fixtures are still written, flagged `"validated": false`.

Run with the repo venv:
    /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py --only classic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gridworld_ref as gw  # noqa: E402

VERSION = 1

# The recording run every trace golden is cut from.
RUN_ROOT = _REPO / ".oak-benchmarks" / "frontend-demo-16env-20260803"
TRACE_STEPS = 200
RECORD_FLOATS = 16

# envId -> the 8 lane seeds that env owns in RUN_ROOT (meta.json envSchedule).
SEEDS = {
    "HalfCheetah-v4": list(range(0, 8)),
    "Acrobot-cont-v0": list(range(32, 40)),
    "MountainCar-cont-v0": list(range(40, 48)),
    "CartPole-cont-v0": list(range(48, 56)),
    "Pendulum-cont-v0": list(range(56, 64)),
}
ENVS = list(SEEDS)
# the lane each trace golden is cut from (the env's first seed)
TRACE_SEED = {e: s[0] for e, s in SEEDS.items()}

# --- kernel constants (Sources/HalfCheetahModel.metal, ClassicControl.metal) --
RESET_NOISE = 0.1        # HC_RESET_NOISE: qpos uniform half-width and qvel sigma
HC_QPOS0 = [0.0] * 9     # HC_QPOS0
CHEETAH_DT = 0.05        # HC_FRAME_SKIP * HC_TIMESTEP
CHEETAH_TELEPORT_DX = 10.0
ROOT_Z0 = 0.7            # world z = 0.7 + qpos[1] (replay_html fk)
STREAM_RESET = 0

MC_GOAL_POS = 0.5
MC_START_POS = -0.5
CP_X_LIMIT = float(np.float32(2.4))
CP_TH_LIMIT = float(np.float32(0.2094395))
CP_POLE_LENGTH = 1.0
CP_FORCE_MAG = 10.0
PD_MAX_TORQUE = 2.0

F32 = np.float32


# --------------------------------------------------------------- primitives
def _u01(x: int) -> np.float32:
    return gw._u01(x)


def _u01open(x: int) -> np.float32:
    """hc_u01open: (float32(x >> 8) + 1) * 2^-24 (OakPhysics.metal L1268)."""
    return (F32(x >> 8) + F32(1.0)) * F32(2.0**-24)


def _uniform(x: int, lo: float, hi: float) -> np.float32:
    """hc_uniform: lo + (hi - lo) * u01(x), in float32."""
    lo32, hi32 = F32(lo), F32(hi)
    return lo32 + (hi32 - lo32) * _u01(x)


def _philox_words(lane_seed: int, stream: int, idx: int, count: int, block0: int) -> list[int]:
    """`count` consecutive stream words from blocks block0, block0+1, ..."""
    out: list[int] = []
    b = block0
    while len(out) < count:
        out += gw._philox4x32_10(lane_seed, stream, [idx, b, 0, 0])
        b += 1
    return out[:count]


def cheetah_reset(lane_seed: int, reset_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Port of hc_rng_reset_noise + hc_env_reset (OakPhysics.metal L1271-1316).

    RESET stream, key = (laneSeed, 0):
      blocks 0..2 -> 12 words, first 9 are qpos noise U(-0.1, 0.1)
      blocks 3..5 -> 12 words, 6 Box-Muller pairs -> z[12], qvel = 0.1 * z[0..8]
    """
    raw_pos = _philox_words(lane_seed, STREAM_RESET, reset_idx, 12, 0)
    qpos = np.array(
        [F32(HC_QPOS0[i]) + _uniform(raw_pos[i], -RESET_NOISE, RESET_NOISE) for i in range(9)],
        dtype=F32)

    raw_vel = _philox_words(lane_seed, STREAM_RESET, reset_idx, 12, 3)
    z = np.empty(12, dtype=F32)
    for p in range(6):
        u1 = _u01open(raw_vel[2 * p])
        u2 = _u01(raw_vel[2 * p + 1])
        rad = F32(np.sqrt(F32(-2.0) * F32(np.log(np.float64(u1)))))
        ang = F32(F32(2.0 * np.pi) * u2)
        z[2 * p] = F32(rad * F32(np.cos(np.float64(ang))))
        z[2 * p + 1] = F32(rad * F32(np.sin(np.float64(ang))))
    qvel = np.array([F32(RESET_NOISE) * z[i] for i in range(9)], dtype=F32)
    return qpos, qvel


def classic_init_pose(env: str) -> np.ndarray:
    """The frame -1 pose[9] each env's oak_fleet_init produces."""
    pose = np.zeros(9, dtype=F32)
    if env == "MountainCar-cont-v0":
        pose[0] = F32(MC_START_POS)
        pose[2] = F32(MC_GOAL_POS)
    elif env == "Pendulum-cont-v0":
        pose[0] = F32(np.pi)
    # Acrobot / CartPole: hc_resetData -> all zeros
    return pose


# ------------------------------------------------------------------ records
def _read_rows(env: str, seed: int, steps: int | None = None) -> np.ndarray:
    p = RUN_ROOT / env / f"seed{seed}" / "actions.bin"
    if not p.exists():
        raise FileNotFoundError(f"missing recording {p}")
    a = np.fromfile(p, dtype="<f4")
    if a.size % RECORD_FLOATS:
        raise ValueError(f"{p}: {a.size} floats is not a multiple of {RECORD_FLOATS}")
    a = a.reshape(-1, RECORD_FLOATS)
    return a if steps is None else a[:steps]


def _d(x) -> float:
    """float32 -> the exact double JSON/JS round-trips."""
    return float(np.float64(x))


def _dl(a) -> list[float]:
    return [_d(v) for v in np.asarray(a).ravel()]


# -------------------------------------------------------------- validations
def _validate_cheetah_reset(seeds: list[int], steps: int = 30) -> list[dict]:
    """Seed fp64 MuJoCo with the derived reset and replay the recorded ctrl."""
    import physics_parity as pp
    import mujoco

    m = pp.load_model()
    d = mujoco.MjData(m)
    out: list[dict] = []
    for seed in seeds:
        rows = _read_rows("HalfCheetah-v4", seed, steps)
        qpos, qvel = cheetah_reset(seed, 0)
        # cross-check the local float32 helpers against the physics_parity ones
        exp = pp._expected_reset_qpos(m, seed, 0)
        assert np.array_equal(exp, qpos), f"seed {seed}: qpos port disagrees with physics_parity"
        d.qpos[:] = qpos.astype(np.float64)
        d.qvel[:] = qvel.astype(np.float64)
        d.qacc_warmstart[:] = 0.0
        errs = []
        for t in range(len(rows)):
            d.ctrl[:] = rows[t, 0:6].astype(np.float64)
            for _ in range(5):
                mujoco.mj_step(m, d)
            errs.append(float(np.abs(d.qpos - rows[t, 7:16].astype(np.float64)).max()))
        assert errs[0] < 1e-6, f"seed {seed}: step-0 replay error {errs[0]:.3e}"
        assert max(errs) < 1e-4, f"seed {seed}: replay error {max(errs):.3e}"
        out.append({"seed": seed, "steps": len(errs), "err0": errs[0],
                    "errLast": errs[-1], "errMax": max(errs)})
    return out


def _validate_classic_onestep(env: str, seed: int, rows: np.ndarray,
                              pre_pose: np.ndarray | None = None) -> dict:
    """F1-style one-step parity: re-step every recorded transition in fp64.

    `pre_pose` is the record row immediately before the window; None means the
    window starts at row 0, so the predecessor is the fleet-init state (which
    is cross-checked against physics_parity.classic_init here)."""
    import physics_parity as pp

    dims = pp.CLASSIC_STATE_DIMS[env]
    angle = pp.CLASSIC_ANGLE_DIMS[env]
    box = env in pp.CLASSIC_BOX
    k = len(dims)
    # pose slots holding the state, in CLASSIC_STATE_DIMS order
    pose_slots = list(range(k))

    if pre_pose is None:
        pre = pp.classic_init(env, np.float64)
        assert np.allclose(pre, classic_init_pose(env)[pose_slots].astype(np.float64)), (
            f"{env}: local init pose disagrees with physics_parity.classic_init")
    else:
        pre = pre_pose[7:7 + k].astype(np.float64)

    worst = 0.0
    done_mismatch = 0
    for t in range(len(rows)):
        c0 = float(rows[t, 0]) if box else float(int(rows[t, 0]))
        st, _r, dn = pp.classic_step(env, pre.copy(), c0)
        rec = rows[t, 7:7 + k].astype(np.float64)
        worst = max(worst, float(pp._state_diff(st, rec, angle).max()))
        if int(dn) != int(rows[t, 6]):
            done_mismatch += 1
        pre = rec  # re-anchor on the recorded state: one-step parity, no drift
    assert done_mismatch == 0, f"{env}: {done_mismatch} done-flag mismatches"
    assert worst < 1e-3, f"{env}: one-step parity max |err| {worst:.3e}"
    return {"env": env, "seed": seed, "steps": int(len(rows)), "maxAbsErr": worst,
            "doneMismatch": done_mismatch}


# ------------------------------------------------------------ decoded state
def _cheetah_state(pose: np.ndarray, prev_pose: np.ndarray | None,
                   init_qvel: np.ndarray | None) -> dict:
    if prev_pose is None:
        vel_x = _d(init_qvel[0]) if init_qvel is not None else 0.0
    else:
        dx = _d(pose[0]) - _d(prev_pose[0])
        vel_x = 0.0 if abs(dx) > CHEETAH_TELEPORT_DX else dx / CHEETAH_DT
    return {
        "rootX": _d(pose[0]),
        "rootZ": ROOT_Z0 + _d(pose[1]),
        "pitch": _d(pose[2]),
        "velX": vel_x,
    }


def _disc(c0: float) -> int:
    """uint(ctrl[0]), the kernel's discrete-action cast."""
    return int(np.uint32(int(c0)))


def _env_state(env: str, pose: np.ndarray, ctrl0: float,
               prev_pose: np.ndarray | None, init_qvel: np.ndarray | None,
               applied: bool = True) -> dict:
    """`applied` is False only for the synthesized init frame, where no action
    has been taken yet and the action-derived fields decode to exactly 0."""
    if env == "HalfCheetah-v4":
        return _cheetah_state(pose, prev_pose, init_qvel)
    if env == "Acrobot-cont-v0":
        th1, th2 = _d(pose[0]), _d(pose[1])
        return {
            "th1": th1, "th2": th2, "w1": _d(pose[2]), "w2": _d(pose[3]),
            "tipHeight": -np.cos(th1) - np.cos(th1 + th2),
            "action": _d(ctrl0),
            "torque": float(_disc(ctrl0) % 3 - 1) if applied else 0.0,
        }
    if env == "MountainCar-cont-v0":
        pos = _d(pose[0])
        return {
            "pos": pos, "vel": _d(pose[1]), "goal": _d(pose[2]),
            "height": np.sin(3 * pos) * 0.45 + 0.55,
            "action": _d(ctrl0),
            "force": float(_disc(ctrl0) % 3 - 1) if applied else 0.0,
        }
    if env == "CartPole-cont-v0":
        x, th = _d(pose[0]), _d(pose[2])
        force = CP_FORCE_MAG if (_disc(ctrl0) & 1) else -CP_FORCE_MAG
        return {
            "x": x, "xDot": _d(pose[1]), "th": th, "thDot": _d(pose[3]),
            "action": _d(ctrl0),
            "force": force if applied else 0.0,
            "xMargin": min(1.0, abs(x) / CP_X_LIMIT),
            "thMargin": min(1.0, abs(th) / CP_TH_LIMIT),
        }
    # Pendulum-cont-v0
    torque = _d(pose[2])
    return {
        "th": _d(pose[0]), "thDot": _d(pose[1]), "torque": torque,
        "action": _d(ctrl0), "effort": min(1.0, abs(torque) / PD_MAX_TORQUE),
    }


# state fields whose value is a transcendental of the pose: numpy libm and V8
# libm may differ in the last ulp, so the test compares these with a tolerance
# instead of exact float equality.
APPROX_FIELDS = {
    "Acrobot-cont-v0": ["tipHeight"],
    "MountainCar-cont-v0": ["height"],
}

# expected SimEvent types on a done row, in emission order
DONE_EVENTS = {
    "HalfCheetah-v4": ["done"],
    "Acrobot-cont-v0": ["goal", "done", "teleport"],
    "MountainCar-cont-v0": ["goal", "done", "teleport"],
    "CartPole-cont-v0": ["done", "teleport"],
    "Pendulum-cont-v0": ["done"],
}


# ----------------------------------------------------------------- emitters
def _write_json(path: Path, obj, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
    else:
        path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")


def _init_states(outdir: Path, cheetah_validation: list[dict]) -> str:
    name = "init_states.json"
    path = outdir / name
    doc = {"version": VERSION, "envs": {}, "generators": []}
    if path.exists():   # preserve other generators' env keys
        try:
            old = json.loads(path.read_text())
            if isinstance(old.get("envs"), dict):
                doc["envs"] = {k: v for k, v in old["envs"].items() if k not in ENVS}
            if isinstance(old.get("generators"), list):
                doc["generators"] = [g for g in old["generators"] if g != "classic"]
        except json.JSONDecodeError:
            pass
    doc["generators"] = sorted(doc["generators"] + ["classic"])
    doc["source"] = ("Sources/OakKernel.metal oak_fleet_init L2228-2428; "
                     "Sources/OakPhysics.metal hc_env_reset L1305-1316")
    doc["note"] = ("pose = the synthesized frame -1 pose[9]. HalfCheetah also "
                   "carries qvel[9] (the reset draw; not recoverable from the "
                   "record, only the init frame knows it). Classic-control "
                   "inits are seed-independent — the 8 rows are identical by "
                   "construction and exist so a decoder is checked per lane.")

    for env in ENVS:
        rows = []
        for seed in SEEDS[env]:
            if env == "HalfCheetah-v4":
                qpos, qvel = cheetah_reset(seed, 0)
                rows.append({"seed": seed, "pose": _dl(qpos), "qvel": _dl(qvel)})
            else:
                rows.append({"seed": seed, "pose": _dl(classic_init_pose(env))})
        doc["envs"][env] = {"seeds": rows}
    doc["envs"]["HalfCheetah-v4"]["resetValidation"] = {
        "method": ("fp64 MuJoCo seeded with the derived (qpos, qvel), replaying "
                   "the recorded ctrl against the recorded pose"),
        "validated": bool(cheetah_validation),
        "lanes": cheetah_validation,
    }
    _write_json(path, doc)
    return name


def _trace_doc(env: str, seed: int, rows: np.ndarray, window_start: int,
               validation: dict | None) -> dict:
    """The expected decode of `rows`, as `decodeRecording` produces it: the
    decoder's synthesized init frame followed by one frame per row."""
    init_qvel = cheetah_reset(seed, 0)[1] if env == "HalfCheetah-v4" else None
    init_pose = (cheetah_reset(seed, 0)[0] if env == "HalfCheetah-v4"
                 else classic_init_pose(env))

    out_rows = []
    prev = init_pose
    for t in range(len(rows)):
        pose = rows[t, 7:16]
        out_rows.append({
            "step": t,
            "done": _d(rows[t, 6]),
            "pose": _dl(pose),
            "state": _env_state(env, pose, float(rows[t, 0]), prev,
                                init_qvel if t == 0 else None),
        })
        prev = pose

    return {
        "version": VERSION,
        "envId": env,
        "seed": seed,
        "recordVersion": 3,
        "runRoot": RUN_ROOT.name,
        # first record row of this window inside the full recording
        "windowStart": int(window_start),
        "steps": int(len(rows)),
        "init": {
            "step": -1,
            "done": 0.0,
            "ctrl": [0.0] * 6,
            "pose": _dl(init_pose),
            "state": _env_state(env, init_pose, 0.0, None, init_qvel,
                                applied=False),
        },
        # full ctrl of the first 8 rows: pins the ctrl[6] view offset
        "ctrlHead": [_dl(rows[t, 0:6]) for t in range(min(8, len(rows)))],
        "rows": out_rows,
        "doneRows": [int(i) for i in np.nonzero(rows[:, 6])[0]],
        "doneEvents": DONE_EVENTS[env],
        "approxFields": APPROX_FIELDS.get(env, []),
        "oneStepParity": validation,
    }


# a done-carrying window starts this many rows before the first done, so the
# transition into it has a real predecessor frame
EVENT_WINDOW_LEAD = 20


def _trace(outdir: Path, env: str, validate) -> list[str]:
    seed = TRACE_SEED[env]
    tdir = outdir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)

    full = _read_rows(env, seed)
    if len(full) < TRACE_STEPS:
        raise ValueError(f"{env}: recording has only {len(full)} rows")

    written: list[str] = []
    head = full[:TRACE_STEPS]
    head.astype("<f4").tofile(tdir / f"{env}.bin")
    doc = _trace_doc(env, seed, head, 0, validate(env, seed, head))
    doc["bin"] = f"{env}.bin"
    doc["kind"] = "head"
    doc["initIsPredecessor"] = True
    _write_json(tdir / f"{env}.json", doc, compact=True)
    written += [f"traces/{env}.bin", f"traces/{env}.json"]

    # Acrobot's and MountainCar's first success is thousands of steps in, so
    # the head window never exercises goal/teleport. Cut a second window
    # around the first done so the event path is covered by real data. The
    # window is mid-stream, so its synthesized init frame is NOT the true
    # predecessor of row 0 — `initIsPredecessor: false` says so, and the
    # decoders whose events need this (Acrobot, MountainCar, CartPole) derive
    # events from the next frame alone.
    dones = np.nonzero(full[:, 6])[0]
    if len(np.nonzero(head[:, 6])[0]) == 0 and len(dones):
        start = max(0, int(dones[0]) - EVENT_WINDOW_LEAD)
        win = full[start:start + TRACE_STEPS]
        if len(win) == TRACE_STEPS and len(np.nonzero(win[:, 6])[0]):
            win.astype("<f4").tofile(tdir / f"{env}.events.bin")
            pre = full[start - 1] if start > 0 else None
            edoc = _trace_doc(env, seed, win, start, validate(env, seed, win, pre))
            edoc["bin"] = f"{env}.events.bin"
            edoc["kind"] = "events"
            edoc["initIsPredecessor"] = False
            _write_json(tdir / f"{env}.events.json", edoc, compact=True)
            written += [f"traces/{env}.events.bin", f"traces/{env}.events.json"]
    return written


# --------------------------------------------------------------- entry point
def generate(outdir: Path) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not RUN_ROOT.exists():
        raise FileNotFoundError(f"recording run root not found: {RUN_ROOT}")

    have_mujoco = True
    try:
        import physics_parity  # noqa: F401
    except Exception as e:  # mujoco missing -> fixtures still generate
        have_mujoco = False
        print(f"   WARNING: physics_parity unavailable ({type(e).__name__}: {e});"
              " writing fixtures WITHOUT the fp64 validations")

    cheetah_val: list[dict] = []
    if have_mujoco:
        cheetah_val = _validate_cheetah_reset(SEEDS["HalfCheetah-v4"])
        print(f"   cheetah reset validated on {len(cheetah_val)} lanes, "
              f"max |err| {max(v['errMax'] for v in cheetah_val):.3e}")

    def validate(env: str, seed: int, rows: np.ndarray,
                 pre: np.ndarray | None = None) -> dict | None:
        # HalfCheetah's fp64 oracle is the reset validation above (a full
        # MuJoCo re-step per row would just repeat it at 200x the cost).
        if not have_mujoco or env == "HalfCheetah-v4":
            return None
        v = _validate_classic_onestep(env, seed, rows, pre)
        print(f"   {env}: one-step parity max |err| {v['maxAbsErr']:.3e} "
              f"({v['steps']} rows)")
        return v

    written = [_init_states(outdir, cheetah_val)]
    for env in ENVS:
        written += _trace(outdir, env, validate)
    return written


if __name__ == "__main__":
    for f in generate(_HERE):
        print(f"wrote {f}")
