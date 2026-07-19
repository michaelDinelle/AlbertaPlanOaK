#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = str(Path(__file__).resolve().parent.parent / "assets" / "half_cheetah_envpool.xml")
FRAME_SKIP = 5
RESET_NOISE = 0.1

def load_model() -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(XML)
    assert (m.nq, m.nv, m.nu, m.nbody, m.ngeom) == (9, 9, 6, 8, 9), "model card mismatch"
    return m

def fmt(x: float) -> str:
    s = f"{x:.9g}"
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"

def arr(name: str, a: np.ndarray, per_row: int = 0) -> str:
    flat = np.asarray(a, dtype=np.float64).ravel()
    vals = [fmt(v) for v in flat]
    if per_row:
        rows = [", ".join(vals[i:i + per_row]) for i in range(0, len(vals), per_row)]
        body = ",\n    ".join(rows)
    else:
        body = ", ".join(vals)
    return f"constant float {name}[{len(flat)}] = {{\n    {body}\n}};"

def iarr(name: str, a: np.ndarray) -> str:
    vals = ", ".join(str(int(v)) for v in np.asarray(a).ravel())
    return f"constant int {name}[{len(np.asarray(a).ravel())}] = {{ {vals} }};"

def dump_metal_constants(m: mujoco.MjModel) -> None:
    print(f"constant float HC_TIMESTEP = {fmt(m.opt.timestep)};")
    print(f"constant float HC_GRAVITY_Z = {fmt(m.opt.gravity[2])};")
    print("constant float HC_SOLVER_TOL = 1e-5f;")
    print(f"constant float HC_LS_TOL = {fmt(m.opt.ls_tolerance)};")
    print(f"constant int HC_SOLVER_ITER = {int(m.opt.iterations)};")
    print(f"constant int HC_LS_ITER = {int(m.opt.ls_iterations)};")
    print(f"constant int HC_FRAME_SKIP = {FRAME_SKIP};")
    print(f"constant float HC_RESET_NOISE = {fmt(RESET_NOISE)};")
    print(iarr("HC_BODY_PARENT", m.body_parentid))
    print(arr("HC_BODY_POS", m.body_pos, 3))
    print(arr("HC_BODY_QUAT", m.body_quat, 4))
    print(arr("HC_BODY_IPOS", m.body_ipos, 3))
    print(arr("HC_BODY_IQUAT", m.body_iquat, 4))
    print(arr("HC_BODY_MASS", m.body_mass))
    print(arr("HC_BODY_INERTIA", m.body_inertia, 3))
    print(iarr("HC_JNT_TYPE", m.jnt_type))
    print(iarr("HC_JNT_BODY", m.jnt_bodyid))
    print(arr("HC_JNT_POS", m.jnt_pos, 3))
    print(arr("HC_JNT_AXIS", m.jnt_axis, 3))
    print(iarr("HC_JNT_LIMITED", m.jnt_limited))
    print(arr("HC_JNT_RANGE", m.jnt_range, 2))
    print(arr("HC_JNT_SOLREF", m.jnt_solref, 2))
    print(arr("HC_JNT_SOLIMP", m.jnt_solimp, 5))
    print(arr("HC_JNT_MARGIN", m.jnt_margin))
    print(arr("HC_DOF_ARMATURE", m.dof_armature))
    print(arr("HC_DOF_DAMPING", m.dof_damping))
    print(arr("HC_JNT_STIFFNESS", m.jnt_stiffness))
    print(arr("HC_QPOS_SPRING", m.qpos_spring))
    print(arr("HC_QPOS0", m.qpos0))
    print(iarr("HC_DOF_PARENT", m.dof_parentid))
    print(iarr("HC_GEOM_BODY", m.geom_bodyid))
    print(iarr("HC_GEOM_TYPE", m.geom_type))
    print(arr("HC_GEOM_POS", m.geom_pos, 3))
    print(arr("HC_GEOM_QUAT", m.geom_quat, 4))
    print(arr("HC_GEOM_SIZE", m.geom_size, 3))
    print(arr("HC_GEOM_FRICTION", m.geom_friction, 3))
    print(arr("HC_GEOM_SOLREF", m.geom_solref, 2))
    print(arr("HC_GEOM_SOLIMP", m.geom_solimp, 5))
    print(iarr("HC_ACT_TRNID", m.actuator_trnid[:, 0]))
    print(arr("HC_ACT_GEAR", m.actuator_gear[:, 0]))
    print(arr("HC_ACT_CTRLRANGE", m.actuator_ctrlrange, 2))

def reset_state(m: mujoco.MjModel, d: mujoco.MjData, rng: np.random.Generator) -> None:
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0 + rng.uniform(-RESET_NOISE, RESET_NOISE, m.nq)
    d.qvel[:] = RESET_NOISE * rng.standard_normal(m.nv)
    mujoco.mj_forward(m, d)

def golden(m: mujoco.MjModel, out: Path, cases: int, seed: int) -> None:
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    qpos_in, qvel_in, ctrl_in = [], [], []
    qpos_out, qvel_out, ncon, niter, warm = [], [], [], [], []
    per_rollout = 200
    while len(qpos_in) < cases:
        reset_state(m, d, rng)
        for t in range(per_rollout):
            if t % FRAME_SKIP == 0:
                d.ctrl[:] = rng.uniform(-1, 1, m.nu)
            if t >= 20 and rng.random() < 0.5 and len(qpos_in) < cases:
                qpos_in.append(d.qpos.copy())
                qvel_in.append(d.qvel.copy())
                ctrl_in.append(d.ctrl.copy())
                warm.append(d.qacc_warmstart.copy())
                mujoco.mj_step(m, d)
                qpos_out.append(d.qpos.copy())
                qvel_out.append(d.qvel.copy())
                ncon.append(d.ncon)
                niter.append(d.solver_niter[0])
            else:
                mujoco.mj_step(m, d)
    np.savez(
        out, qpos_in=qpos_in, qvel_in=qvel_in, ctrl=ctrl_in, warmstart=warm,
        qpos_out=qpos_out, qvel_out=qvel_out, ncon=ncon, niter=niter)
    print(f"golden: {len(qpos_in)} cases -> {out} "
          f"(ncon hist: {np.bincount(np.array(ncon)).tolist()})")

def reference(m: mujoco.MjModel, out: Path, seeds: list[int], steps: int) -> None:
    d = mujoco.MjData(m)
    all_data = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        reset_state(m, d, rng)
        all_data[f"seed{s}_qpos_init"] = d.qpos.copy()
        all_data[f"seed{s}_qvel_init"] = d.qvel.copy()
        qpos = np.empty((steps, m.nq))
        qvel = np.empty((steps, m.nv))
        ctrl = np.empty((steps, m.nu))
        reward = np.empty(steps)
        for t in range(steps):
            c = rng.uniform(-1, 1, m.nu)
            x_before = d.qpos[0]
            d.ctrl[:] = c
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
            xv = (d.qpos[0] - x_before) / (FRAME_SKIP * m.opt.timestep)
            reward[t] = xv - 0.1 * float(c @ c)
            qpos[t], qvel[t], ctrl[t] = d.qpos, d.qvel, c
        all_data[f"seed{s}_qpos"] = qpos
        all_data[f"seed{s}_qvel"] = qvel
        all_data[f"seed{s}_ctrl"] = ctrl
        all_data[f"seed{s}_reward"] = reward
    np.savez(out, seeds=seeds, **all_data)
    print(f"reference: {len(seeds)} seeds x {steps} env-steps -> {out}")

def energy(m: mujoco.MjModel, steps: int, seed: int) -> None:
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    reset_state(m, d, rng)
    d.ctrl[:] = 0
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    es = []
    for _ in range(steps):
        mujoco.mj_step(m, d)
        es.append(d.energy[0] + d.energy[1])
    es = np.array(es)
    print(f"energy reference: start {es[0]:.6f} end {es[-1]:.6f} "
          f"drift {(es[-1] - es[0]) / max(abs(es[0]), 1e-9):.3%} over {steps} steps")

def emit_probe_input(gold_npz: Path, out: Path) -> None:
    g = np.load(gold_npz)
    n = len(g["qpos_in"])
    rows = np.concatenate(
        [g["qpos_in"], g["qvel_in"], g["ctrl"], g["warmstart"]], axis=1
    ).astype(np.float32)
    with open(out, "wb") as f:
        f.write(np.uint32(n).tobytes())
        f.write(rows.tobytes())
    print(f"probe input: {n} cases x {rows.shape[1]} floats -> {out}")

def compare(metal_bin: Path, gold_npz: Path, tol: float) -> int:
    g = np.load(gold_npz)
    n = len(g["qpos_in"])
    raw = np.fromfile(metal_bin, dtype=np.float32)
    assert raw.size == n * 18, f"expected {n * 18} floats, got {raw.size}"
    out = raw.reshape(n, 18)
    ref = np.concatenate([g["qpos_out"], g["qvel_out"]], axis=1)
    scale = np.maximum(np.abs(ref), 1.0)
    rel = np.abs(out - ref) / scale
    worst = rel.max(axis=1)
    print(f"compare: n={n} median worst-component rel err {np.median(worst):.2e}, "
          f"p99 {np.percentile(worst, 99):.2e}, max {worst.max():.2e} (tol {tol:g})")
    bad = np.where(worst > tol)[0]
    if len(bad):
        print(f"FAIL: {len(bad)} cases over tol; first bad case {bad[0]} "
              f"(ncon={g['ncon'][bad[0]]}, niter={g['niter'][bad[0]]})")
        return 1
    print("PASS")
    return 0

EPISODE_LEN = 1000
ROLLOUT_RECORD = 37
DIVERGENCE_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

def emit_rollout_input(ref_npz: Path, out_bin: Path) -> None:
    g = np.load(ref_npz)
    seeds = [int(s) for s in g["seeds"]]
    steps = g[f"seed{seeds[0]}_ctrl"].shape[0]
    init = np.stack(
        [np.concatenate([g[f"seed{s}_qpos_init"], g[f"seed{s}_qvel_init"]]) for s in seeds]
    ).astype(np.float32)
    ctrl = np.stack([g[f"seed{s}_ctrl"] for s in seeds]).astype(np.float32)
    with open(out_bin, "wb") as f:
        f.write(np.array([len(seeds), steps, 0, 0, EPISODE_LEN], dtype=np.uint32).tobytes())
        f.write(init.tobytes())
        f.write(ctrl.tobytes())
    print(f"rollout input: {len(seeds)} lanes x {steps} env-steps -> {out_bin}")

def _replay(m: mujoco.MjModel, qpos_i, qvel_i, ctrl) -> tuple[np.ndarray, np.ndarray]:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos_i
    d.qvel[:] = qvel_i
    mujoco.mj_forward(m, d)
    steps = ctrl.shape[0]
    qpos = np.empty((steps, m.nq))
    qvel = np.empty((steps, m.nv))
    for t in range(steps):
        d.ctrl[:] = ctrl[t]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        qpos[t], qvel[t] = d.qpos, d.qvel
    return qpos, qvel

def rollout_compare(m: mujoco.MjModel, gpu_bin: Path, ref_npz: Path) -> int:
    g = np.load(ref_npz)
    seeds = [int(s) for s in g["seeds"]]
    lanes = len(seeds)
    steps = g[f"seed{seeds[0]}_ctrl"].shape[0]
    raw = np.fromfile(gpu_bin, dtype=np.float32)
    assert raw.size == lanes * steps * ROLLOUT_RECORD, (
        f"expected {lanes * steps * ROLLOUT_RECORD} floats, got {raw.size}")
    gpu = raw.reshape(lanes, steps, ROLLOUT_RECORD)
    gpu_state = gpu[:, :, 19:37].astype(np.float64)
    ref = np.stack(
        [np.concatenate([g[f"seed{s}_qpos"], g[f"seed{s}_qvel"]], axis=1) for s in seeds])

    obs_expected = np.concatenate([gpu[:, :, 20:28], gpu[:, :, 28:37]], axis=2)
    assert np.array_equal(gpu[:, :, :17], obs_expected), "GPU obs/state mismatch"
    done = gpu[:, :, 18]
    assert np.array_equal(done[:, : min(steps, EPISODE_LEN) - 1], np.zeros_like(done[:, :-1])[:, : min(steps, EPISODE_LEN) - 1]), "premature done flag"
    if steps >= EPISODE_LEN:
        assert np.all(done[:, EPISODE_LEN - 1] == 1.0), "missing done at episode end"
    scale = np.maximum(np.abs(ref), 1.0)
    gpu_div = (np.abs(gpu_state - ref) / scale).max(axis=(0, 2))
    twin_div = np.zeros(steps)
    for k, s in enumerate(seeds):
        tq, tv = _replay(
            m, g[f"seed{s}_qpos_init"], g[f"seed{s}_qvel_init"] + 1e-7,
            g[f"seed{s}_ctrl"])
        twin = np.concatenate([tq, tv], axis=1)
        twin_div = np.maximum(
            twin_div, (np.abs(twin - ref[k]) / np.maximum(np.abs(ref[k]), 1.0)).max(axis=1))
    table = [t for t in DIVERGENCE_STEPS if t <= steps]
    print(f"rollout-compare: {lanes} lanes x {steps} env-steps")
    print(f"{'step':>6} {'gpu-vs-cpu':>12} {'twin envelope':>14} {'ratio':>10}")
    ok_env = True
    for t in table:
        gd, td = gpu_div[t - 1], twin_div[t - 1]
        ratio = gd / td if td > 0 else float("inf")
        passed = gd <= 10 * td or gd <= 1e-3
        ok_env &= passed
        print(f"{t:>6} {gd:>12.3e} {td:>14.3e} {ratio:>10.2f} {'' if passed else ' <-- OVER'}")
    early = gpu_div[: min(10, steps)].max()
    ok_early = early < 1e-3
    print(f"first-10 max rel err: {early:.3e} (gate 1e-3) {'PASS' if ok_early else 'FAIL'}")
    print(f"envelope gate (<=10x twin or <=1e-3 at each step): {'PASS' if ok_env else 'FAIL'}")
    print("ROLLOUT PASS" if ok_early and ok_env else "ROLLOUT FAIL")
    return 0 if (ok_early and ok_env) else 1

# ---------------------------------------------------------------- classic control
# fp64 oracle (+ numpy fp32 shadow null) for the continuing classic-control
# envs ported to Sources/ClassicControl.metal. Certification standard: the
# F4-verdict precedent — (a) an F1-style ONE-STEP parity gate (every recorded
# (state, action) transition replayed in fp64; chaotic trajectories are never
# compared over long horizons), and (b) an F4-style POPULATION TOST on
# per-lane metrics (rewardSum, doneCount, state means), unpaired across
# backends (the F4 gate-calibration finding: paired-by-seed tests are invalid
# across physics backends), validated against a same-source null (the fp32
# numpy shadow vs the fp64 oracle at the same bands).
#
# The input run root is a fleet run of the env with the RANDOM policy and all
# lanes recorded, e.g.:
#   OAK_BENCH_ENVS="Acrobot-cont-v0:512" OAK_BENCH_SEED_COUNT=512 \
#     OAK_BENCH_STEPS=500 OAK_BENCH_POLICIES=random OAK_RECORD_ACTIONS=1 \
#     OAK_METRICS_STORAGE=<root> .build/release/OaK benchmark
CLASSIC_ENVS = ("Acrobot-cont-v0", "MountainCar-cont-v0", "CartPole-cont-v0",
                "Pendulum-cont-v0")
CLASSIC_STATE_DIMS = {
    "Acrobot-cont-v0": ["th1", "th2", "w1", "w2"],
    "MountainCar-cont-v0": ["pos", "vel"],
    "CartPole-cont-v0": ["x", "xdot", "th", "thdot"],
    "Pendulum-cont-v0": ["th", "thdot"],
}
CLASSIC_BOX = {"Pendulum-cont-v0"}   # action = ctrl[0]; others: int(ctrl[0])
CLASSIC_ANGLE_DIMS = {   # 2*pi-periodic state dims (wrapped state compared mod 2*pi)
    "Acrobot-cont-v0": [0, 1],
    "MountainCar-cont-v0": [],
    "CartPole-cont-v0": [2],
    "Pendulum-cont-v0": [0],
}

def _state_diff(a: np.ndarray, b: np.ndarray, angle_dims) -> np.ndarray:
    """|a-b| per dim, but angles compared mod 2*pi (the [-pi,pi) wrap makes
    -pi and +pi the same physical angle on both sides)."""
    d = np.abs(a - b)
    for j in angle_dims:
        d[..., j] = np.abs(_wrap_pi(a[..., j] - b[..., j]))
    return d

def _wrap_pi(x):
    # floor-mod into [-pi, pi); identical formula to gw_wrap_pi (Metal side).
    return x - 2.0 * np.pi * np.floor((x + np.pi) / (2.0 * np.pi))

def _acro_dsdt(st, tau):
    th1, th2, w1, w2 = st
    c2 = np.cos(th2)
    d1 = 3.5 + c2
    d2 = 1.25 + 0.5 * c2
    phi2 = 4.9 * np.cos(th1 + th2 - np.pi / 2.0)
    s2 = np.sin(th2)
    phi1 = -0.5 * w2 * w2 * s2 - w2 * w1 * s2 + 14.7 * np.cos(th1 - np.pi / 2.0) + phi2
    a2 = (tau + d2 / d1 * phi1 - 0.5 * w1 * w1 * s2 - phi2) / (d2 - d2 * d2 / d1)
    a1 = -(d2 * a2 + phi1) / d1
    return np.array([w1, w2, a1, a2], dtype=st.dtype)

def classic_step(env: str, st: np.ndarray, ctrl0: float):
    """One env step (dtype-generic: np.float64 oracle / np.float32 shadow).
    Mirrors gw_*_step in Sources/ClassicControl.metal formula for formula."""
    if env == "Acrobot-cont-v0":
        tau = float(int(ctrl0) % 3) - 1.0
        dt = 0.2
        k1 = _acro_dsdt(st, tau)
        k2 = _acro_dsdt(st + (dt / 2.0) * k1, tau)
        k3 = _acro_dsdt(st + (dt / 2.0) * k2, tau)
        k4 = _acro_dsdt(st + dt * k3, tau)
        st = st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        st[2] = np.clip(st[2], -4.0 * np.pi, 4.0 * np.pi)
        st[3] = np.clip(st[3], -9.0 * np.pi, 9.0 * np.pi)
        st[0] = _wrap_pi(st[0]); st[1] = _wrap_pi(st[1])
        r, d = 0.0, 0.0
        if -np.cos(st[0]) - np.cos(st[0] + st[1]) > 1.0:
            r, d = 1.0, 1.0
            st = np.zeros_like(st)
        return st, r, d
    if env == "MountainCar-cont-v0":
        pos, vel = st
        force = float(int(ctrl0) % 3) - 1.0
        vel = vel + force * 0.001 - 0.0025 * np.cos(3.0 * pos)
        vel = np.clip(vel, -0.07, 0.07)
        pos = pos + vel
        pos = np.clip(pos, -1.2, 0.6)
        if pos == -1.2 and vel < 0:
            vel = 0.0
        r, d = 0.0, 0.0
        if pos >= 0.5:
            r, d = 1.0, 1.0
            pos, vel = -0.5, 0.0
        return np.array([pos, vel], dtype=st.dtype), r, d
    if env == "CartPole-cont-v0":
        x, xd, th, thd = st
        force = 10.0 if (int(ctrl0) & 1) else -10.0
        s, c = np.sin(th), np.cos(th)
        temp = (force + 0.05 * thd * thd * s) / 1.1
        thacc = (9.8 * s - c * temp) / (0.5 * (4.0 / 3.0 - 0.1 * c * c / 1.1))
        xacc = temp - 0.05 * thacc * c / 1.1
        x = x + 0.02 * xd
        xd = xd + 0.02 * xacc
        th = th + 0.02 * thd
        thd = thd + 0.02 * thacc
        th = _wrap_pi(th)
        d = 0.0
        out = np.array([x, xd, th, thd], dtype=st.dtype)
        # thresholds as fp32-rounded literals — the Metal side compares the
        # fp32 state against the fp32 constants 2.4f / 0.2094395f
        if abs(x) > float(np.float32(2.4)) or abs(th) > float(np.float32(0.2094395)):
            d = 1.0
            out = np.zeros_like(out)
        return out, 1.0, d
    # Pendulum-cont-v0
    th, thd = st
    u = np.clip(2.0 * ctrl0, -2.0, 2.0)
    thn = _wrap_pi(th)
    cost = thn * thn + 0.1 * thd * thd + 0.001 * u * u
    nthd = thd + (15.0 * np.sin(th + np.pi) + 3.0 * u) * 0.05
    th = th + nthd * 0.05
    nthd = np.clip(nthd, -8.0, 8.0)
    th = _wrap_pi(th)
    return np.array([th, nthd], dtype=st.dtype), -cost, 0.0

def classic_init(env: str, dtype) -> np.ndarray:
    # Fleet-init state (oak_fleet_init); fp32-exact seeds of the stream.
    if env == "Acrobot-cont-v0":
        return np.zeros(4, dtype=dtype)
    if env == "MountainCar-cont-v0":
        return np.array([-0.5, 0.0], dtype=dtype)
    if env == "CartPole-cont-v0":
        return np.zeros(4, dtype=dtype)
    return np.array([float(np.float32(np.pi)), 0.0], dtype=dtype)

def _classic_replay(env: str, ctrls: np.ndarray, dtype):
    """Full-horizon replay from fleet init with the recorded actions.
    Returns (states T x k, rewards T, dones T)."""
    st = classic_init(env, dtype)
    T = len(ctrls)
    k = st.shape[0]
    states = np.empty((T, k), dtype=dtype)
    rewards = np.empty(T, dtype=dtype)
    dones = np.empty(T, dtype=dtype)
    for t in range(T):
        st, r, d = classic_step(env, st, float(ctrls[t]))
        states[t], rewards[t], dones[t] = st, r, d
    return states, rewards, dones

def classic_tost(run_root: Path) -> int:
    """Two-part certification of a ClassicControl env fleet recording:
    1. ONE-STEP parity (F1-style): replay every recorded transition in fp64;
       per-dim abs diff vs the recorded next state must stay <= 1e-4, and the
       done flag must flip on <= 0.01% of transitions.
    2. POPULATION TOST (F4-style): per-lane rewardSum / doneCount / state
       means, metal lanes vs fp64-oracle replay lanes (same action streams),
       unpaired Welch 95% CI inside a practical-irrelevance band of
       max(5% |oracle mean|, 1e-3) + KS p >= 0.01; the fp32-shadow-vs-oracle
       null must read the same verdict."""
    from scipy import stats

    meta = json.loads((run_root / "meta.json").read_text())
    sched = meta.get("envSchedule")
    envs = {e["envId"] for e in sched} if sched else {meta["envId"]}
    assert len(envs) == 1 and next(iter(envs)) in CLASSIC_ENVS, (
        f"--classic-tost needs a single-classic-env run root, got {envs}")
    env = next(iter(envs))
    dims = CLASSIC_STATE_DIMS[env]
    k = len(dims)
    lane_dirs = sorted((run_root / env).glob("seed*/"), key=lambda p: int(p.name[4:]))
    assert lane_dirs, f"no recorded lanes under {run_root / env}"
    fleet = [json.loads(x) for x in (run_root / "fleet.jsonl").read_text().splitlines()]
    finals = {r["seed"]: r for r in fleet if r.get("kind") == "final"}

    ok = True
    # ---- one-step parity (F1-style) --------------------------------------
    angle_dims = CLASSIC_ANGLE_DIMS[env]
    done_flips = 0
    n_trans = 0
    absdiff_by_dim = [[] for _ in range(k)]
    for p in lane_dirs:
        rec = np.fromfile(p / "actions.bin", dtype=np.float32).reshape(-1, 16)
        ctrl, done_rec, pose = rec[:, 0], rec[:, 6], rec[:, 7:7 + 9]
        n_trans += len(rec) - 1
        for t in range(1, len(rec)):
            st1, _, d1 = classic_step(env, pose[t - 1, :k].astype(np.float64),
                                      float(ctrl[t]))
            dd = _state_diff(st1, pose[t, :k].astype(np.float64), angle_dims)
            for j in range(k):
                absdiff_by_dim[j].append(dd[j])
            done_flips += int(d1 != done_rec[t])
    print(f"{env}: one-step parity over {len(lane_dirs)} lanes x "
          f"{n_trans // len(lane_dirs)} transitions")
    for j, name in enumerate(dims):
        a = np.asarray(absdiff_by_dim[j])
        med, p99, mx = np.median(a), np.percentile(a, 99), a.max()
        dim_ok = mx <= 1e-4
        ok &= dim_ok
        print(f"  {name:>6}: median {med:.2e} p99 {p99:.2e} max {mx:.2e} "
              f"(gate max <= 1e-4) {'PASS' if dim_ok else 'FAIL'}")
    flip_rate = done_flips / max(n_trans, 1)
    flips_ok = flip_rate <= 1e-4
    ok &= flips_ok
    print(f"  done flips: {done_flips}/{n_trans} ({flip_rate:.2e}, gate <= 1e-4) "
          f"{'PASS' if flips_ok else 'FAIL'}")

    # ---- population TOST (F4-style, unpaired) -----------------------------
    def lane_metrics(dtype_or_metal):
        """(rewardSum, doneCount, stateMean[k]) per lane."""
        rows = []
        for p in lane_dirs:
            seed = int(p.name[4:])
            rec = np.fromfile(p / "actions.bin", dtype=np.float32).reshape(-1, 16)
            if dtype_or_metal == "metal":
                f = finals[seed]
                rows.append([f["rewardSum"], float(f["doneCount"])]
                            + list(rec[:, 7:7 + k].astype(np.float64).mean(axis=0)))
            else:
                st, rw, dn = _classic_replay(env, rec[:, 0], dtype_or_metal)
                rows.append([float(rw.sum()), float(dn.sum())]
                            + list(st.astype(np.float64).mean(axis=0)))
        return np.array(rows)

    metal_m = lane_metrics("metal")
    ref_m = lane_metrics(np.float64)
    null_m = lane_metrics(np.float32)
    names = ["rewardSum", "doneCount"] + [f"mean({d})" for d in dims]

    # Same-physics null (F4 gate-calibration law): the fp64 oracle's lanes
    # split into disjoint halves; the null CI per metric is the sampling-noise
    # floor — no equivalence band may be tighter than it.
    half = len(ref_m) // 2
    null_a, null_b = ref_m[:half], ref_m[half:2 * half]

    print(f"{env}: population TOST (unpaired, N={len(lane_dirs)} lanes/side; "
          f"same-physics null N={half}/half)")
    for j, name in enumerate(names):
        x, y = metal_m[:, j], ref_m[:, j]
        b0 = max(0.05 * abs(y.mean()), 1e-3)          # strict practical band
        dn = null_a[:, j].mean() - null_b[:, j].mean()
        if null_a[:, j].std() == 0.0 and null_b[:, j].std() == 0.0:
            null_half = abs(dn)
        else:
            se_n = np.sqrt(null_a[:, j].var(ddof=1) / half + null_b[:, j].var(ddof=1) / half)
            null_half = 1.96 * se_n
        band = max(b0, abs(dn) + null_half)            # null-calibrated band
        if x.std() == 0.0 and y.std() == 0.0:
            dmean = x.mean() - y.mean()
            verdict = "EQUIVALENT" if abs(dmean) <= band else "DIFFERENT"
            ci = (dmean, dmean)
            n_req = 0
        else:
            dmean = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            ci = (dmean - 1.96 * se, dmean + 1.96 * se)
            n_req = int(np.ceil(len(x) * (1.96 * se / max(b0, 1e-12)) ** 2))
            if ci[0] >= -b0 and ci[1] <= b0:
                verdict = "EQUIVALENT"
            elif ci[1] < -band or ci[0] > band:
                verdict = "DIFFERENT"
            elif ci[0] >= -band and ci[1] <= band:
                verdict = "EQUIVALENT(null-cal)"
            else:
                verdict = "INCONCLUSIVE"
        ks = stats.ks_2samp(x, y)
        metric_ok = verdict.startswith("EQUIVALENT") and ks.pvalue >= 0.01
        ok &= metric_ok
        extra = f" strict ±{b0:.4g}" if band > b0 else ""
        n_note = f" (strict band needs N~{n_req})" if verdict == "INCONCLUSIVE" else ""
        print(f"  {name:>12}: dmean {dmean:+.4g} CI95 [{ci[0]:+.4g}, {ci[1]:+.4g}] "
              f"band ±{band:.4g}{extra} KS p {ks.pvalue:.3f} -> {verdict}{n_note}"
              f"{'' if metric_ok else ' <-- FAIL'}")

    # fp32 numpy shadow vs fp64 oracle: the same swap verdict, transparently.
    print(f"{env}: fp32-shadow cross-check (should track metal-vs-fp64)")
    for j, name in enumerate(names):
        x, y = null_m[:, j], ref_m[:, j]
        dmean = x.mean() - y.mean()
        ks = stats.ks_2samp(x, y)
        print(f"  {name:>12}: dmean {dmean:+.4g} KS p {ks.pvalue:.3f}")
    print("CLASSIC TOST PASS" if ok else "CLASSIC TOST FAIL")
    return 0 if ok else 1

# ------------------------------------------------- classic boundary case gate
# Single-step case probe (oak_classic_probe kernel) that forces the boundary
# paths the random-policy fleet may never reach — respawn-on-complete
# (MountainCar hilltop, Acrobot swing-up, CartPole fall) and threshold logic.
# fp64 golden at fp32 tolerance (state/reward 1e-4, done flag exact-counted).
CLASSIC_PROBE_W = {
    "MountainCar-cont-v0": (3, 4),
    "Acrobot-cont-v0": (5, 6),
    "CartPole-cont-v0": (7, 6),
    "Pendulum-cont-v0": (4, 4),
}

def classic_probe_emit(env: str, out: Path, n: int, seed: int) -> None:
    assert env in CLASSIC_PROBE_W, f"unknown classic env {env}"
    rng = np.random.default_rng(seed)
    rows = []
    if env == "MountainCar-cont-v0":
        for pos in (-1.2, -1.199, -0.5, 0.0, 0.49, 0.499, 0.5, 0.501, 0.6):
            for vel in (-0.07, -0.01, 0.0, 0.01, 0.07):
                for a in (0, 1, 2):
                    rows.append([pos, vel, float(a)])
        while len(rows) < n:
            rows.append([rng.uniform(-1.2, 0.6), rng.uniform(-0.07, 0.07),
                         float(rng.integers(0, 3))])
    elif env == "Acrobot-cont-v0":
        b = 2.0944   # swing-up height boundary: |th1| at th2 = 0
        for th1 in (-b - 0.001, -b, b, b + 0.001, 0.0, np.pi):
            for th2 in (0.0, -0.5, 0.5, 2.0, -2.0):
                for w in (0.0, 1.0, -1.0, 4.0):
                    for a in (0, 1, 2):
                        rows.append([th1, th2, w, -w, float(a)])
        while len(rows) < n:
            rows.append([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi),
                         rng.uniform(-12.57, 12.57), rng.uniform(-28.3, 28.3),
                         float(rng.integers(0, 3))])
    elif env == "CartPole-cont-v0":
        for x in (-2.41, -2.4, -2.39, 0.0, 1.0, 2.39, 2.4, 2.41):
            for th in (-0.2099, -0.2094, -0.2089, 0.0, 0.2089, 0.2094, 0.2099, 0.5):
                for xd in (0.0, 1.0):
                    for thd in (0.0, 1.0, -1.0):
                        for a in (0, 1):
                            rows.append([x, xd, th, thd, float(a), 0.0, 0.0])
        while len(rows) < n:
            rows.append([rng.uniform(-2.4, 2.4), rng.uniform(-2.0, 2.0),
                         rng.uniform(-0.3, 0.3), rng.uniform(-2.0, 2.0),
                         float(rng.integers(0, 2)), 0.0, 0.0])
    else:  # Pendulum-cont-v0
        for th in (-np.pi, -np.pi + 0.001, -0.001, 0.0, 0.001, np.pi - 0.001,
                   np.pi, 1.5, -1.5):
            for thd in (-8.0, -1.0, 0.0, 1.0, 8.0):
                for u in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    rows.append([th, thd, u, 0.0])
        while len(rows) < n:
            rows.append([rng.uniform(-np.pi, np.pi), rng.uniform(-8.0, 8.0),
                         rng.uniform(-1.0, 1.0), 0.0])
    k = len(CLASSIC_STATE_DIMS[env])
    ins = np.array(rows[:n], dtype=np.float32)
    outs = []
    for r in ins:
        st = r[:k].astype(np.float64)
        act = float(r[k])
        st1, rw, dn = classic_step(env, st, act)
        outs.append([rw] + list(st1) + [dn])
    outs = np.array(outs, dtype=np.float32)
    with open(out, "wb") as f:
        np.array([ins.shape[0], ins.shape[1], outs.shape[1]], dtype=np.uint32).tofile(f)
        ins.tofile(f)
        outs.tofile(f)
    print(f"wrote {ins.shape[0]} {env} boundary cases -> {out} "
          f"(in {ins.shape[1]} f32, out {outs.shape[1]} f32)")

def classic_probe_compare(cases: Path, gpu: Path, tol: float = 1e-4) -> int:
    d = np.fromfile(cases, dtype=np.float32)
    n, wi, wo = np.fromfile(cases, dtype=np.uint32, count=3)
    ins = d[3:3 + n * wi].reshape(n, wi)
    exp = d[3 + n * wi:3 + n * wi + n * wo].reshape(n, wo)
    got = np.fromfile(gpu, dtype=np.float32).reshape(exp.shape)
    env = next(e for e, w in CLASSIC_PROBE_W.items() if w == (int(wi), int(wo)))
    k = len(CLASSIC_STATE_DIMS[env])
    angle_fields = [1 + j for j in CLASSIC_ANGLE_DIMS[env]]
    ok = True
    # state + reward fields: fp32-vs-fp64 tolerance (angles mod 2*pi)
    for j in range(wo - 1):
        if j in angle_fields:
            diff = np.abs(_wrap_pi(
                got[:, j].astype(np.float64) - exp[:, j].astype(np.float64)))
        else:
            diff = np.abs(got[:, j].astype(np.float64) - exp[:, j].astype(np.float64))
        lim = tol + tol * np.abs(exp[:, j].astype(np.float64))
        bad = int((diff > lim).sum())
        ok &= bad == 0
        print(f"  field {j}: max diff {diff.max():.2e} (tol {tol:g}+rel) "
              f"violations {bad} {'PASS' if bad == 0 else 'FAIL'}")
    # done flag (last field): exact, counted (threshold-straddling fp noise
    # may flip a handful; a logic bug flips a quarter of the boundary grid)
    dm = got[:, -1] != exp[:, -1]
    flip_rate = float(dm.mean())
    done_ok = flip_rate <= 0.01
    ok &= done_ok
    print(f"  done flag: {int(dm.sum())}/{n} mismatches ({flip_rate:.3%}, "
          f"gate <= 1%) {'PASS' if done_ok else 'FAIL'}")
    n_done = int((exp[:, -1] == 1).sum())
    print(f"  (golden done=1 cases: {n_done}/{n})")
    print("CLASSIC PROBE PASS" if ok else "CLASSIC PROBE FAIL")
    return 0 if ok else 1


def _philox4x32_10(key0: int, key1: int, ctr) -> list[int]:
    M0, M1 = 0xD2511F53, 0xCD9E8D57
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    c = list(ctr)
    k0, k1 = key0, key1
    for r in range(10):
        if r:
            k0 = (k0 + W0) & 0xFFFFFFFF
            k1 = (k1 + W1) & 0xFFFFFFFF
        p0 = (M0 * c[0]) & 0xFFFFFFFFFFFFFFFF
        p1 = (M1 * c[2]) & 0xFFFFFFFFFFFFFFFF
        hi0, lo0 = p0 >> 32, p0 & 0xFFFFFFFF
        hi1, lo1 = p1 >> 32, p1 & 0xFFFFFFFF
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
    return c

def _u01(x: int) -> np.float32:
    return np.float32(x >> 8) * np.float32(2.0**-24)

def _uniform(x: int, lo: float, hi: float) -> np.float32:
    lo32, hi32 = np.float32(lo), np.float32(hi)
    return lo32 + (hi32 - lo32) * _u01(x)

def _expected_reset_qpos(m: mujoco.MjModel, lane_seed: int, reset_idx: int) -> np.ndarray:
    raw = []
    for b in range(3):
        raw += _philox4x32_10(lane_seed, 0, [reset_idx, b, 0, 0])
    return np.array(
        [np.float32(m.qpos0[i]) + _uniform(raw[i], -RESET_NOISE, RESET_NOISE) for i in range(9)],
        dtype=np.float32)

def _expected_ctrl(lane_seed: int, step_idx: int) -> np.ndarray:
    raw = _philox4x32_10(lane_seed, 1, [step_idx, 0, 0, 0])
    raw += _philox4x32_10(lane_seed, 1, [step_idx, 1, 0, 0])
    return np.array([_uniform(raw[u], -1.0, 1.0) for u in range(6)], dtype=np.float32)

def sampling_test(m: mujoco.MjModel, probe_bin: Path) -> int:
    from scipy import stats

    x = np.fromfile(probe_bin, dtype=np.float32)
    assert x.size % 24 == 0, f"probe size {x.size} not a multiple of 24"
    x = x.reshape(-1, 24)
    n = x.shape[0]
    qpos, qvel, ctrl = x[:, :9], x[:, 9:18], x[:, 18:24]

    worst = 0.0
    for lane in [0, 1, 2, 137, n - 1]:
        worst = max(worst, np.abs(qpos[lane] - _expected_reset_qpos(m, lane, 0)).max())
        worst = max(worst, np.abs(ctrl[lane] - _expected_ctrl(lane, 0)).max())
    chain_ok = worst < 1e-6
    print(f"philox chain cross-check (5 lanes): max abs diff {worst:.2e} "
          f"{'PASS' if chain_ok else 'FAIL'}")
    qpos_noise = (qpos.astype(np.float64) - m.qpos0).ravel()
    qvel_all = qvel.astype(np.float64).ravel()
    ctrl_all = ctrl.astype(np.float64).ravel()
    ok = chain_ok
    checks = [
        ("reset qpos noise U(-0.1,0.1)", qpos_noise, ("uniform", (-RESET_NOISE, 2 * RESET_NOISE)),
         0.0, 2 * RESET_NOISE / np.sqrt(12), True),
        ("reset qvel N(0,0.1)", qvel_all, ("norm", (0.0, RESET_NOISE)), 0.0, RESET_NOISE, False),
        ("ctrl U(-1,1)", ctrl_all, ("uniform", (-1.0, 2.0)), 0.0, 2 / np.sqrt(12), True),
    ]
    for name, vals, (dist, args), mu, sigma, is_unif in checks:
        ks = stats.kstest(vals, dist, args=args)
        nn = vals.size
        se_mean = sigma / np.sqrt(nn)

        kurt = 9 / 5 if is_unif else 3
        se_std = sigma * np.sqrt((kurt - 1) / (4 * nn))
        mean_ok = abs(vals.mean() - mu) <= 3 * se_mean
        std_ok = abs(vals.std() - sigma) <= 3 * se_std
        ks_ok = ks.pvalue >= 0.01
        ok &= mean_ok and std_ok and ks_ok
        print(f"{name}: n={nn} KS D={ks.statistic:.2e} p={ks.pvalue:.3f} "
              f"mean {vals.mean():+.5f} (|z|={abs(vals.mean() - mu) / se_mean:.2f}) "
              f"std {vals.std():.5f} (|z|={abs(vals.std() - sigma) / se_std:.2f}) "
              f"{'PASS' if ks_ok and mean_ok and std_ok else 'FAIL'}")
    print("SAMPLING PASS" if ok else "SAMPLING FAIL")
    return 0 if ok else 1

def energy_emit(m: mujoco.MjModel, out_bin: Path, seed: int, steps: int) -> None:
    d = mujoco.MjData(m)
    reset_state(m, d, np.random.default_rng(seed))
    init = np.concatenate([d.qpos, d.qvel]).astype(np.float32)
    with open(out_bin, "wb") as f:
        f.write(np.array([1, steps, 3, 0, 0], dtype=np.uint32).tobytes())
        f.write(init.tobytes())
    print(f"energy input: 1 lane x {steps} mj-steps (seed {seed}) -> {out_bin}")

def energy_compare(m: mujoco.MjModel, gpu_bin: Path, seed: int, steps: int) -> int:
    raw = np.fromfile(gpu_bin, dtype=np.float32)
    assert raw.size == steps * 2, f"expected {steps * 2} floats, got {raw.size}"
    gpu = raw.reshape(steps, 2).astype(np.float64)
    d = mujoco.MjData(m)
    reset_state(m, d, np.random.default_rng(seed))
    d.ctrl[:] = 0
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    ref = np.empty((steps, 2))
    for t in range(steps):
        mujoco.mj_step(m, d)
        ref[t] = d.energy
    m.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    for t in range(3):
        print(f"  state {t}: gpu pot {gpu[t, 0]:+.6f} kin {gpu[t, 1]:+.6f} | "
              f"fp64 pot {ref[t, 0]:+.6f} kin {ref[t, 1]:+.6f}")
    tot_g, tot_r = gpu.sum(axis=1), ref.sum(axis=1)
    rel = np.abs(tot_g - tot_r) / np.maximum(np.abs(tot_r), 1e-9)
    band = rel[100:].max()
    band_ok = band <= 0.02
    dissipate = tot_g[-1] < tot_g[0] and tot_r[-1] < tot_r[0]
    print(f"energy: start gpu {tot_g[0]:.4f} fp64 {tot_r[0]:.4f} | "
          f"end gpu {tot_g[-1]:.4f} fp64 {tot_r[-1]:.4f}")
    print(f"band (max rel diff, t>=100): {band:.4%} (gate 2%) "
          f"{'PASS' if band_ok else 'FAIL'}; dissipation "
          f"{'PASS' if dissipate else 'FAIL'}")
    print("ENERGY PASS" if band_ok and dissipate else "ENERGY FAIL")
    return 0 if (band_ok and dissipate) else 1

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-metal-constants", action="store_true")
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--emit-probe-input", nargs=2, type=Path, metavar=("GOLD_NPZ", "OUT_BIN"))
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("METAL_BIN", "GOLD_NPZ"))
    ap.add_argument("--rollout-emit", nargs=2, type=Path, metavar=("REF_NPZ", "OUT_BIN"))
    ap.add_argument("--rollout-compare", nargs=2, type=Path, metavar=("GPU_BIN", "REF_NPZ"))
    ap.add_argument("--sampling-test", type=Path, metavar="PROBE_BIN")
    ap.add_argument("--energy-emit", type=Path, metavar="OUT_BIN")
    ap.add_argument("--energy-compare", type=Path, metavar="GPU_BIN")
    ap.add_argument("--energy-steps", type=int, default=10_000)
    ap.add_argument("--classic-tost", type=Path, metavar="RUN_ROOT",
                    help="classic-control env certification: one-step parity + "
                         "population TOST on a random-policy recorded fleet root")
    ap.add_argument("--classic-probe-emit", nargs=2, metavar=("ENV", "OUT_BIN"),
                    help="emit single-step boundary cases for a classic env")
    ap.add_argument("--classic-probe-compare", nargs=2, type=Path,
                    metavar=("CASES_BIN", "GPU_BIN"),
                    help="compare oak_classic_probe output at fp32 tolerance")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = load_model()
    rc = 0
    if args.dump_metal_constants:
        dump_metal_constants(m)
    if args.golden:
        golden(m, args.golden, args.cases, args.seed)
    if args.reference:
        reference(m, args.reference, [int(s) for s in args.seeds.split(",")], args.steps)
    if args.energy:
        energy(m, 10_000, args.seed)
    if args.emit_probe_input:
        emit_probe_input(args.emit_probe_input[0], args.emit_probe_input[1])
    if args.compare:
        rc |= compare(args.compare[0], args.compare[1], args.tol)
    if args.rollout_emit:
        emit_rollout_input(args.rollout_emit[0], args.rollout_emit[1])
    if args.rollout_compare:
        rc |= rollout_compare(m, args.rollout_compare[0], args.rollout_compare[1])
    if args.sampling_test:
        rc |= sampling_test(m, args.sampling_test)
    if args.energy_emit:
        energy_emit(m, args.energy_emit, args.seed, args.energy_steps)
    if args.energy_compare:
        rc |= energy_compare(m, args.energy_compare, args.seed, args.energy_steps)
    if args.classic_tost:
        rc |= classic_tost(args.classic_tost)
    if args.classic_probe_emit:
        classic_probe_emit(args.classic_probe_emit[0], Path(args.classic_probe_emit[1]),
                           args.cases, args.seed)
    if args.classic_probe_compare:
        rc |= classic_probe_compare(*args.classic_probe_compare, tol=args.tol)
    return rc

if __name__ == "__main__":
    sys.exit(main())
