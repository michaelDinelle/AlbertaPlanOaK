#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
    return rc

if __name__ == "__main__":
    sys.exit(main())
