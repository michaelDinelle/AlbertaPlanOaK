#!/usr/bin/env python3
"""Pack OaK pose streams into a zero-dependency HTML replay page ("wall of cheetahs").

Inputs (auto-detected from the path):
  * physics-rollout OUTPUT binary (a file): flat little-endian f32, LANE-MAJOR
    (lanes, steps, 37) records [obs17, reward, done, qpos9, qvel9], NO header.
    Layout confirmed against Sources/Main.swift runRollout (outFloats =
    lanes*steps*37, outBuf written raw) and tools/physics_parity.py
    rollout_compare (np.fromfile(...).reshape(lanes, steps, 37); state at
    [:, :, 19:37]; obs = qpos[1:9] + qvel[0:9]). lanes x steps is inferred from
    the file size (prefers 1024 lanes); override with --shape LANESxSTEPS.
  * record-v3 run root (a directory): <run>/<envId>/seed<N>/meta.json
    {envId, seed, actDim, recordVersion:3} + actions.bin = n x 16 f32 records
    [ctrl[6], done, pose[9]]. Gridworld lanes (envId starts with FourRooms or
    DeepSea) carry pose = [x, y, goalx, goaly, 0...]. recordVersion 2 streams
    (actDim f32 + 1 uint8 done, NO pose) are detected and skipped with a
    warning. If <run>/fleet.jsonl exists, per-lane rewardRate is recovered
    from the last heartbeat/final row (rewardSum/step).

Output: <out>/data.js (`const META = {...}; const QPOS = "<base64>";` — a
<script src> include works from file:// where fetch() does not) plus a copy of
the repo's index.html.

Packed buffer, per lane (three contiguous blocks, all little-endian):
  fp32 first rootx | int32 rootx deltas[F] (x0.01) | int16 angles[F][8].
Cheetah lanes: angles slot 0 (rootz) = round(value / 1e-4); slots 1..7 (rooty +
6 hinge joints) are wrapped to [-pi, pi] then round(value / 1e-4); rootx deltas
are int32 (NOT int16: episode resets teleport rootx by hundreds of meters,
which an int16 delta at 0.01 cannot represent — +-327.67 m max) and are encoded
against a decoder-simulated accumulator (no drift). Gridworld lanes: deltas and
angles all raw ints. Per-lane size = 4 + F*4 + F*16 bytes.

Extras:
  --synth OUT.bin     write a synthetic rollout binary (scripted gait,
                      per-lane phase/amplitude, ~10% flailer lanes, resets)
  --synth-v3 OUTDIR   write a synthetic record-v3 run root (cheetah +
                      FourRooms + DeepSea lanes + fleet.jsonl)
  --selftest          FK hand-value assertions + codec round-trip, then exit
  --verify            after packing, re-decode the emitted data.js and compare
                      against the source poses within quantization tolerance
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RECORD = 37          # physics-rollout output record width (f32)
V3_RECORD = 16       # record-v3 actions.bin record width (f32)
ANG_SCALE = 1e-4     # int16 quantum for rootz/rooty + the 6 joint angles
X_SCALE = 0.01       # int16 quantum for rootx deltas
DATA_VERSION = 1
CHEETAH_ENV = "HalfCheetah-v4"
GRID_PREFIXES = ("FourRooms", "DeepSea")
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- cheetah FK
# Planar x-z, hinges about +y. R_y(t): (x, z) -> (x cos t + z sin t,
# -x sin t + z cos t); angles accumulate down the chain. Body local positions
# are the assets/half_cheetah_envpool.xml body pos values. Foot "toe" tips are
# the XML foot-geom capsules extended to their lower cap:
#   bfoot geom pos (0.03,-0.097) half-len 0.094 axisangle(0,1,0,-0.27)
#     -> toe = pos - 0.094*(sin(-0.27), cos(-0.27)) = (0.0551, -0.1876)
#   ffoot geom pos (0.045,-0.07) half-len 0.07 axisangle(0,1,0,-0.6)
#     -> toe = pos - 0.07*(sin(-0.6), cos(-0.6)) = (0.0845, -0.1278)
TOE_B = (0.055, -0.188)
TOE_F = (0.085, -0.128)


def _rot(t: float, x: float, z: float) -> tuple[float, float]:
    c, s = math.cos(t), math.sin(t)
    return (x * c + z * s, -x * s + z * c)


def _add(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    return (p[0] + q[0], p[1] + q[1])


def fk(qpos) -> dict[str, tuple[float, float]]:
    """qpos = [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot].

    Returns the 10 skeleton points (x, z): torso origin o1, back/front torso
    ends bt/ft, head center, bshin/bfoot origins bs/bf, back toe, fshin/ffoot
    origins fs/ff, front toe. Segments: bt-ft, bt-bs, bs-bf, bf-toe_b,
    ft-fs, fs-ff, ff-toe_f (+ head circle at head, r=0.15).
    """
    q0, q1, q2, q3, q4, q5, q6, q7, q8 = (float(v) for v in qpos)
    t1 = q2
    t_bt = t1 + q3
    t_bs = t_bt + q4
    t_bf = t_bs + q5
    t_ft = t1 + q6
    t_fs = t_ft + q7
    t_ff = t_fs + q8
    o1 = (q0, 0.7 + q1)
    bt = _add(o1, _rot(t1, -0.5, 0.0))
    ft = _add(o1, _rot(t1, 0.5, 0.0))
    head = _add(o1, _rot(t1, 0.6, 0.1))
    bs = _add(bt, _rot(t_bt, 0.16, -0.25))
    bf = _add(bs, _rot(t_bs, -0.28, -0.14))
    toe_b = _add(bf, _rot(t_bf, *TOE_B))
    fs = _add(ft, _rot(t_ft, -0.14, -0.24))
    ff = _add(fs, _rot(t_fs, 0.13, -0.18))
    toe_f = _add(ff, _rot(t_ff, *TOE_F))
    return dict(o1=o1, bt=bt, ft=ft, head=head, bs=bs, bf=bf, toe_b=toe_b,
                fs=fs, ff=ff, toe_f=toe_f)


# ---------------------------------------------------------------- codec
def encode_lane(frames: np.ndarray, grid: bool) -> bytes:
    """frames: (F, 9) float pose -> fp32 x0 + int32 deltas[F] + int16 rest[F][8].

    Block layout keeps JS typed-array views contiguous. rootx deltas are int32
    at X_SCALE: episode resets teleport rootx by hundreds of meters, far past
    the +-327.67 an int16 delta could carry.
    """
    F = frames.shape[0]
    rest = np.empty((F, 8), dtype="<i2")
    if grid:
        rest[:, :] = np.clip(np.rint(frames[:, 1:9]), -32768, 32767).astype("<i2")
        dx = np.diff(np.rint(frames[:, 0]).astype(np.int64), prepend=0)
        deltas = np.clip(dx, -2**31, 2**31 - 1).astype("<i4")
        x0 = np.float32(0.0)
    else:
        # rootz (rest slot 0) is linear: plain clip. rooty + the 6 hinge joints
        # (rest slots 1..7) wrap to [-pi, pi] before quantizing — FK only sees
        # sin/cos, so a tumbling rooty stays exact instead of saturating.
        rest[:, 0] = np.clip(
            np.rint(frames[:, 1] / ANG_SCALE), -32768, 32767).astype("<i2")
        ang = (frames[:, 2:9] + np.pi) % (2 * np.pi) - np.pi
        rest[:, 1:8] = np.clip(
            np.rint(ang / ANG_SCALE), -32768, 32767).astype("<i2")
        x = frames[:, 0].astype(np.float64)
        x0 = np.float32(x[0])
        prev = float(x0)
        deltas = np.empty(F, dtype="<i4")
        for f in range(F):
            d = int(round((x[f] - prev) / X_SCALE))
            d = max(-2**31, min(2**31 - 1, d))
            prev += d * X_SCALE
            deltas[f] = d
    return struct.pack("<f", float(x0)) + deltas.tobytes() + rest.tobytes()


def decode_lane(chunk: bytes, F: int, grid: bool) -> np.ndarray:
    """Inverse of encode_lane -> (F, 9) float64 pose (rootx reconstructed)."""
    x0 = struct.unpack_from("<f", chunk, 0)[0]
    deltas = np.frombuffer(chunk, dtype="<i4", count=F, offset=4)
    rest = (np.frombuffer(chunk, dtype="<i2", count=F * 8, offset=4 + F * 4)
            .reshape(F, 8).astype(np.float64))
    q = np.empty((F, 9))
    if grid:
        q[:, 0] = x0 + np.cumsum(deltas.astype(np.float64))
        q[:, 1:9] = rest
    else:
        q[:, 0] = x0 + np.cumsum(deltas.astype(np.float64)) * X_SCALE
        q[:, 1] = rest[:, 0] * ANG_SCALE
        q[:, 2:9] = rest[:, 1:8] * ANG_SCALE
    return q


def lane_bytes(F: int) -> int:
    return 4 + F * 4 + F * 8 * 2


def is_grid_env(env_id: str) -> bool:
    return any(env_id.startswith(g) for g in GRID_PREFIXES)


def _ang_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference of angles, wrapped to [-pi, pi]."""
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _pose_err(orig: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    """(max angle/rootz error, max rootx error) for a cheetah lane."""
    e_ang = max(float(np.abs(orig[:, 1] - recon[:, 1]).max()),
                float(_ang_err(orig[:, 2:9], recon[:, 2:9]).max()))
    e_x = float(np.abs(orig[:, 0] - recon[:, 0]).max())
    return e_ang, e_x


# ---------------------------------------------------------------- readers
def read_rollout(path: Path, shape: str | None) -> list[dict]:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0 or raw.size % RECORD != 0:
        sys.exit(f"{path}: {raw.size} floats is not a positive multiple of "
                 f"{RECORD} — not a physics-rollout output binary?")
    total = raw.size // RECORD
    if shape:
        try:
            lanes, steps = (int(v) for v in shape.lower().split("x"))
        except ValueError:
            sys.exit(f"--shape must be LANESxSTEPS, got {shape!r}")
        if lanes * steps != total:
            sys.exit(f"--shape {lanes}x{steps} != {total} records in {path}")
    else:
        lanes = 0
        for cand in (1024, 2048, 4096, 8192, 16384, 512, 256, 128, 64, 32,
                     16, 8, 4, 2, 1):
            if total % cand == 0:
                lanes = cand
                break
        steps = total // lanes
        print(f"rollout shape inferred: {lanes} lanes x {steps} steps "
              f"(override with --shape LANESxSTEPS)")
    rec = raw.reshape(lanes, steps, RECORD)
    out = []
    for l in range(lanes):
        pose = rec[l, :, 19:28].astype(np.float64)
        reward = rec[l, :, 17].astype(np.float64)
        out.append(SimpleNamespace(kind="cheetah", envId=CHEETAH_ENV, seed=l, lane=l,
                                   pose=pose, rate=float(reward.mean())))
    return out


def _fleet_rates(root: Path) -> dict[int, float]:
    """seed -> delivered reward rate from the last fleet.jsonl row per lane."""
    p = root / "fleet.jsonl"
    rates: dict[int, float] = {}
    if not p.is_file():
        return rates
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            step, rs = row.get("step"), row.get("rewardSum")
            if step and rs is not None and "seed" in row:
                rates[int(row["seed"])] = float(rs) / float(step)
    return rates


def read_run_v3(root: Path) -> list[dict]:
    lanes: list[dict] = []
    rates = _fleet_rates(root)
    n_v2 = 0
    for meta_path in sorted(root.rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        ver = int(meta.get("recordVersion", 0))
        if ver == 2:
            n_v2 += 1
            continue
        if ver != 3:
            print(f"warning: {meta_path}: recordVersion {ver} unsupported, skipped",
                  file=sys.stderr)
            continue
        bin_path = meta_path.parent / "actions.bin"
        raw = np.fromfile(bin_path, dtype="<f4")
        if raw.size == 0 or raw.size % V3_RECORD != 0:
            print(f"warning: {bin_path}: {raw.size} floats not a multiple of "
                  f"{V3_RECORD}, skipped", file=sys.stderr)
            continue
        rec = raw.reshape(-1, V3_RECORD)
        env_id = str(meta.get("envId", CHEETAH_ENV))
        seed = int(meta.get("seed", len(lanes)))
        lanes.append(SimpleNamespace(
            kind="grid" if is_grid_env(env_id) else "cheetah",
            envId=env_id, seed=seed, lane=len(lanes),
            pose=rec[:, 7:16].astype(np.float64),
            rate=rates.get(seed)))
    if n_v2:
        print(f"warning: {n_v2} lane(s) use recordVersion 2 (actDim f32 + "
              f"1-byte done, NO pose) — skipped; re-record with v3 to replay "
              f"them", file=sys.stderr)
    return lanes


def parse_lane_spec(spec: str, n: int) -> list[int]:
    spec = spec.strip()
    if "," in spec:
        idx = [int(x) for x in spec.split(",") if x.strip()]
    elif "-" in spec:
        a, b = spec.split("-", 1)
        idx = list(range(int(a), int(b) + 1))
    else:
        idx = list(range(min(int(spec), n)))
    bad = [i for i in idx if i < 0 or i >= n]
    if bad:
        sys.exit(f"--lanes indices out of range (n={n}): {bad[:5]}")
    return idx


def decode_data_js(path: Path) -> tuple[dict, list[np.ndarray]]:
    text = path.read_text()
    meta_line = next(l for l in text.splitlines() if l.startswith("const META = "))
    meta = json.loads(meta_line[len("const META = "):].rstrip(";"))
    qpos_line = next(l for l in text.splitlines() if l.startswith('const QPOS = "'))
    blob = base64.b64decode(qpos_line[len('const QPOS = "'):].rstrip('";'))
    F = int(meta["frames"])
    per = lane_bytes(F)
    if len(blob) != per * len(meta["lanes"]):
        sys.exit(f"{path}: blob {len(blob)} bytes != {per} x {len(meta['lanes'])} lanes")
    poses = [decode_lane(blob[i * per:(i + 1) * per], F,
                         is_grid_env(lm["envId"]))
             for i, lm in enumerate(meta["lanes"])]
    return meta, poses


def cmd_pack(args: argparse.Namespace) -> int:
    src = Path(args.path)
    if src.is_dir():
        lanes = read_run_v3(src)
        src_kind = "record-v3"
    elif src.is_file():
        lanes = read_rollout(src, args.shape)
        src_kind = "rollout"
    else:
        sys.exit(f"{src}: not a file or directory")
    if not lanes:
        sys.exit(f"{src}: no readable lanes found")
    idx = (parse_lane_spec(args.lanes, len(lanes)) if args.lanes
           else list(range(len(lanes))))
    sel = [lanes[i] for i in idx]
    stride = max(1, args.stride)
    F = min((l.pose.shape[0] + stride - 1) // stride for l in sel)
    if F < 2:
        sys.exit(f"only {F} frame(s) after stride {stride} — nothing to animate")
    n_trunc = sum(1 for l in sel if (l.pose.shape[0] + stride - 1) // stride > F)
    if n_trunc:
        print(f"note: {n_trunc} lane(s) truncated to {F} frames (shortest stream)")

    blob = b"".join(
        encode_lane(l.pose[::stride][:F], grid=(l.kind == "grid")) for l in sel)
    meta = dict(
        version=DATA_VERSION,
        title=args.title or src.name,
        frames=F,
        stride=stride,
        srcKind=src_kind,
        qposScale=ANG_SCALE,
        xDeltaScale=X_SCALE,
        lanes=[dict(seed=l.seed, envId=l.envId,
                    rate=(None if l.rate is None else round(float(l.rate), 5)),
                    lane=l.lane)
               for l in sel],
    )
    data_js = ("// OaK replay data — generated by tools/replay_html.py; do not edit.\n"
               "const META = " + json.dumps(meta, separators=(",", ":")) + ";\n"
               "const QPOS = \"" + base64.b64encode(blob).decode("ascii") + "\";\n")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(data_js)
    viewer_src = REPO_ROOT / "index.html"
    if not viewer_src.is_file():
        sys.exit(f"viewer {viewer_src} not found")
    shutil.copyfile(viewer_src, out_dir / "index.html")
    print(f"packed {len(sel)} lanes x {F} frames (stride {stride}) "
          f"-> {out_dir}/data.js")
    print(f"  raw {len(blob) / 1e6:.2f} MB | base64 {len(blob) * 4 / 3e6:.2f} MB "
          f"| data.js {len(data_js) / 1e6:.2f} MB")
    if args.verify:
        meta2, poses2 = decode_data_js(out_dir / "data.js")
        assert meta2 == meta, "META round-trip mismatch"
        max_ang = max_x = 0.0
        for l, p2 in zip(sel, poses2):
            orig = l.pose[::stride][:F]
            if l.kind == "grid":
                if not np.array_equal(np.rint(orig), p2):
                    sys.exit("verify: FAIL (grid pose mismatch)")
            else:
                e_ang, e_x = _pose_err(orig, p2)
                max_ang, max_x = max(max_ang, e_ang), max(max_x, e_x)
        ok = max_ang <= ANG_SCALE * 0.5001 + 1e-9 and max_x <= X_SCALE * 0.5 + 1e-6
        print(f"verify: max angle/rootz err {max_ang:.3e} (tol {ANG_SCALE / 2:.1e}), "
              f"max rootx err {max_x:.3e} (tol {X_SCALE / 2:.1e}) "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


# ---------------------------------------------------------------- synth data
def _smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def gait_pose(rng: np.random.Generator, quality: float, steps: int,
              dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Scripted sinusoidal gait. quality in [0,1] -> forward speed ~8*quality
    m/s; quality < 0 -> flailer (tumbling, noisy joints, backward drift).
    Returns (qpos (steps,9), per-step reward)."""
    t = np.arange(steps)
    phase = rng.uniform(7.0, 13.0) * dt * t + rng.uniform(0, 2 * np.pi)
    q = np.empty((steps, 9))
    if quality >= 0:
        A = 0.15 + 0.55 * quality
        q[:, 1] = 0.03 * np.sin(2 * phase) + rng.normal(0, 0.004, steps)
        q[:, 2] = 0.06 * np.sin(0.5 * phase + 1.0)
        q[:, 3] = A * np.sin(phase)
        q[:, 6] = A * np.sin(phase + np.pi)
        q[:, 4] = -0.35 + 0.7 * A * np.sin(phase + 1.3)
        q[:, 7] = -0.35 + 0.7 * A * np.sin(phase + np.pi + 1.3)
        q[:, 5] = 0.25 + 0.5 * A * np.sin(phase + 2.3)
        q[:, 8] = 0.25 + 0.5 * A * np.sin(phase + np.pi + 2.3)
        v = quality * 8.0 * (0.92 + 0.08 * np.sin(phase))
        q[:, 0] = np.cumsum(v) * dt
        reward = v + rng.normal(0, 0.02, steps)
    else:
        q[:, 1] = -0.35 + 0.05 * np.sin(phase)
        q[:, 2] = np.cumsum(rng.normal(0.02, 0.06, steps))
        for j in range(3, 9):
            q[:, j] = _smooth(rng.uniform(-1.5, 1.5, steps))
        v = rng.normal(-0.05, 0.15, steps)
        q[:, 0] = np.cumsum(v) * dt
        reward = -0.05 + 0.1 * rng.standard_normal(steps)
    return q, reward


def synth_rollout(out: Path, lanes: int, steps: int, seed: int) -> None:
    """Synthetic physics-rollout OUTPUT binary: lane-major (lanes, steps, 37)
    f32 records [obs17, reward, done, qpos9, qvel9], no header, with
    1000-step episode resets (rootx re-zeros) like EPISODE_LEN=1000."""
    rng = np.random.default_rng(seed)
    rec = np.empty((lanes, steps, RECORD), dtype="<f4")
    dt = 0.05
    episode = 1000
    for l in range(lanes):
        u = rng.random()
        if u < 0.10:
            quality = -1.0                       # flailer
        elif u < 0.25:
            quality = rng.uniform(0.02, 0.15)    # weak
        else:
            quality = rng.uniform(0.15, 1.0)     # walker
        q, reward = gait_pose(rng, quality, steps)
        for ep in range(0, steps, episode):      # rootx re-zeros on reset
            q[ep:ep + episode, 0] -= q[ep, 0]
        qvel = np.gradient(q, dt, axis=0)
        rec[l, :, 0:8] = q[:, 1:9]
        rec[l, :, 8:17] = qvel
        rec[l, :, 17] = reward
        rec[l, :, 18] = 0.0
        rec[l, episode - 1::episode, 18] = 1.0
        rec[l, :, 19:28] = q
        rec[l, :, 28:37] = qvel
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.tofile(out)
    print(f"synth rollout: {lanes} lanes x {steps} steps -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")


def synth_run_v3(out: Path, seed: int) -> None:
    """Synthetic record-v3 run root: cheetah + FourRooms + DeepSea lanes and a
    minimal fleet.jsonl so the rate-lookup path is exercised."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gridworld_ref as gw

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    fleet_rows = []

    def write_lane(env_id: str, lane_seed: int, act_dim: int,
                   rec: np.ndarray, reward_sum: float) -> None:
        d = out / env_id / f"seed{lane_seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(
            dict(envId=env_id, seed=lane_seed, actDim=act_dim, recordVersion=3),
            indent=2, sort_keys=True))
        rec.astype("<f4").tofile(d / "actions.bin")
        fleet_rows.append(dict(v=1, kind="final", lane=len(fleet_rows),
                               seed=lane_seed, step=rec.shape[0],
                               rewardSum=round(reward_sum, 3)))

    steps = 600
    for i in range(16):                          # cheetah lanes, seeds 0-15
        quality = -1.0 if i % 8 == 7 else rng.uniform(0.1, 1.0)
        q, reward = gait_pose(rng, quality, steps)
        rec = np.zeros((steps, V3_RECORD))
        rec[:, 0:6] = np.clip(q[:, 3:9] * 2.0, -1, 1)   # plausible ctrl
        rec[:, 6] = 0.0                                # done
        rec[:, 7:16] = q
        write_lane(CHEETAH_ENV, i, 6, rec, float(reward.sum()))

    for i in range(8):                           # FourRooms lanes, seeds 100-107
        lane_seed = 100 + i
        x, y, gx, gy, since, respawn = gw.four_reset()
        rec = np.zeros((steps, V3_RECORD))
        reward_sum = 0.0
        for t in range(steps):
            a = int(rng.integers(0, 4))
            r, x, y, gx, gy, since, respawn = gw.four_step(
                lane_seed, respawn, x, y, gx, gy, since, a)[:7]
            reward_sum += float(r)
            rec[t, a] = 1.0
            rec[t, 7:16] = [x, y, gx, gy, 0, 0, 0, 0, 0]
        write_lane("FourRooms-cont-v0", lane_seed, 4, rec, reward_sum)

    for i in range(8):                           # DeepSea lanes, seeds 200-207
        lane_seed = 200 + i
        r_, c_, last_a, ml, mh = gw.deepsea_reset(lane_seed)
        rec = np.zeros((steps, V3_RECORD))
        reward_sum = 0.0
        for t in range(steps):
            a = int(rng.integers(0, 2))
            r, r_, c_, last_a = gw.deepsea_step(ml, mh, r_, c_, last_a, a)[:4]
            reward_sum += float(r)
            rec[t, a] = 1.0
            rec[t, 7:16] = [r_, c_, gw.DEEPSEA_N - 1, gw.DEEPSEA_N - 1,
                            0, 0, 0, 0, 0]
        write_lane("DeepSea-cont-v0", lane_seed, 2, rec, reward_sum)

    with (out / "fleet.jsonl").open("w") as f:
        for row in fleet_rows:
            f.write(json.dumps(row) + "\n")
    print(f"synth record-v3 run: {len(fleet_rows)} lanes -> {out}")


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def check(name: str, got, want, tol: float) -> None:
        nonlocal ok
        err = max(abs(g - w) for g, w in zip(got, want))
        good = err <= tol
        ok &= good
        print(f"  {name}: max err {err:.3e} {'PASS' if good else 'FAIL'}")

    print("FK hand-value assertions")
    pts = fk([0] * 9)
    check("qpos=0 o1", pts["o1"], (0.0, 0.7), 1e-12)
    check("qpos=0 bt", pts["bt"], (-0.5, 0.7), 1e-12)
    check("qpos=0 ft", pts["ft"], (0.5, 0.7), 1e-12)
    check("qpos=0 head", pts["head"], (0.6, 0.8), 1e-12)
    check("qpos=0 bs", pts["bs"], (-0.34, 0.45), 1e-12)
    check("qpos=0 bf", pts["bf"], (-0.62, 0.31), 1e-12)
    check("qpos=0 toe_b", pts["toe_b"], (-0.565, 0.122), 1e-12)
    check("qpos=0 fs", pts["fs"], (0.36, 0.46), 1e-12)
    check("qpos=0 ff", pts["ff"], (0.49, 0.28), 1e-12)
    check("qpos=0 toe_f", pts["toe_f"], (0.575, 0.152), 1e-12)

    bent = [1.5, 0.05, 0.1, 0.5, -0.8, 0.3, -0.4, 0.9, 0.2]
    pts = fk(bent)
    check("bent o1", pts["o1"], (1.5, 0.75), 1e-9)
    check("bent bt", pts["bt"], (1.0024979173609871, 0.7999167083234141), 1e-9)
    check("bent ft", pts["ft"], (1.9975020826390129, 0.7000832916765859), 1e-9)
    check("bent head", pts["head"], (2.1069858408314983, 0.7896003665397057), 1e-9)
    check("bent bs", pts["bs"], (0.9933909973977767, 0.5032400088527889), 1e-9)
    check("bent bf", pts["bf"], (0.7467860619135376, 0.3104032753323979), 1e-9)
    check("bent toe_b", pts["toe_b"], (0.7827426086742173, 0.1178516543443535), 1e-9)
    check("bent fs", pts["fs"], (1.9346798237601495, 0.4294297053538529), 1e-9)
    check("bent ff", pts["ff"], (1.9403378084873013, 0.2074657731287562), 1e-9)
    check("bent toe_f", pts["toe_f"], (1.9077362991466714, 0.0573120466058596), 1e-9)

    print("codec round-trip")
    rng = np.random.default_rng(7)
    for trial in range(20):
        F = int(rng.integers(2, 500))
        q, _ = gait_pose(rng, float(rng.uniform(-1, 1)), F)
        p2 = decode_lane(encode_lane(q, grid=False), F, grid=False)
        e_ang, e_x = _pose_err(q, p2)
        if e_ang > ANG_SCALE * 0.5001 + 1e-9 or e_x > X_SCALE * 0.5 + 1e-6:
            print(f"  trial {trial}: FAIL ang {e_ang:.3e} x {e_x:.3e}")
            ok = False
    print(f"  20 random gait lanes: ang <= {ANG_SCALE / 2:.1e}, "
          f"rootx <= {X_SCALE / 2:.1e} {'PASS' if ok else 'FAIL'}")
    # episode-reset teleport: rootx jumps -400 m mid-stream (int16 deltas would
    # clip at +-327.67 and corrupt the rest of the stream; int32 must not)
    q, _ = gait_pose(rng, 0.9, 64)
    q[32:, 0] -= 400.0
    p2 = decode_lane(encode_lane(q, grid=False), 64, grid=False)
    e_ang, e_x = _pose_err(q, p2)
    good = e_x <= X_SCALE * 0.5 + 1e-6 and e_ang <= ANG_SCALE * 0.5001 + 1e-9
    ok &= good
    print(f"  teleport -400m: rootx err {e_x:.3e}, ang err {e_ang:.3e} "
          f"{'PASS' if good else 'FAIL'}")
    grid = np.rint(rng.uniform(0, 10, (50, 9)))
    p2 = decode_lane(encode_lane(grid, grid=True), 50, grid=True)
    good = np.array_equal(grid, p2)
    ok &= good
    print(f"  grid int16 exact: {'PASS' if good else 'FAIL'}")

    print("gridworld wall map")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import gridworld_ref as gw
        n_free = sum(not gw.four_is_wall(x, y)
                     for y in range(gw.FOUR_H) for x in range(gw.FOUR_W))
        good = n_free == 68 and gw.DEEPSEA_N == 8
        ok &= good
        print(f"  FourRooms free cells {n_free} (expect 68), "
              f"DeepSea N {gw.DEEPSEA_N}: {'PASS' if good else 'FAIL'}")
    except ImportError as e:
        ok = False
        print(f"  gridworld_ref import failed: {e} FAIL")

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="rollout output binary or record-v3 run root")
    ap.add_argument("--out", help="output directory (data.js + index.html)")
    ap.add_argument("--stride", type=int, default=100, help="keep every Nth frame")
    ap.add_argument("--lanes", help="lane subset: '0-1023' | count N | '3,8,17'")
    ap.add_argument("--title", help="run title shown in the viewer header")
    ap.add_argument("--shape", help="rollout LANESxSTEPS (skip shape inference)")
    ap.add_argument("--verify", action="store_true",
                    help="re-decode emitted data.js and compare to source poses")
    ap.add_argument("--synth", type=Path, metavar="OUT.bin",
                    help="write a synthetic rollout binary and exit")
    ap.add_argument("--synth-lanes", type=int, default=1024)
    ap.add_argument("--synth-steps", type=int, default=2000)
    ap.add_argument("--synth-v3", type=Path, metavar="OUTDIR",
                    help="write a synthetic record-v3 run root and exit")
    ap.add_argument("--synth-seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.synth:
        synth_rollout(args.synth, args.synth_lanes, args.synth_steps, args.synth_seed)
        return 0
    if args.synth_v3:
        synth_run_v3(args.synth_v3, args.synth_seed)
        return 0
    if not args.path or not args.out:
        ap.error("pack mode requires <path> and --out (or use --synth/--synth-v3/--selftest)")
    return cmd_pack(args)


if __name__ == "__main__":
    sys.exit(main())
