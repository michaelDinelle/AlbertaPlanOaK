// HalfCheetah-v4 decoder — record-v3 pose = qpos9, plus the reset port that
// synthesizes frame -1.
//
// Authoritative sources:
//   Sources/OakKernel.metal  L2891-2917  (pose[i] = s.qpos[i], all 9)
//   Sources/OakPhysics.metal L1271-1316  (hc_rng_reset_noise, hc_env_reset)
//   Sources/HalfCheetahModel.metal L1,7,8,136 (HC_TIMESTEP, HC_FRAME_SKIP,
//                                             HC_RESET_NOISE, HC_QPOS0 = 0^9)
//
// Records hold only qpos (the env's obs also carries qvel, the record does
// not), so the decoder recovers velocity the same way the env computes its own
// reward term: xv = (rootx_t - rootx_{t-1}) / (frameSkip * timestep). The one
// frame that knows its true qvel is the synthesized init frame, which carries
// the reset draw's qvel[0].
//
// The reset port is validated end-to-end: seeding fp64 MuJoCo with the derived
// (qpos, qvel) and replaying the recorded ctrl reproduces the recorded pose to
// ~1e-8 at step 0 and ~5e-7 at step 30 (pure fp32-vs-fp64 drift). See
// test/fixtures/gen_classic.py `_validate_cheetah_reset`.

import { cheetahGeometry, ROOT_Z0 } from '../fk/cheetah';
import { STREAM_RESET, boxMuller, philoxWords, uniform } from '../philox';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

// ---------------------------------------------------------------- constants

/** `HC_TIMESTEP` — MuJoCo integrator step. */
export const HC_TIMESTEP = 0.01;
/** `HC_FRAME_SKIP` — physics steps per control step. */
export const HC_FRAME_SKIP = 5;
/** `HC_RESET_NOISE` — qpos uniform half-width AND qvel normal scale. */
export const HC_RESET_NOISE = 0.1;
/**
 * Control-step duration, the denominator of the env's own forward-velocity
 * term (`xv = dx / (HC_FRAME_SKIP * HC_TIMESTEP)`). Written as the exact
 * literal 0.05 so the TS and Python fixture sides divide by the same double;
 * the kernel evaluates `5.0f * 0.01f` in fp32, which is the same value.
 */
export const CHEETAH_DT = 0.05;
/**
 * A rootx jump larger than this is an episode reset / respawn, not motion.
 * At the env's own clamped scale a single control step moves the root by
 * well under 1 m, so 10 m is far outside any physical step.
 */
export const CHEETAH_TELEPORT_DX = 10;

/** number of generalized coordinates (record pose width) */
export const QPOS_LEN = 9;

// ------------------------------------------------------------------- types

export interface CheetahStatic {
  /** lane seed the reset was drawn for */
  seed: number;
  /** reset counter the init frame corresponds to (always 0 for frame -1) */
  resetIndex: number;
  /** qpos at frame -1: `HC_QPOS0 + U(-0.1, 0.1)^9`, HC_QPOS0 = 0^9 */
  initQpos: Float32Array;
  /** qvel at frame -1: `0.1 * N(0,1)^9` — not recoverable from the record */
  initQvel: Float32Array;
}

export interface CheetahState {
  /** the frame's qpos9 (the pose view itself — do not mutate) */
  qpos: Float32Array;
  /** qpos[0], world x of the torso root */
  rootX: number;
  /** world height of the torso root: `ROOT_Z0 + qpos[1]` */
  rootZ: number;
  /** qpos[2], torso pitch (rad, +y hinge) */
  pitch: number;
  /**
   * Forward velocity (m/s). Frames >= 0: the env's own
   * `(rootx_t - rootx_{t-1}) / CHEETAH_DT`, zeroed across a teleport.
   * Frame -1: the exact reset draw `qvel[0]`.
   */
  velX: number;
}

// -------------------------------------------------------------- reset port

/**
 * `hc_rng_reset_noise` + `hc_env_reset` (OakPhysics.metal L1271-1316).
 *
 * Draw order, RESET stream, key = (laneSeed, 0):
 *   blocks 0,1,2 -> 12 words, the first 9 are qpos noise U(-0.1, 0.1)
 *   blocks 3,4,5 -> 12 words, 6 Box-Muller pairs -> z[12], qvel = 0.1 * z[0..8]
 * HC_QPOS0 is all zeros, so qpos = the uniform noise itself.
 *
 * Precision: the uniform draws are bitwise-exact; the Box-Muller pair uses
 * libm log/cos/sin in double, so qvel agrees with the GPU to ~1 float32 ulp.
 * Nothing downstream of the init frame depends on qvel bit equality.
 */
export function cheetahReset(
  laneSeed: number,
  resetIndex = 0,
): { qpos: Float32Array; qvel: Float32Array } {
  const qpos = new Float32Array(QPOS_LEN);
  const qvel = new Float32Array(QPOS_LEN);

  const rawPos = philoxWords(laneSeed, STREAM_RESET, resetIndex, 12, 0);
  for (let i = 0; i < QPOS_LEN; i++) {
    qpos[i] = uniform(rawPos[i] ?? 0, -HC_RESET_NOISE, HC_RESET_NOISE);
  }

  const rawVel = philoxWords(laneSeed, STREAM_RESET, resetIndex, 12, 3);
  const z = new Float32Array(12);
  for (let p = 0; p < 6; p++) {
    const [a, b] = boxMuller(rawVel[2 * p] ?? 0, rawVel[2 * p + 1] ?? 0);
    z[2 * p] = a;
    z[2 * p + 1] = b;
  }
  for (let i = 0; i < QPOS_LEN; i++) qvel[i] = Math.fround(HC_RESET_NOISE * (z[i] ?? 0));

  return { qpos, qvel };
}

// ---------------------------------------------------------------- decoder

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'cheetah' as const,
  cheetah: cheetahGeometry,
});

/** Shared no-event result. Frozen: nothing may push into a decoder's events. */
function makeState(pose: Float32Array, velX: number): CheetahState {
  return {
    qpos: pose,
    rootX: pose[0] ?? 0,
    rootZ: ROOT_Z0 + (pose[1] ?? 0),
    pitch: pose[2] ?? 0,
    velX,
  };
}

export const halfCheetahDecoder: EnvDecoder<CheetahState, CheetahStatic> = {
  envId: 'HalfCheetah-v4',
  needsReplay: false,

  initStatic(seed: number): CheetahStatic {
    const { qpos, qvel } = cheetahReset(seed, 0);
    return { seed, resetIndex: 0, initQpos: qpos, initQvel: qvel };
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(_seed: number, st: CheetahStatic): EnvFrame<CheetahState> {
    // copy: the frame's pose must not alias the static's init draw
    const pose = Float32Array.from(st.initQpos);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, st.initQvel[0] ?? 0),
    };
  },

  decodeFrame(
    i: number,
    raw16: Float32Array,
    _st: CheetahStatic,
    prev: EnvFrame<CheetahState>,
  ): EnvFrame<CheetahState> {
    const pose = raw16.subarray(7, 16);
    const dx = (pose[0] ?? 0) - (prev.pose[0] ?? 0);
    // a teleport's dx is a reset artifact, not motion
    const velX = Math.abs(dx) > CHEETAH_TELEPORT_DX ? 0 : dx / CHEETAH_DT;
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, velX),
    };
  },

  deriveEvents(
    prev: EnvFrame<CheetahState>,
    next: EnvFrame<CheetahState>,
  ): SimEvent[] {
    // Continuing streams (OAK_EPISODE_STEPS=0) never emit done and never
    // teleport; both branches exist for gym-1000 recordings, where the
    // episode boundary resets qpos and snaps rootx back toward 0.
    const teleport = Math.abs((next.pose[0] ?? 0) - (prev.pose[0] ?? 0)) > CHEETAH_TELEPORT_DX;
    if (next.done === 0 && !teleport) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'done' });
    if (teleport) out.push({ type: 'teleport' });
    return out;
  },
};
