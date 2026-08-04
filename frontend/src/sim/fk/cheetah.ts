// HalfCheetah planar forward kinematics — an exact port of tools/replay_html.py
// `fk()` (L87-114) and its `_rot` helper (L78-80).
//
// Planar x-z, hinges about +y. R_y(t): (x, z) -> (x cos t + z sin t,
// -x sin t + z cos t); angles accumulate down the chain. Body local positions
// are the assets/half_cheetah_envpool.xml body `pos` values; the two "toe"
// tips are the XML foot-geom capsules extended to their lower cap:
//   bfoot geom pos (0.03,-0.097) half-len 0.094 axisangle(0,1,0,-0.27)
//     -> toe = pos - 0.094*(sin(-0.27), cos(-0.27)) = (0.0551, -0.1876)
//   ffoot geom pos (0.045,-0.07) half-len 0.07 axisangle(0,1,0,-0.6)
//     -> toe = pos - 0.07*(sin(-0.6), cos(-0.6)) = (0.0845, -0.1278)
//
// Segments for the renderer: bt-ft, bt-bs, bs-bf, bf-toe_b, ft-fs, fs-ff,
// ff-toe_f, plus a head circle at `head` (r = headRadius).

import { CHEETAH_FK_POINTS } from '../types';
import type { CheetahGeometry } from '../types';

/** back-foot toe tip, in bfoot-local coordinates */
export const TOE_B: readonly [number, number] = [0.055, -0.188];
/** front-foot toe tip, in ffoot-local coordinates */
export const TOE_F: readonly [number, number] = [0.085, -0.128];

/** torso half-length (bt/ft offsets are -+ this along local x) */
export const TORSO_HALF = 0.5;
/** capsule radius shared by every body geom */
export const CHEETAH_RADIUS = 0.046;
/** head sphere radius */
export const CHEETAH_HEAD_RADIUS = 0.15;
/** root z offset: world z = 0.7 + qpos[1] */
export const ROOT_Z0 = 0.7;

/** index of each named point in the packed [x,z] output (point i -> 2i, 2i+1) */
export const FK_POINT_INDEX: Readonly<Record<(typeof CHEETAH_FK_POINTS)[number], number>> =
  Object.freeze(
    Object.fromEntries(CHEETAH_FK_POINTS.map((n, i) => [n, i])) as Record<
      (typeof CHEETAH_FK_POINTS)[number],
      number
    >,
  );

/** number of floats written by `fk` (10 points x [x, z]) */
export const FK_FLOATS = CHEETAH_FK_POINTS.length * 2;

/**
 * Core FK, writing 20 floats into any float array. Computed entirely in
 * double precision; a Float32Array destination rounds once, on store.
 *
 * qpos = [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot].
 * Point order is CHEETAH_FK_POINTS:
 *   o1, bt, ft, head, bs, bf, toe_b, fs, ff, toe_f.
 */
export function fkInto<T extends Float32Array | Float64Array>(
  qpos: ArrayLike<number>,
  out: T,
): T {
  const q0 = +(qpos[0] ?? 0);
  const q1 = +(qpos[1] ?? 0);
  const q2 = +(qpos[2] ?? 0);
  const q3 = +(qpos[3] ?? 0);
  const q4 = +(qpos[4] ?? 0);
  const q5 = +(qpos[5] ?? 0);
  const q6 = +(qpos[6] ?? 0);
  const q7 = +(qpos[7] ?? 0);
  const q8 = +(qpos[8] ?? 0);

  const t1 = q2;
  const tBt = t1 + q3;
  const tBs = tBt + q4;
  const tBf = tBs + q5;
  const tFt = t1 + q6;
  const tFs = tFt + q7;
  const tFf = tFs + q8;

  const c1 = Math.cos(t1);
  const s1 = Math.sin(t1);
  const cBt = Math.cos(tBt);
  const sBt = Math.sin(tBt);
  const cBs = Math.cos(tBs);
  const sBs = Math.sin(tBs);
  const cBf = Math.cos(tBf);
  const sBf = Math.sin(tBf);
  const cFt = Math.cos(tFt);
  const sFt = Math.sin(tFt);
  const cFs = Math.cos(tFs);
  const sFs = Math.sin(tFs);
  const cFf = Math.cos(tFf);
  const sFf = Math.sin(tFf);

  // o1
  const o1x = q0;
  const o1z = ROOT_Z0 + q1;
  // bt = o1 + rot(t1, -0.5, 0)
  const btx = o1x + (-TORSO_HALF * c1 + 0.0 * s1);
  const btz = o1z + (TORSO_HALF * s1 + 0.0 * c1);
  // ft = o1 + rot(t1, 0.5, 0)
  const ftx = o1x + (TORSO_HALF * c1 + 0.0 * s1);
  const ftz = o1z + (-TORSO_HALF * s1 + 0.0 * c1);
  // head = o1 + rot(t1, 0.6, 0.1)
  const hx = o1x + (0.6 * c1 + 0.1 * s1);
  const hz = o1z + (-0.6 * s1 + 0.1 * c1);
  // bs = bt + rot(t_bt, 0.16, -0.25)
  const bsx = btx + (0.16 * cBt + -0.25 * sBt);
  const bsz = btz + (-0.16 * sBt + -0.25 * cBt);
  // bf = bs + rot(t_bs, -0.28, -0.14)
  const bfx = bsx + (-0.28 * cBs + -0.14 * sBs);
  const bfz = bsz + (0.28 * sBs + -0.14 * cBs);
  // toe_b = bf + rot(t_bf, TOE_B)
  const tbx = bfx + (TOE_B[0] * cBf + TOE_B[1] * sBf);
  const tbz = bfz + (-TOE_B[0] * sBf + TOE_B[1] * cBf);
  // fs = ft + rot(t_ft, -0.14, -0.24)
  const fsx = ftx + (-0.14 * cFt + -0.24 * sFt);
  const fsz = ftz + (0.14 * sFt + -0.24 * cFt);
  // ff = fs + rot(t_fs, 0.13, -0.18)
  const ffx = fsx + (0.13 * cFs + -0.18 * sFs);
  const ffz = fsz + (-0.13 * sFs + -0.18 * cFs);
  // toe_f = ff + rot(t_ff, TOE_F)
  const tfx = ffx + (TOE_F[0] * cFf + TOE_F[1] * sFf);
  const tfz = ffz + (-TOE_F[0] * sFf + TOE_F[1] * cFf);

  out[0] = o1x;
  out[1] = o1z;
  out[2] = btx;
  out[3] = btz;
  out[4] = ftx;
  out[5] = ftz;
  out[6] = hx;
  out[7] = hz;
  out[8] = bsx;
  out[9] = bsz;
  out[10] = bfx;
  out[11] = bfz;
  out[12] = tbx;
  out[13] = tbz;
  out[14] = fsx;
  out[15] = fsz;
  out[16] = ffx;
  out[17] = ffz;
  out[18] = tfx;
  out[19] = tfz;
  return out;
}

/** CheetahGeometry.fk — allocation-free when `out` is supplied. */
export function fk(qpos: ArrayLike<number>, out?: Float32Array): Float32Array {
  return fkInto(qpos, out ?? new Float32Array(FK_FLOATS));
}

/** Double-precision FK, for tests/analysis that need sub-float32 agreement. */
export function fk64(qpos: ArrayLike<number>, out?: Float64Array): Float64Array {
  return fkInto(qpos, out ?? new Float64Array(FK_FLOATS));
}

/** The frozen CheetahGeometry instance every cheetah consumer should use. */
export const cheetahGeometry: CheetahGeometry = Object.freeze({
  fk,
  radius: CHEETAH_RADIUS,
  headRadius: CHEETAH_HEAD_RADIUS,
});

/** Renderer bone list: index pairs into CHEETAH_FK_POINTS. */
export const CHEETAH_BONES: readonly (readonly [number, number])[] = Object.freeze([
  [FK_POINT_INDEX.bt, FK_POINT_INDEX.ft],
  [FK_POINT_INDEX.bt, FK_POINT_INDEX.bs],
  [FK_POINT_INDEX.bs, FK_POINT_INDEX.bf],
  [FK_POINT_INDEX.bf, FK_POINT_INDEX.toe_b],
  [FK_POINT_INDEX.ft, FK_POINT_INDEX.fs],
  [FK_POINT_INDEX.fs, FK_POINT_INDEX.ff],
  [FK_POINT_INDEX.ff, FK_POINT_INDEX.toe_f],
] as const);
