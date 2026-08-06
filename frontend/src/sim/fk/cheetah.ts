
import { CHEETAH_FK_POINTS } from '../types';
import type { CheetahGeometry } from '../types';

export const TOE_B: readonly [number, number] = [0.055, -0.188];
export const TOE_F: readonly [number, number] = [0.085, -0.128];

export const TORSO_HALF = 0.5;
export const CHEETAH_RADIUS = 0.046;
export const CHEETAH_HEAD_RADIUS = 0.15;
export const ROOT_Z0 = 0.7;

export const FK_POINT_INDEX: Readonly<Record<(typeof CHEETAH_FK_POINTS)[number], number>> =
  Object.freeze(
    Object.fromEntries(CHEETAH_FK_POINTS.map((n, i) => [n, i])) as Record<
      (typeof CHEETAH_FK_POINTS)[number],
      number
    >,
  );

export const FK_FLOATS = CHEETAH_FK_POINTS.length * 2;

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

  const o1x = q0;
  const o1z = ROOT_Z0 + q1;
  const btx = o1x + (-TORSO_HALF * c1 + 0.0 * s1);
  const btz = o1z + (TORSO_HALF * s1 + 0.0 * c1);
  const ftx = o1x + (TORSO_HALF * c1 + 0.0 * s1);
  const ftz = o1z + (-TORSO_HALF * s1 + 0.0 * c1);
  const hx = o1x + (0.6 * c1 + 0.1 * s1);
  const hz = o1z + (-0.6 * s1 + 0.1 * c1);
  const bsx = btx + (0.16 * cBt + -0.25 * sBt);
  const bsz = btz + (-0.16 * sBt + -0.25 * cBt);
  const bfx = bsx + (-0.28 * cBs + -0.14 * sBs);
  const bfz = bsz + (0.28 * sBs + -0.14 * cBs);
  const tbx = bfx + (TOE_B[0] * cBf + TOE_B[1] * sBf);
  const tbz = bfz + (-TOE_B[0] * sBf + TOE_B[1] * cBf);
  const fsx = ftx + (-0.14 * cFt + -0.24 * sFt);
  const fsz = ftz + (0.14 * sFt + -0.24 * cFt);
  const ffx = fsx + (0.13 * cFs + -0.18 * sFs);
  const ffz = fsz + (-0.13 * sFs + -0.18 * cFs);
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

export function fk(qpos: ArrayLike<number>, out?: Float32Array): Float32Array {
  return fkInto(qpos, out ?? new Float32Array(FK_FLOATS));
}

export function fk64(qpos: ArrayLike<number>, out?: Float64Array): Float64Array {
  return fkInto(qpos, out ?? new Float64Array(FK_FLOATS));
}

export const cheetahGeometry: CheetahGeometry = Object.freeze({
  fk,
  radius: CHEETAH_RADIUS,
  headRadius: CHEETAH_HEAD_RADIUS,
});

export const CHEETAH_BONES: readonly (readonly [number, number])[] = Object.freeze([
  [FK_POINT_INDEX.bt, FK_POINT_INDEX.ft],
  [FK_POINT_INDEX.bt, FK_POINT_INDEX.bs],
  [FK_POINT_INDEX.bs, FK_POINT_INDEX.bf],
  [FK_POINT_INDEX.bf, FK_POINT_INDEX.toe_b],
  [FK_POINT_INDEX.ft, FK_POINT_INDEX.fs],
  [FK_POINT_INDEX.fs, FK_POINT_INDEX.ff],
  [FK_POINT_INDEX.ff, FK_POINT_INDEX.toe_f],
] as const);
