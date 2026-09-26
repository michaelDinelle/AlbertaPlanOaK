import type * as THREE from 'three/webgpu';
import { CHEETAH_ROOT_ANGLE, CHEETAH_ROOT_X, CHEETAH_ROOT_Z } from '@sim/envs/planar';
import { constant, constantList, type EnvDescriptor } from '../../data/envs';

export interface Capsule {
  body: number;
  x: number;
  z: number;
  ax: number;
  az: number;
  half: number;
}

export interface CheetahModel {
  parent: number[];
  bodyX: number[];
  bodyZ: number[];
  capsules: Capsule[];
  radius: number;
}

export function cheetahModel(env: EnvDescriptor): CheetahModel {
  const list = (key: string) => constantList(env, `PLANAR_${key}`);
  const [body, x, z, w, y, half] = ['CAPSULE_BODY', 'CAPSULE_X', 'CAPSULE_Z', 'CAPSULE_W', 'CAPSULE_Y', 'CAPSULE_HALF'].map(list) as [number[], number[], number[], number[], number[], number[]];
  return {
    parent: list('PARENT'),
    bodyX: list('BODY_X'),
    bodyZ: list('BODY_Z'),
    radius: constant(env, 'PLANAR_RADIUS'),
    capsules: body.map((b, i) => ({ body: b, x: x[i]!, z: z[i]!, ax: 2 * w[i]! * y[i]!, az: w[i]! ** 2 - y[i]! ** 2, half: half[i]! })),
  };
}

export function cheetahFrames(model: CheetahModel, pose: ArrayLike<number>, out: Float32Array): Float32Array {
  for (let k = 0; k < model.parent.length; k++) {
    const p = model.parent[k]!;
    if (p < 0) {
      out[0] = pose[CHEETAH_ROOT_X]! + model.bodyX[0]!;
      out[1] = pose[CHEETAH_ROOT_Z]! + model.bodyZ[0]!;
      out[2] = pose[CHEETAH_ROOT_ANGLE]!;
      continue;
    }
    const phi = out[p * 3 + 2]!, c = Math.cos(phi), s = Math.sin(phi);
    out[k * 3] = out[p * 3]! + c * model.bodyX[k]! + s * model.bodyZ[k]!;
    out[k * 3 + 1] = out[p * 3 + 1]! - s * model.bodyX[k]! + c * model.bodyZ[k]!;
    out[k * 3 + 2] = phi + pose[CHEETAH_ROOT_ANGLE + k]!;
  }
  return out;
}

export function bodyPoint(frames: Float32Array, body: number, x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
  const phi = frames[body * 3 + 2]!, c = Math.cos(phi), s = Math.sin(phi);
  return out.set(frames[body * 3]! + c * x + s * z, frames[body * 3 + 1]! - s * x + c * z, 0);
}

export function capsuleEnd(capsule: Capsule, end: number): [number, number] {
  const sign = end === 0 ? 1 : -1;
  return [capsule.x + sign * capsule.half * capsule.ax, capsule.z + sign * capsule.half * capsule.az];
}

export function farEnd(capsule: Capsule): [number, number] {
  const [ax, az] = capsuleEnd(capsule, 0), [bx, bz] = capsuleEnd(capsule, 1);
  return ax * ax + az * az > bx * bx + bz * bz ? [ax, az] : [bx, bz];
}
