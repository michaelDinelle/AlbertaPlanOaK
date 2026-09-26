import { HOPPER_ROOT_ANGLE, HOPPER_ROOT_X, HOPPER_ROOT_Z, HOPPER_THIGH } from '@sim/envs/planar';
import type { EnvFrame } from '@sim/types';
import { constantRecord, type EnvDescriptor } from '../../data/envs';

export interface Capsule {
  ax: number;
  az: number;
  bx: number;
  bz: number;
  radius: number;
}

export interface HopperModel {
  links: readonly (readonly [number, number])[];
  capsules: readonly Capsule[];
  low: readonly number[];
  high: readonly number[];
  margin: number;
  height: number;
  angle: number;
  alive: number;
  control: number;
  step: number;
  rest: number;
}

export const BODIES = 4;
export const JOINTS = 3;

export function hopperModel(env: EnvDescriptor): HopperModel {
  const leg = constantRecord(env, 'PLANAR_LEG');
  const list = (key: string): number[] => {
    const value = leg[key];
    if (!Array.isArray(value)) throw new Error(`${env.envId} PLANAR_LEG lacks ${key}`);
    return value;
  };
  const scalar = (key: string): number => {
    const value = leg[key];
    if (typeof value !== 'number') throw new Error(`${env.envId} PLANAR_LEG lacks ${key}`);
    return value;
  };
  const pairs = (key: string) => {
    const flat = list(key);
    return Array.from({ length: flat.length / 2 }, (_, i) => [flat[2 * i]!, flat[2 * i + 1]!] as const);
  };
  const top = pairs('top'), bottom = pairs('bottom'), radius = list('radius'), links = pairs('joint');
  if (radius.length !== BODIES || links.length !== JOINTS) throw new Error(`${env.envId} PLANAR_LEG is not a ${BODIES}-body chain`);
  return {
    links,
    capsules: radius.map((r, g) => ({ ax: top[g]![0], az: top[g]![1], bx: bottom[g]![0], bz: bottom[g]![1], radius: r })),
    low: list('low'),
    high: list('high'),
    margin: scalar('margin'),
    height: scalar('height'),
    angle: scalar('angle'),
    alive: scalar('alive'),
    control: scalar('control'),
    step: scalar('step'),
    rest: scalar('rest'),
  };
}

export function solve(model: HopperModel, q: ArrayLike<number>, out: Float64Array): Float64Array {
  out[0] = q[HOPPER_ROOT_X]!;
  out[1] = q[HOPPER_ROOT_Z]!;
  out[2] = q[HOPPER_ROOT_ANGLE]!;
  for (let i = 0; i < JOINTS; i++) {
    const [u, w] = model.links[i]!, a = out[3 * i + 2]!, c = Math.cos(a), s = Math.sin(a);
    out[3 * i + 3] = out[3 * i]! + u * c + w * s;
    out[3 * i + 4] = out[3 * i + 1]! - u * s + w * c;
    out[3 * i + 5] = a - q[HOPPER_THIGH + i]!;
  }
  return out;
}

export interface Point { x: number; y: number }

export function locate(bodies: Float64Array, body: number, u: number, w: number, out: Point): Point {
  const a = bodies[3 * body + 2]!, c = Math.cos(a), s = Math.sin(a);
  out.x = bodies[3 * body]! + u * c + w * s;
  out.y = bodies[3 * body + 1]! - u * s + w * c;
  return out;
}

export function effort(frame: EnvFrame): number {
  let sum = 0;
  for (let u = 0; u < frame.actions.length; u++) sum += frame.actions[u]! * frame.actions[u]!;
  return sum;
}

export function forward(model: HopperModel, frame: EnvFrame): number {
  return frame.reward - (frame.done ? 0 : model.alive) + model.control * effort(frame);
}
