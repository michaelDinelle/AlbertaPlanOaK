import { constant, parameter } from '../../data/envs';
import { philox4x32_10 } from '../philox';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const GARNET_STATE = 0;
export const GARNET_ACTION = 1;

export const u01 = (x: number): number => (x >>> 8) * 2 ** -24;
export const u01open = (x: number): number => ((x >>> 8) + 1) * 2 ** -24;
export const normalFrom = (w0: number, w1: number): number => Math.sqrt(-2 * Math.log(u01open(w0))) * Math.cos(2 * Math.PI * u01(w1));

export interface GarnetModel {
  states: number;
  actions: number;
  branching: number;
  next: Uint32Array;
  prob: Float32Array;
  reward: Float32Array;
  greedy: Uint8Array;
}

export function garnetModel(seed: number, states: number, actions: number, branching: number, discount: number, improvements: number, margin: number, epoch: number): GarnetModel {
  const pairs = states * actions, edges = pairs * branching;
  const next = new Uint32Array(edges), prob = new Float32Array(edges), reward = new Float32Array(pairs), greedy = new Uint8Array(states);
  for (let pair = 0; pair < pairs; pair++) {
    const r = philox4x32_10(seed, 0, epoch, pair, 0, 0);
    let total = 0;
    for (let b = 0; b < branching; b++) {
      const e = philox4x32_10(seed, 0, epoch, pair, 1 + b, 0);
      next[pair * branching + b] = e[0] % states;
      prob[pair * branching + b] = u01open(e[1]);
      total += prob[pair * branching + b]!;
    }
    for (let b = 0; b < branching; b++) prob[pair * branching + b]! /= total;
    reward[pair] = normalFrom(r[1], r[2]);
  }
  const q = (value: Float64Array, s: number, a: number): number => {
    const pair = s * actions + a;
    let sum = reward[pair]!;
    for (let b = 0; b < branching; b++) sum += discount * prob[pair * branching + b]! * value[next[pair * branching + b]!]!;
    return sum;
  };
  for (let s = 0; s < states; s++) {
    let best = 0;
    for (let a = 1; a < actions; a++) if (reward[s * actions + a]! > reward[s * actions + best]!) best = a;
    greedy[s] = best;
  }
  const value = new Float64Array(states);
  for (let improvement = 0; improvement < improvements; improvement++) {
    garnetEvaluate(states, actions, branching, discount, next, prob, reward, greedy, value);
    let changed = false;
    for (let s = 0; s < states; s++) {
      let best = greedy[s]!, bestQ = q(value, s, best);
      for (let a = 0; a < actions; a++) { const candidate = q(value, s, a); if (candidate > bestQ + margin) { bestQ = candidate; best = a; } }
      if (best !== greedy[s]) { greedy[s] = best; changed = true; }
    }
    if (!changed) break;
  }
  return { states, actions, branching, next, prob, reward, greedy };
}

function garnetEvaluate(states: number, actions: number, branching: number, discount: number, next: Uint32Array, prob: Float32Array, reward: Float32Array, greedy: Uint8Array, value: Float64Array): void {
  const n = states, a = new Float64Array(n * n), b = new Float64Array(n);
  for (let s = 0; s < n; s++) {
    const pair = s * actions + greedy[s]!;
    a[s * n + s] = a[s * n + s]! + 1;
    for (let br = 0; br < branching; br++) { const at = s * n + next[pair * branching + br]!; a[at] = a[at]! - discount * prob[pair * branching + br]!; }
    b[s] = reward[pair]!;
  }
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(a[row * n + col]!) > Math.abs(a[pivot * n + col]!)) pivot = row;
    if (pivot !== col) {
      for (let c = 0; c < n; c++) { const t = a[col * n + c]!; a[col * n + c] = a[pivot * n + c]!; a[pivot * n + c] = t; }
      const t = b[col]!; b[col] = b[pivot]!; b[pivot] = t;
    }
    for (let row = col + 1; row < n; row++) {
      const factor = a[row * n + col]! / a[col * n + col]!;
      if (factor === 0) continue;
      for (let c = col; c < n; c++) a[row * n + c] = a[row * n + c]! - factor * a[col * n + c]!;
      b[row] = b[row]! - factor * b[col]!;
    }
  }
  for (let row = n - 1; row >= 0; row--) {
    let sum = b[row]!;
    for (let c = row + 1; c < n; c++) sum -= a[row * n + c]! * value[c]!;
    value[row] = sum / a[row * n + row]!;
  }
}

export function garnetModelOf(env: Parameters<typeof parameter>[0], seed: number, step: number): GarnetModel {
  const epoch = parameter(env, 'resample') ? Math.floor(step / constant(env, 'GARNET_PERIOD')) : 0;
  return garnetModel(seed, parameter(env, 'states'), parameter(env, 'actions'), parameter(env, 'branching'), constant(env, 'GARNET_DISCOUNT'), constant(env, 'GARNET_IMPROVEMENTS'), constant(env, 'GARNET_IMPROVEMENT_MARGIN'), epoch);
}

export const garnetDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[GARNET_STATE] = frame.words[0]!; out[GARNET_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
