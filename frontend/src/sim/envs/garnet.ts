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

export function garnetModel(seed: number, states: number, actions: number, branching: number, sweeps: number, epoch: number): GarnetModel {
  const pairs = states * actions, edges = pairs * branching;
  const next = new Uint32Array(edges), prob = new Float32Array(edges), reward = new Float32Array(pairs), value = new Float32Array(states), greedy = new Uint8Array(states);
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
    reward[pair] = r[0] % 10 === 0 ? normalFrom(r[1], r[2]) : 0;
  }
  for (let sweep = 0; sweep < sweeps; sweep++) {
    const reference = value[0]!;
    for (let s = 0; s < states; s++) {
      let best = -Infinity, bestAction = 0;
      for (let a = 0; a < actions; a++) {
        const pair = s * actions + a;
        let q = reward[pair]!;
        for (let b = 0; b < branching; b++) q += prob[pair * branching + b]! * value[next[pair * branching + b]!]!;
        if (q > best) { best = q; bestAction = a; }
      }
      value[s] = best - reference;
      greedy[s] = bestAction;
    }
  }
  return { states, actions, branching, next, prob, reward, greedy };
}

export function garnetModelOf(env: Parameters<typeof parameter>[0], seed: number, step: number): GarnetModel {
  const epoch = parameter(env, 'resample') ? Math.floor(step / constant(env, 'GARNET_PERIOD')) : 0;
  return garnetModel(seed, parameter(env, 'states'), parameter(env, 'actions'), parameter(env, 'branching'), constant(env, 'GARNET_SWEEPS'), epoch);
}

export const garnetDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[GARNET_STATE] = frame.words[0]!; out[GARNET_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
