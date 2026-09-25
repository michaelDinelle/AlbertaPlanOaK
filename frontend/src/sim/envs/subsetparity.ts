import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const SUBSET_BITS = 0;
export const SUBSET_CALL = 1;
export const SUBSET_REWARD = 2;

export const subsetMask = (frame: EnvFrame): number => frame.words[0]! >>> 0;
export const subsetBits = (frame: EnvFrame): number => frame.words[1]! >>> 0;
export const subsetCall = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;
export const subsetTruth = (judge: EnvFrame): number => subsetCall(judge) ^ (judge.reward > 0 ? 0 : 1);
export const subsetParityOf = (mask: number, bits: number): number => {
  let x = (mask & bits) >>> 0, p = 0;
  while (x) { p ^= 1; x &= x - 1; }
  return p;
};
export const popcount = (v: number): number => {
  let x = v >>> 0, n = 0;
  while (x) { n++; x &= x - 1; }
  return n;
};

const RIGHT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const WRONG = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const subsetParityDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[SUBSET_BITS] = subsetBits(frame);
    out[SUBSET_CALL] = subsetCall(frame);
    out[SUBSET_REWARD] = frame.reward;
  },
  geometry({ environment }, initial) {
    const inputs = parameter(environment, 'inputs'), subset = parameter(environment, 'subset'), mask = subsetMask(initial), bits = subsetBits(initial);
    let ok = environment.words === 2 && environment.view === 0 && environment.obs === inputs + 1 && environment.actions === 2 && environment.box === 0
      && Number.isInteger(inputs) && inputs > 0 && inputs <= 16 && popcount(mask) === subset && mask >>> inputs === 0 && bits >>> inputs === 0
      && initial.observations[inputs] === 1;
    for (let b = 0; ok && b < inputs; b++) ok = initial.observations[b] === ((bits >>> b) & 1);
    if (!ok) throw new Error(`${environment.envId} does not carry a ${subset}-bit mask over ${inputs} input bits, the bits as observations plus a bias, and a two-way call`);
    return {};
  },
  deriveEvents(_prev, next) {
    if (next.step === 0) return NO_EVENTS;
    return next.reward > 0 ? RIGHT : WRONG;
  },
};
