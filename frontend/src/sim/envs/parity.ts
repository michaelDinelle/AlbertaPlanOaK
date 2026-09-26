import { parameter } from '../../data/envs';
import type { EnvDecoder, EnvFrame, SimEvent } from '../types';

export const PARITY_BIT = 0;
export const PARITY_CALL = 1;
export const PARITY_TRUTH = 2;
export const PARITY_HISTORY = 96;

export const parityBit = (frame: EnvFrame, age: number): number => (frame.words[age >>> 5]! >>> (age & 31)) & 1;
export const parityCall = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;
export const parityTruth = (frame: EnvFrame): number => parityCall(frame) ^ (frame.reward > 0 ? 0 : 1);

const RIGHT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const WRONG = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const parityDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[PARITY_BIT] = parityBit(frame, 0);
    out[PARITY_CALL] = parityCall(frame);
    out[PARITY_TRUTH] = parityTruth(frame);
  },
  geometry({ environment }, initial) {
    const delay = parameter(environment, 'delay');
    if (environment.words * 32 !== PARITY_HISTORY || environment.view !== 0 || environment.obs !== 2 || environment.actions !== 2 || environment.box !== 0
      || !Number.isInteger(delay) || delay < 0 || delay + 3 >= PARITY_HISTORY || initial.observations[0] !== parityBit(initial, 0) || initial.observations[1] !== 1) {
      throw new Error(`${environment.envId} does not carry a ${PARITY_HISTORY}-bit history reaching past its judged pair at delay ${delay}, its newest bit plus a bias, and a two-way call`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 ? RIGHT : WRONG;
  },
};
