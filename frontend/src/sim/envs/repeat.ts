import { constant, parameter } from '../../data/envs';
import type { EnvDecoder, EnvFrame, SimEvent } from '../types';

export const REPEAT_SHOWN = 0;
export const REPEAT_CALL = 1;
export const REPEAT_HIT = 2;

const SYMBOL_BITS = 2;
const PER_WORD = 32 / SYMBOL_BITS;

export const repeatSymbol = (frame: EnvFrame, age: number): number => (frame.words[Math.floor(age / PER_WORD)]! >>> (SYMBOL_BITS * (age % PER_WORD))) & ((1 << SYMBOL_BITS) - 1);

export const repeatCall = (frame: EnvFrame): number => {
  const action = frame.actions[0]!;
  return (action > 0 ? Math.min(Math.trunc(action), 0xffffffff) : 0) % (1 << SYMBOL_BITS);
};

export const repeatHistory = (words: number): number => words * PER_WORD;

const RIGHT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const WRONG = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const repeatDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[REPEAT_SHOWN] = repeatSymbol(frame, 0);
    out[REPEAT_CALL] = repeatCall(frame);
    out[REPEAT_HIT] = frame.reward > 0 ? 1 : 0;
  },
  geometry({ environment }, initial) {
    const lag = parameter(environment, 'lag'), symbols = constant(environment, 'REPEAT_SYMBOLS'), shown = repeatSymbol(initial, 0);
    const onehot = Array.from({ length: symbols }, (_, i) => initial.observations[i]);
    if (symbols !== 1 << SYMBOL_BITS || environment.view !== 0 || environment.obs !== symbols + 1 || environment.actions !== symbols || environment.box !== 0
      || !Number.isInteger(lag) || lag < 1 || lag + 1 >= repeatHistory(environment.words) || onehot.some((v, i) => v !== (i === shown ? 1 : 0)) || initial.observations[symbols] !== 1) {
      throw new Error(`${environment.envId} does not carry a ${SYMBOL_BITS}-bit symbol history reaching past its lag of ${lag}, the newest symbol one-hot plus a bias, and a ${symbols}-way call`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 ? RIGHT : WRONG;
  },
};
