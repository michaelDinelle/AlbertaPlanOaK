import { constant, parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const RECITE_PHASE = 0;
export const RECITE_CALL = 1;
export const RECITE_HIT = 2;

const SYMBOL_BITS = 2;
const SYMBOLS = 1 << SYMBOL_BITS;
const RECALL = SYMBOLS;

export const recitePhase = (frame: EnvFrame): number => frame.words[1]!;

export const reciteSymbol = (frame: EnvFrame, position: number): number => (frame.view[0]! >>> (SYMBOL_BITS * position)) & (SYMBOLS - 1);

export const reciteCall = (frame: EnvFrame): number => {
  const action = frame.actions[0]!;
  return (action > 0 ? Math.min(Math.trunc(action), 0xffffffff) : 0) % SYMBOLS;
};

export const reciteRecalling = (frame: EnvFrame): boolean => frame.observations[RECALL] === 1;

const freeze = (events: SimEvent[]): SimEvent[] => Object.freeze(events) as unknown as SimEvent[];
const RIGHT = freeze([{ type: 'goal' }]);
const WRONG = freeze([{ type: 'harm' }]);
const RIGHT_DONE = freeze([{ type: 'goal' }, { type: 'done' }]);
const WRONG_DONE = freeze([{ type: 'harm' }, { type: 'done' }]);

export const reciteDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[RECITE_PHASE] = recitePhase(frame);
    out[RECITE_CALL] = reciteCall(frame);
    out[RECITE_HIT] = frame.reward > 0 ? 1 : 0;
  },
  geometry({ environment }, initial) {
    const length = parameter(environment, 'length'), symbols = constant(environment, 'RECITE_SYMBOLS'), phase = recitePhase(initial);
    const watching = phase < length, shown = watching ? reciteSymbol(initial, phase) : -1, o = initial.observations;
    const onehot = Array.from({ length: symbols }, (_, i) => o[i]);
    if (symbols !== SYMBOLS || environment.view !== 1 || environment.words !== 2 || environment.obs !== symbols + 3 || environment.actions !== symbols || environment.box !== 0
      || !Number.isInteger(length) || length < 4 || length % 4 !== 0 || SYMBOL_BITS * length > 32 || phase >= 2 * length
      || onehot.some((v, i) => v !== (i === shown ? 1 : 0)) || o[RECALL] !== (watching ? 0 : 1) || Math.abs(o[RECALL + 1]! - (phase % length) / length) > 1e-6 || o[RECALL + 2] !== 1) {
      throw new Error(`${environment.envId} does not carry a sequence and phase word, one view word of ${length} ${SYMBOL_BITS}-bit symbols, the shown symbol one-hot with a recall flag, position and bias, and a ${symbols}-way call`);
    }
    return {};
  },
  deriveEvents(prev, next) {
    if (!reciteRecalling(prev)) return NO_EVENTS;
    const hit = next.reward > 0;
    if (next.done > 0) return hit ? RIGHT_DONE : WRONG_DONE;
    return hit ? RIGHT : WRONG;
  },
};
