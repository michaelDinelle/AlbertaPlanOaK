import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const KEYDOOR_PHASE = 0;
export const KEYDOOR_HELD = 1;
export const KEYDOOR_TAKE = 2;

export const keyDoorTake = (frame: EnvFrame): number => (frame.actions[0]! >>> 0) & 1;

const OPENED: SimEvent[] = [{ type: 'door', index: 0, open: true }, { type: 'goal' }, { type: 'done' }];
const LOCKED: SimEvent[] = [{ type: 'door', index: 0, open: false }, { type: 'harm' }, { type: 'done' }];
const TAKEN: SimEvent[] = [{ type: 'key', index: 0 }];

export const keyDoorDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[KEYDOOR_PHASE] = frame.words[0]!;
    out[KEYDOOR_HELD] = frame.words[1]!;
    out[KEYDOOR_TAKE] = keyDoorTake(frame);
  },
  geometry({ environment }, initial) {
    parameter(environment, 'delay');
    if (initial.words.length !== 2 || environment.actions !== 2 || environment.box !== 0) throw new Error(`${environment.envId} is not a phase and key word over two actions`);
    return {};
  },
  deriveEvents(prev, next) {
    if (next.done) return next.reward > 0 ? OPENED : LOCKED;
    return prev.words[1] === 0 && next.words[1] === 1 ? TAKEN : NO_EVENTS;
  },
};
