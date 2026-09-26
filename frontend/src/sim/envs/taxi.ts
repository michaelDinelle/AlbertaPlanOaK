import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';
import { constant } from '../../data/envs';

export const TAXI_X = 0;
export const TAXI_Y = 1;
export const TAXI_PASSENGER = 2;
export const TAXI_DESTINATION = 3;

export const taxiDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) {
    out[TAXI_X] = frame.words[0]!;
    out[TAXI_Y] = frame.words[1]!;
    out[TAXI_PASSENGER] = frame.words[2]!;
    out[TAXI_DESTINATION] = frame.words[3]!;
  },
  geometry({ environment }) {
    const side = constant(environment, 'TAXI_SIDE');
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(prev, next) {
    if (next.done) return [{ type: 'goal' }, { type: 'done' }];
    const events: SimEvent[] = [];
    if (next.words[2] !== prev.words[2]) events.push({ type: 'key', index: prev.words[2]! });
    if (next.reward < -1) events.push({ type: 'harm' });
    return events.length ? events : NO_EVENTS;
  },
};
