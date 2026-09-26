import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const CHAIN_POS = 0;

export const walkDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) { out[CHAIN_POS] = frame.words[0]!; },
  geometry({ environment }) {
    const length = parameter(environment, 'limit') + 1;
    return { grid: { w: length, h: 1, walls: new Uint8Array(length), goal: { x: length - 1, y: 0 } } };
  },
  deriveEvents(_prev, next) {
    return next.done ? [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }] : NO_EVENTS;
  },
};
