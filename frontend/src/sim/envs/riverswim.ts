import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const RIVERSWIM_POS = 0;

export const riverswimDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) { out[RIVERSWIM_POS] = frame.words[0]!; },
  geometry({ environment }) {
    const states = parameter(environment, 'states');
    return { grid: { w: states, h: 1, walls: new Uint8Array(states), goal: { x: states - 1, y: 0 } } };
  },
  deriveEvents(_prev, next) { return next.reward >= 1 ? [{ type: 'goal' }] : NO_EVENTS; },
};
