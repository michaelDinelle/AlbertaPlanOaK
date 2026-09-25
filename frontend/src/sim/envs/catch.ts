import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const CATCH_BALL_X = 0;
export const CATCH_BALL_Y = 1;
export const CATCH_PADDLE = 2;

export const catchDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[CATCH_BALL_X] = frame.words[0]!;
    out[CATCH_BALL_Y] = frame.words[1]!;
    out[CATCH_PADDLE] = frame.words[2]!;
  },
  geometry({ environment }) {
    const w = parameter(environment, 'width'), h = parameter(environment, 'height');
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(prev, next) {
    if (!next.done) return NO_EVENTS;
    return [prev.words[0] === next.words[2] ? { type: 'goal' } : { type: 'harm' }, { type: 'done' }];
  },
};
