import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const QUADROTOR_X = 0;
export const QUADROTOR_Y = 1;
export const QUADROTOR_ANGLE = 2;
export const QUADROTOR_VX = 3;
export const QUADROTOR_VY = 4;
export const QUADROTOR_SPIN = 5;
export const QUADROTOR_GOAL_X = 6;
export const QUADROTOR_GOAL_Y = 7;
export const QUADROTOR_WIND_X = 8;
export const QUADROTOR_WIND_Y = 9;
const QUADROTOR_SLOTS = 10;

export const QUADROTOR_MOVE = 0;
export const QUADROTOR_GUST = 1;

const moved = (prev: EnvFrame, next: EnvFrame): boolean =>
  next.words[QUADROTOR_GOAL_X] !== prev.words[QUADROTOR_GOAL_X] || next.words[QUADROTOR_GOAL_Y] !== prev.words[QUADROTOR_GOAL_Y];

const gusted = (prev: EnvFrame, next: EnvFrame): boolean =>
  next.words.length > QUADROTOR_WIND_Y && (next.words[QUADROTOR_WIND_X] !== prev.words[QUADROTOR_WIND_X] || next.words[QUADROTOR_WIND_Y] !== prev.words[QUADROTOR_WIND_Y]);

export const quadrotorDecoder: EnvDecoder = {
  slots: QUADROTOR_SLOTS,
  wrap: 1 << QUADROTOR_ANGLE,
  decode(frame, out) {
    for (let i = 0; i < QUADROTOR_SLOTS; i++) out[i] = i < frame.words.length ? wordFloat(frame.words[i]!) : 0;
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    const jump = moved(prev, next), gust = gusted(prev, next);
    if (!next.done && !jump && !gust) return NO_EVENTS;
    const events: SimEvent[] = next.done ? [{ type: 'harm' }, { type: 'done' }, { type: 'teleport' }] : [];
    if (jump) events.push({ type: 'cue', value: QUADROTOR_MOVE });
    if (gust) events.push({ type: 'cue', value: QUADROTOR_GUST });
    return events;
  },
};
