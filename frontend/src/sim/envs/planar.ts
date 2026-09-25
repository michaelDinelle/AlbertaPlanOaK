import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat, wordsDouble } from '../words';

export const CHEETAH_ROOT_X = 0;
export const CHEETAH_ROOT_Z = 1;
export const CHEETAH_ROOT_ANGLE = 2;
export const CHEETAH_BACK_THIGH = 3;
export const CHEETAH_FRONT_THIGH = 6;

export const REACHER_SHOULDER = 0;
export const REACHER_ELBOW = 1;
export const REACHER_TARGET_X = 2;
export const REACHER_TARGET_Y = 3;

export const PENDULUM_CART = 0;
export const PENDULUM_HINGE = 1;
export const PENDULUM_HINGE2 = 2;

export const HOPPER_ROOT_X = 0;
export const HOPPER_ROOT_Z = 1;
export const HOPPER_ROOT_ANGLE = 2;
export const HOPPER_THIGH = 3;
export const HOPPER_LEG = 4;
export const HOPPER_FOOT = 5;

const restart = (_prev: EnvFrame, next: EnvFrame): SimEvent[] => next.done ? [{ type: 'teleport' }, { type: 'done' }] : NO_EVENTS;

export const halfCheetahDecoder: EnvDecoder = {
  slots: 9,
  wrap: 1 << CHEETAH_ROOT_ANGLE,
  decode(frame, out) {
    out[CHEETAH_ROOT_X] = wordsDouble(frame.words, 0);
    for (let i = 1; i < 9; i++) out[i] = wordFloat(frame.words[i + 1]!);
  },
  geometry: () => ({}),
  deriveEvents: restart,
};

export const reacherDecoder: EnvDecoder = {
  slots: 4,
  wrap: 1 << REACHER_SHOULDER | 1 << REACHER_ELBOW,
  decode(frame, out) {
    out[REACHER_SHOULDER] = wordsDouble(frame.words, 0);
    out[REACHER_ELBOW] = wordsDouble(frame.words, 2);
    out[REACHER_TARGET_X] = wordFloat(frame.words[24]!);
    out[REACHER_TARGET_Y] = wordFloat(frame.words[25]!);
  },
  geometry: () => ({}),
  deriveEvents: restart,
};

export const invertedDoublePendulumDecoder: EnvDecoder = {
  slots: 3,
  wrap: 1 << PENDULUM_HINGE | 1 << PENDULUM_HINGE2,
  decode(frame, out) {
    for (let i = 0; i < 3; i++) out[i] = wordsDouble(frame.words, 2 * i);
  },
  geometry: () => ({}),
  deriveEvents: restart,
};

export const hopperDecoder: EnvDecoder = {
  slots: 6,
  wrap: 1 << HOPPER_ROOT_ANGLE,
  decode(frame, out) {
    for (let i = 0; i < 6; i++) out[i] = wordsDouble(frame.words, 2 * i);
  },
  geometry: () => ({}),
  deriveEvents: restart,
};
