import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const CP_TRACK_HALF = 2.4;
export const CP_ACTIONS = 3;
export const CP_SPEED_SCALE = 5;

export const CARTPOLE_X = 0;
export const CARTPOLE_XDOT = 1;
export const CARTPOLE_THETA = 2;
export const CARTPOLE_THETADOT = 3;
export const CARTPOLE_ACTION = 4;

export function cartpoleForce(action: number): number {
  return ((Math.trunc(action) >>> 0) % CP_ACTIONS) - 1;
}

const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  railLeft: [-CP_TRACK_HALF, 0, 0],
  railRight: [CP_TRACK_HALF, 0, 0],
  start: [0, 0, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({ classic: Object.freeze({ anchors: ANCHORS }) });

export const cartpoleDecoder: EnvDecoder = {
  slots: 5,
  wrap: 0b100,
  decode(frame, out) {
    for (let i = 0; i < 4; i++) out[i] = wordFloat(frame.words[i]!);
    out[CARTPOLE_ACTION] = frame.actions[0]!;
  },
  geometry(): EnvGeometry { return GEOMETRY; },
  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    return next.done === 0 ? NO_EVENTS : [{ type: 'done' }, { type: 'teleport' }];
  },
};
