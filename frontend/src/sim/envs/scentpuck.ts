import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const SP_HALF = 1;
export const SP_STEP = 0.05;
export const SP_CATCH = 0.05;
export const SP_SCENT_FALLOFF = 4;
export const SP_DECOY_FALLOFF = 16;
export const SP_ACTIONS = 5;

export const SCENT_X = 0;
export const SCENT_Y = 1;
export const SCENT_GOAL_X = 2;
export const SCENT_GOAL_Y = 3;
export const SCENT_DECOY_X = 4;
export const SCENT_DECOY_Y = 5;
export const SCENT_ACTION = 6;

export const scentAt = (dx: number, dy: number): number => Math.exp(-SP_SCENT_FALLOFF * (dx * dx + dy * dy));

const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  boundsMin: [-SP_HALF, 0, -SP_HALF],
  boundsMax: [SP_HALF, 0, SP_HALF],
  start: [0, 0, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({ classic: Object.freeze({ anchors: ANCHORS }) });

export const scentpuckDecoder: EnvDecoder = {
  slots: 7,
  decode(frame, out) {
    for (let i = 0; i < 6; i++) out[i] = wordFloat(frame.words[i]!);
    out[SCENT_ACTION] = frame.actions[0]!;
  },
  geometry(): EnvGeometry { return GEOMETRY; },
  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    return next.done === 0 ? NO_EVENTS : [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
