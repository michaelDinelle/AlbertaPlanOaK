import { NO_EVENTS, type EnvDecoder, type EnvGeometry } from '../types';
import { wordFloat } from '../words';

export const TRACK_GAIN = 3;
export const TRACK_OBS_NOISE = 0.5;

export const TRACK_TARGET = 0;
export const TRACK_ESTIMATE = 1;
export const TRACK_OBSERVED = 2;

const GEOMETRY: EnvGeometry = Object.freeze({ classic: Object.freeze({ anchors: Object.freeze({ origin: [0, 0, 0] as const }) }) });

export const trackingDecoder: EnvDecoder = {
  slots: 3,
  decode(frame, out) {
    out[TRACK_TARGET] = wordFloat(frame.words[0]!);
    out[TRACK_ESTIMATE] = TRACK_GAIN * frame.actions[0]!;
    out[TRACK_OBSERVED] = frame.observations[0]!;
  },
  geometry(): EnvGeometry { return GEOMETRY; },
  deriveEvents: () => NO_EVENTS,
};
