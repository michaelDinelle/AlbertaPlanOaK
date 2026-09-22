import { NO_EVENTS, type EnvDecoder } from '../types';
import { wordFloat } from '../words';

export const CALIBRATION_FREQUENCY = 0;
export const CALIBRATION_ACTION = 1;

export const calibrationDecoder: EnvDecoder = {
  slots: 2,
  decode(frame, out) { out[CALIBRATION_FREQUENCY] = wordFloat(frame.words[0]!); out[CALIBRATION_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
