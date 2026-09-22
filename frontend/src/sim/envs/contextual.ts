import { philox4x32_10 } from '../philox';
import { NO_EVENTS, type EnvDecoder } from '../types';
import { normalFrom } from './garnet';

export const CONTEXTUAL_ACTION = 0;

export function contextualWeights(seed: number, arms: number, inputs: number): Float32Array {
  const weights = new Float32Array(arms * inputs), scale = 1 / Math.sqrt(inputs);
  for (let i = 0; i < weights.length; i++) {
    const r = philox4x32_10(seed, 0, i, 0, 0, 0);
    weights[i] = scale * normalFrom(r[0], r[1]);
  }
  return weights;
}

export function contextualDrift(seed: number, step: number, drift: number, weights: Float32Array): void {
  for (let i = 0; i < weights.length; i += 2) {
    const d = philox4x32_10(seed, 3, step, 1 + i, 0, 0);
    weights[i]! += drift * normalFrom(d[0], d[1]);
    if (i + 1 < weights.length) weights[i + 1]! += drift * normalFrom(d[2], d[3]);
  }
}

export const contextualDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) { out[CONTEXTUAL_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
