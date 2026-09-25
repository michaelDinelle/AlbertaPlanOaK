import { wordFloat } from '../words';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const TRACKING_TRUTH = 0;
export const TRACKING_CALL = 1;
export const TRACKING_READING = 2;
export const TRACKING_REACH = 3;
export const TRACKING_BULLSEYE = 0.01;

export const trackingTruth = (frame: EnvFrame): number => wordFloat(frame.words[0]!);
export const trackingCall = (frame: EnvFrame): number => TRACKING_REACH * frame.actions[0]!;
export const trackingReading = (frame: EnvFrame): number => frame.observations[0]!;

const BULLSEYE = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];

export const trackingDecoder: EnvDecoder = {
  slots: 3,
  decode(frame, out) {
    out[TRACKING_TRUTH] = trackingTruth(frame);
    out[TRACKING_CALL] = trackingCall(frame);
    out[TRACKING_READING] = trackingReading(frame);
  },
  geometry({ environment }) {
    if (environment.words !== 1 || environment.view !== 0 || environment.obs !== 2 || environment.actions !== 1 || environment.box !== 1) {
      throw new Error(`${environment.envId} does not carry one signal word, a reading with a bias and one continuous prediction`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    return next.step > 0 && -next.reward < TRACKING_BULLSEYE ? BULLSEYE : NO_EVENTS;
  },
};
