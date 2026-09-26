import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const OPERANT_LIGHT = 0;
export const OPERANT_PRESS = 1;

export const operantLight = (frame: EnvFrame): number => (frame.words[0] !== 0 ? 1 : 0);
export const operantPress = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;

export const operantDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  sequential: true,
  decode(frame, out) {
    out[OPERANT_LIGHT] = operantLight(frame);
    out[OPERANT_PRESS] = operantPress(frame);
  },
  geometry({ environment }, initial) {
    if (initial.words.length !== 1 || environment.actions !== 2) throw new Error(`${environment.envId} does not carry one light word and a two-way lever`);
    return {};
  },
  deriveEvents(_prev, next) {
    const light = operantLight(next) === 1;
    if (!light && next.reward === 0) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (light) events.push({ type: 'cue', value: 1 });
    if (next.reward > 0) events.push({ type: 'goal' });
    else if (next.reward < 0) events.push({ type: 'harm' });
    return events;
  },
};
