import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const FROST_POSITION = 0;
export const FROST_HEAT = 1;
export const FROST_PHASE = 2;
export const FROST_CALM = 3;
export const FROST_CLOCK = 4;
export const FROST_TOKEN = 5;
export const FROST_PREDICTION = 6;

export const frostHeat = (frame: EnvFrame): number => wordFloat(frame.words[FROST_HEAT]!);
export const frostPrediction = (frame: EnvFrame): number => wordFloat(frame.words[FROST_PREDICTION]!);
export const frostHazard = (frame: EnvFrame): boolean => frame.words[FROST_PHASE]! >= frame.words[FROST_CALM]!;
export const frostToken = (frame: EnvFrame): boolean => frame.words[FROST_TOKEN] === 1;

export const frostDecoder: EnvDecoder = {
  slots: 7,
  discrete: true,
  decode(frame, out) {
    out[FROST_POSITION] = frame.words[FROST_POSITION]!;
    out[FROST_HEAT] = frostHeat(frame);
    out[FROST_PHASE] = frame.words[FROST_PHASE]!;
    out[FROST_CALM] = frame.words[FROST_CALM]!;
    out[FROST_CLOCK] = frame.words[FROST_CLOCK]!;
    out[FROST_TOKEN] = frame.words[FROST_TOKEN]!;
    out[FROST_PREDICTION] = frostPrediction(frame);
  },
  geometry({ environment }, initial) {
    if (initial.words.length !== 7) throw new Error('FrostHollow state does not carry its position, heat, phase, calm, clock, token and prediction words');
    const w = constant(environment, 'FROST_CELLS');
    return { grid: { w, h: 1, walls: new Uint8Array(w), goal: { x: constant(environment, 'FROST_CENTER'), y: 0 } } };
  },
  deriveEvents(prev, next) {
    const events: SimEvent[] = [];
    if (next.reward > 0) events.push({ type: 'goal' });
    if (!next.done && frostHazard(next) && frostHeat(prev) > 0 && frostHeat(next) === 0) events.push({ type: 'harm' });
    if (frostToken(next) && !frostToken(prev)) events.push({ type: 'cue', value: 1 });
    if (next.done) events.push({ type: 'teleport' }, { type: 'done' });
    return events.length ? events : NO_EVENTS;
  },
};
