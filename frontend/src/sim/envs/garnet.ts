import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export interface GarnetShape {
  states: number;
  actions: number;
  branching: number;
}

const shapes = new Map<number, GarnetShape>();

export const garnetState = (frame: EnvFrame): number => frame.words[0]!;
export const garnetAction = (frame: EnvFrame, shape: GarnetShape): number => (Math.trunc(frame.actions[0]!) >>> 0) % shape.actions;
export const garnetEpoch = (frame: EnvFrame): number => frame.view[0]!;
export const garnetNext = (frame: EnvFrame, shape: GarnetShape, pair: number, b: number): number => frame.view[1 + pair * shape.branching + b]!;
export const garnetProb = (frame: EnvFrame, shape: GarnetShape, pair: number, b: number): number =>
  wordFloat(frame.view[1 + shape.states * shape.actions * shape.branching + pair * shape.branching + b]!);
export const garnetReward = (frame: EnvFrame, shape: GarnetShape, pair: number): number => wordFloat(frame.view[1 + 2 * shape.states * shape.actions * shape.branching + pair]!);

const bestCache = { key: '', best: 0 };

export function garnetBest(frame: EnvFrame, shape: GarnetShape): number {
  const pairs = shape.states * shape.actions, start = 1 + 2 * pairs * shape.branching;
  const key = `${frame.view.length}:${frame.view[0]}:${frame.view[start]}:${frame.view[start + pairs - 1]}`;
  if (key === bestCache.key) return bestCache.best;
  let best = 0;
  for (let p = 1; p < pairs; p++) if (garnetReward(frame, shape, p) > garnetReward(frame, shape, best)) best = p;
  bestCache.key = key;
  bestCache.best = best;
  return best;
}

export const garnetDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[0] = garnetState(frame);
    out[1] = frame.actions[0]!;
  },
  geometry({ environment }, initial) {
    const states = Number(environment.parameters.states), actions = Number(environment.parameters.actions), branching = Number(environment.parameters.branching);
    const count = 1 + states * actions * (2 * branching + 1);
    if (initial.words.length !== 1 || environment.actions !== actions || initial.view.length !== count) throw new Error(`${environment.envId} is not a Garnet(${states}, ${actions}, ${branching}) tape with a ${count}-word view`);
    shapes.set(count, { states, actions, branching });
    return {};
  },
  deriveEvents(prev, next) {
    if (!next.view.length || prev.view.length !== next.view.length) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (garnetEpoch(next) !== garnetEpoch(prev)) events.push({ type: 'cue', value: garnetEpoch(next) });
    const shape = shapes.get(next.view.length);
    if (shape && garnetState(prev) * shape.actions + garnetAction(next, shape) === garnetBest(next, shape)) events.push({ type: 'goal' });
    return events.length ? events : NO_EVENTS;
  },
};

export const garnetShape = (viewWords: number): GarnetShape | undefined => shapes.get(viewWords);
