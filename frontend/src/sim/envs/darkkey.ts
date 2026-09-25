import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const DARKKEY_X = 0;
export const DARKKEY_Y = 1;
export const DARKKEY_FOUND = 2;
export const DARKKEY_KEY_X = 3;
export const DARKKEY_KEY_Y = 4;
export const DARKKEY_DOOR_X = 5;
export const DARKKEY_DOOR_Y = 6;

export const darkKeyFound = (prev: { words: Uint32Array }, next: { words: Uint32Array; reward: number; done: number }): number => {
  const before = prev.words[3]!, after = next.done ? before + next.reward : next.words[3]!;
  if (after - before !== next.reward || after > 2) throw new Error(`DarkKeyToDoor went from found ${before} to ${after} for reward ${next.reward}`);
  return after;
};

export const darkKeyToDoorDecoder: EnvDecoder = {
  slots: 7,
  discrete: true,
  decode(frame, out) {
    const w = frame.words, task = w[2]!;
    out[DARKKEY_X] = w[0]!;
    out[DARKKEY_Y] = w[1]!;
    out[DARKKEY_FOUND] = w[3]!;
    out[DARKKEY_KEY_X] = task & 15;
    out[DARKKEY_KEY_Y] = (task >>> 4) & 15;
    out[DARKKEY_DOOR_X] = (task >>> 8) & 15;
    out[DARKKEY_DOOR_Y] = (task >>> 12) & 15;
  },
  geometry({ environment }, initial) {
    const side = parameter(environment, 'side');
    if (initial.words.length !== 4) throw new Error(`${environment.envId} state does not carry its position, task and found words`);
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(prev, next) {
    const before = prev.words[3]!, after = darkKeyFound(prev, next);
    if (after === before && !next.done) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (before < 1 && after >= 1) events.push({ type: 'key', index: 0 });
    if (before < 2 && after >= 2) events.push({ type: 'door', index: 0, open: true });
    if (next.done) events.push({ type: 'teleport' }, { type: 'done' });
    return events;
  },
};
