import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type GridCell, type SimEvent } from '../types';

export const CORRIDOR_X = 0;
export const CORRIDOR_Y = 1;
export const CORRIDOR_GX = 2;
export const CORRIDOR_GY = 3;
export const CORRIDOR_HELD = 4;
export const CORRIDOR_OPEN = 5;
export const CORRIDOR_GOALS = 6;
const CORRIDOR_KEYS = 7;
const CORRIDOR_MAX_DOORS = 3;

const KEY_WORDS = 7;

export const corridorKeyX = (i: number): number => CORRIDOR_KEYS + 2 * i;
export const corridorKeyY = (i: number): number => CORRIDOR_KEYS + 2 * i + 1;
export const corridorBit = (mask: number, i: number): boolean => ((mask >>> i) & 1) === 1;

export const corridorDecoder: EnvDecoder = {
  slots: CORRIDOR_KEYS + 2 * CORRIDOR_MAX_DOORS,
  discrete: true,
  decode(frame, out) {
    const w = frame.words, doors = w.length - KEY_WORDS, mask = (1 << doors) - 1;
    for (let i = 0; i < 4; i++) out[i] = w[i]!;
    out[CORRIDOR_HELD] = w[4]! & mask;
    out[CORRIDOR_OPEN] = (w[4]! >>> doors) & mask;
    out[CORRIDOR_GOALS] = w[6]!;
    for (let i = 0; i < doors; i++) {
      out[corridorKeyX(i)] = w[KEY_WORDS + i]! & 15;
      out[corridorKeyY(i)] = w[KEY_WORDS + i]! >>> 4;
    }
  },
  geometry({ environment }, initial) {
    const grid = environment.grid, doors = parameter(environment, 'doors');
    if (!grid || grid.walls.length !== grid.w * grid.h) throw new Error(`${environment.envId} walls are missing`);
    if (doors > CORRIDOR_MAX_DOORS || initial.words.length !== KEY_WORDS + doors) throw new Error(`${environment.envId} words do not carry its ${doors} keys`);
    const walls = Uint8Array.from(grid.walls);
    const wall = (x: number, y: number) => y < 0 || y >= grid.h || walls[y * grid.w + x] !== 0;
    const doorways: GridCell[] = [];
    for (let x = 0; x < grid.w; x++) for (let y = 0; y < grid.h; y++) if (!wall(x, y) && wall(x, y - 1) && wall(x, y + 1)) doorways.push({ x, y });
    if (doorways.length !== doors) throw new Error(`${environment.envId} walls open ${doorways.length} doorways for ${doors} doors`);
    return { grid: { w: grid.w, h: grid.h, walls, doors: doorways } };
  },
  deriveEvents(prev, next) {
    const doors = next.words.length - KEY_WORDS, before = prev.words[4]!, after = next.words[4]!;
    if (next.done) {
      if (next.reward !== 1) throw new Error('A corridor reset has no goal reward');
      const events: SimEvent[] = [{ type: 'goal' }];
      for (let i = 0; i < doors; i++) if (corridorBit(before, doors + i)) events.push({ type: 'door', index: i, open: false });
      events.push({ type: 'teleport' }, { type: 'done' });
      return events;
    }
    const gained = after & ~before;
    if (gained === 0) return NO_EVENTS;
    const events: SimEvent[] = [];
    for (let i = 0; i < doors; i++) {
      if (corridorBit(gained, i)) events.push({ type: 'key', index: i });
      if (corridorBit(gained, doors + i)) events.push({ type: 'door', index: i, open: true });
    }
    return events;
  },
};
