import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const SOKOBAN_BOXES = 4;
export const SOKOBAN_PX = 0;
export const SOKOBAN_PY = 1;
export const SOKOBAN_BOX = 2;
export const SOKOBAN_TARGET = SOKOBAN_BOX + 2 * SOKOBAN_BOXES;
export const SOKOBAN_SLOTS = SOKOBAN_TARGET + 2 * SOKOBAN_BOXES;

export interface SokobanLevel {
  side: number;
  floor: boolean[];
  targets: number[];
  boxes: number[];
  player: number;
}

export const sokobanOpen = (frame: EnvFrame) => frame.words.length >= 7;

const half = (view: Uint32Array, i: number) => (view[i >> 1]! >>> (16 * (i & 1))) & 0xffff;

export const cellX = (cell: number) => (cell & 15) - 2;
export const cellY = (cell: number) => (cell >> 4) - 2;

const bytes = (word: number) => [0, 8, 16, 24].map((s) => (word >>> s) & 255).filter((c) => c !== 0);

export function sokobanLevel(frame: EnvFrame): SokobanLevel | null {
  const view = frame.view, count = view.length;
  if (sokobanOpen(frame) || count === 0) return null;
  const odd = half(view, 2 * count - 1) !== 0, rows = 2 * count - (odd ? 5 : 6), side = rows + 2;
  const floor = new Array<boolean>(side * side).fill(false);
  for (let r = 0; r < rows; r++) for (let x = 0; x < side; x++) floor[(r + 1) * side + x] = ((half(view, r) >> (x + 2)) & 1) === 1;
  return {
    side,
    floor,
    targets: bytes(half(view, rows) | (half(view, rows + 1) << 16)),
    boxes: bytes(half(view, rows + 2) | (half(view, rows + 3) << 16)),
    player: half(view, rows + 4),
  };
}

export function sokobanCrates(frame: EnvFrame, level: SokobanLevel): { player: number; boxes: number[] } {
  const player = frame.words[0]!;
  return player === 0 ? { player: level.player, boxes: level.boxes } : { player, boxes: bytes(frame.words[1]!) };
}

export const sokobanDecoder: EnvDecoder = {
  slots: SOKOBAN_SLOTS,
  discrete: true,
  decode(frame, out) {
    out.fill(-1);
    const w = frame.words;
    if (sokobanOpen(frame)) {
      out[SOKOBAN_PX] = w[0]!; out[SOKOBAN_PY] = w[1]!;
      out[SOKOBAN_BOX] = w[2]!; out[SOKOBAN_BOX + 1] = w[3]!;
      out[SOKOBAN_TARGET] = w[4]!; out[SOKOBAN_TARGET + 1] = w[5]!;
      return;
    }
    const level = sokobanLevel(frame);
    if (!level) return;
    const { player, boxes } = sokobanCrates(frame, level);
    out[SOKOBAN_PX] = cellX(player); out[SOKOBAN_PY] = cellY(player);
    boxes.forEach((c, i) => { out[SOKOBAN_BOX + 2 * i] = cellX(c); out[SOKOBAN_BOX + 2 * i + 1] = cellY(c); });
    level.targets.forEach((c, i) => { out[SOKOBAN_TARGET + 2 * i] = cellX(c); out[SOKOBAN_TARGET + 2 * i + 1] = cellY(c); });
  },
  geometry({ environment }, initial) {
    const level = sokobanLevel(initial), side = level ? level.side : (environment.parameters['side'] as number);
    if (!Number.isInteger(side) || side < 3) throw new Error(`${environment.envId} lacks its side`);
    const walls = Uint8Array.from({ length: side * side }, (_, i) => (level && !level.floor[i] ? 1 : 0));
    return { grid: { w: side, h: side, walls } };
  },
  deriveEvents(prev, next): SimEvent[] {
    if (sokobanOpen(next)) {
      if (next.done) return [{ type: 'goal' }, { type: 'done' }];
      return next.words[6] === 0 && prev.words[6] !== 0 ? [{ type: 'teleport' }] : NO_EVENTS;
    }
    const solved = next.done !== 0 && next.reward > 5, gain = Math.round(next.reward + 0.1) - (solved ? 10 : 0);
    if (solved) return [{ type: 'goal' }, { type: 'done' }];
    const events: SimEvent[] = gain !== 0 ? [{ type: 'cue', value: gain }] : [];
    if (next.done) events.push({ type: 'harm' }, { type: 'done' });
    return events.length ? events : NO_EVENTS;
  },
};
