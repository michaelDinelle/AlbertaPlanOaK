import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const G2048_CELLS = 16;
export const G2048_WIN = Math.log2(2048);

export const g2048Tile = (words: ArrayLike<number>, cell: number): number => (words[cell >>> 2]! >>> ((cell & 3) << 3)) & 255;

export function g2048Highest(words: ArrayLike<number>): number {
  let highest = 0;
  for (let cell = 0; cell < G2048_CELLS; cell++) highest = Math.max(highest, g2048Tile(words, cell));
  return highest;
}

const RESET: SimEvent[] = [{ type: 'teleport' }, { type: 'done' }];
const WON: SimEvent[] = [{ type: 'goal' }];

export const g2048Decoder: EnvDecoder = {
  slots: G2048_CELLS,
  discrete: true,
  decode(frame, out) {
    for (let cell = 0; cell < G2048_CELLS; cell++) out[cell] = g2048Tile(frame.words, cell);
  },
  geometry(_descriptor, initial) {
    const side = Math.sqrt(initial.words.length * 4);
    if (side * side !== G2048_CELLS) throw new Error(`A 2048 board holds ${G2048_CELLS} tiles, not ${initial.words.length * 4}`);
    return { grid: { w: side, h: side, walls: new Uint8Array(G2048_CELLS) } };
  },
  deriveEvents(prev, next) {
    if (next.done) return RESET;
    return g2048Highest(next.words) >= G2048_WIN && g2048Highest(prev.words) < G2048_WIN ? WON : NO_EVENTS;
  },
};
