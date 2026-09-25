import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const TETRIS_W = 10;
export const TETRIS_H = 20;

export const tetrisByte = (words: ArrayLike<number>, at: number): number => (words[at >>> 2]! >>> ((at & 3) << 3)) & 255;
export const tetrisFilled = (words: ArrayLike<number>, cell: number): boolean => ((words[cell >>> 5]! >>> (cell & 31)) & 1) === 1;

export interface TetrisPiece { x: number; y: number; orient: number; type: number; next: number; delay: number; dealt: number; level: number }

export function tetrisPiece(words: ArrayLike<number>): TetrisPiece {
  return {
    x: tetrisByte(words, 52), y: tetrisByte(words, 53), orient: tetrisByte(words, 54), type: tetrisByte(words, 55),
    next: tetrisByte(words, 56), delay: tetrisByte(words, 57), dealt: tetrisByte(words, 63), level: tetrisByte(words, 65),
  };
}

export const tetrisPlaced = (value: number): Pick<TetrisPiece, 'x' | 'y' | 'orient' | 'type'> => ({ x: value & 255, y: (value >>> 8) & 255, orient: (value >>> 16) & 255, type: (value >>> 24) & 255 });

export const tetrisScore = (words: ArrayLike<number>): number => words[8]! >>> 0;
export const tetrisLines = (words: ArrayLike<number>): number => words[9]! >>> 0;
export const tetrisGame = (words: ArrayLike<number>): number => words[10]! >>> 0;
export const tetrisCleared = (words: ArrayLike<number>): number => words[11]! & 7;
export const tetrisClearedRow = (words: ArrayLike<number>, i: number): number => ((words[11]! >>> (3 + 5 * i)) & 31) - 2;

const DONE: SimEvent[] = [{ type: 'done' }];

export const tetrisDecoder: EnvDecoder = {
  slots: 7,
  discrete: true,
  sequential: true,
  decode(frame, out) {
    const p = tetrisPiece(frame.words);
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.orient;
    out[3] = p.type;
    out[4] = p.next;
    out[5] = p.delay;
    out[6] = p.level;
  },
  geometry(_descriptor, initial) {
    if (initial.words.length !== 20) throw new Error(`A Tetris state is 20 words, not ${initial.words.length}`);
    return { grid: { w: TETRIS_W, h: TETRIS_H, walls: new Uint8Array(TETRIS_W * TETRIS_H) } };
  },
  deriveEvents(prev, next) {
    if (next.done) return DONE;
    if (tetrisByte(prev.words, 57) !== 0 || tetrisByte(next.words, 57) === 0) return NO_EVENTS;
    const place: SimEvent = { type: 'place', value: next.words[13]! };
    return tetrisCleared(next.words) > 0 ? [{ type: 'goal' }, place] : [place];
  },
};
