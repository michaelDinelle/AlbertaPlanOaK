import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const CHESS_SQUARES = 64;
export const CHESS_CURSOR = 8;
export const CHESS_STATE = 9;
export const CHESS_GAME = 10;
export const CHESS_LAST = 11;
export const CHESS_THINK = 12;
export const CHESS_PLY = 13;
const CHESS_WORDS = 14;

export const CHESS_PAWN = 1;
export const CHESS_KNIGHT = 2;
export const CHESS_BISHOP = 3;
export const CHESS_ROOK = 4;
export const CHESS_QUEEN = 5;
export const CHESS_KING = 6;
export const CHESS_BLACK = 8;

export const CHESS_UP = 0;
export const CHESS_DOWN = 1;
export const CHESS_LEFT = 2;
export const CHESS_RIGHT = 3;
export const CHESS_SELECT = 4;
export const CHESS_CANCEL = 5;

export const CHESS_WIN = 1;
export const CHESS_LOSS = 2;
export const CHESS_DRAW = 3;

export const CHESS_CURSOR_SLOT = CHESS_SQUARES;
export const CHESS_SELECTED_SLOT = CHESS_SQUARES + 1;
export const CHESS_PROMOTING_SLOT = CHESS_SQUARES + 2;
export const CHESS_CANDIDATE_SLOT = CHESS_SQUARES + 3;
export const CHESS_SIDE_SLOT = CHESS_SQUARES + 4;
export const CHESS_CASTLE_SLOT = CHESS_SQUARES + 5;
export const CHESS_PASSANT_SLOT = CHESS_SQUARES + 6;
export const CHESS_CLOCK_SLOT = CHESS_SQUARES + 7;
export const CHESS_GAME_SLOT = CHESS_SQUARES + 8;
export const CHESS_FROM_SLOT = CHESS_SQUARES + 9;
export const CHESS_TO_SLOT = CHESS_SQUARES + 10;
export const CHESS_CHECK_SLOT = CHESS_SQUARES + 11;
export const CHESS_OUTCOME_SLOT = CHESS_SQUARES + 12;
export const CHESS_RESULT_SLOT = CHESS_SQUARES + 13;
export const CHESS_THINK_SLOT = CHESS_SQUARES + 14;
export const CHESS_PLY_SLOT = CHESS_SQUARES + 15;
export const CHESS_AGENT_SLOT = CHESS_SQUARES + 16;

export const chessPiece = (words: ArrayLike<number>, square: number): number => (words[square >> 3]! >>> (4 * (square & 7))) & 15;
export const chessCursor = (words: ArrayLike<number>): number => words[CHESS_CURSOR]! & 63;
export const chessSelected = (words: ArrayLike<number>): number => (words[CHESS_CURSOR]! >>> 8) & 127;
export const chessPromoting = (words: ArrayLike<number>): number => (words[CHESS_CURSOR]! >>> 16) & 127;
export const chessCandidate = (words: ArrayLike<number>): number => (words[CHESS_CURSOR]! >>> 24) & 3;
export const chessSide = (words: ArrayLike<number>): number => words[CHESS_STATE]! & 1;
export const chessAgent = (words: ArrayLike<number>): number => chessSide(words) ^ (words[CHESS_THINK]! ? 1 : 0);
export const chessFlip = (words: ArrayLike<number>): number => (chessAgent(words) ? 63 : 0);

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }]);
const DRAWN: readonly SimEvent[] = Object.freeze([{ type: 'teleport' }, { type: 'done' }]);

export const chessDecoder: EnvDecoder = {
  slots: CHESS_AGENT_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, cursor = words[CHESS_CURSOR]!, state = words[CHESS_STATE]!, last = words[CHESS_LAST]!;
    for (let square = 0; square < CHESS_SQUARES; square++) out[square] = chessPiece(words, square);
    out[CHESS_CURSOR_SLOT] = cursor & 63;
    out[CHESS_SELECTED_SLOT] = (cursor >>> 8) & 127;
    out[CHESS_PROMOTING_SLOT] = (cursor >>> 16) & 127;
    out[CHESS_CANDIDATE_SLOT] = (cursor >>> 24) & 3;
    out[CHESS_SIDE_SLOT] = state & 1;
    out[CHESS_CASTLE_SLOT] = (state >>> 1) & 15;
    out[CHESS_PASSANT_SLOT] = (state >>> 8) & 255;
    out[CHESS_CLOCK_SLOT] = (state >>> 16) & 255;
    out[CHESS_GAME_SLOT] = words[CHESS_GAME]!;
    out[CHESS_FROM_SLOT] = last & 127;
    out[CHESS_TO_SLOT] = (last >>> 8) & 127;
    out[CHESS_CHECK_SLOT] = (last >>> 16) & 1;
    out[CHESS_OUTCOME_SLOT] = (last >>> 20) & 15;
    out[CHESS_RESULT_SLOT] = (last >>> 24) & 3;
    out[CHESS_THINK_SLOT] = words[CHESS_THINK]!;
    out[CHESS_PLY_SLOT] = words[CHESS_PLY]!;
    out[CHESS_AGENT_SLOT] = chessAgent(words);
  },
  geometry({ environment }, initial) {
    if (environment.words !== CHESS_WORDS) throw new Error(`${environment.envId} records ${environment.words} words where a chess position packs into ${CHESS_WORDS}`);
    const kings = [0, 0];
    for (let square = 0; square < CHESS_SQUARES; square++) {
      const code = chessPiece(initial.words, square);
      if ((code & 7) === CHESS_KING) kings[code >> 3]!++;
    }
    if (kings[0] !== 1 || kings[1] !== 1) throw new Error(`${environment.envId} opens with ${kings[0]} white and ${kings[1]} black kings`);
    return { grid: { w: 8, h: 8, walls: new Uint8Array(CHESS_SQUARES) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST : DRAWN) as SimEvent[];
  },
};
