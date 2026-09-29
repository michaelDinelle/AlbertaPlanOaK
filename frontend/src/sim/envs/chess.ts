import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const CHESS_SQUARES = 64;
export const CHESS_OFFER = 8;
export const CHESS_STATE = 9;
export const CHESS_GAME = 10;
export const CHESS_LAST = 11;
export const CHESS_PROPOSAL = 12;
export const CHESS_PLY = 13;
export const CHESS_PLAYED = 14;
const CHESS_WORDS = 15;

export const CHESS_PAWN = 1;
export const CHESS_KNIGHT = 2;
export const CHESS_BISHOP = 3;
export const CHESS_ROOK = 4;
export const CHESS_QUEEN = 5;
export const CHESS_KING = 6;
export const CHESS_BLACK = 8;

export const CHESS_PLAY = 0;
export const CHESS_NEXT = 1;

export const CHESS_PASSANT = 2;
export const CHESS_CASTLE = 3;

export const CHESS_WIN = 1;
export const CHESS_LOSS = 2;
export const CHESS_DRAW = 3;

export const CHESS_PROPOSED_FROM_SLOT = CHESS_SQUARES;
export const CHESS_PROPOSED_TO_SLOT = CHESS_SQUARES + 1;
export const CHESS_PROPOSED_PIECE_SLOT = CHESS_SQUARES + 2;
export const CHESS_INDEX_SLOT = CHESS_SQUARES + 3;
export const CHESS_COUNT_SLOT = CHESS_SQUARES + 4;
export const CHESS_SIDE_SLOT = CHESS_SQUARES + 5;
export const CHESS_CASTLE_SLOT = CHESS_SQUARES + 6;
export const CHESS_PASSANT_SLOT = CHESS_SQUARES + 7;
export const CHESS_CLOCK_SLOT = CHESS_SQUARES + 8;
export const CHESS_GAME_SLOT = CHESS_SQUARES + 9;
export const CHESS_FROM_SLOT = CHESS_SQUARES + 10;
export const CHESS_TO_SLOT = CHESS_SQUARES + 11;
export const CHESS_CHECK_SLOT = CHESS_SQUARES + 12;
export const CHESS_OUTCOME_SLOT = CHESS_SQUARES + 13;
export const CHESS_RESULT_SLOT = CHESS_SQUARES + 14;
export const CHESS_PLY_SLOT = CHESS_SQUARES + 15;
export const CHESS_PLAYED_FROM_SLOT = CHESS_SQUARES + 16;
export const CHESS_PLAYED_TO_SLOT = CHESS_SQUARES + 17;
export const CHESS_AGENT_SLOT = CHESS_SQUARES + 18;

export const chessPiece = (words: ArrayLike<number>, square: number): number => (words[square >> 3]! >>> (4 * (square & 7))) & 15;
export const chessMoveFrom = (move: number): number => move & 63;
export const chessMoveTo = (move: number): number => (move >>> 6) & 63;
export const chessMovePiece = (move: number): number => (move >>> 12) & 7;
export const chessMoveTaken = (move: number): number => (move >>> 15) & 7;
export const chessMovePromotion = (move: number): number => (move >>> 18) & 7;
export const chessMoveSpecial = (move: number): number => (move >>> 21) & 3;
export const chessMoveLands = (move: number): number => chessMovePromotion(move) || chessMovePiece(move);
export const chessProposal = (words: ArrayLike<number>): number => words[CHESS_PROPOSAL]!;
export const chessPlayed = (words: ArrayLike<number>): number => words[CHESS_PLAYED]!;
export const chessIndex = (words: ArrayLike<number>): number => words[CHESS_OFFER]! & 255;
export const chessCount = (words: ArrayLike<number>): number => (words[CHESS_OFFER]! >>> 8) & 255;
export const chessSide = (words: ArrayLike<number>): number => words[CHESS_STATE]! & 1;
export const chessAgent = chessSide;
export const chessFlip = (words: ArrayLike<number>): number => (chessAgent(words) ? 63 : 0);

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }]);
const DRAWN: readonly SimEvent[] = Object.freeze([{ type: 'teleport' }, { type: 'done' }]);

export const chessDecoder: EnvDecoder = {
  slots: CHESS_AGENT_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, proposal = chessProposal(words), played = chessPlayed(words), state = words[CHESS_STATE]!, last = words[CHESS_LAST]!;
    for (let square = 0; square < CHESS_SQUARES; square++) out[square] = chessPiece(words, square);
    out[CHESS_PROPOSED_FROM_SLOT] = chessMoveFrom(proposal);
    out[CHESS_PROPOSED_TO_SLOT] = chessMoveTo(proposal);
    out[CHESS_PROPOSED_PIECE_SLOT] = chessMoveLands(proposal);
    out[CHESS_INDEX_SLOT] = chessIndex(words);
    out[CHESS_COUNT_SLOT] = chessCount(words);
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
    out[CHESS_PLY_SLOT] = words[CHESS_PLY]!;
    out[CHESS_PLAYED_FROM_SLOT] = played ? chessMoveFrom(played) + 1 : 0;
    out[CHESS_PLAYED_TO_SLOT] = played ? chessMoveTo(played) + 1 : 0;
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
    const proposal = chessProposal(initial.words), mover = chessPiece(initial.words, chessMoveFrom(proposal));
    if (!chessCount(initial.words) || mover !== (chessMovePiece(proposal) | (chessAgent(initial.words) << 3)))
      throw new Error(`${environment.envId} opens proposing a move for a piece the agent does not have`);
    return { grid: { w: 8, h: 8, walls: new Uint8Array(CHESS_SQUARES) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST : DRAWN) as SimEvent[];
  },
};
