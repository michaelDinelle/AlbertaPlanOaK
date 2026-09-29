import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const REVERSI_OWN = 0;
export const REVERSI_OPP = 2;
export const REVERSI_LEGAL = 4;
const REVERSI_PROPOSAL = 6;
const REVERSI_SHOWN = 7;
const REVERSI_GAME = 8;
const REVERSI_LAST = 9;
const REVERSI_MEMBER = 10;
const REVERSI_WORDS = 11;
const REVERSI_STRIDE = 8;

export const REVERSI_PASS = 65;
export const REVERSI_PLAY = 0;
export const REVERSI_NEXT = 1;

export const REVERSI_EMPTY = 0;
export const REVERSI_AGENT = 1;
export const REVERSI_RIVAL = 2;

export const REVERSI_X = 0;
export const REVERSI_Y = 1;
const REVERSI_SHOWN_SLOT = 2;
const REVERSI_GAME_SLOT = 3;
const REVERSI_LAST_SLOT = 4;
const REVERSI_MEMBER_SLOT = 5;
const REVERSI_AGENT_SLOT = 6;
const REVERSI_RIVAL_SLOT = 7;

const RAYS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]] as const;

const bits = (word: number): number => {
  let v = word >>> 0;
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

export const reversiSquare = (column: number, row: number): number => REVERSI_STRIDE * row + column;
export const reversiColumn = (square: number): number => square & 7;
export const reversiRow = (square: number): number => square >>> 3;
export const reversiHas = (words: ArrayLike<number>, plane: number, square: number): boolean => ((words[plane + (square >>> 5)]! >>> (square & 31)) & 1) === 1;
export const reversiCount = (words: ArrayLike<number>, plane: number): number => bits(words[plane]!) + bits(words[plane + 1]!);
export const reversiOwner = (words: ArrayLike<number>, square: number): number =>
  reversiHas(words, REVERSI_OWN, square) ? REVERSI_AGENT : reversiHas(words, REVERSI_OPP, square) ? REVERSI_RIVAL : REVERSI_EMPTY;
export const reversiProposal = (words: ArrayLike<number>): number => words[REVERSI_PROPOSAL]!;
export const reversiShown = (words: ArrayLike<number>): number => words[REVERSI_SHOWN]!;
export const reversiGame = (words: ArrayLike<number>): number => words[REVERSI_GAME]!;
export const reversiLast = (words: ArrayLike<number>): number => words[REVERSI_LAST]!;
export const reversiMember = (words: ArrayLike<number>): number => words[REVERSI_MEMBER]!;
export const reversiAgentWhite = (words: ArrayLike<number>): boolean => (reversiGame(words) & 1) === 1;
export const reversiAction = (actions: ArrayLike<number>): number => {
  const control = actions[0]!;
  return control >= 0 && control < 2 ? Math.trunc(control) : REVERSI_PLAY;
};
export const reversiFlips = (words: ArrayLike<number>, square: number): number[] => {
  const flipped: number[] = [];
  for (const [dx, dy] of RAYS) {
    const run: number[] = [];
    for (let x = reversiColumn(square) + dx, y = reversiRow(square) + dy; x >= 0 && y >= 0 && x < REVERSI_STRIDE && y < REVERSI_STRIDE; x += dx, y += dy) {
      const at = reversiSquare(x, y), owner = reversiOwner(words, at);
      if (owner === REVERSI_RIVAL) run.push(at);
      else {
        if (owner === REVERSI_AGENT) flipped.push(...run);
        break;
      }
    }
  }
  return flipped;
};

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const DRAWN: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const reversiDecoder: EnvDecoder = {
  slots: REVERSI_RIVAL_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, proposal = reversiProposal(words);
    out[REVERSI_X] = reversiColumn(proposal);
    out[REVERSI_Y] = reversiRow(proposal);
    out[REVERSI_SHOWN_SLOT] = reversiShown(words);
    out[REVERSI_GAME_SLOT] = reversiGame(words);
    out[REVERSI_LAST_SLOT] = reversiLast(words);
    out[REVERSI_MEMBER_SLOT] = reversiMember(words);
    out[REVERSI_AGENT_SLOT] = reversiCount(words, REVERSI_OWN);
    out[REVERSI_RIVAL_SLOT] = reversiCount(words, REVERSI_OPP);
  },
  geometry({ environment }, initial) {
    const size = parameter(environment, 'size'), words = initial.words;
    if (environment.words !== REVERSI_WORDS) throw new Error(`${environment.envId} records ${environment.words} words where a Reversi position packs into ${REVERSI_WORDS}`);
    if (!(size >= 1 && size <= REVERSI_STRIDE)) throw new Error(`${environment.envId} declares a ${size}-square board that does not fit its bitboards`);
    if (environment.obs !== size * size) throw new Error(`${environment.envId} observes ${environment.obs} values where a ${size}×${size} afterstate holds ${size * size}`);
    if (environment.actions !== 2) throw new Error(`${environment.envId} offers ${environment.actions} actions where Reversi answers play or next`);
    for (let square = 0; square < REVERSI_STRIDE * REVERSI_STRIDE; square++) {
      const inside = reversiColumn(square) < size && reversiRow(square) < size;
      const own = reversiHas(words, REVERSI_OWN, square), opp = reversiHas(words, REVERSI_OPP, square);
      if (own && opp) throw new Error(`${environment.envId} opens with both players on square ${square}`);
      if (!inside && (own || opp || reversiHas(words, REVERSI_LEGAL, square))) throw new Error(`${environment.envId} opens with a disc off its ${size}×${size} board`);
    }
    const proposal = reversiProposal(words);
    if (proposal >= REVERSI_STRIDE * REVERSI_STRIDE || !reversiHas(words, REVERSI_LEGAL, proposal)) throw new Error(`${environment.envId} opens proposing ${proposal}, which is not a legal move`);
    if (reversiShown(words) >= reversiCount(words, REVERSI_LEGAL)) throw new Error(`${environment.envId} opens having shown more proposals than it has legal moves`);
    return { grid: { w: size, h: size, walls: new Uint8Array(size * size) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST : DRAWN) as SimEvent[];
  },
};
