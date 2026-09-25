import { constantRecords, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const ELEVEN_BALL_X = 0;
export const ELEVEN_BALL_Y = 1;
export const ELEVEN_BALL_Z = 2;
export const ELEVEN_PLAYER = 3;
const ELEVEN_SLOTS = 24;
const ELEVEN_WORDS = 7;
const ELEVEN_TAIL = 7;
const ELEVEN_OBSERVED = 18;
const ELEVEN_CONTROLS = 6;

export const ELEVEN_NORMAL = 0;
export const ELEVEN_KICKOFF = 1;
export const ELEVEN_GOALKICK = 2;
export const ELEVEN_FREEKICK = 3;
export const ELEVEN_CORNER = 4;
export const ELEVEN_THROWIN = 5;
export const ELEVEN_PENALTY = 6;

export const ELEVEN_DOWN = 1;
export const ELEVEN_DIVING = 2;
export const ELEVEN_ACTIVE = 4;
export const ELEVEN_TAKER = 8;
export const ELEVEN_OFFSIDE = 16;

export interface ElevenPlayer { x: number; y: number; vx: number; vy: number; stamina: number; flags: number }

export interface ElevenState {
  clock: number;
  game: number;
  us: number;
  them: number;
  mode: number;
  team: number;
  live: boolean;
  held: boolean;
  half: number;
  kickoff: number;
  owner: number;
  active: number;
  taker: number;
  bx: number;
  by: number;
  bz: number;
  sx: number;
  sy: number;
  ready: number;
  scan: number;
  switched: boolean;
  bvx: number;
  bvy: number;
  bvz: number;
  last: number;
  member: number;
  offside: number;
  wins: number;
  losses: number;
  draws: number;
  close: number;
  near: number;
  players: ElevenPlayer[];
}

export const elevenFine = (value: number): number => value / 512 - 64;
export const elevenCoarse = (value: number): number => (value - 128) / 10;
export const elevenPlayers = (view: ArrayLike<number>): number => (view.length - ELEVEN_TAIL) / 2;

export function elevenState(): ElevenState {
  return {
    clock: 0, game: 0, us: 0, them: 0, mode: 0, team: 0, live: false, held: false, half: 0, kickoff: 0, owner: 0, active: 0, taker: 0,
    bx: 0, by: 0, bz: 0, sx: 0, sy: 0, ready: 0, scan: 0, switched: false, bvx: 0, bvy: 0, bvz: 0, last: 0, member: 0, offside: 0,
    wins: 0, losses: 0, draws: 0, close: 0, near: 0,
    players: Array.from({ length: ELEVEN_SLOTS }, () => ({ x: 0, y: 0, vx: 0, vy: 0, stamina: 0, flags: 0 })),
  };
}

export function readEleven(frame: EnvFrame, out: ElevenState): ElevenState {
  const words = frame.words, view = frame.view, count = elevenPlayers(view), tail = 2 * count, status = words[3]!;
  out.clock = words[0]!;
  out.game = words[1]!;
  out.us = words[2]! & 0xffff;
  out.them = words[2]! >>> 16;
  out.mode = status & 7;
  out.team = (status >>> 3) & 1;
  out.live = ((status >>> 4) & 1) === 1;
  out.held = ((status >>> 5) & 1) === 1;
  out.half = (status >>> 6) & 1;
  out.kickoff = (status >>> 7) & 1;
  out.owner = (status >>> 8) & 255;
  out.active = (status >>> 16) & 255;
  out.taker = status >>> 24;
  out.bx = wordFloat(words[4]!);
  out.by = wordFloat(words[5]!);
  out.bz = wordFloat(words[6]!);
  for (let k = 0; k < count; k++) {
    const place = view[2 * k]!, motion = view[2 * k + 1]!, player = out.players[k]!;
    player.x = elevenFine(place & 0xffff);
    player.y = elevenFine(place >>> 16);
    player.vx = elevenCoarse(motion & 255);
    player.vy = elevenCoarse((motion >>> 8) & 255);
    player.stamina = ((motion >>> 16) & 255) / 255;
    player.flags = motion >>> 24;
  }
  out.sx = elevenFine(view[tail]! & 0xffff);
  out.sy = elevenFine(view[tail]! >>> 16);
  out.ready = view[tail + 1]! & 0xffff;
  out.scan = (view[tail + 1]! >>> 16) & 255;
  out.switched = view[tail + 1]! >>> 24 !== 0;
  out.bvx = elevenFine(view[tail + 2]! & 0xffff);
  out.bvy = elevenFine(view[tail + 2]! >>> 16);
  out.bvz = elevenFine(view[tail + 3]! & 0xffff);
  out.last = (view[tail + 3]! >>> 16) & 255;
  out.member = view[tail + 3]! >>> 24;
  out.offside = view[tail + 4]!;
  out.wins = view[tail + 5]! & 0xffff;
  out.losses = view[tail + 5]! >>> 16;
  out.draws = view[tail + 6]! & 0xffff;
  out.close = (view[tail + 6]! >>> 16) & 255;
  out.near = view[tail + 6]! >>> 24;
  return out;
}

export function elevenTeamSlots(env: EnvDescriptor): number {
  const shapes = constantRecords(env, 'ELE_SHAPES'), sizes = new Set(shapes.flatMap((shape) => ['u', 'v', 'follow', 'lift', 'mark'].map((key) => (shape[key] as number[]).length)));
  if (sizes.size !== 1) throw new Error(`${env.envId} declares formation shapes of unequal team size`);
  return [...sizes][0]!;
}

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }]);
const WON_END: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST_END: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const END: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const elevenDecoder: EnvDecoder = {
  slots: ELEVEN_PLAYER + 2 * ELEVEN_SLOTS,
  decode(frame, out) {
    const words = frame.words, view = frame.view, count = elevenPlayers(view);
    out[ELEVEN_BALL_X] = wordFloat(words[4]!);
    out[ELEVEN_BALL_Y] = wordFloat(words[5]!);
    out[ELEVEN_BALL_Z] = wordFloat(words[6]!);
    for (let k = 0; k < count; k++) {
      out[ELEVEN_PLAYER + 2 * k] = elevenFine(view[2 * k]! & 0xffff);
      out[ELEVEN_PLAYER + 2 * k + 1] = elevenFine(view[2 * k]! >>> 16);
    }
  },
  geometry({ environment }) {
    const perSide = parameter(environment, 'perSide'), team = elevenTeamSlots(environment);
    if (environment.words !== ELEVEN_WORDS) throw new Error(`${environment.envId} records ${environment.words} words where a football state packs into ${ELEVEN_WORDS}`);
    if (environment.view !== 4 * perSide + ELEVEN_TAIL) throw new Error(`${environment.envId} views ${environment.view} words where ${perSide} a side take ${4 * perSide + ELEVEN_TAIL}`);
    if (environment.obs !== ELEVEN_OBSERVED || environment.actions !== ELEVEN_CONTROLS) throw new Error(`${environment.envId} observes ${environment.obs} and controls ${environment.actions} where a footballer takes ${ELEVEN_OBSERVED} and ${ELEVEN_CONTROLS}`);
    if (perSide > team || 2 * perSide > ELEVEN_SLOTS) throw new Error(`${environment.envId} fields ${perSide} a side in teams of ${team} slots`);
    return {};
  },
  deriveEvents(_prev, next) {
    if (next.reward !== 0 && Math.abs(next.reward) !== 1) throw new Error(`Eleven step ${next.step} pays ${next.reward}, but only goals pay`);
    if (next.reward > 0) return (next.done ? WON_END : WON) as SimEvent[];
    if (next.reward < 0) return (next.done ? LOST_END : LOST) as SimEvent[];
    return next.done ? (END as SimEvent[]) : NO_EVENTS;
  },
};
