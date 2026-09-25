import { constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const ANC_X = 0;
export const ANC_Y = 1;

export const EV_LASTHIT = 1;
export const EV_KILL = 2;
export const EV_DEATH = 3;
export const EV_TOWER = 4;
export const EV_RESPAWN = 5;
export const EV_LEVEL = 6;
export const EV_ACQUIRED = 7;
export const EV_RELEASED = 8;
export const EV_ALLY = 10;

export const KIND_HERO = 1;
export const KIND_CREEP = 2;
export const KIND_TOWER = 3;
export const KIND_CAMP = 4;

export const FLAG_DEAD = 1;
export const FLAG_STUN = 2;
export const FLAG_SLOW = 4;
export const FLAG_FRENZY = 8;
export const FLAG_FORTIFY = 16;
export const FLAG_WARD = 32;
export const FLAG_TAUNT = 64;
export const FLAG_ATTACK = 128;

export const ARCHETYPES = ['carry', 'nuker', 'stunner', 'tank', 'support', 'blinker'] as const;

const FIFO = 16;
const FIFO_WORDS = FIFO + 1;
const CAMP_WORDS = 3;

export interface AncientsLayout {
  heroes: number;
  lanes: number;
  laneSlots: number;
  side: number;
  own: number;
  camps: number;
  fog: boolean;
  jungle: boolean;
  fifo: number;
  towerHp: number;
  towerTarget: number;
  camp: number;
  memory: number;
  tail: number;
}

export function ancientsLayout(env: EnvDescriptor): AncientsLayout {
  const heroes = parameter(env, 'heroes'), lanes = parameter(env, 'lanes'), fog = parameter(env, 'fog') !== 0, jungle = parameter(env, 'jungle') !== 0;
  const laneSlots = constantList(env, 'ANCIENTS_LANE_LENGTH').length, side = constantList(env, 'ANCIENTS_SEAT_LANE').length, own = constantList(env, 'ANCIENTS_STRUCTURE_HP').length, camps = constantList(env, 'ANCIENTS_CAMP_X').length;
  const fifo = 8 * heroes, towerHp = fifo + 2 * lanes * FIFO_WORDS, towerTarget = towerHp + own, camp = towerTarget + own;
  const memory = camp + (jungle ? camps * CAMP_WORDS : 0), tail = memory + (fog ? 2 * heroes : 0);
  if (tail + 1 !== env.view) throw new Error(`${env.envId} records ${env.view} view words where its parameters describe ${tail + 1}`);
  return { heroes, lanes, laneSlots, side, own, camps, fog, jungle, fifo, towerHp, towerTarget, camp, memory, tail };
}

export const heroX = (view: Uint32Array, slot: number): number => view[4 * slot]! & 0xffff;
export const heroY = (view: Uint32Array, slot: number): number => view[4 * slot]! >>> 16;
export const heroHp = (view: Uint32Array, slot: number): number => view[4 * slot + 1]! & 0xffff;
export const heroMaxHp = (view: Uint32Array, slot: number): number => view[4 * slot + 1]! >>> 16;
export const heroMana = (view: Uint32Array, slot: number): number => view[4 * slot + 2]! & 0xffff;
export const heroLevel = (view: Uint32Array, slot: number): number => (view[4 * slot + 2]! >>> 16) & 31;
export const heroArch = (view: Uint32Array, slot: number): number => (view[4 * slot + 2]! >>> 21) & 7;
export const heroFlags = (view: Uint32Array, slot: number): number => view[4 * slot + 2]! >>> 24;
export const heroGold = (view: Uint32Array, slot: number): number => view[4 * slot + 3]! & 0xffff;
export const heroTarget = (view: Uint32Array, slot: number): number => view[4 * slot + 3]! >>> 19;

const fifoAt = (layout: AncientsLayout, fifo: number): number => layout.fifo + ((fifo >= layout.laneSlots ? layout.lanes : 0) + fifo % layout.laneSlots) * FIFO_WORDS;
export const creepCount = (view: Uint32Array, layout: AncientsLayout, fifo: number): number => view[fifoAt(layout, fifo)]!;
export const creepArc = (view: Uint32Array, layout: AncientsLayout, fifo: number, k: number): number => (view[fifoAt(layout, fifo) + 1 + k]! << 16) >> 16;
export const creepHp = (view: Uint32Array, layout: AncientsLayout, fifo: number, k: number): number => view[fifoAt(layout, fifo) + 1 + k]! >>> 16;

export const towerHp = (view: Uint32Array, layout: AncientsLayout, s: number): number => (view[layout.towerHp + (s >> 1)]! >>> ((s & 1) << 4)) & 0xffff;
export const towerTarget = (view: Uint32Array, layout: AncientsLayout, s: number): number => (view[layout.towerTarget + (s >> 1)]! >>> ((s & 1) << 4)) & 0xffff;

export const campX = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.camp + CAMP_WORDS * i]! & 0xffff;
export const campY = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.camp + CAMP_WORDS * i]! >>> 16;
export const campHp = (view: Uint32Array, layout: AncientsLayout, i: number, j: number): number => (view[layout.camp + CAMP_WORDS * i + 1 + (j >> 1)]! >>> ((j & 1) << 4)) & 0xffff;
export const campAggro = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.camp + CAMP_WORDS * i + 2]! >>> 16;

export const memoryX = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.memory + 2 * i]! & 0xffff;
export const memoryY = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.memory + 2 * i]! >>> 16;
export const memoryAge = (view: Uint32Array, layout: AncientsLayout, i: number): number => view[layout.memory + 2 * i + 1]! >>> 16;

const tailOf = (view: Uint32Array): number => view[view.length - 1]!;
export const shownEvent = (view: Uint32Array): number => (tailOf(view) >>> 8) & 15;
export const leagueMember = (view: Uint32Array): number => (tailOf(view) >>> 16) & 15;

export const codeKind = (code: number): number => code >>> 10;
export const codeIndex = (code: number): number => code & 1023;
export const heroSlotOf = (layout: AncientsLayout, hero: number): number => hero < layout.side ? hero : layout.heroes + hero - layout.side;
export const creepFifoOf = (code: number): number => (code & 1023) >>> 4;
export const creepSlotOf = (code: number): number => code & 15;

export interface LaneShape {
  x0: number[];
  y0: number[];
  ax1: number[];
  ay1: number[];
  corner: number[];
  ax2: number[];
  ay2: number[];
  length: number[];
}

export function laneShape(env: EnvDescriptor): LaneShape {
  return {
    x0: constantList(env, 'ANCIENTS_LANE_X0'), y0: constantList(env, 'ANCIENTS_LANE_Y0'),
    ax1: constantList(env, 'ANCIENTS_LANE_AX1'), ay1: constantList(env, 'ANCIENTS_LANE_AY1'), corner: constantList(env, 'ANCIENTS_LANE_CORNER'),
    ax2: constantList(env, 'ANCIENTS_LANE_AX2'), ay2: constantList(env, 'ANCIENTS_LANE_AY2'), length: constantList(env, 'ANCIENTS_LANE_LENGTH'),
  };
}

const offset = (s: number, a: number): number => (s * a + (a < 0 ? 65535 : 0)) >> 16;

export function lanePoint(shape: LaneShape, fifo: number, arc: number, out: { x: number; y: number }): { x: number; y: number } {
  const corner = shape.corner[fifo]!, s = arc < 0 ? 0 : arc, s1 = s < corner ? s : corner, s2 = s > corner ? s - corner : 0;
  out.x = shape.x0[fifo]! + offset(s1, shape.ax1[fifo]!) + offset(s2, shape.ax2[fifo]!);
  out.y = shape.y0[fifo]! + offset(s1, shape.ay1[fifo]!) + offset(s2, shape.ay2[fifo]!);
  return out;
}

export function laneHeading(shape: LaneShape, fifo: number, arc: number): number {
  const second = arc > shape.corner[fifo]!;
  return Math.atan2(second ? shape.ay2[fifo]! : shape.ay1[fifo]!, second ? shape.ax2[fifo]! : shape.ax1[fifo]!);
}

const CUES = new Set([EV_LASTHIT, EV_KILL, EV_DEATH, EV_TOWER, EV_RESPAWN, EV_LEVEL, EV_ACQUIRED, EV_RELEASED, EV_ALLY]);

export const ancientsDecoder: EnvDecoder = {
  slots: 2,
  decode(frame, out) {
    out[ANC_X] = heroX(frame.view, 0);
    out[ANC_Y] = heroY(frame.view, 0);
  },
  geometry: () => ({}),
  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    const code = shownEvent(next.view);
    if (!next.done && !CUES.has(code)) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (CUES.has(code)) events.push({ type: 'cue', value: code });
    if (next.done) {
      if (next.reward > 0) events.push({ type: 'goal' });
      else if (next.reward < 0) events.push({ type: 'harm' });
      events.push({ type: 'done' });
    }
    return events;
  },
};
