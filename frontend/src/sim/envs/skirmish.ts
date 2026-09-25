import { constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const SKW = { TICK: 0, SLICE: 1, BATTLE: 2, WINS: 3, LOSSES: 4, DRAWN: 5, TIMEOUTS: 6, RACE: 7, START: 8, SELECT: 9, CALL: 10, ALLIES: 11, FOES: 12, MEMBER: 13 } as const;
export const SPAWN = { REFLECT_NEAR: 0, REFLECT_FAR: 1, CENTRE: 2, RING_NEAR: 3, RING_FAR: 4 } as const;
export const RULE = { MIN_DAMAGE: 0, SLOWDOWN: 1, REGEN_DELAY: 2, REGEN_RATE: 3, FORM: 4, ARRIVE: 5 } as const;
export const FOE = 20;
export const NONE = 63;
export const MOVES = 4;
export const ATTACK = 4;
export const CYCLE = 5;
export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;

const WORDS = 14;
const STRIDE = 3;
const CAPACITY = 64;
const OBSERVED = 18;
const KEYS = 6;
const FIXED = 1024;

export interface Kind {
  hp: number;
  shield: number;
  armor: number;
  range: number;
  sight: number;
  damage: number;
  bonus: number;
  bonusVs: number;
  armored: boolean;
  cooldown: number;
  speed: number;
  splash: number;
  heal: number;
  slow: number;
  suicide: boolean;
  threat: number;
}

export interface SkirmishShape {
  allies: number;
  enemies: number;
  races: number;
  league: boolean;
  slices: number;
  battleTicks: number;
  surround: number;
  tick: number;
  arena: number;
  timeoutReward: number;
  perRace: number;
  kinds: Kind[];
  spawn: number[];
  rules: number[];
}

export interface Unit {
  index: number;
  slot: number;
  x: number;
  y: number;
  hp: number;
  shield: number;
  kind: number;
  target: number;
  slowed: boolean;
  hunt: boolean;
  wait: number;
}

export type Outcome = 'won' | 'lost' | 'drawn' | 'timeout';

const LISTS = ['SKIRMISH_HP', 'SKIRMISH_SHIELD', 'SKIRMISH_ARMOR', 'SKIRMISH_RANGE', 'SKIRMISH_SIGHT', 'SKIRMISH_DAMAGE', 'SKIRMISH_BONUS', 'SKIRMISH_BONUS_VS', 'SKIRMISH_ARMORED',
  'SKIRMISH_COOLDOWN', 'SKIRMISH_SPEED', 'SKIRMISH_SPLASH', 'SKIRMISH_HEAL', 'SKIRMISH_SLOW', 'SKIRMISH_SUICIDE', 'SKIRMISH_THREAT'] as const;

export function skirmishShape(env: EnvDescriptor): SkirmishShape {
  const allies = parameter(env, 'allies'), enemies = parameter(env, 'enemies'), races = parameter(env, 'races');
  if (env.words !== WORDS) throw new Error(`${env.envId} records ${env.words} words where a skirmish packs ${WORDS}`);
  if (env.obs !== OBSERVED || env.actions !== KEYS) throw new Error(`${env.envId} observes ${env.obs} and offers ${env.actions} keys where a skirmisher takes ${OBSERVED} and ${KEYS}`);
  if (allies < 1 || allies > FOE || enemies < 1 || FOE + enemies > CAPACITY) throw new Error(`${env.envId} fields ${allies} against ${enemies} in ${CAPACITY} slots split at ${FOE}`);
  if (env.view !== STRIDE * (allies + enemies)) throw new Error(`${env.envId} views ${env.view} words where ${allies + enemies} units take ${STRIDE * (allies + enemies)}`);
  const lists = LISTS.map((key) => constantList(env, key)), perRace = constantList(env, 'SKIRMISH_WEIGHT').length, count = lists[0]!.length;
  if (lists.some((list) => list.length !== count) || perRace < 1 || count % perRace !== 0 || races < 1 || races > count / perRace) throw new Error(`${env.envId} lists unit tables that do not split into ${races} races of ${perRace}`);
  const kinds = Array.from({ length: count }, (_, t) => {
    const [hp, shield, armor, range, sight, damage, bonus, bonusVs, armored, cooldown, speed, splash, heal, slow, suicide, threat] = lists.map((list) => list[t]!) as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
    return { hp, shield, armor, range, sight, damage, bonus, bonusVs, armored: armored !== 0, cooldown, speed, splash, heal, slow, suicide: suicide !== 0, threat };
  });
  const spawn = constantList(env, 'SKIRMISH_SPAWN'), rules = constantList(env, 'SKIRMISH_RULES');
  if (spawn.length !== 5 || rules.length !== 6) throw new Error(`${env.envId} lists ${spawn.length} spawn distances and ${rules.length} rules where the view reads 5 and 6`);
  return {
    allies, enemies, races, league: parameter(env, 'league') !== 0, slices: parameter(env, 'slices'), battleTicks: parameter(env, 'battleTicks'), surround: parameter(env, 'surround'),
    tick: parameter(env, 'tick'), arena: parameter(env, 'arena'), timeoutReward: parameter(env, 'timeoutReward'), perRace, kinds, spawn, rules,
  };
}

export const slotOf = (index: number, allies: number): number => (index < allies ? index : FOE + index - allies);
export const indexOf = (slot: number, allies: number): number => (slot < FOE ? slot : allies + slot - FOE);
export const unitCount = (frame: EnvFrame): number => frame.view.length / STRIDE;
export const xOf = (frame: EnvFrame, index: number): number => (frame.view[STRIDE * index]! & 0xffff) / FIXED;
export const yOf = (frame: EnvFrame, index: number): number => (frame.view[STRIDE * index]! >>> 16) / FIXED;
export const hpOf = (frame: EnvFrame, index: number): number => frame.view[STRIDE * index + 1]! & 0xffff;
export const shieldOf = (frame: EnvFrame, index: number): number => frame.view[STRIDE * index + 1]! >>> 16;
export const kindOf = (frame: EnvFrame, index: number): number => frame.view[STRIDE * index + 2]! & 15;
export const targetOf = (frame: EnvFrame, index: number): number => (frame.view[STRIDE * index + 2]! >>> 4) & 63;
export const waitOf = (frame: EnvFrame, index: number): number => (frame.view[STRIDE * index + 2]! >>> 16) & 255;
export const nowOf = (frame: EnvFrame, slices: number): number => frame.words[SKW.TICK]! * slices + frame.words[SKW.SLICE]!;
export const keyOf = (frame: EnvFrame): number => {
  const pressed = frame.actions[0]!;
  return pressed >= 0 && pressed < KEYS ? Math.trunc(pressed) : 0;
};

export const blankUnit = (): Unit => ({ index: 0, slot: 0, x: 0, y: 0, hp: 0, shield: 0, kind: 0, target: NONE, slowed: false, hunt: false, wait: 0 });

export function readUnit(frame: EnvFrame, index: number, allies: number, out: Unit): Unit {
  const state = frame.view[STRIDE * index + 2]!;
  out.index = index;
  out.slot = slotOf(index, allies);
  out.x = xOf(frame, index);
  out.y = yOf(frame, index);
  out.hp = hpOf(frame, index);
  out.shield = shieldOf(frame, index);
  out.kind = state & 15;
  out.target = (state >>> 4) & 63;
  out.slowed = ((state >>> 10) & 1) === 1;
  out.hunt = ((state >>> 11) & 1) === 1;
  out.wait = (state >>> 16) & 255;
  return out;
}

const COUNTERS = [[SKW.WINS, 'won'], [SKW.LOSSES, 'lost'], [SKW.DRAWN, 'drawn'], [SKW.TIMEOUTS, 'timeout']] as const;

export function outcomeOf(before: EnvFrame, after: EnvFrame): Outcome {
  let found: Outcome | null = null;
  for (const [word, outcome] of COUNTERS) {
    const rise = after.words[word]! - before.words[word]!;
    if (rise === 1 && found === null) found = outcome;
    else if (rise !== 0) throw new Error(`Skirmish step ${after.step} moves the ${outcome} count by ${rise}`);
  }
  if (found === null || after.words[SKW.BATTLE] !== before.words[SKW.BATTLE]! + 1) throw new Error(`Skirmish step ${after.step} ends battle ${before.words[SKW.BATTLE]} without exactly one result`);
  return found;
}

function checkOpening(env: EnvDescriptor, shape: SkirmishShape, initial: EnvFrame): void {
  const w = initial.words;
  for (const word of [SKW.TICK, SKW.SLICE, SKW.BATTLE, SKW.WINS, SKW.LOSSES, SKW.DRAWN, SKW.TIMEOUTS]) if (w[word] !== 0) throw new Error(`${env.envId} opens with word ${word} at ${w[word]}`);
  if (w[SKW.ALLIES] !== shape.allies || w[SKW.FOES] !== shape.enemies || w[SKW.CALL] !== NONE || w[SKW.RACE]! >= shape.races) throw new Error(`${env.envId} opens with ${w[SKW.ALLIES]} allies, ${w[SKW.FOES]} foes, call ${w[SKW.CALL]} and race ${w[SKW.RACE]}`);
  const select = w[SKW.SELECT]!;
  if (select < FOE || select >= FOE + shape.enemies) throw new Error(`${env.envId} opens selecting slot ${select}, not a foe`);
  const unit = blankUnit();
  for (let k = 0; k < shape.allies + shape.enemies; k++) {
    readUnit(initial, k, shape.allies, unit);
    const kind = shape.kinds[unit.kind], race = Math.floor(unit.kind / shape.perRace), word = initial.view[STRIDE * k + 2]!;
    if (!kind || race !== w[SKW.RACE] || unit.hp !== Math.ceil(kind.hp) || unit.shield !== Math.ceil(kind.shield) || unit.target !== NONE || unit.wait !== 0 || ((word >>> 12) & 15) !== 0 || word >>> 24 !== 0)
      throw new Error(`${env.envId} opens unit ${k} as kind ${unit.kind} with ${unit.hp} hp, ${unit.shield} shield and target ${unit.target}`);
    if (unit.x < 0 || unit.y < 0 || unit.x > shape.arena || unit.y > shape.arena) throw new Error(`${env.envId} opens unit ${k} outside the ${shape.arena} arena`);
    if (k < shape.allies && unit.kind !== kindOf(initial, shape.allies + k)) throw new Error(`${env.envId} opens ally ${k} unlike its mirror foe`);
  }
}

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const ENDED: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const skirmishDecoder: EnvDecoder = {
  slots: 2 * CAPACITY,
  decode(frame, out) {
    const count = unitCount(frame);
    for (let k = 0; k < count; k++) {
      out[2 * k] = xOf(frame, k);
      out[2 * k + 1] = yOf(frame, k);
    }
  },
  geometry({ environment }, initial) {
    checkOpening(environment, skirmishShape(environment), initial);
    return {};
  },
  deriveEvents(prev, next) {
    if (!next.done) return NO_EVENTS;
    const outcome = outcomeOf(prev, next), reward = next.reward;
    if ((outcome === 'won' && reward !== 1) || (outcome === 'lost' && reward !== -1) || (outcome === 'drawn' && reward !== 0)) throw new Error(`Skirmish step ${next.step} pays ${reward} for a battle ${outcome}`);
    return (reward > 0 ? WON : reward < 0 ? LOST : ENDED) as SimEvent[];
  },
};
