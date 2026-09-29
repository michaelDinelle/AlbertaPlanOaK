import { constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const RW = { MODE: 0, CURSOR: 1, FOCUS: 2, CLOCK: 3, ROSTER: 4, GAME: 5, STEP: 6, MATCH: 7, OUR_POINTS: 8, THEIR_POINTS: 9, OUR_WINS: 10, THEIR_WINS: 11, GAIN: 12, MEMBER: 13, COUNT: 14 } as const;

export const SIDE = 24;
export const TILES = SIDE * SIDE;
export const UNITS = 16;
const RELICS = 6;
const PAIRS = RELICS / 2;
export const MATCHES = 5;
export const MATCH_STEPS = 100;
export const MAX_ENERGY = 400;
const FOCUS = 8;

export const ORDER = 1;
export const TARGET = 2;

export const CENTER = 0;
export const SAP = 5;
const KEYS = 6;

const SCOUT = 0;
const SAPPER = 1;
export const LEAGUE = 2;

export const NEBULA = 1;
export const ASTEROID = 2;

export const RF_CURSOR_X = 0;
export const RF_CURSOR_Y = 1;

const OBSERVED = 64;
const GRID_DELTA = [[0, -1], [0, 1], [-1, 0], [1, 0]] as const;
const PARAMETERS = ['moveCost', 'sensorRange', 'visionReduction', 'energyReduction', 'sapCost', 'sapRange', 'dropoff', 'voidFactor', 'nebulaSpeed', 'nodeSpeed', 'nodeMagnitude'] as const;
const CHOICES: Record<Parameter, string> = {
  moveCost: 'RELICFOG_MOVE_COSTS', sensorRange: 'RELICFOG_SENSOR_RANGES', visionReduction: 'RELICFOG_VISION_REDUCTIONS', energyReduction: 'RELICFOG_ENERGY_REDUCTIONS',
  sapCost: 'RELICFOG_SAP_COSTS', sapRange: 'RELICFOG_SAP_RANGES', dropoff: 'RELICFOG_DROPOFF_QUARTERS', voidFactor: 'RELICFOG_VOID_SIXTEENTHS',
  nebulaSpeed: 'RELICFOG_NEBULA_DRIFT_FORTIETHS', nodeSpeed: 'RELICFOG_NODE_DRIFT_HUNDREDTHS', nodeMagnitude: 'RELICFOG_NODE_DRIFT_MAGNITUDES',
};

const AT = (() => {
  const units = TILES / 2, relics = units + 4 * UNITS, weights = relics + RELICS, node = weights + TILES / 8, sight = node + 1, params = sight + TILES / 8;
  const last = params + PARAMETERS.length, orders = last + 2 * UNITS;
  return { tiles: 0, units, relics, weights, node, sight, params, last, orders, end: orders + UNITS };
})();

type Parameter = (typeof PARAMETERS)[number];
export type Params = Record<Parameter, number>;

export interface Order { type: number; dx: number; dy: number }
export interface Unit extends Order { x: number; y: number; alive: boolean; energy: number }
export interface Relic { x: number; y: number; revealed: boolean }

export interface World {
  kind: Uint8Array;
  energy: Int8Array;
  weight: Uint8Array;
  sight: readonly [Uint8Array, Uint8Array];
  units: readonly [Unit[], Unit[]];
  relics: Relic[];
  node: { x: number; y: number };
  params: Params;
  pairs: number;
}

export interface RelicShape {
  opponent: number;
  steps: readonly (readonly [number, number])[];
  choices: Record<Parameter, readonly number[]>;
}

export const field16 = (view: ArrayLike<number>, k: number): number => ((view[k >> 1]! >>> (16 * (k & 1))) & 0xffff);
const signed16 = (value: number): number => (value << 16) >> 16;
const signed8 = (value: number): number => (value << 24) >> 24;

const orderOf = (code: number): Order => ({ type: code & 7, dx: ((code >>> 3) & 15) - FOCUS, dy: ((code >>> 7) & 15) - FOCUS });
export const pendingOf = (view: ArrayLike<number>, i: number): Order => orderOf(field16(view, AT.orders + i));
export const focusOf = (words: ArrayLike<number>): { dx: number; dy: number } => ({ dx: (words[RW.FOCUS]! & 15) - FOCUS, dy: ((words[RW.FOCUS]! >>> 4) & 15) - FOCUS });

export function keyOf(frame: EnvFrame): number {
  const pressed = frame.actions[0]!;
  return pressed >= 0 && pressed < KEYS ? Math.trunc(pressed) : CENTER;
}

export function relicShape(env: EnvDescriptor): RelicShape {
  if (env.words !== RW.COUNT) throw new Error(`${env.envId} records ${env.words} state words where the fleet command reads ${RW.COUNT}`);
  if (2 * env.view !== AT.end) throw new Error(`${env.envId} views ${env.view} words where a ${SIDE}-tile sector of ${UNITS} ships a side packs ${AT.end / 2}`);
  if (env.obs !== OBSERVED) throw new Error(`${env.envId} observes ${env.obs} values where the fleet command senses ${OBSERVED}`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} keys where the fleet command reads ${KEYS}`);
  const opponent = parameter(env, 'opponent'), headings = constantList(env, 'RELICFOG_HEADINGS');
  if (opponent !== SCOUT && opponent !== SAPPER && opponent !== LEAGUE) throw new Error(`${env.envId} fields opponent ${opponent}`);
  if (headings.length !== SAP) throw new Error(`${env.envId} heads ${headings.length} keys where the fleet moves on ${SAP}`);
  const steps = headings.map((heading, key) => key === CENTER ? [0, 0] as const : GRID_DELTA[heading & 3]!);
  const choices = Object.fromEntries(PARAMETERS.map((name) => [name, constantList(env, CHOICES[name])])) as Record<Parameter, number[]>;
  return { opponent, steps, choices };
}

export function makeWorld(): World {
  const unit = (): Unit => ({ x: 0, y: 0, alive: false, energy: 0, type: 0, dx: 0, dy: 0 });
  return {
    kind: new Uint8Array(TILES), energy: new Int8Array(TILES), weight: new Uint8Array(TILES), sight: [new Uint8Array(TILES), new Uint8Array(TILES)],
    units: [Array.from({ length: UNITS }, unit), Array.from({ length: UNITS }, unit)], relics: Array.from({ length: RELICS }, () => ({ x: 0, y: 0, revealed: false })),
    node: { x: 0, y: 0 }, params: Object.fromEntries(PARAMETERS.map((name) => [name, 0])) as Params, pairs: 0,
  };
}

export function readWorld(view: ArrayLike<number>, shape: RelicShape, out: World): World {
  for (let i = 0; i < TILES; i++) {
    const cell = (field16(view, AT.tiles + (i >> 1)) >>> (8 * (i & 1))) & 255;
    out.kind[i] = cell & 3;
    out.energy[i] = (cell >>> 2) - 32;
    out.weight[i] = (field16(view, AT.weights + (i >> 3)) >>> (2 * (i & 7))) & 3;
    for (let team = 0; team < 2; team++) out.sight[team]![i] = (field16(view, AT.sight + team * (TILES / 16) + (i >> 4)) >>> (i & 15)) & 1;
  }
  for (let team = 0; team < 2; team++) for (let i = 0; i < UNITS; i++) {
    const at = AT.units + 2 * (team * UNITS + i), place = field16(view, at), unit = out.units[team]![i]!, last = orderOf(field16(view, AT.last + team * UNITS + i));
    unit.x = place & 31;
    unit.y = (place >>> 5) & 31;
    unit.alive = ((place >>> 10) & 1) === 1;
    unit.energy = signed16(field16(view, at + 1));
    unit.type = last.type;
    unit.dx = last.dx;
    unit.dy = last.dy;
  }
  let revealed = 0;
  for (let r = 0; r < RELICS; r++) {
    const code = field16(view, AT.relics + r), relic = out.relics[r]!;
    relic.x = code & 31;
    relic.y = (code >>> 5) & 31;
    relic.revealed = ((code >>> 10) & 1) === 1;
    if (relic.revealed) revealed++;
  }
  out.pairs = revealed >> 1;
  const node = field16(view, AT.node);
  out.node.x = node & 31;
  out.node.y = (node >>> 5) & 31;
  PARAMETERS.forEach((name, k) => {
    const raw = field16(view, AT.params + k), value = name === 'nebulaSpeed' ? signed8(raw) : raw;
    if (!shape.choices[name].includes(value)) throw new Error(`RelicFog draws ${name} ${value}, which is none of ${CHOICES[name]}`);
    out.params[name] = value;
  });
  return out;
}

export const live = (world: World, tile: number): boolean => world.weight[tile]! > 0 && world.weight[tile]! <= world.pairs;

export function scoredBy(world: World, team: number, out: Uint8Array): number {
  out.fill(0);
  let count = 0;
  for (const unit of world.units[team]!) {
    const tile = unit.y * SIDE + unit.x;
    if (!unit.alive || unit.energy < 0 || !live(world, tile) || out[tile]) continue;
    out[tile] = 1;
    count++;
  }
  return count;
}

export function fired(before: World, after: World, team: number, i: number): boolean {
  const unit = after.units[team]![i]!, was = before.units[team]![i]!, { sapRange, sapCost } = before.params;
  if (unit.type !== SAP || !was.alive || Math.abs(unit.dx) > sapRange || Math.abs(unit.dy) > sapRange || was.energy < sapCost) return false;
  const x = was.x + unit.dx, y = was.y + unit.dy;
  return x >= 0 && y >= 0 && x < SIDE && y < SIDE;
}

export function reborn(before: World, after: World, team: number, i: number, steps: RelicShape['steps']): boolean {
  const was = before.units[team]![i]!, unit = after.units[team]![i]!;
  if (!was.alive || !unit.alive) return false;
  if (was.energy < 0) return true;
  const dx = unit.x - was.x, dy = unit.y - was.y, step = unit.type >= 1 && unit.type <= 4 ? steps[unit.type]! : null;
  return (dx !== 0 || dy !== 0) && (!step || dx !== step[0] || dy !== step[1]);
}

export const engaged = (before: World, team: number, i: number): boolean => before.units[team]![i]!.alive && before.units[team]![i]!.energy >= 0;

export const driftOf = (world: World): readonly [number, number] => world.params.nebulaSpeed > 0 ? [1, -1] : [-1, 1];

export const driftsBy = (world: World, ticks: number): number => ticks < 1 ? 0 : 1 + Math.floor((ticks - 1) * Math.abs(world.params.nebulaSpeed) / 40);

export function shifted(before: World, after: World, dx: number, dy: number): boolean {
  for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) if (after.kind[y * SIDE + x] !== before.kind[((y - dy + SIDE) % SIDE) * SIDE + ((x - dx + SIDE) % SIDE)]) return false;
  return true;
}

const GOAL: SimEvent = { type: 'goal' }, HARM: SimEvent = { type: 'harm' }, DONE: SimEvent = { type: 'done' };
const WON = Object.freeze([GOAL]) as SimEvent[];
const LOST = Object.freeze([HARM]) as SimEvent[];
const WON_GAME = Object.freeze([GOAL, DONE]) as SimEvent[];
const LOST_GAME = Object.freeze([HARM, DONE]) as SimEvent[];
const REVEALS = [0, 1, 2].map((value) => Object.freeze([{ type: 'cue', value }]) as SimEvent[]);
const RELIC_WORDS = [AT.relics >> 1, (AT.relics >> 1) + 1, (AT.relics >> 1) + 2] as const;

export const relicFogDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    const place = field16(frame.view, AT.units + 2 * frame.words[RW.CURSOR]!);
    out[RF_CURSOR_X] = place & 31;
    out[RF_CURSOR_Y] = (place >>> 5) & 31;
  },
  geometry({ environment }, initial) {
    const shape = relicShape(environment), world = readWorld(initial.view, shape, makeWorld());
    if (initial.words[RW.STEP] !== 1 || initial.words[RW.MATCH] !== 1 || initial.words[RW.MODE] !== ORDER) throw new Error(`${environment.envId} opens at step ${initial.words[RW.STEP]} of match step ${initial.words[RW.MATCH]} in mode ${initial.words[RW.MODE]}`);
    if (world.units.some((team) => team.filter((unit) => unit.alive).length !== 1)) throw new Error(`${environment.envId} opens without exactly one ship a side`);
    return { grid: { w: SIDE, h: SIDE, walls: new Uint8Array(TILES) } };
  },
  deriveEvents(prev, next) {
    if (next.done) return next.reward > 0 ? WON_GAME : LOST_GAME;
    if (prev.words[RW.GAME] !== next.words[RW.GAME]) return NO_EVENTS;
    if (next.words[RW.OUR_WINS]! > prev.words[RW.OUR_WINS]!) return WON;
    if (next.words[RW.THEIR_WINS]! > prev.words[RW.THEIR_WINS]!) return LOST;
    if (RELIC_WORDS.every((k) => prev.view[k] === next.view[k])) return NO_EVENTS;
    for (let q = 0; q < PAIRS; q++) if (((field16(next.view, AT.relics + q) >>> 10) & 1) > ((field16(prev.view, AT.relics + q) >>> 10) & 1)) return REVEALS[q]!;
    return NO_EVENTS;
  },
};
