import { IDLE, MOVE, MW, STEP_X, STEP_Y, aimOf, directionOf, goldOf, sightOf, timeOf, unitsOf, type MicroShape, type Unit } from '@sim/envs/microrts';
import type { EnvFrame } from '@sim/types';

export type Fetch = (step: number) => EnvFrame | null;

export interface Piece extends Unit {
  from: number;
  start: number;
  heading: number;
  born: number;
}

export interface Turn {
  step: number;
  length: number;
  time: number;
  game: number;
  side: number;
  foe: number;
  member: number;
  gold: readonly [number, number];
  opening: boolean;
  exact: boolean;
  prior: number;
  pieces: Piece[];
  fallen: Piece[];
  sight: Uint8Array;
  made: number;
}

const LIMIT = 256;
const KEEP = 768;
const RETRY = 0.25;

const continues = (a: Piece, b: Unit): boolean => {
  if (a.kind !== b.kind || a.owner !== b.owner) return false;
  if (a.cell === b.cell) return true;
  if (a.act !== MOVE) return false;
  const d = directionOf(a);
  return b.x === a.x + STEP_X[d]! && b.y === a.y + STEP_Y[d]!;
};

export class Chronicle {
  private readonly cache = new Map<number, Turn>();
  clock = 0;

  constructor(private readonly shape: MicroShape) {}

  headingToward(x: number, y: number, tx: number, ty: number): number {
    return Math.atan2(-(tx - x), -(ty - y));
  }

  private rest(unit: Unit): number {
    const centre = (this.shape.width - 1) / 2;
    return this.headingToward(unit.x, unit.y, centre, centre);
  }

  private facing(unit: Unit, prior: Piece | null): number {
    if (this.shape.types[unit.kind]!.mobile && unit.act !== IDLE && unit.act !== 0) {
      const aim = aimOf(unit);
      return this.headingToward(unit.x, unit.y, aim.x, aim.y);
    }
    return prior ? prior.heading : this.rest(unit);
  }

  private build(step: number, frame: EnvFrame, prior: Turn | null, rooted: boolean): Turn {
    const shape = this.shape, units = unitsOf(frame, shape), time = timeOf(frame), side = frame.words[MW.SIDE]!;
    const opening = time === 0;
    const before = prior?.pieces ?? [], links = new Int32Array(units.length).fill(-1), taken = new Uint8Array(before.length);
    if (prior) {
      let j = 0;
      for (let i = 0; i < before.length && j < units.length; i++) {
        if (!continues(before[i]!, units[j]!)) continue;
        links[j++] = i;
        taken[i] = 1;
      }
      for (; j < units.length; j++) if (units[j]!.act !== IDLE) throw new Error(`MicroRTS turn at step ${step} lists a new unit ${j} that is already acting`);
    }
    const pieces = units.map((unit, k): Piece => {
      const i = links[k]!, p = i >= 0 ? before[i]! : null;
      const same = p !== null && p.act === unit.act && p.arg === unit.arg && p.cell === unit.cell;
      const start = unit.act === IDLE ? NaN : same ? p.start : prior || opening ? time - 1 : NaN;
      const born = p ? p.born : prior && !opening ? time : -Infinity;
      return { ...unit, from: i, start, heading: this.facing(unit, p), born };
    });
    const fallen = prior ? before.filter((_, i) => !taken[i]) : [];
    return {
      step, length: frame.words[MW.LENGTH]!, time, game: frame.words[MW.GAME]!, side, foe: frame.words[MW.FOE]!, member: frame.words[MW.MEMBER]!,
      gold: [goldOf(frame, 0), goldOf(frame, 1)], opening, exact: prior ? prior.exact : opening || rooted, prior: prior?.step ?? -1,
      pieces, fallen, sight: sightOf(units, side, shape, new Uint8Array(shape.width * shape.width)), made: this.clock,
    };
  }

  private turn(step: number, sample: EnvFrame, fetch: Fetch): Turn {
    const hit = this.cache.get(step);
    if (hit && (hit.exact || this.clock - hit.made < RETRY)) return hit;
    const chain: { step: number; frame: EnvFrame }[] = [];
    let base: Turn | null = null, cut = false, key = step, frame = sample;
    for (;;) {
      const cached = chain.length > 0 ? this.cache.get(key) : undefined;
      if (cached?.exact) { base = cached; break; }
      chain.push({ step: key, frame });
      if (timeOf(frame) === 0 || chain.length > LIMIT) break;
      const last = fetch(key - 1);
      if (!last) { cut = true; break; }
      frame = last;
      key = key - 1 - last.words[MW.PHASE]!;
    }
    let prior = base;
    for (let i = chain.length - 1; i >= 0; i--) {
      const turn = this.build(chain[i]!.step, chain[i]!.frame, prior, i === chain.length - 1 && !cut);
      this.cache.delete(turn.step);
      this.cache.set(turn.step, turn);
      prior = turn;
    }
    while (this.cache.size > KEEP) this.cache.delete(this.cache.keys().next().value!);
    return prior!;
  }

  cached(step: number): Turn | null {
    return this.cache.get(step) ?? null;
  }

  at(step: number, fetch: Fetch): Turn | null {
    const f = fetch(step);
    return f ? this.turn(step - f.words[MW.PHASE]!, f, fetch) : null;
  }

  finale(opening: Turn, fetch: Fetch): Turn | null {
    if (opening.step === 0) return null;
    return this.at(opening.step - 1, fetch);
  }

  clear(): void {
    this.cache.clear();
  }
}
