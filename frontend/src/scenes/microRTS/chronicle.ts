import {
  ATTACK, BASE, HARVEST, IDLE, MODE_KEY, MOVE, MW, NEUTRAL, NOBODY, RANGED, RESOURCE, RETURN, STEP_X, STEP_Y, WORKER, aimOf, directionOf, etaOf, goldOf, keyOf, sightOf, timeOf, unitsOf, type MicroShape, type Unit,
} from '@sim/envs/microrts';
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
  orders: Piece[];
  sight: Uint8Array;
  made: number;
}

const LIMIT = 256;
const KEEP = 768;
const RETRY = 0.25;

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

  private continues(a: Piece, b: Unit, elapsed: number): boolean {
    if (a.kind !== b.kind || a.owner !== b.owner) return false;
    if (a.cell === b.cell) return true;
    const type = this.shape.types[a.kind]!;
    if (elapsed >= type.move + (a.act === MOVE ? 1 : 0)) return type.mobile && Math.abs(b.x - a.x) + Math.abs(b.y - a.y) <= 1 + Math.floor((elapsed - 1) / type.move);
    if (a.act !== MOVE) return false;
    const d = directionOf(a);
    return b.x === a.x + STEP_X[d]! && b.y === a.y + STEP_Y[d]!;
  }

  private begun(unit: Unit, p: Piece | null, prior: Turn, time: number): number {
    if (time - prior.time === 1 || unit.owner === prior.side) return prior.time;
    const types = this.shape.types, eta = etaOf(unit, types), after = p && !Number.isNaN(p.start) ? p.start + etaOf(p, types) : prior.time;
    return Math.min(time - 1, Math.max(after, prior.time, eta > 0 ? time - eta + 1 : prior.time));
  }

  private order(earlier: Turn, time: number, pieces: readonly Piece[], piece: Piece, key: number, mode: number): Piece | null {
    const types = this.shape.types, type = types[piece.kind]!, width = this.shape.width;
    const foe = (other: Piece) => other.owner !== NEUTRAL && other.owner !== piece.owner;
    const toward = (act: number, x: number, y: number): Piece | null => {
      const order = { ...piece, act, arg: act === ATTACK ? (x - piece.x + 3) | ((y - piece.y + 3) << 3) : key - 1, start: earlier.time, heading: this.headingToward(piece.x, piece.y, x, y) };
      return earlier.time + etaOf(order, types) <= time ? order : null;
    };
    if (piece.kind === RANGED && mode > 0) {
      const span = (other: Piece) => (other.x - piece.x) ** 2 + (other.y - piece.y) ** 2;
      const target = earlier.pieces.filter((other) => foe(other) && span(other) <= type.range * type.range).sort((a, b) => span(a) - span(b))[mode - 1];
      return target ? toward(ATTACK, target.x, target.y) : null;
    }
    const x = piece.x + STEP_X[key - 1]!, y = piece.y + STEP_Y[key - 1]!;
    if (x < 0 || y < 0 || x >= width || y >= width) return null;
    const other = earlier.pieces.find((candidate) => candidate.x === x && candidate.y === y);
    if (!other) {
      const arrived = pieces.some((moved) => moved.x === x && moved.y === y && moved.owner === piece.owner);
      return type.mobile && !(piece.kind === WORKER && mode > 0) && arrived ? toward(MOVE, x, y) : null;
    }
    if (piece.kind === WORKER && mode === 0 && other.kind === RESOURCE && piece.carry === 0) return toward(HARVEST, x, y);
    if (piece.kind === WORKER && mode === 0 && other.kind === BASE && other.owner === piece.owner && piece.carry > 0) return toward(RETURN, x, y);
    return type.fighter && mode === 0 && foe(other) ? toward(ATTACK, x, y) : null;
  }

  private orders(earlier: Turn, step: number, time: number, pieces: readonly Piece[], fetch: Fetch): Piece[] {
    const out: Piece[] = [];
    for (let s = earlier.step + 1; s <= step; s++) {
      const now = fetch(s), was = fetch(s - 1);
      if (!now || !was) continue;
      const key = keyOf(now), picked = was.words[MW.SELECTED]!, at = was.words[MW.CURSOR]!;
      if (picked === NOBODY || key === 0 || key === MODE_KEY || (s < step && now.words[MW.SELECTED] === picked)) continue;
      const piece = earlier.pieces.find((candidate) => candidate.cell === at && candidate.owner === earlier.side);
      const order = piece ? this.order(earlier, time, pieces, piece, key, was.words[MW.MODE]!) : null;
      if (order) out.push(order);
    }
    return out;
  }

  private build(step: number, frame: EnvFrame, prior: Turn | null, rooted: boolean, fetch: Fetch): Turn {
    const shape = this.shape, units = unitsOf(frame, shape), time = timeOf(frame), side = frame.words[MW.SIDE]!;
    const opening = time === 0, elapsed = prior ? time - prior.time : 1;
    const before = prior?.pieces ?? [], links = new Int32Array(units.length).fill(-1), taken = new Uint8Array(before.length);
    if (prior) {
      let j = 0;
      for (let i = 0; i < before.length && j < units.length; i++) {
        if (!this.continues(before[i]!, units[j]!, elapsed)) continue;
        links[j++] = i;
        taken[i] = 1;
      }
      if (elapsed === 1) for (; j < units.length; j++) if (units[j]!.act !== IDLE) throw new Error(`MicroRTS turn at step ${step} lists a new unit ${j} that is already acting`);
    }
    const pieces = units.map((unit, k): Piece => {
      const i = links[k]!, p = i >= 0 ? before[i]! : null, lapsed = p !== null && etaOf(p, shape.types) > 0 && p.start + etaOf(p, shape.types) <= time;
      const same = p !== null && p.act === unit.act && p.arg === unit.arg && p.cell === unit.cell && !lapsed;
      const start = unit.act === IDLE ? NaN : same ? p.start : opening ? time - 1 : prior ? this.begun(unit, p, prior, time) : NaN;
      const born = p ? p.born : prior && !opening ? time : -Infinity;
      return { ...unit, from: i, start, heading: this.facing(unit, p), born };
    });
    const fallen = prior ? before.filter((_, i) => !taken[i]) : [];
    const orders = prior && elapsed > 1 ? this.orders(prior, step, time, pieces, fetch) : [];
    return {
      step, length: frame.words[MW.LENGTH]!, time, game: frame.words[MW.GAME]!, side, foe: frame.words[MW.FOE]!, member: frame.words[MW.MEMBER]!,
      gold: [goldOf(frame, 0), goldOf(frame, 1)], opening, exact: prior ? prior.exact : opening || rooted, prior: prior?.step ?? -1,
      pieces, fallen, orders, sight: sightOf(units, side, shape, new Uint8Array(shape.width * shape.width)), made: this.clock,
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
      const turn = this.build(chain[i]!.step, chain[i]!.frame, prior, i === chain.length - 1 && !cut, fetch);
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
