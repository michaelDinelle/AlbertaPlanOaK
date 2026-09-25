import * as THREE from 'three/webgpu';
import { atan, clamp, cos, dot, float, floor, length, max, min, select, sin, vec2 } from 'three/tsl';

type N<T extends string> = THREE.Node<T>;

export const HALF = 0.62;
const BEND = 1.3;
const PAD = 0.5;
export const CAP = 0.36;
const SHORE = 1.05;
const SWAY = 0.5;
const MIN_STRAIGHT = 2;
const FRAME = 16 / 9;

interface Piece { arc: boolean; s0: number; length: number; ax: number; ay: number; dx: number; dy: number; cx: number; cy: number; radius: number; angle: number; sweep: number }

export interface Near { d: number; s: number; side: number }

export interface Located { s: N<'float'>; side: N<'float'>; d: N<'float'> }

class Pen {
  readonly pieces: Piece[] = [];
  private x: number;
  private y: number;
  private heading: number;
  private s = 0;

  constructor(x: number, y: number, heading: number) {
    this.x = x;
    this.y = y;
    this.heading = heading;
  }

  line(length: number): this {
    const dx = Math.cos(this.heading), dy = Math.sin(this.heading);
    this.pieces.push({ arc: false, s0: this.s, length, ax: this.x, ay: this.y, dx, dy, cx: 0, cy: 0, radius: 0, angle: 0, sweep: 0 });
    this.x += dx * length;
    this.y += dy * length;
    this.s += length;
    return this;
  }

  turn(radius: number, angle: number): this {
    const sweep = Math.sign(angle), cx = this.x - Math.sin(this.heading) * radius * sweep, cy = this.y + Math.cos(this.heading) * radius * sweep;
    const start = Math.atan2(this.y - cy, this.x - cx), length = radius * Math.abs(angle);
    this.pieces.push({ arc: true, s0: this.s, length, ax: 0, ay: 0, dx: 0, dy: 0, cx, cy, radius, angle: start, sweep });
    this.x = cx + radius * Math.cos(start + angle);
    this.y = cy + radius * Math.sin(start + angle);
    this.heading += angle;
    this.s += length;
    return this;
  }
}

function draw(stations: number, rows: number): Piece[] | null {
  const total = stations - 1 + 2 * PAD;
  if (rows === 1) {
    const end = Math.min(0.8, total * 0.12), radius = (total - 2 * end) / (6 * SWAY);
    const pen = new Pen(0, 0, 0).line(end).turn(radius, SWAY).turn(radius, -2 * SWAY).turn(radius, 2 * SWAY).turn(radius, -SWAY).line(end);
    return centred(pen.pieces);
  }
  const straight = (total - (rows - 1) * Math.PI * BEND) / rows;
  if (straight < MIN_STRAIGHT) return null;
  const pen = new Pen(-straight / 2, -(rows - 1) * BEND, 0);
  for (let j = 0; j < rows; j++) {
    pen.line(straight);
    if (j < rows - 1) pen.turn(BEND, j % 2 === 0 ? Math.PI : -Math.PI);
  }
  return pen.pieces;
}

function centred(pieces: Piece[]): Piece[] {
  const box = extent(pieces), cx = (box.min.x + box.max.x) / 2, cy = (box.min.y + box.max.y) / 2;
  return pieces.map((p) => ({ ...p, ax: p.arc ? 0 : p.ax - cx, ay: p.arc ? 0 : p.ay - cy, cx: p.arc ? p.cx - cx : 0, cy: p.arc ? p.cy - cy : 0 }));
}

function pointOn(pieces: readonly Piece[], s: number, out: THREE.Vector2): THREE.Vector2 {
  let p = pieces[pieces.length - 1]!;
  for (const piece of pieces) if (s < piece.s0 + piece.length) { p = piece; break; }
  const t = s - p.s0;
  if (!p.arc) return out.set(p.ax + p.dx * t, p.ay + p.dy * t);
  const theta = p.angle + (p.sweep * t) / p.radius;
  return out.set(p.cx + p.radius * Math.cos(theta), p.cy + p.radius * Math.sin(theta));
}

function extent(pieces: readonly Piece[]): THREE.Box2 {
  const last = pieces[pieces.length - 1]!, total = last.s0 + last.length, box = new THREE.Box2(), at = new THREE.Vector2();
  for (let s = -CAP - SHORE * 0.55; s <= total + CAP + SHORE * 0.45; s += 0.05) box.expandByPoint(pointOn(pieces, s, at));
  return box.expandByScalar(HALF + 0.5);
}

function layout(stations: number): Piece[] {
  let best: { pieces: Piece[]; score: number } | null = null;
  for (let rows = 1; ; rows += 2) {
    const pieces = draw(stations, rows);
    if (!pieces) break;
    const box = extent(pieces), score = Math.abs(Math.log((box.max.x - box.min.x) / (box.max.y - box.min.y) / FRAME));
    if (!best || score < best.score) best = { pieces, score };
  }
  if (!best) throw new Error(`RiverSwim cannot lay ${stations} stations along a river`);
  return best.pieces;
}

export class Course {
  readonly pieces: Piece[];
  readonly length: number;

  constructor(readonly stations: number) {
    this.pieces = layout(stations);
    const last = this.pieces[this.pieces.length - 1]!;
    this.length = last.s0 + last.length;
  }

  station(k: number): number {
    return PAD + k;
  }

  point(s: number, out: THREE.Vector2): THREE.Vector2 {
    return pointOn(this.pieces, s, out);
  }

  heading(s: number): number {
    const clamped = Math.min(Math.max(s, 0), this.length);
    let p = this.pieces[this.pieces.length - 1]!;
    for (const piece of this.pieces) if (clamped < piece.s0 + piece.length) { p = piece; break; }
    if (!p.arc) return Math.atan2(p.dy, p.dx);
    const theta = p.angle + (p.sweep * (clamped - p.s0)) / p.radius;
    return Math.atan2(p.sweep * Math.cos(theta), -p.sweep * Math.sin(theta));
  }

  world(s: number, lateral: number, out: THREE.Vector3): THREE.Vector3 {
    const at = this.point(s, new THREE.Vector2()), h = this.heading(s);
    return out.set(at.x - Math.sin(h) * lateral, 0, -(at.y + Math.cos(h) * lateral));
  }

  bounds(): THREE.Box2 {
    return extent(this.pieces);
  }

  near(x: number, y: number): Near {
    let best: Near = { d: Infinity, s: 0, side: 0 };
    const last = this.pieces.length - 1;
    this.pieces.forEach((p, i) => {
      let d: number, s: number, side: number;
      if (!p.arc) {
        const rx = x - p.ax, ry = y - p.ay, raw = rx * p.dx + ry * p.dy, t = Math.min(Math.max(raw, 0), p.length);
        d = Math.hypot(x - (p.ax + p.dx * t), y - (p.ay + p.dy * t));
        s = p.s0 + (i === 0 && raw < 0 ? raw : i === last && raw > p.length ? raw : t);
        side = p.dx * ry - p.dy * rx;
      } else {
        const vx = x - p.cx, vy = y - p.cy, r = Math.hypot(vx, vy), span = p.length / p.radius;
        let rel = p.sweep * (Math.atan2(vy, vx) - p.angle);
        rel -= 2 * Math.PI * Math.floor((rel + Math.PI - span / 2) / (2 * Math.PI));
        const t = Math.min(Math.max(rel, 0), span), theta = p.angle + p.sweep * t;
        d = Math.hypot(x - (p.cx + p.radius * Math.cos(theta)), y - (p.cy + p.radius * Math.sin(theta)));
        s = p.s0 + p.radius * t;
        side = p.sweep * (p.radius - r);
      }
      if (d < best.d) best = { d, s, side };
    });
    const over = Math.max(-best.s, 0) + Math.max(best.s - this.length, 0);
    return { d: Math.hypot(best.side, (over * HALF) / CAP), s: best.s, side: best.side };
  }

  locate(q: N<'vec2'>): Located {
    const bestD = float(1e9).toVar(), bestS = float(0).toVar(), bestSide = float(0).toVar();
    const last = this.pieces.length - 1;
    this.pieces.forEach((p, i) => {
      let d: N<'float'>, s: N<'float'>, side: N<'float'>;
      if (!p.arc) {
        const rel = q.sub(vec2(p.ax, p.ay)), raw = dot(rel, vec2(p.dx, p.dy)), t = clamp(raw, 0, p.length);
        d = length(q.sub(vec2(p.ax, p.ay).add(vec2(p.dx, p.dy).mul(t))));
        const open = i === 0 && i === last ? raw : i === 0 ? min(raw, p.length) : i === last ? max(raw, 0) : t;
        s = open.add(p.s0);
        side = rel.y.mul(p.dx).sub(rel.x.mul(p.dy));
      } else {
        const v = q.sub(vec2(p.cx, p.cy)), r = length(v), span = p.length / p.radius;
        const rel = atan(v.y, v.x).sub(p.angle).mul(p.sweep), wrapped = rel.sub(floor(rel.add(Math.PI - span / 2).div(2 * Math.PI)).mul(2 * Math.PI));
        const t = clamp(wrapped, 0, span), theta = t.mul(p.sweep).add(p.angle);
        d = length(q.sub(vec2(p.cx, p.cy).add(vec2(cos(theta), sin(theta)).mul(p.radius))));
        s = t.mul(p.radius).add(p.s0);
        side = r.negate().add(p.radius).mul(p.sweep);
      }
      const closer = d.lessThan(bestD);
      bestS.assign(select(closer, s, bestS));
      bestSide.assign(select(closer, side, bestSide));
      bestD.assign(min(bestD, d));
    });
    const over = max(bestS.negate(), 0).add(max(bestS.sub(this.length), 0));
    return { s: bestS, side: bestSide, d: length(vec2(bestSide, over.mul(HALF / CAP))).toVar() };
  }
}
