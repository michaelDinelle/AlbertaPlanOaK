import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export interface Palette { agent: number; rival: number }

export const WALL = 0.12;
export const PAD = WALL + 0.03;
export const NOSE = 0.46;
export const RIM = 0.34;
export const CYCLE = 1.8;
export const DIAL = 1.8;
export const DIAL_RADIUS = 0.68;
export const LAP = 16;
export const SHARD = 0.24;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));
const tint = (palette: Palette, owner: N<'float'>) => mix(rgb(palette.agent), rgb(palette.rival), owner);
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

const tapered = (q: N<'vec2'>, from: number, r1: number, r2: number, span: number): N<'float'> => {
  const p = vec2(abs(q.y), q.x.sub(from)), b = (r1 - r2) / span, a = Math.sqrt(1 - b * b), k = dot(p, vec2(-b, a));
  return select(k.lessThan(0), length(p).sub(r1), select(k.greaterThan(a * span), length(p.sub(vec2(0, span))).sub(r2), dot(p, vec2(a, b)).sub(r1)));
};

interface Layer { color: N<'vec3'>; alpha: N<'float'> }

const over = (below: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(below.color.mul(alpha.oneMinus())),
  alpha: alpha.add(below.alpha.mul(alpha.oneMinus())),
});

const settle = (material: THREE.MeshBasicNodeMaterial, layer: Layer, alpha: N<'float'>) => {
  material.colorNode = layer.color.div(max(layer.alpha, 1e-4));
  material.opacityNode = layer.alpha.mul(alpha);
};

export function arenaMaterial(w: number, h: number, claims: THREE.Texture, ghost: Level, palette: Palette, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = positionWorld.xz, g = p.add(vec2(w / 2, h / 2)), cell = floor(g), local = g.sub(cell);
  const seam = min(min(local.x, local.x.oneMinus()), min(local.y, local.y.oneMinus()));
  const edge = sdBox(p, vec2(w / 2, h / 2)), inside = coverage(edge);
  const band = coverage(abs(edge.sub(RIM / 2)).sub(RIM / 2));
  const line = coverage(seam.sub(0.011));
  const far = exp(max(edge, 0).mul(-0.32));
  const mottle = hash(cell).mul(0.1).add(0.95);
  const bevel = smoothstep(0.14, 0, seam).mul(0.35).add(1);
  const tile = rgb(0x172536).mul(mottle).mul(bevel);
  const plate = rgb(0x111a26).mul(smoothstep(0, RIM, edge).mul(-0.35).add(1.1));
  const albedo = mix(mix(rgb(0x04070c), plate, band), tile, inside);

  const at = clamp(cell, vec2(0), vec2(w - 1, h - 1)).add(0.5).div(vec2(w, h)), claim = texture(claims, at);
  const pad = coverage(sdBox(local.sub(0.5), vec2(0.43), 0.12)).mul(inside);
  const owned = rgb(palette.agent).mul(claim.r.add(claim.b.mul(ghost))).add(rgb(palette.rival).mul(claim.g.add(claim.a.mul(ghost))));

  const grid = rgb(0x2a7bb0).mul(line).mul(mix(far.mul(0.09).mul(band.oneMinus()), float(0.3), inside));
  const lip = coverage(abs(edge.sub(0.018)).sub(0.014)), outer = coverage(abs(edge.sub(RIM - 0.025)).sub(0.009));
  const rims = rgb(0xcdf3ff).mul(lip.mul(1.7).add(outer.mul(0.3)));
  material.colorNode = lit(albedo.add(owned.mul(pad).mul(0.08))).add(grid).add(owned.mul(pad).mul(0.05)).add(rims);
  return material;
}

export function wallMaterial(palette: Palette, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const span = attribute<'vec4'>('aSpan', 'vec4'), look = attribute<'vec4'>('aWall', 'vec4');
  const p = positionWorld.xz, a = span.xy, ba = span.zw.sub(a), len = length(ba), dir = ba.div(max(len, 1e-4));
  const t = clamp(dot(p.sub(a), dir), 0, len), d = length(p.sub(a).sub(dir.mul(t)));
  const along = look.x.add(t), owner = look.z, fade = look.w, dead = step(1e-4, fade);
  const color = mix(tint(palette, owner), vec3(0.42, 0.47, 0.55), dead.mul(0.55)).mul(dead.mul(-0.4).add(1));
  const half = fade.mul(-0.45).add(1).mul(WALL), v = clamp(d.div(half), 0, 1);
  const recent = smoothstep(look.y.sub(7), look.y, along);
  const pulse = exp(fract(along.add(time.mul(2.2)).div(3.2)).sub(0.5).div(0.05).pow(2).negate());
  const core = exp(d.div(half.mul(0.2)).pow(2).negate());
  const rim = coverage(abs(d.sub(half.sub(0.022))).sub(0.01));
  const body = color.mul(v.mul(v).mul(-0.45).add(1)).mul(recent.mul(0.4).add(0.72).add(pulse.mul(0.3)));
  const hot = mix(color, vec3(1), 0.7).mul(core.mul(1.7).add(rim.mul(0.35))).mul(dead.mul(-0.6).add(1));
  const crumble = hash(vec2(floor(along.mul(2.2)), owner.mul(17).add(3))).mul(0.6).add(hash(vec2(floor(along.mul(6.5)), owner.mul(29).add(11))).mul(0.4));
  const keep = step(fade.mul(1.2).sub(0.1), crumble);
  material.colorNode = body.add(hot);
  material.opacityNode = coverage(d.sub(half)).mul(keep).mul(fade.mul(fade).oneMinus());
  return material;
}

export function cycleMaterial(palette: Palette, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aRider', 'vec4');
  const owner = state.x, spawn = state.z, heat = state.w;
  const q = uv().sub(0.5).mul(CYCLE), color = tint(palette, owner), white = vec3(3.2, 3.1, 2.9), hot = mix(color, vec3(1), 0.55).mul(2.2);
  const shell = tapered(q, -0.25, 0.165, 0.105, 0.56);
  const front = sdBox(q.sub(vec2(0.37, 0)), vec2(0.1, 0.05), 0.05), rear = sdBox(q.sub(vec2(-0.375, 0)), vec2(0.1, 0.062), 0.062);
  const tyres = min(front, rear), outline = min(shell, tyres);
  const canopy = ellipse(q, [-0.085, 0], [0.17, 0.095]);
  const visor = coverage(abs(ellipse(q, [-0.065, 0], [0.125, 0.064])).sub(0.009)).mul(smoothstep(-0.08, 0.02, q.x));
  const stripe = coverage(abs(shell.add(0.03)).sub(0.011));
  const lamp = length(q.sub(vec2(0.375, 0))).sub(0.03);
  const rubber = mix(mix(lit(rgb(0x05080c)), color.mul(2.4), coverage(abs(tyres.add(0.016)).sub(0.01))), hot, heat.mul(0.6));
  const paint = lit(color.mul(0.34)).add(color.mul(0.2)).mul(q.y.mul(1.3).add(1));
  let body: N<'vec3'> = mix(paint, lit(rgb(0x05090f)).add(rgb(0xbfeaff).mul(visor.mul(0.8))), coverage(canopy));
  body = mix(mix(body, color.mul(2.3), stripe), hot, heat.mul(0.6));

  const halo = exp(length(q.div(vec2(0.6, 0.34))).pow(2).mul(-2.2)).mul(0.5);
  const ring = coverage(abs(length(q).sub(spawn.mul(0.46).add(0.26))).sub(spawn.oneMinus().mul(0.022).add(0.005))).mul(spawn.oneMinus().pow(1.5)).mul(0.8);
  let layer: Layer = { color: vec3(0), alpha: float(0) };
  layer = over(layer, color.mul(0.9), halo);
  layer = over(layer, vec3(0.004, 0.006, 0.01), coverage(outline.sub(0.03)));
  layer = over(layer, rubber, coverage(tyres));
  layer = over(layer, body, coverage(shell));
  layer = over(layer, white, coverage(lamp));
  layer = over(layer, mix(color, vec3(1), 0.3).mul(2.4), ring);
  settle(material, layer, state.y);
  return material;
}

export function shardMaterial(palette: Palette): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aShard', 'vec4');
  const q = uv().sub(0.5).mul(SHARD), heat = state.z;
  const d = sdBox(q, mix(vec2(0.042, 0.042), vec2(0.072, 0.022), state.w), 0.004);
  const color = mix(tint(palette, state.x), vec3(1), heat.mul(0.8)).mul(heat.mul(2.4).add(1.7));
  const glow = exp(max(d, 0).mul(-40)).mul(smoothstep(SHARD / 2, SHARD / 3, length(q))).mul(0.5);
  material.colorNode = color;
  material.opacityNode = max(coverage(d), glow).mul(state.y);
  return material;
}

export interface Dial { material: THREE.MeshBasicNodeMaterial; progress: Level; alpha: Level; lock: Level }

export function dialMaterial(color: number): Dial {
  const material = sheet(), progress = uniform(0), alpha = uniform(0), lock = uniform(0);
  const q = uv().sub(0.5).mul(DIAL), r = length(q);
  const turn = fract(atan(q.x, q.y).div(2 * Math.PI).add(1));
  const laps = floor(progress.div(LAP)), within = progress.sub(laps.mul(LAP));
  const notch = floor(turn.mul(LAP)), inside = fract(turn.mul(LAP));
  const arc = coverage(abs(inside.sub(0.5)).mul((2 * Math.PI * DIAL_RADIUS) / LAP).sub(0.085));
  const band = coverage(abs(r.sub(DIAL_RADIUS)).sub(0.03));
  const done = clamp(within.sub(notch), 0, 1), level = max(done, step(1, laps).mul(0.42));
  const head = within.div(LAP).mul(2 * Math.PI);
  const spark = exp(length(q.sub(vec2(sin(head), cos(head)).mul(DIAL_RADIUS))).mul(-24)).mul(step(0.01, progress));
  const flash = coverage(abs(r.sub(DIAL_RADIUS)).sub(0.07)).mul(lock);
  const hue = rgb(color);
  material.colorNode = mix(hue.mul(level.mul(1.9).add(0.35)), mix(hue, vec3(1), 0.6).mul(3), max(spark, flash));
  material.opacityNode = clamp(arc.mul(band).mul(level.mul(0.85).add(0.15)).add(spark).add(flash), 0, 1).mul(alpha);
  return { material, progress, alpha, lock };
}

export class Strips {
  readonly geometry = new THREE.BufferGeometry();
  private readonly position: THREE.BufferAttribute;
  private used = 0;

  constructor(private readonly capacity: number) {
    const index = new Uint32Array(capacity * 6);
    for (let i = 0; i < capacity; i++) index.set([4 * i, 4 * i + 2, 4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * i + 1], 6 * i);
    this.position = new THREE.BufferAttribute(new Float32Array(capacity * 12), 3).setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('position', this.position);
    this.geometry.setIndex(new THREE.BufferAttribute(index, 1));
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
    this.geometry.setDrawRange(0, 0);
  }

  get count(): number {
    return this.used;
  }

  begin(): void {
    this.used = 0;
  }

  add(ax: number, az: number, bx: number, bz: number, half: number): void {
    if (this.used >= this.capacity) throw new Error(`light strips hold ${this.capacity} runs`);
    const x0 = Math.min(ax, bx) - half, x1 = Math.max(ax, bx) + half, z0 = Math.min(az, bz) - half, z1 = Math.max(az, bz) + half;
    (this.position.array as Float32Array).set([x0, 0, z0, x1, 0, z0, x0, 0, z1, x1, 0, z1], this.used * 12);
    this.used++;
  }

  commit(): void {
    this.geometry.setDrawRange(0, this.used * 6);
    this.position.needsUpdate = true;
  }
}
