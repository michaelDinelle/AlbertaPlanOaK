import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, pow, round, select, smoothstep, texture, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';
import { AGENT, EQUAL, GATE, LAMP, LOG, PAD, PIN, RESISTOR, TRACE, WIRE, along, pathArcs, type Layout, type Point } from './layout';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const BENCH = 0x08070c;
export const SIGNAL = TOKENS.cueCyan;
export const HIT = TOKENS.celebrationGold;
export const MISS = TOKENS.failureBlue;
const MASK = 0x2f1f4d;
const MASK_EDGE = 0x4b3d66;
const SILK = 0xd9d0e8;
const COPPER = 0x9a7462;
const COPPER_EDGE = 0x5a4048;
const GOLD = 0xd8b25c;
const GOLD_DARK = 0x8a6a2c;
const EPOXY = 0x1a1720;
const EPOXY_TOP = 0x2a2533;
const METAL = 0xc4c0cf;
const HOT = 0xf2feff;

export const flat = (width: number, depth: number): THREE.BufferGeometry => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const disc = (radius: number): THREE.BufferGeometry => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);

const sheet = (): THREE.MeshBasicNodeMaterial => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });

const fall = (from: number | N<'float'>, to: number | N<'float'>, x: N<'float'>): N<'float'> => float(1).sub(smoothstep(to, from, x));

const local = (centre: THREE.Vector2): N<'vec2'> => positionWorld.xz.sub(vec2(centre.x, centre.y));

const glowOf = (color: number, on: N<'float'>, core: N<'float'>): N<'vec3'> => mix(rgb(color).mul(0.8), mix(rgb(color), rgb(HOT), 0.22).mul(1.45), core).mul(on);

function ribbon(points: readonly Point[], half: number, arcs?: readonly number[]): THREE.BufferGeometry {
  const count = points.length, position = new Float32Array(count * 6), across = new Float32Array(count * 2), u = new Float32Array(count * 2), index: number[] = [];
  let total = 0;
  const lengths = [0];
  for (let i = 1; i < count; i++) lengths.push((total += Math.hypot(points[i]![0] - points[i - 1]![0], points[i]![1] - points[i - 1]![1])));
  for (let i = 0; i < count; i++) {
    const [x, z] = points[i]!, prev = points[Math.max(0, i - 1)]!, next = points[Math.min(count - 1, i + 1)]!;
    const ax = x - prev[0], az = z - prev[1], bx = next[0] - x, bz = next[1] - z;
    const la = Math.hypot(ax, az) || 1, lb = Math.hypot(bx, bz) || 1;
    const na = i > 0 ? [-az / la, ax / la] : [-bz / lb, bx / lb], nb = i < count - 1 ? [-bz / lb, bx / lb] : na;
    let mx = na[0]! + nb[0]!, mz = na[1]! + nb[1]!;
    const ml = Math.hypot(mx, mz) || 1;
    mx /= ml; mz /= ml;
    const miter = half / Math.max(0.35, mx * na[0]! + mz * na[1]!);
    position.set([x + mx * miter, 0, z + mz * miter, x - mx * miter, 0, z - mz * miter], i * 6);
    across.set([half, -half], i * 2);
    const at = arcs ? arcs[i]! : lengths[i]! / total;
    u.set([at, at], i * 2);
    if (i > 0) index.push(2 * i - 2, 2 * i, 2 * i - 1, 2 * i - 1, 2 * i, 2 * i + 1);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('across', new THREE.BufferAttribute(across, 1));
  geometry.setAttribute('u', new THREE.BufferAttribute(u, 1));
  geometry.setIndex(index);
  return geometry;
}

export function wireGeometry(points: readonly Point[]): THREE.BufferGeometry {
  return ribbon(points, WIRE);
}

export function lineGeometry(layout: Layout, half: number, tail: number): THREE.BufferGeometry {
  const arcs = pathArcs(layout, tail), station = { x: 0, z: 0, tx: 1, tz: 0 };
  return ribbon(arcs.map((arc) => { along(layout, arc, station); return [station.x, station.z] as const; }), half, arcs);
}

export interface LineState { offset: Level; bits: THREE.DataTexture }

function lineBit(layout: Layout, bits: THREE.DataTexture, j: N<'float'>): N<'float'> {
  return select(j.lessThan(0).or(j.greaterThan(layout.cells)), float(0), texture(bits).load(uvec2(vec2(clamp(j, 0, layout.cells), 0))).r);
}

function lineOn(layout: Layout, state: LineState, smooth: boolean): N<'float'> {
  const arc = attribute<'float'>('u', 'float'), t = arc.sub(state.offset), age = floor(t), j = age.add(1), here = lineBit(layout, state.bits, j);
  const shown = smoothstep(-0.02, 0.02, arc);
  if (!smooth) return here.mul(shown);
  const e = t.sub(age), fw = max(fwidth(t), 1e-4);
  const low = clamp(e.div(fw).add(0.5), 0, 1), high = clamp(e.oneMinus().div(fw).add(0.5), 0, 1);
  return mix(lineBit(layout, state.bits, j.sub(1)), mix(lineBit(layout, state.bits, j.add(1)), here, high), low).mul(shown);
}

const crownOf = (x: N<'float'>, half: number): N<'float'> => clamp(float(1).sub(pow(x.div(half), 2)), 0, 1);

const copperOf = (crown: N<'float'>): N<'vec3'> => mix(rgb(COPPER_EDGE), rgb(COPPER), crown.pow(0.7));

export function lineMaterial(layout: Layout, state: LineState, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), s = layout.cell, n = layout.cells;
  const arc = attribute<'float'>('u', 'float'), x = abs(attribute<'float'>('across', 'float'));
  const on = lineOn(layout, state, true), crown = crownOf(x, TRACE);
  const trace = coverage(x.sub(TRACE)), core = exp(x.div(TRACE).pow(2).mul(-2.6));
  const body = mix(lit(copperOf(crown)), glowOf(SIGNAL, float(1), core), on);
  const k = round(arc), major = mod(k, 8).lessThan(0.5);
  const reach = select(major, float(0.17), float(0.1));
  const tick = max(abs(arc.sub(k)).mul(s).sub(0.013), max(float(TRACE + 0.06).sub(x), x.sub(reach.add(TRACE + 0.06))));
  const ticks = coverage(tick).mul(select(k.lessThan(-0.5).or(k.greaterThan(n + 0.5)), float(0), float(1))).mul(0.72);
  const box = (from: number, to: number, grow: number) => abs(sdBox(vec2(arc.sub((from + to) / 2).mul(s), x), vec2(((to - from) * s) / 2 + grow, TRACE + grow), 0.08)).sub(0.013);
  const gate = coverage(box(n - 2, n, 0.12)).mul(0.9);
  const watch = coverage(box(0, 1, 0.1).sub(0.005));
  let color: N<'vec3'> = lit(rgb(SILK));
  let alpha: N<'float'> = ticks;
  color = mix(color, body, trace);
  alpha = max(alpha, trace);
  color = mix(color, rgb(SIGNAL).mul(0.95), gate);
  alpha = max(alpha, gate);
  color = mix(color, rgb(TOKENS.accents.Parity!).mul(1.1), watch);
  alpha = max(alpha, watch);
  material.colorNode = color;
  material.opacityNode = alpha;
  return material;
}

export function lineLight(layout: Layout, state: LineState, strength: number): N<'vec3'> {
  return rgb(SIGNAL).mul(strength).mul(lineOn(layout, state, false));
}

export interface Signal { from: Level; to: Level; front: Level }

function signalOn(signal: Signal): N<'float'> {
  const u = attribute<'float'>('u', 'float'), fw = max(fwidth(u), 1e-5);
  return mix(signal.to, signal.from, clamp(u.sub(signal.front).div(fw).add(0.5), 0, 1));
}

export function wireMaterial(color: number, signal: Signal, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), x = abs(attribute<'float'>('across', 'float')), u = attribute<'float'>('u', 'float');
  const on = signalOn(signal), core = exp(x.div(WIRE).pow(2).mul(-2.2));
  const spark = exp(u.sub(signal.front).div(0.035).pow(2).negate()).mul(abs(signal.to.sub(signal.from))).mul(smoothstep(0, 0.02, signal.front)).mul(fall(1, 0.98, signal.front));
  const shade = mix(lit(copperOf(crownOf(x, WIRE))), glowOf(color, float(1), core), on);
  material.colorNode = shade.add(mix(rgb(color), rgb(HOT), 0.5).mul(spark.mul(2.4)));
  material.opacityNode = coverage(x.sub(WIRE));
  return material;
}

export function wireLight(color: number, signal: Signal, strength: number): N<'vec3'> {
  return rgb(color).mul(strength).mul(signalOn(signal));
}

export function boardMaterial(layout: Layout, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const box = layout.board, centre = box.getCenter(new THREE.Vector2()), half = box.getSize(new THREE.Vector2()).multiplyScalar(0.5);
  const p = positionWorld.xz, q = local(centre), edge = sdBox(q, vec2(half.x, half.y), 0.34), inside = coverage(edge);
  const bench = rgb(BENCH).mul(fractal(p.mul(0.3), 3).mul(0.3).add(0.95));
  const w = p.div(0.15), parity = mod(floor(w.x).add(floor(w.y)), 2), bundle = (t: N<'float'>) => float(1).sub(abs(fract(t).sub(0.5)).mul(2).pow(2).mul(0.7));
  const weave = mix(bundle(w.y), bundle(w.x), parity);
  let albedo: N<'vec3'> = rgb(MASK).mul(weave.mul(0.1).add(0.92)).mul(fractal(p.mul(0.45), 3).mul(0.16).add(1));
  albedo = mix(albedo, rgb(MASK_EDGE), coverage(abs(edge.add(0.045)).sub(0.03)));
  const outline = abs(sdBox(q, vec2(half.x - 0.26, half.y - 0.26), 0.16)).sub(0.012);
  albedo = mix(albedo, rgb(SILK), coverage(outline).mul(0.45));
  const coin = local(layout.coin), rim = abs(sdCircle(coin, 0.22)).sub(0.014), face = sdCircle(coin, 0.15);
  const halfFace = max(face, coin.x.negate());
  albedo = mix(albedo, rgb(SILK), coverage(min(rim, halfFace)).mul(0.62));
  albedo = mix(albedo, rgb(SILK), coverage(abs(face).sub(0.012)).mul(0.62));
  const g = local(layout.ground);
  const bars = min(min(sdSegment(g, vec2(-0.42, 0), vec2(-0.1, 0), 0.014), sdSegment(g, vec2(-0.1, -0.16), vec2(-0.1, 0.16), 0.016)), min(sdSegment(g, vec2(-0.02, -0.1), vec2(-0.02, 0.1), 0.016), sdSegment(g, vec2(0.06, -0.04), vec2(0.06, 0.04), 0.016)));
  albedo = mix(albedo, rgb(SILK), coverage(bars).mul(0.62));
  const r = layout.resistor, outlineR = abs(sdBox(local(r), vec2(RESISTOR.x + 0.07, RESISTOR.y + 0.07), 0.03)).sub(0.01);
  albedo = mix(albedo, rgb(SILK), coverage(outlineR).mul(0.45));
  let holes: N<'float'> = float(1e3);
  let rings: N<'float'> = float(1e3);
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const at = local(new THREE.Vector2(centre.x + sx * (half.x - 0.42), centre.y + sz * (half.y - 0.42)));
    holes = min(holes, sdCircle(at, 0.11));
    rings = min(rings, sdCircle(at, 0.2));
  }
  const gold = mix(rgb(GOLD_DARK), rgb(GOLD), fall(0.2, 0.11, rings.add(0.2)));
  albedo = mix(albedo, gold, coverage(rings));
  const surface = mix(lit(bench), lit(albedo), inside);
  material.colorNode = mix(surface, rgb(BENCH).mul(0.4), coverage(holes).mul(inside));
  return material;
}

export function padMaterial(layout: Layout, glow: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = local(layout.pad), r = length(q);
  const gold = mix(rgb(GOLD_DARK), rgb(GOLD), fall(PAD, 0.09, r)).add(rgb(0xfff2c8).mul(exp(length(q.sub(vec2(-0.07, -0.07))).div(0.05).pow(2).negate()).mul(0.5)));
  const lit1 = mix(lit(gold), glowOf(SIGNAL, float(1), exp(r.div(PAD).pow(2).mul(-1.8))), glow);
  material.colorNode = mix(lit1, rgb(BENCH).mul(0.5), coverage(sdCircle(q, 0.085)));
  material.opacityNode = coverage(sdCircle(q, PAD));
  return material;
}

export function resistorMaterial(layout: Layout, heat: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = local(layout.resistor);
  const body = sdBox(q, vec2(RESISTOR.x, RESISTOR.y), 0.02), caps = max(body, float(RESISTOR.x - 0.07).sub(abs(q.x)));
  const warm = mix(rgb(0xff7a3a), rgb(0xffd2a0), heat).mul(heat.mul(2.2));
  let color: N<'vec3'> = lit(rgb(EPOXY_TOP).mul(q.y.mul(-1.2).add(1))).add(warm.mul(exp(q.x.div(0.16).pow(2).negate())));
  color = mix(color, lit(rgb(METAL).mul(abs(q.y).mul(-2).add(1.05))), coverage(caps));
  material.colorNode = color;
  material.opacityNode = coverage(body);
  return material;
}

export type Glyph = 'xor' | 'equal' | 'agent';

export interface ChipSpec { centre: THREE.Vector2; half: THREE.Vector2; pins: readonly Point[]; glyph: Glyph; glow: Level | null }

export function chipSpecs(layout: Layout, glow: Level): ChipSpec[] {
  const g = GATE, e = EQUAL, a = AGENT, pinsOf = (half: THREE.Vector2, side: 'top' | 'bottom' | 'left' | 'right', at: readonly number[]): Point[] =>
    at.map((v) => side === 'top' ? [v, -half.y - PIN / 2] : side === 'bottom' ? [v, half.y + PIN / 2] : side === 'left' ? [-half.x - PIN / 2, v] : [half.x + PIN / 2, v]);
  const around = (half: THREE.Vector2) => [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5].map((v) => v * (half.x / 0.8));
  return [
    { centre: layout.gate, half: g, glyph: 'xor', glow: null, pins: [...pinsOf(g, 'top', [-0.18, 0.18]), ...pinsOf(g, 'right', [0]), ...pinsOf(g, 'bottom', [-0.18, 0.18]), ...pinsOf(g, 'left', [0])] },
    { centre: layout.equal, half: e, glyph: 'equal', glow: null, pins: [...pinsOf(e, 'top', [0]), ...pinsOf(e, 'left', [0]), ...pinsOf(e, 'right', [0]), ...pinsOf(e, 'bottom', [0])] },
    { centre: layout.agent, half: a, glyph: 'agent', glow, pins: [...pinsOf(a, 'top', around(a)), ...pinsOf(a, 'bottom', around(a)), ...pinsOf(a, 'left', around(a)), ...pinsOf(a, 'right', around(a))] },
  ];
}

export function chipMaterial(spec: ChipSpec, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = local(spec.centre), { half } = spec;
  const body = sdBox(q, vec2(half.x, half.y), 0.045);
  let pins: N<'float'> = float(1e3);
  for (const [px, pz] of spec.pins) {
    const vertical = Math.abs(pz) > half.y;
    pins = min(pins, sdBox(q.sub(vec2(px, pz)), vertical ? vec2(0.032, PIN / 2) : vec2(PIN / 2, 0.032), 0.008));
  }
  const top = mix(rgb(EPOXY), rgb(EPOXY_TOP), fall(half.y, -half.y, q.y).mul(0.8)).mul(smoothstep(0, 0.07, body.negate()).mul(0.35).add(0.75));
  let face: N<'vec3'> = lit(top);
  const silk = rgb(SILK).mul(0.8);
  const dot = sdCircle(q.sub(vec2(-half.x + 0.1, -half.y + 0.1)), 0.035);
  face = mix(face, lit(silk), coverage(dot).mul(0.7));
  if (spec.glyph === 'xor') {
    const g = q.sub(vec2(0.04, 0));
    const mark = min(abs(sdCircle(g, 0.14)).sub(0.017), min(sdSegment(g, vec2(-0.1, 0), vec2(0.1, 0), 0.017), sdSegment(g, vec2(0, -0.1), vec2(0, 0.1), 0.017)));
    face = mix(face, silk, coverage(mark).mul(0.85));
  } else if (spec.glyph === 'equal') {
    const mark = min(sdSegment(q, vec2(-0.1, -0.05), vec2(0.1, -0.05), 0.018), sdSegment(q, vec2(-0.1, 0.05), vec2(0.1, 0.05), 0.018));
    face = mix(face, silk, coverage(mark).mul(0.85));
  } else if (spec.glow) {
    const accent = rgb(TOKENS.accents.Parity!), glow = spec.glow;
    const window = sdBox(q, vec2(0.44, 0.44), 0.1), grid = q.div(0.22).add(0.5), cell = fract(grid).sub(0.5), node = length(cell).mul(0.22).sub(0.048);
    const inside = coverage(window), lattice = coverage(node).mul(step3(grid));
    const glass = rgb(0x221a33).mul(fall(0.44, -0.26, q.y).mul(0.5).add(0.7));
    face = mix(face, lit(glass).add(accent.mul(glow.mul(0.35).add(0.05))), inside);
    face = mix(face, accent.mul(glow.mul(2.2).add(0.45)), lattice.mul(inside));
    face = mix(face, accent.mul(0.8), coverage(abs(window).sub(0.012)).mul(0.9));
  }
  material.colorNode = mix(lit(rgb(METAL).mul(0.9)), face, coverage(body));
  material.opacityNode = max(coverage(body), coverage(pins));
  return material;
}

function step3(grid: N<'vec2'>): N<'float'> {
  const g = floor(grid);
  return select(g.x.greaterThanEqual(-1).and(g.x.lessThanEqual(1)).and(g.y.greaterThanEqual(-1)).and(g.y.lessThanEqual(1)), float(1), float(0));
}

export function ledMaterial(centre: THREE.Vector2, radius: number, color: number, level: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = local(centre), r = length(q);
  const rim = sdCircle(q, radius), lens = sdCircle(q, radius * 0.76);
  const off = lit(rgb(color).mul(0.16)).add(rgb(0xffffff).mul(exp(length(q.sub(vec2(-radius * 0.3, -radius * 0.3))).div(radius * 0.18).pow(2).negate()).mul(0.35)));
  const on = glowOf(color, float(1), exp(r.div(radius * 0.76).pow(2).mul(-2)));
  const housing = lit(rgb(0x3b3447).mul(fall(radius, radius * 0.76, r).mul(0.5).add(0.7)));
  material.colorNode = mix(housing, mix(off, on, level), coverage(lens));
  material.opacityNode = coverage(rim);
  return material;
}

export interface VerdictLevels { hit: Level; shown: Level; flash: Level }

export function verdictMaterial(layout: Layout, levels: VerdictLevels, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = local(layout.verdict), r = length(q);
  const rim = sdCircle(q, LAMP), lens = sdCircle(q, LAMP * 0.8), ring = abs(sdCircle(q, LAMP + 0.12)).sub(0.012);
  const tone = mix(rgb(MISS), rgb(HIT), levels.hit), strength = mix(float(0.42), float(1), levels.hit).mul(levels.flash.mul(0.45).add(1));
  const on = mix(tone.mul(0.9), mix(tone, rgb(HOT), mix(float(0.05), float(0.3), levels.hit)).mul(1.9), exp(r.div(LAMP * 0.8).pow(2).mul(-2))).mul(strength);
  const off = lit(rgb(0x2a2433)).add(rgb(0xffffff).mul(exp(length(q.sub(vec2(-0.08, -0.08))).div(0.05).pow(2).negate()).mul(0.3)));
  const housing = lit(rgb(0x3b3447).mul(fall(LAMP, LAMP * 0.8, r).mul(0.5).add(0.7)));
  const lamp = mix(housing, mix(off, on, levels.shown), coverage(lens));
  material.colorNode = mix(lit(rgb(SILK)), lamp, coverage(rim));
  material.opacityNode = max(coverage(rim), coverage(ring).mul(0.45));
  return material;
}

export function verdictLight(levels: VerdictLevels, strength: number): N<'vec3'> {
  return mix(rgb(MISS).mul(0.3), rgb(HIT), levels.hit).mul(levels.shown).mul(levels.flash.mul(0.6).add(1)).mul(strength);
}

export interface LogState { verdicts: THREE.DataTexture; shift: Level }

export function logMaterial(layout: Layout, state: LogState, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), { left, right, z, pitch } = layout.log, p = positionWorld.xz;
  const c = float(right).sub(p.x).div(pitch).sub(state.shift), k = round(c), along = c.sub(k).mul(pitch);
  const valid = select(k.lessThan(-0.5).or(k.greaterThan(LOG - 0.5)), float(0), float(1));
  const verdict = texture(state.verdicts).load(uvec2(vec2(clamp(k, 0, LOG - 1), 0))).r;
  const cell = sdBox(vec2(along, p.y.sub(z)), vec2(pitch * 0.32, 0.075), 0.02), socket = sdBox(vec2(along, p.y.sub(z)), vec2(pitch * 0.42, 0.1), 0.025);
  const fade = fall(LOG - 0.5, LOG - 12, c);
  const hit = verdict.equal(1), miss = verdict.equal(2);
  const glow = select(hit, rgb(HIT).mul(1.25), select(miss, rgb(MISS).mul(0.55), vec3(0)));
  const cellColor = select(hit.or(miss), glow, lit(rgb(0x1c1726)));
  const rail = abs(sdBox(vec2(p.x.sub((left + right) / 2), p.y.sub(z)), vec2((right - left) / 2 + pitch * 0.2, 0.15), 0.06)).sub(0.01);
  let color: N<'vec3'> = lit(rgb(SILK));
  let alpha: N<'float'> = coverage(rail).mul(0.35);
  const inSocket = coverage(socket).mul(valid);
  color = mix(color, lit(rgb(0x120f18)), inSocket);
  alpha = max(alpha, inSocket.mul(fade).mul(0.9));
  const inCell = coverage(cell).mul(valid);
  color = mix(color, cellColor, inCell);
  alpha = max(alpha, inCell.mul(fade));
  material.colorNode = color;
  material.opacityNode = alpha;
  return material;
}

export const logBounds = (layout: Layout): { centre: THREE.Vector2; size: THREE.Vector2 } => {
  const { left, right, z, pitch } = layout.log;
  return { centre: new THREE.Vector2((left + right) / 2, z), size: new THREE.Vector2(right - left + pitch * 2 + 0.2, 0.5) };
};

