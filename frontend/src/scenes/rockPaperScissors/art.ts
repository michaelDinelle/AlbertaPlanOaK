import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, fwidth, int, length, max, min, mix, mod, positionWorld, pow, select, sin, smoothstep, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Scalar = THREE.UniformNode<'float', number>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

export const TAU = 2 * Math.PI;
export const DIAL_X = 0;
export const DIAL_Y = 1.95;
export const R_CASE = 2.42;
export const R_BEZEL = 2.25;
export const R_TILE_OUT = 2.15;
export const R_TILE_IN = 1.73;
export const R_TILE = (R_TILE_OUT + R_TILE_IN) / 2;
export const R_FACE = 1.62;
export const R_NODE = 0.98;
export const BADGE = 0.43;
export const CLASH_X = 0;
export const CLASH_Y = -1.55;
export const WRIST = 2.95;
export const BEADS_Y = -3.05;
export const RECORD = 24;
const BEAD_STEP = 0.235;
const BEADS_X = -((RECORD - 1) / 2) * BEAD_STEP;
const BEAD = 0.088;
const GAP = 0.016;
const ARROW_CLEAR = 0.62;
const HEAD = 0.16;

export const MOVE_NAMES = ['rock', 'paper', 'scissors'] as const;
export const MOVE_FILL = [0x868b90, 0xe3d9c1, 0x46a57d] as const;
const MOVE_INK = [0xf1f3f6, 0x4a3b2c, 0xeef9f2] as const;
export const MOVE_GLOW = [0xc2cee2, 0xfff3dc, 0x7ff0bd] as const;
const NODE_ANGLE = [0, (2 * TAU) / 3, TAU / 3] as const;
export const WIN = TOKENS.celebrationGold;
export const LOSS = TOKENS.failureBlue;
export const TIE = 0xc9c0ae;
export const BOT = 0x9fe6ff;
const BRASS = 0xc9a45c;
const BRASS_DEEP = 0x5e4a26;
const FACE = 0x131a2b;
const CHANNEL = 0x06080d;
const ENAMEL = 0x0d1220;
const SOCKET = 0x0a0d15;
const STEEL = 0xdfe6e2;
const DIGIT = 0xffc27a;
const WOOD_DARK = 0x2e1c11;
const WOOD_LIGHT = 0x5e3d25;
const PLANK = 1.15;
const LIGHT = vec2(-0.55, 0.83).normalize();

export const DIGIT_MASKS = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f] as const;
const SEGMENTS: readonly (readonly [number, number, number, number])[] = [
  [-0.5, 1, 0.5, 1], [0.5, 1, 0.5, 0], [0.5, 0, 0.5, -1], [-0.5, -1, 0.5, -1], [-0.5, 0, -0.5, -1], [-0.5, 1, -0.5, 0], [-0.5, 0, 0.5, 0],
];

export const nodeX = (move: number): number => DIAL_X + R_NODE * Math.sin(NODE_ANGLE[move]!);
export const nodeY = (move: number): number => DIAL_Y + R_NODE * Math.cos(NODE_ANGLE[move]!);
const slotAngle = (index: number, slots: number): number => (index * TAU) / slots;
export const slotX = (index: number, slots: number): number => DIAL_X + R_TILE * Math.sin(slotAngle(index, slots));
export const slotY = (index: number, slots: number): number => DIAL_Y + R_TILE * Math.cos(slotAngle(index, slots));
const glyphs = (slots: number): boolean => (TAU * R_TILE) / slots > 0.5;

export function slotsOf(count: number) {
  const values = new Array<number>(count).fill(-1);
  return { node: uniformArray(values, 'float' as const), values };
}
export type Slots = ReturnType<typeof slotsOf>;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);

const board = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (width: number, height: number): N<'vec2'> => uv().sub(0.5).mul(vec2(width, height));
const pixel = fwidth(positionWorld.x);
const aa = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel)), 0, 1);
const soft = (d: N<'float'>, blur: number): N<'float'> => smoothstep(-0.04, blur, d).oneMinus();
const facing = (p: N<'vec2'>): N<'float'> => dot(p.div(max(length(p), 1e-4)), LIGHT);
const pick3 = <T extends 'float' | 'vec3'>(move: N<'float'>, a: N<T>, b: N<T>, c: N<T>): N<T> => select(move.lessThan(0.5), a, select(move.lessThan(1.5), b, c)) as N<T>;

function sprite(geometry: THREE.BufferGeometry, order: number, color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

const taper = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, r0: number, r1: number): N<'float'> => {
  const pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(dot(ba, ba)), 0, 1);
  return length(pa.sub(ba.mul(h))).sub(mix(float(r0), float(r1), h));
};

const BOULDER = [[-0.8, -0.34], [-0.58, 0.34], [-0.1, 0.62], [0.5, 0.48], [0.82, -0.08], [0.52, -0.56], [-0.34, -0.62]] as const;
const RIDGE = [[-0.58, 0.34], [0.04, 0.08], [0.5, 0.48], [0.04, 0.08], [0.14, -0.6]] as const;

function boulder(p: N<'vec2'>): N<'float'> {
  let d: N<'float'> = float(-1e3);
  BOULDER.forEach(([ax, ay], k) => {
    const [bx, by] = BOULDER[(k + 1) % BOULDER.length]!, ex = bx - ax, ey = by - ay, l = Math.hypot(ex, ey);
    d = max(d, dot(p.sub(vec2(ax, ay)), vec2(-ey / l, ex / l)));
  });
  return d.add(0.02);
}

const side = (p: N<'vec2'>, a: readonly [number, number], b: readonly [number, number]): N<'float'> => dot(p.sub(vec2(a[0], a[1])), vec2(b[1] - a[1], a[0] - b[0]));

function facet(p: N<'vec2'>): N<'float'> {
  const top = side(p, RIDGE[0], RIDGE[1]).lessThan(0).and(side(p, RIDGE[1], RIDGE[2]).lessThan(0));
  const left = side(p, RIDGE[3], RIDGE[4]).greaterThan(0);
  return select(top, float(1.18), select(left, float(0.92), float(0.7)));
}

const ridges = (p: N<'vec2'>): N<'float'> =>
  min(min(sdSegment(p, vec2(RIDGE[0][0], RIDGE[0][1]), vec2(RIDGE[1][0], RIDGE[1][1]), 0.04), sdSegment(p, vec2(RIDGE[1][0], RIDGE[1][1]), vec2(RIDGE[2][0], RIDGE[2][1]), 0.04)), sdSegment(p, vec2(RIDGE[3][0], RIDGE[3][1]), vec2(RIDGE[4][0], RIDGE[4][1]), 0.04));

const FOLD = 0.3;
const CORNER = vec2(0.56, 0.74);
const DIAGONAL = vec2(Math.SQRT1_2, Math.SQRT1_2);
const ACROSS = vec2(Math.SQRT1_2, -Math.SQRT1_2);

const sheet = (p: N<'vec2'>): N<'float'> => max(sdBox(p, vec2(0.56, 0.74), 0.06), dot(p.sub(CORNER), DIAGONAL).add(FOLD));

function flap(p: N<'vec2'>): N<'float'> {
  const c = p.sub(CORNER), s = dot(c, DIAGONAL).negate().sub(FOLD);
  return max(s.negate(), abs(dot(c, ACROSS)).add(s).sub(FOLD).mul(Math.SQRT1_2));
}

const lines = (p: N<'vec2'>): N<'float'> =>
  min(min(sdSegment(p, vec2(-0.34, 0.36), vec2(0.02, 0.36), 0.045), sdSegment(p, vec2(-0.34, 0.1), vec2(0.3, 0.1), 0.045)), min(sdSegment(p, vec2(-0.34, -0.16), vec2(0.3, -0.16), 0.045), sdSegment(p, vec2(-0.34, -0.42), vec2(0.12, -0.42), 0.045)));

const blades = (p: N<'vec2'>): N<'float'> => min(taper(p, vec2(0.21, -0.2), vec2(-0.3, 0.86), 0.12, 0.035), taper(p, vec2(-0.21, -0.2), vec2(0.3, 0.86), 0.12, 0.035));
const bows = (p: N<'vec2'>): N<'float'> => min(abs(length(p.sub(vec2(0.33, -0.5))).sub(0.2)).sub(0.08), abs(length(p.sub(vec2(-0.33, -0.5))).sub(0.2)).sub(0.08));
const pivot = (p: N<'vec2'>): N<'float'> => abs(length(p.sub(vec2(0, 0.08))).sub(0.05)).sub(0.03);

function glyphInk(move: N<'float'>, p: N<'vec2'>): { body: N<'float'>; detail: N<'float'> } {
  const body = pick3<'float'>(move, boulder(p), sheet(p), min(blades(p), bows(p)));
  const detail = pick3<'float'>(move, ridges(p), min(lines(p), abs(flap(p)).sub(0.03)), pivot(p));
  return { body, detail };
}

function glyphColor(move: number, p: N<'vec2'>, scale: number): { color: N<'vec3'>; body: N<'float'> } {
  const cov = (d: N<'float'>) => aa(d.mul(scale));
  if (move === 0) {
    const body = boulder(p), grit = noise(p.mul(9)).mul(0.06).add(1);
    const tone = rgb(0x9aa2ae).mul(facet(p)).mul(grit);
    const color = mix(mix(tone, rgb(0x3c424c), cov(ridges(p)).mul(0.8)), rgb(0x1c2027), cov(abs(body).sub(0.035)));
    return { color, body };
  }
  if (move === 1) {
    const body = sheet(p);
    let color: N<'vec3'> = rgb(MOVE_FILL[1]).mul(smoothstep(-1, 1, p.y.sub(p.x.mul(0.3))).mul(0.14).add(0.9));
    color = mix(color, rgb(0xbfb193), cov(flap(p)));
    color = mix(color, rgb(0x7c6c55), cov(lines(p)));
    color = mix(color, rgb(0x6a5a44), cov(abs(flap(p)).sub(0.028)));
    color = mix(color, rgb(0x3b2f22), cov(abs(body).sub(0.035)));
    return { color, body };
  }
  const blade = blades(p), bow = bows(p), body = min(blade, bow);
  let color: N<'vec3'> = mix(rgb(MOVE_FILL[2]).mul(smoothstep(-0.8, -0.2, p.y).mul(0.2).add(0.85)), rgb(STEEL).mul(smoothstep(-0.3, 0.6, p.x.negate().add(p.y)).mul(0.25).add(0.78)), cov(blade));
  color = mix(color, rgb(0x16261f), cov(pivot(p)));
  color = mix(color, rgb(0x0f1f18), cov(abs(body).sub(0.035)));
  return { color, body };
}

export function table(lit: Lit): THREE.Mesh {
  const q = board(), yy = q.y.div(PLANK), row = floor(yy), fy = fract(yy);
  const tone = noise(vec2(row.mul(1.37).add(0.31), 7.13)).mul(0.5);
  const grain = fractal(vec2(q.x.mul(0.2).add(row.mul(5.3)), q.y.mul(4.4)), 4);
  const fine = noise(vec2(q.x.mul(1.6), q.y.mul(40)));
  const figure = sin(q.y.mul(26).add(fractal(q.mul(0.45).add(row), 3).mul(9))).mul(0.5).add(0.5);
  let wood: N<'vec3'> = mix(rgb(WOOD_DARK), rgb(WOOD_LIGHT), clamp(grain.mul(0.9).add(0.45).add(fine.mul(0.12)).add(tone.mul(0.35)).add(figure.mul(0.12)), 0, 1));
  wood = mix(wood, rgb(0x080403), aa(min(fy, fy.oneMinus()).mul(PLANK).sub(0.007)).mul(0.85));
  const dial = soft(sdCircle(q.sub(vec2(DIAL_X + 0.14, DIAL_Y - 0.2)), R_CASE), 0.55).mul(0.7);
  const rail = soft(sdBox(q.sub(vec2(0.04, BEADS_Y - 0.06)), vec2((RECORD / 2 + 0.6) * BEAD_STEP, 0.05), 0.05), 0.14).mul(0.45);
  const vignette = smoothstep(3.5, 11, length(q.sub(vec2(0, 0.4)).mul(vec2(0.8, 1.2)))).mul(0.6).oneMinus();
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(wood.mul(max(dial, rail).oneMinus()).mul(vignette));
  const mesh = new THREE.Mesh(flat(80, 50), material);
  mesh.renderOrder = 0;
  mesh.frustumCulled = false;
  return mesh;
}

export interface DialState {
  slots: Slots;
  arrows: Slots;
  newest: Scalar;
  flash: Scalar;
  fading: Scalar;
  fade: Scalar;
  pointer: Scalar;
}

function triangle(p: N<'vec2'>, apex: number, base: number, half: number): N<'float'> {
  const depth = base - apex, slant = Math.hypot(depth, half);
  return max(p.y.sub(base), abs(p.x).mul(depth).sub(p.y.sub(apex).mul(half)).div(slant));
}

function arrowSdf(q: N<'vec2'>, r: N<'float'>, angle: N<'float'>, target: number): N<'float'> {
  const theta = NODE_ANGLE[target]!, b = mod(angle.sub(theta).add(Math.PI), TAU).sub(Math.PI);
  const from = -TAU / 3 + ARROW_CLEAR, to = -ARROW_CLEAR - HEAD / R_NODE / 2;
  const band = max(abs(r.sub(R_NODE)).sub(0.02), max(float(from).sub(b), b.sub(to)).mul(R_NODE));
  const tip = theta - ARROW_CLEAR, tx = R_NODE * Math.sin(tip), ty = R_NODE * Math.cos(tip);
  const dx = Math.cos(tip), dy = -Math.sin(tip), c = q.sub(vec2(tx, ty));
  const head = triangle(vec2(dot(c, vec2(-dy, dx)), dot(c, vec2(dx, dy)).negate()), 0, HEAD, 0.085);
  return min(band, head);
}

export function dial(lit: Lit, count: number, accent: number, state: DialState): THREE.Mesh {
  const q = board().sub(vec2(DIAL_X, DIAL_Y)), r = length(q), light = facing(q), angle = mod(atan(q.x, q.y).add(TAU), TAU);

  const s = angle.div(TAU).mul(count), nearest = floor(s.add(0.5)), k = mod(nearest, count), off = s.sub(nearest);
  const arc = off.mul(TAU / count).mul(r), rad = r.sub(R_TILE), halfRad = (R_TILE_OUT - R_TILE_IN) / 2;
  const half = r.mul(Math.PI / count).sub(GAP / 2);
  const tile = sdBox(vec2(arc, rad), vec2(half, halfRad), min(float(0.05), half.mul(0.6)));
  const move = state.slots.node.element(int(k)), empty = move.lessThan(-0.5);
  const isNewest = abs(k.sub(state.newest)).lessThan(0.5), isFading = abs(k.sub(state.fading)).lessThan(0.5);

  const across = clamp(r.sub(R_BEZEL).div(R_CASE - R_BEZEL), 0, 1);
  const brass = rgb(BRASS).mul(cos(across.mul(Math.PI)).negate().mul(light).mul(0.4).add(0.82)).mul(sin(r.mul(290)).mul(0.02).add(1));
  const guilloche = sin(angle.mul(48).add(sin(r.mul(13)).mul(1.3))).mul(0.5).add(0.5);
  let albedo: N<'vec3'> = rgb(FACE).mul(guilloche.mul(0.1).add(0.95)).mul(noise(q.mul(9)).mul(0.05).add(1));
  albedo = albedo.mul(smoothstep(R_FACE - 0.3, R_FACE, r).mul(0.3).oneMinus());

  let glow: N<'vec3'> = vec3(0);
  for (let target = 0; target < 3; target++) {
    const d = arrowSdf(q, r, angle, target), hot = state.arrows.node.element(target);
    albedo = mix(albedo, mix(rgb(0x8a7040), rgb(0x2a3552), clamp(hot, 0, 1)), aa(d));
    albedo = mix(albedo, rgb(0x2a1f10), aa(abs(d).sub(0.006)).mul(0.7).mul(clamp(hot, 0, 1).oneMinus()));
    glow = glow.add(rgb(BOT).mul(hot).mul(aa(d).mul(1.4).add(exp(max(d, 0).mul(-26)).mul(0.5))));
  }

  const hub = aa(r.sub(0.13));
  albedo = mix(albedo, rgb(BRASS).mul(pow(clamp(r.div(0.13).oneMinus(), 0, 1), 0.5).mul(0.3).add(light.mul(0.2)).add(0.62)), hub);
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(r.sub(0.13)).sub(0.01)));

  albedo = mix(albedo, rgb(CHANNEL), aa(float(R_FACE).sub(r)));
  albedo = mix(albedo, rgb(BRASS).mul(light.mul(0.3).add(0.8)), aa(abs(r.sub(R_FACE)).sub(0.022)));

  const socket = mix(rgb(SOCKET), rgb(0x232b3f), aa(abs(tile.add(0.012)).sub(0.01)).mul(0.8));
  const fill = pick3<'vec3'>(move, rgb(MOVE_FILL[0]), rgb(MOVE_FILL[1]), rgb(MOVE_FILL[2]));
  const bevel = smoothstep(0, 0.045, tile.negate());
  let enamel: N<'vec3'> = fill.mul(mix(float(0.66), float(1.02), bevel)).mul(rad.div(halfRad).mul(0.08).add(1));
  if (glyphs(count)) {
    const centre = vec2(sin(k.mul(TAU / count)), cos(k.mul(TAU / count))).mul(R_TILE), scale = 0.17, gp = q.sub(centre).div(scale);
    const ink = glyphInk(move, gp), inkColor = pick3<'vec3'>(move, rgb(MOVE_INK[0]), rgb(MOVE_INK[1]), rgb(MOVE_INK[2]));
    enamel = mix(enamel, inkColor, aa(ink.body.mul(scale)).mul(aa(ink.detail.mul(scale).negate())));
  }
  enamel = mix(enamel, vec3(1), select(isNewest, state.flash, float(0)).mul(0.55));
  const shown = select(isFading, state.fade.oneMinus(), float(1));
  const tileColor = select(empty, socket, mix(socket, enamel, shown));
  albedo = mix(albedo, tileColor, aa(tile).mul(aa(r.sub(R_FACE + 0.03).negate())));

  const tickDistance = float(0.5).sub(abs(off)).mul(TAU / count).mul(r);
  const inBezel = aa(r.sub(R_BEZEL + 0.035).negate()).mul(aa(r.sub(R_CASE - 0.05)));
  const major = count > 24 ? mod(nearest.add(count), count / 10).lessThan(0.5) : float(1).greaterThan(0);
  albedo = mix(albedo, brass, aa(float(R_BEZEL).sub(r)));
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(tickDistance.sub(0.006)).mul(inBezel).mul(select(major, float(0.85), float(0.45))));
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(r.sub(R_BEZEL)).sub(0.012)).mul(0.8));
  albedo = mix(albedo, rgb(0x2a1f10), aa(abs(r.sub(R_CASE - 0.012)).sub(0.012)).mul(0.7));

  const pc = state.pointer, er = vec2(sin(pc), cos(pc)), et = vec2(cos(pc), sin(pc).negate());
  const pl = vec2(dot(q, et), dot(q, er)), pointer = triangle(pl, R_TILE_OUT + 0.02, R_CASE - 0.04, 0.12);
  albedo = mix(albedo, rgb(0xf0d48c).mul(facing(pl.sub(vec2(0, R_BEZEL))).mul(0.25).add(0.85)), aa(pointer));
  albedo = mix(albedo, rgb(0x3a2a12), aa(abs(pointer).sub(0.008)).mul(0.8));

  const rim = aa(abs(tile.add(0.016)).sub(0.012)).add(exp(max(tile, 0).mul(-60)).mul(0.3).mul(aa(tile.negate()))).mul(aa(r.sub(R_FACE + 0.03).negate()));
  glow = glow.add(rgb(accent).mul(select(isNewest.and(empty.not()), rim, float(0))).mul(state.flash.mul(1.4).add(0.75)));

  return sprite(flat(2 * R_CASE + 0.1, 2 * R_CASE + 0.1), 2, lit(albedo).add(glow), aa(r.sub(R_CASE)));
}

export interface BadgeState {
  masks: Scalar[];
  fill: Scalar;
  read: Scalar;
  pick: Scalar;
  flash: Scalar;
}

function sevenSegment(p: N<'vec2'>, mask: N<'float'>, width: number, height: number, thick: number): { lit: N<'float'>; ghost: N<'float'> } {
  let lit: N<'float'> = float(0), ghost: N<'float'> = float(1e3);
  const gap = thick * 1.6;
  SEGMENTS.forEach(([ax, ay, bx, by], bit) => {
    const sx = Math.sign(bx - ax) * gap, sy = Math.sign(by - ay) * gap;
    const d = sdSegment(p, vec2(ax * width + sx, (ay * height) / 2 + sy), vec2(bx * width - sx, (by * height) / 2 - sy), thick);
    ghost = min(ghost, d);
    lit = max(lit, aa(d).mul(mod(floor(mask.div(2 ** bit)), 2)));
  });
  return { lit, ghost: aa(ghost) };
}

export function badge(lit: Lit, move: number, digits: number, accent: number, state: BadgeState): { body: THREE.Mesh; halo: THREE.Mesh } {
  const size = 2 * (BADGE + 0.3), p = local(size, size), r = length(p), light = facing(p), angle = mod(atan(p.x, p.y).add(TAU), TAU);
  const inner = BADGE - 0.05;
  let albedo: N<'vec3'> = rgb(ENAMEL).mul(smoothstep(inner, 0, r).mul(0.35).add(0.8));
  const glyph = glyphColor(move, p.sub(vec2(0, 0.11)).div(0.21), 0.21), glyphMask = aa(glyph.body.mul(0.21));
  albedo = mix(albedo, glyph.color, glyphMask);
  let digitGlow: N<'float'> = float(0), ghost: N<'float'> = float(0);
  for (let j = 0; j < digits; j++) {
    const x = (j - (digits - 1) / 2) * 0.105;
    const seg = sevenSegment(p.sub(vec2(x + 0.014, -0.215)).sub(vec2(p.y.add(0.215).mul(0.12), 0)), state.masks[j]!, 0.072, 0.15, 0.013);
    digitGlow = max(digitGlow, seg.lit);
    ghost = max(ghost, seg.ghost);
  }
  albedo = mix(albedo, rgb(0x2a1a10), ghost.mul(0.5));
  albedo = mix(albedo, rgb(BRASS).mul(cos(clamp(r.sub(inner).div(BADGE - inner), 0, 1).mul(Math.PI)).negate().mul(light).mul(0.4).add(0.82)), aa(float(inner).sub(r)));
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(r.sub(inner)).sub(0.008)));
  const track = abs(r.sub(BADGE + 0.06)).sub(0.022);
  albedo = mix(albedo, rgb(CHANNEL), aa(track.sub(0.008)));
  const filled = aa(track).mul(clamp(state.fill.mul(TAU).sub(angle).mul(r).div(pixel).add(0.5), 0, 1));
  const tint = rgb(MOVE_GLOW[move]!);
  const emissive = tint.mul(filled.mul(0.9)).add(rgb(DIGIT).mul(digitGlow.mul(1.6))).add(vec3(state.flash.mul(aa(r.sub(inner)).mul(0.25)))).add(glyph.color.mul(glyphMask.mul(0.28)));
  const body = sprite(flat(size, size), 5, lit(albedo).add(emissive), max(aa(r.sub(BADGE)), aa(track.sub(0.008))));
  const out = max(r.sub(BADGE + 0.1), 0), ring = aa(abs(r.sub(BADGE + 0.12)).sub(0.014)), outside = smoothstep(BADGE + 0.07, BADGE + 0.11, r);
  const halo = rgb(accent).mul(state.read).mul(exp(out.mul(-22)).mul(0.5).mul(outside).add(ring.mul(1.1))).add(rgb(BOT).mul(state.pick).mul(exp(out.mul(-18)).mul(0.55).mul(outside).add(ring.mul(1.1))));
  return { body, halo: sprite(flat(size, size), 4, halo.mul(smoothstep(size / 2, size / 2 - 0.08, r)), float(1), true) };
}

export interface RailState {
  values: Slots;
  slide: Scalar;
  latest: Scalar;
  fresh: Scalar;
}

export function rail(lit: Lit, state: RailState): THREE.Mesh {
  const width = (RECORD + 2) * BEAD_STEP, height = 0.5, p = local(width, height), x = p.x.sub(BEADS_X);
  const reach = (RECORD / 2 + 0.55) * BEAD_STEP;
  const rod = sdBox(p, vec2(reach, 0.018), 0.018);
  const caps = min(sdCircle(p.sub(vec2(-reach, 0)), 0.05), sdCircle(p.sub(vec2(reach, 0)), 0.05));
  let albedo: N<'vec3'> = rgb(BRASS).mul(smoothstep(-0.02, 0.02, p.y).mul(0.35).add(0.7));
  albedo = mix(albedo, rgb(BRASS).mul(facing(p.sub(vec2(p.x.sign().mul(reach), 0))).mul(0.3).add(0.8)), aa(caps));
  const j = clamp(floor(x.div(BEAD_STEP).add(state.slide).add(0.5)), 0, RECORD), cx = j.sub(state.slide).mul(BEAD_STEP);
  const value = state.values.node.element(int(j)), present = value.greaterThan(-0.5);
  const b = vec2(x.sub(cx), p.y), d = length(b).sub(BEAD), u = clamp(length(b).div(BEAD), 0, 1);
  const tint = pick3<'vec3'>(value, rgb(WIN), rgb(LOSS), rgb(TIE));
  const sphere = tint.mul(u.mul(-0.45).add(1.12)).add(vec3(exp(length(b.div(BEAD).sub(vec2(-0.35, 0.4))).mul(-5)).mul(0.55)));
  const position = j.sub(state.slide), edge = smoothstep(-0.9, 0, position).mul(smoothstep(RECORD, RECORD - 0.9, position));
  const bead = aa(d).mul(select(present, edge, float(0)));
  albedo = mix(albedo, sphere, bead);
  const fresh = select(abs(j.sub(state.latest)).lessThan(0.5).and(present), state.fresh, float(0));
  const halo = exp(max(d, 0).mul(-20)).mul(fresh).mul(edge);
  const win = select(value.lessThan(0.5).and(present), float(0.3), float(0));
  const glow = rgb(WIN).mul(win).mul(bead).add(tint.mul(halo.mul(0.8)));
  return sprite(flat(width, height), 3, lit(albedo).add(glow), max(aa(min(rod, caps)), max(bead, halo.mul(0.6))));
}

export function flash(tint: Tint, strength: Scalar): THREE.Mesh {
  const p = local(4, 4), r = length(p), a = atan(p.y, p.x);
  const star = exp(r.mul(-4.5)).mul(pow(abs(cos(a.mul(2))), 16).mul(1.1).add(pow(abs(cos(a.mul(2).add(Math.PI / 2))), 30).mul(0.35)).add(0.18));
  const core = exp(r.mul(-16)).mul(1.8).add(exp(r.mul(-2.6)).mul(0.22));
  return sprite(flat(4, 4), 9, tint.mul(star.add(core)).mul(strength).mul(smoothstep(2, 1.6, r)), float(1), true);
}

export function token(move: Scalar): THREE.Mesh {
  const p = local(1, 1), tile = sdBox(p, vec2(0.36, 0.36), 0.09);
  const fill = pick3<'vec3'>(move, rgb(MOVE_FILL[0]), rgb(MOVE_FILL[1]), rgb(MOVE_FILL[2])).mul(smoothstep(0, 0.07, tile.negate()).mul(0.35).add(0.7));
  const ink = glyphInk(move, p.div(0.3)), inkColor = pick3<'vec3'>(move, rgb(MOVE_INK[0]), rgb(MOVE_INK[1]), rgb(MOVE_INK[2]));
  let color: N<'vec3'> = mix(fill, inkColor, coverage(ink.body).mul(coverage(ink.detail.negate())));
  color = mix(color, rgb(0x0b0d14), coverage(abs(tile).sub(0.012)).mul(0.6));
  const halo = exp(max(tile, 0).mul(-9)).mul(0.5).mul(smoothstep(-0.01, 0.02, tile)).mul(smoothstep(0.5, 0.4, max(abs(p.x), abs(p.y))));
  return sprite(flat(1, 1), 10, color.add(rgb(0xffffff).mul(halo.mul(0.25))), max(coverage(tile), halo.mul(0.4)));
}
