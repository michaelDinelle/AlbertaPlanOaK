import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, round, select, sin, smoothstep, sqrt, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const GOLD = 0xffc85e;
export const ERROR = 0xff5563;
export const CABLES = [0xff993d, 0x5fd68d, 0xb48cff, 0xff78c6, 0xe9e0c8] as const;
export const INK = 0xd7cfbd;
export const HISTORY = 50;
export const SLOT = 0.1;

const RAIL = 4.2;
const METER_RANGE = 6;
const ARENA = 5.2;
const LANE = 0.24;
const FACE_MARGIN = 0.5;
const GLASS_MARGIN = 0.24;
const FACE = 0x15171b;
const GLASS = 0x04070b;
const GRID = 0x223445;
const LED = 0xf6ead0;
const AMBER = 0xffae45;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const LIGHT = vec2(-0.6, 0.8);

export interface Layout {
  rows: number;
  signs: number;
  period: number;
  scale: number;
  unit: number;
  rail: number;
  pitch: number;
  half: number;
  top: number;
  labelX: number;
  meterX: number;
  meterHalf: number;
  jackX: number;
  a: readonly [number, number];
  b: readonly [number, number];
  c: readonly [number, number];
  mixJackX: number;
  knobX: number;
  glyphX: number;
  glass: readonly [number, number];
  laneX: readonly number[];
  arenaX: number;
  ruleX: number;
  historyX: number;
  clockTop: number;
  clockBottom: number;
}

export function layout(inputs: number, signs: number, scale: number, period: number): Layout {
  const rows = inputs + 1, unit = RAIL / scale, pitch = (2 * RAIL) / rows, meterHalf = METER_RANGE * unit;
  const labelX = 0.42, meterX = 0.74 + meterHalf, jackX = meterX + meterHalf + 0.32, a1 = jackX + 0.3;
  const b0 = a1 + 0.06, mixJackX = b0 + 0.3, knobX = mixJackX + 0.56, glyphX = knobX + 0.4, b1 = glyphX + 0.28;
  const c0 = b1 + 0.06, glass0 = c0 + 0.12, lane0 = glass0 + 0.3;
  const laneX = Array.from({ length: signs }, (_, k) => lane0 + k * LANE);
  const arenaX = lane0 + LANE * (signs - 1) + 0.26, ruleX = arenaX + ARENA, historyX = ruleX + 0.92;
  const glass1 = historyX + SLOT * (HISTORY - 1) + 0.2, c1 = glass1 + 0.12, s = -c1 / 2;
  const rowY = (r: number) => RAIL - (r + 0.5) * pitch;
  return {
    rows, signs, period, scale, unit, rail: RAIL, pitch, half: c1 / 2, top: RAIL + FACE_MARGIN,
    labelX: labelX + s, meterX: meterX + s, meterHalf, jackX: jackX + s,
    a: [s, a1 + s], b: [b0 + s, b1 + s], c: [c0 + s, c1 + s],
    mixJackX: mixJackX + s, knobX: knobX + s, glyphX: glyphX + s,
    glass: [glass0 + s, glass1 + s], laneX: laneX.map((x) => x + s), arenaX: arenaX + s, ruleX: ruleX + s, historyX: historyX + s,
    clockTop: rowY(signs - 1) - 0.58, clockBottom: rowY(rows - 1) + 0.62,
  };
}

export const rowY = (l: Layout, r: number) => l.rail - (r + 0.5) * l.pitch;
export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const pixel = (): N<'float'> => max(fwidth(positionWorld.x), fwidth(positionWorld.z));
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel())), 0, 1);
const ring = (d: N<'float'>, half: number | N<'float'>): N<'float'> => abs(d).sub(half);

interface Paint { color: N<'vec3'>; alpha: N<'float'> }

const over = (under: Paint, color: N<'vec3'>, alpha: N<'float'>): Paint => {
  const a = alpha.add(under.alpha.mul(alpha.oneMinus()));
  return { color: color.mul(alpha).add(under.color.mul(under.alpha).mul(alpha.oneMinus())).div(max(a, 1e-4)), alpha: a };
};

const clear = (): Paint => ({ color: vec3(0), alpha: float(0) });

function sprite(width: number, depth: number, x: number, y: number, order: number, paint: Paint): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = paint.color;
  material.opacityNode = paint.alpha;
  const mesh = new THREE.Mesh(flat(width, depth), material);
  mesh.position.set(x, 0, -y);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function hexagon(p: N<'vec2'>, r: number): N<'float'> {
  const k = vec2(-0.866025404, 0.5), a = abs(p);
  const b = a.sub(k.mul(min(dot(k, a), 0).mul(2)));
  const c = b.sub(vec2(clamp(b.x, -0.577350269 * r, 0.577350269 * r), r));
  return length(c).mul(select(c.y.greaterThanEqual(0), float(1), float(-1)));
}

function pointer(p: N<'vec2'>, tip: N<'vec2'>, along: N<'vec2'>, h: N<'float'>, b: N<'float'>): N<'float'> {
  const local = p.sub(tip), y = dot(local, along), x = abs(local.x.mul(along.y).sub(local.y.mul(along.x)));
  return max(x.mul(h).add(y.mul(b)).div(max(length(vec2(h, b)), 1e-5)), y.negate().sub(h));
}

function jack(under: N<'vec3'>, p: N<'vec2'>, lit: Lit): N<'vec3'> {
  const nut = hexagon(p, 0.098), r = length(p), unit = p.div(max(r, 1e-4));
  const metal = mix(rgb(0x4a4e55), rgb(0xd5dae1), dot(unit, LIGHT).mul(0.5).add(0.5));
  let color = mix(under, rgb(0x050506), crisp(sdCircle(p.sub(vec2(0.012, -0.016)), 0.108)).mul(0.6));
  color = mix(color, lit(metal.mul(0.8)), crisp(nut));
  color = mix(color, lit(mix(rgb(0x2c2f34), rgb(0x9ba1a9), dot(unit, LIGHT).mul(-0.5).add(0.5))), crisp(sdCircle(p, 0.068)));
  return mix(color, rgb(0x020203), crisp(sdCircle(p, 0.05)));
}

export function desk(lit: Lit): THREE.Mesh {
  const q = chart();
  const weave = noise(q.mul(vec2(9, 1.2))).mul(0.05).add(noise(q.mul(0.6)).mul(0.12)).add(1);
  return sprite(90, 90, 0, 0, 0, { color: lit(rgb(0x0c0e12).mul(weave)), alpha: float(1) });
}

export function rack(l: Layout, lit: Lit): THREE.Mesh {
  const q = chart(), faceHalf = l.top, shellHalf = vec2(l.half + 0.2, l.top + 0.34);
  const shellAt = (p: N<'vec2'>) => sdBox(p, shellHalf, 0.16), shell = shellAt(q), e = 0.01;
  const shellNormal = vec2(shellAt(q.add(vec2(e, 0))).sub(shellAt(q.sub(vec2(e, 0)))), shellAt(q.add(vec2(0, e))).sub(shellAt(q.sub(vec2(0, e)))));
  const shellFacing = dot(shellNormal.div(max(length(shellNormal), 1e-5)), LIGHT), shellRim = smoothstep(-0.07, 0, shell);
  const brushed = noise(vec2(q.x.mul(0.5), q.y.mul(48))).mul(0.05).add(noise(vec2(q.x.mul(2.3), q.y.mul(130))).mul(0.03)).add(1);
  let color: N<'vec3'> = lit(rgb(0x1d2025).mul(brushed).add(vec3(0.2).mul(shellRim).mul(shellFacing.max(0))).mul(float(1).sub(shellRim.mul(shellFacing.min(0).abs()).mul(0.55))));
  const railY = abs(q.y).sub(l.top + 0.11), rail = sdBox(vec2(q.x, railY), vec2(l.half + 0.06, 0.13), 0.02);
  const railTone = mix(rgb(0x4d535b), rgb(0x8d949c), smoothstep(-0.13, 0.13, railY.mul(select(q.y.greaterThan(0), float(-1), float(1))))).mul(brushed);
  color = mix(color, rgb(0x060708), crisp(rail.sub(0.012)).mul(0.7));
  color = mix(color, lit(railTone), crisp(rail));
  color = mix(color, rgb(0x0b0c0e), crisp(abs(railY.sub(0.03)).sub(0.012)).mul(crisp(rail)).mul(0.8));

  const faceAt = (p: N<'vec2'>) => [l.a, l.b, l.c].map(([x0, x1]) => sdBox(p.sub(vec2((x0 + x1) / 2, 0)), vec2((x1 - x0) / 2, faceHalf), 0.045)).reduce((u, v) => min(u, v));
  const faces = faceAt(q);
  const normal = vec2(faceAt(q.add(vec2(e, 0))).sub(faceAt(q.sub(vec2(e, 0)))), faceAt(q.add(vec2(0, e))).sub(faceAt(q.sub(vec2(0, e)))));
  const bevel = smoothstep(-0.05, 0, faces), facing = dot(normal.div(max(length(normal), 1e-5)), LIGHT);
  const grain = noise(q.mul(34)).mul(0.035).add(noise(q.mul(3.3)).mul(0.03)).add(1);
  const panel = rgb(FACE).mul(grain).add(vec3(0.16).mul(bevel).mul(facing.max(0))).mul(float(1).sub(bevel.mul(facing.min(0).abs()).mul(0.5)));
  color = mix(color, rgb(0x020203), crisp(faces.sub(0.03)).mul(0.85));
  color = mix(color, lit(panel), crisp(faces));

  const r = clamp(floor(float(l.rail).sub(q.y).div(l.pitch)), 0, l.rows - 1), cy = float(l.rail).sub(r.add(0.5).mul(l.pitch));
  color = jack(color, vec2(q.x.sub(l.jackX), q.y.sub(cy)), lit);
  const patched = select(r.lessThan(l.signs).or(r.equal(l.rows - 1)), float(1), float(0));
  color = mix(color, jack(color, vec2(q.x.sub(l.mixJackX), q.y.sub(cy)), lit), patched);

  for (const [x0, x1] of [l.a, l.b, l.c]) for (const x of [x0 + 0.17, x1 - 0.17]) for (const sign of [1, -1]) {
    const p = q.sub(vec2(x, sign * (faceHalf - 0.14))), unit = p.div(max(length(p), 1e-4));
    color = mix(color, rgb(0x050506), crisp(sdCircle(p.sub(vec2(0.01, -0.012)), 0.07)).mul(0.6));
    color = mix(color, lit(mix(rgb(0x1b1c1f), rgb(0x767b84), dot(unit, LIGHT).mul(0.5).add(0.5))), crisp(sdCircle(p, 0.062)));
    color = mix(color, rgb(0x050506), crisp(hexagon(p, 0.022)));
  }
  color = mix(color, rgb(0x3a3f46), crisp(ring(shell, 0.012)).mul(0.8));
  return sprite(2 * l.half + 0.8, 2 * l.top + 1.0, 0, 0, 1, { color, alpha: crisp(shell.sub(0.012)) });
}

export function meters(l: Layout, data: THREE.DataTexture): THREE.Mesh {
  const q = chart(), store = texture(data);
  const r = clamp(floor(float(l.rail).sub(q.y).div(l.pitch)), 0, l.rows - 1), cy = float(l.rail).sub(r.add(0.5).mul(l.pitch));
  const p = vec2(q.x.sub(l.meterX), q.y.sub(cy)), row = store.load(uvec2(vec2(r, 0)));
  const reach = clamp(row.x.mul(l.unit), -l.meterHalf, l.meterHalf), last = row.y, overflow = select(abs(row.x).greaterThan(METER_RANGE), float(1), float(0));
  let paint = clear();
  const track = sdBox(p, vec2(l.meterHalf + 0.05, 0.058), 0.058);
  paint = over(paint, rgb(0x2a2d33), crisp(track.sub(0.012)));
  paint = over(paint, mix(rgb(0x050607), rgb(0x0d0f12), smoothstep(-0.05, 0.05, p.y)), crisp(track));
  const k = round(p.x.div(l.unit)), tick = sdBox(vec2(p.x.sub(k.mul(l.unit)), p.y.add(0.1)), vec2(0.005, select(abs(k).lessThan(0.5), float(0.035), float(0.018))), 0);
  paint = over(paint, rgb(INK), crisp(max(tick, abs(k).sub(METER_RANGE))).mul(0.45));
  const bar = sdBox(vec2(p.x.sub(reach.mul(0.5)), p.y), vec2(abs(reach).mul(0.5).add(0.028), 0.03), 0.028);
  const segment = abs(fract(p.x.div(l.unit * 0.5).add(0.5)).sub(0.5)).mul(l.unit * 0.5);
  const tint = mix(rgb(LED), rgb(GOLD), last);
  const lamp = tint.mul(mix(float(0.42), float(0.85), smoothstep(0.004, 0.012, segment)));
  paint = over(paint, tint.mul(0.5), exp(max(bar, 0).div(0.03).negate()).mul(0.3).mul(crisp(track.add(0.004))));
  paint = over(paint, lamp, crisp(bar));
  const outward = select(row.x.greaterThan(0), float(1), float(-1)), chevron = pointer(p, vec2(outward.mul(l.meterHalf + 0.1), 0), vec2(outward, 0), float(0.075), float(0.055));
  paint = over(paint, tint.mul(0.95), crisp(chevron).mul(overflow));
  paint = over(paint, rgb(INK), crisp(sdBox(p, vec2(0.006, 0.07), 0)).mul(0.5));
  return sprite(2 * l.meterHalf + 0.5, 2 * l.rail + 0.2, l.meterX, 0, 3, paint);
}

export function glass(l: Layout): THREE.Mesh {
  const q = chart(), [x0, x1] = l.glass, cx = (x0 + x1) / 2, hx = (x1 - x0) / 2, hy = l.rail + GLASS_MARGIN, u = l.unit;
  const box = sdBox(q.sub(vec2(cx, 0)), vec2(hx, hy), 0.1);
  const depth = smoothstep(0, 0.6, box.negate());
  let color: N<'vec3'> = rgb(GLASS).mul(mix(float(0.55), float(1.1), depth)).add(rgb(0x0a1824).mul(exp(abs(q.y).div(-3.2)).mul(0.45)));
  const band = crisp(abs(q.y).sub(l.rail + 0.004));
  const gutter = crisp(q.x.sub(l.ruleX + 0.01)).max(crisp(float(l.historyX - SLOT * 0.5).sub(q.x)));
  const plot = crisp(float(l.arenaX - 0.02).sub(q.x)).mul(gutter);
  const weight = (k: N<'float'>) => select(abs(k).lessThan(0.5), float(0.5), select(mod(abs(k), 4).lessThan(0.5), float(0.26), float(0.1)));
  const ky = round(q.y.div(u)), dy = abs(q.y.sub(ky.mul(u)));
  const kx = round(float(l.ruleX).sub(q.x).div(u)), dx = abs(float(l.ruleX).sub(q.x).sub(kx.mul(u)));
  const lineY = crisp(dy.sub(0.004)).mul(weight(ky)).mul(gutter), lineX = crisp(dx.sub(0.004)).mul(weight(kx).min(0.26)).mul(plot);
  color = color.add(rgb(GRID).mul(max(lineY, lineX).mul(band)));
  color = mix(color, rgb(0x3b4148), crisp(ring(box, 0.018)));
  return sprite(2 * hx + 0.1, 2 * hy + 0.1, cx, 0, 2, { color, alpha: crisp(box.sub(0.018)) });
}

export function lanes(l: Layout, sums: readonly Level[]): THREE.Mesh {
  const q = chart(), u = l.unit;
  let paint = clear();
  for (let k = 0; k < l.signs; k++) {
    const x = l.laneX[k]!, guide = crisp(abs(q.x.sub(x)).sub(0.004)).mul(crisp(abs(q.y).sub(l.rail))).mul(exp(abs(q.y).div(-2.2)));
    paint = over(paint, rgb(CABLES[k]!), guide.mul(0.16));
  }
  for (let k = 0; k < l.signs; k++) {
    const x = l.laneX[k]!, from = sums[k]!.mul(u), to = sums[k + 1]!.mul(u), span = abs(to.sub(from));
    const up = select(to.greaterThanEqual(from), float(1), float(-1)), h = max(min(float(0.095), span), 0.002), stem = to.sub(up.mul(min(h.mul(0.7), span)));
    const shaft = sdSegment(q, vec2(x, from), vec2(x, stem), 0.019);
    const head = pointer(q, vec2(x, to), vec2(0, up), h, h.mul(0.62));
    const arrow = min(shaft, head);
    const tint = rgb(CABLES[k]!);
    paint = over(paint, tint.mul(0.8), exp(max(arrow, 0).div(0.03).negate()).mul(0.4));
    paint = over(paint, tint.mul(1.25), crisp(arrow));
    if (k + 1 < l.signs) {
      const next = l.laneX[k + 1]!, link = sdSegment(q, vec2(x, to), vec2(next, to), 0.006);
      const dash = crisp(abs(fract(q.x.div(0.05)).sub(0.5)).sub(0.22).mul(0.05));
      paint = over(paint, vec3(0.85), crisp(link).mul(dash).mul(0.7));
    }
  }
  const origin = sdBox(q.sub(vec2(l.laneX[0]!, 0)), vec2(0.07, 0.007), 0);
  paint = over(paint, vec3(0.9), crisp(origin));
  const width = l.laneX[l.signs - 1]! - l.laneX[0]! + 0.5, cx = (l.laneX[0]! + l.laneX[l.signs - 1]!) / 2;
  return sprite(width, 2 * l.rail + 0.4, cx, 0, 5, paint);
}

export interface SquareLevels { low: Level; high: Level; truth: Level; fade: Level }

export function square(l: Layout, s: SquareLevels): THREE.Mesh {
  const q = chart(), u = l.unit, side = s.high.sub(s.low), centre = vec2(float(l.ruleX).sub(side.mul(0.5)), s.low.add(s.high).mul(0.5));
  const box = sdBox(q.sub(centre), vec2(side.mul(0.5), side.mul(0.5)), 0);
  const inside = crisp(box);
  const gx = abs(fract(float(l.ruleX).sub(q.x).div(u).add(0.5)).sub(0.5)).mul(u), gy = abs(fract(q.y.sub(s.truth).div(u).add(0.5)).sub(0.5)).mul(u);
  const cells = crisp(min(gx, gy).sub(0.005)).mul(inside);
  const present = smoothstep(0, 0.012, side).mul(s.fade);
  let paint = clear();
  paint = over(paint, rgb(ERROR).mul(0.9), exp(max(box, 0).div(0.07).negate()).mul(0.28).mul(present));
  paint = over(paint, rgb(ERROR).mul(0.55), inside.mul(0.26).mul(present));
  paint = over(paint, rgb(0xff9aa2), cells.mul(0.45).mul(present));
  paint = over(paint, rgb(ERROR).mul(1.35), crisp(ring(box, 0.011)).mul(present));
  const left = l.glass[0] + 0.02, width = l.ruleX + 0.2 - left;
  return sprite(width, 2 * l.rail + 0.6, left + width / 2, 0, 4, { color: paint.color, alpha: paint.alpha.mul(crisp(float(left).sub(q.x))) });
}

export interface MarkLevels { truth: Level; pen: Level; penOn: Level; railHigh: Level; railLow: Level }

export function truthLine(l: Layout, m: MarkLevels): THREE.Mesh {
  const q = chart(), x0 = l.laneX[l.signs - 1]!, x1 = l.ruleX - 0.12;
  const line = sdSegment(q, vec2(x0, m.truth), vec2(x1, m.truth), 0.007);
  let paint = clear();
  paint = over(paint, rgb(GOLD).mul(0.8), exp(max(line, 0).div(0.03).negate()).mul(0.35));
  paint = over(paint, rgb(GOLD).mul(1.2), crisp(line).mul(0.9));
  paint = over(paint, rgb(GOLD).mul(1.3), crisp(sdCircle(q.sub(vec2(x0, m.truth)), 0.03)));
  return sprite(x1 - x0 + 0.3, 2 * l.rail + 0.6, (x0 + x1) / 2, 0, 6, paint);
}

export function ruler(l: Layout, m: MarkLevels, accent: number): THREE.Mesh {
  const q = chart(), u = l.unit, x = l.ruleX, p = q.sub(vec2(x, 0));
  let paint = clear();
  paint = over(paint, rgb(0x8c9cab), crisp(sdBox(p, vec2(0.007, l.rail), 0)));
  const k = clamp(round(q.y.div(u)), -l.scale, l.scale), major = mod(abs(k), 4).lessThan(0.5), zero = abs(k).lessThan(0.5);
  const reach = select(zero, float(0.2), select(major, float(0.13), float(0.065)));
  const tick = sdBox(vec2(p.x.sub(reach.mul(0.5)), q.y.sub(k.mul(u))), vec2(reach.mul(0.5), 0.005), 0);
  paint = over(paint, rgb(0xa9b8c6), crisp(tick).mul(select(major, float(0.95), float(0.6))));
  for (const [sign, level] of [[1, m.railHigh], [-1, m.railLow]] as const) {
    const stop = sdBox(p.sub(vec2(0.02, sign * (l.rail + 0.025))), vec2(0.3, 0.025), 0.01);
    const hatch = crisp(abs(fract(p.x.add(q.y.mul(sign)).div(0.07)).sub(0.5)).mul(0.07).sub(0.012)).mul(crisp(sdBox(p.sub(vec2(0.02, sign * (l.rail + 0.1))), vec2(0.3, 0.05), 0)));
    const hot = mix(rgb(0x5d6670), rgb(ERROR).mul(1.6), level);
    paint = over(paint, hot.mul(0.7), hatch.mul(mix(float(0.4), float(0.9), level)));
    paint = over(paint, hot, crisp(stop));
    paint = over(paint, rgb(ERROR), exp(max(stop, 0).div(0.08).negate()).mul(0.6).mul(level));
  }
  const truthTip = vec2(x - 0.018, m.truth), truth = pointer(q, truthTip, vec2(1, 0), float(0.15), float(0.085));
  paint = over(paint, rgb(0x120c02), crisp(truth.sub(0.012)));
  paint = over(paint, rgb(GOLD).mul(1.25), crisp(truth));
  const penTip = vec2(x + 0.018, m.pen), head = pointer(q, penTip, vec2(-1, 0), float(0.15), float(0.085));
  const arm = sdBox(q.sub(vec2(x + 0.25, m.pen)), vec2(0.1, 0.022), 0.02), pen = min(head, arm);
  paint = over(paint, rgb(accent).mul(0.7), exp(max(pen, 0).div(0.05).negate()).mul(0.45).mul(m.penOn));
  paint = over(paint, rgb(0x040910), crisp(pen.sub(0.012)).mul(m.penOn));
  paint = over(paint, rgb(accent).mul(1.3), crisp(pen).mul(m.penOn));
  return sprite(0.9, 2 * l.rail + 0.6, x + 0.05, 0, 7, paint);
}

export function trace(l: Layout, data: THREE.DataTexture, scroll: Level, accent: number): THREE.Mesh {
  const q = chart(), store = texture(data), u = l.unit;
  const m = q.x.sub(l.historyX).div(SLOT).sub(scroll).add(1), m0 = floor(m);
  const load = (k: N<'float'>): N<'vec4'> => store.load(uvec2(vec2(clamp(k, 0, HISTORY + 1), 0)));
  const xOf = (k: N<'float'>): N<'float'> => float(l.historyX).add(k.sub(1).add(scroll).mul(SLOT));
  const cells = [-1, 0, 1, 2].map((o) => {
    const k = m0.add(o), cell = load(k), inRange = k.greaterThanEqual(0).and(k.lessThanEqual(HISTORY + 1));
    const age = clamp(k.sub(1).add(scroll).div(HISTORY), 0, 1), entry = select(k.lessThan(0.5), scroll, float(1));
    const fade = mix(float(1), float(0.28), age).mul(entry).mul(smoothstep(HISTORY - 0.6, HISTORY + 0.4, k.sub(1).add(scroll)).oneMinus());
    const valid = select(inRange.and(cell.w.greaterThan(0.5)), float(1), float(0));
    return { k, cell, x: xOf(k), fade: fade.mul(valid) };
  });
  let paint = clear();
  for (let i = 0; i < 3; i++) {
    const a = cells[i]!, b = cells[i + 1]!;
    const d = sdSegment(q, vec2(a.x, a.cell.y.mul(u)), vec2(b.x, b.cell.y.mul(u)), 0.009);
    const fade = min(a.fade, b.fade);
    paint = over(paint, rgb(accent).mul(0.6), exp(max(d, 0).div(0.03).negate()).mul(0.3).mul(fade));
    paint = over(paint, rgb(accent).mul(1.15), crisp(d).mul(fade));
  }
  for (const c of cells.slice(1, 3)) {
    const hair = sdSegment(q, vec2(c.x, c.cell.x.mul(u)), vec2(c.x, c.cell.y.mul(u)), 0.005);
    paint = over(paint, rgb(ERROR), crisp(hair).mul(c.fade).mul(0.75));
  }
  for (const c of cells.slice(1, 3)) {
    const dot = sdCircle(q.sub(vec2(c.x, c.cell.x.mul(u))), 0.03);
    paint = over(paint, rgb(0x140d02), crisp(dot.sub(0.01)).mul(c.fade));
    paint = over(paint, rgb(GOLD).mul(1.2), crisp(dot).mul(c.fade));
  }
  const dashes = crisp(abs(fract(q.y.div(0.09)).sub(0.5)).sub(0.2).mul(0.09)), column = crisp(abs(q.y).sub(l.rail));
  for (const c of cells.slice(0, 3)) {
    const code = c.cell.z, index = abs(code).sub(1), flipped = select(abs(code).greaterThan(0.5), c.fade, float(0));
    let tint: N<'vec3'> = rgb(CABLES[0]!);
    for (let k = 1; k < l.signs; k++) tint = select(index.greaterThan(k - 0.5), rgb(CABLES[k]!), tint);
    const bx = c.x.add(SLOT * 0.5), tag = q.sub(vec2(bx, l.rail + 0.13));
    paint = over(paint, tint, crisp(abs(q.x.sub(bx)).sub(0.005)).mul(column).mul(dashes).mul(flipped).mul(0.75));
    const minus = sdBox(tag, vec2(0.048, 0.012), 0.004), sign = select(code.greaterThan(0), min(minus, sdBox(tag, vec2(0.012, 0.048), 0.004)), minus);
    paint = over(paint, tint.mul(1.05), crisp(sdCircle(tag, 0.088)).mul(flipped));
    paint = over(paint, rgb(0x0a0a0c), crisp(sign).mul(flipped));
  }
  const x0 = l.historyX - SLOT, x1 = l.historyX + SLOT * (HISTORY - 1) + 0.12;
  const mask = crisp(float(x0 - 0.02).sub(q.x));
  return sprite(x1 - x0 + 0.4, 2 * l.rail + 0.8, (x0 + x1) / 2, 0, 5, { color: paint.color, alpha: paint.alpha.mul(mask) });
}

export function knob(x: number, y: number, tint: number, angle: Level, lit: Lit): THREE.Mesh {
  const R = 0.16, q = chart(), p = q.sub(vec2(x, y)), r = length(p), unit = p.div(max(r, 1e-4));
  const bearing = atan(p.x, p.y), sweep = (3 * Math.PI) / 4;
  let paint = clear();
  const arc = max(ring(r.sub(R + 0.055), 0.005), abs(bearing).sub(sweep).mul(r));
  paint = over(paint, lit(rgb(INK)), crisp(arc).mul(0.55));
  for (const at of [-sweep, 0, sweep]) {
    const dir = vec2(Math.sin(at), Math.cos(at));
    paint = over(paint, lit(rgb(INK)), crisp(sdSegment(p, dir.mul(R + 0.035), dir.mul(R + 0.1), 0.007)).mul(0.8));
  }
  const shadow = smoothstep(R - 0.03, R + 0.07, length(p.sub(vec2(0.035, -0.045)))).oneMinus();
  paint = over(paint, vec3(0), shadow.mul(0.6));
  const turned = bearing.sub(angle), ridges = sin(turned.mul(34)).mul(0.5).add(0.5);
  const skirt = mix(rgb(0x1a1b1e), rgb(0x565b62), dot(unit, LIGHT).mul(0.5).add(0.5)).mul(ridges.mul(0.35).add(0.75));
  paint = over(paint, lit(skirt), crisp(r.sub(R)));
  const cap = mix(rgb(0x0b0b0d), rgb(0x3a3d43), dot(p.div(R), LIGHT).mul(0.5).add(0.5)).add(vec3(0.06).mul(exp(length(p.sub(vec2(-0.04, 0.05))).div(0.04).negate())));
  paint = over(paint, lit(cap), crisp(r.sub(R * 0.74)));
  paint = over(paint, rgb(tint).mul(0.95), crisp(ring(r.sub(R * 0.8), 0.009)));
  const dir = vec2(sin(angle), cos(angle));
  paint = over(paint, rgb(0xf1ece0).mul(0.95), crisp(sdSegment(p, dir.mul(R * 0.2), dir.mul(R * 0.9), 0.015)));
  return sprite(2 * R + 0.3, 2 * R + 0.3, x, y, 6, paint);
}

export function glyph(x: number, y: number, tint: number, stroke: Level): THREE.Mesh {
  const p = chart().sub(vec2(x, y));
  const d = min(sdBox(p, vec2(0.078, 0.018), 0.008), sdBox(p, vec2(0.018, max(stroke.mul(0.078), 0.018)), 0.008));
  let paint = clear();
  paint = over(paint, rgb(tint).mul(0.8), exp(max(d, 0).div(0.035).negate()).mul(0.5));
  paint = over(paint, rgb(tint).mul(1.4), crisp(d));
  return sprite(0.34, 0.34, x, y, 6, paint);
}

export function clock(l: Layout, count: Level, lit: Lit): THREE.Mesh {
  const q = chart(), n = l.period, gap = (l.clockTop - l.clockBottom) / Math.max(1, n - 1);
  const j = clamp(round(q.y.sub(l.clockBottom).div(gap)), 0, n - 1), cy = float(l.clockBottom).add(j.mul(gap));
  const p = vec2(q.x.sub(l.knobX), q.y.sub(cy)), cell = sdBox(p, vec2(0.08, Math.min(0.045, gap * 0.3)), 0.022);
  const level = clamp(count.sub(j), 0, 1), last = j.greaterThan(n - 1.5);
  const hot = select(last, rgb(ERROR), rgb(AMBER));
  let paint = clear();
  paint = over(paint, lit(rgb(0x2b2e33)), crisp(cell.sub(0.016)));
  paint = over(paint, mix(select(last, rgb(0x2a0d10), rgb(0x2a1b0c)), hot.mul(1.45), level), crisp(cell));
  paint = over(paint, hot.mul(0.6), exp(max(cell, 0).div(0.04).negate()).mul(0.4).mul(level));
  const cx = l.knobX, height = l.clockTop - l.clockBottom + 0.3;
  return sprite(0.4, height, cx, (l.clockTop + l.clockBottom) / 2, 3, paint);
}

export function cable(from: readonly [number, number], to: readonly [number, number], sag: number, tint: number, pulse: Level, lit: Lit): THREE.Mesh {
  const W = 0.062, PAD = 0.03, SEGMENTS = 48, half = W / 2 + PAD;
  const points: THREE.Vector2[] = [];
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = i / SEGMENTS;
    points.push(new THREE.Vector2(from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t - 4 * sag * t * (1 - t)));
  }
  const position = new Float32Array((SEGMENTS + 1) * 6), texcoord = new Float32Array((SEGMENTS + 1) * 4), index: number[] = [];
  points.forEach((point, i) => {
    const a = points[Math.max(0, i - 1)]!, b = points[Math.min(SEGMENTS, i + 1)]!, tx = b.x - a.x, ty = b.y - a.y, len = Math.hypot(tx, ty) || 1, nx = -ty / len, ny = tx / len;
    for (const [side, v] of [[1, 0], [-1, 1]] as const) {
      const o = (i * 2 + v) * 3;
      position[o] = point.x + nx * half * side;
      position[o + 1] = 0;
      position[o + 2] = -(point.y + ny * half * side);
      texcoord[(i * 2 + v) * 2] = i / SEGMENTS;
      texcoord[(i * 2 + v) * 2 + 1] = v;
    }
    if (i < SEGMENTS) index.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(texcoord, 2));
  geometry.setIndex(index);
  const across = uv().y.sub(0.5).mul(2 * half), edge = abs(across).sub(W / 2), body = clamp(across.div(W / 2), -1, 1);
  const round = sqrt(max(float(1).sub(body.mul(body)), 0));
  const shine = exp(body.add(0.35).div(0.22).pow(2).negate()).mul(0.3);
  const band = exp(uv().x.sub(pulse).div(0.09).pow(2).negate()).mul(smoothstep(-0.05, 0.05, pulse)).mul(smoothstep(0.95, 1.05, pulse).oneMinus());
  const color = lit(rgb(tint).mul(round.mul(0.6).add(0.3))).add(vec3(shine)).add(rgb(tint).mul(band.mul(0.75)));
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = crisp(edge);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 8;
  mesh.frustumCulled = false;
  return mesh;
}

export function plug(x: number, y: number, tint: number, lit: Lit): THREE.Mesh {
  const p = chart().sub(vec2(x, y)), r = length(p), unit = p.div(max(r, 1e-4)), facing = dot(unit, LIGHT);
  let paint = clear();
  paint = over(paint, vec3(0), smoothstep(0.08, 0.15, length(p.sub(vec2(0.022, -0.028)))).oneMinus().mul(0.55));
  paint = over(paint, lit(rgb(tint).mul(facing.mul(0.3).add(0.7))), crisp(r.sub(0.112)));
  paint = over(paint, lit(rgb(tint).mul(0.4)), crisp(ring(r.sub(0.084), 0.007)));
  paint = over(paint, lit(rgb(tint).mul(facing.mul(-0.25).add(0.85))), crisp(r.sub(0.07)));
  paint = over(paint, lit(mix(rgb(0x2a2d31), rgb(0xa9b0b9), facing.mul(0.5).add(0.5))), crisp(r.sub(0.034)));
  return sprite(0.4, 0.4, x, y, 9, paint);
}

export interface Label { mesh: THREE.Mesh; width: number; texture: THREE.DataTexture }

export function label(message: string, height: number, tint: number, spacing: number, lit?: Lit): Label {
  const px = 96, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('IDBD labels need a 2D canvas');
  const font = `600 ${px}px ${FONT}`, tracking = `${spacing * px}px`;
  context.font = font;
  context.letterSpacing = tracking;
  const w = Math.ceil(context.measureText(message).width + px * 0.3), h = Math.ceil(px * 1.4);
  canvas.width = w;
  canvas.height = h;
  context.font = font;
  context.letterSpacing = tracking;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(message, w / 2, h / 2);
  const pixels = context.getImageData(0, 0, w, h).data, alpha = new Uint8Array(w * h);
  for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) alpha[(h - 1 - row) * w + column] = pixels[(row * w + column) * 4 + 3]!;
  const map = new THREE.DataTexture(alpha, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  const tall = height * 1.4, wide = (tall * w) / h, ink = rgb(tint);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit ? lit(ink) : ink;
  material.opacityNode = texture(map, uv()).r;
  const mesh = new THREE.Mesh(flat(wide, tall), material);
  mesh.renderOrder = 10;
  mesh.frustumCulled = false;
  return { mesh, width: wide, texture: map };
}
