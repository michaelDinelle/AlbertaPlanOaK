import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, dot, exp, float, floor, fract, fwidth, length, max, min, mix, normalize, positionWorld, pow, round, select, sign, smoothstep, sqrt, step, texture, uniformArray, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Scalar = THREE.UniformNode<'float', number>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
export type Points = ReturnType<typeof uniformArray>;

export const HALF = 6.2;
export const TOP = 4.35;
export const ROWS = 3.95;
export const AXIS = TOP - ROWS - 0.3;
export const FLOOR = AXIS - 2.45;
export const FAN = FLOOR - 0.42;
export const PLATE_W = 2.16;
export const PLATE_H = 0.8;
export const PLATE_Y = FAN - PLATE_H / 2;
export const MOUTH = TOP + 0.16;
const HOPPER = MOUTH + 0.5;
export const BEAD = 0.15;
export const PITCH = 0.17;
export const STACK = AXIS - 0.22 - FLOOR;
export const GAUGE = 1.5;
const GAP = 0.34;
export const XG = -HALF - GAP - GAUGE / 2;
export const WG = HALF + GAP + GAUGE / 2;
export const LABEL_X = XG - GAUGE / 2 - 0.1;
export const HEAD = TOP + 0.24;
export const X_RANGE = 3;
const FRAME_X = HALF + GAP + GAUGE + 0.86;
const FRAME_TOP = HOPPER + 0.44;
const FRAME_BOTTOM = FAN - PLATE_H - 0.3;
const FRAME = 0.3;
const MIDDLE = (FRAME_TOP + FRAME_BOTTOM) / 2;
const TALL = (FRAME_TOP - FRAME_BOTTOM) / 2;
export const EXTENT = new THREE.Box2(new THREE.Vector2(-FRAME_X - 0.12, -(FRAME_TOP + 0.12)), new THREE.Vector2(FRAME_X + 0.12, -(FRAME_BOTTOM - 0.12)));

export const IVORY = 0xf3ead8;
export const WARM = 0xffb45e;
export const COOL = 0x8f96ff;
export const STEEL = 0xc5cedb;
export const INK = 0x9c8a74;
const BRASS = 0xc9a45c;
const ENAMEL = 0x100c0a;
const STOPS = [0x7a82ff, 0xa9b3ff, 0xece4cf, 0xffc98a, 0xff9152] as const;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

export const rowY = (j: number, rows: number) => TOP - ((j + 0.5) * ROWS) / rows;
export const plateX = (b: number, bins: number) => (b - (bins - 1) / 2) * ((2 * HALF) / bins);
export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const area = (x0: number, y0: number, x1: number, y1: number) => flat(x1 - x0, y1 - y0).translate((x0 + x1) / 2, 0, -(y0 + y1) / 2);
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const unit = (scale = 1): N<'vec2'> => uv().sub(0.5).mul(2 * scale);
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(max(fwidth(positionWorld.x), fwidth(positionWorld.z)), 1e-5))), 0, 1);

export function classColor(b: number, bins: number): THREE.Color {
  const t = (bins > 1 ? b / (bins - 1) : 0.5) * (STOPS.length - 1), k = Math.min(Math.floor(t), STOPS.length - 2);
  return new THREE.Color(STOPS[k]).lerp(new THREE.Color(STOPS[k + 1]), t - k);
}

function sprite(color: N<'vec3'>, opacity: N<'float'>, order: number, geometry: THREE.BufferGeometry, additive = false): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  if (additive) material.blending = THREE.AdditiveBlending;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

const brass = (shine: N<'float'>): N<'vec3'> => mix(rgb(0x5e4520), rgb(BRASS), shine).add(rgb(0xfff0c8).mul(pow(shine, float(8)).mul(0.3)));

function sphere(p: N<'vec2'>, base: N<'vec3'>): { color: N<'vec3'>; spec: N<'float'> } {
  const s = p.div(max(length(p), 1)), n = vec3(s.x, s.y, sqrt(max(float(1).sub(dot(s, s)), 0)));
  const key = normalize(vec3(-0.45, 0.62, 0.64));
  const diffuse = max(dot(n, key), 0);
  const spec = pow(max(dot(n, normalize(key.add(vec3(0, 0, 1)))), 0), float(36));
  const sky = smoothstep(-0.7, 0.9, n.y).mul(0.55).add(0.45);
  return { color: base.mul(diffuse.mul(0.62).add(0.22)).mul(sky), spec };
}

function walnut(along: N<'float'>, across: N<'float'>): N<'vec3'> {
  const warp = fractal(vec2(along.mul(0.16), across.mul(1.1)), 3);
  const ring = fract(across.mul(2.4).add(warp.mul(2.2)).add(fractal(vec2(along.mul(0.5), across.mul(0.35)), 2).mul(0.5)));
  const figure = smoothstep(0, 0.45, ring).mul(smoothstep(1, 0.55, ring));
  const pores = noise(vec2(along.mul(1.7), across.mul(38))).mul(0.5).add(0.5);
  return mix(rgb(0x4a2c17), rgb(0x9a6437), figure.mul(0.6).add(pores.mul(0.25)));
}

export function backdrop(lit: Lit): THREE.Mesh {
  const q = chart(), local = q.sub(vec2(0, MIDDLE));
  const outer = sdBox(local, vec2(FRAME_X, TALL), 0.34), inner = outer.add(FRAME);
  const sideways = step(float(TALL).sub(abs(local.y)), float(FRAME_X).sub(abs(local.x)));
  const wood = walnut(select(sideways.greaterThan(0.5), q.x, q.y), select(sideways.greaterThan(0.5), q.y, q.x));
  const depth = clamp(outer.negate().div(FRAME), 0, 1);
  const bead = float(1).sub(abs(depth.mul(2).sub(1))).mul(0.42).add(0.72);
  const light = local.y.div(TALL).mul(0.12).sub(local.x.div(FRAME_X).mul(0.05)).add(1);
  const frame = wood.mul(bead).mul(light);
  const veneer = mix(rgb(0x24170d), rgb(0x3b2615), fractal(vec2(q.x.mul(6.5), q.y.mul(0.22)), 3).mul(0.5).add(0.5)).mul(noise(q.mul(vec2(0.35, 2.4))).mul(0.14).add(1));
  const face = veneer.mul(smoothstep(-0.02, -0.55, inner).mul(0.45).add(0.55));
  const wall = rgb(0x0b0907).mul(fractal(q.mul(0.9), 3).mul(0.08).add(1)).mul(smoothstep(0.1, 3.5, outer).mul(-0.45).add(1));
  let color: N<'vec3'> = mix(wall.mul(smoothstep(0, 0.35, outer).mul(0.6).add(0.4)), frame, coverage(outer));
  color = mix(color, face, coverage(inner));
  color = mix(color, rgb(0x050302), coverage(abs(inner).sub(0.012)).mul(0.9));
  color = mix(color, rgb(0xd9b98a).mul(0.55), coverage(abs(outer.add(0.018)).sub(0.008)).mul(0.6));
  let screws: N<'float'> = float(1e3), slots: N<'float'> = float(1e3);
  for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const at = q.sub(vec2(sx * (FRAME_X - FRAME / 2), MIDDLE + sy * (TALL - FRAME / 2)));
    screws = min(screws, sdCircle(at, 0.075));
    slots = min(slots, sdBox(vec2(at.x.add(at.y).mul(0.7071), at.x.sub(at.y).mul(0.7071)), vec2(0.06, 0.011), 0));
  }
  color = mix(color, brass(float(0.72)).mul(smoothstep(0.075, -0.075, screws).mul(0.35).add(0.75)), coverage(screws));
  color = mix(color, rgb(0x241a0c), coverage(slots).mul(coverage(screws)));
  return sprite(lit(color), float(1), 0, flat(80, 50));
}

export interface BoardUniforms {
  span: Scalar;
  minor: Scalar;
  major: Scalar;
  regions: Points;
}

export function board(lit: Lit, rows: number, bins: number, u: BoardUniforms): THREE.Mesh {
  const q = chart(), pitch = ROWS / rows, region = (k: number) => u.regions.element(k) as unknown as N<'vec4'>;
  const r = float(TOP).sub(q.y).div(pitch), inRows = step(0, r).mul(step(r, rows)), across = fract(r).sub(0.5).mul(pitch);
  const onBoard = step(abs(q.x), HALF);
  const wellHalf = Math.min(0.085, 0.3 * pitch);
  let color: N<'vec3'> = rgb(0x120b07), alpha: N<'float'> = float(0);
  const paint = (tint: N<'vec3'>, cover: N<'float'>) => {
    color = mix(color, tint, cover);
    alpha = max(alpha, cover);
  };

  const groove = crisp(abs(across.sub(0.022)).sub(0.012)).mul(inRows).mul(onBoard);
  paint(rgb(0x050302), groove.mul(0.55));
  const rail = coverage(abs(across).sub(0.013)).mul(inRows).mul(onBoard);
  paint(brass(clamp(float(0.62).sub(across.div(0.026)), 0, 1)), rail);
  const bracket = sdBox(vec2(abs(q.x).sub(HALF + 0.04), across), vec2(0.05, 0.032), 0.012);
  paint(brass(float(0.55)), coverage(bracket).mul(inRows));

  const well = (centre: number, y: N<'float'>) => sdBox(vec2(q.x.sub(centre), y), vec2(GAUGE / 2, wellHalf), wellHalf);
  const wells = min(select(inRows.greaterThan(0.5), min(well(XG, across), well(WG, across)), float(1e3)), well(XG, q.y.sub(AXIS)));
  const wellsTop = min(select(inRows.greaterThan(0.5), min(well(XG, across.add(0.012)), well(WG, across.add(0.012))), float(1e3)), well(XG, q.y.sub(AXIS).sub(0.012)));
  paint(rgb(0x7a5d36), coverage(wells.sub(0.014)));
  paint(rgb(0x050403), coverage(wells).mul(coverage(wellsTop)));
  const inner = GAUGE / 2 - 0.08, ticks = abs(fract(q.x.sub(XG).div(inner / X_RANGE).add(0.5)).sub(0.5)).mul(inner / X_RANGE);
  const inXWell = coverage(well(XG, select(inRows.greaterThan(0.5), across, q.y.sub(AXIS))));
  paint(rgb(0x3b3024), coverage(ticks.sub(0.005)).mul(step(abs(q.x.sub(XG)), inner + 0.01)).mul(inXWell).mul(0.8));
  paint(rgb(0x6e5b44), coverage(abs(q.x.sub(XG)).sub(0.007)).mul(inXWell));
  paint(rgb(0x3f3326), coverage(abs(q.x.sub(WG)).sub(0.007)).mul(coverage(well(WG, across))).mul(inRows));

  const zero = coverage(abs(q.x).sub(0.012)).mul(step(AXIS + 0.02, q.y)).mul(step(q.y, AXIS + 0.17));
  const value = q.x.div(u.span);
  const minor = abs(fract(value.div(u.minor).add(0.5)).sub(0.5)).mul(u.minor).mul(u.span);
  const major = abs(fract(value.div(u.major).add(0.5)).sub(0.5)).mul(u.major).mul(u.span);
  const band = q.y.sub(AXIS);
  paint(rgb(0x8a7456), coverage(minor.sub(0.006)).mul(step(0.02, band)).mul(step(band, 0.075)).mul(onBoard));
  paint(rgb(0xb89a6c), coverage(major.sub(0.008)).mul(step(0.02, band)).mul(step(band, 0.12)).mul(onBoard));
  paint(rgb(IVORY).mul(0.85), zero);
  paint(brass(clamp(float(0.7).sub(band.div(0.04)), 0, 1)), coverage(abs(band).sub(0.02)).mul(step(abs(q.x), HALF + 0.06)));

  const pile = step(FLOOR, q.y).mul(step(q.y, AXIS));
  let divider: N<'float'> = abs(abs(q.x).sub(HALF + 0.02)).sub(0.03);
  for (let k = 0; k + 1 < bins; k++) divider = min(divider, abs(q.x.sub(region(k).y)).sub(0.024));
  let a: N<'float'> = region(0).x, c: N<'float'> = region(0).y, m: N<'float'> = region(0).z;
  for (let k = 1; k < bins; k++) {
    const right = step(region(k).x, q.x);
    a = mix(a, region(k).x, right);
    c = mix(c, region(k).y, right);
    m = mix(m, region(k).z, right);
  }
  const slot = q.x.sub(a).div(max(c.sub(a), 1e-4)).mul(m);
  const pin = abs(slot.sub(round(slot))).mul(c.sub(a)).div(max(m, 1)).sub(0.0055);
  const stack = step(FLOOR, q.y).mul(step(q.y, FLOOR + STACK));
  paint(rgb(0x6b5436), coverage(pin).mul(stack).mul(onBoard).mul(0.75));
  paint(brass(float(0.72)), coverage(divider).mul(pile));
  let notches: N<'float'> = float(1e3);
  for (let k = 0; k + 1 < bins; k++) notches = min(notches, abs(q.x.sub(region(k).y)));
  paint(brass(float(0.85)), coverage(notches.sub(float(0.06).mul(clamp(q.y.sub(AXIS - 0.02).div(0.14), 0, 1)))).mul(step(AXIS - 0.02, q.y)).mul(step(q.y, AXIS + 0.12)));
  paint(brass(clamp(float(0.66).sub(q.y.sub(FLOOR).div(0.05)), 0, 1)), coverage(abs(q.y.sub(FLOOR)).sub(0.026)).mul(step(abs(q.x), HALF + 0.06)));

  const t = float(FLOOR).sub(q.y).div(FLOOR - FAN), inFan = step(0, t).mul(step(t, 1));
  let lines: N<'float'> = float(1e3);
  for (let b = 0; b < bins; b++) {
    const left = plateX(b, bins) - PLATE_W / 2, right = plateX(b, bins) + PLATE_W / 2;
    const from = mix(region(b).x, float(left), t), to = mix(region(b).y, float(right), t);
    paint(rgb(classColor(b, bins)).mul(0.5), step(from, q.x).mul(step(q.x, to)).mul(inFan).mul(0.16));
    lines = min(lines, min(sdSegment(q, vec2(region(b).x, FLOOR), vec2(left, FAN), 0), sdSegment(q, vec2(region(b).y, FLOOR), vec2(right, FAN), 0)));
  }
  paint(brass(float(0.6)), coverage(lines.sub(0.011)).mul(step(FAN - 0.02, q.y)).mul(step(q.y, FLOOR)));

  for (let b = 0; b < bins; b++) {
    const at = q.sub(vec2(plateX(b, bins), PLATE_Y)), d = sdBox(at, vec2(PLATE_W / 2, PLATE_H / 2), 0.1);
    paint(rgb(ENAMEL).mul(at.y.div(PLATE_H).mul(0.5).add(1)), coverage(d));
    paint(brass(clamp(float(0.55).add(at.y.div(PLATE_H)), 0, 1)), coverage(abs(d.add(0.03)).sub(0.022)));
  }

  const funnel = min(sdSegment(q, vec2(-0.58, HOPPER), vec2(-0.12, MOUTH + 0.02), 0.035), sdSegment(q, vec2(0.58, HOPPER), vec2(0.12, MOUTH + 0.02), 0.035));
  const height = q.y.sub(MOUTH).div(HOPPER - MOUTH), half = mix(float(0.12), float(0.58), clamp(height, 0, 1));
  paint(rgb(0x070504), step(abs(q.x), half).mul(step(MOUTH, q.y)).mul(step(q.y, HOPPER)));
  for (const [bx, by] of [[-0.3, 0.3], [-0.14, 0.3], [0.02, 0.3], [0.18, 0.3], [0.34, 0.3], [-0.06, 0.16], [0.1, 0.16], [0.02, 0.03]] as const) {
    const p = q.sub(vec2(bx, MOUTH + by)).div(BEAD / 2);
    const ball = sphere(p, rgb(STEEL));
    paint(ball.color.mul(0.8).add(vec3(ball.spec.mul(0.5))), coverage(length(p).sub(1)));
  }
  paint(brass(clamp(float(0.5).add(q.x.mul(0.6)), 0.2, 1)), coverage(funnel));
  paint(brass(float(0.8)), coverage(sdBox(q.sub(vec2(0, HOPPER + 0.02)), vec2(0.7, 0.035), 0.03)));

  return sprite(lit(color), alpha, 1, area(-HALF - GAP - GAUGE - 0.3, FAN - PLATE_H - 0.1, HALF + GAP + GAUGE + 0.3, HOPPER + 0.1));
}

export function gauges(side: 'x' | 'w', data: THREE.DataTexture, rows: number, scale: Scalar, lit: Lit): THREE.Mesh {
  const q = chart(), pitch = ROWS / rows, store = texture(data), centre = side === 'x' ? XG : WG;
  const r = float(TOP).sub(q.y).div(pitch), inRows = step(0, r).mul(step(r, rows)), index = clamp(floor(r), 0, rows - 1);
  const bias = side === 'x' ? step(abs(q.y.sub(AXIS)), Math.min(pitch, 0.4) / 2) : float(0);
  const which = select(bias.greaterThan(0.5), float(rows), index), middle = select(bias.greaterThan(0.5), float(AXIS), float(TOP).sub(index.add(0.5).mul(pitch)));
  const row = store.load(uvec2(vec2(which, 0))), y = q.y.sub(middle), shown = max(inRows, bias);
  const thick = Math.min(0.1, 0.34 * pitch), inner = GAUGE / 2 - 0.08, x = q.x.sub(centre);
  const bar = (value: N<'float'>, range: N<'float'> | number) => {
    const reach = clamp(value.div(range), -1, 1).mul(inner);
    return { reach, d: sdBox(vec2(x.sub(reach.mul(0.5)), y), vec2(abs(reach).mul(0.5).add(thick / 2), thick / 2), thick / 2) };
  };
  const chevron = (tip: N<'float'>, direction: N<'float'>) => {
    const along = x.sub(tip).mul(direction);
    return max(max(along, along.negate().sub(0.07)), abs(y).sub(along.add(0.07).mul(0.75)));
  };
  if (side === 'x') {
    const { d } = bar(row.x, X_RANGE), over = step(X_RANGE, abs(row.x));
    const arrow = chevron(sign(row.x).mul(inner + 0.075), sign(row.x));
    const glow = exp(max(d, 0).negate().div(0.03)).mul(0.14);
    const tint = rgb(IVORY).mul(select(bias.greaterThan(0.5), float(0.5), float(0.92)));
    return sprite(tint, max(max(crisp(d), crisp(arrow).mul(over)), glow).mul(shown), 2, area(XG - GAUGE / 2 - 0.1, AXIS - 0.3, XG + GAUGE / 2 + 0.1, TOP + 0.1));
  }
  const { reach, d } = bar(row.y, scale), start = clamp(row.z.div(scale), -1, 1).mul(inner);
  const tick = sdBox(vec2(x.sub(start), y), vec2(0.008, thick / 2 + 0.03), 0);
  const trail = sdBox(vec2(x.sub(start.add(reach).mul(0.5)), y.add(thick / 2 + 0.028)), vec2(abs(reach.sub(start)).mul(0.5), 0.005), 0);
  const wellD = sdBox(vec2(x, y), vec2(GAUGE / 2, Math.min(0.085, 0.3 * pitch)), Math.min(0.085, 0.3 * pitch));
  const sheen = smoothstep(0.1, 0, abs(fract(q.x.mul(0.9).add(q.y.mul(0.5)).mul(1.3)).sub(0.5))).mul(0.5).add(0.5);
  const veil = crisp(wellD.add(0.012)), fill = crisp(d), line = crisp(trail), mark = crisp(tick);
  let color: N<'vec3'> = rgb(0x222a35).mul(sheen);
  color = mix(color, lit(select(row.y.greaterThan(0), rgb(WARM), rgb(COOL)).mul(0.78)), fill);
  color = mix(color, rgb(0x8a7458), line);
  color = mix(color, brass(float(0.9)), mark);
  const opacity = max(max(veil.mul(0.32), fill.mul(0.9)), max(line.mul(0.8), mark));
  return sprite(color, opacity.mul(inRows), 2, area(WG - GAUGE / 2 - 0.1, AXIS, WG + GAUGE / 2 + 0.1, TOP + 0.1));
}

export function path(points: Points, count: number, reveal: Scalar, alpha: Scalar, rows: number): THREE.Mesh {
  const q = chart(), point = (k: number) => points.element(k) as unknown as N<'vec4'>, thick = Math.min(0.085, 0.27 * (ROWS / rows));
  let kicks: N<'float'> = float(1e3), drops: N<'float'> = float(1e3), heads: N<'float'> = float(1e3), beyond: N<'float'> = float(1e3);
  let tint: N<'vec3'> = rgb(WARM);
  for (let k = 0; k + 1 < count; k++) {
    const a = point(k), b = point(k + 1), f = clamp(reveal.sub(k), 0, 1);
    const d = select(f.greaterThan(0), sdSegment(q, a.xy, mix(a.xy, b.xy, f), 0), float(1e3));
    if (k % 2 === 0) {
      drops = min(drops, d);
      continue;
    }
    const run = b.x.sub(a.x), direction = select(run.greaterThan(0), float(1), float(-1)), size = min(abs(run), thick * 1.9);
    tint = select(d.lessThan(kicks), select(run.greaterThan(0), rgb(WARM), rgb(COOL)), tint);
    kicks = min(kicks, d);
    const along = q.x.sub(b.x).mul(direction), up = abs(q.y.sub(b.y));
    const head = max(max(along.sub(thick * 0.35), along.negate().sub(size)), up.sub(along.negate().add(thick * 0.35).mul(thick * 1.25).div(size.add(thick * 0.35))));
    heads = min(heads, select(f.greaterThanEqual(1).and(abs(run).greaterThan(0.03)), head, float(1e3)));
    const out = sign(b.x), past = q.x.sub(b.x).mul(out);
    const flag = max(max(past.sub(0.12), past.negate()), abs(q.y.sub(b.y)).sub(float(0.12).sub(past).mul(0.6)));
    beyond = min(beyond, select(b.w.greaterThan(0.5).and(f.greaterThanEqual(1)), flag, float(1e3)));
  }
  const bar = coverage(kicks.sub(thick / 2)), tip = coverage(heads), line = coverage(drops.sub(0.008)).mul(0.5);
  const glow = exp(max(kicks, 0).negate().div(0.07)).mul(0.28), mark = coverage(beyond);
  const color = mix(mix(rgb(IVORY).mul(0.9), tint.mul(1.25), max(max(bar, tip), glow)), rgb(IVORY).mul(1.6), mark);
  return sprite(color, max(max(max(bar, tip), max(line, glow)), mark).mul(alpha), 5, area(-HALF - 0.4, AXIS - 0.2, HALF + 0.4, MOUTH + 0.1));
}

export function ball(alpha: Scalar): THREE.Mesh {
  const p = unit(1.9), r = length(p);
  const lit = sphere(p, rgb(0xdfe4ec));
  const body = coverage(r.sub(1)), glow = exp(r.negate().mul(1.5)).mul(smoothstep(1.9, 1, r)).mul(0.55);
  const color = mix(vec3(1.6, 1.49, 1.28), lit.color.mul(1.25).add(vec3(lit.spec.mul(1.4))), body);
  return sprite(color, max(body, glow).mul(alpha), 8, flat(BEAD * 1.9, BEAD * 1.9));
}

export function beadMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const look = attribute<'vec4'>('aBead', 'vec4'), p = unit(1.08), r = length(p);
  const shade = sphere(p, mix(rgb(STEEL).mul(0.85), rgb(0xffcf5a), look.x));
  const body = coverage(r.sub(1)), fresh = coverage(abs(r.sub(0.98)).sub(0.08)).mul(look.z);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = mix(lit(shade.color.mul(1.2)).add(vec3(shade.spec.mul(0.55))), rgb(0xfff4d6).mul(1.4), fresh);
  material.opacityNode = max(body, fresh).mul(look.y);
  return material;
}

export function token(tint: N<'vec3'>, alpha: Scalar): THREE.Mesh {
  const width = PLATE_W + 0.5, height = PLATE_H + 0.5, p = uv().sub(0.5).mul(vec2(width, height));
  const frame = sdBox(p, vec2(PLATE_W / 2, PLATE_H / 2), 0.1);
  const rim = coverage(abs(frame.add(0.03)).sub(0.026)), halo = exp(abs(frame).negate().div(0.07)).mul(0.6);
  const at = p.sub(vec2(PLATE_W / 2 - 0.36, 0)), disc = length(at);
  const jewel = sphere(at.div(0.15), tint.mul(0.9));
  const inner = coverage(disc.sub(0.15)), ring = coverage(abs(disc.sub(0.19)).sub(0.035)), shine = exp(disc.negate().div(0.12)).mul(0.5);
  let color: N<'vec3'> = tint.mul(1.5);
  color = mix(color, brass(clamp(float(0.6).add(at.y.mul(3)), 0, 1)), ring);
  color = mix(color, jewel.color.mul(1.6).add(vec3(jewel.spec)), inner);
  return sprite(color, max(max(rim, halo), max(max(ring, inner), shine)).mul(alpha), 7, flat(width, height));
}

export function verdict(tint: Tint, alpha: Scalar): THREE.Mesh {
  const width = PLATE_W + 0.7, height = PLATE_H + 0.7, p = uv().sub(0.5).mul(vec2(width, height));
  const frame = sdBox(p, vec2(PLATE_W / 2, PLATE_H / 2), 0.1);
  const band = abs(frame.sub(0.1)), glow = exp(band.negate().div(0.045)).mul(0.85).mul(smoothstep(-0.02, 0.06, frame));
  return sprite(tint.mul(1.4), glow.mul(alpha), 6, flat(width, height), true);
}

export function strip(tint: N<'vec3'>, alpha: Scalar): THREE.Mesh {
  const p = uv().sub(0.5).mul(vec2(1, 2)), fade = smoothstep(0.5, 0.36, abs(p.x));
  const glow = exp(abs(p.y).negate().mul(9)).mul(fade);
  return sprite(tint.mul(1.3), glow.mul(alpha), 3, flat(1, 0.5), true);
}

export interface Label { mesh: THREE.Mesh; width: number; texture: THREE.DataTexture; alpha: Scalar }

export function label(text: string, height: number, tint: N<'vec3'>, alpha: Scalar, weight = 600): Label {
  const px = 96, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Regression labels need a 2D canvas');
  const font = `${weight} ${px}px ${FONT}`;
  context.font = font;
  const w = Math.ceil(context.measureText(text).width + px * 0.3), h = Math.ceil(px * 1.4);
  canvas.width = w;
  canvas.height = h;
  context.font = font;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, w / 2, h / 2);
  const pixels = context.getImageData(0, 0, w, h).data, values = new Uint8Array(w * h);
  for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) values[(h - 1 - row) * w + column] = pixels[(row * w + column) * 4 + 3]!;
  const map = new THREE.DataTexture(values, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  const tall = height * 1.4, wide = (tall * w) / h;
  return { mesh: sprite(tint, texture(map, uv()).r.mul(alpha), 6, flat(wide, tall)), width: wide, texture: map, alpha };
}

