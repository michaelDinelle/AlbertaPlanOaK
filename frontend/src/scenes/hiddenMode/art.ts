import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, round, select, sin, smoothstep, step, texture, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = THREE.UniformNode<'float', number>;

export const CABINET = { w: 7.9, h: 4.45, r: 0.55 } as const;
const PLATE = { w: 7.42, h: 3.98, r: 0.3 } as const;
export const SCREEN = { x0: -6.9, x1: 1.9, y0: -3.55, y1: 1.85 } as const;
export const DIAL = { x0: -6.9, x1: 1.9, y0: 2.35, y1: 3.55 } as const;
export const BANDS = [-4.7, -0.3] as const;
export const MID = (BANDS[0] + BANDS[1]) / 2;
export const DIAL_Y = (DIAL.y0 + DIAL.y1) / 2;
export const NEEDLE = { y0: DIAL.y0 + 0.1, y1: DIAL.y0 + 0.64 } as const;
export const ROWS = 108;
export const ROW = (SCREEN.y1 - SCREEN.y0) / ROWS;
export const PASS = 0.62;
export const EYE = { x: 4.8, y: 1.62, r: 1.2 } as const;
export const KNOB = { x: 4.8, y: -1.95, r: 1.1, swing: 0.7 } as const;

export const CARRIER = 0xffb13d;
export const PHOSPHOR = 0x4dff95;
const CARRIER_HOT = 0xfff1d2;
const STATIC_DARK = 0x03041a;
const STATIC_LIGHT = 0x2c3396;
const DIAL_WARM = 0xf6dba4;
const DIAL_EDGE = 0x7c4816;
const INK = 0x2a1709;
const WALNUT_DARK = 0x2a160b;
const WALNUT_LIGHT = 0x6c4226;
const FACE = 0x1d1e22;
const CHROME_DARK = 0x363941;
const CHROME_LIGHT = 0xe4e8ee;
const BAKELITE = 0x141417;
const CREAM = 0xe9dfc4;

const light = vec2(-0.55, 0.83);
const TICK = 0.22;
const MAJOR = 1.1;
const SCALE_Y = DIAL.y0 + 0.36;
const DIGIT_Y = DIAL.y0 + 0.9;
const DIGIT_H = 0.34;
export const STATION_Y = DIAL.y0 + 0.19;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 48).rotateX(-Math.PI / 2);
export const bandX = (band: number) => BANDS[0] + (BANDS[1] - BANDS[0]) * band;

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const bandOf = (band: N<'float'>): N<'float'> => mix(float(BANDS[0]), float(BANDS[1]), band);
const ring = (d: N<'float'>, width: number): N<'float'> => abs(d).sub(width);
const cover = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(fwidth(positionWorld.x), fwidth(positionWorld.z)))), 0, 1);

function sprite(width: number, depth: number, x: number, y: number, order: number, color: N<'vec3'>, opacity: N<'float'>): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(flat(width, depth), material);
  mesh.position.set(x, 0, -y);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function zero(p: N<'vec2'>, h: number): N<'float'> {
  return ring(sdBox(p, vec2(h * 0.25, h * 0.42), h * 0.25), h * 0.07);
}

function one(p: N<'vec2'>, h: number): N<'float'> {
  const stem = sdSegment(p, vec2(h * 0.04, h * -0.47), vec2(h * 0.04, h * 0.47), h * 0.07);
  const flag = sdSegment(p, vec2(h * 0.04, h * 0.47), vec2(h * -0.2, h * 0.28), h * 0.065);
  const foot = sdSegment(p, vec2(h * -0.18, h * -0.47), vec2(h * 0.26, h * -0.47), h * 0.055);
  return min(min(stem, flag), foot);
}

const numeral = (p: N<'vec2'>, band: number, h: number): N<'float'> => (band === 0 ? zero(p, h) : one(p, h));

function wood(q: N<'vec2'>): N<'vec3'> {
  const warp = noise(q.mul(0.45)).mul(1.6);
  const lines = noise(vec2(q.x.mul(0.3).add(warp), q.y.mul(7.5).add(warp.mul(0.6)))).mul(0.5).add(0.5);
  const figure = noise(q.mul(1.7).add(3.1)).mul(0.5).add(0.5);
  return mix(rgb(WALNUT_DARK), rgb(WALNUT_LIGHT), lines.mul(0.75).add(figure.mul(0.25)));
}

function chrome(unit: N<'vec2'>): N<'vec3'> {
  return mix(rgb(CHROME_DARK), rgb(CHROME_LIGHT), dot(unit, light).mul(0.5).add(0.5));
}

function screw(under: N<'vec3'>, q: N<'vec2'>, x: number, y: number, slot: number): N<'vec3'> {
  const p = q.sub(vec2(x, y)), r = length(p), slit = vec2(Math.cos(slot), Math.sin(slot)).mul(0.06);
  const dome = chrome(p.div(max(r, 1e-4))).mul(r.div(0.085).mul(0.35).oneMinus());
  return mix(under, mix(dome, rgb(0x121316), coverage(sdSegment(p, slit.negate(), slit, 0.012))), coverage(sdCircle(p, 0.085)));
}

export function desk(lit: Lit): THREE.Mesh {
  const q = chart();
  const lines = noise(vec2(q.x.mul(0.16), q.y.mul(2.4)).add(noise(q.mul(0.3)).mul(1.2))).mul(0.5).add(0.5);
  const top = mix(rgb(0x100a07), rgb(0x291a10), lines);
  const shade = smoothstep(-0.3, 1.2, sdBox(q.sub(vec2(0.24, -0.34)), vec2(CABINET.w, CABINET.h), CABINET.r));
  return sprite(80, 80, 0, 0, 0, lit(top.mul(shade.mul(0.7).add(0.3))), float(1));
}

export function cabinet(lit: Lit): THREE.Mesh {
  const q = chart(), e = 0.02;
  const outer = sdBox(q, vec2(CABINET.w, CABINET.h), CABINET.r);
  const gx = sdBox(q.add(vec2(e, 0)), vec2(CABINET.w, CABINET.h), CABINET.r).sub(sdBox(q.sub(vec2(e, 0)), vec2(CABINET.w, CABINET.h), CABINET.r));
  const gy = sdBox(q.add(vec2(0, e)), vec2(CABINET.w, CABINET.h), CABINET.r).sub(sdBox(q.sub(vec2(0, e)), vec2(CABINET.w, CABINET.h), CABINET.r));
  const normal = vec2(gx, gy).div(max(length(vec2(gx, gy)), 1e-5));
  const bevel = smoothstep(-0.34, 0, outer);
  let color: N<'vec3'> = wood(q).mul(dot(normal, light).mul(bevel).mul(0.55).add(1)).mul(bevel.mul(0.25).oneMinus());

  const plate = sdBox(q, vec2(PLATE.w, PLATE.h), PLATE.r);
  const crinkle = noise(q.mul(13)).mul(0.05).add(noise(q.mul(29).add(7.7)).mul(0.03)).add(1);
  const face = rgb(FACE).mul(crinkle).mul(smoothstep(0.5, 0, plate.negate()).mul(-0.35).add(1));
  color = mix(color, rgb(0x0a0706), coverage(plate.sub(0.07)).mul(0.8));
  color = mix(color, face, coverage(plate));
  color = mix(color, chrome(normal).mul(0.85), coverage(ring(plate.sub(0.025), 0.018)));

  const recess = (centre: N<'vec2'>, half: N<'vec2'>, radius: number, lip: number) => {
    const inner = sdBox(q.sub(centre), half, radius), outerRim = inner.sub(lip), across = q.sub(centre).div(half);
    const wall = rgb(0x09090b).mul(smoothstep(lip, 0, inner).mul(0.6).add(0.4));
    color = mix(color, wall, coverage(outerRim));
    color = mix(color, chrome(across.div(max(length(across), 1e-4)).negate()).mul(0.7), coverage(ring(outerRim.add(0.012), 0.012)));
  };
  recess(vec2((SCREEN.x0 + SCREEN.x1) / 2, (SCREEN.y0 + SCREEN.y1) / 2), vec2((SCREEN.x1 - SCREEN.x0) / 2, (SCREEN.y1 - SCREEN.y0) / 2), 0.24, 0.16);
  recess(vec2((DIAL.x0 + DIAL.x1) / 2, DIAL_Y), vec2((DIAL.x1 - DIAL.x0) / 2, (DIAL.y1 - DIAL.y0) / 2), 0.2, 0.14);
  color = mix(color, rgb(0x070708), coverage(sdCircle(q.sub(vec2(EYE.x, EYE.y)), EYE.r + 0.1)));

  const k = q.sub(vec2(KNOB.x, KNOB.y));
  let marks: N<'float'> = float(1e3);
  for (let t = -2; t <= 2; t++) {
    const a = (KNOB.swing * t) / 2, reach = Math.abs(t) === 2 ? 0.36 : 0.26;
    const dir = vec2(Math.sin(a), Math.cos(a));
    marks = min(marks, sdSegment(k, dir.mul(KNOB.r + 0.12), dir.mul(KNOB.r + reach), 0.016));
  }
  for (const band of [0, 1]) {
    const a = KNOB.swing * (band * 2 - 1), at = vec2(Math.sin(a), Math.cos(a)).mul(KNOB.r + 0.6);
    marks = min(marks, numeral(k.sub(at), band, 0.26));
  }
  color = mix(color, rgb(CREAM).mul(0.72), coverage(marks));

  for (const [x, y, slot] of [[-1, 1, 0.4], [1, 1, 1.9], [-1, -1, 2.6], [1, -1, 0.9]] as const) color = screw(color, q, x * (PLATE.w - 0.28), y * (PLATE.h - 0.28), slot);
  return sprite(2 * CABINET.w + 0.2, 2 * CABINET.h + 0.2, 0, 0, 1, lit(color), coverage(outer));
}

export interface DialLevels { needle: Level; station: Level; lock: Level; flare: Level }

export function dial(accent: number, levels: DialLevels): THREE.Mesh {
  const q = chart();
  const cx = (DIAL.x0 + DIAL.x1) / 2, hx = (DIAL.x1 - DIAL.x0) / 2, hy = (DIAL.y1 - DIAL.y0) / 2;
  const glass = sdBox(q.sub(vec2(cx, DIAL_Y)), vec2(hx, hy), 0.2);
  const lamps = exp(q.x.sub(BANDS[0] - 1.4).div(2.6).pow(2).negate()).add(exp(q.x.sub(BANDS[1] + 1.4).div(2.6).pow(2).negate()));
  const v = q.y.sub(DIAL.y0).div(DIAL.y1 - DIAL.y0);
  const back = mix(rgb(DIAL_EDGE), rgb(DIAL_WARM), clamp(lamps.mul(0.55).add(0.35), 0, 1).mul(smoothstep(-0.1, 0.45, v).mul(0.25).add(0.75)));
  const column = exp(q.x.sub(levels.station).div(0.42).pow(2).negate()).mul(smoothstep(1.1, 0, v).mul(0.6).add(0.4)).mul(levels.flare.mul(1.6).add(1));
  const wave = length(q.sub(vec2(levels.station, STATION_Y)).mul(vec2(1, 1.4))).sub(levels.flare.oneMinus().mul(1.1).add(0.25));
  let color: N<'vec3'> = back.mul(0.78).add(rgb(0xff7a1c).mul(column.mul(0.75).add(exp(wave.div(0.05).pow(2).negate()).mul(levels.flare).mul(1.2))));

  const along = abs(q.x.sub(MID)).sub(hx - 0.32);
  const minor = max(max(abs(fract(q.x.sub(BANDS[0]).div(TICK).add(0.5)).sub(0.5)).mul(TICK).sub(0.009), abs(q.y.sub(SCALE_Y + 0.045)).sub(0.045)), along);
  const major = max(max(abs(fract(q.x.sub(BANDS[0]).div(MAJOR).add(0.5)).sub(0.5)).mul(MAJOR).sub(0.013), abs(q.y.sub(SCALE_Y + 0.1)).sub(0.1)), along);
  const base = max(abs(q.y.sub(SCALE_Y)).sub(0.011), along);
  let ink: N<'float'> = min(min(minor, major), base);
  for (const band of [0, 1]) ink = min(ink, numeral(q.sub(vec2(bandX(band), DIGIT_Y)), band, DIGIT_H));
  color = mix(color, rgb(INK), coverage(ink).mul(0.92));
  color = color.add(vec3(0.06).mul(smoothstep(0.7, 1, v)));
  const pip = sdBox(q.sub(vec2(levels.station, STATION_Y)), vec2(0.17, 0.045), 0.045);
  color = mix(color, rgb(0x4a1c04), coverage(pip.sub(0.018)));
  color = mix(color, mix(rgb(0xff6a10), rgb(0xffd08a), exp(abs(q.x.sub(levels.station)).div(0.08).negate())).mul(levels.flare.add(1).mul(2.4)), coverage(pip));

  const nx = q.x.sub(levels.needle);
  const halo = exp(nx.div(0.16).pow(2).add(q.y.sub((NEEDLE.y0 + NEEDLE.y1) / 2).div(0.5).pow(2)).negate()).mul(levels.lock);
  color = color.add(rgb(accent).mul(halo.mul(1.1)));
  const tall = 0.16, wide = 0.085, slant = Math.hypot(tall, wide);
  const arrow = max(abs(nx).mul(tall).add(q.y.sub(NEEDLE.y1).mul(wide)).div(slant), float(NEEDLE.y1 - tall).sub(q.y));
  const needle = min(sdSegment(q, vec2(levels.needle, NEEDLE.y0), vec2(levels.needle, NEEDLE.y1 - tall), 0.024), arrow);
  color = mix(color, rgb(0x1a0d2e), coverage(needle.sub(0.012)));
  color = mix(color, rgb(accent).mul(levels.lock.mul(0.5).add(0.85)), coverage(needle));
  return sprite(2 * hx + 0.1, 2 * hy + 0.1, cx, DIAL_Y, 3, color, coverage(glass));
}

export function screen(accent: number, rows: THREE.DataTexture, scroll: Level, head: Level): THREE.Mesh {
  const q = chart(), table = texture(rows);
  const cx = (SCREEN.x0 + SCREEN.x1) / 2, cy = (SCREEN.y0 + SCREEN.y1) / 2, hx = (SCREEN.x1 - SCREEN.x0) / 2, hy = (SCREEN.y1 - SCREEN.y0) / 2;
  const load = (m: N<'float'>): N<'vec4'> => table.load(uvec2(vec2(clamp(m, 0, ROWS), 0)));
  const u = float(SCREEN.y1).sub(q.y).div(ROW).add(1).sub(scroll);
  const m = max(floor(u), select(scroll.greaterThan(0), float(0), float(1))), f = u.sub(m);
  const own = load(m), above = load(m.sub(1)), below = load(m.add(1));
  const column = floor(q.x.sub(SCREEN.x0).div(ROW));

  const rowColor = (k: N<'float'>, row: N<'vec4'>): N<'vec3'> => {
    const j = head.add(1).sub(k);
    const speck = grain(vec2(column.mul(1.618034).add(0.1375), mod(j, 4096).mul(1.324718).add(0.2917))).x;
    const hiss = speck.mul(speck).mul(speck);
    const pass = cover(abs(q.x.sub(bandOf(row.y))).sub(PASS));
    const hush = mix(rgb(STATIC_DARK), rgb(STATIC_LIGHT), hiss.mul(0.55));
    let color: N<'vec3'> = hush.add(rgb(accent).mul(pass.mul(hiss.mul(0.5).add(0.07))));
    const dx = q.x.sub(bandOf(row.x)), heard = row.z, power = mix(float(0.42), float(1.25), heard);
    color = color.add(rgb(CARRIER).mul(exp(dx.mul(dx).div(-0.05)).mul(power.mul(0.5))));
    color = mix(color, mix(rgb(CARRIER), rgb(CARRIER_HOT), heard).mul(power.mul(1.15)), cover(abs(dx).sub(0.024)));
    return mix(hush.mul(0.3), color, row.w);
  };

  const upper = f.lessThan(0.5);
  const other = select(upper, above, below), otherRow = select(upper, m.sub(1), m.add(1));
  const edge = min(f, f.oneMinus()).mul(ROW), px = fwidth(q.y);
  const offscreen = upper.and(m.add(scroll).lessThan(1.001));
  let color: N<'vec3'> = mix(rowColor(otherRow, other), rowColor(m, own), select(offscreen, float(1), clamp(edge.div(px).add(0.5), 0, 1)));

  const top = float(SCREEN.y1).sub(m.sub(1).add(scroll).mul(ROW)), bottom = top.sub(ROW);
  const j = head.add(1).sub(m);
  const tick = (count: N<'float'>, y: N<'float'>, live: N<'float'>) => {
    const tens = abs(count.sub(round(count.div(10)).mul(10))).lessThan(0.5), fifties = abs(count.sub(round(count.div(50)).mul(50))).lessThan(0.5);
    return cover(abs(q.y.sub(y)).sub(0.006)).mul(select(fifties, float(0.13), select(tens, float(0.06), float(0)))).mul(live);
  };
  const grid = max(tick(j, top, own.w), tick(j.sub(1), bottom, below.w));
  const verticals = cover(abs(fract(q.x.sub(BANDS[0]).div(MAJOR).add(0.5)).sub(0.5)).mul(MAJOR).sub(0.006)).mul(0.05).mul(own.w);
  color = color.add(rgb(0x8c96ff).mul(max(grid, verticals)));

  const hop = (newer: N<'vec4'>, older: N<'vec4'>, y: N<'float'>) => {
    const live = newer.w.mul(older.w).mul(step(0.5, abs(newer.x.sub(older.x))));
    const d = sdSegment(q, vec2(bandOf(older.x), y), vec2(bandOf(newer.x), y), 0.018);
    return { line: cover(d).mul(live), glow: exp(d.div(0.07).pow(2).negate()).mul(live) };
  };
  const a = hop(above, own, top), b = hop(own, below, bottom);
  color = color.add(rgb(CARRIER).mul(max(a.glow, b.glow).mul(0.6)));
  color = mix(color, rgb(CARRIER_HOT).mul(1.4), max(a.line, b.line));

  const box = sdBox(q.sub(vec2(cx, cy)), vec2(hx, hy), 0.24);
  color = color.mul(smoothstep(0, 0.45, box.negate()).mul(0.45).add(0.55));
  return sprite(2 * hx + 0.1, 2 * hy + 0.1, cx, cy, 2, color, coverage(box));
}

export function eye(signal: Level, lit: Lit): THREE.Mesh {
  const q = chart(), p = q.sub(vec2(EYE.x, EYE.y)), r = length(p), R = EYE.r;
  const unit = p.div(max(r, 1e-4));
  const bezel = mix(chrome(unit), chrome(unit.negate()), smoothstep(R * 0.9, R * 0.94, r));
  const bearing = atan(p.x, p.y), angle = abs(bearing), open = mix(float(0.95), float(0.05), signal);
  const shadow = open.sub(angle).mul(r);
  const target = abs(r.sub(R * 0.53)).sub(R * 0.24);
  const radial = smoothstep(R * 0.29, R * 0.77, r).mul(0.5).add(0.45).mul(sin(bearing.mul(36)).mul(0.06).add(0.94));
  const lip = exp(shadow.div(0.022).pow(2).negate()).mul(0.7);
  const glow = coverage(target).mul(coverage(shadow).mul(radial).add(lip.mul(coverage(target.add(0.01)))));
  const halo = exp(abs(target).div(0.09).negate()).mul(0.1).mul(coverage(target).oneMinus());
  let color: N<'vec3'> = rgb(0x03100a).mul(smoothstep(R * 0.86, R * 0.2, r).mul(0.6).add(0.4)).add(rgb(PHOSPHOR).mul(glow.mul(0.85).add(halo)));
  const cap = sdCircle(p, R * 0.25);
  color = mix(color, lit(chrome(unit).mul(0.35)).add(vec3(0.05).mul(exp(length(p.sub(vec2(-0.08, 0.1))).div(0.08).negate()))), coverage(cap));
  const crescent = coverage(sdCircle(p, R * 0.8)).mul(coverage(sdCircle(p.sub(vec2(0.1, -0.1)), R * 0.8).negate())).mul(0.07);
  color = color.add(vec3(crescent));
  color = mix(color, lit(bezel), coverage(ring(r.sub(R * 0.93), R * 0.07)));
  return sprite(2 * R + 0.1, 2 * R + 0.1, EYE.x, EYE.y, 3, color, coverage(r.sub(R)));
}

export function knob(turn: Level, lit: Lit): THREE.Mesh {
  const q = chart(), p = q.sub(vec2(KNOB.x, KNOB.y)), R = KNOB.r, r = length(p);
  const c = cos(turn), s = sin(turn);
  const k = vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
  const unit = p.div(max(r, 1e-4));
  const ridges = sin(atan(k.x, k.y).mul(52)).mul(0.5).add(0.5);
  const skirt = chrome(unit).mul(ridges.mul(0.32).add(0.7));
  const dome = dot(unit, light).mul(smoothstep(0, R * 0.76, r)).mul(0.5).add(1);
  const spot = exp(length(p.sub(vec2(-0.3, 0.36).mul(R))).div(R * 0.26).pow(2).negate()).mul(0.3);
  let color: N<'vec3'> = mix(skirt, rgb(BAKELITE).mul(dome.mul(1.6)).add(vec3(spot)), coverage(r.sub(R * 0.76)));
  color = mix(color, rgb(0x050506), coverage(ring(r.sub(R * 0.77), 0.018)));
  color = mix(color, rgb(CREAM), coverage(sdSegment(k, vec2(0, R * 0.2), vec2(0, R * 0.66), 0.034)));
  const drop = smoothstep(R + 0.25, R - 0.1, length(p.sub(vec2(0.1, -0.14)))).mul(0.55);
  const body = coverage(r.sub(R));
  const alpha = max(body, drop);
  return sprite(2 * R + 0.8, 2 * R + 0.8, KNOB.x, KNOB.y, 4, lit(color).mul(body).div(max(alpha, 1e-4)), alpha);
}
