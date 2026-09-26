import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionGeometry, positionWorld, pow, round, select, sin, smoothstep, step, vec2 } from 'three/tsl';
import { coverage, fractal, grain, rgb, sdBox, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Paint = (p: N<'vec2'>) => readonly [N<'vec3'>, N<'float'>];

export const PALETTE = {
  night: 0x04060c,
  skyHigh: 0x0a0f2c,
  skyLow: 0x1d2a5a,
  dusk: 0x4a3a6e,
  glow: 0xd88a5f,
  ridgeFar: 0x191f3d,
  ridgeNear: 0x0c1022,
  haze: 0x252b45,
  star: 0xdfe6ff,
  steel: 0xaab4c8,
  gleam: 0xeef3fb,
  web: 0x6a7590,
  shade: 0x111623,
  pylon: 0x3a4460,
  tyre: 0x1b1f2a,
  deck: 0x2c3140,
  hazard: 0xf2b134,
  warn: 0xff4332,
  ember: 0xff5a2c,
  filament: 0xffd98a,
  cool: 0x55689f,
  push: 0x5ec1ff,
} as const;

export const CART = { bodyY: -0.035, halfW: 0.22, halfH: 0.085, wheelX: 0.14, wheelY: -0.155, wheel: 0.05 } as const;
const TOP = CART.wheelY - CART.wheel;
export const RAIL = { top: TOP, flange: 0.034, web: 0.13, chord: 0.024, panel: 0.2, bottom: TOP - 0.188 } as const;
export const LAMP = { x: 0.018, y: (RAIL.top + RAIL.bottom) / 2, radius: 0.015 } as const;
const HORIZON = -0.62;

const OUTLINE = 0.0045;
const stroke = (d: N<'float'>, half: number): N<'float'> => coverage(abs(d).sub(half));
const inside = (d: N<'float'>): N<'float'> => coverage(d.add(OUTLINE));
const layer = (color: N<'vec3'>, d: N<'float'>, tint: N<'vec3'>): N<'vec3'> => mix(mix(color, rgb(PALETTE.shade), coverage(d)), tint, inside(d));

export const flat = (width: number, height: number): THREE.PlaneGeometry => new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);

export function sprite(x0: number, y0: number, x1: number, y1: number, order: number, paint: Paint): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const [color, alpha] = paint(vec2(positionGeometry.x, positionGeometry.z.negate()));
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(x1 - x0, y1 - y0).translate((x0 + x1) / 2, 0, -(y0 + y1) / 2), material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function backdrop(lit: Lit): Paint {
  return () => {
    const q = vec2(positionWorld.x, positionWorld.z.negate()), y = q.y;
    const sky = mix(mix(rgb(PALETTE.dusk), rgb(PALETTE.skyLow), smoothstep(HORIZON, HORIZON + 0.8, y)), rgb(PALETTE.skyHigh), smoothstep(HORIZON + 0.5, 1.9, y));
    const cell = floor(q.div(0.13)), seed = grain(cell), spot = grain(cell.add(31.7));
    const star = clamp(float(0.5).sub(length(q.sub(cell.add(spot.mul(0.7).add(0.15)).mul(0.13))).sub(seed.y.mul(0.003).add(0.0022)).div(fwidth(q.x))), 0, 1);
    const far = fractal(vec2(q.x.mul(0.45), 0.37), 4).mul(0.14).add(HORIZON - 0.02);
    const near = fractal(vec2(q.x.mul(0.7).add(5.3), 1.91), 4).mul(0.2).add(HORIZON - 0.42);
    let color = sky.add(rgb(PALETTE.glow).mul(exp(abs(y.sub(HORIZON)).mul(-9)).mul(0.5)));
    color = color.add(rgb(PALETTE.star).mul(star.mul(step(seed.x, 0.14)).mul(seed.y.mul(0.8).add(0.2)).mul(smoothstep(HORIZON + 0.4, HORIZON + 1.4, y))));
    color = mix(color, rgb(PALETTE.ridgeFar), coverage(y.sub(far)));
    color = mix(color, rgb(PALETTE.ridgeNear), coverage(y.sub(near)));
    return [color.add(lit(rgb(PALETTE.haze))), float(1)];
  };
}

export function pylon(lit: Lit, height: number): Paint {
  const top = 0.045, foot = 0.13, bay = 0.22;
  const halfAt = (y: N<'float'>): N<'float'> => mix(float(top), float(foot), clamp(y.negate().div(height), 0, 1));
  return (p) => {
    const legs = min(sdSegment(p, vec2(-top, 0), vec2(-foot, -height), 0.011), sdSegment(p, vec2(top, 0), vec2(foot, -height), 0.011));
    const y0 = floor(p.y.negate().div(bay)).mul(-bay), y1 = y0.sub(bay), w0 = halfAt(y0), w1 = halfAt(y1);
    const braces = min(sdSegment(p, vec2(w0.negate(), y0), vec2(w1, y1), 0.0045), sdSegment(p, vec2(w0, y0), vec2(w1.negate(), y1), 0.0045));
    const cap = sdBox(p.add(vec2(0, 0.013)), vec2(top + 0.035, 0.013), 0.003);
    const dim = float(1).sub(smoothstep(0.1, height * 0.6, p.y.negate()).mul(0.8));
    const color = mix(rgb(PALETTE.pylon).mul(dim), rgb(PALETTE.steel).mul(0.55), coverage(cap));
    return [lit(color), coverage(min(min(legs, braces), cap))];
  };
}

export function rail(lit: Lit, track: number, accent: number): Paint {
  const flangeY = RAIL.top - RAIL.flange / 2, chordY = RAIL.bottom + RAIL.chord / 2, webTop = RAIL.top - RAIL.flange, webBottom = RAIL.bottom + RAIL.chord;
  const panels = Math.max(2, 2 * Math.round(track / RAIL.panel)), panel = 2 * track / panels;
  return (p) => {
    const flange = sdBox(p.sub(vec2(0, flangeY)), vec2(track, RAIL.flange / 2), 0.003);
    const chord = sdBox(p.sub(vec2(0, chordY)), vec2(track, RAIL.chord / 2), 0.003);
    const k = clamp(floor(p.x.add(track).div(panel)), 0, panels - 1);
    const diagonal = (i: N<'float'>): N<'float'> => {
      const x0 = i.mul(panel).sub(track), up = mod(i, 2);
      return sdSegment(p, vec2(x0, mix(float(webTop), float(webBottom), up)), vec2(x0.add(panel), mix(float(webBottom), float(webTop), up)), 0.0075);
    };
    const web = max(min(diagonal(k), min(diagonal(max(k.sub(1), 0)), diagonal(min(k.add(1), panels - 1)))), abs(p.x).sub(track - 0.018));
    const end = vec2(abs(p.x).sub(track), p.y.sub(LAMP.y));
    const post = sdBox(end.add(vec2(0.018, 0)), vec2(0.018, (RAIL.top - RAIL.bottom) / 2), 0.003);
    const housing = sdBox(end.sub(vec2(LAMP.x - 0.004, 0)), vec2(0.017, 0.025), 0.007);
    const node = p.x.sub(round(p.x.add(track).div(panel)).mul(panel).sub(track));
    const lower = coverage(p.y.sub(flangeY));
    const stripes = coverage(abs(fract(abs(p.x).add(p.y).div(0.034)).sub(0.5)).sub(0.25).mul(0.024));
    let color = mix(rgb(PALETTE.shade), rgb(PALETTE.web), coverage(web.add(0.003)));
    color = layer(color, chord, rgb(PALETTE.steel).mul(0.72));
    color = layer(color, flange, rgb(PALETTE.steel));
    color = mix(color, rgb(PALETTE.gleam), stroke(p.y.sub(RAIL.top - 0.0065), 0.0022).mul(inside(flange)));
    color = mix(color, rgb(PALETTE.shade), max(stroke(node, 0.0017).mul(lower), stroke(p.x.sub(round(p.x)), 0.0017)).mul(inside(flange)).mul(0.8));
    color = mix(color, rgb(accent), stroke(p.x, 0.0032).mul(inside(flange)));
    color = layer(color, post, mix(rgb(PALETTE.shade), rgb(PALETTE.hazard), stripes));
    color = layer(color, housing, rgb(PALETTE.deck));
    return [lit(color), coverage(min(min(min(flange, chord), web), min(post, housing)))];
  };
}

export function cart(lit: Lit, accent: number, roll: N<'float'>): Paint {
  const { bodyY, halfW, halfH, wheelX, wheelY, wheel } = CART;
  return (p) => {
    const w = p.sub(vec2(select(p.x.lessThan(0), float(-wheelX), float(wheelX)), wheelY)), r = length(w);
    const body = max(sdBox(p.sub(vec2(0, bodyY)), vec2(halfW, halfH), 0.032), r.sub(wheel + 0.013).negate());
    const rise = smoothstep(bodyY - halfH, bodyY + halfH, p.y);
    const deck = sdBox(p.sub(vec2(0, bodyY + halfH - 0.013)), vec2(halfW - 0.012, 0.013), 0.004);
    const skirt = sdBox(p.sub(vec2(0, bodyY - halfH + 0.015)), vec2(halfW - 0.012, 0.015), 0.004);
    const row = clamp(round(p.y.sub(bodyY).div(0.024)), -1, 1);
    const slot = sdBox(vec2(abs(p.x).sub(0.158), p.y.sub(bodyY).sub(row.mul(0.024))), vec2(0.03, 0.0036), 0.0034);
    const bolt = length(vec2(abs(p.x).sub(0.198), abs(p.y.sub(bodyY)).sub(0.046))).sub(0.0065);
    const mount = abs(length(p).sub(0.06)).sub(0.0035);
    const turned = vec2(w.x.mul(cos(roll)).add(w.y.mul(sin(roll))), w.y.mul(cos(roll)).sub(w.x.mul(sin(roll))));
    const rim = r.sub(wheel * 0.64);
    let color = layer(rgb(PALETTE.shade), body, rgb(accent).mul(mix(float(0.62), float(1.1), rise)));
    color = mix(color, rgb(PALETTE.deck), coverage(deck).mul(inside(body)));
    color = mix(color, rgb(accent).mul(0.42), coverage(skirt).mul(inside(body)));
    color = mix(color, rgb(PALETTE.shade), coverage(slot).mul(0.85));
    color = mix(color, rgb(accent).mul(0.45), coverage(mount));
    color = mix(color, rgb(PALETTE.steel), coverage(bolt));
    color = layer(color, r.sub(wheel), rgb(PALETTE.tyre));
    color = mix(color, rgb(PALETTE.steel).mul(0.85), coverage(rim));
    color = mix(color, rgb(PALETTE.shade), coverage(min(abs(turned.x), abs(turned.y)).sub(0.0042)).mul(coverage(rim)));
    color = mix(color, rgb(accent), coverage(r.sub(0.011)));
    return [lit(color), max(coverage(body), coverage(r.sub(wheel)))];
  };
}

export function rod(lit: Lit, reach: number, radius: number, lift: N<'float'>): Paint {
  return (p) => {
    const d = sdSegment(p, vec2(0, 0), vec2(reach, 0), radius);
    const hot = mix(rgb(PALETTE.ember), rgb(PALETTE.filament), lift).mul(mix(float(0.85), float(2.3), lift));
    const core = float(1.25).sub(smoothstep(0, radius, abs(p.y)).mul(0.5));
    const collar = sdBox(p.sub(vec2(0.03, 0)), vec2(0.055, radius * 1.5), 0.005);
    const color = layer(mix(rgb(PALETTE.shade), hot.mul(core), coverage(d.add(0.0035))), collar, lit(rgb(PALETTE.steel).mul(0.8)));
    return [color, max(coverage(d), coverage(collar))];
  };
}

export function hub(lit: Lit, radius: number, accent: number): Paint {
  return (p) => {
    const r = length(p);
    let color = layer(rgb(PALETTE.shade), r.sub(radius), rgb(PALETTE.steel));
    color = mix(color, rgb(PALETTE.shade), stroke(r.sub(radius * 0.6), 0.0022));
    color = mix(color, rgb(PALETTE.shade), coverage(sdBox(p.sub(vec2(radius * 0.8, 0)), vec2(radius * 0.14, 0.004), 0.002)));
    color = mix(color, rgb(accent), coverage(r.sub(radius * 0.3)));
    return [lit(color), coverage(r.sub(radius))];
  };
}

export function dial(reach: number, accent: number): Paint {
  const dashes = 120;
  return (p) => {
    const r = length(p), worth = p.y.div(max(r, 1e-4)).add(1).mul(0.5);
    const gap = abs(fract(atan(p.y, p.x).mul(dashes / (2 * Math.PI))).sub(0.5)).sub(0.2).mul(r.mul(2 * Math.PI / dashes));
    const ring = coverage(max(abs(r.sub(reach)).sub(0.003), gap)).mul(mix(float(0.14), float(0.75), worth.mul(worth)));
    const level = coverage(sdSegment(vec2(abs(p.x), p.y), vec2(reach - 0.035, 0), vec2(reach + 0.035, 0), 0.0028));
    const m = p.sub(vec2(0, reach + 0.06)), marker = coverage(sdBox(vec2(m.x.add(m.y), m.y.sub(m.x)).mul(Math.SQRT1_2), vec2(0.016, 0.016), 0.002));
    const color = mix(mix(mix(rgb(PALETTE.cool), rgb(PALETTE.filament), worth), rgb(PALETTE.cool).mul(1.3), level), rgb(accent).mul(1.6), marker);
    return [color, max(max(ring, level.mul(0.55)), marker.mul(0.95))];
  };
}

export function chevrons(flow: N<'float'>): Paint {
  return (p) => {
    let glow: N<'float'> = float(0);
    for (let i = 0; i < 3; i++) {
      const x = -0.05 + i * 0.05, tip = vec2(x + 0.012, 0);
      const d = min(sdSegment(p, vec2(x - 0.022, 0.034), tip, 0.0085), sdSegment(p, vec2(x - 0.022, -0.034), tip, 0.0085));
      glow = max(glow, coverage(d).mul(pow(float(1).sub(fract(flow.sub(i / 3))), 2).mul(0.7).add(0.3)));
    }
    return [rgb(PALETTE.push).mul(2.4), glow];
  };
}

export function lamp(glow: N<'float'>): Paint {
  return (p) => [rgb(PALETTE.warn).mul(glow.mul(0.45).add(0.5)), coverage(length(p).sub(LAMP.radius))];
}

export function ghost(reach: number, radius: number, slant: N<'vec2'>, fade: N<'float'>): Paint {
  const { bodyY, halfW, halfH, wheelX, wheelY, wheel } = CART;
  return (p) => {
    const body = sdBox(p.sub(vec2(0, bodyY)), vec2(halfW, halfH), 0.032);
    const wheels = length(vec2(abs(p.x).sub(wheelX), p.y.sub(wheelY))).sub(wheel);
    const shape = min(min(body, wheels), sdSegment(p, vec2(0, 0), slant.mul(reach), radius));
    return [rgb(PALETTE.ember).mul(1.4), max(stroke(shape, 0.0035), coverage(shape).mul(0.22)).mul(fade)];
  };
}

export interface Ribbon {
  mesh: THREE.Mesh;
  positions: Float32Array;
  attribute: THREE.BufferAttribute;
}

export function ribbon(points: number, reach: number): Ribbon {
  const positions = new Float32Array(points * 6), side = new Float32Array(points * 2), age = new Float32Array(points * 2), index: number[] = [];
  for (let i = 0; i < points; i++) {
    side[2 * i] = 1;
    side[2 * i + 1] = -1;
    age[2 * i] = age[2 * i + 1] = i / (points - 1);
    if (i > 0) index.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
  }
  const geometry = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', position);
  geometry.setAttribute('side', new THREE.BufferAttribute(side, 1));
  geometry.setAttribute('age', new THREE.BufferAttribute(age, 1));
  geometry.setIndex(index);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
  const fade = float(1).sub(attribute<'float'>('age', 'float')), worth = clamp(positionWorld.z.negate().div(reach).add(1).mul(0.5), 0, 1);
  material.colorNode = mix(rgb(PALETTE.ember), rgb(PALETTE.filament), worth).mul(fade.mul(1.1).add(0.25));
  material.opacityNode = fade.mul(fade).mul(0.9).mul(coverage(abs(attribute<'float'>('side', 'float')).sub(1)));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return { mesh, positions, attribute: position };
}
