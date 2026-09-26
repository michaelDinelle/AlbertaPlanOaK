import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dFdx, dFdy, dot, exp, float, floor, fract, length, max, min, mix, mod, positionGeometry, pow, round, select, sin, smoothstep, step, vec2 } from 'three/tsl';
import { coverage, fractal, grain, rgb, sdBox, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Paint = (p: N<'vec2'>, px: N<'float'>) => readonly [N<'vec3'>, N<'float'>];

export const PALETTE = {
  void: 0x06080d,
  exterior: 0x232a39,
  wall: 0x46546f,
  seam: 0x6f83aa,
  mat: 0x3a414e,
  slab: 0x1b2029,
  amber: 0xf2b134,
  hazard: 0x16181d,
  shade: 0x07090d,
  steel: 0x8e9ab0,
  girder: 0x3d4960,
  lamp: 0xfff0d2,
  net: 0x88a3c4,
  carbon: 0x283040,
  motor: 0x505b70,
  shell: 0xf2f7fc,
  blade: 0xe3eaf3,
  port: 0xff4d5e,
  starboard: 0x49f59a,
  plume: 0xc4e9ff,
  gold: 0xffd166,
  ember: 0xff6a3d,
  wind: 0xd2ecff,
  lens: 0xff3b4e,
} as const;

const STRIPE = 0.045;
const NET = 0.2;
const POST = 0.045;
export const PROP = 0.075;
export const LIFT = 0.05;
const TRUSS = { low: 0.07, high: 0.36, chord: 0.035, bay: 0.5 } as const;
const COLUMN = { gap: 0.62, half: 0.09 } as const;
export const RING = 0.13;
export const HALO = 0.42;

export interface Arena { wall: number; ceiling: number; reach: number; low: number; high: number; arm: number }

const OUTLINE = 0.008;
const fill = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const stroke = (d: N<'float'>, half: number, px: N<'float'>): N<'float'> => fill(abs(d).sub(half), px);
const layer = (color: N<'vec3'>, d: N<'float'>, tint: N<'vec3'>, px: N<'float'>): N<'vec3'> => mix(mix(color, rgb(PALETTE.shade), fill(d.sub(OUTLINE), px)), tint, fill(d, px));
const band = (v: N<'float'>, from: number, to: number): N<'float'> => abs(v.sub((from + to) / 2)).sub((to - from) / 2);

export const lampXs = (wall: number): number[] => [-0.75, -0.25, 0.25, 0.75].map((k) => k * wall);
const cameraAt = (a: Arena): readonly [number, number] => [a.wall + STRIPE + NET + POST + COLUMN.gap - 0.2, a.ceiling - 0.62];
export const lensAt = (a: Arena): readonly [number, number] => [cameraAt(a)[0] - 0.075, cameraAt(a)[1]];

export const flat = (width: number, height: number): THREE.PlaneGeometry => new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);

export function sprite(x0: number, y0: number, x1: number, y1: number, order: number, paint: Paint, additive = false): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  if (additive) material.blending = THREE.AdditiveBlending;
  const p = vec2(positionGeometry.x, positionGeometry.z.negate());
  const [color, alpha] = paint(p, max(length(vec2(dFdx(p.x), dFdy(p.x))), 1e-6));
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(x1 - x0, y1 - y0).translate((x0 + x1) / 2, 0, -(y0 + y1) / 2), material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function hall(lit: Lit, a: Arena, accent: number, launch: N<'vec2'>, known: N<'float'>): Paint {
  const { wall, ceiling, reach, low, high, arm } = a;
  const span = wall + STRIPE, netOut = span + NET, postOut = netOut + POST, top = ceiling + TRUSS.high + TRUSS.chord;
  const column = postOut + COLUMN.gap, [cameraX, cameraY] = cameraAt(a);
  return (p, px) => {
    const x = abs(p.x), y = p.y;
    const edge = max(x.sub(wall), max(y.negate(), y.sub(ceiling)));
    const cage = fill(edge, px);
    const mottle = fractal(p.mul(0.9), 3).mul(0.035).add(1);
    const g = abs(fract(p.add(0.5)).sub(0.5));
    const grid = stroke(min(g.x, g.y), 0.0045, px);
    const half = abs(fract(p.mul(2).add(0.5)).sub(0.5)).mul(0.5);
    const tick = fill(max(min(half.x, half.y).sub(0.0035), max(half.x, half.y).sub(0.035)), px);
    let albedo = mix(rgb(PALETTE.exterior), rgb(PALETTE.wall), cage).mul(mottle);
    albedo = mix(albedo, rgb(PALETTE.seam), max(grid.mul(0.5), tick.mul(0.4)).mul(cage));

    const corner = vec2(x.sub(reach), select(y.lessThan((low + high) / 2), y.sub(low).negate(), y.sub(high)));
    const bracket = min(max(abs(corner.y).sub(0.007), max(corner.x.negate().sub(0.22), corner.x.sub(0.007))), max(abs(corner.x).sub(0.007), max(corner.y.negate().sub(0.22), corner.y.sub(0.007))));
    albedo = mix(albedo, rgb(PALETTE.gold).mul(0.8), fill(bracket, px).mul(0.7));

    const l = p.sub(launch);
    const level = max(abs(l.y).sub(0.006), abs(l.x).sub(arm));
    const ends = max(abs(abs(l.x).sub(arm)).sub(0.006), abs(l.y).sub(0.05));
    const gap = abs(fract(l.x.div(0.05)).sub(0.5)).sub(0.25).mul(0.05);
    const pad = min(max(level, gap), ends);

    const mat = fill(y.add(STRIPE), px);
    const slab = fill(y.add(0.28), px);
    const seams = stroke(abs(fract(p.x.add(0.5)).sub(0.5)), 0.004, px).mul(0.5);
    const face = mix(rgb(PALETTE.mat), rgb(PALETTE.mat).mul(0.7), smoothstep(-STRIPE, -0.28, y)).mul(fractal(vec2(p.x.mul(7), y.mul(30)), 3).mul(0.06).add(1));
    const floorColor = mix(mix(face, rgb(PALETTE.shade), seams), rgb(PALETTE.slab), slab);
    albedo = mix(albedo, floorColor, mat);
    albedo = mix(albedo, rgb(PALETTE.shade), stroke(y.add(0.28), 0.005, px).mul(mat));

    const netZone = fill(max(band(x, span, netOut), max(y.negate().sub(STRIPE), y.sub(ceiling + STRIPE))), px);
    const d1 = abs(fract(p.x.add(y).div(0.07)).sub(0.5)).mul(0.07), d2 = abs(fract(p.x.sub(y).div(0.07)).sub(0.5)).mul(0.07);
    albedo = mix(albedo, rgb(PALETTE.net), stroke(min(d1, d2), 0.003, px).mul(netZone).mul(0.8));
    const rope = max(band(x, netOut - 0.012, netOut), max(y.negate().sub(STRIPE), y.sub(ceiling + STRIPE)));
    albedo = mix(albedo, rgb(PALETTE.net), fill(rope, px));
    const post = max(band(x, netOut, postOut), max(y.add(STRIPE).negate(), y.sub(top)));
    albedo = layer(albedo, post, rgb(PALETTE.steel).mul(0.8), px);

    const beam = max(band(x, column - COLUMN.half, column + COLUMN.half), y.sub(top + 0.3));
    const flange = max(min(band(x, column - COLUMN.half, column - COLUMN.half + 0.025), band(x, column + COLUMN.half - 0.025, column + COLUMN.half)), y.sub(top + 0.3));
    albedo = layer(albedo, beam, rgb(PALETTE.girder), px);
    albedo = mix(albedo, rgb(PALETTE.girder).mul(1.5), fill(flange, px));
    const bracketArm = max(band(y, cameraY - 0.012, cameraY + 0.012), band(x, cameraX, column - COLUMN.half));
    const camera = sdBox(vec2(x.sub(cameraX), y.sub(cameraY)), vec2(0.075, 0.045), 0.012);
    albedo = layer(albedo, min(bracketArm, camera), rgb(PALETTE.girder).mul(1.25), px);

    const trussZone = band(x, 0, postOut);
    const lowChord = max(band(y, ceiling + TRUSS.low, ceiling + TRUSS.low + TRUSS.chord), trussZone);
    const highChord = max(band(y, ceiling + TRUSS.high, ceiling + TRUSS.high + TRUSS.chord), trussZone);
    const webLow = ceiling + TRUSS.low + TRUSS.chord, webHigh = ceiling + TRUSS.high, panels = Math.max(2, Math.round(2 * postOut / TRUSS.bay)), pitch = 2 * postOut / panels;
    const k = clamp(floor(p.x.add(postOut).div(pitch)), 0, panels - 1);
    const diagonal = (i: N<'float'>): N<'float'> => {
      const x0 = i.mul(pitch).sub(postOut), up = mod(i, 2);
      return sdSegment(p, vec2(x0, mix(float(webLow), float(webHigh), up)), vec2(x0.add(pitch), mix(float(webHigh), float(webLow), up)), 0.009);
    };
    const web = max(min(diagonal(k), min(diagonal(max(k.sub(1), 0)), diagonal(min(k.add(1), panels - 1)))), trussZone);
    albedo = mix(albedo, rgb(PALETTE.girder).mul(1.2), fill(web, px));
    albedo = layer(albedo, min(lowChord, highChord), rgb(PALETTE.steel), px);

    const lamp = clamp(round(p.x.div(wall).add(0.75).div(0.5)), 0, 3).mul(0.5).sub(0.75).mul(wall);
    const housing = sdBox(vec2(p.x.sub(lamp), y.sub(ceiling + 0.2)), vec2(0.17, 0.06), 0.02);
    const lens = sdBox(vec2(p.x.sub(lamp), y.sub(ceiling + 0.148)), vec2(0.13, 0.011), 0.006);
    albedo = layer(albedo, housing, rgb(PALETTE.carbon), px);

    const floorStripe = max(band(y, -STRIPE, 0), x.sub(span));
    const ceilStripe = max(band(y, ceiling, ceiling + STRIPE), x.sub(span));
    const wallStripe = max(band(x, wall, span), max(y.negate().sub(STRIPE), y.sub(ceiling + STRIPE)));
    const stripes = min(floorStripe, min(ceilStripe, wallStripe));
    const hazard = fill(abs(fract(p.x.add(y).div(0.14)).sub(0.5)).sub(0.25).mul(0.14), px);
    albedo = layer(albedo, stripes, mix(rgb(PALETTE.amber), rgb(PALETTE.hazard), hazard.mul(0.9)), px);

    let color = lit(albedo);
    color = mix(color, rgb(PALETTE.amber).mul(1.25), stroke(edge, 0.0045, px).mul(fill(stripes.sub(0.005), px)));
    color = mix(color, rgb(accent).mul(0.9), fill(pad, px).mul(known).mul(0.8));
    color = mix(color, rgb(PALETTE.lamp).mul(2.6), fill(lens, px));
    color = mix(color, rgb(PALETTE.lens).mul(1.8), fill(length(vec2(x.sub(cameraX - 0.075), y.sub(cameraY))).sub(0.024), px));
    return [color, float(1)];
  };
}

export function envelope(a: Arena, near: N<'vec4'>, at: N<'vec2'>, crash: N<'vec4'>, wreck: N<'vec2'>): Paint {
  const { wall, ceiling } = a, span = wall + STRIPE;
  return (p, px) => {
    const x = p.x, y = p.y;
    const floorD = max(band(y, -STRIPE, 0), abs(x).sub(span));
    const ceilD = max(band(y, ceiling, ceiling + STRIPE), abs(x).sub(span));
    const leftD = max(band(x, -span, -wall), max(y.negate().sub(STRIPE), y.sub(ceiling + STRIPE)));
    const rightD = max(band(x, wall, span), max(y.negate().sub(STRIPE), y.sub(ceiling + STRIPE)));
    const inside = fill(max(abs(x).sub(wall), max(y.negate(), y.sub(ceiling))), px);
    const bell = (d: N<'float'>): N<'float'> => exp(d.mul(d).mul(-5));
    const faces = [
      [fill(floorD, px), exp(y.mul(-16)), x, 'x'],
      [fill(ceilD, px), exp(y.sub(ceiling).mul(16)), x, 'y'],
      [fill(leftD, px), exp(x.add(wall).mul(-16)), y, 'z'],
      [fill(rightD, px), exp(x.sub(wall).mul(16)), y, 'w'],
    ] as const;
    let glow: N<'float'> = float(0), burn: N<'float'> = float(0);
    faces.forEach(([face, wash, along, key], i) => {
      const shape = max(face, wash.mul(inside).mul(0.45));
      glow = glow.add(near[key].mul(bell(along.sub(i < 2 ? at.x : at.y))).mul(shape));
      burn = burn.add(crash[key].mul(bell(along.sub(i < 2 ? wreck.x : wreck.y))).mul(shape));
    });
    const total = glow.add(burn);
    return [mix(rgb(PALETTE.amber).mul(2.4), rgb(PALETTE.ember).mul(2.8), clamp(burn.div(max(total, 1e-4)), 0, 1)), clamp(total, 0, 1)];
  };
}

export function waypoint(closeness: N<'float'>, pulse: N<'float'>, spin: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p);
    const ring = stroke(r.sub(RING), 0.0075, px);
    const arm = max(min(abs(p.x), abs(p.y)).sub(0.006), band(max(abs(p.x), abs(p.y)), RING + 0.03, RING + 0.085));
    const turn = atan(p.y, p.x).div(2 * Math.PI).mul(18).add(spin);
    const dash = abs(fract(turn).sub(0.5)).sub(0.24).mul(r.mul(2 * Math.PI / 18));
    const outer = fill(max(abs(r.sub(RING + 0.11)).sub(0.0045), dash), px);
    const dot = fill(r.sub(0.02), px);
    const halo = exp(r.mul(-10)).mul(float(1).sub(smoothstep(0.3, 0.48, r))).mul(closeness.mul(0.5).add(0.18).add(pulse.mul(0.6)));
    const shape = max(max(ring, fill(arm, px)), max(dot, outer.mul(0.6)));
    const gain = closeness.mul(1.6).add(1.15).add(pulse.mul(2.5));
    return [rgb(PALETTE.gold).mul(gain), clamp(max(shape, halo), 0, 1)];
  };
}

export function ghostRing(fade: N<'float'>): Paint {
  return (p, px) => [rgb(PALETTE.gold).mul(1.4), max(stroke(length(p).sub(RING), 0.006, px), fill(length(p).sub(0.016), px)).mul(fade)];
}

export function guides(drone: N<'vec2'>, goal: N<'vec2'>, from: N<'vec2'>, hop: N<'float'>, closeness: N<'float'>): Paint {
  const dashed = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, trim: number, period: number, px: N<'float'>): N<'float'> => {
    const span = max(length(b.sub(a)), 1e-4), along = dot(p.sub(a), b.sub(a).div(span));
    const gap = abs(fract(along.div(period)).sub(0.5)).sub(0.22).mul(period);
    const ends = max(float(trim).sub(along), along.sub(span.sub(trim)));
    return fill(max(max(sdSegment(p, a, b, 0).sub(0.005), gap), ends), px);
  };
  return (p, px) => {
    const tether = dashed(p, drone, goal, RING + 0.02, 0.11, px).mul(closeness.mul(-0.4).add(0.7));
    const jump = dashed(p, from, goal, RING + 0.02, 0.06, px).mul(hop);
    return [rgb(PALETTE.gold).mul(1.2), max(tether, jump)];
  };
}

function parts(p: N<'vec2'>, arm: number) {
  const q = vec2(abs(p.x).sub(arm), p.y);
  return {
    q,
    beam: sdSegment(p, vec2(-arm, 0), vec2(arm, 0), 0.014),
    hull: sdBox(p.sub(vec2(0, 0.006)), vec2(0.085, 0.03), 0.024),
    motor: sdBox(q.sub(vec2(0, 0.024)), vec2(0.02, 0.024), 0.007),
    disc: sdBox(q.sub(vec2(0, LIFT + 0.006)), vec2(PROP, 0.005), 0.005),
  };
}

export function drone(lit: Lit, accent: number, arm: number, right: N<'float'>, left: N<'float'>, spawn: N<'float'>): Paint {
  return (p, px) => {
    const { q, beam, hull, motor, disc } = parts(p, arm), starboard = p.x.greaterThan(0);
    const bell = sdBox(q.sub(vec2(0, 0.042)), vec2(0.022, 0.007), 0.005);
    const hub = length(q.sub(vec2(0, LIFT + 0.006))).sub(0.009);
    const reach = abs(cos(select(starboard, right, left))).mul(PROP);
    const blade = sdSegment(q, vec2(reach.negate(), LIFT + 0.006), vec2(reach, LIFT + 0.006), 0.0055);
    const led = length(q.sub(vec2(0, -0.004))).sub(0.0085);
    const strip = max(abs(p.y.sub(0.002)).sub(0.0045), abs(p.x).sub(0.064));
    const visor = sdBox(p.sub(vec2(0, 0.022)), vec2(0.045, 0.008), 0.006);
    let albedo = layer(rgb(PALETTE.shade), beam, rgb(PALETTE.carbon), px);
    albedo = mix(albedo, rgb(PALETTE.steel).mul(0.75), stroke(p.y.sub(0.007), 0.0022, px).mul(fill(beam, px)));
    albedo = layer(albedo, motor, rgb(PALETTE.motor), px);
    albedo = mix(albedo, rgb(PALETTE.steel), fill(bell, px));
    albedo = layer(albedo, hull, mix(rgb(PALETTE.shell).mul(0.72), rgb(PALETTE.shell), smoothstep(-0.024, 0.036, p.y)), px);
    albedo = mix(albedo, rgb(PALETTE.carbon), fill(visor, px));
    let color = lit(albedo).add(rgb(accent).mul(0.12));
    color = mix(color, rgb(accent).mul(2.4), fill(strip, px));
    color = mix(color, rgb(PALETTE.blade), fill(disc, px).mul(0.35));
    color = mix(color, lit(rgb(PALETTE.blade)).add(0.1), fill(blade, px));
    color = mix(color, lit(rgb(PALETTE.steel)), fill(hub, px));
    color = mix(color, select(starboard, rgb(PALETTE.starboard), rgb(PALETTE.port)).mul(3.4), fill(led, px));
    color = color.add(rgb(accent).mul(spawn.mul(1.8)));
    const solid = max(fill(min(min(beam, hull), motor).sub(OUTLINE), px), max(fill(blade, px), fill(hub, px)));
    return [color, max(solid, max(fill(disc, px).mul(0.35), fill(led, px)))];
  };
}

export function halo(accent: number, strength: N<'float'>): Paint {
  return (p) => [rgb(accent).mul(0.9), pow(clamp(float(1).sub(length(p).div(HALO)), 0, 1), 3).mul(strength)];
}

export function wreck(arm: number, fade: N<'float'>): Paint {
  return (p, px) => {
    const { beam, hull, motor, disc } = parts(p, arm), d = min(min(beam, hull), min(motor, disc));
    return [rgb(PALETTE.ember).mul(2.2), max(stroke(d, 0.0045, px), fill(d, px).mul(0.4)).mul(fade)];
  };
}

export function plume(power: N<'float'>, flow: N<'float'>, reach: number): Paint {
  return (p) => {
    const span = max(power.mul(reach), 1e-3), t = p.y.negate().div(span);
    const along = smoothstep(0, 0.08, t).mul(float(1).sub(smoothstep(0.35, 1, t)));
    const width = clamp(t, 0, 1).mul(0.055).add(0.026);
    const across = exp(pow(p.x.div(width), 2).mul(-2.4));
    const streams = pow(fract(p.y.mul(9).add(flow).add(abs(p.x).mul(14))), 3).mul(0.4).add(0.6);
    const strength = clamp(power, 0, 1).mul(0.55).add(clamp(power.sub(1), 0, 1).mul(0.35));
    return [rgb(PALETTE.plume).mul(1.5), along.mul(across).mul(streams).mul(strength)];
  };
}

export function wind(a: Arena, direction: N<'vec2'>, strength: N<'float'>, flow: N<'float'>, flash: N<'float'>): Paint {
  const lane = 0.21, period = 2.1;
  return (p, px) => {
    const across = vec2(direction.y.negate(), direction.x);
    const u = dot(p, direction).sub(flow), v = dot(p, across);
    const row = floor(v.div(lane)), shift = grain(vec2(row, 7.13)).x.mul(period * 5);
    const shifted = u.add(shift), cell = floor(shifted.div(period)), seed = grain(vec2(row, cell));
    const local = shifted.sub(cell.mul(period)), size = seed.x.mul(0.55).add(0.35);
    const offset = v.sub(row.add(0.5).mul(lane)).sub(seed.y.sub(0.5).mul(lane * 0.4));
    const wave = sin(local.mul(8).add(seed.x.mul(6.283))).mul(0.016);
    const head = clamp(local.sub(0.1).div(size), 0, 1);
    const along = abs(local.sub(size.mul(0.5).add(0.1))).sub(size.mul(0.5));
    const d = max(abs(offset.sub(wave)).sub(mix(float(0.001), float(0.0048), head)), along);
    const live = step(0.45, seed.y);
    const cage = fill(max(abs(p.x).sub(a.wall), max(p.y.negate(), p.y.sub(a.ceiling))), px);
    const alpha = fill(d, px).mul(pow(head, 1.4)).mul(live).mul(cage).mul(strength.mul(0.45).add(0.06)).mul(flash.mul(0.8).add(1));
    return [rgb(PALETTE.wind).mul(flash.mul(0.8).add(1.1)), alpha];
  };
}

export const GAUGE = 0.24;

export const gaugeAt = (a: Arena): readonly [number, number, number] => [-(a.wall + STRIPE + NET + POST + COLUMN.gap / 2), a.ceiling / 2, COLUMN.gap / 2 - COLUMN.half];

export function gauge(lit: Lit, direction: N<'vec2'>, strength: N<'float'>, flash: N<'float'>, reach: number): Paint {
  return (p, px) => {
    const r = length(p), sector = Math.PI / 4, angle = atan(p.y, p.x), k = round(angle.div(sector));
    const tick = max(abs(sin(angle.sub(k.mul(sector)))).mul(r).sub(0.004), band(r, GAUGE - 0.045, GAUGE - 0.018));
    const tip = direction.mul(max(strength.mul(GAUGE - 0.05), 0.001)), side = vec2(direction.y.negate(), direction.x);
    const shaft = sdSegment(p, vec2(0, 0), tip, 0.009);
    const barbs = min(sdSegment(p, tip, tip.sub(direction.mul(0.06)).add(side.mul(0.04)), 0.009), sdSegment(p, tip, tip.sub(direction.mul(0.06)).sub(side.mul(0.04)), 0.009));
    const arrow = min(shaft, select(strength.greaterThan(0.2), barbs, float(1)));
    const mount = max(band(p.y, -0.014, 0.014), band(p.x, -reach, -GAUGE + 0.01));
    let albedo = layer(rgb(PALETTE.shade), mount, rgb(PALETTE.girder).mul(1.25), px);
    albedo = layer(albedo, r.sub(GAUGE), rgb(PALETTE.carbon).mul(0.8), px);
    albedo = mix(albedo, rgb(PALETTE.steel), stroke(r.sub(GAUGE - 0.008), 0.006, px));
    albedo = mix(albedo, rgb(PALETTE.steel).mul(0.8), fill(tick, px));
    let color = lit(albedo);
    color = mix(color, rgb(PALETTE.wind).mul(flash.mul(2).add(0.7)), stroke(r.sub(GAUGE - 0.024), 0.003, px));
    color = mix(color, rgb(PALETTE.wind).mul(flash.mul(1.5).add(1.6)), fill(arrow, px));
    color = mix(color, rgb(PALETTE.steel), fill(r.sub(0.016), px));
    return [color, max(fill(min(r.sub(GAUGE), mount).sub(0.008), px), fill(arrow, px))];
  };
}

export interface Ribbon { mesh: THREE.Mesh; positions: Float32Array; attribute: THREE.BufferAttribute; points: number }

export function ribbon(points: number, color: number, gain: number): Ribbon {
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
  const fade = float(1).sub(attribute<'float'>('age', 'float'));
  material.colorNode = rgb(color).mul(fade.mul(gain).add(0.3));
  material.opacityNode = fade.mul(fade).mul(0.8).mul(coverage(abs(attribute<'float'>('side', 'float')).sub(1)));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return { mesh, positions, attribute: position, points };
}

export function lay(ribbon: Ribbon, path: Float32Array, head: number, tail: number): void {
  const { points, positions: out } = ribbon;
  let nx = 0, ny = 1;
  for (let i = points - 1; i >= 0; i--) {
    const a = Math.max(0, i - 1), b = Math.min(points - 1, i + 1);
    const dx = path[2 * a]! - path[2 * b]!, dy = path[2 * a + 1]! - path[2 * b + 1]!, chord = Math.hypot(dx, dy);
    if (chord > 1e-5) {
      nx = -dy / chord;
      ny = dx / chord;
    }
    const width = head + (tail - head) * i / (points - 1), x = path[2 * i]!, y = path[2 * i + 1]!, o = 6 * i;
    out[o] = x + nx * width;
    out[o + 2] = -(y + ny * width);
    out[o + 3] = x - nx * width;
    out[o + 5] = -(y - ny * width);
  }
  ribbon.attribute.needsUpdate = true;
}
