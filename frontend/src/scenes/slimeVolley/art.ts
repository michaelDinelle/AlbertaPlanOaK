import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dFdx, dFdy, dot, exp, float, floor, fract, length, max, min, mix, mod, positionGeometry, positionWorld, pow, sign, sin, smoothstep, sqrt, step, vec2, vec3 } from 'three/tsl';
import { coverage, grain, noise, rgb, sdBox, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Paint = (p: N<'vec2'>, px: N<'float'>) => readonly [N<'vec3'>, N<'float'>];

export interface Court { half: number; ground: number; top: number; stub: number; radius: number; ball: number }

export const PALETTE = {
  void: 0x05070c,
  blockA: 0x283048,
  blockB: 0x313a55,
  mortar: 0x161b28,
  wainscot: 0x1d2436,
  trim: 0x3a4560,
  band: 0x34436b,
  pad: 0x1c2c52,
  padSeam: 0x0a1120,
  padEdge: 0x7188bf,
  concrete: 0x151a26,
  woodA: 0x8c5830,
  woodB: 0xa9703f,
  plankSeam: 0x2e1b0e,
  lip: 0xf0c68d,
  steel: 0x2d3548,
  steelLight: 0x56637f,
  housing: 0x0c1017,
  lens: 0xfff1d8,
  post: 0xe3e8f1,
  strap: 0x252d45,
  outline: 0x070a11,
  ball: 0xfff3dd,
  panel: 0xffd9a8,
  seam: 0x8a5a33,
  glow: 0xffc27a,
  rival: 0xc07bff,
  board: 0x0c0f16,
  window: 0x040508,
  digit: 0xffa53d,
  digitOff: 0x24170c,
  bulbOff: 0x191d27,
  gold: 0xffd166,
} as const;

export const CEILING = 23.4;
export const LAMPS = [11, 20] as const;
export const LENS = { width: 1.5, height: 0.14, y: CEILING - 0.64 } as const;
export const BANNER = { x: 15.5, half: 1.25, bottom: 15.9, notch: 0.9 } as const;
export const BOARD = { half: 7.8, bottom: 19, top: 22.4, cable: 5.6 } as const;
export const BULB = { first: 3.7, spacing: 0.8, radius: 0.27, y: 0.38 } as const;
export const EYE = { x: 0.36, y: 0.5, radius: 0.21, pupil: 0.1 } as const;
export const RING = 0.95;
export const SEVEN = [63, 6, 91, 79, 102, 109, 125, 7, 127, 111] as const;

const OUTLINE = 0.06;
const WHITE = vec3(1, 1, 1);
const fill = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const band = (v: N<'float'>, from: number, to: number): N<'float'> => abs(v.sub((from + to) / 2)).sub((to - from) / 2);
const turn = (p: N<'vec2'>, c: N<'float'> | number, s: N<'float'> | number): N<'vec2'> => vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
const cell = (v: N<'float'>, size: number): N<'float'> => float(0.5).sub(abs(fract(v.div(size)).sub(0.5))).mul(size);
const below = (ground: number, px: N<'float'>, depth: number, gain: number): N<'float'> => {
  const y = positionWorld.z.negate();
  return smoothstep(ground - depth, ground, y).mul(gain).mul(fill(y.sub(ground), px));
};

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

export function hall(lit: Lit, court: Court, accent: number): Paint {
  const { half, ground } = court, wainscot = ground + 2.6, pads = ground + 9.9;
  return (p, px) => {
    const x = p.x, y = p.y, ax = abs(x), up = y.sub(ground);
    const row = floor(up);
    const u = x.div(2).add(mod(row, 2).mul(0.5));
    const edge = min(float(0.5).sub(abs(fract(u).sub(0.5))).mul(2), cell(up, 1));
    const tone = grain(vec2(floor(u), row)).x;
    const block = mix(rgb(PALETTE.blockA), rgb(PALETTE.blockB), tone).mul(noise(p.mul(1.9)).mul(0.08).add(0.95));
    let wall = mix(rgb(PALETTE.mortar), block, fill(float(0.04).sub(edge), px));
    wall = mix(wall, rgb(PALETTE.band).mul(tone.mul(0.12).add(0.94)), fill(band(y, 12.4, 13.2), px).mul(0.9));
    wall = mix(wall, rgb(accent).mul(0.3), fill(min(abs(y.sub(12.26)), abs(y.sub(13.34))).sub(0.035), px).mul(0.7));
    const panel = mix(rgb(PALETTE.wainscot), rgb(PALETTE.mortar), fill(float(0.03).sub(cell(x.add(1.5), 3)), px));
    wall = mix(wall, panel, fill(y.sub(wainscot), px));
    wall = mix(wall, rgb(PALETTE.trim), fill(band(y, wainscot, wainscot + 0.2), px));
    wall = mix(wall, rgb(accent).mul(0.5), fill(abs(y.sub(wainscot + 0.3)).sub(0.035), px));
    wall = wall.mul(mix(float(1), float(0.72), smoothstep(9, CEILING, y)));

    const out = ax.sub(half), mat = sdBox(vec2(fract(out.div(3)).sub(0.5).mul(3), fract(up.div(3.3)).sub(0.5).mul(3.3)), vec2(1.42, 1.58), 0.28);
    const cushion = rgb(PALETTE.pad).mul(smoothstep(0, 0.7, mat.negate()).mul(0.45).add(0.6));
    let side = mix(rgb(PALETTE.padSeam), cushion, fill(mat, px));
    const concrete = rgb(PALETTE.concrete).mul(mix(float(0.8), float(1), fill(float(0.05).sub(cell(out.add(0.4), 2.6)), px)));
    side = mix(side, concrete, fill(float(pads).sub(y), px));
    side = mix(side, rgb(PALETTE.trim), fill(band(y, pads, pads + 0.18), px));
    side = mix(side, rgb(PALETTE.padEdge), fill(band(out, 0, 0.1), px));
    let color = mix(wall, side, fill(out.negate(), px));

    const depth = float(ground).sub(y), line = floor(depth.div(0.46)), shift = grain(vec2(line, 7)).x.mul(5.5);
    const board = x.add(shift).div(5.5), plank = vec2(floor(board), line);
    const wood = mix(rgb(PALETTE.woodA), rgb(PALETTE.woodB), grain(plank).x).mul(noise(vec2(x.mul(0.55).add(grain(plank).y.mul(40)), depth.mul(11))).mul(0.12).add(0.95));
    const seams = min(cell(depth, 0.46), cell(board, 1).mul(5.5));
    let floorColor = mix(rgb(PALETTE.plankSeam), wood, fill(float(0.022).sub(seams), px));
    floorColor = floorColor.mul(mix(float(1), float(0.34), smoothstep(0, 6, depth))).mul(mix(float(1), float(0.6), fill(out.negate(), px)));
    floorColor = mix(floorColor, rgb(PALETTE.lip), fill(band(y, ground - 0.07, ground), px));
    color = mix(color, floorColor, fill(y.sub(ground), px));

    const chord = min(band(y, CEILING, CEILING + 0.2), band(y, CEILING + 1.15, CEILING + 1.35));
    const k = floor(x.div(2.5)), x0 = k.mul(2.5), rise = mod(k, 2);
    const brace = sdSegment(p, vec2(x0, mix(float(CEILING + 0.2), float(CEILING + 1.15), rise)), vec2(x0.add(2.5), mix(float(CEILING + 1.15), float(CEILING + 0.2), rise)), 0.06);
    const truss = max(min(chord, brace), float(CEILING).sub(y));
    let roof = mix(rgb(PALETTE.void), rgb(PALETTE.steel), fill(truss, px));
    roof = mix(roof, rgb(PALETTE.steelLight), fill(band(y, CEILING + 0.14, CEILING + 0.2), px));
    color = mix(color, roof, fill(float(CEILING).sub(y), px));

    const bx = ax.sub(BANNER.x), hang = y.sub(BANNER.bottom), tall = CEILING - BANNER.bottom;
    const cloth = max(sdBox(vec2(bx, hang.sub(tall / 2)), vec2(BANNER.half, tall / 2), 0), float(BANNER.notch).mul(float(1).sub(abs(bx).div(BANNER.half))).sub(hang));
    const team = mix(rgb(PALETTE.rival), rgb(accent), fill(x.negate(), px));
    const fold = sin(bx.mul(5.2)).mul(0.08).add(0.92);
    let flag = team.mul(0.42).mul(fold);
    flag = mix(flag, team.mul(0.8).mul(fold), fill(abs(cloth.add(0.16)).sub(0.05), px));
    const crest = vec2(bx, hang.sub(BANNER.notch + 2.1)), dome = max(length(crest).sub(0.78), crest.y.negate());
    flag = mix(flag, team.mul(0.95), fill(dome, px));
    flag = mix(flag, vec3(0.92, 0.94, 0.97), fill(length(crest.sub(vec2(sign(x).negate().mul(0.27), 0.38))).sub(0.15), px));
    flag = mix(flag, rgb(PALETTE.outline), fill(length(crest.sub(vec2(sign(x).negate().mul(0.31), 0.38))).sub(0.07), px));
    flag = mix(flag, team.mul(0.8), fill(min(abs(hang.sub(BANNER.notch + 0.75)), abs(hang.sub(BANNER.notch + 0.95))).sub(0.05), px).mul(fill(cloth.add(0.16), px)));
    color = mix(color, mix(rgb(PALETTE.outline), flag, fill(cloth, px)), fill(cloth.sub(0.05), px));
    color = mix(color, rgb(PALETTE.steelLight), fill(sdBox(vec2(bx, y.sub(CEILING - 0.12)), vec2(BANNER.half + 0.22, 0.07), 0.05), px));

    const lampX = min(abs(ax.sub(LAMPS[0])), abs(ax.sub(LAMPS[1])));
    const housing = sdBox(vec2(lampX, y.sub(CEILING - 0.3)), vec2(0.95, 0.3), 0.08);
    const stem = sdBox(vec2(lampX, y.sub(CEILING + 0.1)), vec2(0.12, 0.2), 0);
    color = mix(color, rgb(PALETTE.housing), fill(min(housing, stem), px));
    const lens = sdBox(vec2(lampX, y.sub(LENS.y)), vec2(LENS.width / 2, LENS.height / 2), 0.05);
    return [lit(color).add(rgb(PALETTE.lens).mul(fill(lens, px).mul(2.2))), float(1)];
  };
}

export function fence(lit: Lit, court: Court, accent: number): Paint {
  const { ground, top, stub } = court;
  return (p, px) => {
    const capsule = max(sdSegment(p, vec2(0, ground), vec2(0, top), stub), float(ground).sub(p.y));
    const socket = sdBox(p.sub(vec2(0, ground - 0.16)), vec2(stub + 0.24, 0.16), 0.05);
    const round = sqrt(clamp(float(1).sub(p.x.div(stub).mul(p.x.div(stub))), 0, 1));
    let color = rgb(PALETTE.post).mul(mix(float(0.5), float(1.05), round));
    const straps = min(band(p.y, ground + 0.95, ground + 1.3), band(p.y, ground + 2.4, ground + 2.75));
    color = mix(color, rgb(PALETTE.strap).mul(mix(float(0.6), float(1.2), round)), fill(straps, px));
    color = mix(color, rgb(accent).mul(mix(float(0.6), float(1.1), round)), fill(min(abs(p.y.sub(ground + 1.125)), abs(p.y.sub(ground + 2.575))).sub(0.035), px));
    color = mix(color, rgb(PALETTE.steelLight).mul(1.4), fill(length(p.sub(vec2(0, top))).sub(0.11), px));
    color = mix(rgb(PALETTE.outline), color, fill(capsule, px));
    color = mix(color, mix(rgb(PALETTE.steel), rgb(PALETTE.steelLight), smoothstep(ground - 0.32, ground, p.y)), fill(socket, px));
    return [lit(color), max(fill(capsule.sub(OUTLINE), px), fill(socket, px))];
  };
}

export interface Look { look: N<'vec2'>; flash: N<'float'>; lid: N<'float'>; clock: N<'float'> }

export function slime(court: Court, tint: number, facing: number, u: Look, mirror: boolean): Paint {
  const R = court.radius;
  return (p0, px) => {
    const p = mirror ? vec2(p0.x, p0.y.negate()) : p0;
    const r = length(p), dome = max(r.sub(R), p.y.negate());
    const c = rgb(tint).mul(u.flash.mul(0.6).add(1));
    const height = clamp(p.y.div(R), 0, 1), inner = clamp(dome.negate().div(R * 0.5), 0, 1);
    let color = mix(c.mul(0.42), c.mul(1.04), smoothstep(0, 1, height));
    color = mix(c.mul(0.3), color, inner.mul(0.6).add(0.4));
    for (let i = 0; i < 3; i++) {
      const rise = fract(u.clock.mul(0.11 + 0.05 * i).add(i * 0.37)), bubble = vec2(facing * (-0.4 + 0.34 * i) * R, rise.mul(R * 0.62).add(R * 0.08));
      color = mix(color, c.mul(1.35).add(0.06), fill(length(p.sub(bubble)).sub(0.05 + 0.025 * i), px).mul(float(1).sub(rise).mul(0.55)));
    }
    const rim = fill(abs(r.sub(R - 0.12)).sub(0.035), px).mul(smoothstep(R * 0.25, R * 0.85, p.y));
    color = mix(color, c.mul(1.5).add(0.18), rim.mul(0.55));
    const gloss = turn(p.sub(vec2(-facing * 0.44 * R, 0.63 * R)), Math.cos(facing * 0.6), Math.sin(facing * 0.6));
    color = mix(color, color.mul(0.35).add(0.9), fill(length(vec2(gloss.x.div(0.3 * R), gloss.y.div(0.13 * R))).sub(1).mul(0.13 * R), px).mul(0.6));
    color = mix(color, WHITE, fill(length(p.sub(vec2(-facing * 0.2 * R, 0.83 * R))).sub(0.05 * R), px).mul(0.8));
    const e = p.sub(vec2(facing * EYE.x * R, EYE.y * R)), er = EYE.radius * R, pr = EYE.pupil * R;
    const sclera = length(e).sub(er), inside = fill(sclera, px), covered = e.y.sub(u.lid.mul(-2 * er).add(er)), open = fill(covered, px);
    color = mix(color, rgb(PALETTE.outline), fill(sclera.sub(0.045), px));
    color = mix(color, mix(c.mul(0.55), vec3(0.97, 0.98, 1), open), inside);
    const iris = e.sub(u.look);
    color = mix(color, rgb(PALETTE.outline), fill(length(iris).sub(pr), px).mul(inside).mul(open));
    color = mix(color, WHITE, fill(length(iris.sub(vec2(-0.03, 0.035))).sub(0.028), px).mul(inside).mul(open));
    color = mix(color, rgb(PALETTE.outline), fill(abs(covered).sub(0.025), px).mul(inside).mul(step(0.02, u.lid)));
    color = mix(rgb(PALETTE.outline), color, fill(dome, px));
    const alpha = fill(dome.sub(OUTLINE), px);
    return [color, mirror ? alpha.mul(below(court.ground, px, 1.8, 0.12)) : alpha];
  };
}

export function ball(court: Court, spin: N<'float'>, mirror: boolean): Paint {
  const r = court.ball;
  return (p0, px) => {
    const p = mirror ? vec2(p0.x, p0.y.negate()) : p0;
    const d = length(p).sub(r), q = turn(p, cos(spin), sin(spin));
    let seam: N<'float'> = float(1e3), panel: N<'float'> = float(0);
    for (let k = 0; k < 3; k++) {
      const a = (2 * Math.PI * k) / 3, arc = length(turn(q, Math.cos(a), Math.sin(a)).sub(vec2(0, -1.3 * r))).sub(1.02 * r);
      seam = min(seam, abs(arc));
      if (k === 0) panel = fill(arc, px);
    }
    const n = p.div(r), light = clamp(dot(vec2(-0.45, 0.6), n).add(sqrt(clamp(float(1).sub(dot(n, n)), 0, 1)).mul(0.72)), 0, 1);
    let color = mix(rgb(PALETTE.ball), rgb(PALETTE.panel), panel.mul(0.7)).mul(mix(float(0.48), float(1.12), light));
    color = mix(color, rgb(PALETTE.seam), fill(seam.sub(0.026), px).mul(0.85));
    color = mix(color, vec3(1.3, 1.3, 1.3), fill(length(p.sub(vec2(-0.2 * r, 0.3 * r))).sub(0.1 * r), px).mul(0.75));
    color = mix(rgb(PALETTE.outline), color, fill(d, px));
    const alpha = fill(d.sub(0.04), px);
    return [color, mirror ? alpha.mul(below(court.ground, px, 1.6, 0.16)) : alpha];
  };
}

export function halo(tint: number, radius: number, reach: number, strength: N<'float'>): Paint {
  return (p) => {
    const r = length(p);
    return [rgb(tint).mul(strength), exp(max(r.sub(radius), 0).mul(-2.6)).mul(float(1).sub(smoothstep(reach * 0.55, reach, r))).mul(0.55)];
  };
}

export function shadow(strength: N<'float'>): Paint {
  return (p) => [vec3(0, 0, 0), strength.mul(float(1).sub(smoothstep(0.3, 1, length(p))))];
}

export function ring(fraction: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p), arc = fract(atan(p.x, p.y).div(2 * Math.PI).add(1));
    const d = abs(r.sub(RING)).sub(0.055), on = fill(max(arc.sub(fraction), arc.negate()).mul(r.mul(2 * Math.PI)), px);
    return [rgb(PALETTE.gold).mul(1.6), fill(d, px).mul(mix(float(0.18), float(1), on))];
  };
}

export function pop(age: N<'float'>, tint: N<'vec3'>): Paint {
  return (p, px) => {
    const t = clamp(age, 0, 1), grow = float(1).sub(pow(float(1).sub(t), 2));
    const d = abs(length(p).sub(mix(float(0.35), float(1.2), grow))).sub(mix(float(0.1), float(0.02), t));
    return [tint.mul(1.8), fill(d, px).mul(pow(float(1).sub(t), 2))];
  };
}

export function puff(age: N<'float'>, radius: number): Paint {
  return (p, px) => {
    const t = clamp(age, 0, 1), grow = float(1).sub(pow(float(1).sub(t), 3)), side = vec2(abs(p.x), p.y);
    const blob = min(length(side.sub(vec2(mix(float(radius * 0.55), float(radius * 1.45), grow), mix(float(0.1), float(0.3), grow)))).sub(mix(float(0.16), float(0.34), grow)), length(side.sub(vec2(mix(float(radius * 0.35), float(radius * 0.95), grow), mix(float(0.05), float(0.18), grow)))).sub(mix(float(0.1), float(0.22), grow)));
    return [rgb(PALETTE.lip).mul(0.8), fill(blob, px).mul(pow(float(1).sub(t), 2)).mul(0.42)];
  };
}

export function strip(span: number, tint: N<'vec3'>, strength: N<'float'>): Paint {
  return (p) => [tint.mul(1.2), exp(abs(p.y).mul(-3.2)).mul(float(1).sub(smoothstep(span / 2 - 2, span / 2, abs(p.x)))).mul(strength)];
}

export function chevrons(accent: number, dir: N<'float'>, jump: N<'float'>, flow: N<'float'>): Paint {
  return (p, px) => {
    const q = vec2(p.x.mul(dir), p.y);
    let move: N<'float'> = float(0);
    for (let i = 0; i < 3; i++) {
      const x = (i - 1) * 0.42, d = min(sdSegment(q, vec2(x - 0.1, 0.17), vec2(x + 0.1, 0), 0.05), sdSegment(q, vec2(x - 0.1, -0.17), vec2(x + 0.1, 0), 0.05));
      move = max(move, fill(d, px).mul(pow(float(1).sub(fract(flow.sub(i / 3))), 2).mul(0.7).add(0.3)));
    }
    const caret = min(sdSegment(p, vec2(-0.2, 0.46), vec2(0, 0.66), 0.05), sdSegment(p, vec2(0.2, 0.46), vec2(0, 0.66), 0.05));
    const rest = fill(length(p).sub(0.07), px).mul(float(1).sub(abs(dir))).mul(float(1).sub(jump)).mul(0.45);
    return [rgb(accent).mul(0.95), max(max(move.mul(abs(dir)), fill(caret, px).mul(jump)), rest)];
  };
}

const SEGMENTS: readonly (readonly [number, number, boolean])[] = [[0, 1, false], [1, 0.5, true], [1, -0.5, true], [0, -1, false], [-1, -0.5, true], [-1, 0.5, true], [0, 0, false]];

function digit(p: N<'vec2'>, mask: N<'float'>, width: number, height: number, thick: number): readonly [N<'float'>, N<'float'>] {
  let on: N<'float'> = float(1e3), off: N<'float'> = float(1e3);
  SEGMENTS.forEach(([sx, sy, upright], k) => {
    const centre = vec2(sx * width, sy * height), extent = upright ? vec2(thick, height / 2 - thick * 0.6) : vec2(width - thick * 0.6, thick);
    const d = sdBox(p.sub(centre), extent, thick * 0.9), lit = mod(floor(mask.div(2 ** k)), 2);
    on = min(on, d.add(float(1).sub(lit).mul(1e3)));
    off = min(off, d.add(lit.mul(1e3)));
  });
  return [on, off];
}

export interface Board { lives: readonly [N<'float'>, N<'float'>]; pops: readonly [N<'float'>, N<'float'>]; digits: readonly [N<'float'>, N<'float'>, N<'float'>]; member: N<'float'>; result: N<'vec3'>; glow: N<'float'> }

export function scoreboard(lit: Lit, accent: number, u: Board, league: boolean, centre: number, count: number): Paint {
  const tints = [PALETTE.rival, accent] as const, slots = [-1.175, 0.275, 1.175] as const, half = (BOARD.top - BOARD.bottom) / 2;
  const middle = BULB.first + (count - 1) * BULB.spacing / 2, reach = (count - 1) * BULB.spacing / 2;
  return (p, px) => {
    const body = sdBox(p, vec2(BOARD.half, half), 0.22);
    const cables = sdSegment(vec2(abs(p.x), p.y), vec2(BOARD.cable, half), vec2(BOARD.cable, CEILING - centre), 0.04);
    const clockWindow = sdBox(p, vec2(1.95, 1.18), 0.14);
    const bulbWindow = sdBox(vec2(abs(p.x).sub(middle), p.y.sub(BULB.y)), vec2(reach + 0.5, 0.5), 0.24);
    let albedo = mix(rgb(PALETTE.steel), rgb(PALETTE.board), fill(body.add(0.1), px));
    albedo = mix(albedo, rgb(PALETTE.window), fill(min(clockWindow, bulbWindow), px));
    let color = mix(lit(rgb(PALETTE.steelLight)), lit(albedo), fill(body, px));
    color = color.add(u.result.mul(u.glow).mul(fill(abs(body.add(0.05)).sub(0.06), px)).mul(2.2));
    for (const side of [0, 1] as const) {
      const sign = side === 0 ? -1 : 1, tint = rgb(tints[side]), lives = u.lives[side];
      for (let i = 0; i < count; i++) {
        const at = vec2(sign * (BULB.first + i * BULB.spacing), BULB.y), b = length(p.sub(at)).sub(BULB.radius);
        const on = step(i + 0.5, lives), popped = u.pops[side].mul(float(1).sub(min(abs(lives.sub(i)), 1)));
        color = mix(color, mix(rgb(PALETTE.bulbOff), tint.mul(1.45), on).add(popped.mul(3)), fill(b, px));
        color = mix(color, vec3(1.6, 1.6, 1.6), fill(length(p.sub(at).sub(vec2(-0.08, 0.09))).sub(0.055), px).mul(on).mul(0.8));
        color = color.add(tint.mul(on.mul(exp(max(b, 0).mul(-7))).mul(0.08)));
      }
      color = mix(color, tint.mul(0.9), fill(sdBox(p.sub(vec2(sign * middle, -0.5)), vec2(reach + 0.25, 0.07), 0.07), px));
    }
    for (const k of [0, 1, 2] as const) {
      const [on, off] = digit(p.sub(vec2(slots[k], 0)), u.digits[k], 0.34, 0.72, 0.075);
      color = mix(color, rgb(PALETTE.digitOff), fill(off, px));
      color = mix(color, rgb(PALETTE.digit).mul(2.1), fill(on, px));
    }
    const colon = min(length(p.sub(vec2(-0.475, 0.3))), length(p.sub(vec2(-0.475, -0.3)))).sub(0.085);
    color = mix(color, rgb(PALETTE.digit).mul(2.1), fill(colon, px));
    if (league) {
      const tag = p.sub(vec2(-2.575, -0.05)), frame = abs(sdBox(tag, vec2(0.46, 0.72), 0.12)).sub(0.035), [on, off] = digit(tag, u.member, 0.24, 0.47, 0.07);
      color = mix(color, rgb(PALETTE.rival).mul(0.9), fill(frame, px));
      color = mix(color, rgb(PALETTE.rival).mul(0.14), fill(off, px));
      color = mix(color, rgb(PALETTE.rival).mul(2.6), fill(on, px));
    }
    return [color, max(fill(body, px), fill(cables, px))];
  };
}

export interface Ribbon { mesh: THREE.Mesh; positions: Float32Array; attribute: THREE.BufferAttribute; points: number }

export function ribbon(points: number, tint: number): Ribbon {
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
  material.colorNode = rgb(tint).mul(fade.mul(0.9).add(0.2));
  material.opacityNode = fade.mul(fade).mul(0.42).mul(coverage(abs(attribute<'float'>('side', 'float')).sub(1)));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return { mesh, positions, attribute: position, points };
}

export function lay(ribbon: Ribbon, path: Float32Array, used: number, head: number, tail: number): void {
  const { points, positions: out } = ribbon;
  let nx = 0, ny = 1;
  for (let i = points - 1; i >= 0; i--) {
    const j = Math.min(i, Math.max(0, used - 1)), a = Math.max(0, j - 1), b = Math.min(Math.max(0, used - 1), j + 1);
    const dx = path[2 * a]! - path[2 * b]!, dy = path[2 * a + 1]! - path[2 * b + 1]!, chord = Math.hypot(dx, dy);
    if (chord > 1e-5) {
      nx = -dy / chord;
      ny = dx / chord;
    }
    const width = i < used ? head + (tail - head) * i / (points - 1) : 0, x = path[2 * j]!, y = path[2 * j + 1]!, o = 6 * i;
    out[o] = x + nx * width;
    out[o + 2] = -(y + ny * width);
    out[o + 3] = x - nx * width;
    out[o + 5] = -(y - ny * width);
  }
  ribbon.attribute.needsUpdate = true;
}
