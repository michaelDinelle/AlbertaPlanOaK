import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, positionWorld, select, sign, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const ACCENT = TOKENS.accents.MicroRTS!;
export const FOE = 0xff7a5a;
export const MINT = 0x5fe3b0;
export const GOLD = TOKENS.celebrationGold;
export const HURT = 0xd83a3a;
export const TEAMS = [ACCENT, FOE] as const;
const FILLS = [0x3f73e0, 0xd24a30] as const;
export const RIM = 0.46;
export const TOKEN = 1;
export const TROOP_QUAD = 1.2;
const TROOP = 1.62;

const INK = 0x0a0d13;
const SLATE = [0x404b5d, 0x323b4a] as const;
const GROUT = 0x121821;
const STEEL = [0x3a4252, 0x252b36] as const;
const STONE = [0x8a92a3, 0x505868] as const;
const MIST = 0x141c2a;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

export const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

export const spin = (p: N<'vec2'>, angle: N<'float'>) => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
};

export const hexagon = (p: N<'vec2'>, radius: N<'float'> | number) => {
  const r = typeof radius === 'number' ? float(radius) : radius;
  const k = vec2(-0.866025404, 0.5), a = abs(p), folded = a.sub(k.mul(min(dot(k, a), 0).mul(2)));
  const limit = r.mul(0.577350269), edge = folded.sub(vec2(clamp(folded.x, limit.negate(), limit), r));
  return length(edge).mul(sign(edge.y));
};

export const octagon = (p: N<'vec2'>, r: number) => {
  const kx = -0.9238795325, ky = 0.3826834323, kz = 0.4142135623;
  let q = abs(p);
  q = q.sub(vec2(kx, ky).mul(min(dot(vec2(kx, ky), q), 0).mul(2)));
  q = q.sub(vec2(-kx, ky).mul(min(dot(vec2(-kx, ky), q), 0).mul(2)));
  const edge = q.sub(vec2(clamp(q.x, -kz * r, kz * r), r));
  return length(edge).mul(sign(edge.y));
};

const edgeOf = (p: N<'vec2'>, a: readonly [number, number], b: readonly [number, number]) => {
  const ex = b[0] - a[0], ey = b[1] - a[1], norm = Math.hypot(ex, ey);
  return dot(p.sub(vec2(a[0], a[1])), vec2(ey / norm, -ex / norm));
};

export const triangle = (p: N<'vec2'>, a: readonly [number, number], b: readonly [number, number], c: readonly [number, number]) => {
  const turning = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const [u, v] = turning > 0 ? [b, c] : [c, b];
  return max(max(edgeOf(p, a, u), edgeOf(p, u, v)), edgeOf(p, v, a));
};

export class Stack {
  color: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(color: N<'vec3'>, alpha: N<'float'>): this {
    this.color = color.mul(alpha).add(this.color.mul(alpha.oneMinus()));
    this.alpha = alpha.add(this.alpha.mul(alpha.oneMinus()));
    return this;
  }

  paint(material: THREE.MeshBasicNodeMaterial, fade: N<'float'>): void {
    material.colorNode = this.color.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha.mul(fade);
  }
}

export const teamOf = (team: N<'float'>) => select(team.lessThan(0.5), rgb(ACCENT), select(team.lessThan(1.5), rgb(FOE), rgb(MINT)));
export const fillOf = (team: N<'float'>) => select(team.lessThan(0.5), rgb(FILLS[0]), select(team.lessThan(1.5), rgb(FILLS[1]), rgb(MINT)));

const pips = (p: N<'vec2'>, hp: N<'float'>, most: N<'float'>, width: N<'float'>, y: number, tint: N<'vec3'>) => {
  const along = p.x.div(width).add(0.5), slot = floor(along.mul(most)), inside = fract(along.mul(most));
  const bar = sdBox(p.sub(vec2(0, y)), vec2(width.mul(0.5), 0.021), 0.008);
  const gap = min(inside, inside.oneMinus()).mul(width).div(most).sub(0.006);
  const lit = step(slot.add(0.5), hp);
  const color = mix(rgb(HURT).mul(0.55), mix(tint, vec3(1), 0.35), lit);
  return { shape: max(bar, gap.negate()), frame: bar.sub(0.012), color, on: step(1.5, most) };
};

export interface Board {
  material: THREE.MeshBasicNodeMaterial;
  clock: Level;
  time: Level;
  textures: THREE.Texture[];
}

export function boardMaterial(side: number, seed: number, lit: Lit): Board {
  const clock = uniform(0), time = uniform(0);
  const mottle = fbm(seed ^ 0x3c71, { size: 256, frequency: 4, octaves: 5 }), grain = fbm(seed ^ 0x51d2, { size: 256, frequency: 32, octaves: 3 });
  const q = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2)), c = q.sub(side / 2);
  const cell = floor(q), local = fract(q).sub(0.5);
  const mott = texture(mottle, q.mul(0.09)).r, fine = texture(grain, q.mul(0.41)).r, speck = texture(grain, q.mul(2.7).add(0.37)).r;
  const tone = fract(sin(dot(cell, vec2(12.9898, 78.233))).mul(43758.5453));

  const tile = sdBox(local, vec2(0.47), 0.05);
  let slate: N<'vec3'> = mix(rgb(SLATE[1]), rgb(SLATE[0]), clamp(mott.mul(0.8).add(tone.mul(0.35)).sub(0.1), 0, 1));
  slate = slate.mul(fine.sub(0.5).mul(0.16).add(1)).mul(smoothstep(0.74, 0.8, speck).mul(0.1).add(1));
  const rim = smoothstep(-0.05, 0, tile), facing = clamp(local.x.add(local.y).mul(-2.6), -1, 1);
  slate = slate.mul(rim.mul(facing).mul(0.28).add(1));
  const etch = coverage(abs(sdBox(local, vec2(0.33), 0.02)).sub(0.004)).mul(0.1);
  slate = mix(slate, rgb(0x55607a), etch);
  const bed = mix(rgb(GROUT), slate, coverage(tile));

  const reach = side / 2 + RIM, outer = sdBox(c, vec2(reach), 0.14), onFrame = coverage(outer);
  const horizontal = step(abs(c.x), abs(c.y)), along = mix(c.y, c.x, horizontal), across = mix(abs(c.x), abs(c.y), horizontal).sub(side / 2);
  const brushed = texture(grain, vec2(along.mul(0.08), across.mul(3.1))).r;
  let steel: N<'vec3'> = mix(rgb(STEEL[1]), rgb(STEEL[0]), smoothstep(0.25, 0.75, brushed));
  steel = steel.mul(sin(clamp(across.div(RIM), 0, 1).mul(Math.PI)).mul(0.3).add(0.78));
  const lip = coverage(abs(sdBox(c, vec2(side / 2 + 0.03), 0.02)).sub(0.02));
  steel = mix(steel, rgb(0x0d1118), lip.mul(0.9));
  const groove = side / 2 + RIM * 0.55, channel = abs(sdBox(c, vec2(groove), 0.06)).sub(0.045);
  const perimeter = 8 * groove;
  const u = select(horizontal.greaterThan(0.5), select(c.y.lessThan(0), c.x.add(groove), float(5 * groove).sub(c.x)), select(c.x.greaterThan(0), c.y.add(3 * groove), float(7 * groove).sub(c.y)));
  const filled = step(u, clock.mul(perimeter)), head = exp(u.sub(clock.mul(perimeter)).div(0.18).pow(2).negate()).mul(step(0.0005, clock));
  steel = mix(steel, rgb(0x080a0f), coverage(channel));
  const glow = rgb(ACCENT).mul(filled.mul(0.55).add(head.mul(1.6))).mul(coverage(channel));
  const corner = min(abs(c.x), abs(c.y)).sub(side / 2 + RIM * 0.55).mul(-1);
  const brackets = coverage(max(abs(sdBox(c, vec2(groove), 0.06)).sub(0.012), corner.sub(0.45))).mul(step(side / 2, min(abs(c.x), abs(c.y))));
  steel = mix(steel, rgb(0x8d97ab), brackets.mul(0.8));

  const inBoard = coverage(sdBox(c, vec2(side / 2), 0));
  const table = mix(rgb(0x07090d), rgb(0x0e131b), smoothstep(side * 0.9, side * 0.35, length(c)));
  const albedo = mix(mix(table, steel, onFrame), bed, inBoard);
  const shadow = smoothstep(-0.05, 0.6, sdBox(c.sub(vec2(0.14, 0.18)), vec2(reach), 0.14)).mul(0.55).add(0.45);
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo.mul(mix(shadow, float(1), onFrame))).add(glow);
  return { material, clock, time, textures: [mottle, grain] };
}

export interface Fog {
  material: THREE.MeshBasicNodeMaterial;
  now: THREE.DataTexture;
  was: THREE.DataTexture;
  blend: Level;
  time: Level;
  textures: THREE.Texture[];
}

const sightMap = (side: number) => {
  const map = new THREE.DataTexture(new Uint8Array(side * side * 4), side, side, THREE.RGBAFormat);
  map.minFilter = map.magFilter = THREE.LinearFilter;
  map.needsUpdate = true;
  return map;
};

export function fogMaterial(side: number, seed: number): Fog {
  const now = sightMap(side), was = sightMap(side), blend = uniform(1), time = uniform(0);
  const drift = fbm(seed ^ 0x6a09, { size: 256, frequency: 3, octaves: 5 });
  const q = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2));
  const at = q.div(side);
  const seen = mix(texture(was, at).r, texture(now, at).r, blend);
  const swirl = texture(drift, q.mul(0.07).add(vec2(time.mul(0.013), time.mul(-0.009)))).r;
  const curl = texture(drift, q.mul(0.19).add(vec2(time.mul(-0.021), time.mul(0.017))).add(swirl.mul(0.3))).r;
  const edge = seen.add(curl.sub(0.5).mul(0.26));
  const hidden = smoothstep(0.64, 0.36, edge);
  const inBoard = coverage(sdBox(q.sub(side / 2), vec2(side / 2), 0));
  const wisps = smoothstep(0.35, 0.75, curl.mul(0.55).add(swirl.mul(0.45)));
  const material = sheet();
  material.colorNode = mix(rgb(MIST), rgb(0x3a4a66), wisps);
  material.opacityNode = hidden.mul(inBoard).mul(wisps.mul(0.14).add(0.66));
  return { material, now, was, blend, time, textures: [drift] };
}

export function mineralMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aMineral', 'vec4');
  const share = state.x, fade = state.y, flash = state.z, seed = state.w;
  const p = uv().sub(0.5).mul(TOKEN);
  const places: readonly (readonly [number, number, number])[] = [
    [0, 0.03, 0.105], [0.17, 0.12, 0.078], [-0.17, 0.1, 0.082], [0.2, -0.08, 0.07], [-0.19, -0.1, 0.074], [0.02, -0.19, 0.068], [0.03, 0.24, 0.06],
  ];
  const shown = share.mul(places.length).ceil();
  let crystals: N<'float'> = float(9), shade: N<'float'> = float(0), core: N<'float'> = float(9);
  places.forEach(([x, y, r], k) => {
    const tilt = fract(sin(seed.add(k * 7.13)).mul(9311.7)).sub(0.5).mul(0.5);
    const local = spin(p.sub(vec2(x, y)), tilt), size = float(r).mul(share.mul(0.25).add(0.85));
    const d = select(float(k).lessThan(shown), hexagon(local, size), float(9));
    const facet = clamp(local.x.mul(-1.4).add(local.y.mul(1.9)).div(size).mul(0.5).add(0.5), 0, 1);
    shade = select(d.lessThan(crystals), facet, shade);
    core = min(core, select(float(k).lessThan(shown), hexagon(local, size.mul(0.45)), float(9)));
    crystals = min(crystals, d);
  });
  const rock = min(min(ellipse(p, [0.02, -0.02], [0.3, 0.24]), ellipse(p, [-0.12, 0.08], [0.18, 0.15])), ellipse(p, [0.14, -0.1], [0.17, 0.14]));
  const bed = lit(mix(rgb(0x1a2320), rgb(0x3a4a44), smoothstep(0.1, -0.2, p.x.sub(p.y)))).mul(0.9);
  const shadow = smoothstep(-0.05, 0.08, min(rock, crystals).add(0.02).max(sdCircle(p.sub(vec2(0.04, -0.05)), 0.34))).oneMinus().mul(0.35);
  const glass = mix(rgb(0x1f6f58), rgb(0xc8ffe8), shade).mul(share.mul(0.35).add(0.75));
  const glow = rgb(MINT).mul(coverage(core).mul(0.9).add(0.25)).mul(share.mul(0.6).add(0.4));
  new Stack()
    .over(vec3(0), shadow)
    .over(bed, coverage(rock).mul(0.95))
    .over(mix(rgb(INK), lit(glass).add(glow).add(vec3(flash.mul(1.4))), coverage(crystals.add(0.012))), coverage(crystals))
    .paint(material, fade);
  return material;
}

export function structureMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const body = attribute<'vec4'>('aHall', 'vec4'), life = attribute<'vec4'>('aHallLife', 'vec4');
  const kind = body.x, team = body.y, fade = body.z, busy = body.w;
  const hp = life.x, most = life.y, hit = life.z, ghost = life.w;
  const p = uv().sub(0.5).mul(TOKEN), tint = teamOf(team), fill = fillOf(team), isBase = kind.lessThan(1.5);
  const bevel = clamp(p.x.sub(p.y).mul(-1.6), -0.5, 0.5).mul(0.34).add(1);

  const wall = sdBox(p, vec2(0.38), 0.075);
  const corners = [[0.325, 0.325], [-0.325, 0.325], [0.325, -0.325], [-0.325, -0.325]] as const;
  const towers = corners.map(([x, y]) => sdCircle(p.sub(vec2(x, y)), 0.09)).reduce((a, b) => min(a, b));
  const caps = corners.map(([x, y]) => sdCircle(p.sub(vec2(x, y)), 0.05)).reduce((a, b) => min(a, b));
  const keep = sdBox(p.sub(vec2(0, 0.025)), vec2(0.235), 0.03);
  const hips = coverage(abs(abs(p.x).sub(abs(p.y.sub(0.025)))).sub(0.007)).mul(coverage(keep.add(0.03)));
  const coreMark = sdCircle(p.sub(vec2(0, 0.025)), 0.058);
  const gate = sdBox(p.sub(vec2(0, -0.34)), vec2(0.07, 0.045), 0.02);
  const baseShape = min(wall, towers);
  const pulse = select(busy.greaterThanEqual(0), sin(busy.mul(40)).mul(0.2).add(1.25), float(1));
  let baseColor: N<'vec3'> = lit(mix(rgb(STONE[1]), rgb(STONE[0]), 0.55).mul(bevel));
  baseColor = mix(baseColor, lit(rgb(STONE[1]).mul(0.75)), coverage(abs(wall.add(0.05)).sub(0.006)).mul(0.9));
  baseColor = mix(baseColor, lit(rgb(STONE[0]).mul(bevel)), coverage(towers));
  baseColor = mix(baseColor, lit(fill.mul(bevel)), coverage(caps));
  baseColor = mix(baseColor, lit(fill.mul(p.y.mul(0.9).add(1))).mul(hips.mul(-0.45).add(1)), coverage(keep));
  baseColor = mix(baseColor, mix(rgb(INK), tint.mul(pulse).add(vec3(0.15)), coverage(coreMark)), coverage(coreMark.sub(0.012)));
  baseColor = mix(baseColor, rgb(INK), coverage(gate));

  const hall = sdBox(p.sub(vec2(0, 0.03)), vec2(0.37, 0.29), 0.035);
  const roof = sdBox(p.sub(vec2(0, 0.05)), vec2(0.33, 0.23), 0.02);
  const courses = coverage(abs(fract(p.y.sub(0.05).div(0.058)).sub(0.5)).mul(0.058).sub(0.004)).mul(coverage(roof)).mul(0.4);
  const ridge = coverage(abs(p.y.sub(0.05)).sub(0.012)).mul(coverage(roof));
  const crest = max(sdBox(p.sub(vec2(0, 0.08)), vec2(0.085, 0.06), 0.02), sdCircle(p.sub(vec2(0, 0.1)), 0.13)).min(triangle(p, [-0.085, 0.04], [0.085, 0.04], [0, -0.06]));
  const swords = abs(crest).sub(0.013);
  const guards = sdBox(p.sub(vec2(0, 0.04)), vec2(0.012, 0.075), 0.006);
  const door = sdBox(p.sub(vec2(0, -0.235)), vec2(0.07, 0.045), 0.02);
  const flag = triangle(p, [-0.345, 0.36], [-0.345, 0.2], [-0.2, 0.28]), pole = sdSegment(p, vec2(-0.35, 0.38), vec2(-0.35, 0.08), 0.013);
  const barracksShape = min(hall, min(flag, pole));
  let barracksColor: N<'vec3'> = lit(mix(rgb(STONE[1]), rgb(STONE[0]), 0.45).mul(bevel));
  barracksColor = mix(barracksColor, lit(mix(fill.mul(0.72), fill, step(0.05, p.y)).mul(courses.oneMinus())), coverage(roof));
  barracksColor = mix(barracksColor, lit(fill.mul(0.45)), ridge);
  barracksColor = mix(barracksColor, mix(rgb(INK), lit(tint.mul(1.2).add(vec3(0.1))), coverage(min(swords, guards))), coverage(min(swords, guards).sub(0.01)));
  barracksColor = mix(barracksColor, rgb(INK), coverage(door));
  barracksColor = mix(barracksColor, lit(rgb(0x6b5a48)), coverage(pole));
  barracksColor = mix(barracksColor, lit(tint), coverage(flag));

  const shape = select(isBase, baseShape, barracksShape);
  const color = select(isBase, baseColor, barracksColor).add(vec3(hit.mul(0.9), hit.mul(0.25), hit.mul(0.2)));
  const bar = pips(p, hp, most, select(isBase, float(0.72), float(0.56)), -0.455, tint);
  const drop = smoothstep(-0.04, 0.08, sdBox(p.sub(vec2(0.035, -0.045)), select(isBase, vec2(0.37), vec2(0.36, 0.29)), 0.07)).oneMinus().mul(0.45);

  const dashes = step(0.5, fract(atan(p.y, p.x).mul(12 / Math.PI).add(ghost.mul(3))));
  const blueprint = coverage(abs(shape).sub(0.012)).mul(dashes.mul(0.6).add(0.4));
  const built = step(p.y.add(0.42), ghost.mul(0.84));
  const isGhost = ghost.greaterThanEqual(0);

  const solid = new Stack()
    .over(vec3(0), drop)
    .over(mix(rgb(INK), color, coverage(shape.add(0.014))), coverage(shape))
    .over(mix(rgb(INK), bar.color, coverage(bar.shape)), coverage(bar.frame).mul(bar.on));
  const solidColor = solid.color.div(max(solid.alpha, 1e-4));
  const frame = coverage(shape).mul(built.mul(0.55).add(0.12));
  material.colorNode = select(isGhost, mix(solidColor, tint.mul(1.3).add(vec3(0.1)), blueprint.max(built.oneMinus().mul(0.6))), solidColor);
  material.opacityNode = select(isGhost, max(frame, blueprint.mul(0.9)), solid.alpha).mul(fade);
  return material;
}

export function troopMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const body = attribute<'vec4'>('aTroop', 'vec4'), life = attribute<'vec4'>('aTroopLife', 'vec4'), motion = attribute<'vec4'>('aTroopFx', 'vec4');
  const kind = body.x, team = body.y, heading = body.z, fade = body.w;
  const hp = life.x, most = life.y, carry = life.z, ghost = life.w;
  const hit = motion.x, swing = motion.y;
  const p = uv().sub(0.5).mul(TROOP_QUAD), b = p.div(TROOP), tint = teamOf(team), fill = fillOf(team);
  const isWorker = kind.lessThan(3.5), isLight = kind.lessThan(4.5), isHeavy = kind.lessThan(5.5), isGhost = ghost.greaterThanEqual(0);
  const lunge = vec2(0, swing.mul(0.06));

  const shapeAt = (q: N<'vec2'>) => {
    const tools = select(isGhost, float(9), min(sdSegment(q, vec2(0.07, -0.03), vec2(0.19, 0.15), 0.016), sdSegment(q, vec2(0.11, 0.19), vec2(0.26, 0.1), 0.022)));
    const worker = min(ellipse(q, [0, -0.015], [0.155, 0.135]), tools);
    const dart = min(triangle(q, [0, 0.25], [-0.18, -0.16], [0, -0.06]), triangle(q, [0, 0.25], [0, -0.06], [0.18, -0.16]));
    const heavy = min(octagon(q, 0.205), sdBox(q.sub(vec2(0, 0.195)), vec2(0.165, 0.045), 0.02));
    const ranged = min(sdCircle(q, 0.13), abs(sdCircle(q.sub(vec2(0, -0.04)), 0.225)).sub(0.022).max(q.y.sub(0.02).negate()));
    return select(isWorker, worker, select(isLight, dart, select(isHeavy, heavy, ranged)));
  };

  const s = spin(b, heading).sub(lunge);
  const shape = shapeAt(s);
  const shine = (centre: readonly [number, number], radius: number) => smoothstep(radius, 0, length(s.sub(vec2(centre[0], centre[1])))).mul(0.35);

  const torso = ellipse(s, [0, -0.015], [0.155, 0.135]);
  const hat = sdCircle(s.sub(vec2(0, 0.015)), 0.078), brim = abs(sdCircle(s.sub(vec2(0, 0.015)), 0.078)).sub(0.01);
  const handle = select(isGhost, float(9), sdSegment(s, vec2(0.07, -0.03), vec2(0.19, 0.15), 0.016));
  const head = select(isGhost, float(9), sdSegment(s, vec2(0.11, 0.19), vec2(0.26, 0.1), 0.022));
  let worker: N<'vec3'> = lit(fill.add(tint.mul(shine([-0.05, 0.03], 0.16))));
  worker = mix(worker, lit(rgb(0x7a5634)), coverage(handle));
  worker = mix(worker, lit(rgb(0xd5dae4).mul(s.y.sub(0.1).mul(2).add(1))), coverage(head));
  worker = mix(worker, lit(rgb(0xf4cc4e).mul(s.x.mul(-1.6).add(s.y.mul(1.6)).add(1))), coverage(hat));
  worker = mix(worker, lit(rgb(0x8a6a1c)), coverage(brim).mul(0.8));
  worker = mix(rgb(INK), worker, coverage(min(torso, min(handle, head)).add(0.012)).max(coverage(hat)));

  const stripe = coverage(abs(s.x).sub(0.02)).mul(step(-0.07, s.y)).mul(step(s.y, 0.2));
  let dart: N<'vec3'> = lit(mix(fill.mul(0.72), fill.add(tint.mul(0.15)), step(0, s.x)));
  dart = mix(dart, lit(tint.mul(1.2).add(vec3(0.15))), stripe);

  const plate = octagon(s, 0.14);
  let heavy: N<'vec3'> = lit(fill.mul(0.55));
  heavy = mix(heavy, lit(fill.add(tint.mul(shine([-0.04, 0.05], 0.14)))), coverage(plate));
  heavy = mix(heavy, lit(rgb(0xb4bccb).mul(s.y.sub(0.19).mul(3).add(1))), coverage(sdBox(s.sub(vec2(0, 0.195)), vec2(0.165, 0.045), 0.02)));
  heavy = mix(heavy, lit(fill.mul(0.35)), coverage(abs(plate).sub(0.008)));

  const bow = abs(sdCircle(s.sub(vec2(0, -0.04)), 0.225)).sub(0.022).max(s.y.sub(0.02).negate());
  const string = sdSegment(s, vec2(-0.195, 0.02), vec2(0.195, 0.02), 0.006);
  const arrow = min(sdSegment(s, vec2(0, -0.09), vec2(0, 0.22), 0.011), triangle(s, [0, 0.28], [-0.045, 0.2], [0.045, 0.2]));
  let ranged: N<'vec3'> = lit(fill.add(tint.mul(shine([-0.04, 0.04], 0.13))));
  ranged = mix(ranged, lit(rgb(0x9a6e40)), coverage(bow));
  ranged = mix(ranged, lit(rgb(0xece6d6)), coverage(min(string, arrow)));

  const color = select(isWorker, worker, select(isLight, dart, select(isHeavy, heavy, ranged))).add(vec3(hit.mul(0.95), hit.mul(0.3), hit.mul(0.25)));

  const ring = abs(length(p).sub(0.445)).sub(0.022);
  const around = fract(atan(p.x, p.y).div(2 * Math.PI).add(1)), slot = floor(around.mul(most)), within = fract(around.mul(most));
  const gap = min(within, within.oneMinus()).mul(2 * Math.PI * 0.445).div(most).sub(0.012);
  const segments = max(ring, gap.negate());
  const ringColor = mix(rgb(HURT).mul(0.6), mix(tint, vec3(1), 0.3), step(slot.add(0.5), hp));
  const showRing = step(1.5, most);

  const shadow = smoothstep(-0.04, 0.07, shapeAt(spin(b.sub(vec2(0.03, -0.038)), heading).sub(lunge))).oneMinus().mul(0.42);
  const nugget = hexagon(s.sub(vec2(-0.06, -0.12)), 0.068);
  const glint = coverage(sdCircle(s.sub(vec2(-0.08, -0.1)), 0.016));

  const built = step(p.y.add(0.4), ghost.mul(0.8));
  const outline = coverage(abs(shape).sub(0.01));

  const solid = new Stack()
    .over(vec3(0), shadow)
    .over(mix(rgb(INK), color, coverage(shape.add(0.014))), coverage(shape))
    .over(mix(rgb(0x0d3a2c), mix(rgb(MINT).mul(1.15), vec3(1.5), glint), coverage(nugget.add(0.012))), coverage(nugget).mul(step(0.5, carry)))
    .over(mix(rgb(INK), ringColor, coverage(segments.add(0.008))), coverage(segments.sub(0.006)).mul(showRing));
  const solidColor = solid.color.div(max(solid.alpha, 1e-4));
  material.colorNode = select(isGhost, mix(solidColor, tint.mul(1.3).add(vec3(0.1)), outline.max(built.oneMinus().mul(0.6))), solidColor);
  material.opacityNode = select(isGhost, max(coverage(shape).mul(built.mul(0.55).add(0.12)), outline.mul(0.9)), solid.alpha).mul(fade);
  return material;
}

export const TRAY_WIDTH = 1.04;
const PER_ROW = 3;
const CHIP_PITCH = 0.3;

export interface Treasury {
  material: THREE.MeshBasicNodeMaterial;
  count: Level;
  team: Level;
  fresh: Level;
}

export function treasuryMaterial(height: number, fromTop: boolean, lit: Lit): Treasury {
  const material = sheet(), count = uniform(0), team = uniform(0), fresh = uniform(0);
  const p = uv().sub(0.5).mul(vec2(TRAY_WIDTH, height)), tint = teamOf(team);
  const tray = sdBox(p, vec2(TRAY_WIDTH / 2 - 0.04, height / 2 - 0.04), 0.1);
  const down = fromTop ? p.y.negate().add(height / 2 - 0.34) : p.y.add(height / 2 - 0.34);
  const inner = height - 0.52, rows = count.div(PER_ROW).ceil().max(1), pitch = min(float(CHIP_PITCH), float(inner).div(rows));
  const row = floor(down.div(pitch).add(0.5)), column = floor(p.x.div(CHIP_PITCH).add(PER_ROW / 2));
  const index = row.mul(PER_ROW).add(column);
  const centre = vec2(column.sub((PER_ROW - 1) / 2).mul(CHIP_PITCH), fromTop ? float(height / 2 - 0.34).sub(row.mul(pitch)) : row.mul(pitch).sub(height / 2 - 0.34));
  const local = p.sub(centre), size = pitch.mul(0.42).min(0.125);
  const present = step(0, row).mul(step(index, count.sub(0.5))).mul(step(0, column)).mul(step(column, PER_ROW - 1));
  const newest = step(count.sub(1.5), index).mul(fresh);
  const chip = hexagon(local, size.mul(newest.mul(0.35).add(1)));
  const facet = clamp(local.x.mul(-1.4).add(local.y.mul(1.9)).div(size).mul(0.5).add(0.5), 0, 1);
  const crest = sdBox(p.sub(vec2(0, fromTop ? height / 2 - 0.13 : -(height / 2 - 0.13))), vec2(0.36, 0.028), 0.02);
  const glass = mix(rgb(0x1f6f58), rgb(0xc8ffe8), facet);
  new Stack()
    .over(vec3(0), smoothstep(-0.05, 0.12, sdBox(p.sub(vec2(0.05, -0.07)), vec2(TRAY_WIDTH / 2 - 0.04, height / 2 - 0.04), 0.1)).oneMinus().mul(0.5))
    .over(lit(mix(rgb(0x0c1017), rgb(0x1b222d), smoothstep(0.02, -0.1, tray))), coverage(tray))
    .over(lit(tint).mul(0.9), coverage(abs(tray).sub(0.012)))
    .over(tint.mul(1.3), coverage(crest))
    .over(mix(rgb(INK), lit(glass).add(rgb(MINT).mul(newest.mul(1.2).add(0.25))), coverage(chip.add(0.01))), coverage(chip).mul(present))
    .paint(material, float(1));
  return { material, count, team, fresh };
}
