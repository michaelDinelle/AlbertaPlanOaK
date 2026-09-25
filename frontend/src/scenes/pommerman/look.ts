import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, normalize, positionWorld, select, sin, smoothstep, sqrt, step, texture, uniform, uvec2, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const RIM = 0.42;
const WAVE = 14;
export const BOMB_QUAD = 1.5;
export const BOMBER_QUAD = 1.7;
export const FX_QUAD = 2.4;
export const BADGE_W = 3.4;
export const BADGE_H = 1.5;

export const HUE = {
  void: 0x07090d, rim: 0x565d68, lip: 0xa2abb7,
  tileA: 0x505b64, tileB: 0x4a545d, grout: 0x22272d,
  basalt: 0x707985, basaltFront: 0x1c2025,
  iron: 0x3d3431, rivet: 0x9a8a7c, seam: 0xff5a24, hazard: 0x6b5a1e,
  plank: 0xa87440, plankDark: 0x62401f, plankFront: 0x553619, nail: 0xd6c6a6,
  flameCore: 0xfff1c0, flameMid: 0xffa23a, flameEdge: 0xff4418, smoke: 0x2c2a2a, ember: 0xff7a2a,
  bomb: 0x15171b, bombLit: 0x9aa6bb, fuse: 0xd7c59e, spark: 0xffd27a, danger: 0xff3a1a,
  helmet: 0xf1e9da, visor: 0x14171d, sole: 0x191b20,
  warn: 0xff3b2f,
} as const;

export const ITEM_HUES = [0, 0x3fb8f0, 0xff6040, 0x74d650] as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));
const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

interface Layer { color: N<'vec3'>; alpha: N<'float'> }
const clear = (): Layer => ({ color: vec3(0), alpha: float(0) });
const over = (below: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(below.color.mul(alpha.oneMinus())),
  alpha: alpha.add(below.alpha.mul(alpha.oneMinus())),
});
const settle = (material: THREE.MeshBasicNodeMaterial, layer: Layer, fade: N<'float'> | number = 1) => {
  material.colorNode = layer.color.div(max(layer.alpha, 1e-4));
  material.opacityNode = layer.alpha.mul(fade);
};

const rotate = (p: N<'vec2'>, angle: N<'float'>) => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};

const bombIcon = (q: N<'vec2'>, r: number) => min(sdCircle(q, r), sdBox(q.sub(vec2(r * 0.62, r * 0.62)), vec2(r * 0.26), r * 0.08));
const bootIcon = (q: N<'vec2'>, k: number) => min(sdBox(q.sub(vec2(-0.045 * k, 0.05 * k)), vec2(0.06 * k, 0.1 * k), 0.03 * k), sdBox(q.sub(vec2(0.025 * k, -0.075 * k)), vec2(0.125 * k, 0.05 * k), 0.045 * k));
const plusIcon = (q: N<'vec2'>, arm: number, half: number) => min(sdBox(q, vec2(arm, half), half * 0.5), sdBox(q, vec2(half, arm), half * 0.5));
const flameIcon = (q: N<'vec2'>, k: number) => {
  const body = sdCircle(q.add(vec2(0, 0.03 * k)), 0.085 * k);
  const tip = max(abs(q.x).mul(1.45).add(q.y.sub(0.17 * k).mul(0.62)), q.y.negate().sub(0.02 * k));
  return min(body, tip);
};

export const itemIcon = (q: N<'vec2'>, kind: N<'float'>, k: number) =>
  select(kind.lessThan(1.5), bombIcon(q.add(vec2(0.02 * k, 0.02 * k)), 0.1 * k), select(kind.lessThan(2.5), flameIcon(q.add(vec2(0, 0.02 * k)), 1.25 * k), bootIcon(q, k)));

export interface Board {
  cells: THREE.DataTexture;
  marks: THREE.DataTexture;
  u: Level;
  time: Level;
  start: Level;
  ground: THREE.MeshBasicNodeMaterial;
  walls(): N<'float'>;
  fire(): { color: N<'vec3'>; mask: N<'float'> };
  dispose(): void;
}

const dataTexture = (side: number) => {
  const map = new THREE.DataTexture(new Float32Array(side * side * 4), side, side, THREE.RGBAFormat, THREE.FloatType);
  map.minFilter = map.magFilter = THREE.NearestFilter;
  map.needsUpdate = true;
  return map;
};

export function makeBoard(side: number, flameLife: number, lit: Lit): Board {
  const cells = dataTexture(side), marks = dataTexture(side);
  const u = uniform(0), time = uniform(0), start = uniform(1e3);

  const space = () => {
    const g = positionWorld.xz.add(side / 2), cell = floor(g), f = fract(g);
    const cellMap = texture(cells), markMap = texture(marks);
    const clampAt = (dx: number, dy: number) => uvec2(clamp(cell.add(vec2(dx, dy)), vec2(0), vec2(side - 1)));
    const on = (dx: number, dy: number) => {
      const c = cell.add(vec2(dx, dy));
      return c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.x.lessThan(side)).and(c.y.lessThan(side));
    };
    const read = (dx: number, dy: number) => cellMap.load(clampAt(dx, dy));
    const solid = (dx: number, dy: number) => select(on(dx, dy), step(0.5, read(dx, dy).x), float(0));
    const heat = (dx: number, dy: number) => {
      const age = read(dx, dy).y;
      return select(on(dx, dy), clamp(age.sub(u).div(flameLife), 0, 1).mul(step(0.5, age)), float(0));
    };
    return { g, cell, p: vec2(f.x.sub(0.5), float(0.5).sub(f.y)), on, read, mark: markMap.load(clampAt(0, 0)), solid, heat };
  };

  const ground = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  {
    const { g, cell, p, on, read, mark, solid, heat } = space();
    const inside = on(0, 0), here = read(0, 0), terrain = select(inside, here.x, float(0));
    const is = (k: number) => step(abs(terrain.sub(k)), 0.5);
    const grain = noise(g.mul(2.3)), fine = noise(g.mul(8.7).add(3.1)), speck = hash(cell);
    const centre = length(cell.add(0.5).sub(side / 2));
    const opening = smoothstep(0, 0.08, start);
    const crest = start.mul(WAVE).sub(centre), wave = exp(crest.mul(crest).mul(-1.6)).mul(smoothstep(1.4, 0.2, start)).mul(opening);
    const swell = start.mul(WAVE).sub(length(g.sub(side / 2))), ring = exp(swell.mul(swell).mul(-5)).mul(smoothstep(1.2, 0.1, start)).mul(opening);

    const tile = sdBox(p, vec2(0.468), 0.06);
    let floorColor: N<'vec3'> = mix(rgb(HUE.tileA), rgb(HUE.tileB), mod(cell.x.add(cell.y), 2)).mul(speck.mul(0.08).add(0.95)).mul(grain.mul(0.07).add(1)).mul(fine.mul(0.05).add(1));
    const bevel = smoothstep(-0.07, 0, tile).mul(dot(normalize(p.add(1e-4)), vec2(-0.7071, 0.7071)));
    floorColor = floorColor.mul(bevel.mul(0.18).add(1));
    floorColor = mix(rgb(HUE.grout), floorColor, coverage(tile));
    const reach = 0.36;
    const shadow = max(max(solid(0, -1).mul(smoothstep(0.5 - reach, 0.5, p.y)), solid(-1, 0).mul(smoothstep(reach - 0.5, -0.5, p.x))),
      solid(-1, -1).mul(smoothstep(reach, 0, length(p.sub(vec2(-0.5, 0.5))))));
    floorColor = floorColor.mul(shadow.mul(-0.5).add(1));

    const hC = heat(0, 0);
    const burnt = select(mark.x.greaterThanEqual(0), smoothstep(3.6, 1.4, mark.x.add(u)), float(0));
    const soot = max(hC.mul(0.8), burnt.mul(0.6));
    floorColor = floorColor.mul(soot.mul(smoothstep(0.52, 0.1, length(p)).mul(0.5).add(0.25)).oneMinus());
    const chunk = smoothstep(0.18, 0.3, noise(g.mul(5.3).add(11.7))).mul(coverage(sdBox(p, vec2(0.36), 0.12))).mul(burnt);
    floorColor = mix(floorColor, rgb(0x19140f), chunk.mul(0.85));
    const ember = smoothstep(0.42, 0.5, noise(g.mul(11.3).add(4.2))).mul(chunk).mul(sin(time.mul(7).add(hash(cell).mul(9))).mul(0.3).add(0.7));

    const kind = select(inside.and(terrain.lessThan(0.5)).and(here.z.greaterThan(0.5)), here.z, float(0));
    const pop = select(mark.z.greaterThanEqual(0), smoothstep(0, 0.8, mark.z.add(u)), float(1));
    const bob = sin(time.mul(2.4).add(speck.mul(6.283))).mul(0.018);
    const size = max(pop.mul(float(1.18).sub(pop.mul(0.18))), 1e-3);
    const q = p.sub(vec2(0, bob)).div(size);
    const plate = sdBox(q, vec2(0.25), 0.09), shade = sdBox(p.sub(vec2(0.05, -0.06)).div(size), vec2(0.25), 0.12);
    const plateHue = select(kind.lessThan(1.5), rgb(ITEM_HUES[1]), select(kind.lessThan(2.5), rgb(ITEM_HUES[2]), rgb(ITEM_HUES[3])));
    const plateColor = plateHue.mul(dot(q, vec2(-0.9, 0.9)).mul(0.9).add(0.72)).mul(0.62);
    const icon = itemIcon(q, kind, 1);
    const iconColor = select(kind.lessThan(1.5), rgb(HUE.bomb), rgb(0xfff6e2));
    const plus = select(kind.lessThan(1.5), plusIcon(q.sub(vec2(0.115, -0.115)), 0.065, 0.02), float(1));
    const rim = coverage(abs(plate.add(0.018)).sub(0.012));
    const onItem = step(0.5, kind);
    floorColor = mix(floorColor, floorColor.mul(0.45), coverage(shade.sub(0.03)).mul(0.6).mul(onItem));
    let itemColor: N<'vec3'> = mix(plateColor, plateHue.mul(1.35).add(0.12), rim);
    itemColor = mix(itemColor, iconColor, coverage(icon));
    itemColor = mix(itemColor, rgb(0xffffff), coverage(plus));
    const itemCover = coverage(plate).mul(onItem);

    const warn = mark.w;
    const stripes = step(0.5, fract(g.x.add(g.y).mul(1.6).add(time.mul(0.6))));
    floorColor = mix(floorColor, mix(floorColor, rgb(HUE.warn).mul(0.5), 0.55), warn.mul(stripes.mul(0.5).add(0.5)).mul(0.6));

    const below = solid(0, 1), front = float(0.17).mul(below.oneMinus());
    const topCentre = vec2(0, front.mul(0.5)), topHalf = vec2(0.47, float(0.47).sub(front.mul(0.5)));
    const lift = select(is(1).add(is(2)).add(is(3)).greaterThan(0.5), wave.mul(0.1).add(1), float(1));
    const pb = p.div(lift);
    const top = sdBox(pb.sub(topCentre), topHalf, 0.07), foot = sdBox(pb, vec2(0.47), 0.07);
    const edge = clamp(top.negate().div(0.1), 0, 1).oneMinus();
    const tilt = dot(normalize(pb.sub(topCentre).add(1e-4)), vec2(-0.7071, 0.7071));
    const frontShade = smoothstep(float(-0.5), float(-0.5).add(front), pb.y).mul(0.5).add(0.5);

    let rock: N<'vec3'> = rgb(HUE.basalt).mul(grain.mul(0.12).add(1)).mul(edge.mul(tilt).mul(0.55).add(1)).mul(smoothstep(0.72, 0.9, fine.mul(0.5).add(0.5)).mul(-0.25).add(1));
    rock = mix(rgb(HUE.basaltFront).mul(frontShade), rock, coverage(top));

    const rivet = sdCircle(abs(pb.sub(topCentre)).sub(vec2(0.31, topHalf.y.sub(0.13))), 0.045);
    let iron: N<'vec3'> = rgb(HUE.iron).mul(grain.mul(0.15).add(1)).mul(edge.mul(tilt).mul(0.45).add(1));
    iron = mix(iron, rgb(HUE.hazard), step(0.5, fract(pb.x.add(pb.y).mul(3.2))).mul(0.3).mul(coverage(sdBox(pb.sub(topCentre), topHalf.sub(0.1), 0.02))));
    iron = mix(iron, rgb(HUE.rivet).mul(dot(normalize(abs(pb.sub(topCentre)).sub(vec2(0.31, topHalf.y.sub(0.13))).add(1e-4)), vec2(-0.7, 0.7)).mul(0.4).add(1)), coverage(rivet));
    iron = mix(rgb(0x1b1615).mul(frontShade), iron, coverage(top));

    const inner = sdBox(pb.sub(topCentre), topHalf.sub(0.085), 0.03);
    const qp = pb.sub(topCentre);
    const seam = coverage(abs(abs(qp.y).sub(topHalf.y.mul(0.34))).sub(0.009));
    const brace = sdSegment(qp, topHalf.mul(-0.78), topHalf.mul(0.78), 0.045);
    const woodGrain = noise(vec2(g.x.mul(3), g.y.mul(26))).mul(0.14).add(1);
    let crate: N<'vec3'> = rgb(HUE.plank).mul(woodGrain).mul(speck.mul(0.1).add(0.93));
    crate = mix(crate, rgb(HUE.plankDark), seam.mul(0.8));
    crate = mix(crate, rgb(HUE.plank).mul(1.12).mul(woodGrain), coverage(brace));
    crate = mix(crate, rgb(HUE.plankDark).mul(0.8), coverage(abs(brace).sub(0.012)).mul(0.7));
    crate = mix(rgb(HUE.plankDark).mul(woodGrain), crate, coverage(inner));
    crate = mix(crate, rgb(HUE.nail), coverage(sdCircle(abs(qp).sub(topHalf.sub(0.045)), 0.022)));
    crate = crate.mul(edge.mul(tilt).mul(0.4).add(1));
    const planks = step(0.5, fract(pb.x.mul(4).add(0.5))).mul(0.12).add(0.9);
    crate = mix(rgb(HUE.plankFront).mul(frontShade).mul(planks), crate, coverage(top));

    const block = select(is(1).greaterThan(0.5), rock, select(is(2).greaterThan(0.5), crate, iron));
    const blockCover = coverage(foot).mul(is(1).add(is(2)).add(is(3)));
    const blockShadow = coverage(sdBox(pb.sub(vec2(0.07, -0.08)), vec2(0.47), 0.1)).mul(is(1).add(is(2)).add(is(3)));

    let albedo: N<'vec3'> = mix(floorColor, floorColor.mul(0.35), blockShadow.mul(0.5));
    albedo = mix(albedo, itemColor, itemCover);
    albedo = mix(albedo, block, blockCover);

    const slam = select(mark.y.greaterThanEqual(0), smoothstep(2.2, 0, mark.y.add(u)), float(0));
    const seamGlow = exp(abs(top).mul(-60)).mul(is(3)).mul(slam.mul(slam));
    const warnGlow = rgb(HUE.warn).mul(warn.mul(warn)).mul(sin(time.mul(9)).mul(0.35).add(0.65)).mul(0.35).mul(select(is(0).greaterThan(0.5), stripes.mul(0.6).add(0.4), coverage(abs(top).sub(0.03)).mul(1.4)));

    const flick = fractal(g.mul(2.2).add(vec2(time.mul(0.7), time.mul(-2.2))), 3).add(noise(g.mul(6.5).add(vec2(time.mul(-1.1), time.mul(-3.7)))).mul(0.5));
    const lattice = g.sub(0.5), base = floor(lattice), fr = fract(lattice);
    const blaze = (ox: number, oy: number) => {
      const c = base.add(vec2(ox, oy)), valid = c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.x.lessThan(side)).and(c.y.lessThan(side));
      const age = texture(cells).load(uvec2(clamp(c, vec2(0), vec2(side - 1)))).y;
      return { lit: select(valid, step(0.5, age), float(0)), heat: select(valid, clamp(age.sub(u).div(flameLife), 0, 1).mul(step(0.5, age)), float(0)) };
    };
    const b00 = blaze(0, 0), b10 = blaze(1, 0), b01 = blaze(0, 1), b11 = blaze(1, 1);
    const bilinear = (w: N<'vec2'>, a: N<'float'>, b: N<'float'>, c: N<'float'>, d: N<'float'>) => mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
    const sharp = smoothstep(0.3, 0.7, fr);
    const spread = bilinear(sharp, b00.lit, b10.lit, b01.lit, b11.lit), core = bilinear(fr, b00.lit, b10.lit, b01.lit, b11.lit);
    const warmth = bilinear(sharp, b00.heat, b10.heat, b01.heat, b11.heat).div(max(spread, 1e-3));
    const floorLine = mix(float(0.94), float(0.5), warmth);
    const lick = floorLine.sub(spread.add(flick.mul(0.16)));
    const fireCover = coverage(lick).mul(step(1e-3, spread));
    const depth = clamp(core.add(flick.mul(0.12)).sub(0.45).div(0.55), 0, 1).mul(smoothstep(0, 0.25, lick.negate()));
    let fireColor: N<'vec3'> = mix(rgb(HUE.flameEdge), rgb(HUE.flameMid), smoothstep(0, 0.45, depth));
    fireColor = mix(fireColor, rgb(0xffbf36), smoothstep(0.4, 0.8, depth).mul(warmth));
    fireColor = mix(fireColor, rgb(HUE.flameCore), smoothstep(0.86, 1, depth).mul(warmth.mul(warmth)).mul(0.8));
    const smoke = smoothstep(0.36, 0.05, warmth).mul(fireCover);
    const fire = mix(fireColor.mul(depth.mul(0.7).add(0.75).mul(warmth.mul(0.45).add(0.6))), rgb(HUE.smoke).mul(0.8), smoke.mul(0.85));

    const world = positionWorld.xz, board = sdBox(world, vec2(side / 2), 0);
    const outer = board.sub(RIM);
    const band = coverage(board.negate()).mul(coverage(outer));
    let rimColor: N<'vec3'> = rgb(HUE.rim).mul(grain.mul(0.1).add(1)).mul(smoothstep(RIM, 0, board).mul(0.35).add(0.75));
    rimColor = mix(rimColor, rgb(HUE.lip), coverage(abs(board.sub(0.035)).sub(0.022)).mul(0.8));
    rimColor = mix(rimColor, rgb(HUE.lip).mul(0.8), coverage(abs(outer.add(0.03)).sub(0.012)).mul(0.6));
    const beyond = rgb(HUE.void).mul(grain.mul(0.15).add(1)).mul(exp(outer.mul(-0.35)).mul(0.6).add(0.4));
    const inset = max(smoothstep(0.34, 0, world.y.add(side / 2)), smoothstep(0.34, 0, world.x.add(side / 2))).mul(0.35);

    const onBoard = step(board, 0);
    const boardLit = lit(albedo.mul(inset.oneMinus()));
    const surface = mix(mix(lit(beyond), lit(rimColor), band), boardLit, onBoard);
    let color: N<'vec3'> = surface.add(rgb(HUE.ember).mul(ember).mul(2.2).mul(onBoard));
    color = color.add(rgb(HUE.seam).mul(seamGlow).mul(0.9)).add(warnGlow.mul(onBoard));
    color = color.add(plateHue.mul(0.35).mul(rim).mul(itemCover));
    color = color.add(rgb(0xcfe4ff).mul(ring).mul(0.32).mul(onBoard));
    color = mix(color, fire, fireCover.mul(onBoard));
    ground.colorNode = color;
  }

  return {
    cells, marks, u, time, start, ground,
    walls() {
      const { on, read, p } = space(), core = step(sdBox(p, vec2(0.36), 0.05), 0);
      const wall = select(on(0, 0), step(0.5, read(0, 0).x).mul(core), float(0));
      return wall;
    },
    fire() {
      const { heat } = space(), h = heat(0, 0);
      return { color: mix(rgb(HUE.flameEdge), rgb(HUE.flameMid), h).mul(h.mul(h).mul(1.7)), mask: step(0.02, h) };
    },
    dispose() {
      cells.dispose();
      marks.dispose();
      ground.dispose();
    },
  };
}

export class Batch {
  readonly mesh: THREE.InstancedMesh;
  private readonly channels = new Map<string, THREE.InstancedBufferAttribute>();
  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  used = 0;

  constructor(readonly capacity: number, width: number, depth: number, names: readonly string[], paint: (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => THREE.MeshBasicNodeMaterial, order: number) {
    const geometry = flat(width, depth);
    for (const name of names) {
      const attr = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attr);
      this.channels.set(name, attr);
    }
    const p = attribute<'vec2'>('uv', 'vec2').sub(0.5).mul(vec2(width, depth));
    this.mesh = instancedMesh(geometry, paint(p, (name) => attribute<'vec4'>(name, 'vec4')), capacity);
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  put(x: number, z: number, scale = 1): number {
    if (this.used >= this.capacity) throw new Error(`${this.mesh.name || 'batch'} holds ${this.capacity} sprites`);
    const i = this.used++;
    this.mesh.setMatrixAt(i, this.matrix.compose(this.position.set(x, 0, z), this.quaternion, this.scale.setScalar(Math.max(1e-4, scale))));
    return i;
  }

  set(name: string, i: number, x: number, y = 0, z = 0, w = 0): void {
    this.channels.get(name)!.setXYZW(i, x, y, z, w);
  }

  commit(): void {
    this.mesh.count = this.used;
    this.mesh.visible = this.used > 0;
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const attr of this.channels.values()) attr.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.dispose();
  }
}

export function bombShade(lit: Lit, time: Level) {
  return (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => {
    const material = sheet(), state = channel('aBomb'), tint = channel('aTint');
    const fuse = state.x, spawn = state.y;
    const hurry = fuse.oneMinus();
    const beat = sin(time.mul(hurry.mul(hurry).mul(26).add(5))).mul(0.5).add(0.5);
    const swell = beat.mul(hurry.mul(0.07).add(0.015)).add(1).mul(mix(float(0.3), float(1), smoothstep(0, 1, spawn)));
    const r = float(0.3).mul(swell);
    const shadow = coverage(length(p.sub(vec2(0.07, -0.1)).div(vec2(1, 0.72))).sub(r.mul(1.02))).mul(0.42);
    const d = length(p).sub(r);
    const n = clamp(p.div(r), -1, 1), z = sqrt(max(float(1).sub(dot(n, n)), 0));
    const diffuse = clamp(dot(vec3(n.x, n.y, z), normalize(vec3(-0.55, 0.62, 0.56))), 0, 1);
    const spec = exp(length(n.sub(vec2(-0.38, 0.42))).mul(-11)).mul(1.3);
    const sheen = smoothstep(0.6, 1, length(n)).mul(smoothstep(-0.2, 0.6, dot(normalize(n.add(1e-4)), vec2(0.7, -0.7)))).mul(0.3);
    let body: N<'vec3'> = lit(mix(rgb(HUE.bomb), rgb(0x5a6474), diffuse.mul(diffuse))).add(rgb(0xe4ecff).mul(spec)).add(rgb(0x8fa3c8).mul(sheen));
    const band = coverage(abs(length(p).sub(r.mul(0.8))).sub(r.mul(0.08)));
    body = mix(body, lit(tint.xyz.mul(0.8)).add(tint.xyz.mul(0.35)), band);
    body = body.add(rgb(HUE.danger).mul(smoothstep(0.55, 1, hurry).mul(beat).mul(z).mul(1.1)));
    const capAt = vec2(0.72, 0.72).mul(r);
    const cap = sdBox(rotate(p.sub(capAt), float(-Math.PI / 4)), vec2(0.06, 0.05), 0.018);
    const tipAt = capAt.add(vec2(0.04, 0.2).mul(fuse).add(vec2(0.03, 0.03)));
    const cord = sdSegment(p, capAt, tipAt, 0.021);
    const flicker = sin(time.mul(41)).mul(sin(time.mul(23.7))).mul(0.25).add(0.85);
    const spark = exp(length(p.sub(tipAt)).mul(-26)).mul(flicker);
    const rays = exp(abs(rotate(p.sub(tipAt), time.mul(7))).x.mul(-90)).add(exp(abs(rotate(p.sub(tipAt), time.mul(7))).y.mul(-90))).mul(exp(length(p.sub(tipAt)).mul(-11))).mul(0.5);
    const sweep = fract(atan(p.x, p.y).div(Math.PI * 2).add(1));
    const track = coverage(abs(length(p).sub(0.43)).sub(0.02)).mul(mix(float(1), float(0.3), step(fuse, sweep)));
    let layer = over(clear(), vec3(0), shadow);
    layer = over(layer, mix(vec3(0.02), tint.xyz.mul(1.5).add(0.05), step(sweep, fuse)), track.mul(0.8));
    layer = over(layer, rgb(0x040506), coverage(d.sub(0.016)));
    layer = over(layer, body, coverage(d));
    layer = over(layer, rgb(0x05070a), coverage(cord.sub(0.012)).mul(0.8));
    layer = over(layer, lit(rgb(HUE.fuse)), coverage(cord));
    layer = over(layer, rgb(0x05070a), coverage(cap.sub(0.012)).mul(0.8));
    layer = over(layer, lit(rgb(0x9aa2b0)), coverage(cap));
    layer = over(layer, rgb(0xf4f7ff).mul(1.2), coverage(length(p.sub(r.mul(vec2(-0.38, 0.4)))).sub(r.mul(0.16))).mul(0.85));
    const glow = spark.add(rays);
    layer = over(layer, rgb(HUE.spark).mul(4), clamp(glow, 0, 1));
    settle(material, layer, step(0.001, spawn));
    return material;
  };
}

export function bomberShade(lit: Lit, time: Level, accent: number) {
  return (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => {
    const material = sheet(), body = channel('aBody'), tint = channel('aTint'), mark = channel('aMark');
    const heading = body.x, stride = body.y, alpha = body.z, down = body.w;
    const hue = tint.xyz, agent = tint.w, spawn = mark.x, ghost = mark.y, flash = mark.z;
    const gray = vec3(dot(hue, vec3(0.3, 0.55, 0.15)));
    const color = mix(hue, gray.mul(0.8), down);
    const rise = vec2(0, ghost.mul(0.42));
    const q = rotate(p.sub(rise), heading.negate());
    const squash = mix(float(1), float(0.55), down);
    const qs = vec2(q.x, q.y.div(squash));

    const shadow = coverage(length(p.sub(vec2(0.05, -0.08)).div(vec2(1, 0.7))).sub(0.3)).mul(0.45).mul(ghost.oneMinus());
    const ring = coverage(abs(length(p).sub(0.43)).sub(0.022)).mul(agent).mul(sin(time.mul(4)).mul(0.25).add(0.75));
    const halo = exp(length(p).mul(-5.5)).mul(agent).mul(0.35);
    const footL = sdCircle(q.sub(vec2(-0.12, stride.mul(0.1).add(0.01))).div(vec2(1, 1.35)), 0.075);
    const footR = sdCircle(q.sub(vec2(0.12, stride.mul(-0.1).add(0.01))).div(vec2(1, 1.35)), 0.075);
    const feet = min(footL, footR);
    const torso = length(qs).sub(0.26);
    const pack = sdBox(q.sub(vec2(0, -0.2)), vec2(0.14, 0.08), 0.06);
    const hands = min(sdCircle(q.sub(vec2(-0.25, stride.mul(-0.05).add(0.07))), 0.068), sdCircle(q.sub(vec2(0.25, stride.mul(0.05).add(0.07))), 0.068));
    const head = length(q.sub(vec2(0, 0.02))).sub(0.2);
    const visor = sdBox(q.sub(vec2(0, 0.115)), vec2(0.15, 0.062), 0.06);
    const eyes = min(length(q.sub(vec2(-0.062, 0.12))).sub(0.03), length(q.sub(vec2(0.062, 0.12))).sub(0.03));
    const crest = sdBox(q.sub(vec2(0, -0.075)), vec2(0.032, 0.1), 0.03);
    const cross = min(sdSegment(q, vec2(-0.1, 0.085), vec2(-0.035, 0.15), 0.013), sdSegment(q, vec2(-0.1, 0.15), vec2(-0.035, 0.085), 0.013));
    const crossR = min(sdSegment(q, vec2(0.035, 0.085), vec2(0.1, 0.15), 0.013), sdSegment(q, vec2(0.035, 0.15), vec2(0.1, 0.085), 0.013));
    const dead = min(cross, crossR);
    const bobble = sin(time.mul(4.2)).mul(0.035);
    const tip = vec2(0, bobble.add(0.62));
    const chevron = min(sdSegment(p, tip.add(vec2(-0.11, 0.1)), tip, 0.03), sdSegment(p, tip.add(vec2(0.11, 0.1)), tip, 0.03));

    const n = clamp(qs.div(0.26), -1, 1), z = sqrt(max(float(1).sub(dot(n, n)), 0));
    const light = clamp(dot(vec3(rotate(n, heading).x, rotate(n, heading).y, z), normalize(vec3(-0.55, 0.62, 0.56))), 0, 1);
    const coat = lit(color.mul(light.mul(0.8).add(0.45))).add(color.mul(0.12));
    const hn = clamp(q.sub(vec2(0, 0.02)).div(0.2), -1, 1), hz = sqrt(max(float(1).sub(dot(hn, hn)), 0));
    const hlight = clamp(dot(vec3(rotate(hn, heading).x, rotate(hn, heading).y, hz), normalize(vec3(-0.55, 0.62, 0.56))), 0, 1);
    const helm = lit(mix(rgb(HUE.helmet), gray, down.mul(0.5)).mul(hlight.mul(0.6).add(0.55))).add(rgb(0xffffff).mul(exp(length(hn.sub(rotate(vec2(-0.35, 0.4), heading.negate()))).mul(-7)).mul(0.25)));
    const glow = mix(hue.mul(1.6).add(0.25), vec3(0.9), down);

    let layer = over(clear(), vec3(0), shadow);
    layer = over(layer, hue.mul(1.3), halo.mul(ghost.oneMinus()));
    layer = over(layer, rgb(accent).mul(2.2), ring.mul(ghost.oneMinus()));
    layer = over(layer, lit(rgb(HUE.sole)), coverage(feet).mul(down.oneMinus()));
    layer = over(layer, coat.mul(0.55), coverage(pack));
    layer = over(layer, rgb(0x05070a), coverage(hands.sub(0.014)).mul(0.6));
    layer = over(layer, helm, coverage(hands));
    layer = over(layer, coat, coverage(torso));
    layer = over(layer, rgb(0x05070a), coverage(abs(torso).sub(0.012)).mul(0.5));
    layer = over(layer, rgb(0x05070a), coverage(abs(head).sub(0.01)).mul(0.35));
    layer = over(layer, helm, coverage(head));
    layer = over(layer, lit(color.mul(0.95)).add(color.mul(0.15)), coverage(crest).mul(coverage(head)));
    layer = over(layer, lit(rgb(HUE.visor)).add(rgb(0x4a5a70).mul(exp(length(q.sub(vec2(-0.08, 0.15))).mul(-18)).mul(0.5))), coverage(visor).mul(coverage(head)));
    layer = over(layer, glow.mul(mix(float(2.4), float(0.6), down)), coverage(eyes).mul(down.oneMinus()));
    layer = over(layer, rgb(0x1a1c22), coverage(dead).mul(down));
    layer = over(layer, rgb(0xffffff).mul(1.6), flash.mul(0.55).mul(coverage(torso.sub(0.02))));
    layer = over(layer, rgb(0x05070a), coverage(chevron.sub(0.018)).mul(agent).mul(ghost.oneMinus()).mul(0.7));
    layer = over(layer, rgb(accent).mul(2), coverage(chevron).mul(agent).mul(ghost.oneMinus()));
    const spawnRing = coverage(abs(length(p).sub(spawn.mul(0.5).add(0.12))).sub(0.03)).mul(spawn.oneMinus()).mul(step(0.001, spawn));
    layer = over(layer, hue.mul(2.2), spawnRing);
    settle(material, layer, alpha.mul(mix(float(1), float(0.55), ghost)));
    return material;
  };
}

export const FX = { SPLINTER: 0, PICKUP: 1, STREAK: 2, PUFF: 3, DUST: 4, MEMORY: 5, SLAM: 6 } as const;

export function fxShade(time: Level) {
  return (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => {
    const material = sheet(), fx = channel('aFx'), tint = channel('aTint');
    const kind = fx.x, t = clamp(fx.y, 0, 1), seed = fx.z, arg = fx.w, hue = tint.xyz;
    const is = (k: number) => step(abs(kind.sub(k)), 0.5);

    let splinter: N<'float'> = float(9);
    for (let i = 0; i < 7; i++) {
      const a = seed.mul(13.7).add(i * 0.9 + (i % 2) * 0.4), speed = hash(vec2(seed, i)).mul(0.5).add(0.45);
      const at = vec2(cos(a), sin(a)).mul(speed.mul(float(1).sub(t.oneMinus().mul(t.oneMinus()))).mul(0.7).add(0.22));
      const chip = sdBox(rotate(p.sub(at), a.add(t.mul(9 + i))), vec2(0.075, 0.028), 0.01);
      splinter = min(splinter, chip);
    }
    const splinterA = coverage(splinter).mul(t.oneMinus()).mul(is(FX.SPLINTER));

    let burst: N<'float'> = float(0);
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2 + 0.3, spark = vec2(Math.cos(a), Math.sin(a)).mul(t.mul(0.5).add(0.18));
      burst = max(burst, coverage(length(p.sub(spark)).sub(t.oneMinus().mul(0.035).add(0.008))));
    }
    burst = burst.mul(t.oneMinus());
    const star = exp(abs(p.x).mul(-40).add(abs(p.y).mul(-4))).add(exp(abs(p.y).mul(-40).add(abs(p.x).mul(-4)))).mul(t.oneMinus()).mul(smoothstep(0, 0.15, t));
    const icon = itemIcon(p.sub(vec2(0, t.mul(0.55))).div(0.9), arg, 1);
    const pickupA = max(max(burst, clamp(star, 0, 1)), coverage(icon).mul(t.oneMinus())).mul(is(FX.PICKUP));

    const dir = vec2(cos(arg), sin(arg));
    const along = dot(p, dir), across = dot(p, vec2(dir.y.negate(), dir.x));
    const lines = coverage(abs(fract(across.mul(6.5).add(0.5)).sub(0.5)).div(6.5).sub(0.012)).mul(smoothstep(-0.05, -0.2, along)).mul(smoothstep(-0.75, -0.3, along)).mul(step(abs(across), 0.24));
    const streakA = lines.mul(0.7).mul(is(FX.STREAK));

    const puffN = noise(p.mul(5).add(seed.mul(7)));
    const puff = smoothstep(0.1, -0.12, length(p).sub(t.mul(0.4).add(0.14)).add(puffN.mul(0.1))).mul(t.oneMinus().mul(t.oneMinus())).mul(smoothstep(0, 0.12, t));
    const puffA = puff.mul(0.5).mul(is(FX.PUFF));

    const dust = coverage(abs(length(p.div(vec2(1, 0.8))).sub(t.mul(0.5).add(0.18))).sub(t.oneMinus().mul(0.05).add(0.01))).mul(t.oneMinus()).mul(0.55);
    const dustA = dust.mul(is(FX.DUST));

    const angle = atan(p.y, p.x), dash = step(0.5, fract(angle.div(Math.PI * 2).mul(14).add(time.mul(0.4))));
    const memory = coverage(abs(length(p).sub(0.3)).sub(0.02)).mul(dash).mul(0.85);
    const hookAt = p.sub(vec2(0, 0.06)), hook = max(abs(length(hookAt).sub(0.058)).sub(0.019), min(hookAt.x.negate(), hookAt.y.negate()));
    const mark = coverage(min(min(hook, sdSegment(p, vec2(0, 0.002), vec2(0, -0.03), 0.019)), sdCircle(p.sub(vec2(0, -0.085)), 0.024)));
    const memoryA = clamp(memory.add(mark.mul(0.7)), 0, 1).mul(t.oneMinus()).mul(is(FX.MEMORY));

    const slam = coverage(abs(sdBox(p, vec2(t.mul(0.3).add(0.4)), 0.05)).sub(t.oneMinus().mul(0.05).add(0.01))).mul(t.oneMinus());
    const slamA = slam.mul(0.45).mul(is(FX.SLAM));

    const color = mix(hue, hue.mul(2.4).add(0.3), max(max(pickupA, slamA), star.mul(is(FX.PICKUP))));
    const embers = mix(rgb(HUE.flameMid).mul(1.8), hue.mul(0.7), smoothstep(0.1, 0.55, t));
    const shaded = select(is(FX.SPLINTER).greaterThan(0.5), embers.mul(sin(p.x.mul(40)).mul(0.1).add(0.9)), select(is(FX.PUFF).add(is(FX.DUST)).greaterThan(0.5), hue.mul(puffN.mul(0.2).add(0.9)), color));
    material.colorNode = shaded;
    material.opacityNode = clamp(splinterA.add(pickupA).add(streakA).add(puffA).add(dustA).add(memoryA).add(slamA), 0, 1);
    return material;
  };
}

export function badgeShade(lit: Lit, time: Level, accent: number, maxPips: number) {
  return (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => {
    const material = sheet(), seat = channel('aSeat'), tint = channel('aTint'), meta = channel('aMeta');
    const ammo = seat.x, blast = seat.y, kick = seat.z, alive = seat.w, hue = tint.xyz, agent = tint.w, flip = meta.x, role = meta.y;
    const q = vec2(p.x.mul(flip), p.y);
    const panel = sdBox(p, vec2(BADGE_W / 2 - 0.04, BADGE_H / 2 - 0.04), 0.16);
    const faded = mix(float(0.35), float(1), alive);
    const gray = vec3(dot(hue, vec3(0.3, 0.55, 0.15)));
    const face = mix(gray.mul(0.7), hue, alive);
    let layer = over(clear(), rgb(0x0b0e13).mul(0.9), coverage(panel).mul(0.82));
    layer = over(layer, face.mul(0.5), coverage(abs(panel).sub(0.012)).mul(0.9));
    layer = over(layer, rgb(accent).mul(1.8), coverage(abs(panel.sub(0.04)).sub(0.018)).mul(agent).mul(sin(time.mul(4)).mul(0.25).add(0.75)));

    const at = q.sub(vec2(-BADGE_W / 2 + 0.62, 0));
    const disc = length(at).sub(0.42);
    const n = clamp(at.div(0.42), -1, 1), z = sqrt(max(float(1).sub(dot(n, n)), 0));
    const light = clamp(dot(vec3(n.x, n.y, z), normalize(vec3(-0.55, 0.62, 0.56))), 0, 1);
    layer = over(layer, lit(face.mul(light.mul(0.8).add(0.45))), coverage(disc));
    const head = length(at.sub(vec2(0, 0.06))).sub(0.27);
    layer = over(layer, lit(mix(rgb(HUE.helmet), gray, alive.oneMinus().mul(0.5)).mul(light.mul(0.5).add(0.62))), coverage(head));
    const visor = sdBox(at.sub(vec2(0, 0.02)), vec2(0.2, 0.065), 0.06);
    layer = over(layer, rgb(HUE.visor), coverage(visor));
    const eyes = min(length(at.sub(vec2(-0.09, 0.02))).sub(0.035), length(at.sub(vec2(0.09, 0.02))).sub(0.035));
    layer = over(layer, hue.mul(2), coverage(eyes).mul(alive));
    const crossed = min(sdSegment(at, vec2(-0.3, -0.3), vec2(0.3, 0.3), 0.04), sdSegment(at, vec2(-0.3, 0.3), vec2(0.3, -0.3), 0.04));
    layer = over(layer, rgb(0xd04a3a).mul(1.4), coverage(crossed).mul(alive.oneMinus()));
    const tag = sdCircle(at.sub(vec2(0.3, -0.3)), 0.11);
    layer = over(layer, select(role.greaterThan(1.5), rgb(accent), select(role.greaterThan(0.5), mix(hue, rgb(0xffffff), 0.3), rgb(0x2a3140))), coverage(tag).mul(step(0.5, role)));

    const left = -BADGE_W / 2 + 1.16, pitch = (BADGE_W - 1.34) / maxPips;
    const slot = (x: N<'float'>) => floor(x.sub(left).div(pitch));
    const local = (x: N<'float'>) => fract(x.sub(left).div(pitch)).sub(0.5).mul(pitch);
    const row = (y: number) => vec2(local(q.x), q.y.sub(y));
    const k = slot(q.x);
    const within = step(0, k).mul(step(k, maxPips - 1));
    const has = (count: N<'float'>) => step(k.add(0.5), count).mul(within).mul(faded);
    const shell = row(0.34), ammoPip = sdCircle(shell.add(vec2(0.012, 0.012)), 0.066);
    const fuse = sdSegment(shell, vec2(0.03, 0.03), vec2(0.058, 0.064), 0.011), spark = length(shell.sub(vec2(0.064, 0.072))).sub(0.019);
    const shine = length(shell.sub(vec2(-0.03, 0.02))).sub(0.018);
    layer = over(layer, rgb(0x9aa6b8), coverage(ammoPip.sub(0.012)).mul(has(ammo)));
    layer = over(layer, rgb(0x2a2f38), coverage(ammoPip).mul(has(ammo)));
    layer = over(layer, rgb(0xe8eef8), coverage(shine).mul(has(ammo)));
    layer = over(layer, rgb(HUE.fuse), coverage(fuse).mul(has(ammo)));
    layer = over(layer, rgb(HUE.spark).mul(2.2), coverage(spark).mul(has(ammo)));
    layer = over(layer, rgb(HUE.flameMid).mul(1.7), coverage(flameIcon(row(-0.02), 0.78)).mul(has(blast)));
    layer = over(layer, rgb(HUE.flameCore).mul(1.8), coverage(flameIcon(row(-0.02).sub(vec2(0, -0.012)), 0.4)).mul(has(blast)));
    const boot = bootIcon(q.sub(vec2(left + 0.1, -0.4)), 0.95);
    layer = over(layer, mix(rgb(0x2b313b), rgb(ITEM_HUES[3]).mul(1.5), kick).mul(faded), coverage(boot));
    settle(material, layer);
    return material;
  };
}

export interface Fog { material: THREE.MeshBasicNodeMaterial; rect: THREE.UniformNode<'vec4', THREE.Vector4>; strength: Level; veil: Level }

export function fogShade(accent: number, time: Level): Fog {
  const material = sheet(), rect = uniform(new THREE.Vector4()), strength = uniform(0), veil = uniform(0);
  const q = positionWorld.xz, window = sdBox(q.sub(rect.xy.add(rect.zw).mul(0.5)), rect.zw.sub(rect.xy).mul(0.5), 0.1);
  const dark = smoothstep(-0.04, 0.2, window).mul(strength);
  const edge = coverage(abs(window).sub(0.016)).mul(strength).mul(sin(time.mul(2.2)).mul(0.15).add(0.55));
  const alpha = max(max(dark, veil), edge);
  material.colorNode = mix(rgb(0x04060b), rgb(accent).mul(1.4), edge.div(max(alpha, 1e-4)));
  material.opacityNode = alpha;
  return { material, rect, strength, veil };
}
