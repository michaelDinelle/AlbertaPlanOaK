import * as THREE from 'three/webgpu';
import { abs, clamp, dot, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, vec2 } from 'three/tsl';
import { hash2f } from '@kit/rng';
import { coverage, rgb, sdBox } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import type { GridSize } from '@kit/grid';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;

export const FACE = 0.36;
export const WALL = 0.2;

export interface Look {
  frost: number;
  turf: readonly [number, number];
  earth: readonly [number, number];
  moor: readonly [number, number];
  lip: number;
  rime: number;
  sand: readonly [number, number, number];
  water: readonly [number, number, number];
  foam: number;
  flag: readonly [number, number, number];
  stone: number;
  print: number;
  ambient: number;
  ambientLevel: number;
}

export const LOOKS: Record<'dusk' | 'frost', Look> = {
  dusk: {
    frost: 0,
    turf: [0x5f7d3c, 0x47662f],
    earth: [0x8a6644, 0x654a33],
    moor: [0x4d3f47, 0x3b3a2b],
    lip: 0xb9d38a,
    rime: 0xd6e6ee,
    sand: [0xd9a071, 0xa9694b, 0x5e3326],
    water: [0x103449, 0x1d5872, 0x5aa6b6],
    foam: 0xeaf6f2,
    flag: [0xa29a8c, 0x837b6f, 0x3f3a33],
    stone: 0x9a978d,
    print: 0x2c3a1e,
    ambient: 0x9ea6d6,
    ambientLevel: 0.9,
  },
  frost: {
    frost: 1,
    turf: [0x4e6a4c, 0x3c5640],
    earth: [0x56684b, 0x3f4f3a],
    moor: [0x4a4552, 0x363a3a],
    lip: 0xe6f3f8,
    rime: 0xb8c9d6,
    sand: [0xc0987f, 0x8f6655, 0x4f3431],
    water: [0x0b263a, 0x173f58, 0x4a8aa2],
    foam: 0xf0f8fb,
    flag: [0x9da3a6, 0x7c8286, 0x3a3e42],
    stone: 0x9fa4a6,
    print: 0x243522,
    ambient: 0x8ea2d8,
    ambientLevel: 0.78,
  },
};

const NEIGHBOURS = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] as const;

export interface CoastMaps {
  mask: THREE.DataTexture;
  wear: THREE.DataTexture;
  mottle: THREE.DataTexture;
  grain: THREE.DataTexture;
  sea: (x: number, y: number) => boolean;
}

export function coastMaps(grid: GridSize, hazards: ReadonlySet<number>, seed: number): CoastMaps {
  const width = grid.w + 2, height = grid.h + 2, data = new Uint8Array(width * height * 4);
  const sea = (x: number, y: number): boolean => y >= grid.h || (x >= 0 && x < grid.w && y >= 0 && hazards.has(y * grid.w + x));
  for (let j = 0; j < height; j++) for (let i = 0; i < width; i++) data[(j * width + i) * 4] = sea(i - 1, j - 1) ? 255 : 0;
  const mask = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
  mask.magFilter = mask.minFilter = THREE.NearestFilter;
  mask.needsUpdate = true;
  const wear = new THREE.DataTexture(new Uint8Array(grid.w * grid.h * 4), grid.w, grid.h, THREE.RGBAFormat);
  wear.magFilter = wear.minFilter = THREE.LinearFilter;
  wear.needsUpdate = true;
  return { mask, wear, mottle: fbm(seed ^ 0xc11f, { size: 256, frequency: 4, octaves: 5 }), grain: fbm(seed ^ 0x5eaf, { size: 256, frequency: 16, octaves: 3 }), sea };
}

export interface CoastUniforms {
  time: { value: number };
  start: { value: THREE.Vector2 };
}

export function coastMaterial(grid: GridSize, maps: CoastMaps, look: Look, lit: Lit): { material: THREE.MeshBasicNodeMaterial; uniforms: CoastUniforms } {
  const time = uniform(0), start = uniform(new THREE.Vector2(-9, -9));
  const w = grid.w, h = grid.h, size = vec2(w + 2, h + 2);
  const g = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2));
  const cell = floor(g), local = g.sub(cell);
  const seaAt = (c: N<'vec2'>) => texture(maps.mask, c.add(1.5).div(size)).r;
  const signed = (p: N<'vec2'>) => {
    const base = floor(p), self = seaAt(base);
    let d: N<'float'> = float(4);
    for (const [dx, dy] of NEIGHBOURS) {
      const c = base.add(vec2(dx, dy)), gap = length(max(abs(p.sub(c).sub(0.5)).sub(0.5), 0));
      d = min(d, select(abs(seaAt(c).sub(self)).greaterThan(0.5), gap, float(4)));
    }
    return select(self.greaterThan(0.5), min(d, 1), min(d, 1).negate());
  };
  const ragged = texture(maps.grain, g.mul(0.71).add(0.43)).r.sub(0.5);
  const s0 = signed(g).add(ragged.mul(0.03));
  const s1 = signed(g.sub(vec2(0, ragged.mul(0.04).add(FACE)))).add(ragged.mul(0.015));
  const land = coverage(s0), water = coverage(s1.negate());

  const mott = texture(maps.mottle, g.mul(0.17)).r, broad = texture(maps.mottle, g.mul(0.061).add(0.37)).r;
  const fine = texture(maps.grain, g.mul(0.93)).r, blades = texture(maps.grain, g.mul(4.7).add(0.21)).r;
  const inside = step(0, g.x).mul(step(g.x, w)).mul(step(0, g.y)).mul(step(g.y, h));
  const checker = mod(cell.x.add(cell.y), 2);

  let turf = mix(rgb(look.turf[1]), rgb(look.turf[0]), smoothstep(0.28, 0.72, mott)).mul(blades.mul(0.24).add(0.88));
  const wear = texture(maps.wear, g.div(vec2(w, h))).r;
  const trodden = smoothstep(0.24, 0.74, wear.add(fine.sub(0.5).mul(0.45)).add(blades.sub(0.5).mul(0.12)));
  turf = mix(turf, mix(rgb(look.earth[1]), rgb(look.earth[0]), fine).mul(blades.mul(0.18).add(0.9)), trodden);
  turf = turf.mul(checker.sub(0.5).mul(0.11).add(1));
  const edge = min(min(local.x, local.x.oneMinus()), min(local.y, local.y.oneMinus()));
  turf = turf.mul(coverage(edge.sub(0.007)).mul(-0.2).add(1));

  const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));
  const q = g.mul(3.1), qi = floor(q), qf = q.sub(qi);
  let near: N<'float'> = float(9), second: N<'float'> = float(9);
  for (const [dx, dy] of NEIGHBOURS) {
    const o = vec2(dx, dy), id = qi.add(o), jitter = vec2(hash(id), hash(id.add(17.3)));
    const r = length(o.add(jitter.mul(0.8).add(0.1)).sub(qf));
    second = min(max(near, r), second);
    near = min(near, r);
  }
  const joint = second.sub(near).div(3.1);
  const slab = mix(rgb(look.flag[1]), rgb(look.flag[0]), fine).mul(mott.mul(0.2).add(0.9));
  const onStart = step(abs(cell.x.sub(start.x)).add(abs(cell.y.sub(start.y))), 0.5);
  turf = mix(turf, mix(rgb(look.flag[2]), slab, smoothstep(0.012, 0.03, joint)), onStart);

  const heath = mix(rgb(look.moor[1]), rgb(look.moor[0]), smoothstep(0.3, 0.7, broad)).mul(smoothstep(0.35, 0.7, mott).mul(0.16).add(0.88)).mul(blades.mul(0.14).add(0.93));
  let ground = mix(heath, turf, inside);
  let glint: N<'float'> = float(0);
  if (look.frost > 0) {
    const at = g.div(0.09).add(7.7), id = floor(at), pick = hash(id), spot = vec2(hash(id.add(17.3)), hash(id.add(41.9))).mul(0.6).add(0.2);
    const flake = coverage(length(at.sub(id).sub(spot)).mul(0.09).sub(0.005)).mul(step(pick, 0.02));
    glint = flake.mul(sin(time.mul(2.3).add(pick.mul(2000))).max(0).pow(10)).mul(trodden.oneMinus()).mul(inside).mul(0.7);
    const rime = blades.sub(0.5).mul(0.2).add(mott.sub(0.5).mul(0.12)).add(0.5).mul(trodden.mul(inside).mul(-0.92).add(1)).mul(onStart.mul(-0.7).add(1)).mul(inside.mul(0.25).add(0.75));
    ground = mix(ground, rgb(look.rime), rime).mul(checker.sub(0.5).mul(inside).mul(0.09).add(1));
  }
  const rim = abs(sdBox(g.sub(vec2(w / 2, h / 2)), vec2(w / 2, h / 2)));
  const open = seaAt(vec2(clamp(floor(g.x), 0, w - 1), h - 1)).mul(step(h - 0.5, g.y));
  const wall = select(open.greaterThan(0.5), float(9), rim);
  ground = ground.mul(smoothstep(WALL / 2 - 0.01, WALL / 2 + 0.14, wall).mul(0.5).add(0.5));
  ground = mix(ground, rgb(look.stone).mul(0.3), coverage(wall.sub(WALL / 2 - 0.012)));
  const lip = s0.negate();
  ground = mix(ground, rgb(look.lip), smoothstep(0.012, 0.07, lip).oneMinus().mul(onStart.oneMinus().mul(0.3)));
  ground = mix(ground, ground.mul(0.4), smoothstep(0, 0.012, lip).oneMinus());

  const depth = clamp(s0.div(FACE), 0, 1);
  const column = smoothstep(0.3, 0.7, texture(maps.grain, vec2(g.x.mul(0.42), 0.37)).r).mul(0.6).add(texture(maps.grain, vec2(g.x.mul(1.1).add(0.2), depth.mul(0.05).add(0.8))).r.mul(0.4));
  const tier = (level: number, seed: number) => {
    const at = texture(maps.mottle, vec2(g.x.mul(0.3), seed)).r.sub(0.5).mul(0.24).add(level), rise = at.sub(depth);
    return [smoothstep(0.004, 0.022, abs(rise)).oneMinus(), smoothstep(0, 0.05, rise).mul(smoothstep(0.05, 0.13, rise).oneMinus())] as const;
  };
  const [line1, top1] = tier(0.36, 0.21), [line2, top2] = tier(0.68, 0.73);
  const seam = max(line1, line2), shelf = max(top1, top2);
  let rock = mix(rgb(look.sand[1]), rgb(look.sand[0]), column).mul(texture(maps.mottle, vec2(g.x.mul(0.6), depth.mul(0.3).add(0.4))).r.mul(0.3).add(0.85));
  rock = rock.mul(shelf.mul(0.28).add(1)).mul(seam.mul(-0.4).add(1));
  rock = rock.mul(mix(float(1.28), float(0.5), smoothstep(0, 1, depth))).mul(smoothstep(0, 0.08, depth).mul(0.65).add(0.35));
  rock = mix(rock, rgb(look.water[0]).mul(1.2), smoothstep(0.8, 1, depth).mul(0.6));
  if (look.frost > 0) rock = mix(rock, rgb(look.rime), shelf.mul(0.22).add(smoothstep(0.04, 0.14, depth).oneMinus().mul(0.4)));

  const flow = time.mul(0.6);
  const swell = texture(maps.mottle, g.mul(0.21).add(vec2(flow.mul(0.012), flow.mul(0.005)))).r.add(texture(maps.mottle, g.mul(0.37).add(vec2(flow.mul(-0.008), flow.mul(0.011))).add(0.5)).r).mul(0.5);
  const reach = max(s1, 0);
  let sea = mix(rgb(look.water[1]), rgb(look.water[0]), smoothstep(0.3, 0.75, swell));
  sea = mix(sea, rgb(look.water[2]), smoothstep(0.6, 0.8, swell).mul(0.35));
  const crestField = texture(maps.mottle, g.mul(vec2(0.11, 0.42)).add(vec2(flow.mul(0.004), flow.mul(-0.018)))).r;
  sea = mix(sea, rgb(look.water[2]), smoothstep(0, 0.012, abs(crestField.sub(0.56))).oneMinus().mul(smoothstep(0.15, 0.6, reach)).mul(0.16));
  sea = sea.mul(smoothstep(0, 0.4, reach).mul(0.38).add(0.62));
  const wobble = texture(maps.mottle, vec2(g.x.mul(0.19), 0.61)).r;
  const cycle = fract(time.mul(0.11).add(wobble.mul(0.8)));
  const wash = exp(reach.sub(cycle.mul(0.24).add(0.03)).div(0.014).pow(2).negate()).mul(cycle.oneMinus()).mul(smoothstep(0.3, 0.6, texture(maps.mottle, g.mul(0.9)).r));
  const hug = exp(reach.div(0.018).negate()).mul(swell.mul(0.25).add(0.75));
  const lace = smoothstep(0.5, 0.7, texture(maps.mottle, g.mul(0.8).add(vec2(flow.mul(0.01), 0))).r).mul(exp(reach.div(0.07).negate())).mul(0.5);
  sea = mix(sea, rgb(look.foam), clamp(hug.add(wash.mul(0.7)).add(lace), 0, 1));

  const color = mix(mix(rock, sea, water), ground, land);
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(color).add(rgb(0xe8f4ff).mul(glint.mul(land).mul(0.9)));
  return { material, uniforms: { time, start } };
}

export interface WallSegment { from: THREE.Vector2; to: THREE.Vector2 }

export function wallSegments(grid: GridSize, sea: (x: number, y: number) => boolean): WallSegment[] {
  const edges: WallSegment[] = [];
  const x0 = -grid.w / 2, z0 = -grid.h / 2;
  for (let y = 0; y < grid.h; y++) for (let x = 0; x < grid.w; x++) {
    if (sea(x, y)) continue;
    if (y === 0) edges.push({ from: new THREE.Vector2(x0 + x, z0), to: new THREE.Vector2(x0 + x + 1, z0) });
    if (y === grid.h - 1) edges.push({ from: new THREE.Vector2(x0 + x, z0 + grid.h), to: new THREE.Vector2(x0 + x + 1, z0 + grid.h) });
    if (x === 0) edges.push({ from: new THREE.Vector2(x0, z0 + y), to: new THREE.Vector2(x0, z0 + y + 1) });
    if (x === grid.w - 1) edges.push({ from: new THREE.Vector2(x0 + grid.w, z0 + y), to: new THREE.Vector2(x0 + grid.w, z0 + y + 1) });
  }
  const merged: WallSegment[] = [];
  for (const edge of edges) {
    const joined = merged.find((run) => run.to.equals(edge.from) && (run.to.x - run.from.x) * (edge.to.y - edge.from.y) === (run.to.y - run.from.y) * (edge.to.x - edge.from.x));
    if (joined) joined.to.copy(edge.to);
    else merged.push({ from: edge.from.clone(), to: edge.to.clone() });
  }
  const meets = (run: WallSegment, p: THREE.Vector2) => merged.some((other) => other !== run && (other.from.equals(p) || other.to.equals(p)));
  const joins = merged.map((run) => [meets(run, run.from), meets(run, run.to)] as const);
  merged.forEach((run, i) => {
    const along = run.to.clone().sub(run.from).normalize().multiplyScalar(WALL / 2);
    if (joins[i]![0]) run.from.sub(along);
    if (joins[i]![1]) run.to.add(along);
  });
  return merged;
}

export interface StoneLayout { matrices: THREE.Matrix4[]; tones: number[] }

export function layStones(segments: readonly WallSegment[], seed: number): StoneLayout {
  const matrices: THREE.Matrix4[] = [], tones: number[] = [];
  const quaternion = new THREE.Quaternion(), axis = new THREE.Vector3(0, 1, 0);
  let n = 0;
  for (const run of segments) {
    const span = run.from.distanceTo(run.to), dir = run.to.clone().sub(run.from).normalize(), heading = Math.atan2(-dir.y, dir.x);
    for (let at = 0.035; at < span - 0.02; n++) {
      const r = hash2f(n, 1, seed), s = hash2f(n, 2, seed), t = hash2f(n, 3, seed);
      const thick = 0.06 + 0.05 * r;
      const centre = run.from.clone().addScaledVector(dir, Math.min(at + thick / 2, span - thick / 2));
      quaternion.setFromAxisAngle(axis, heading + (s - 0.5) * 0.16);
      matrices.push(new THREE.Matrix4().compose(new THREE.Vector3(centre.x, 0, centre.y), quaternion, new THREE.Vector3(thick / 0.1, 1, (0.82 + 0.18 * t) * WALL / 0.2)));
      tones.push(0.72 + 0.4 * hash2f(n, 4, seed));
      at += thick + 0.004 + 0.008 * s;
    }
  }
  return { matrices, tones };
}

export function stoneMaterial(look: Look, grain: THREE.Texture, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const p = uv().sub(0.5).mul(2), d = sdBox(p, vec2(0.9), 0.3);
  const speck = texture(grain, vec2(positionWorld.x, positionWorld.z).mul(3.7)).r;
  const shade = p.y.mul(0.16).sub(p.x.mul(0.08)).add(1).mul(speck.mul(0.34).add(0.83)).mul(smoothstep(-0.34, 0, d).mul(-0.4).add(1));
  material.colorNode = lit(rgb(look.stone).mul(shade));
  material.opacityNode = coverage(d);
  return material;
}
