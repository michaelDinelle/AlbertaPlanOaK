import * as THREE from 'three/webgpu';
import { abs, dot, float, floor, fract, length, max, min, mix, positionWorld, select, sin, smoothstep, texture, uniform, vec2, vec3 } from 'three/tsl';
import type { GridSize } from '@kit/grid';
import { coverage, fractal, noise, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import type { GridCell } from '@sim/types';
import type { EnvDescriptor } from '../../data/envs';

type N<T extends string> = THREE.Node<T>;
type Mask = { w: number; h: number; walls: number[] };
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const PAD = 2;
export const DECK = 0.33;
export const PIER = 0.27;
export const SHADOW = new THREE.Vector2(0.05, 0.075);

const PARAPET = 0.07;
const BAND = 0.022;
const KERB = 0.07;
const BRICK = 1 / 8;
const JOINT = 0.09;
const LANE = 0.1;
const FAR = 0.9;
const SETT = 1 / 12;
const REFLECT = 0.16;
const PAINT = 0.013;
const NEIGHBOURS = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] as const;

export const PALETTE = {
  deep: 0x0d2633,
  mid: 0x1f4553,
  sheen: 0xa9c6cc,
  brick: [0x74463a, 0x8a5643, 0x5c3a30] as const,
  mortar: 0x3a2f2b,
  granite: [0x7d7a74, 0x98948c] as const,
  kerb: 0xaaa69d,
  arris: 0x302b28,
  parapet: [0xb9ae98, 0xd6ccb6] as const,
  sett: [0x6a6864, 0x8d8a84] as const,
  pier: 0x8f8b82,
  ring: 0x3b3e42,
  polish: 0xc9a488,
  paint: 0xe9e6dc,
};

export interface Layout {
  grid: GridSize;
  phases: Uint8Array[];
  goal: GridCell;
  land: Uint8Array;
  bridges: GridCell[];
  gates: GridCell[];
}

export function layoutOf(env: EnvDescriptor): Layout {
  const phases = env.parameters['phase'] as Mask[] | undefined, first = phases?.[0], goal = env.goal;
  if (!phases || !first || !goal || phases.some((p) => p.w !== first.w || p.h !== first.h || p.walls.length !== p.w * p.h)) throw new Error(`${env.envId} lacks its wall phases or goal`);
  const { w, h } = first, walls = phases.map((p) => Uint8Array.from(p.walls, (v) => (v ? 1 : 0)));
  const wet = Array.from({ length: h }, (_, y) => walls.some((m) => m.subarray(y * w, y * w + w).some(Boolean)));
  if (wet.some((row, y) => row && (y === 0 || y === h - 1 || wet[y - 1] || wet[y + 1]))) throw new Error(`${env.envId} has a barrier the canal cannot carry`);
  const land = new Uint8Array(w * h), bridges: GridCell[] = [], gates: GridCell[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, count = walls.reduce((n, m) => n + m[i]!, 0);
    if (!wet[y]) land[i] = 1;
    else if (count === 0) bridges.push({ x, y });
    else if (count < walls.length) gates.push({ x, y });
  }
  if (!land[goal[1] * w + goal[0]]) throw new Error(`${env.envId} puts its goal off the quays`);
  return { grid: { w, h }, phases: walls, goal: { x: goal[0], y: goal[1] }, land, bridges, gates };
}

export function paddedTexture(w: number, h: number): THREE.DataTexture {
  const texture = new THREE.DataTexture(new Uint8Array((w + 2 * PAD) * (h + 2 * PAD) * 4), w + 2 * PAD, h + 2 * PAD, THREE.RGBAFormat);
  texture.magFilter = texture.minFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

export const texelOf = (grid: GridSize, x: number, y: number) => ((y + PAD) * (grid.w + 2 * PAD) + x + PAD) * 4;

const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));

export interface Ground {
  material: THREE.MeshBasicNodeMaterial;
  map: THREE.DataTexture;
  wear: THREE.DataTexture;
  time: Level;
  glint: Level;
  start: THREE.UniformNode<'vec2', THREE.Vector2>;
}

export function groundMaterial(layout: Layout, lit: Lit, ambient: THREE.Color): Ground {
  const { grid } = layout, { w, h } = grid, size = vec2(w + 2 * PAD, h + 2 * PAD);
  const map = paddedTexture(w, h), wear = paddedTexture(w, h), data = map.image.data as Uint8Array;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[texelOf(grid, x, y)] = layout.land[y * w + x] ? 255 : 0;
  for (const cell of layout.bridges) data[texelOf(grid, cell.x, cell.y) + 1] = 255;
  for (const cell of layout.gates) data[texelOf(grid, cell.x, cell.y) + 2] = 255;
  const time = uniform(0), glint = uniform(1), start = uniform(new THREE.Vector2(-9, -9));

  const g = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2));
  const cell = floor(g), local = g.sub(cell);
  const read = (source: THREE.Texture, c: N<'vec2'>) => texture(source, c.add(PAD + 0.5).div(size));
  const self = read(map, cell), onLand = self.r.greaterThan(0.5);
  const cast = g.sub(vec2(SHADOW.x, SHADOW.y));

  const far = float(FAR);
  let toLand: N<'float'> = far, toWater: N<'float'> = far, deck: N<'float'> = far, parapet: N<'float'> = far;
  let pier: N<'float'> = far, bridgeShade: N<'float'> = far, pierShade: N<'float'> = far, pillar: N<'float'> = far;
  for (const [dx, dy] of NEIGHBOURS) {
    const n = cell.add(vec2(dx, dy)), m = read(map, n), q = g.sub(n).sub(0.5), s = cast.sub(n).sub(0.5);
    const gap = length(max(abs(q).sub(0.5), 0)), isLand = m.r.greaterThan(0.5), isBridge = m.g.greaterThan(0.5), isGate = m.b.greaterThan(0.5);
    toLand = min(toLand, select(isLand, gap, far));
    toWater = min(toWater, select(isLand, far, gap));
    deck = min(deck, select(isBridge, sdBox(q, vec2(DECK, 0.5)), far));
    parapet = min(parapet, select(isBridge, max(abs(abs(q.x).sub(DECK + PARAPET / 2)).sub(PARAPET / 2), abs(q.y).sub(0.58)), far));
    pillar = min(pillar, select(isBridge, sdBox(vec2(abs(q.x).sub(DECK + PARAPET / 2), abs(q.y).sub(0.53)), vec2(0.06, 0.055), 0.012), far));
    bridgeShade = min(bridgeShade, select(isBridge, sdBox(s, vec2(DECK + PARAPET, 0.5), 0.02), far));
    pier = min(pier, select(isGate, length(q).sub(PIER), far));
    pierShade = min(pierShade, select(isGate, length(s).sub(PIER), far));
  }
  const shore = select(onLand, toWater.negate(), toLand);

  const flow = time.mul(0.05);
  const broad = noise(g.mul(0.16).add(vec2(flow.mul(0.4), flow.mul(-0.15)))).mul(0.5).add(0.5);
  const chop = noise(vec2(g.x.mul(1.1).sub(flow.mul(2.4)), g.y.mul(4.6))).mul(0.6).add(noise(vec2(g.x.mul(2.4).add(flow.mul(1.5)), g.y.mul(7.5).add(3.1))).mul(0.4));
  const patches = smoothstep(-0.2, 0.6, noise(g.mul(0.45).add(vec2(flow.mul(0.8), 1.7))));
  let water: N<'vec3'> = mix(rgb(PALETTE.deep), rgb(PALETTE.mid), smoothstep(0.15, 0.85, broad));
  water = mix(water, rgb(PALETTE.sheen), smoothstep(0.2, 0.7, chop).mul(patches).mul(0.2));
  water = water.mul(smoothstep(0, 0.24, max(shore, 0)).mul(0.5).add(0.5));
  water = water.mul(smoothstep(-0.03, 0.07, bridgeShade).mul(0.5).add(0.5)).mul(smoothstep(-0.03, 0.07, pierShade).mul(0.45).add(0.55));

  const hb = local.div(BRICK), row = floor(hb.y), fy = hb.y.sub(row), t = hb.x.sub(row), band = floor(t.div(4)), m = t.sub(band.mul(4));
  const flatBrick = m.lessThan(2), upper = m.lessThan(3);
  const lx = select(flatBrick, m, select(upper, m.sub(2), m.sub(3))), ly = select(flatBrick, fy, select(upper, fy.add(1), fy));
  const sx = select(flatBrick, float(2), float(1)), sy = select(flatBrick, float(1), float(2));
  const edge = min(min(lx, sx.sub(lx)), min(ly, sy.sub(ly)));
  const brickId = vec2(select(flatBrick, row, select(upper, row.sub(2), row.sub(1))).add(select(flatBrick, float(0.5), float(0))), select(flatBrick, band, band.add(1)));
  const tone = hash(brickId.add(cell.mul(vec2(17.3, 31.9)))), dark = hash(brickId.add(cell.mul(vec2(5.1, 3.7))).add(9.1));
  const grain = noise(g.mul(13)).mul(0.5).add(0.5);
  const face = mix(mix(rgb(PALETTE.brick[0]), rgb(PALETTE.brick[1]), tone), rgb(PALETTE.brick[2]), smoothstep(0.92, 0.96, dark)).mul(grain.mul(0.14).add(0.92)).mul(smoothstep(JOINT, JOINT + 0.25, edge).mul(0.16).add(0.84));
  const bricks = mix(face.mul(0.55).add(rgb(PALETTE.mortar).mul(0.3)), face, coverage(float(JOINT).sub(edge).mul(BRICK)));
  const toEdge = min(min(local.x, local.x.oneMinus()), min(local.y, local.y.oneMinus()));
  const granite = mix(rgb(PALETTE.granite[0]), rgb(PALETTE.granite[1]), noise(g.mul(17)).mul(0.5).add(0.5));
  let quay: N<'vec3'> = mix(bricks, granite, coverage(toEdge.sub(BAND)));
  quay = mix(quay, rgb(PALETTE.kerb).mul(noise(g.mul(23)).mul(0.08).add(0.96)), coverage(toWater.sub(KERB)));
  quay = mix(quay, rgb(PALETTE.arris), coverage(toWater.sub(0.007)).mul(0.75));

  const crest = float(1).sub(abs(local.y.sub(0.5)).mul(0.36));
  const sett = vec2(local.x.div(SETT).add(floor(local.y.div(SETT)).mul(0.5)), local.y.div(SETT)), settId = floor(sett), settAt = fract(sett).sub(0.5);
  const settEdge = float(0.5).sub(max(abs(settAt.x), abs(settAt.y)));
  const settFace = mix(rgb(PALETTE.sett[0]), rgb(PALETTE.sett[1]), hash(settId.add(cell.mul(7.7)))).mul(smoothstep(0.06, 0.3, settEdge).mul(0.2).add(0.8)).mul(grain.mul(0.12).add(0.94));
  const setts = mix(rgb(PALETTE.mortar), settFace, coverage(float(0.07).sub(settEdge)));
  const bridgeDeck = mix(setts, granite, coverage(abs(local.y.sub(0.5)).negate().add(0.5).sub(BAND))).mul(crest);
  const courses = float(0.5).sub(abs(fract(g.y.mul(5)).sub(0.5))).div(5);
  const stone = mix(rgb(PALETTE.parapet[0]), rgb(PALETTE.parapet[1]), smoothstep(-PARAPET / 2, 0, parapet).oneMinus().mul(0.7)).mul(coverage(courses.sub(0.006)).mul(-0.35).add(1));
  const cap = mix(rgb(PALETTE.parapet[1]), rgb(PALETTE.parapet[0]).mul(0.8), smoothstep(-0.05, 0, pillar)).mul(coverage(abs(pillar.add(0.02)).sub(0.004)).mul(-0.25).add(1));

  const ring = coverage(abs(pier.add(0.05)).sub(0.012));
  const pierTop = mix(rgb(PALETTE.pier).mul(noise(g.mul(19)).mul(0.1).add(0.95)), rgb(PALETTE.ring), ring);

  const here = read(wear, cell), west = read(wear, cell.sub(vec2(1, 0))).r, north = read(wear, cell.sub(vec2(0, 1))).g;
  const lane = (end: readonly [number, number], level: N<'float'>) => smoothstep(LANE * 0.25, LANE, sdSegment(local, vec2(0.5, 0.5), vec2(...end), 0)).oneMinus().mul(level);
  const worn = max(max(lane([1, 0.5], here.r), lane([0, 0.5], west)), max(lane([0.5, 1], here.g), lane([0.5, 0], north)));

  const icon = g.sub(start).sub(0.5);
  const spoke = (a: readonly [number, number], b: readonly [number, number]) => sdSegment(icon, vec2(...a), vec2(...b), PAINT);
  const wheel = (x: number) => abs(length(icon.sub(vec2(x, 0.05))).sub(0.11)).sub(PAINT);
  const frameLines = min(min(min(spoke([-0.17, 0.05], [-0.01, 0.05]), spoke([-0.17, 0.05], [-0.06, -0.1])), min(spoke([-0.01, 0.05], [-0.06, -0.1]), spoke([-0.06, -0.1], [0.12, -0.1]))), min(min(spoke([-0.01, 0.05], [0.12, -0.1]), spoke([0.12, -0.1], [0.17, 0.05])), min(spoke([-0.1, -0.118], [-0.02, -0.118]), spoke([0.12, -0.1], [0.09, -0.155]))));
  const pictogram = min(min(wheel(-0.17), wheel(0.17)), frameLines);
  const paint = coverage(pictogram).mul(noise(g.mul(31)).mul(0.25).add(0.62));

  let color: N<'vec3'> = water;
  color = mix(color, quay, coverage(shore));
  color = mix(color, bridgeDeck, coverage(deck));
  const walk = max(coverage(shore), coverage(deck));
  color = mix(color, rgb(PALETTE.polish), worn.mul(walk).mul(0.38));
  color = mix(color, rgb(PALETTE.paint), paint.mul(walk));
  color = mix(color, stone, coverage(parapet));
  color = mix(color, rgb(PALETTE.arris), coverage(pillar.sub(0.012)).mul(0.8));
  color = mix(color, cap, coverage(pillar));
  color = mix(color, pierTop, coverage(pier));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const light = max(lit(vec3(1)).sub(vec3(ambient.r, ambient.g, ambient.b)), vec3(0)).mul(glint);
  const glare = smoothstep(0.1, 0.75, chop).mul(0.55).add(0.12).mul(coverage(shore).oneMinus()).mul(coverage(deck).oneMinus()).mul(coverage(pier).oneMinus());
  material.colorNode = lit(color).add(light.mul(glare).mul(REFLECT));
  return { material, map, wear, time, glint, start };
}
