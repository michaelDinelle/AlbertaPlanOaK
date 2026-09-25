import * as THREE from 'three/webgpu';
import { abs, atan, clamp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, smoothstep, texture, uniform, uvec2, vec2 } from 'three/tsl';
import { hash2i } from '@kit/rng';
import { coverage, rgb, sdBox } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { brightAt, CORRIDOR, DOOR, FLOOR, open, ROCK, SIDE, STAIRS, terrainAt, WALL } from '@sim/envs/dungeon';
import { hash, type Lit, type N } from './art';

export const HALF = SIDE / 2;
const ERODE = 0.3;
const FAR = 9;

export interface Room { x0: number; y0: number; x1: number; y1: number; lit: boolean; index: number }

export class Board {
  readonly terrain = new Uint8Array(SIDE * SIDE);
  readonly bright = new Uint8Array(SIDE * SIDE);
  readonly map: THREE.DataTexture;
  readonly glow: THREE.DataTexture;
  readonly mottle: THREE.DataTexture;
  readonly grain: THREE.DataTexture;
  rooms: Room[] = [];
  key = -1;

  constructor(seed: number) {
    this.map = new THREE.DataTexture(new Uint8Array(SIDE * SIDE * 4), SIDE, SIDE, THREE.RGBAFormat);
    this.map.magFilter = this.map.minFilter = THREE.NearestFilter;
    this.glow = new THREE.DataTexture(new Uint8Array(SIDE * SIDE * 4), SIDE, SIDE, THREE.RGBAFormat);
    this.glow.magFilter = this.glow.minFilter = THREE.LinearFilter;
    this.mottle = fbm(seed ^ 0x6d07, { size: 256, frequency: 4, octaves: 5 });
    this.grain = fbm(seed ^ 0x9a1e, { size: 256, frequency: 32, octaves: 3 });
  }

  at(x: number, y: number): number {
    return x < 0 || y < 0 || x >= SIDE || y >= SIDE ? ROCK : this.terrain[y * SIDE + x]!;
  }

  open(x: number, y: number): boolean {
    return open(this.at(x, y));
  }

  roomAt(x: number, y: number): Room | null {
    for (const room of this.rooms) if (x >= room.x0 - 1 && x <= room.x1 + 1 && y >= room.y0 - 1 && y <= room.y1 + 1) return room;
    return null;
  }

  paint(view: Uint32Array, key: number): void {
    this.key = key;
    const { terrain, bright } = this;
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) {
      terrain[y * SIDE + x] = terrainAt(view, x, y);
      bright[y * SIDE + x] = brightAt(view, x, y) ? 1 : 0;
    }
    this.rooms = [];
    const seen = new Uint8Array(SIDE * SIDE);
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) {
      const t = terrain[y * SIDE + x]!;
      if (seen[y * SIDE + x] || (t !== FLOOR && t !== STAIRS)) continue;
      const room: Room = { x0: x, y0: y, x1: x, y1: y, lit: bright[y * SIDE + x] === 1, index: 0 };
      const stack = [y * SIDE + x];
      seen[y * SIDE + x] = 1;
      while (stack.length) {
        const at = stack.pop()!, cx = at % SIDE, cy = (at - cx) / SIDE;
        room.x0 = Math.min(room.x0, cx); room.x1 = Math.max(room.x1, cx); room.y0 = Math.min(room.y0, cy); room.y1 = Math.max(room.y1, cy);
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const nx = cx + dx, ny = cy + dy, n = ny * SIDE + nx, k = this.at(nx, ny);
          if (seen[n] || nx < 0 || ny < 0 || nx >= SIDE || ny >= SIDE || (k !== FLOOR && k !== STAIRS)) continue;
          seen[n] = 1;
          stack.push(n);
        }
      }
      room.index = Math.floor((room.y0 - 1) / 10) * 3 + Math.floor((room.x0 - 1) / 10);
      this.rooms.push(room);
    }
    const ring = (x: number, y: number) => { const k = this.at(x, y); return k === WALL || k === DOOR; };
    const cells = this.map.image.data as Uint8Array, glow = this.glow.image.data as Uint8Array;
    const distance = new Float32Array(SIDE * SIDE).fill(FAR), queue: number[] = [];
    for (let i = 0; i < SIDE * SIDE; i++) if (open(terrain[i]!)) { distance[i] = 0; queue.push(i); }
    for (let head = 0; head < queue.length; head++) {
      const at = queue[head]!, cx = at % SIDE, cy = (at - cx) / SIDE;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = cx + dx, ny = cy + dy, n = ny * SIDE + nx;
        if (nx < 0 || ny < 0 || nx >= SIDE || ny >= SIDE || distance[n]! <= distance[at]! + 1) continue;
        distance[n] = distance[at]! + 1;
        queue.push(n);
      }
    }
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) {
      const i = y * SIDE + x, t = terrain[i]!, o = i * 4;
      const across = ring(x - 1, y) || ring(x + 1, y), along = ring(x, y - 1) || ring(x, y + 1);
      const shape = t === WALL ? (across && along ? 3 : along ? 2 : 1) : t === DOOR ? (along ? 2 : 1) : 0;
      cells[o] = t * 32;
      cells[o + 1] = bright[i]! * 255;
      cells[o + 2] = shape * 64;
      cells[o + 3] = hash2i(x, y, key) & 255;
      glow[o] = Math.round(255 * Math.exp(-distance[i]! / 2.4));
      glow[o + 1] = bright[i]! * 255;
      glow[o + 2] = 0;
      glow[o + 3] = 255;
    }
    this.map.needsUpdate = true;
    this.glow.needsUpdate = true;
  }

  dispose(): void {
    for (const texture of [this.map, this.glow, this.mottle, this.grain]) texture.dispose();
  }
}

const cellSpace = (): N<'vec2'> => vec2(positionWorld.x.add(HALF), positionWorld.z.add(HALF));

interface Cell { t: N<'float'>; lit: N<'float'>; shape: N<'float'>; seed: N<'float'> }

function reader(board: Board) {
  const map = texture(board.map);
  return (c: N<'vec2'>): Cell => {
    const inside = c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.x.lessThan(SIDE)).and(c.y.lessThan(SIDE));
    const texel = map.load(uvec2(clamp(c, vec2(0), vec2(SIDE - 1))));
    const t = select(inside, floor(texel.r.mul(255 / 32).add(0.5)), float(ROCK));
    return { t, lit: select(inside, texel.g, float(0)), shape: floor(texel.b.mul(255 / 64).add(0.5)), seed: texel.a };
  };
}

interface Field { dOpen: N<'float'>; dSolid: N<'float'>; dCorridor: N<'float'>; dRing: N<'float'>; shade: N<'float'> }

function field(q: N<'vec2'>, read: (c: N<'vec2'>) => Cell, shadow: N<'vec2'>): Field {
  const base = floor(q);
  let dOpen: N<'float'> = float(4), dSolid: N<'float'> = float(4), dCorridor: N<'float'> = float(4), dRing: N<'float'> = float(4), shade: N<'float'> = float(4);
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
    const n = base.add(vec2(i, j)), k = read(n).t, box = sdBox(q.sub(n).sub(0.5), vec2(0.5)), cast = sdBox(q.sub(shadow).sub(n).sub(0.5), vec2(0.5));
    const isOpen = k.greaterThanEqual(FLOOR), isCorridor = k.equal(CORRIDOR);
    dOpen = min(dOpen, select(isOpen, box, float(4)));
    dSolid = min(dSolid, select(isOpen, float(4), box));
    dCorridor = min(dCorridor, select(isCorridor, box, float(4)));
    dRing = min(dRing, select(k.equal(WALL), box, float(4)));
    shade = min(shade, select(isOpen, float(4), cast));
  }
  return { dOpen, dSolid, dCorridor, dRing, shade };
}

export function occluderMask(board: Board): N<'float'> {
  const q = cellSpace(), read = reader(board), here = read(floor(q));
  const { dOpen } = field(q, read, vec2(0));
  return select(here.t.lessThan(FLOOR), select(dOpen.greaterThan(ERODE), float(1), float(0)), float(0));
}

export function groundMaterial(board: Board, lit: Lit): THREE.MeshBasicNodeMaterial {
  const q = cellSpace(), read = reader(board), cell = floor(q), f = fract(q), here = read(cell);
  const { dOpen, dCorridor, dRing, shade } = field(q, read, vec2(0.1, 0.14));
  const mott = texture(board.mottle, q.mul(0.045)).r, broad = texture(board.mottle, q.mul(0.013).add(0.37)).r;
  const fine = texture(board.grain, q.mul(0.11)).r, speck = texture(board.grain, q.mul(0.53).add(0.21)).r;
  const openness = texture(board.glow, q.div(SIDE)).r;

  let rock: N<'vec3'> = mix(rgb(0x17181d), rgb(0x3a3a42), mott.mul(0.7).add(broad.mul(0.3)));
  rock = rock.mul(fine.mul(0.4).add(0.8)).mul(mix(float(0.3), float(1), openness));
  rock = mix(rock, rgb(0x3a3a42).mul(openness), smoothstep(0.72, 0.9, speck).mul(0.5));
  rock = mix(rock, rgb(0x4a4744).mul(openness.add(0.2)), smoothstep(0, 0.16, dOpen).oneMinus().mul(0.55));

  const pebbleAt = q.mul(5.2), pebbleCell = floor(pebbleAt), jitter = vec2(hash(pebbleCell), hash(pebbleCell.add(19.7))).sub(0.5).mul(0.45);
  const pebble = length(fract(pebbleAt).sub(0.5).sub(jitter)).sub(hash(pebbleCell.add(7.1)).mul(0.16).add(0.14));
  let gravel: N<'vec3'> = mix(rgb(0x564838), rgb(0x7a6852), mott).mul(fine.mul(0.3).add(0.85));
  gravel = mix(gravel, mix(rgb(0x6d6153), rgb(0x857a6b), hash(pebbleCell.add(3.3))), coverage(pebble.mul(0.19)).mul(0.75));
  const ragged = texture(board.grain, q.mul(0.21).add(0.61)).r.sub(0.5).mul(0.16).add(mott.sub(0.5).mul(0.1));
  const edge = dCorridor.add(ragged);
  const tunnel = coverage(edge.add(0.02));
  gravel = gravel.mul(smoothstep(0, 0.26, edge.negate()).mul(0.4).add(0.6));

  const row = floor(q.y.mul(2)), slabAt = q.x.mul(1.4).add(hash(vec2(row, 5.1))), slab = floor(slabAt);
  const joint = min(float(0.5).sub(abs(fract(slabAt).sub(0.5))).div(1.4), float(0.5).sub(abs(fract(q.y.mul(2)).sub(0.5))).div(2));
  let flag: N<'vec3'> = mix(rgb(0x6f6353), rgb(0x8e8069), hash(vec2(slab, row))).mul(mott.mul(0.24).add(0.88)).mul(fine.mul(0.16).add(0.92));
  flag = flag.mul(smoothstep(0, 0.07, joint).mul(0.16).add(0.84));
  flag = mix(flag, rgb(0x2d2822), coverage(joint.sub(0.012)));
  flag = flag.mul(smoothstep(0, 0.42, shade).mul(0.42).add(0.58));

  const vertical = here.shape.equal(2), u = select(vertical, f.y, f.x), v = select(vertical, f.x, f.y);
  const course = floor(v.mul(3)), brickAt = u.mul(2).add(mod(course, 2).mul(0.5)), brick = floor(brickAt);
  const brickEdge = min(float(0.5).sub(abs(fract(brickAt).sub(0.5))).div(2), float(0.5).sub(abs(fract(v.mul(3)).sub(0.5))).div(3));
  const corner = here.shape.equal(3), block = sdBox(f.sub(0.5), vec2(0.4), 0.05);
  let masonry: N<'vec3'> = mix(rgb(0x5f584e), rgb(0x857c6e), hash(vec2(brick, course).add(cell.mul(7.3)))).mul(fine.mul(0.22).add(0.88));
  masonry = masonry.mul(select(corner, smoothstep(-0.2, 0.05, block).oneMinus().mul(0.25).add(0.85), smoothstep(0, 0.06, brickEdge).mul(0.18).add(0.82)));
  masonry = mix(masonry, rgb(0x221e1a), select(corner, coverage(block).oneMinus(), coverage(brickEdge.sub(0.014))));
  masonry = masonry.mul(select(corner, float(1), smoothstep(0, 0.08, min(v, v.oneMinus())).mul(0.3).add(0.7)));

  const jamb = min(u, u.oneMinus()).lessThan(0.17), across = u.sub(0.17).div(0.66).mul(3);
  const plankEdge = float(0.5).sub(abs(fract(across).sub(0.5))).mul(0.22);
  let wood: N<'vec3'> = mix(rgb(0x5a3a22), rgb(0x7d5533), hash(vec2(floor(across), cell.x.add(cell.y.mul(31))))).mul(texture(board.grain, vec2(u.mul(0.2), v.mul(0.02)).add(cell.mul(0.13))).r.mul(0.35).add(0.8));
  wood = mix(rgb(0x2a1a10), wood, smoothstep(0.004, 0.02, plankEdge));
  wood = mix(wood, rgb(0x3b3a3c), coverage(min(abs(v.sub(0.25)), abs(v.sub(0.75))).sub(0.04)).mul(0.9));
  const door = select(jamb, masonry, wood);

  const centre = f.sub(0.5), radius = length(centre), turnAt = fract(atan(centre.y, centre.x).div(2 * Math.PI).add(0.25)), tread = floor(turnAt.mul(9));
  const well = radius.sub(0.4), pit = radius.sub(0.13), riser = float(0.5).sub(abs(fract(turnAt.mul(9)).sub(0.5))).mul(radius).mul(0.7);
  let stair: N<'vec3'> = mix(rgb(0xd8ccb4), rgb(0x3a332b), tread.div(9)).mul(fine.mul(0.2).add(0.9)).mul(smoothstep(0.13, 0.4, radius).mul(0.3).add(0.7));
  stair = mix(stair, rgb(0x1c1814), coverage(riser.sub(0.006)).mul(smoothstep(0.1, 0.16, radius)));
  stair = mix(stair, rgb(0x080706), coverage(pit));
  stair = mix(flag, mix(rgb(0x1a1612), stair, coverage(well.add(0.03))), coverage(well));
  stair = mix(stair, rgb(0xe8dcc4), coverage(abs(well.add(0.015)).sub(0.012)).mul(0.7));

  let albedo: N<'vec3'> = rock;
  albedo = mix(albedo, gravel, tunnel);
  albedo = select(here.t.equal(FLOOR), flag, albedo);
  albedo = select(here.t.equal(STAIRS), stair, albedo);
  albedo = select(here.t.equal(WALL), masonry, albedo);
  albedo = select(here.t.equal(DOOR), door, albedo);
  const ink = coverage(abs(dRing).sub(0.016)).mul(select(here.t.equal(ROCK).or(here.t.equal(CORRIDOR)), float(1), float(0)));
  albedo = mix(albedo, rgb(0x08080a), ink.mul(0.85));

  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  material.colorNode = lit(albedo);
  return material;
}

export interface VeilUniforms {
  now: THREE.UniformNode<'vec4', THREE.Vector4>[];
  was: THREE.UniformNode<'vec4', THREE.Vector4>[];
  blend: THREE.UniformNode<'float', number>;
  strength: THREE.UniformNode<'float', number>;
  iris: THREE.UniformNode<'vec4', THREE.Vector4>;
  tint: THREE.UniformNode<'color', THREE.Color>;
}

export const EMPTY = new THREE.Vector4(-99, -99, -98, -98);

export function veilMaterial(): { material: THREE.MeshBasicNodeMaterial; uniforms: VeilUniforms } {
  const uniforms: VeilUniforms = {
    now: [uniform(EMPTY.clone()), uniform(EMPTY.clone())],
    was: [uniform(EMPTY.clone()), uniform(EMPTY.clone())],
    blend: uniform(1),
    strength: uniform(0.46),
    iris: uniform(new THREE.Vector4(0, 0, 999, 0)),
    tint: uniform(new THREE.Color(0x03050c)),
  };
  const q = cellSpace();
  const box = (r: N<'vec4'>) => sdBox(q.sub(r.xy.add(r.zw).mul(0.5)), r.zw.sub(r.xy).mul(0.5));
  const was = min(box(uniforms.was[0]!), box(uniforms.was[1]!)), now = min(box(uniforms.now[0]!), box(uniforms.now[1]!));
  const veil = mix(smoothstep(-0.03, 0.2, was), smoothstep(-0.03, 0.2, now), uniforms.blend).mul(uniforms.strength);
  const border = mix(coverage(abs(was.add(0.02)).sub(0.022)), coverage(abs(now.add(0.02)).sub(0.022)), uniforms.blend).mul(0.4);
  const iris = smoothstep(-1.2, 1.2, length(q.sub(uniforms.iris.xy)).sub(uniforms.iris.z)).mul(uniforms.iris.w);
  const shade = max(veil, iris);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const alpha = max(shade, border.mul(iris.oneMinus()));
  material.colorNode = mix(uniforms.tint, rgb(0xffdca8).mul(1.2), border.mul(iris.oneMinus()).div(max(alpha, 1e-4)));
  material.opacityNode = alpha;
  return { material, uniforms };
}
