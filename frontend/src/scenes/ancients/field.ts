import * as THREE from 'three/webgpu';
import { abs, atan, dot, exp, float, floor, fract, length, max, min, mix, mx_fractal_noise_float, mx_noise_float, mx_worley_noise_float, positionWorld, sin, smoothstep, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdSegment } from '@kit/shapes2d';
import type { Light2D } from '@kit/light2d';
import { constantList, type EnvDescriptor } from '../../data/envs';
import { lanePoint, laneShape, type AncientsLayout, type LaneShape } from '@sim/envs/ancients';
import { SpriteBatch } from './sprites';

type N<T extends string> = THREE.Node<T>;

const SCALE = 1 / 512;
const LANE_HALF = 0.46;
const PLAZA = 3.7;
const CLEARING = 1.25;
const TAU = Math.PI * 2;

export interface Point { x: number; z: number }

export class Board {
  readonly shape: LaneShape;
  readonly structureX: number[];
  readonly structureY: number[];
  readonly structureHp: number[];
  readonly campX: number[];
  readonly campY: number[];
  readonly fountainX: number[];
  readonly fountainY: number[];
  readonly cx: number;
  readonly cy: number;

  constructor(env: EnvDescriptor, readonly layout: AncientsLayout) {
    this.shape = laneShape(env);
    this.structureX = constantList(env, 'ANCIENTS_STRUCTURE_X');
    this.structureY = constantList(env, 'ANCIENTS_STRUCTURE_Y');
    this.structureHp = constantList(env, 'ANCIENTS_STRUCTURE_HP');
    this.campX = constantList(env, 'ANCIENTS_CAMP_X');
    this.campY = constantList(env, 'ANCIENTS_CAMP_Y');
    this.fountainX = constantList(env, 'ANCIENTS_FOUNTAIN_X');
    this.fountainY = constantList(env, 'ANCIENTS_FOUNTAIN_Y');
    this.cx = (this.fountainX[0]! + this.fountainX[1]!) / 2;
    this.cy = (this.fountainY[0]! + this.fountainY[1]!) / 2;
  }

  x(x: number): number { return (x - this.cx) * SCALE; }
  z(y: number): number { return (y - this.cy) * SCALE; }

  at(x: number, y: number, out: Point): Point {
    out.x = this.x(x);
    out.z = this.z(y);
    return out;
  }

  ancient(side: number): number { return (side + 1) * this.layout.own - 1; }

  exists(s: number): boolean {
    const i = s % this.layout.own;
    return i === this.layout.own - 1 || i >> 1 < this.layout.lanes;
  }

  lanePath(lane: number): Point[] {
    const at = { x: 0, y: 0 }, arcs = [0, this.shape.corner[lane]!, this.shape.length[lane]!].filter((arc, i, all) => i === 0 || arc !== all[i - 1]);
    return arcs.map((arc) => { lanePoint(this.shape, lane, arc, at); return { x: this.x(at.x), z: this.z(at.y) }; });
  }

  extent(pad: number): THREE.Box2 {
    const box = new THREE.Box2();
    const add = (x: number, y: number) => box.expandByPoint(new THREE.Vector2(this.x(x), this.z(y)));
    this.fountainX.forEach((x, i) => add(x, this.fountainY[i]!));
    this.structureX.forEach((x, i) => add(x, this.structureY[i]!));
    return box.expandByScalar(pad);
  }
}

export function buildGround(board: Board, light: Light2D): THREE.Mesh {
  const layout = board.layout, p = vec2(positionWorld.x, positionWorld.z);
  let lane: N<'float'> = float(1e3);
  for (let l = 0; l < layout.lanes; l++) {
    const path = board.lanePath(l);
    for (let i = 1; i < path.length; i++) lane = min(lane, sdSegment(p, vec2(path[i - 1]!.x, path[i - 1]!.z), vec2(path[i]!.x, path[i]!.z), 0));
  }
  const ancients = [0, 1].map((side) => vec2(board.x(board.structureX[board.ancient(side)]!), board.z(board.structureY[board.ancient(side)]!)));
  const axis = ancients[1]!.sub(ancients[0]!).normalize();
  const territory = smoothstep(-2.6, 2.6, dot(p, axis).add(mx_noise_float(vec3(p.mul(0.3), 3)).mul(1.6)));

  const broad = mx_fractal_noise_float(vec3(p.mul(0.13), 0), 4, 2, 0.5).mul(0.5).add(0.5);
  const fine = mx_noise_float(vec3(p.mul(2.1), 1)).mul(0.5).add(0.5);
  const grit = mx_noise_float(vec3(p.mul(4.2), 2)).mul(0.5).add(0.5);
  const relief = mx_noise_float(vec3(p.mul(0.9), 4)).sub(mx_noise_float(vec3(p.add(vec2(0.06, 0.06)).mul(0.9), 4))).mul(2);
  const tufts = smoothstep(0.6, 0.85, mx_noise_float(vec3(p.mul(3.1), 5)).mul(0.5).add(0.5));
  const pebbles = smoothstep(0.05, 0.02, mx_worley_noise_float(vec3(p.mul(1.4), 7), 1)).mul(smoothstep(0.5, 0.75, mx_noise_float(vec3(p.mul(0.45), 8)).mul(0.5).add(0.5)));
  const moss = mix(mix(rgb(0x34422c), rgb(0x505f38), broad), rgb(0x677a42), tufts.mul(0.6));
  const ash = mix(mix(rgb(0x3a2f2c), rgb(0x584439), broad), rgb(0x6b4a3c), tufts.mul(0.55));
  let ground: N<'vec3'> = mix(moss, ash, territory).mul(grit.mul(0.08).add(0.96)).mul(relief.add(1));
  ground = mix(ground, rgb(0x8a857a), pebbles.mul(0.45));

  const stones = mx_worley_noise_float(vec3(p.mul(2.6), 0), 1);
  const path = mix(rgb(0x7a6c56), rgb(0x9c8c6e), broad.mul(0.6).add(fine.mul(0.4))).mul(smoothstep(0.02, 0.2, stones).mul(0.3).add(0.72));
  const rut = exp(lane.mul(lane).mul(-30)).mul(0.12);
  const road = coverage(lane.sub(LANE_HALF));
  const curb = coverage(abs(lane.sub(LANE_HALF)).sub(0.045));
  const verge = smoothstep(LANE_HALF + 0.4, LANE_HALF, lane).mul(float(1).sub(road));
  ground = mix(ground, ground.mul(0.8).add(rgb(0x4a4030).mul(0.2)), verge);
  ground = mix(ground, path.mul(float(1).sub(rut)), road);
  ground = mix(ground, rgb(0x3a3328), curb.mul(0.7));

  for (let side = 0; side < 2; side++) {
    const d = p.sub(ancients[side]!), r = length(d), angle = atan(d.y, d.x);
    const band = floor(r.mul(1.35)), spokes = band.mul(3).add(6);
    const radial = float(0.5).sub(abs(fract(r.mul(1.35)).sub(0.5))).div(1.35);
    const around = float(0.5).sub(abs(fract(angle.div(TAU).mul(spokes).add(band.mul(0.37))).sub(0.5))).mul(r).mul(TAU).div(spokes);
    const joint = min(radial, around);
    const tint = side === 0 ? rgb(0x4c7a64) : rgb(0x84504a);
    const slab = mix(rgb(0x6b6870), tint, 0.3).mul(fine.mul(0.22).add(0.86));
    const tiles = mix(rgb(0x2a2a2e), slab, smoothstep(0.018, 0.05, joint));
    const plaza = coverage(r.sub(PLAZA));
    ground = mix(ground, ground.mul(0.7), smoothstep(PLAZA + 0.7, PLAZA, r).mul(float(1).sub(plaza)));
    ground = mix(ground, tiles, plaza);
    ground = mix(ground, rgb(0x2b2926), coverage(abs(r.sub(PLAZA)).sub(0.06)));
  }

  if (layout.jungle) for (let i = 0; i < board.campX.length; i++) {
    const r = length(p.sub(vec2(board.x(board.campX[i]!), board.z(board.campY[i]!))));
    const soil = mix(rgb(0x3b2f22), rgb(0x584632), fine).mul(grit.mul(0.2).add(0.85));
    ground = mix(ground, soil, coverage(r.sub(CLEARING)).mul(0.9));
    ground = mix(ground, rgb(0x8a8272), coverage(abs(r.sub(CLEARING)).sub(0.05)).mul(smoothstep(0.45, 0.6, fine)));
  }

  const box = board.extent(0.6), reach = max(max(abs(p.x).sub(box.max.x), abs(p.y).sub(box.max.y)), 0);
  ground = ground.mul(exp(reach.mul(-0.9)).mul(0.9).add(0.1));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = light.lit(ground);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(160, 100).rotateX(-Math.PI / 2), material);
  mesh.name = 'ancientsGround';
  mesh.renderOrder = 0;
  return mesh;
}

const POOL_SIZE = 2.2;

export function buildFountains(board: Board, clock: N<'float'>, light: Light2D): SpriteBatch {
  const pools = new SpriteBatch('ancientsFountains', 2, POOL_SIZE, POOL_SIZE, ['aPool'], (channel) => {
    const side = channel('aPool').x;
    const q = uv().sub(0.5).mul(2), r = length(q);
    const water = mix(rgb(0x1f7a66), rgb(0x8a2a3a), side);
    const shimmer = sin(r.mul(26).sub(clock.mul(3))).mul(0.5).add(0.5).mul(smoothstep(0.78, 0.2, r));
    let color: N<'vec3'> = light.lit(mix(rgb(0x4a4744), rgb(0x6e6a64), smoothstep(0.84, 1, r)));
    color = mix(color, water.mul(float(1.1).add(shimmer.mul(0.9))), coverage(r.sub(0.8)));
    color = mix(color, mix(rgb(0xbff7e6), rgb(0xffc2b8), side).mul(2.2), coverage(r.sub(0.16)));
    return { color, opacity: coverage(r.sub(0.98)) };
  }, 1);
  pools.begin();
  for (let side = 0; side < 2; side++) pools.set('aPool', pools.add(board.x(board.fountainX[side]!), board.z(board.fountainY[side]!)), side);
  pools.commit();
  return pools;
}
