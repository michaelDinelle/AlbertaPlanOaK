import * as THREE from 'three/webgpu';
import { abs, clamp, dot, float, length, max, min, mix, select, sign, smoothstep, sqrt, uniform, vec2 } from 'three/tsl';
import { coverage, noise, rgb, sdCircle, sdSegment } from '@kit/shapes2d';
import { FLOOR, LEVER_DROP, LEVER_HALF, LEVER_LENGTH, PIVOT, picture, sheet, type Lit, type Point } from './art';

type N<T extends string> = THREE.Node<T>;
const RUMP = 0.47;
const CHEST = 0.34;
const SPAN = 1.3;
const HIP = new THREE.Vector2(-3.95, FLOOR + 0.6);
const REAR_SHIFT = new THREE.Vector2(0.62, -0.06);
const EAT_SHIFT = 0.5;
const REAR_TILT = 0.95;
const TAIL = 8;
export const RAT = { x0: -6.9, x1: -1.2, y0: FLOOR - 0.2, y1: 0.15 };

export interface RatPose {
  rear: number;
  lever: number;
  dip: number;
  look: number;
  chew: number;
  breath: number;
  sniff: number;
  sway: number;
  glint: number;
}

const rotate = (x: number, y: number, angle: number, out: THREE.Vector2): THREE.Vector2 =>
  out.set(x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle));
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export class RatRig {
  readonly hip: Point = uniform(new THREE.Vector2());
  readonly axis: Point = uniform(new THREE.Vector2(1, 0));
  readonly chest: Point = uniform(new THREE.Vector2());
  readonly skull: Point = uniform(new THREE.Vector2());
  readonly snout: Point = uniform(new THREE.Vector2());
  readonly face: Point = uniform(new THREE.Vector2(1, 0));
  readonly ear: Point = uniform(new THREE.Vector2());
  readonly eye: Point = uniform(new THREE.Vector2());
  readonly whisker: Point = uniform(new THREE.Vector2());
  readonly shoulder: Point = uniform(new THREE.Vector2());
  readonly paw: Point = uniform(new THREE.Vector2());
  readonly farShoulder: Point = uniform(new THREE.Vector2());
  readonly farPaw: Point = uniform(new THREE.Vector2());
  readonly knee: Point = uniform(new THREE.Vector2());
  readonly heel: Point = uniform(new THREE.Vector2());
  readonly tail: Point[] = Array.from({ length: TAIL }, () => uniform(new THREE.Vector2()));
  readonly breath = uniform(1);
  readonly glint = uniform(0);
  readonly whiskerSpread = uniform(0);
  private readonly v = new THREE.Vector2();
  private readonly w = new THREE.Vector2();

  private at(base: THREE.Vector2, x: number, y: number, angle: number, out: THREE.Vector2): THREE.Vector2 {
    return out.copy(base).add(rotate(x, y, angle, this.v));
  }

  leverPoint(along: number, drop: number, out: THREE.Vector2): THREE.Vector2 {
    const angle = drop * LEVER_DROP, c = Math.cos(angle), s = Math.sin(angle), lift = LEVER_HALF + 0.045;
    return out.set(PIVOT.x - along * c - lift * s, PIVOT.y - along * s + lift * c);
  }

  pose(p: RatPose): void {
    const rear = p.rear, dip = p.dip * (1 - rear);
    const hip = this.hip.value.set(HIP.x + REAR_SHIFT.x * rear + EAT_SHIFT * smooth(0, 1, dip), HIP.y + REAR_SHIFT.y * rear - 0.02 * dip);
    const theta = REAR_TILT * rear - 0.08 * dip;
    this.axis.value.set(Math.cos(theta), Math.sin(theta));
    const chest = this.at(hip, SPAN, 0.02 + 0.012 * (p.breath - 1) * 20, theta, this.chest.value);
    const neck = this.at(chest, 0.22, 0.12, theta, this.w);
    const psi = theta + 0.5 * p.look * (1 - rear) - 0.6 * rear - 0.62 * dip + p.sniff + 0.05 * p.chew * dip;
    this.face.value.set(Math.cos(psi), Math.sin(psi));
    const skull = this.at(neck, 0.14, 0, psi, this.skull.value);
    this.at(skull, 0.53, -0.06, psi, this.snout.value);
    this.at(skull, -0.13, 0.2, psi, this.ear.value);
    this.at(skull, 0.17, 0.07, psi, this.eye.value);
    this.at(skull, 0.47, -0.035, psi, this.whisker.value);
    const shoulder = this.at(chest, 0.06, -0.18, theta, this.shoulder.value);
    const far = this.at(chest, -0.04, -0.13, theta, this.farShoulder.value);
    const reach = smooth(0.3, 0.85, rear), lift = 0.22 * Math.sin(Math.PI * reach);
    const floorNear = this.v.set(shoulder.x + 0.1, FLOOR + 0.045);
    const onLever = this.leverPoint(LEVER_LENGTH * 0.84, p.lever, this.w);
    this.paw.value.set(floorNear.x + (onLever.x - floorNear.x) * reach, floorNear.y + (onLever.y - floorNear.y) * reach + lift);
    const floorFar = this.v.set(far.x - 0.12, FLOOR + 0.045);
    const onLeverFar = this.leverPoint(LEVER_LENGTH * 0.68, p.lever, this.w);
    this.farPaw.value.set(floorFar.x + (onLeverFar.x - floorFar.x) * reach, floorFar.y + (onLeverFar.y - floorFar.y) * reach + lift * 0.8);
    this.at(hip, 0.2, -0.3, theta, this.knee.value);
    this.heel.value.set(hip.x - 0.1 - 0.12 * rear, FLOOR + 0.07);
    const base = this.at(hip, -RUMP + 0.07, -0.13, theta, this.w);
    const bx = base.x, by = base.y;
    const cx = bx - 0.55, cy = FLOOR + 0.1 + 0.05 * rear;
    const tx = bx - 2.0 + 0.06 * p.sway, ty = FLOOR + 0.05 + 0.035 * p.sway;
    this.tail.forEach((point, i) => {
      const s = i / (TAIL - 1), a = (1 - s) * (1 - s), b = 2 * (1 - s) * s, c = s * s;
      point.value.set(a * bx + b * cx + c * tx, a * by + b * cy + c * ty + 0.02 * p.sway * s * s);
    });
    this.breath.value = p.breath;
    this.glint.value = p.glint;
    this.whiskerSpread.value = p.sniff * 2.5 + 0.08 * p.chew * dip;
  }
}

function roundCone(p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, r1: number, r2: number): N<'float'> {
  const ba = b.sub(a), l2 = dot(ba, ba), rr = r1 - r2, a2 = l2.sub(rr * rr), il2 = float(1).div(l2);
  const pa = p.sub(a), y = dot(pa, ba), z = y.sub(l2);
  const xv = pa.mul(l2).sub(ba.mul(y)), x2 = dot(xv, xv);
  const y2 = y.mul(y).mul(l2), z2 = z.mul(z).mul(l2);
  const k = x2.mul(Math.sign(rr) * rr * rr);
  const tip = sqrt(x2.add(z2)).mul(il2).sub(r2);
  const root = sqrt(x2.add(y2)).mul(il2).sub(r1);
  const side = sqrt(x2.mul(a2).mul(il2)).add(y.mul(rr)).mul(il2).sub(r1);
  return select(sign(z).mul(a2).mul(z2).greaterThan(k), tip, select(sign(y).mul(a2).mul(y2).lessThan(k), root, side));
}

const smin = (a: N<'float'>, b: N<'float'>, k: number): N<'float'> => {
  const h = max(float(k).sub(abs(a.sub(b))), 0).div(k);
  return min(a, b).sub(h.mul(h).mul(k * 0.25));
};

export interface RatShape {
  silhouette: N<'float'>;
  color: N<'vec3'>;
  alpha: N<'float'>;
}

export function ratShape(rig: RatRig, lit: Lit): RatShape {
  const q = picture();
  const d = q.sub(rig.hip), u = dot(d, rig.axis), v = dot(d, vec2(rig.axis.y.negate(), rig.axis.x));
  const torso = roundCone(q, rig.hip, rig.chest, RUMP, CHEST).sub(rig.breath.sub(1).mul(0.5));
  const head = roundCone(q, rig.skull, rig.snout, 0.24, 0.07);
  const body = smin(torso, head, 0.3);
  const ear = sdCircle(q.sub(rig.ear), 0.13);
  const thigh = roundCone(q, rig.knee, rig.heel, 0.12, 0.05);
  const foot = sdSegment(q, rig.heel, rig.heel.add(vec2(0.3, -0.02)), 0.042);
  const leg = roundCone(q, rig.shoulder, rig.paw, 0.085, 0.04);
  const paw = sdCircle(q.sub(rig.paw), 0.05);
  const farLeg = min(roundCone(q, rig.farShoulder, rig.farPaw, 0.08, 0.036), sdCircle(q.sub(rig.farPaw), 0.046));
  let tail: N<'float'> = float(1e3);
  for (let i = 0; i < TAIL - 1; i++) tail = min(tail, roundCone(q, rig.tail[i]!, rig.tail[i + 1]!, 0.07 - (0.052 * i) / (TAIL - 1), 0.07 - (0.052 * (i + 1)) / (TAIL - 1)));
  const near = min(min(body, ear), min(min(thigh, foot), min(leg, paw)));
  const silhouette = min(min(near, farLeg), tail);

  const fu = dot(q.sub(rig.skull), rig.face), fv = dot(q.sub(rig.skull), vec2(rig.face.y.negate(), rig.face.x));
  const inHead = smoothstep(0.08, -0.04, head);
  const fur = noise(vec2(u.mul(6), v.mul(22))).mul(0.05).add(noise(vec2(fu.mul(9), fv.mul(20))).mul(0.03).mul(inHead)).add(1);
  const top = mix(v, fv, inHead);
  const roundness = smoothstep(-0.5, 0.42, top).mul(0.28).add(0.72).mul(smoothstep(0, 0.1, body.negate()).mul(0.16).add(0.84));
  const white = rgb(0xf6f1ea), shade = rgb(0xc7bdb6), pink = rgb(0xeeaaa9), rose = rgb(0xe38d95);
  const furColor = mix(shade, white, roundness).mul(fur);
  const along = clamp(length(q.sub(rig.tail[0]!)).div(2.1), 0, 1);
  let color: N<'vec3'> = lit(mix(shade, pink, 0.35).mul(0.72));
  color = mix(color, lit(mix(rgb(0xe9b7b6), rgb(0xc98f93), along)), coverage(tail));
  color = mix(color, lit(pink.mul(0.92)), coverage(foot));
  color = mix(color, lit(mix(shade, white, 0.55).mul(fur)), coverage(thigh));
  color = mix(color, lit(furColor), coverage(body));
  color = mix(color, lit(mix(rgb(0xd9c4c0), rgb(0xf1dcd8), smoothstep(-0.13, 0.13, q.y.sub(rig.ear.y)))), coverage(ear));
  color = mix(color, lit(rose.mul(0.95)), coverage(sdCircle(q.sub(rig.ear.add(rig.face.mul(0.03)).sub(vec2(0, 0.012))), 0.075)));
  color = mix(color, lit(mix(shade, white, 0.75).mul(fur)), coverage(leg));
  color = mix(color, lit(pink), coverage(paw));
  const eyeQ = q.sub(rig.eye);
  color = mix(color, rgb(0x5a0c14), coverage(sdCircle(eyeQ, 0.047)));
  color = mix(color, rgb(0xb3202e), coverage(sdCircle(eyeQ.add(vec2(0.008, 0.008)), 0.03)));
  const catchlight = coverage(sdCircle(eyeQ.sub(vec2(0.016, 0.018)), 0.014));
  color = mix(color, mix(rgb(0xffffff), rgb(0xffd9a0).mul(2.6), rig.glint), catchlight);
  color = mix(color, lit(rose), coverage(sdCircle(q.sub(rig.snout).sub(rig.face.mul(0.045)), 0.04)));
  let whiskers: N<'float'> = float(0);
  for (const bend of [0.42, 0.12, -0.2]) {
    const angle = rig.whiskerSpread.add(bend);
    const dir = vec2(rig.face.x.mul(angle.cos()).sub(rig.face.y.mul(angle.sin())), rig.face.x.mul(angle.sin()).add(rig.face.y.mul(angle.cos())));
    whiskers = max(whiskers, coverage(sdSegment(q, rig.whisker, rig.whisker.add(dir.mul(0.36)), 0.005)));
  }
  const outlined = mix(rgb(0x1a1413), color, coverage(silhouette));
  return {
    silhouette,
    color: mix(outlined, lit(rgb(0xf4efe6)), whiskers.mul(0.8)),
    alpha: max(coverage(silhouette.sub(0.016)), whiskers.mul(0.6)),
  };
}

export function ratMaterial(shape: RatShape): THREE.MeshBasicNodeMaterial {
  return sheet(shape.color, shape.alpha);
}
