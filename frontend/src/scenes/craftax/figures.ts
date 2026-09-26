import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, dot, float, floor, fract, max, min, mix, positionWorld, select, sin, smoothstep, uv, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { dimmed, ellipse, rotate, smin, type Shade, type WorldUniforms } from './world';

type N<T extends string> = THREE.Node<T>;
type Ink = { color: N<'vec3'>; opacity: N<'float'> };
export type FigureShade = (a: N<'vec4'>, b: N<'vec4'>, p: N<'vec2'>) => Ink;

export const FIGURE = 1.5;
const INK = 0x16120f;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export class Figures {
  readonly mesh: THREE.InstancedMesh;
  private readonly a: THREE.InstancedBufferAttribute;
  private readonly b: THREE.InstancedBufferAttribute;
  private readonly matrix = new THREE.Matrix4();
  private used = 0;

  constructor(name: string, readonly capacity: number, shade: FigureShade, side: number, u: WorldUniforms, order: number, zoom: number, size = FIGURE) {
    const geometry = flat(size, size);
    this.a = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.b = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aFigA', this.a);
    geometry.setAttribute('aFigB', this.b);
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = uv().sub(0.5).mul(size / zoom), g = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2));
    const ink = shade(attribute<'vec4'>('aFigA', 'vec4'), attribute<'vec4'>('aFigB', 'vec4'), p);
    material.colorNode = dimmed(ink.color, g, u);
    material.opacityNode = ink.opacity;
    this.mesh = instancedMesh(geometry, material, capacity);
    this.mesh.name = name;
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  add(x: number, z: number, a: readonly [number, number, number, number], b: readonly [number, number, number, number] = [0, 0, 0, 0]): void {
    if (this.used >= this.capacity) throw new Error(`${this.mesh.name} holds ${this.capacity} figures`);
    this.mesh.setMatrixAt(this.used, this.matrix.makeTranslation(x, 0, z));
    this.a.setXYZW(this.used, ...a);
    this.b.setXYZW(this.used, ...b);
    this.used++;
  }

  commit(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.a.needsUpdate = true;
    this.b.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.removeFromParent();
  }
}

const composite = (layers: readonly (readonly [N<'float'>, N<'vec3'>])[], shadow: N<'float'>, alpha: N<'float'>): Ink => {
  let color: N<'vec3'> = rgb(INK), solid: N<'float'> = float(0);
  for (const [d, tint] of layers) {
    color = mix(color, rgb(INK), coverage(d.sub(0.018)));
    color = mix(color, tint, coverage(d));
    solid = max(solid, coverage(d.sub(0.018)));
  }
  const opacity = solid.add(shadow.mul(solid.oneMinus()));
  return { color: color.mul(solid).div(max(opacity, 1e-4)), opacity: opacity.mul(alpha) };
};

const heading = (p: N<'vec2'>, angle: N<'float'>) => rotate(p, angle.negate());
const castShadow = (p: N<'vec2'>, radii: readonly [number, number], strength: number) => smoothstep(0.05, -0.06, ellipse(p.sub(vec2(0.05, -0.07)), [0, 0], radii)).mul(strength);

const pips = (p: N<'vec2'>, health: N<'float'>, most: number): { d: N<'float'>; full: N<'float'> } => {
  const width = 0.085, x0 = -(most - 1) * width / 2;
  const slot = clamp(floor(p.x.sub(x0).div(width).add(0.5)), 0, most - 1);
  const centre = vec2(slot.mul(width).add(x0), 0.5);
  return { d: sdCircle(p.sub(centre), 0.03), full: select(slot.lessThan(health.sub(0.5)), float(1), float(0)) };
};

export function playerShade(accent: number, shade: Shade, clock: N<'float'>): FigureShade {
  return (a, b, p) => {
    const alpha = a.y, hurt = a.z, phase = a.w, strike = b.x, sleep = b.y, death = b.z;
    const lying = max(sleep, death);
    const l = rotate(heading(p, a.x), lying.mul(-1.45)).div(lying.mul(0.08).add(1));
    const step = sin(phase.mul(Math.PI * 2)).mul(0.09).mul(lying.oneMinus());
    const boots = min(ellipse(l.sub(vec2(-0.1, step)), [0, 0], [0.065, 0.085]), ellipse(l.sub(vec2(0.1, step.negate())), [0, 0], [0.065, 0.085]));
    const pack = sdBox(l.sub(vec2(0, -0.15)), vec2(0.13, 0.07), 0.04);
    const torso = sdBox(l.sub(vec2(0, -0.02)), vec2(0.21, 0.11), 0.1);
    const reach = strike.mul(0.2);
    const swing = step.mul(-0.7);
    const leftHand = vec2(-0.25, swing.add(0.06)), rightHand = vec2(reach.mul(-0.45).add(0.25), swing.negate().add(0.06).add(reach));
    const sleeves = min(sdSegment(l, vec2(-0.2, -0.02), leftHand, 0.06), sdSegment(l, vec2(0.2, -0.02), rightHand, 0.06));
    const hands = min(sdCircle(l.sub(leftHand), 0.052), sdCircle(l.sub(rightHand), 0.052));
    const head = sdCircle(l.sub(vec2(0, 0.02)), 0.135);
    const face = max(head, sdCircle(l.sub(vec2(0, 0.075)), 0.105));
    const hair = max(head, sdCircle(l.sub(vec2(0, -0.02)), 0.125));
    const eyes = min(sdCircle(l.sub(vec2(-0.045, 0.125)), 0.017), sdCircle(l.sub(vec2(0.045, 0.125)), 0.017));
    const tunic = rgb(accent).mul(smoothstep(0.25, -0.2, p.x.sub(p.y)).mul(0.3).add(0.8));
    const skin = rgb(0xe9b98e), shut = sleep.greaterThan(0.5), gone = death.greaterThan(0.5);
    const cross = (x: number) => min(sdSegment(l.sub(vec2(x, 0.12)), vec2(-0.024, -0.024), vec2(0.024, 0.024), 0.009), sdSegment(l.sub(vec2(x, 0.12)), vec2(-0.024, 0.024), vec2(0.024, -0.024), 0.009));
    const body = composite([
      [boots, rgb(0x3b2a1d)],
      [pack, rgb(0x6d4a2b)],
      [sleeves, rgb(accent).mul(0.72)],
      [torso, tunic],
      [hands, skin],
      [face, skin],
      [hair, mix(rgb(0x4b301c), rgb(0x6e4a2c), smoothstep(0.1, -0.1, l.x.add(l.y).sub(0.02)))],
      [select(gone, min(cross(-0.05), cross(0.05)), select(shut, max(abs(l.y.sub(0.123)).sub(0.006), abs(abs(l.x).sub(0.045)).sub(0.026)), eyes)), rgb(0x1a1411)],
    ], castShadow(p, [0.29, 0.25], 0.38), alpha);
    const red = mix(shade(body.color), rgb(0xe83a2a), hurt.mul(0.5));
    const grey = mix(red, vec3(dot(red, vec3(0.3, 0.5, 0.2))).mul(0.85), death.mul(0.8));
    let zInk: N<'float'> = float(0), zFace: N<'float'> = float(0);
    for (let k = 0; k < 3; k++) {
      const rise = fract(clock.mul(0.45).add(k / 3)), grow = rise.mul(0.6).add(0.9), at = p.sub(vec2(0.24, 0.16).add(vec2(rise.mul(0.12), rise.mul(0.38)))).div(grow);
      const glyph = min(min(sdSegment(at, vec2(-0.06, 0.06), vec2(0.06, 0.06), 0.016), sdSegment(at, vec2(0.06, 0.06), vec2(-0.06, -0.06), 0.016)), sdSegment(at, vec2(-0.06, -0.06), vec2(0.06, -0.06), 0.016)).mul(grow);
      const fade = sin(rise.mul(Math.PI)).mul(sleep);
      zInk = max(zInk, coverage(glyph.sub(0.014)).mul(fade));
      zFace = max(zFace, coverage(glyph).mul(fade));
    }
    const color = mix(mix(grey, rgb(0x141a2a), zInk.mul(body.opacity.oneMinus().add(0.4).min(1))), rgb(0xe8efff).mul(1.5), zFace);
    return { color, opacity: max(body.opacity, zInk.mul(alpha)) };
  };
}

export function zombieShade(shade: Shade, most: number): FigureShade {
  return (a, b, p) => {
    const alpha = a.y, hurt = a.z, phase = a.w, lunge = b.x, health = b.z, night = b.w;
    const l = heading(p, a.x);
    const step = sin(phase.mul(Math.PI * 2)).mul(0.08);
    const feet = min(ellipse(l.sub(vec2(-0.1, step)), [0, 0], [0.065, 0.085]), ellipse(l.sub(vec2(0.1, step.negate())), [0, 0], [0.065, 0.085]));
    const torso = sdBox(l.sub(vec2(0, -0.03)), vec2(0.2, 0.11), 0.09);
    const out = lunge.mul(0.12).add(0.3);
    const arms = min(sdSegment(l, vec2(-0.17, 0), vec2(-0.14, out.add(step)), 0.055), sdSegment(l, vec2(0.17, 0), vec2(0.14, out.sub(step)), 0.055));
    const claws = min(sdCircle(l.sub(vec2(-0.14, out.add(step).add(0.02))), 0.05), sdCircle(l.sub(vec2(0.14, out.sub(step).add(0.02))), 0.05));
    const head = sdCircle(l.sub(vec2(0, 0.03)), 0.13);
    const scalp = max(head, sdCircle(l.sub(vec2(0.03, -0.03)), 0.1));
    const eyes = min(sdCircle(l.sub(vec2(-0.05, 0.125)), 0.022), sdCircle(l.sub(vec2(0.05, 0.125)), 0.022));
    const rags = rgb(0x35524a).mul(smoothstep(0.25, -0.2, p.x.sub(p.y)).mul(0.3).add(0.8));
    const flesh = rgb(0x7fb069);
    const hp = pips(p, health, most);
    const body = composite([[feet, rgb(0x2a2a2c)], [arms, flesh.mul(0.85)], [torso, rags], [claws, flesh], [head, flesh], [scalp, rgb(0x5f8a4f)], [eyes, rgb(0x2a0f0c)]], castShadow(p, [0.28, 0.25], 0.36), alpha);
    const lit = mix(shade(body.color), rgb(0xffffff), hurt.mul(0.75));
    const glare = coverage(eyes.add(0.006)).mul(night).mul(alpha);
    const showPips = select(health.lessThan(most - 0.5), float(1), float(0)).mul(alpha);
    const pipCover = coverage(hp.d).mul(showPips);
    const color = mix(mix(lit, rgb(0xff4b3a).mul(2.2), glare), mix(rgb(0x2b2b2b), rgb(0xff5a5a), hp.full), pipCover);
    return { color, opacity: max(body.opacity, pipCover) };
  };
}

export function skeletonShade(shade: Shade, most: number): FigureShade {
  return (a, b, p) => {
    const alpha = a.y, hurt = a.z, phase = a.w, draw = b.x, health = b.z;
    const l = heading(p, a.x);
    const step = sin(phase.mul(Math.PI * 2)).mul(0.07);
    const feet = min(ellipse(l.sub(vec2(-0.08, step)), [0, 0], [0.05, 0.075]), ellipse(l.sub(vec2(0.08, step.negate())), [0, 0], [0.05, 0.075]));
    const spine = sdSegment(l, vec2(0, -0.14), vec2(0, 0.06), 0.035);
    const ribs = min(sdSegment(l, vec2(-0.14, -0.02), vec2(0.14, -0.02), 0.03), sdSegment(l, vec2(-0.12, -0.09), vec2(0.12, -0.09), 0.026));
    const bowArm = sdSegment(l, vec2(-0.14, -0.02), vec2(-0.05, 0.28), 0.03);
    const pullArm = sdSegment(l, vec2(0.14, -0.02), vec2(0.04, draw.mul(-0.08).add(0.2)), 0.03);
    const arc = abs(sdCircle(l.sub(vec2(0, 0.1)), 0.22)).sub(0.022);
    const bow = max(arc, l.y.sub(0.15).negate().max(abs(l.x).sub(0.21)));
    const string = sdSegment(l, vec2(-0.2, 0.2), vec2(0.2, 0.2), 0.006);
    const pulled = min(sdSegment(l, vec2(-0.2, 0.2), vec2(0, draw.mul(-0.08).add(0.2)), 0.006), sdSegment(l, vec2(0.2, 0.2), vec2(0, draw.mul(-0.08).add(0.2)), 0.006));
    const skull = sdCircle(l.sub(vec2(0, 0.04)), 0.115);
    const sockets = min(sdCircle(l.sub(vec2(-0.045, 0.1)), 0.026), sdCircle(l.sub(vec2(0.045, 0.1)), 0.026));
    const bone = rgb(0xe9e3d0).mul(smoothstep(0.25, -0.2, p.x.sub(p.y)).mul(0.25).add(0.82));
    const hp = pips(p, health, most);
    const body = composite([[feet, bone.mul(0.8)], [select(draw.greaterThan(0.01), pulled, string), rgb(0xd8d2c2)], [bow, rgb(0x7a5230)], [ribs, bone], [spine, bone.mul(0.9)], [bowArm, bone], [pullArm, bone], [skull, bone], [sockets, rgb(0x1b1715)]], castShadow(p, [0.25, 0.22], 0.34), alpha);
    const lit = mix(shade(body.color), rgb(0xffffff), hurt.mul(0.75));
    const showPips = select(health.lessThan(most - 0.5), float(1), float(0)).mul(alpha);
    const pipCover = coverage(hp.d).mul(showPips);
    return { color: mix(lit, mix(rgb(0x2b2b2b), rgb(0xff5a5a), hp.full), pipCover), opacity: max(body.opacity, pipCover) };
  };
}

export function cowShade(shade: Shade, most: number): FigureShade {
  return (a, b, p) => {
    const alpha = a.y, hurt = a.z, phase = a.w, health = b.z, graze = b.x;
    const l = heading(p, a.x);
    const bob = sin(phase.mul(Math.PI * 2)).mul(0.02);
    const hooves = [[-0.14, 0.17], [0.14, 0.17], [-0.14, -0.2], [0.14, -0.2]].reduce<N<'float'>>((d, [x, y], k) => min(d, sdCircle(l.sub(vec2(x!, y! + (k % 2 ? 0.035 : -0.035) * 1)).sub(vec2(0, bob.mul(k < 2 ? 1 : -1))), 0.045)), float(9));
    const trunk = ellipse(l.sub(vec2(0, -0.03)), [0, 0], [0.19, 0.29]);
    const nod = graze.mul(0.05);
    const headAt = vec2(0, nod.add(0.3));
    const skull = ellipse(l.sub(headAt), [0, 0], [0.11, 0.1]);
    const muzzle = ellipse(l.sub(headAt).sub(vec2(0, 0.08)), [0, 0], [0.085, 0.055]);
    const horns = min(sdSegment(l.sub(headAt), vec2(-0.08, 0.02), vec2(-0.15, 0.06), 0.018), sdSegment(l.sub(headAt), vec2(0.08, 0.02), vec2(0.15, 0.06), 0.018));
    const ears = min(ellipse(l.sub(headAt), [-0.12, -0.02], [0.05, 0.03]), ellipse(l.sub(headAt), [0.12, -0.02], [0.05, 0.03]));
    const tail = sdSegment(l, vec2(0, -0.3), vec2(sin(phase.mul(6.28).add(1)).mul(0.04), -0.4), 0.014);
    const patch = smin(sdCircle(l.sub(vec2(0.07, 0.05)), 0.09), sdCircle(l.sub(vec2(-0.06, -0.15)), 0.08), 0.05);
    const hide = mix(rgb(0xf3efe6), rgb(0x2a2624), coverage(patch)).mul(smoothstep(0.25, -0.2, p.x.sub(p.y)).mul(0.25).add(0.82));
    const hp = pips(p, health, most);
    const body = composite([[hooves, rgb(0x2b2320)], [tail, rgb(0x2b2320)], [trunk, hide], [ears, rgb(0x3a302c)], [horns, rgb(0xe6dcc2)], [skull, rgb(0xf2ede4)], [muzzle, rgb(0xf0aaa2)]], castShadow(p, [0.25, 0.33], 0.34), alpha);
    const lit = mix(shade(body.color), rgb(0xffffff), hurt.mul(0.75));
    const showPips = select(health.lessThan(most - 0.5), float(1), float(0)).mul(alpha);
    const pipCover = coverage(hp.d).mul(showPips);
    return { color: mix(lit, mix(rgb(0x2b2b2b), rgb(0xff5a5a), hp.full), pipCover), opacity: max(body.opacity, pipCover) };
  };
}

export function arrowShade(shade: Shade): FigureShade {
  return (a, _b, p) => {
    const alpha = a.y, l = heading(p, a.x);
    const shaft = sdSegment(l, vec2(0, -0.26), vec2(0, 0.2), 0.024);
    const tip = sdBox(rotate(l.sub(vec2(0, 0.24)), Math.PI / 4), vec2(0.06), 0.008);
    const fletch = min(sdSegment(l, vec2(0, -0.2), vec2(-0.07, -0.3), 0.02), sdSegment(l, vec2(0, -0.2), vec2(0.07, -0.3), 0.02));
    const streak = smoothstep(0.025, 0, abs(l.x)).mul(smoothstep(-0.75, -0.3, l.y)).mul(smoothstep(-0.18, -0.3, l.y)).mul(0.4);
    const body = composite([[fletch, rgb(0xe8e2d4)], [shaft, rgb(0x8a6a45)], [tip, rgb(0xc9ced4)]], castShadow(p, [0.06, 0.2], 0.25), alpha);
    const weight = body.opacity.div(max(body.opacity.add(streak.mul(alpha)), 1e-4));
    return { color: mix(rgb(0xfff2d8), shade(body.color), weight), opacity: max(body.opacity, streak.mul(alpha)) };
  };
}
