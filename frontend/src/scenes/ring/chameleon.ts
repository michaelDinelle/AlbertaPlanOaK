import * as THREE from 'three/webgpu';
import { abs, clamp, dot, float, floor, fract, length, max, min, mix, select, sin, smoothstep, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, grain, rgb } from '@kit/shapes2d';
import { TAU } from './flower';

type N<T extends string> = THREE.Node<T>;
type Level = THREE.UniformNode<'float', number>;
type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
type Gaze = THREE.UniformNode<'vec2', THREE.Vector2>;

export const SPRITE = new THREE.Vector2(1.56, 0.84);

const INK = 0x120d09;
const IRIS = 0x0b0907;
const HALO = 0x9fb8e8;

export interface Chameleon {
  light: Tint;
  lights: Level;
  from: Tint;
  to: Tint;
  front: Level;
  swingA: Level;
  swingB: Level;
  startle: Level;
  blend: Level;
  breath: Level;
  outer: Gaze;
  inner: Gaze;
}

export const chameleon = (): Chameleon => ({
  light: uniform(new THREE.Vector3()),
  lights: uniform(1),
  from: uniform(new THREE.Vector3(1, 1, 1)),
  to: uniform(new THREE.Vector3(1, 1, 1)),
  front: uniform(0),
  swingA: uniform(0),
  swingB: uniform(0),
  startle: uniform(0),
  blend: uniform(0),
  breath: uniform(1),
  outer: uniform(new THREE.Vector2(1, 0)),
  inner: uniform(new THREE.Vector2(0, -1)),
});

const TAIL: readonly (readonly [number, number, number])[] = (() => {
  const points: [number, number, number][] = [[-0.25, 0, 0.054], [-0.36, 0, 0.044], [-0.47, -0.01, 0.035]];
  const centre = [-0.56, -0.1] as const, steps = 24, turns = 1.75;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps, a = Math.PI / 2 + t * turns * TAU, radius = 0.095 * (1 - 0.8 * t);
    points.push([centre[0] + radius * Math.cos(a), centre[1] + radius * Math.sin(a), 0.031 * (1 - 0.68 * t)]);
  }
  return points;
})();

const ellipse = (p: N<'vec2'>, x: number, y: number, rx: N<'float'> | number, ry: N<'float'> | number): N<'float'> => {
  const q = p.sub(vec2(x, y)), r = vec2(rx, ry), k0 = length(q.div(r)), k1 = length(q.div(r.mul(r)));
  return k0.sub(1).mul(select(k1.greaterThan(1e-5), k0.div(k1), min(r.x, r.y)));
};

const smin = (a: N<'float'>, b: N<'float'>, k: number): N<'float'> => {
  const h = max(float(k).sub(abs(a.sub(b))), 0).div(k);
  return min(a, b).sub(h.mul(h).mul(k * 0.25));
};

const limb = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, ra: number, rb: number): N<'float'> => {
  const pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(dot(ba, ba)), 0, 1);
  return length(pa.sub(ba.mul(h))).sub(mix(float(ra), float(rb), h));
};

type Joint = readonly [number, number];

function leg(p: N<'vec2'>, root: Joint, bend: Joint, foot: Joint, swing: N<'float'>, side: number): N<'float'> {
  const a = vec2(root[0], root[1] * side), b = vec2(swing.mul(0.35).add(bend[0]), bend[1] * side), c = vec2(swing.add(foot[0]), foot[1] * side);
  const upper = limb(p, a, b, 0.028, 0.021), lower = limb(p, b, c, 0.021, 0.016);
  const ahead = limb(p, c, c.add(vec2(0.046, 0.016 * side)), 0.014, 0.01), behind = limb(p, c, c.add(vec2(-0.04, 0.026 * side)), 0.014, 0.01);
  return min(min(upper, lower), min(ahead, behind));
}

const ARM: readonly [Joint, Joint, Joint] = [[0.14, 0.06], [0.085, 0.2], [0.2, 0.235]];
const THIGH: readonly [Joint, Joint, Joint] = [[-0.18, 0.055], [-0.115, 0.2], [-0.25, 0.235]];

export function chameleonMaterial(eye: number, ambient: THREE.Color, state: Chameleon): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(vec2(SPRITE.x, SPRITE.y)), breath = state.breath;
  const torso = ellipse(p, -0.025, 0, 0.26, breath.mul(0.108));
  const neck = ellipse(p, 0.18, 0, 0.1, 0.075);
  const skull = ellipse(p, 0.31, 0, 0.125, 0.088);
  const snout = ellipse(p, 0.425, 0, 0.078, 0.05);
  const casque = ellipse(p, 0.23, 0, 0.125, 0.046);
  let tail: N<'float'> = float(9);
  for (let i = 0; i + 1 < TAIL.length; i++) {
    const [ax, ay, ar] = TAIL[i]!, [bx, by, br] = TAIL[i + 1]!;
    tail = min(tail, limb(p, vec2(ax, ay), vec2(bx, by), ar, br));
  }
  const legs = min(
    min(leg(p, ...ARM, state.swingA, 1), leg(p, ...ARM, state.swingB, -1)),
    min(leg(p, ...THIGH, state.swingB, 1), leg(p, ...THIGH, state.swingA, -1)),
  );
  const head = smin(smin(skull, snout, 0.05), casque, 0.03);
  const trunk = smin(smin(torso, neck, 0.07), head, 0.05);
  const eyeSize = state.startle.mul(0.12).add(1).mul(0.06);
  const outerEye = p.sub(vec2(0.34, 0.088)), innerEye = p.sub(vec2(0.34, -0.088));
  const eyes = min(length(outerEye).sub(eyeSize), length(innerEye).sub(eyeSize));
  const body = smin(trunk, smin(tail, legs, 0.025), 0.035);
  const silhouette = min(body, eyes);

  const sigma = float(0.5).sub(p.x);
  const skin: N<'vec3'> = mix(state.from, state.to, smoothstep(sigma.sub(0.1), sigma.add(0.1), state.front));
  const depth = smoothstep(0, 0.075, silhouette.negate());
  const flank = smoothstep(0.014, 0.006, abs(abs(p.y).sub(0.058))).mul(smoothstep(0.01, -0.02, torso)).mul(smoothstep(-0.25, -0.15, p.x)).mul(smoothstep(0.16, 0.06, p.x));
  const blotch = min(min(ellipse(vec2(p.x, abs(p.y)), -0.14, 0.066, 0.034, 0.02), ellipse(vec2(p.x, abs(p.y)), -0.03, 0.07, 0.038, 0.021)), ellipse(vec2(p.x, abs(p.y)), 0.08, 0.064, 0.03, 0.018));
  const spots = smoothstep(0.004, -0.004, blotch).mul(0.3);
  const bands = smoothstep(0.25, 0.75, sin(p.x.mul(25).add(1.2)).mul(0.5).add(0.5)).mul(smoothstep(0.01, -0.02, min(torso, tail)));
  const scales = grain(floor(p.mul(110))).x;
  const crest = smoothstep(0.004, -0.002, abs(p.y).sub(abs(fract(p.x.mul(22)).sub(0.5)).mul(0.014).add(0.005))).mul(smoothstep(-0.27, -0.2, p.x)).mul(smoothstep(0.34, 0.26, p.x));
  const ridge = smoothstep(0.02, -0.03, casque).mul(0.2).sub(smoothstep(0.012, 0.004, abs(p.y)).mul(smoothstep(0.13, 0.17, p.x)).mul(smoothstep(0.43, 0.36, p.x)).mul(0.3));
  let albedo: N<'vec3'> = skin.mul(depth.mul(0.45).add(0.62)).mul(scales.mul(0.14).add(0.92));
  albedo = albedo.mul(flank.mul(0.32).add(1)).mul(bands.mul(0.22).oneMinus()).mul(ridge.add(1)).mul(spots.add(1));
  albedo = mix(albedo, albedo.mul(0.45), crest);

  const eyeBall = (local: N<'vec2'>, gaze: Gaze) => {
    const r = length(local), rings = smoothstep(0.0035, 0.001, abs(fract(r.div(0.0125)).sub(0.5)).mul(0.0125)).mul(smoothstep(0.016, 0.03, r));
    const offset = local.sub(gaze.mul(0.024)), pupil = length(offset), size = state.startle.mul(0.01).add(0.011);
    const glint = coverage(length(offset.sub(vec2(0.006, 0.007))).sub(0.0045));
    let color: N<'vec3'> = skin.mul(smoothstep(0.056, 0.01, r).mul(0.42).add(0.66)).mul(rings.mul(0.22).oneMinus());
    color = mix(color, rgb(eye), smoothstep(0.005, 0.002, abs(pupil.sub(size.add(0.008)))).mul(0.9));
    color = mix(color, rgb(IRIS), coverage(pupil.sub(size)));
    return { color: mix(color, vec3(1), glint.mul(0.9)), cover: coverage(r.sub(eyeSize)) };
  };
  const left = eyeBall(outerEye, state.outer), right = eyeBall(innerEye, state.inner);
  const shine = smoothstep(0.05, 0, length(p.sub(vec2(0.29, 0.035)))).mul(0.2);
  const self = (color: N<'vec3'>) => mix(color, color.mul(rgb(ambient).add(state.light).mul(0.75).add(0.42)), state.lights);
  let color: N<'vec3'> = self(albedo).add(vec3(shine));
  color = mix(color, self(left.color), left.cover);
  color = mix(color, self(right.color), right.cover);
  color = mix(color, rgb(INK), smoothstep(-0.02, -0.006, silhouette).mul(state.blend.mul(-0.25).add(0.9)));

  const shadow = smoothstep(0.075, -0.01, silhouette).mul(0.62);
  const halo = smoothstep(0.045, 0.004, silhouette).mul(state.startle).mul(0.6);
  let alpha: N<'float'> = shadow;
  let premultiplied: N<'vec3'> = rgb(HALO).mul(halo);
  alpha = halo.add(alpha.mul(halo.oneMinus()));
  const cover = coverage(silhouette);
  premultiplied = color.mul(cover).add(premultiplied.mul(cover.oneMinus()));
  alpha = cover.add(alpha.mul(cover.oneMinus()));

  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.blending = THREE.CustomBlending;
  material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
  material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  material.colorNode = premultiplied;
  material.opacityNode = alpha;
  return material;
}
