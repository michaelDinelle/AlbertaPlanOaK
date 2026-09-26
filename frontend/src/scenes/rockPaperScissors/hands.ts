import * as THREE from 'three/webgpu';
import { abs, clamp, cos, dot, float, fract, max, min, mix, normalize, pow, sin, smoothstep, sqrt, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, noise, rgb } from '@kit/shapes2d';
import { flat, type Lit, type N, type Scalar, type Tint } from './art';

export type Pose = { readonly ext: readonly number[]; readonly angle: readonly number[] };

export const FIST: Pose = { ext: [0, 0, 0, 0, 0], angle: [-0.24, 0.06, 0, -0.06, -0.13] };
export const THROWS: readonly Pose[] = [
  FIST,
  { ext: [1, 1, 1, 1, 1], angle: [0.66, 0.13, 0.03, -0.08, -0.2] },
  { ext: [0, 1, 1, 0, 0], angle: [0.02, 0.27, -0.21, -0.06, -0.13] },
];

export interface HandPose {
  ext: Scalar[];
  angle: Scalar[];
  lift: Scalar;
  glow: Tint;
}

export const handPose = (): HandPose => ({
  ext: Array.from({ length: 5 }, () => uniform(0)),
  angle: FIST.angle.map((value) => uniform(value)),
  lift: uniform(0),
  glow: uniform(new THREE.Vector3()),
});

export function setPose(pose: HandPose, from: Pose, to: Pose, k: number): void {
  for (let f = 0; f < 5; f++) {
    pose.ext[f]!.value = from.ext[f]! + (to.ext[f]! - from.ext[f]!) * k;
    pose.angle[f]!.value = from.angle[f]! + (to.angle[f]! - from.angle[f]!) * k;
  }
}

export function easePose(pose: HandPose, to: Pose, rate: number): void {
  for (let f = 0; f < 5; f++) {
    pose.ext[f]!.value += (to.ext[f]! - pose.ext[f]!.value) * rate;
    pose.angle[f]!.value += (to.angle[f]! - pose.angle[f]!.value) * rate;
  }
}

const BASES = [[0.27, 0.4], [0.93, 0.33], [0.95, 0.11], [0.93, -0.11], [0.88, -0.31]] as const;
const LENGTHS = [[0.74, 0.84], [0.22, 0.9], [0.22, 1.0], [0.22, 0.93], [0.2, 0.74]] as const;
const RADII = [0.14, 0.118, 0.124, 0.116, 0.1] as const;
const ORDER = [4, 3, 2, 1, 0] as const;
const U0 = -14;
const U1 = 2.7;
const V = 1.55;

interface Part {
  d: N<'float'>;
  n: N<'vec3'>;
  along: N<'float'>;
  radial: N<'float'>;
}

function capsule(p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, r: number): Part {
  const pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(dot(ba, ba)), 0, 1), off = pa.sub(ba.mul(h)), l = off.length();
  const m = clamp(l.div(r), 0, 1);
  return { d: l.sub(r), n: vec3(off.div(max(l, 1e-5)).mul(m), sqrt(max(m.mul(m).oneMinus(), 0))), along: h.mul(ba.length()), radial: m };
}

function slab(p: N<'vec2'>, centre: readonly [number, number], half: readonly [number, number], round: number): Part {
  const c = p.sub(vec2(centre[0], centre[1])), inner = vec2(half[0] - round, half[1] - round);
  const k = c.sub(clamp(c, inner.negate(), inner)), l = k.length(), m = clamp(l.div(round), 0, 1);
  return { d: l.sub(round), n: vec3(k.div(max(l, 1e-5)).mul(m), sqrt(max(m.mul(m).oneMinus(), 0))), along: c.x, radial: m };
}

function anatomy(p: N<'vec2'>, pose: HandPose): { palm: Part; fingers: Part[]; hand: N<'float'> } {
  const pv = vec2(p.x, p.y.mul(smoothstep(0.95, 0.05, p.x).mul(0.15).add(1)));
  const palm = slab(pv, [0.5, 0], [0.52, 0.47], 0.27);
  const fingers = BASES.map(([bx, by], f) => {
    const base = vec2(bx, by), angle = pose.angle[f]!, reach = mix(float(LENGTHS[f]![0]), float(LENGTHS[f]![1]), pose.ext[f]!);
    return capsule(p, base, base.add(vec2(cos(angle), sin(angle)).mul(reach)), RADII[f]!);
  });
  let hand = palm.d;
  for (const finger of fingers) hand = min(hand, finger.d);
  return { palm, fingers, hand };
}

export interface HandLook {
  kind: 'glove' | 'automaton';
  accent: number;
}

export function hand(lit: Lit, pose: HandPose, mirror: 1 | -1, look: HandLook): THREE.Mesh {
  const p = vec2(mix(float(U0), float(U1), uv().x), mix(float(-V), float(V), uv().y));
  const key = normalize(vec3(-0.45 * mirror, 0.55, 0.72)), halfway = normalize(key.add(vec3(0, 0, 1)));
  const diffuse = (n: N<'vec3'>) => max(dot(n, key), 0);
  const gloss = (n: N<'vec3'>, power: number) => pow(max(dot(n, halfway), 0), float(power));
  const { palm, fingers, hand: handD } = anatomy(p, pose);
  const glove = look.kind === 'glove';

  const armHalf = (x: N<'float'>): N<'float'> => (glove ? smoothstep(-0.3, -3.5, x).mul(0.07).add(0.52) : float(0.34));
  const armOf = (q: N<'vec2'>): N<'float'> => max(abs(q.y).sub(armHalf(q.x)), q.x.add(0.3));
  const armD = armOf(p);
  const armRound = clamp(abs(p.y).div(armHalf(p.x)), 0, 1), armN = vec3(0, p.y.sign().mul(armRound), sqrt(max(armRound.mul(armRound).oneMinus(), 0)));

  let color: N<'vec3'>;
  let silhouette: N<'float'>;
  let aura: N<'float'>;
  const lift = pose.lift;

  if (glove) {
    const pink = rgb(look.accent), deep = rgb(look.accent).mul(0.62), crease = rgb(0x5a2046);
    const knit = noise(p.mul(vec2(60, 60))).mul(0.035).add(1);
    const cloth = (n: N<'vec3'>) => pink.mul(diffuse(n).mul(0.55).add(0.52)).mul(knit).add(vec3(gloss(n, 18).mul(0.08)));
    const folds = sin(p.x.mul(2.3).add(sin(p.x.mul(0.6)).mul(2))).mul(0.06).add(1);
    const sleeve = rgb(0x70406f).mul(diffuse(armN).mul(0.6).add(0.45)).mul(folds).mul(noise(p.mul(vec2(3, 24))).mul(0.05).add(1));
    color = sleeve;
    const cuff = slab(p, [-0.2, 0], [0.24, 0.54], 0.1);
    const ribs = abs(fract(p.y.mul(13)).sub(0.5)).mul(2);
    const cuffColor = deep.mul(diffuse(cuff.n).mul(0.5).add(0.55)).mul(ribs.mul(0.22).add(0.82));
    color = mix(color, rgb(0x1d0f1f), coverage(abs(armD).sub(0.012)).mul(0.8));
    color = mix(color, cloth(palm.n), coverage(palm.d));
    const stitch = (y0: number, y1: number) => {
      const a = vec2(0.14, y0), b = vec2(0.74, y1), pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(dot(ba, ba)), 0, 1);
      return coverage(pa.sub(ba.mul(h)).length().sub(0.009)).mul(smoothstep(0.45, 0.55, fract(h.mul(9))));
    };
    color = mix(color, pink.mul(1.25), max(max(stitch(0.19, 0.22), stitch(0, 0.01)), stitch(-0.19, -0.2)).mul(coverage(palm.d.add(0.05))).mul(0.55));
    for (const f of ORDER) {
      const finger = fingers[f]!, inPalm = smoothstep(-0.1, 0.02, palm.d).mul(0.65).add(0.35);
      color = mix(color, cloth(finger.n), coverage(finger.d));
      color = mix(color, crease, coverage(abs(finger.d).sub(0.011)).mul(inPalm).mul(0.75));
    }
    color = mix(color, cuffColor, coverage(cuff.d));
    color = mix(color, crease, coverage(abs(cuff.d).sub(0.011)).mul(0.8));
    silhouette = min(min(handD, armD), cuff.d);
    aura = min(handD, cuff.d);
  } else {
    const steel = (n: N<'vec3'>, tone: number) => rgb(0x7d8898).mul(tone).mul(diffuse(n).mul(0.6).add(0.45)).add(rgb(0xdfe9f7).mul(gloss(n, 34).mul(0.55)));
    const brass = (n: N<'vec3'>) => rgb(0xc9a45c).mul(diffuse(n).mul(0.6).add(0.45)).add(rgb(0xfff0c8).mul(gloss(n, 24).mul(0.45)));
    const seam = rgb(0x151a22);
    color = steel(armN, 0.8).mul(smoothstep(0.3, 1, abs(sin(p.x.mul(0.8)))).mul(0.06).add(0.97));
    const band = (u: number, w: number) => slab(p, [u, 0], [w, 0.4], 0.06);
    for (const u of [-0.95, -2.9, -5.2]) {
      const b = band(u, 0.1);
      color = mix(color, brass(b.n), coverage(b.d));
      color = mix(color, seam, coverage(abs(b.d).sub(0.009)).mul(0.8));
    }
    const cylinder = slab(p, [-2.6, 0.33], [0.85, 0.12], 0.1), rod = slab(p, [-1.2, 0.33], [0.62, 0.04], 0.038);
    color = mix(color, steel(rod.n, 1.2), coverage(rod.d));
    color = mix(color, seam, coverage(abs(rod.d).sub(0.007)));
    color = mix(color, steel(cylinder.n, 0.62), coverage(cylinder.d));
    color = mix(color, seam, coverage(abs(cylinder.d).sub(0.009)));
    color = mix(color, seam, coverage(abs(armD).sub(0.012)).mul(0.9));
    const collar = slab(p, [-0.2, 0], [0.16, 0.44], 0.08);
    color = mix(color, brass(collar.n), coverage(collar.d));
    color = mix(color, seam, coverage(abs(collar.d).sub(0.009)));
    const plate = steel(palm.n, 1).mul(smoothstep(0.2, 0.27, palm.radial).mul(0.18).add(0.9));
    color = mix(color, plate, coverage(palm.d));
    color = mix(color, seam, coverage(abs(palm.d.add(0.075)).sub(0.006)).mul(0.55));
    for (const [rx, ry] of [[0.2, 0.28], [0.2, -0.28], [0.74, 0.28], [0.74, -0.28]] as const) {
      const rivet = p.sub(vec2(rx, ry)).length().sub(0.042);
      color = mix(color, rgb(0xd9b56a).mul(smoothstep(0.042, -0.03, p.sub(vec2(rx - 0.012, ry + 0.012)).length().sub(0.01)).mul(0.5).add(0.6)), coverage(rivet));
    }
    for (const f of ORDER) {
      const finger = fingers[f]!, reach = f === 0 ? 0.34 : 0.3;
      const joint = min(abs(finger.along.sub(reach)), abs(finger.along.sub(reach + 0.28)));
      let part: N<'vec3'> = steel(finger.n, 1.05);
      part = mix(part, brass(finger.n), coverage(joint.sub(0.04)));
      part = mix(part, seam, coverage(joint.sub(0.01)).mul(0.8));
      color = mix(color, part, coverage(finger.d));
      color = mix(color, seam, coverage(abs(finger.d).sub(0.011)).mul(0.9));
    }
    for (const [f, [bx, by]] of BASES.entries()) {
      if (f === 0) continue;
      const knuckle = p.sub(vec2(bx, by)).length().sub(0.07);
      color = mix(color, brass(vec3(p.sub(vec2(bx, by)).div(0.07).mul(0.6), 0.8)), coverage(knuckle).mul(0.9));
    }
    silhouette = min(min(handD, armD), min(cylinder.d, collar.d));
    aura = min(handD, collar.d);
  }

  color = mix(color, rgb(0x0a0508), coverage(abs(silhouette).sub(0.013)).mul(0.85));
  const offset = vec2(lift.mul(0.12).add(0.08).mul(mirror), lift.mul(0.14).add(0.12).negate());
  const shadowShape = min(anatomy(p.sub(offset), pose).hand, armOf(p.sub(offset)));
  const shade = smoothstep(lift.mul(0.1).add(0.07), lift.mul(-0.05).sub(0.02), shadowShape).mul(lift.mul(-0.2).add(0.55));
  const body = coverage(silhouette), halo = smoothstep(0.32, 0, aura).mul(smoothstep(-0.05, 0.02, silhouette)).mul(dot(pose.glow, vec3(0.3, 0.5, 0.2)).min(1));
  const lamp = pose.glow.mul(coverage(abs(aura.add(0.02)).sub(0.018)).mul(0.9).add(smoothstep(-0.25, 0, aura).mul(coverage(aura)).mul(0.12)));
  const front = lit(color).add(lamp), under = body.oneMinus();
  const alpha = body.add(under.mul(halo.add(halo.oneMinus().mul(shade))));
  const geometry = flat(U1 - U0, 2 * V).translate((U0 + U1) / 2, 0, 0);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });
  material.colorNode = front.mul(body).add(pose.glow.mul(under.mul(halo))).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.scale.x = mirror;
  return mesh;
}
