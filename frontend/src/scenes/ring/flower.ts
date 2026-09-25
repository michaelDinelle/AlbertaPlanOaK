import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, mix, mod, positionWorld, pow, select, sin, smoothstep, sqrt, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { smoothstep01 } from '@kit/mix';
import { coverage, fractal, grain, noise, rgb } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = THREE.UniformNode<'float', number>;

export const TAU = 2 * Math.PI;
const DISC = 2.3;
export const BASE = 2.12;
export const TIP = 5.15;
export const WALK = 3.85;
const HALF = 0.215;
export const REACH = 5.45;
const SHOULDER = BASE + 1.1;
const NOTCH = 0.085;
const LEAVES = 12;
const LEAF_FROM = 3.4;
const LEAF_TO = 8.9;
const LEAF_HALF = 1.05;

export const HUES = [0xffbf47, 0xa585ff] as const;
export const NAMES = ['gold', 'violet'] as const;
const PETAL = [0xf2bd5a, 0xae96f2] as const;
const DEEP = [0x9a5716, 0x4b33a0] as const;
const TIPS = [0xffe7ad, 0xddd2ff] as const;
const FROST = 0x8aa4d6;
const SOIL = [0x151c18, 0x262a1f] as const;
const MOSS = 0x2a4a2e;
const LEAF = [0x1c4230, 0x386d46] as const;
const RIB = 0x5f8a58;
const DISC_INNER = 0x2a160c;
const DISC_OUTER = 0x4a2512;
const BUD = 0x4a5a2a;
const FLORET = 0x7a3818;
const OPEN = 0xc9782a;
const POLLEN = 0xffd76a;

export const flat = (width: number, depth: number): THREE.BufferGeometry => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const onRing = (position: number, cells: number, radius: number, out: THREE.Vector3): THREE.Vector3 => {
  const angle = (TAU * (position + 0.5)) / cells;
  return out.set(radius * Math.sin(angle), 0, -radius * Math.cos(angle));
};

const petalWidth = (s: number): number => HALF * (0.58 + 0.42 * smoothstep01((s - BASE) / (SHOULDER - BASE)));

export const reachOf = (cell: number): number => TIP - 0.14 * ((((Math.sin(cell * 78.233) * 43758.5453) % 1) + 1) % 1);

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const pixel = (): N<'float'> => fwidth(positionWorld.x).max(1e-5);
const cover = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const hueOf = (cell: N<'float'>, arc: number): N<'float'> => mod(floor(cell.div(arc)), 2);
const pick = (hue: N<'float'>, pair: readonly [number, number]): N<'vec3'> => mix(rgb(pair[0]), rgb(pair[1]), hue);
const sectorOf = (q: N<'vec2'>, count: number, offset = 0): N<'float'> => floor(fract(atan(q.x, q.y).div(TAU).sub(offset)).mul(count));

interface Layer { color: N<'vec3'>; alpha: N<'float'> }
const over = (under: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(under.color.mul(alpha.oneMinus())),
  alpha: alpha.add(under.alpha.mul(alpha.oneMinus())),
});

function sheet(): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.blending = THREE.CustomBlending;
  material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
  material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  return material;
}

export interface PetalState { verdicts: THREE.DataTexture; phase: Level; lights: Level; ambient: THREE.Color }

interface Verdict { level: N<'float'>; flash: N<'float'>; frost: N<'float'>; tip: N<'float'> }

function verdictOf(state: PetalState, cell: N<'float'>, cells: number): Verdict {
  const data = texture(state.verdicts).load(uvec2(vec2(cell, 0))), code = data.r, age = data.g.add(state.phase);
  return {
    tip: data.b,
    level: select(code.equal(1), float(1).sub(age.div(cells).mul(0.5)), float(0)),
    flash: select(code.equal(1), exp(age.div(-0.6)), float(0)),
    frost: select(code.equal(2), float(1).sub(age.div(cells).mul(0.45)), float(0)),
  };
}

interface Petal { d: N<'float'>; color: N<'vec3'>; lower: N<'float'>; upper: N<'float'> }

function petalAt(q: N<'vec2'>, cell: N<'float'>, cells: number, arc: number, state: PetalState): Petal {
  const angle = cell.add(0.5).mul(TAU / cells), dir = vec2(sin(angle), cos(angle)), side = vec2(cos(angle), sin(angle).negate());
  const s = dot(q, dir), a = dot(q, side), hue = hueOf(cell, arc), parity = mod(cell, 2);
  const { level, flash, frost, tip } = verdictOf(state, cell, cells);
  const width = mix(float(HALF * 0.58), float(HALF), smoothstep(BASE, SHOULDER, s)), cap = tip.sub(HALF);
  const plain = select(s.greaterThan(cap), length(vec2(s.sub(cap), a)).sub(HALF), max(abs(a).sub(width), float(BASE).sub(s)));
  const d = select(hue.greaterThan(0.5), max(plain, float(NOTCH).sub(length(vec2(s.sub(tip.add(0.03)), a)))), plain);
  const t = clamp(s.sub(BASE).div(tip.sub(BASE)), 0, 1), across = a.div(width);
  const vein = abs(abs(across).sub(0.5)).mul(width);
  const streak = noise(vec2(s.mul(2.3), across.mul(9).add(cell.mul(3.7)))), tone = grain(vec2(cell, 7.3)).x.mul(0.1).add(0.95);
  let albedo: N<'vec3'> = mix(pick(hue, DEEP), pick(hue, PETAL), smoothstep(0.02, 0.42, t)).mul(tone);
  albedo = mix(albedo, pick(hue, TIPS), smoothstep(0.7, 1, t).mul(0.4));
  albedo = albedo.mul(streak.mul(0.07).add(1)).mul(across.mul(0.07).add(1));
  albedo = albedo.mul(sin(across.mul(Math.PI * 6)).mul(smoothstep(1, 0.35, t)).mul(0.045).add(1));
  albedo = albedo.mul(smoothstep(0.016, 0.004, vein).mul(smoothstep(0.95, 0.5, t)).mul(0.09).oneMinus());
  albedo = albedo.mul(smoothstep(0.016, 0.003, abs(a)).mul(smoothstep(0.97, 0.2, t)).mul(0.18).oneMinus());
  albedo = albedo.mul(smoothstep(-0.035, 0, d).mul(0.14).oneMinus());
  albedo = albedo.mul(smoothstep(DISC + 0.28, DISC - 0.02, length(q)).mul(0.55).oneMinus());
  const core = smoothstep(1.05, 0.15, abs(across)).mul(0.25).add(0.75);
  const rim = smoothstep(-0.06, -0.008, d).mul(0.3);
  const glow = albedo.mul(pick(hue, HUES)).mul(level.mul(1.05).add(flash.mul(0.6))).mul(core.add(rim));
  const near = mod(cell.add(select(a.greaterThan(0), float(1), float(cells - 1))), cells), beside = verdictOf(state, near, cells);
  const spill = pick(hueOf(near, arc), HUES).mul(beside.level.add(beside.flash.mul(0.5))).mul(smoothstep(0.3, 1, abs(across))).mul(level.oneMinus()).mul(0.22);
  const dim = mix(float(0.75), float(0.5), frost);
  const shaded = albedo.mul(rgb(state.ambient).mul(mix(dim, float(0.3), level)).add(spill)).add(glow).add(rgb(FROST).mul(frost.mul(smoothstep(-0.05, -0.005, d).mul(0.16).add(0.035))));
  return { d, color: mix(albedo, shaded, state.lights), lower: parity.oneMinus(), upper: parity };
}

export function petalMaterial(cells: number, arc: number, state: PetalState): THREE.MeshBasicNodeMaterial {
  const material = sheet(), q = chart(), px = pixel(), base = sectorOf(q, cells);
  const petals = [-1, 0, 1].map((k) => petalAt(q, mod(base.add(k + cells), cells), cells, arc, state));
  let layer: Layer = { color: vec3(0), alpha: float(0) };
  for (const petal of petals) layer = over(layer, petal.color, cover(petal.d, px).mul(petal.lower));
  let shade: N<'float'> = float(1);
  for (const petal of petals) shade = shade.mul(smoothstep(-0.005, 0.06, petal.d).oneMinus().mul(petal.upper).mul(0.38).oneMinus());
  layer = { color: layer.color.mul(shade), alpha: layer.alpha };
  for (const petal of petals) layer = over(layer, petal.color, cover(petal.d, px).mul(petal.upper));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return material;
}

export function petalLight(cells: number, arc: number, state: PetalState, strength: number): N<'vec3'> {
  const cell = attribute<'float'>('cell', 'float'), { level, flash } = verdictOf(state, cell, cells);
  return pick(hueOf(cell, arc), HUES).mul(level.add(flash.mul(0.5))).mul(strength);
}

export function petalEmitters(cells: number): THREE.BufferGeometry {
  const rows = 24, from = DISC + 0.03, position: number[] = [], cell: number[] = [], index: number[] = [];
  for (const c of Array.from({ length: cells }, (_, i) => (i < cells / 2 ? 2 * i : 2 * (i - cells / 2) + 1))) {
    const angle = (TAU * (c + 0.5)) / cells, dx = Math.sin(angle), dy = Math.cos(angle), sx = Math.cos(angle), sy = -Math.sin(angle), first = position.length / 3, tip = reachOf(c), to = tip + 0.02;
    for (let i = 0; i <= rows; i++) {
      const s = from + ((to - from) * i) / rows, end = Math.min(HALF, Math.max(0, s - (tip - HALF))), w = 0.03 + Math.sqrt(Math.max(0, petalWidth(s) ** 2 - end * end));
      for (const side of [-1, 1]) {
        position.push(s * dx + side * w * sx, 0, -(s * dy + side * w * sy));
        cell.push(c);
      }
      if (i > 0) {
        const a = first + (i - 1) * 2;
        index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  geometry.setAttribute('cell', new THREE.Float32BufferAttribute(cell, 1));
  geometry.setIndex(index);
  return geometry;
}

export function discMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const q = chart(), r = length(q);
  let albedo: N<'vec3'> = mix(rgb(DISC_INNER), rgb(DISC_OUTER), smoothstep(0, DISC, r));
  albedo = albedo.mul(fractal(q.mul(9), 3).mul(0.14).add(1));
  albedo = albedo.mul(smoothstep(DISC - 0.1, DISC, r).mul(0.35).oneMinus());
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo);
  material.opacityNode = cover(r.sub(DISC), pixel());
  return material;
}

export function discGeometry(): THREE.BufferGeometry {
  return new THREE.CircleGeometry(DISC + 0.05, 160).rotateX(-Math.PI / 2);
}

function floretMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const info = attribute<'vec2'>('aFloret', 'vec2'), t = info.x, seed = info.y;
  const p = uv().sub(0.5).mul(2), r = length(p);
  const dome = sqrt(clamp(float(1).sub(r.div(0.86).pow(2)), 0, 1));
  const star = cos(atan(p.y, p.x).mul(5)).mul(0.5).add(0.5), open = smoothstep(0.62, 0.92, t);
  let albedo: N<'vec3'> = mix(mix(rgb(BUD), rgb(FLORET), smoothstep(0.1, 0.4, t)), rgb(OPEN), open);
  albedo = albedo.mul(dome.mul(0.65).add(0.35)).mul(seed.mul(0.3).add(0.85));
  albedo = mix(albedo, rgb(POLLEN), open.mul(smoothstep(0.4, 0.1, r)).mul(star.mul(0.6).add(0.4)).mul(0.6));
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo);
  material.opacityNode = coverage(r.sub(0.86));
  return material;
}

export function florets(lit: Lit, count: number): THREE.InstancedMesh {
  const golden = Math.PI * (3 - Math.sqrt(5)), spacing = (DISC - 0.06) / Math.sqrt(count), geometry = flat(1, 1), data = new Float32Array(count * 2);
  for (let n = 0; n < count; n++) {
    data[n * 2] = (n + 0.5) / count;
    data[n * 2 + 1] = ((Math.sin(n * 12.9898) * 43758.5453) % 1 + 1) % 1;
  }
  geometry.setAttribute('aFloret', new THREE.InstancedBufferAttribute(data, 2));
  const mesh = instancedMesh(geometry, floretMaterial(lit), count), matrix = new THREE.Matrix4(), at = new THREE.Vector3(), turn = new THREE.Quaternion(), size = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  for (let n = 0; n < count; n++) {
    const r = spacing * Math.sqrt(n + 0.5), a = n * golden;
    at.set(r * Math.sin(a), 0, -r * Math.cos(a));
    turn.setFromAxisAngle(up, -a);
    size.set(spacing * 2.25, 1, spacing * 2.25);
    mesh.setMatrixAt(n, matrix.compose(at, turn, size));
  }
  return mesh;
}

interface Leaf { d: N<'float'>; color: N<'vec3'>; lower: N<'float'>; upper: N<'float'> }

function leafAt(q: N<'vec2'>, index: N<'float'>, lit: Lit): Leaf {
  const angle = index.add(0.5).mul(TAU / LEAVES).add(0.13), dir = vec2(sin(angle), cos(angle)), side = vec2(cos(angle), sin(angle).negate());
  const s = dot(q, dir), bend = grain(vec2(index, 3.1)).x.sub(0.5).mul(0.1), reach = s.sub(LEAF_FROM);
  const a = dot(q, side).sub(bend.mul(reach.mul(reach))), t = clamp(reach.div(LEAF_TO - LEAF_FROM), 0, 1);
  const width = pow(max(sin(t.mul(Math.PI)), 0), 0.72).mul(LEAF_HALF), rel = abs(a).div(width.max(1e-3));
  const teeth = abs(fract(s.mul(5.5)).sub(0.5)).mul(0.05);
  const d = max(abs(a).sub(width).add(teeth), max(float(LEAF_FROM).sub(s), s.sub(LEAF_TO)));
  const rib = smoothstep(t.oneMinus().mul(0.022).add(0.014), t.oneMinus().mul(0.022).add(0.004), abs(a));
  const chevron = s.sub(abs(a).mul(0.85)), vein = smoothstep(0.012, 0.003, abs(fract(chevron.mul(2.2)).sub(0.5)).div(2.2)).mul(smoothstep(0.04, 0.2, rel));
  let albedo: N<'vec3'> = mix(rgb(LEAF[0]), rgb(LEAF[1]), smoothstep(-0.5, 0.6, fractal(q.mul(0.9).add(index.mul(4.7)), 3)));
  albedo = albedo.mul(smoothstep(0.35, 1, rel).mul(-0.3).add(1.05)).mul(vein.mul(0.2).oneMinus());
  albedo = mix(albedo, rgb(RIB), rib.mul(0.55));
  const parity = mod(index, 2);
  return { d, color: lit(albedo), lower: parity.oneMinus(), upper: parity };
}

export function groundMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const q = chart(), px = pixel(), base = sectorOf(q, LEAVES, 0.13 / TAU);
  let soil: N<'vec3'> = mix(rgb(SOIL[0]), rgb(SOIL[1]), smoothstep(-0.4, 0.5, fractal(q.mul(0.7), 4)));
  soil = soil.mul(grain(floor(q.mul(40))).x.mul(0.2).add(0.9));
  soil = mix(soil, rgb(MOSS), smoothstep(0.08, 0.42, fractal(q.mul(1.3).add(5.3), 3)).mul(0.7));
  let layer: Layer = { color: lit(soil), alpha: float(1) };
  const leaves = [-1, 0, 1].map((k) => leafAt(q, mod(base.add(k + LEAVES), LEAVES), lit));
  for (const leaf of leaves) layer = over(layer, leaf.color, cover(leaf.d, px).mul(leaf.lower));
  let shade: N<'float'> = float(1);
  for (const leaf of leaves) shade = shade.mul(smoothstep(-0.02, 0.22, leaf.d).oneMinus().mul(leaf.upper).mul(0.45).oneMinus());
  layer = { color: layer.color.mul(shade), alpha: layer.alpha };
  for (const leaf of leaves) layer = over(layer, leaf.color, cover(leaf.d, px).mul(leaf.upper));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = layer.color;
  return material;
}
