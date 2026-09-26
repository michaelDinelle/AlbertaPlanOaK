import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, normalize, positionWorld, select, sign, sin, smoothstep, sqrt, step, uniform, uv, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { LIGHT_LAYER, type Light2D } from '@kit/light2d';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { SPAWN, type SkirmishShape } from '@sim/envs/skirmish';

export type N<T extends string> = THREE.Node<T>;
export type Level = THREE.UniformNode<'float', number>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Scalar = N<'float'> | number;

export const SPRITE = 3.2;
export const RING_PAD = 1.6;

export const HUE = {
  ash: 0x6f675c, ashDark: 0x4f4941, ashLight: 0x8c8273, crack: 0x2a2622, pebble: 0x9a9080,
  void: 0x0b0c0f, voidLift: 0x17181c, lip: 0x8a8174, cliff: 0x121214,
  theirs: 0x4fc3d9, ink: 0x0b0c10, metal: 0x1b1d22, trim: 0xf3d79a,
  hpGood: 0x74e08a, hpMid: 0xf2cf5b, hpLow: 0xf0604c, shield: 0x8fd6ff, frost: 0xbfeaff,
  acid: 0xc6f25a, heal: 0x9dffb4, focus: 0xffd166, fire: 0xffa24a, fireCore: 0xfff2c8, smoke: 0x2a2826,
  panel: 0x0e1014, row: 0x171a20,
} as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

interface Layer { color: N<'vec3'>; alpha: N<'float'> }
const clear = (): Layer => ({ color: vec3(0), alpha: float(0) });
const over = (below: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(below.color.mul(alpha.oneMinus())),
  alpha: alpha.add(below.alpha.mul(alpha.oneMinus())),
});
const finish = (material: THREE.MeshBasicNodeMaterial, layer: Layer, fade: Scalar = 1) => {
  material.colorNode = layer.color.div(max(layer.alpha, 1e-4));
  material.opacityNode = layer.alpha.mul(fade);
  return material;
};

const rotate = (p: N<'vec2'>, angle: N<'float'>): N<'vec2'> => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
};

const hexagon = (p: N<'vec2'>, apothem: N<'float'>): N<'float'> => {
  const k = vec2(-0.866025404, 0.5), a = abs(p);
  const b = a.sub(k.mul(min(dot(k, a), 0).mul(2)));
  const c = b.sub(vec2(clamp(b.x, apothem.mul(-0.577350269), apothem.mul(0.577350269)), apothem));
  return length(c).mul(sign(c.y));
};

const smin = (a: N<'float'>, b: N<'float'>, k: N<'float'>): N<'float'> => {
  const h = clamp(float(0.5).add(b.sub(a).mul(0.5).div(k)), 0, 1);
  return mix(b, a, h).sub(k.mul(h).mul(h.oneMinus()));
};

const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));

const pick = (index: N<'float'>, values: readonly N<'float'>[]): N<'float'> =>
  values.slice(0, -1).reduceRight<N<'float'>>((rest, value, i) => select(index.lessThan(i + 0.5), value, rest), values.at(-1)!);

const pickColor = (index: N<'float'>, values: readonly N<'vec3'>[]): N<'vec3'> =>
  values.slice(0, -1).reduceRight<N<'vec3'>>((rest, value, i) => select(index.lessThan(i + 0.5), value, rest), values.at(-1)!);

export function radiusOf(hp: number, shield: number): number {
  return 0.4 + 0.33 * Math.sqrt((hp + shield) / 100);
}

export interface Ground { material: THREE.MeshBasicNodeMaterial; surround: Level }

export function groundMaterial(shape: SkirmishShape, lit: Lit, ours: number): Ground {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const surround = uniform(0);
  const half = shape.arena / 2, world = vec2(positionWorld.x, positionWorld.z), e = world.add(half);
  const near = shape.spawn[SPAWN.REFLECT_NEAR]!, far = shape.spawn[SPAWN.REFLECT_FAR]!, centre = shape.spawn[SPAWN.CENTRE]!;
  const ringNear = shape.spawn[SPAWN.RING_NEAR]! * Math.SQRT1_2, ringFar = shape.spawn[SPAWN.RING_FAR]! * Math.SQRT1_2;

  const broad = fractal(world.mul(0.09), 4), mid = noise(world.mul(0.55).add(7.3)), fine = noise(world.mul(3.1).add(1.7));
  const plates = world.mul(0.34), base = floor(plates), local = fract(plates);
  let nearest: N<'float'> = float(9), second: N<'float'> = float(9), tone: N<'float'> = float(0);
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
    const id = base.add(vec2(i, j)), offset = vec2(hash(id), hash(id.add(31.7))).mul(0.8).add(0.1), r = vec2(i, j).add(offset).sub(local), d = dot(r, r);
    const closer = d.lessThan(nearest);
    second = select(closer, nearest, min(second, d));
    tone = select(closer, hash(id.add(5.3)), tone);
    nearest = min(nearest, d);
  }
  const seam = sqrt(second).sub(sqrt(nearest)).div(0.34), crackMask = smoothstep(-0.25, 0.25, noise(world.mul(0.16).add(2.7)));
  let ash: N<'vec3'> = mix(rgb(HUE.ashDark), rgb(HUE.ash), smoothstep(-0.5, 0.45, broad.mul(0.8).add(tone.sub(0.5).mul(0.22))));
  ash = mix(ash, rgb(HUE.ashLight), smoothstep(0.3, 0.8, mid).mul(0.25));
  ash = ash.mul(fine.mul(0.07).add(1)).mul(smoothstep(0, 0.5, seam).mul(0.07).mul(crackMask).add(0.95));
  ash = mix(ash, rgb(HUE.crack), coverage(seam.sub(0.018)).mul(crackMask).mul(0.5));
  const cell = floor(world.mul(1.6)), spot = fract(world.mul(1.6)).sub(0.5), jitter = vec2(hash(cell), hash(cell.add(17.3))).sub(0.5).mul(0.5);
  const pebble = coverage(sdCircle(spot.sub(jitter), hash(cell.add(4.1)).mul(0.07).add(0.03))).mul(step(0.86, hash(cell.add(9.7))));
  ash = mix(ash, rgb(HUE.pebble), pebble.mul(0.6));

  const minor = abs(fract(e.add(0.5)).sub(0.5)), major = abs(fract(e.div(8).add(0.5)).sub(0.5)).mul(8);
  const lines = max(coverage(min(minor.x, minor.y).sub(0.012)).mul(0.05), coverage(min(major.x, major.y).sub(0.02)).mul(0.1));
  ash = mix(ash, rgb(0xb9b0a0), lines);

  const oursTint = rgb(ours), theirsTint = rgb(HUE.theirs);
  const zone = (d: N<'float'>, tint: N<'vec3'>, base: N<'vec3'>): N<'vec3'> => {
    const inside = coverage(d), dash = step(0.5, fract(e.x.add(e.y).mul(0.9)));
    return mix(mix(base, tint.mul(0.55), inside.mul(0.07)), tint.mul(0.9), coverage(abs(d).sub(0.035)).mul(dash).mul(0.32));
  };
  const reflectHalf = vec2((far - near) / 2, half - near);
  const reflectOurs = sdBox(e.sub(vec2((near + far) / 2, half)), reflectHalf, 0.4);
  const reflectTheirs = sdBox(e.sub(vec2(shape.arena - (near + far) / 2, half)), reflectHalf, 0.4);
  const huddle = sdBox(e.sub(half), vec2(centre), 0.4);
  const diagonal = abs(e.sub(half)), along = clamp(diagonal.x.add(diagonal.y).mul(0.5), ringNear, ringFar);
  const ring = length(diagonal.sub(vec2(along))).sub(0.45);
  const reflected = zone(reflectTheirs, theirsTint, zone(reflectOurs, oursTint, ash));
  const surrounded = zone(ring, theirsTint, zone(huddle, oursTint, ash));
  const field = mix(reflected, surrounded, surround);

  const edge = sdBox(world, vec2(half), 0), outer = edge.sub(0.55);
  const lipBand = coverage(edge.negate()).mul(coverage(outer));
  let lip: N<'vec3'> = mix(rgb(HUE.lip), rgb(HUE.cliff), smoothstep(0.05, 0.55, edge)).mul(fine.mul(0.08).add(0.95));
  lip = mix(lip, rgb(0xd6cbb8), coverage(abs(edge.sub(0.03)).sub(0.02)).mul(0.6));
  const drop = exp(outer.mul(-1.6)).mul(0.55);
  const beyond = mix(rgb(HUE.void), rgb(HUE.voidLift), smoothstep(-0.5, 0.6, broad)).mul(fine.mul(0.1).add(1)).mul(drop.oneMinus());
  const inset = smoothstep(0.9, 0, edge.negate()).mul(0.22);

  const onArena = coverage(edge);
  const surface = mix(mix(lit(beyond), lit(lip), lipBand), lit(field.mul(inset.oneMinus())), onArena);
  material.colorNode = surface;
  return { material, surround };
}

export interface UnitArt { geometry: THREE.BufferGeometry; body: THREE.InstancedBufferAttribute; life: THREE.InstancedBufferAttribute; mark: THREE.InstancedBufferAttribute }

export function unitGeometry(capacity: number): UnitArt {
  const geometry = flat(SPRITE, SPRITE);
  const channel = (name: string) => {
    const attr = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, attr);
    return attr;
  };
  return { geometry, body: channel('aBody'), life: channel('aLife'), mark: channel('aMark') };
}

interface Parts { body: N<'float'>; metal: N<'float'>; detail: N<'float'>; energy: N<'float'> }

function partsOf(kind: N<'float'>, q: N<'vec2'>, R: N<'float'>, time: Level): Parts {
  const m = vec2(q.x, abs(q.y)), at = (x: number, y: number) => vec2(R.mul(x), R.mul(y)), r = (k: number) => R.mul(k);
  const seg = (p: N<'vec2'>, ax: number, ay: number, bx: number, by: number, w: number) => sdSegment(p, at(ax, ay), at(bx, by), r(w));
  const dot2 = (p: N<'vec2'>, x: number, y: number, w: number) => sdCircle(p.sub(at(x, y)), r(w));
  const none = float(1e3);
  const strider: Parts = {
    body: hexagon(q, r(0.55)),
    metal: min(seg(m, 0.15, 0.25, 0.82, 0.95, 0.085), seg(m, -0.15, 0.25, -0.8, 0.92, 0.085), dot2(m, 0.82, 0.95, 0.11), dot2(m, -0.8, 0.92, 0.11)),
    detail: hexagon(q.add(at(0.08, 0)), r(0.34)),
    energy: dot2(q, 0.32, 0, 0.17),
  };
  const sweep = atan(m.y, m.x), crescent = max(abs(length(m).sub(r(0.95))).sub(r(0.075)), max(float(0.18).sub(sweep), sweep.sub(1.2)).mul(length(m)));
  const blade: Parts = {
    body: hexagon(q, r(0.6)),
    metal: dot2(m, 0.05, 0.6, 0.2),
    detail: hexagon(q, r(0.34)),
    energy: crescent,
  };
  const titan: Parts = {
    body: hexagon(q, r(0.5)),
    metal: min(seg(m, 0.25, 0.35, 0.92, 1.0, 0.075), seg(m, -0.25, 0.35, -0.88, 1.0, 0.075), dot2(m, 0.92, 1.0, 0.1), dot2(m, -0.88, 1.0, 0.1), seg(m, -0.15, 0.68, 1.12, 0.68, 0.12), dot2(m, 0, 0.62, 0.17)),
    detail: hexagon(q, r(0.3)),
    energy: min(dot2(m, 1.16, 0.68, 0.1), hexagon(q, r(0.16))),
  };
  const trooper: Parts = {
    body: sdBox(q, at(0.52, 0.5), r(0.26)),
    metal: min(seg(q, 0.05, -0.32, 1.2, -0.32, 0.085), sdBox(q.add(at(0.5, 0)), at(0.18, 0.34), r(0.06))),
    detail: sdBox(q.sub(at(0.26, 0)), at(0.1, 0.3), r(0.05)),
    energy: none,
  };
  const breacher: Parts = {
    body: min(sdBox(q, at(0.6, 0.56), r(0.22)), dot2(m, -0.05, 0.58, 0.28)),
    metal: seg(m, 0.1, 0.42, 1.08, 0.42, 0.14),
    detail: sdBox(q.sub(at(0.22, 0)), at(0.12, 0.34), r(0.05)),
    energy: dot2(m, 1.08, 0.42, 0.075),
  };
  const medic: Parts = {
    body: seg(q, -0.7, 0, 0.72, 0, 0.3),
    metal: seg(m, -0.05, 0.2, -0.4, 0.92, 0.2),
    detail: seg(q, 0.3, 0, 0.55, 0, 0.13),
    energy: dot2(m, -0.52, 0.9, 0.15),
  };
  const skitter: Parts = {
    body: smin(sdCircle(q.add(at(0.12, 0)), r(0.5)), sdCircle(q.sub(at(0.42, 0)), r(0.34)), r(0.2)),
    metal: min(seg(m, 0.25, 0.2, 0.5, 0.8, 0.06), seg(m, -0.05, 0.25, -0.05, 0.85, 0.06), seg(m, -0.35, 0.2, -0.65, 0.75, 0.06), seg(m, 0.6, 0.28, 1.05, 0.08, 0.07)),
    detail: seg(q, -0.45, 0, 0.3, 0, 0.08),
    energy: dot2(m, 0.62, 0.14, 0.07),
  };
  const larva = smin(smin(sdCircle(q.sub(at(0.55, 0)), r(0.27)), sdCircle(q.sub(at(0.02, 0)), r(0.4)), r(0.1)), sdCircle(q.add(at(0.55, 0)), r(0.33)), r(0.1));
  const spitter: Parts = {
    body: larva,
    metal: min(seg(m, -0.05, 0.3, -0.35, 0.72, 0.075), seg(m, 0.2, 0.3, 0.05, 0.7, 0.075), seg(m, -0.62, 0.22, -0.9, 0.5, 0.065), seg(m, 0.72, 0.12, 1.1, 0.06, 0.06)),
    detail: max(min(abs(q.x.sub(r(0.3))).sub(r(0.03)), abs(q.x.add(r(0.27))).sub(r(0.03))), larva.add(r(0.07))),
    energy: dot2(q, 0.6, 0, 0.12),
  };
  const swell = sin(time.mul(7)).mul(0.04).add(1);
  const burster: Parts = {
    body: sdCircle(q, r(0.66)),
    metal: min(seg(m, 0.3, 0.45, 0.55, 0.8, 0.08), seg(m, -0.3, 0.45, -0.55, 0.8, 0.08)),
    detail: abs(sdCircle(q, r(0.52))).sub(r(0.03)),
    energy: sdCircle(q.add(at(0.06, 0)), r(0.38).mul(swell)),
  };
  const all = [strider, blade, titan, trooper, breacher, medic, skitter, spitter, burster];
  return {
    body: pick(kind, all.map((part) => part.body)),
    metal: pick(kind, all.map((part) => part.metal)),
    detail: pick(kind, all.map((part) => part.detail)),
    energy: pick(kind, all.map((part) => part.energy)),
  };
}

export function unitMaterial(ours: number, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const body = attribute<'vec4'>('aBody', 'vec4'), life = attribute<'vec4'>('aLife', 'vec4'), mark = attribute<'vec4'>('aMark', 'vec4');
  const kind = body.x, team = body.y, heading = body.z, R = body.w;
  const shieldFrac = life.y, hit = life.z, alpha = life.w, slowed = mark.x, race = floor(kind.div(3).add(0.01));
  const p = uv().sub(0.5).mul(SPRITE), q = rotate(p, heading);
  const parts = partsOf(kind, q, R, time);
  const tint = mix(rgb(ours), rgb(HUE.theirs), team);
  const metal = mix(tint, rgb(HUE.metal), 0.74), detail = mix(tint, rgb(0xffffff), 0.38), glow = mix(tint, rgb(0xffffff), 0.62).mul(1.7);
  const radial = length(p).div(R);
  const dome = float(1.14).sub(radial.mul(radial).mul(0.38)).add(dot(normalize(p.add(1e-4)), vec2(-0.6, 0.6)).mul(radial).mul(0.14));
  const stripes = pick(race, [
    coverage(abs(parts.body.add(R.mul(0.1))).sub(R.mul(0.025))).mul(0.8),
    coverage(abs(q.x.add(R.mul(0.08))).sub(R.mul(0.025))).mul(coverage(parts.body.add(R.mul(0.12)))).mul(0.5),
    step(0.62, fract(q.x.div(R).mul(2.6).add(0.15))).mul(coverage(parts.body.add(R.mul(0.1)))).mul(0.3),
  ]);
  const trimColor = pickColor(race, [mix(rgb(HUE.trim), tint, 0.25), tint.mul(0.55), tint.mul(0.62)]);
  let fill: N<'vec3'> = mix(tint.mul(dome), trimColor, stripes);
  fill = mix(fill, rgb(0xffffff), hit.mul(0.3));
  fill = mix(fill, mix(fill, rgb(HUE.frost), 0.35), slowed);
  const shape = min(parts.body, parts.metal, parts.detail, parts.energy);
  const core = select(kind.greaterThan(7.5), mix(rgb(HUE.acid), tint, 0.2).mul(sin(time.mul(7)).mul(0.25).add(1.6)), glow);
  const cross = min(sdBox(q.add(vec2(R.mul(0.2), 0)), vec2(R.mul(0.2), R.mul(0.06)), R.mul(0.02)), sdBox(q.add(vec2(R.mul(0.2), 0)), vec2(R.mul(0.06), R.mul(0.2)), R.mul(0.02)));
  const medic = step(abs(kind.sub(5)), 0.5);
  const bubble = abs(parts.body.sub(R.mul(0.2))).sub(R.mul(0.03));
  const shielded = step(race, 0.5).mul(step(0.01, shieldFrac));
  const frost = abs(length(q).sub(R.mul(1.02))).sub(R.mul(0.022)), frostDash = step(0.45, fract(atan(q.y, q.x).div(Math.PI / 4).add(time.mul(0.3))));
  let layer = clear();
  layer = over(layer, rgb(HUE.ink), coverage(shape.sub(R.mul(0.07))).mul(0.9));
  layer = over(layer, metal.mul(dome.mul(0.4).add(0.7)), coverage(parts.metal));
  layer = over(layer, fill, coverage(parts.body));
  layer = over(layer, mix(detail, rgb(0xffffff), hit.mul(0.3)), coverage(parts.detail).mul(0.9));
  layer = over(layer, core, coverage(parts.energy));
  layer = over(layer, rgb(0xffffff), coverage(cross).mul(medic));
  layer = over(layer, mix(rgb(HUE.shield), tint, 0.3).mul(1.4), coverage(bubble).mul(shielded).mul(shieldFrac.mul(0.22).add(hit.mul(0.7))));
  layer = over(layer, rgb(HUE.frost), coverage(frost).mul(frostDash).mul(slowed).mul(0.6));
  return finish(material, layer, alpha);
}

export function shadowMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const body = attribute<'vec4'>('aBody', 'vec4'), life = attribute<'vec4'>('aLife', 'vec4');
  const kind = body.x, R = body.w, air = step(abs(kind.sub(5)), 0.5);
  const p = uv().sub(0.5).mul(SPRITE), offset = vec2(mix(float(0.14), float(0.55), air), mix(float(-0.2), float(-0.7), air)).mul(R);
  const d = length(p.sub(offset).div(vec2(1, 0.85))).sub(R.mul(mix(float(0.78), float(0.7), air)));
  material.colorNode = vec3(0);
  material.opacityNode = smoothstep(R.mul(0.35), R.mul(-0.25), d).mul(mix(float(0.45), float(0.3), air)).mul(life.w);
  return material;
}

export interface BarArt { geometry: THREE.BufferGeometry; state: THREE.InstancedBufferAttribute; size: THREE.InstancedBufferAttribute }

export function barGeometry(capacity: number): BarArt {
  const geometry = flat(1, 1);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const size = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aBar', state);
  geometry.setAttribute('aSize', size);
  return { geometry, state, size };
}

export function barMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const bar = attribute<'vec4'>('aBar', 'vec4'), size = attribute<'vec4'>('aSize', 'vec4');
  const hp = bar.x, shield = bar.y, alpha = bar.z, W = size.x, H = size.y;
  const p = uv().sub(0.5).mul(vec2(W, H)), has = step(0, shield);
  const frame = sdBox(p, vec2(W, H).mul(0.5), H.mul(0.18));
  const inner = vec2(W.mul(0.5).sub(H.mul(0.12)), H.mul(0.5).sub(H.mul(0.12)));
  const bottom = inner.y.negate(), hpTop = mix(inner.y, H.mul(-0.05), has), shieldBottom = H.mul(0.05);
  const hpRow = sdBox(p.sub(vec2(0, bottom.add(hpTop).mul(0.5))), vec2(inner.x, hpTop.sub(bottom).mul(0.5)), 0);
  const shieldRow = sdBox(p.sub(vec2(0, shieldBottom.add(inner.y).mul(0.5))), vec2(inner.x, inner.y.sub(shieldBottom).mul(0.5)), 0);
  const x = p.x.add(inner.x).div(inner.x.mul(2));
  const hpColor = mix(mix(rgb(HUE.hpLow), rgb(HUE.hpMid), smoothstep(0.2, 0.45, hp)), rgb(HUE.hpGood), smoothstep(0.5, 0.75, hp));
  let layer = clear();
  layer = over(layer, rgb(HUE.ink), coverage(frame).mul(0.85));
  layer = over(layer, rgb(0x262a31), coverage(hpRow).mul(0.9));
  layer = over(layer, hpColor.mul(1.15), coverage(hpRow).mul(step(x, hp)));
  layer = over(layer, rgb(0x1d2733), coverage(shieldRow).mul(has));
  layer = over(layer, rgb(HUE.shield).mul(1.2), coverage(shieldRow).mul(has).mul(step(x, shield)));
  return finish(material, layer, alpha);
}

export const FX = { MUZZLE: 0, IMPACT: 1, SLASH: 2, FROST: 3, BOOM: 4, ACID: 5, SPLASH: 6, WARP: 7, HEAL: 8 } as const;

export interface FxArt { geometry: THREE.BufferGeometry; state: THREE.InstancedBufferAttribute; tint: THREE.InstancedBufferAttribute }

export function fxGeometry(capacity: number): FxArt {
  const geometry = flat(2, 2);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const tint = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aFx', state);
  geometry.setAttribute('aTint', tint);
  return { geometry, state, tint };
}

export function fxMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const fx = attribute<'vec4'>('aFx', 'vec4'), tintAttr = attribute<'vec4'>('aTint', 'vec4');
  const kind = fx.x, t = clamp(fx.y, 0, 1), seed = fx.w, tint = tintAttr.xyz;
  const p = uv().sub(0.5).mul(2), r = length(p), angle = atan(p.y, p.x), fade = t.oneMinus();
  const light = mix(tint, rgb(0xffffff), 0.35);
  const solo = (color: N<'vec3'>, alpha: N<'float'>): Layer => over(clear(), color, alpha);

  const spikes = abs(sin(angle.mul(2))).mul(0.55).add(0.45);
  const muzzle = solo(light.mul(2.2), coverage(r.sub(spikes.mul(0.75).mul(fade.mul(0.6).add(0.4)))).mul(fade));

  const spark = coverage(abs(r.sub(t.mul(0.7).add(0.25))).sub(fade.mul(0.08).add(0.02)));
  const impact = solo(light.mul(2), max(spark.mul(fade), coverage(r.sub(fade.mul(0.3))).mul(fade.mul(fade))));

  const sweep = t.mul(2.4).sub(1.2), arc = abs(r.sub(0.7)).sub(float(0.09).mul(smoothstep(1.3, 0, abs(angle.sub(sweep)))));
  const slash = solo(light.mul(2.2), coverage(arc).mul(step(abs(angle.sub(sweep.mul(0.5))), 1.1)).mul(fade));

  const shard = fract(angle.div(Math.PI / 5).add(seed));
  const shards = coverage(max(abs(shard.sub(0.5)).mul(0.9).sub(0.12), abs(r.sub(t.mul(0.55).add(0.3))).sub(0.1)));
  const frost = solo(rgb(HUE.frost).mul(1.8), max(shards, coverage(r.sub(0.35)).mul(0.35)).mul(fade));

  const blast = smoothstep(0.1, 0.55, t), puff = fractal(p.mul(2.6).add(seed.mul(13)), 3).mul(0.25);
  const ball = r.sub(mix(float(0.2), float(0.62), blast).add(puff.mul(0.4)));
  const shock = abs(r.sub(t.mul(0.85).add(0.1))).sub(fade.mul(0.025).add(0.006));
  const spoke = angle.div(Math.PI / 6).add(seed.mul(7)), lane = floor(spoke);
  const debris = sdCircle(vec2(fract(spoke).sub(0.5).mul(0.35), r.sub(t.mul(0.75).add(0.12).mul(hash(vec2(lane, seed)).mul(0.4).add(0.8)))), 0.05);
  const fireColor = mix(rgb(HUE.fireCore).mul(2.2), mix(rgb(HUE.fire).mul(1.8), rgb(HUE.smoke), smoothstep(0.2, 0.7, t)), smoothstep(0, 0.2, t).mul(smoothstep(0, 0.45, r.add(t.mul(0.5)))));
  let boom = solo(light.mul(1.6), coverage(shock).mul(fade).mul(0.6));
  boom = over(boom, fireColor, coverage(ball).mul(fade.mul(fade).mul(0.9).add(0.1)).mul(smoothstep(1, 0.7, t)));
  boom = over(boom, tint.mul(1.8), coverage(debris).mul(fade));

  const blobs = fractal(p.mul(3.2).add(seed.mul(5)), 3);
  const acidFill = coverage(r.sub(t.mul(0.4).add(0.6))).mul(smoothstep(-0.1, 0.4, blobs)).mul(0.55);
  const acid = solo(rgb(HUE.acid).mul(1.8), max(coverage(abs(r.sub(0.93)).sub(0.035)).mul(0.9), acidFill).mul(fade));

  const splash = solo(light.mul(2), max(coverage(abs(r.sub(0.9)).sub(0.05)), coverage(r.sub(0.9)).mul(0.25)).mul(fade));

  const warpRing = abs(r.sub(t.mul(0.8).add(0.2))).sub(fade.mul(0.06).add(0.01));
  const warp = solo(tint.mul(1.6), max(coverage(warpRing), coverage(r.sub(t.mul(0.5).add(0.15))).mul(fade.mul(0.3))).mul(fade));

  const rise = p.sub(vec2(0, t.mul(0.5)));
  const plus = min(sdBox(rise, vec2(0.28, 0.08), 0.03), sdBox(rise, vec2(0.08, 0.28), 0.03));
  const heal = solo(rgb(HUE.heal).mul(1.9), coverage(plus).mul(fade).mul(smoothstep(0, 0.15, t)));

  const layers = [muzzle, impact, slash, frost, boom, acid, splash, warp, heal];
  const alpha = clamp(pick(kind, layers.map((layer) => layer.alpha)), 0, 1);
  material.colorNode = pickColor(kind, layers.map((layer) => layer.color)).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  return material;
}

export const STREAK = { BOLT: 0, TRACER: 1, BEAM: 2, SHELL: 3, SPINE: 4, HEAL: 5 } as const;
export const ARRIVE = 0.3;

export interface StreakArt { geometry: THREE.BufferGeometry; state: THREE.InstancedBufferAttribute; tint: THREE.InstancedBufferAttribute }

export function streakGeometry(capacity: number): StreakArt {
  const geometry = flat(1, 1);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const tint = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aStreak', state);
  geometry.setAttribute('aTint', tint);
  return { geometry, state, tint };
}

export function streakMaterial(time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const s = attribute<'vec4'>('aStreak', 'vec4'), tintAttr = attribute<'vec4'>('aTint', 'vec4');
  const kind = s.x, t = clamp(s.y, 0, 1), span = s.z, width = s.w, tint = tintAttr.xyz;
  const p = uv().sub(0.5).mul(vec2(span, width)), along = p.x.add(span.mul(0.5)), fade = t.oneMinus();
  const light = mix(tint, rgb(0xffffff), 0.3);
  const head = span.mul(clamp(t.div(ARRIVE), 0, 1)), flying = step(t, ARRIVE), trail = smoothstep(1, ARRIVE, t);
  const tail = (length: number, radius: number) => coverage(sdSegment(vec2(along, p.y), vec2(max(head.sub(length), 0), 0), vec2(head, 0), float(radius))).mul(trail);
  const boltAlpha = max(tail(0.6, 0.06).mul(0.7), coverage(sdCircle(vec2(along.sub(head), p.y), 0.14)).mul(flying));
  const tracer = coverage(abs(p.y).sub(0.03)).mul(step(0.35, fract(along.mul(1.2).sub(t.mul(5))))).mul(fade.mul(fade));
  const beamCore = coverage(abs(p.y).sub(fade.mul(0.07).add(0.015))), beamHalo = exp(abs(p.y).mul(-9)).mul(0.5);
  const beam = max(beamCore, beamHalo).mul(fade).mul(smoothstep(0, 0.08, t));
  const shell = max(coverage(sdCircle(vec2(along.sub(head), p.y), 0.17)).mul(flying), tail(0.45, 0.05).mul(0.6));
  const spine = tail(0.55, 0.055);
  const flow = fract(along.mul(1.4).sub(time.mul(2.5)));
  const heal = max(coverage(abs(p.y).sub(0.03)).mul(0.55), coverage(abs(p.y).sub(0.07)).mul(step(0.7, flow))).mul(fade.mul(0.6).add(0.4));
  const colors = [light.mul(2.2), light.mul(2.4), light.mul(2.6), mix(rgb(HUE.frost), light, 0.4).mul(2.2), mix(rgb(HUE.acid), tint, 0.4).mul(1.8), rgb(HUE.heal).mul(1.8)];
  const alphas = [boltAlpha, tracer, beam, shell, spine, heal];
  material.colorNode = pickColor(kind, colors);
  material.opacityNode = clamp(pick(kind, alphas), 0, 1);
  return material;
}

export interface WreckArt { geometry: THREE.BufferGeometry; state: THREE.InstancedBufferAttribute; fade: THREE.InstancedBufferAttribute }

export function wreckGeometry(capacity: number): WreckArt {
  const geometry = flat(3, 3);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const fade = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aWreck', state);
  geometry.setAttribute('aFade', fade);
  return { geometry, state, fade };
}

export function wreckMaterial(ours: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const w = attribute<'vec4'>('aWreck', 'vec4');
  const race = w.x, team = w.y, age = w.z, seed = w.w;
  const p = uv().sub(0.5).mul(3), r = length(p), angle = atan(p.y, p.x);
  const tint = mix(rgb(ours), rgb(HUE.theirs), team);
  const rough = fractal(p.mul(2.2).add(seed.mul(17)), 3);
  const scorch = smoothstep(1.2, 0.2, r.add(rough.mul(0.35))).mul(0.62);
  const sector = floor(angle.div(Math.PI / 4).add(seed.mul(8))), local = fract(angle.div(Math.PI / 4).add(seed.mul(8))).sub(0.5);
  const distance = hash(vec2(sector, seed)).mul(0.5).add(0.25);
  const chip = sdBox(vec2(local.mul(r).mul(0.8), r.sub(distance)), vec2(0.07, 0.05), 0.02);
  const goo = smoothstep(0.62, 0.3, r.add(rough.mul(0.4)));
  const ember = smoothstep(0.15, 0.55, rough).mul(smoothstep(0.7, 0.2, r)).mul(exp(age.mul(-0.9)));
  const brood = step(1.5, race);
  let layer = over(clear(), rgb(0x0d0c0b), scorch);
  layer = over(layer, lit(mix(tint, rgb(0x2a2a2a), 0.6)), coverage(chip).mul(step(0.3, hash(vec2(sector, seed.add(3.3))))).mul(brood.oneMinus()).mul(0.9));
  layer = over(layer, lit(mix(tint.mul(0.4), rgb(0x2b3a18), 0.5)), goo.mul(brood).mul(0.8));
  layer = over(layer, rgb(HUE.fire).mul(1.6), ember.mul(0.8));
  return finish(material, layer, smoothstep(0, 0.25, age).mul(attribute<'vec4'>('aFade', 'vec4').x));
}

export interface AgentRing { material: THREE.MeshBasicNodeMaterial; radius: Level; range: Level; cool: Level; reach: Level; alpha: Level; aimX: Level; aimY: Level; push: Level }

export function agentRingMaterial(accent: number, time: Level, size: number): AgentRing {
  const material = sheet();
  const radius = uniform(0.5), range = uniform(5), cool = uniform(0), reach = uniform(0), alpha = uniform(1), aimX = uniform(0), aimY = uniform(0), push = uniform(0);
  const p = uv().sub(0.5).mul(size), r = length(p), turn = fract(atan(p.x, p.y).div(2 * Math.PI).add(1));
  const tone = rgb(accent);
  const ring = abs(r.sub(radius.add(0.22))).sub(0.05);
  const arc = abs(r.sub(radius.add(0.42))).sub(0.06);
  const ready = step(cool, 0.001);
  const dashes = step(0.5, fract(atan(p.y, p.x).mul(range).div(Math.PI).mul(1.6).add(time.mul(0.25))));
  const rim = abs(r.sub(range)).sub(mix(float(0.03), float(0.05), reach));
  const inside = smoothstep(range, range.sub(1.4), r).mul(step(r, range)).mul(0.05);
  const aim = vec2(aimX, aimY), facing = rotate(p, atan(aim.y, aim.x));
  const chevron = max(abs(facing.y).sub(float(0.36).sub(facing.x.sub(radius.add(0.72))).mul(0.9)), radius.add(0.66).sub(facing.x));
  let layer = clear();
  layer = over(layer, tone, inside.mul(reach.mul(0.6).add(0.4)));
  layer = over(layer, tone.mul(mix(float(1), float(1.8), reach)), coverage(rim).mul(dashes).mul(mix(float(0.35), float(0.8), reach)));
  layer = over(layer, rgb(HUE.ink), coverage(ring.sub(0.04)).mul(0.6));
  layer = over(layer, tone.mul(1.8), coverage(ring));
  layer = over(layer, tone.mul(0.45), coverage(arc).mul(0.7));
  layer = over(layer, tone.mul(ready.mul(0.8).add(1.4)), coverage(arc).mul(step(cool, turn)));
  layer = over(layer, tone.mul(1.9), coverage(chevron).mul(push));
  return { material: finish(material, layer, alpha), radius, range, cool, reach, alpha, aimX, aimY, push };
}

export interface Reticle { material: THREE.MeshBasicNodeMaterial; radius: Level; jump: Level; alpha: Level }

export function reticleMaterial(accent: number, time: Level, size: number): Reticle {
  const material = sheet();
  const radius = uniform(0.6), jump = uniform(0), alpha = uniform(1);
  const p = uv().sub(0.5).mul(size), spin = time.mul(0.6).add(jump.mul(1.2));
  const q = rotate(p, spin), a = abs(q), reach = radius.add(0.38).add(jump.mul(0.8));
  const corner = min(sdBox(a.sub(vec2(reach, reach.sub(0.18))), vec2(0.05, 0.22), 0.02), sdBox(a.sub(vec2(reach.sub(0.18), reach)), vec2(0.22, 0.05), 0.02));
  let layer = over(clear(), rgb(HUE.ink), coverage(corner.sub(0.05)).mul(0.7));
  layer = over(layer, rgb(accent).mul(2), coverage(corner));
  return { material: finish(material, layer, alpha), radius, jump, alpha };
}

export interface Focus { material: THREE.MeshBasicNodeMaterial; radius: Level; pulse: Level; alpha: Level }

export function focusMaterial(time: Level, size: number): Focus {
  const material = sheet();
  const radius = uniform(0.6), pulse = uniform(0), alpha = uniform(1);
  const p = uv().sub(0.5).mul(size), r = length(p);
  const beat = sin(time.mul(5)).mul(0.08).add(pulse.mul(0.5));
  const reach = radius.add(0.9).add(beat);
  const turn = fract(atan(p.y, p.x).div(Math.PI / 2).add(0.5)).sub(0.5).mul(Math.PI / 2);
  const q = vec2(r.mul(cos(turn)), r.mul(sin(turn)));
  const chevron = max(abs(q.y).sub(q.x.sub(reach.sub(0.34)).mul(0.75)), q.x.sub(reach));
  const ring = abs(r.sub(radius.add(0.3))).sub(0.035);
  let layer = over(clear(), rgb(HUE.ink), coverage(chevron.sub(0.06)).mul(0.7));
  layer = over(layer, rgb(HUE.focus).mul(2), coverage(chevron));
  layer = over(layer, rgb(HUE.focus).mul(1.4), coverage(ring).mul(0.55).mul(step(0.5, fract(atan(p.y, p.x).div(Math.PI / 8)))));
  return { material: finish(material, layer, alpha), radius, pulse, alpha };
}

export interface RowArt { geometry: THREE.BufferGeometry; state: THREE.InstancedBufferAttribute; size: THREE.InstancedBufferAttribute }

export function rowGeometry(capacity: number): RowArt {
  const geometry = flat(1, 1);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const size = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aRow', state);
  geometry.setAttribute('aSize', size);
  return { geometry, state, size };
}

export function rowMaterial(ours: number, accent: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const row = attribute<'vec4'>('aRow', 'vec4'), size = attribute<'vec4'>('aSize', 'vec4');
  const team = row.x, role = row.y, dead = row.z, flash = row.w, W = size.x, H = size.y;
  const p = uv().sub(0.5).mul(vec2(W, H));
  const box = sdBox(p, vec2(W, H).mul(0.5).sub(0.04), H.mul(0.2));
  const tint = mix(rgb(ours), rgb(HUE.theirs), team);
  const outline = pickColor(role, [tint.mul(0.5), rgb(accent), rgb(accent), rgb(HUE.focus)]);
  const outlined = step(0.5, role);
  let layer = over(clear(), mix(rgb(HUE.row), tint.mul(0.25), flash.mul(0.8).add(0.15)), coverage(box).mul(mix(float(0.85), float(0.45), dead)));
  layer = over(layer, outline, coverage(abs(box).sub(mix(float(0.015), float(0.045), outlined))).mul(mix(float(0.35), float(1), outlined)));
  layer = over(layer, tint.mul(0.9), coverage(sdBox(p.add(vec2(W.mul(0.5).sub(0.08), 0)), vec2(0.04, H.mul(0.32)), 0.02)).mul(dead.oneMinus()));
  return finish(material, layer);
}

export function panelMaterial(tint: number, width: number, height: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(vec2(width, height));
  const box = sdBox(p, vec2(width / 2 - 0.05, height / 2 - 0.05), 0.35);
  const rule = sdBox(p.sub(vec2(0, height / 2 - 0.32)), vec2(width / 2 - 0.5, 0.05), 0.04);
  let layer = over(clear(), rgb(HUE.panel), coverage(box).mul(0.82));
  layer = over(layer, rgb(0x2a2e36), coverage(abs(box).sub(0.02)).mul(0.9));
  layer = over(layer, rgb(tint), coverage(rule));
  return finish(material, layer);
}

export function lampBank(light: Light2D, capacity: number): THREE.InstancedMesh {
  const template = light.emitter(new THREE.CircleGeometry(1, 24).rotateX(-Math.PI / 2), vec3(1));
  const mesh = instancedMesh(template.geometry, template.material, capacity);
  mesh.layers.set(LIGHT_LAYER);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage);
  mesh.count = 0;
  mesh.frustumCulled = false;
  return mesh;
}
