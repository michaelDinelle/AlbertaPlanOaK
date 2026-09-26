import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fwidth, int, length, max, min, mix, mod, positionWorld, pow, select, sin, smoothstep, step, uniformArray, uv, vec2 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { LIGHT_LAYER } from '@kit/light2d';
import { noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Scalar = THREE.UniformNode<'float', number>;
export type Point = THREE.UniformNode<'vec2', THREE.Vector2>;

export const TAU = 2 * Math.PI;
export const R_CASE = 3.75;
export const R_BEZEL = 3.44;
export const R_LANE = 3.08;
export const R_SOCKET = 2.5;
export const R_ROSE = 1.92;
export const R_KEY = 0.98;
export const R_JEWEL = 1.6;
export const KEY = 0.42;
export const BELL = 0.34;
export const BELL_Y = R_CASE + 0.18;
export const LEDGER = 12;
const LEDGER_STEP = 0.112;
const R_LEDGER = (R_BEZEL + R_CASE) / 2;
const JEWEL = 0.1;
const LANE = 0.17;
const BAND = 0.36;
const LAMP = 0.082;

export const BEAD_TINT = TOKENS.cueCyan;
export const BELL_TINT = TOKENS.cueMagenta;
export const GOLD = TOKENS.celebrationGold;
export const MISS = TOKENS.failureBlue;
export const LAMP_TINT = 0xffd9a0;
const BEAD_CORE = 0xe4fbff;
const FACE = 0x121a2a;
const BAND_TINT = 0x18223a;
const GROOVE = 0x05070c;
const ROSE = 0x1a2336;
const BRASS = 0xc9a45c;
const BRASS_DEEP = 0x5e4a26;
const IVORY = 0xe9e1cd;
const INK = 0x231e2a;
const FELT = 0x0d1017;
const LIGHT = vec2(-0.55, 0.83).normalize();

export function slots(count: number) {
  const values = new Array<number>(count).fill(0);
  return { node: uniformArray(values, 'float' as const), values };
}
export type Slots = ReturnType<typeof slots>;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 48).rotateX(-Math.PI / 2);
export const along = (angle: number, radius: number, out: THREE.Vector3) => out.set(radius * Math.sin(angle), 0, -radius * Math.cos(angle));

const board = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (size: number): N<'vec2'> => uv().sub(0.5).mul(size);
const toward = (angle: N<'float'>): N<'vec2'> => vec2(sin(angle), cos(angle));
const pixel = fwidth(positionWorld.x);
const aa = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel)), 0, 1);
const soft = (d: N<'float'>, blur: number): N<'float'> => smoothstep(-0.04, blur, d).oneMinus();
const nearest = (q: N<'vec2'>, count: number): N<'float'> => mod(floor(atan(q.x, q.y).div(TAU).mul(count).add(0.5)).add(count), count);
const facing = (p: N<'vec2'>): N<'float'> => dot(p.div(max(length(p), 1e-4)), LIGHT);

function sprite(geometry: THREE.BufferGeometry, order: number, color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function desk(lit: Lit): THREE.Mesh {
  const q = board();
  const felt = rgb(FELT).mul(noise(q.mul(2.7)).mul(0.1).add(noise(q.mul(19)).mul(0.05)).add(1));
  const caseShadow = soft(sdCircle(q.sub(vec2(0.14, -0.2)), R_CASE), 0.6).mul(0.72);
  const bellShadow = soft(sdCircle(q.sub(vec2(0.08, BELL_Y - 0.12)), BELL), 0.28).mul(0.6);
  const vignette = smoothstep(4.2, 12, length(q)).mul(0.55).oneMinus();
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(felt.mul(max(caseShadow, bellShadow).oneMinus()).mul(vignette));
  const mesh = new THREE.Mesh(flat(60, 40), material);
  mesh.renderOrder = 0;
  return mesh;
}

export function dial(lit: Lit, period: number, modulus: number, bead: number, sweep: Scalar): THREE.Mesh {
  const q = board(), r = length(q), light = facing(q), angle = mod(atan(q.x, q.y).add(TAU), TAU);
  const across = clamp(r.sub(R_BEZEL).div(R_CASE - R_BEZEL), 0, 1);
  const brass = rgb(BRASS).mul(cos(across.mul(Math.PI)).negate().mul(light).mul(0.4).add(0.82)).mul(sin(r.mul(310)).mul(0.025).add(1));

  const innerShade = smoothstep(float(R_BEZEL - 0.14).sub(max(light, 0).mul(0.22)), R_BEZEL, r).mul(0.6).oneMinus();
  let albedo: N<'vec3'> = rgb(FACE).mul(noise(q.mul(13)).mul(0.05).add(1)).mul(innerShade);

  albedo = mix(albedo, rgb(BAND_TINT).mul(noise(q.mul(5)).mul(0.05).add(1)), aa(abs(r.sub(R_SOCKET)).sub(BAND)));
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(abs(r.sub(R_SOCKET)).sub(BAND)).sub(0.012)));

  const groove = aa(abs(r.sub(R_LANE)).sub(LANE / 2));
  albedo = mix(albedo, rgb(GROOVE), groove);
  albedo = mix(albedo, rgb(0x3b4561), aa(abs(r.sub(R_LANE - LANE / 2)).sub(0.01)).mul(0.8));

  const s = angle.div(TAU).mul(period), k = floor(s.add(0.5)), arc = s.sub(k).mul(TAU / period).mul(r);
  const top = mod(k, period).lessThan(0.5);
  const tick = aa(sdBox(vec2(arc, r.sub(R_LANE + LANE / 2 + 0.15)), select(top, vec2(0.05, 0.1), vec2(0.016, 0.065)), 0.012));
  albedo = mix(albedo, select(top, rgb(BRASS), rgb(0x8a94ab)), tick);

  const cup = bead * 1.28, p = q.sub(toward(k.mul(TAU / period)).mul(R_SOCKET)), pd = length(p);
  albedo = mix(albedo, mix(rgb(0x04060b), rgb(0x141c2e), smoothstep(-1, 1, facing(p).negate()).mul(pd.div(cup))), aa(pd.sub(cup)));
  albedo = mix(albedo, rgb(0x55607a).mul(facing(p).negate().mul(0.35).add(0.75)), aa(abs(pd.sub(cup)).sub(0.014)));

  const guilloche = sin(angle.mul(modulus * 12).add(sin(r.mul(11)).mul(1.4))).mul(0.5).add(0.5);
  albedo = mix(albedo, rgb(ROSE).mul(guilloche.mul(0.12).add(0.94)), aa(r.sub(R_ROSE)));
  albedo = mix(albedo, rgb(BRASS).mul(light.mul(0.3).add(0.8)), aa(abs(r.sub(R_ROSE)).sub(0.035)));
  albedo = mix(albedo, rgb(0x0b1019), aa(abs(r.sub(R_JEWEL)).sub(0.011)));
  const jewel = q.sub(toward(nearest(q, modulus).mul(TAU / modulus)).mul(R_JEWEL)), jd = length(jewel);
  albedo = mix(albedo, rgb(BRASS).mul(facing(jewel).mul(0.35).add(0.8)), aa(abs(jd.sub(JEWEL + 0.03)).sub(0.026)));
  albedo = mix(albedo, mix(rgb(0x06121a), rgb(0x2b4a5a), smoothstep(-0.05, 0.1, dot(jewel, LIGHT))), aa(jd.sub(JEWEL)));
  const hub = aa(r.sub(0.24));
  albedo = mix(albedo, rgb(BRASS).mul(pow(clamp(r.div(0.24).oneMinus(), 0, 1), 0.5).mul(0.3).add(light.mul(0.25)).add(0.65)), hub);
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(sdBox(q, vec2(0.13, 0.018), 0.01)).mul(hub));

  albedo = mix(albedo, brass, aa(float(R_BEZEL).sub(r)));
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(r.sub(R_BEZEL)).sub(0.014)).mul(0.7));
  const fill = clamp(sweep.sub(angle).mul(r).div(pixel).add(0.5), 0, 1).mul(groove);
  const trace = aa(abs(r.sub(R_LANE)).sub(0.02)).mul(0.9).add(groove.mul(0.1));
  return sprite(flat(2 * R_CASE + 0.2, 2 * R_CASE + 0.2), 1, lit(albedo).add(rgb(LAMP_TINT).mul(fill.mul(trace))), aa(r.sub(R_CASE)));
}

function pearl(p: N<'vec2'>, radius: number, presence: N<'float'>, gain: N<'float'> | number): N<'vec3'> {
  const size = float(radius).mul(presence.mul(0.7).add(0.3)), d = length(p).sub(size), u = length(p).div(size);
  const shade = mix(rgb(BEAD_CORE), rgb(BEAD_TINT), smoothstep(0.05, 1, u)).mul(u.mul(-0.35).add(1.35));
  const spec = exp(length(p.div(size).sub(vec2(-0.34, 0.4))).mul(-6)).mul(1.1);
  const halo = exp(max(d, 0).mul(-40)).mul(0.4);
  return shade.add(spec).mul(aa(d)).add(rgb(BEAD_TINT).mul(halo)).mul(presence).mul(gain);
}

export interface Beads {
  mesh: THREE.Mesh;
  presence: Slots;
}

export function beads(period: number, bead: number, drop: { at: Point; presence: Scalar; gain: Scalar }, arrive: { at: Point; presence: Scalar }): Beads {
  const presence = slots(period), q = board(), k = nearest(q, period);
  const inner = pearl(q.sub(toward(k.mul(TAU / period)).mul(R_SOCKET)), bead, presence.node.element(int(k)), 0.72);
  const color = inner.add(pearl(q.sub(drop.at), bead, drop.presence, drop.gain)).add(pearl(q.sub(arrive.at), bead, arrive.presence, 1.3));
  const geometry = new THREE.RingGeometry(R_SOCKET - 0.5, R_LANE + 0.5, 192, 1).rotateX(-Math.PI / 2);
  return { mesh: sprite(geometry, 2, color, float(1), true), presence };
}

export function glowBank(count: number, radius: number, tint: number, gain: number): THREE.InstancedMesh {
  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  material.colorNode = rgb(tint).mul(gain);
  const mesh = instancedMesh(disc(radius), material, count);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  mesh.layers.set(LIGHT_LAYER);
  mesh.frustumCulled = false;
  return mesh;
}

function pip(p: N<'vec2'>, count: number): N<'float'> {
  if (count === 0) return abs(length(p).sub(0.12)).sub(0.028);
  if (count === 1) return length(p).sub(0.078);
  const spots = count === 2 ? [[0.12, 0.12], [-0.12, -0.12]] : count === 3 ? [[0.15, 0.15], [0, 0], [-0.15, -0.15]]
    : Array.from({ length: count }, (_, i) => [0.19 * Math.sin((TAU * i) / count), 0.19 * Math.cos((TAU * i) / count)]);
  const radius = count > 3 ? 0.052 : 0.068;
  return spots.map(([x, y]) => length(p.sub(vec2(x!, y!))).sub(radius)).reduce((a, b) => min(a, b));
}

export interface Keys {
  body: THREE.Mesh;
  glow: THREE.Mesh;
  press: Slots;
  halo: Slots;
  gold: Slots;
  blue: Slots;
}

export function keys(lit: Lit, modulus: number, accent: number): Keys {
  const press = slots(modulus), halo = slots(modulus), gold = slots(modulus), blue = slots(modulus), q = board();
  const j = nearest(q, modulus), p = q.sub(toward(j.mul(TAU / modulus)).mul(R_KEY)), down = press.node.element(int(j));
  const radius = float(KEY).mul(down.mul(-0.05).add(1)), d = length(p).sub(radius), u = length(p).div(radius), light = facing(p);
  const shadow = soft(length(p.sub(vec2(0.06, -0.085).mul(down.mul(-0.65).add(1)))).sub(radius), 0.14).mul(0.62);
  const bevel = smoothstep(0.7, 0.97, u), dish = smoothstep(0, 0.7, u).oneMinus();
  const ivory = rgb(IVORY).mul(bevel.mul(light).mul(0.32).sub(dish.mul(light).mul(0.1)).add(0.9)).mul(down.mul(-0.12).add(1));
  let pips: N<'float'> = pip(p, modulus - 1);
  for (let k = modulus - 2; k >= 0; k--) pips = select(j.lessThan(k + 0.5), pip(p, k), pips);
  let albedo: N<'vec3'> = mix(ivory, rgb(accent).mul(1.05), down.mul(0.42));
  albedo = mix(albedo, rgb(GOLD), gold.node.element(int(j)).mul(0.35));
  albedo = mix(albedo, rgb(0x6f86b0).mul(0.8), blue.node.element(int(j)).mul(0.55));
  albedo = mix(albedo, rgb(INK), aa(pips));
  albedo = mix(albedo, rgb(0x2e2936), aa(abs(d.add(0.016)).sub(0.012)).mul(0.75));
  const cap = aa(d);
  const body = sprite(disc(R_ROSE - 0.02), 3, mix(rgb(0x000000), lit(albedo), cap), max(cap, shadow));

  let glow: N<'vec3'> = rgb(0x000000);
  for (let k = 0; k < modulus; k++) {
    const a = (TAU * k) / modulus, rk = length(q.sub(vec2(R_KEY * Math.sin(a), R_KEY * Math.cos(a)))), out = max(rk.sub(KEY), 0), inside = aa(rk.sub(KEY)), outside = inside.oneMinus();
    const ring = aa(abs(rk.sub(KEY + 0.15)).sub(0.017)).add(exp(abs(rk.sub(KEY + 0.15)).mul(-26)).mul(0.25));
    glow = glow
      .add(rgb(accent).mul(press.node.element(k)).mul(exp(out.mul(-12)).mul(0.35).mul(outside)))
      .add(rgb(GOLD).mul(gold.node.element(k)).mul(exp(out.mul(-7)).mul(0.9).mul(outside).add(inside.mul(0.1))))
      .add(rgb(MISS).mul(blue.node.element(k)).mul(exp(out.mul(-10)).mul(0.25).mul(outside).add(aa(abs(rk.sub(KEY + 0.03)).sub(0.014)).mul(0.6))))
      .add(rgb(BELL_TINT).mul(halo.node.element(k)).mul(ring).mul(1.3));
  }
  return { body, glow: sprite(disc(R_ROSE - 0.02), 4, glow.mul(smoothstep(R_ROSE - 0.3, R_ROSE - 0.04, length(q)).oneMinus()), float(1), true), press, halo, gold, blue };
}

export function pilot(alpha: Scalar): THREE.Mesh {
  const r = length(local(0.9)), fence = smoothstep(0.34, 0.44, r).oneMinus();
  return sprite(flat(0.9, 0.9), 5, rgb(BEAD_CORE).mul(aa(r.sub(JEWEL * 0.78))).mul(1.5).add(rgb(BEAD_TINT).mul(exp(r.mul(-11)).mul(1.1))).mul(alpha).mul(fence), float(1), true);
}

export function loupe(lit: Lit): THREE.Group {
  const p = local(1), r = length(p), light = facing(p);
  const shell = max(aa(abs(r.sub(0.3)).sub(0.035)), aa(sdSegment(p, vec2(0, -0.34), vec2(0, -0.43), 0.022)));
  const metal = rgb(BRASS).mul(select(r.lessThan(0.3), light.negate(), light).mul(0.35).add(0.85));
  const lens = aa(r.sub(0.27)), glass = lens.mul(exp(length(p.sub(vec2(-0.1, 0.12))).mul(-14)).mul(0.25).add(0.14));
  const group = new THREE.Group();
  group.add(
    sprite(flat(1, 1), 5, rgb(LAMP_TINT).mul(exp(r.mul(-7)).mul(0.5).mul(aa(r.sub(0.3)))), float(1), true),
    sprite(flat(1, 1), 6, mix(rgb(0xfff4dc), lit(metal), shell), max(shell, glass)),
  );
  return group;
}

export function bell(lit: Lit, ring: Scalar, wave: Scalar): THREE.Group {
  const p = local(1.2), r = length(p), light = facing(p);
  const bracket = aa(sdBox(p.sub(vec2(0, -0.34)), vec2(0.07, 0.13), 0.03));
  const dome = aa(r.sub(BELL));
  const sheen = exp(length(p.div(BELL).sub(vec2(-0.38, 0.4))).mul(-3.2)).mul(0.7);
  const metal = rgb(BRASS).mul(pow(clamp(r.div(BELL).oneMinus(), 0, 1), 0.35).mul(0.45).add(0.5).add(sheen)).mul(light.mul(0.12).add(1));
  let albedo: N<'vec3'> = mix(rgb(BRASS).mul(light.mul(0.3).add(0.75)), metal, dome);
  albedo = mix(albedo, rgb(BRASS_DEEP), aa(abs(r.sub(BELL - 0.04)).sub(0.014)).mul(dome));
  albedo = mix(albedo, rgb(BRASS).mul(1.25), aa(r.sub(0.07)));
  const g = length(local(2.8)), fence = smoothstep(1.15, 1.38, g).oneMinus();
  const halo = exp(max(g.sub(BELL), 0).mul(-12)).mul(0.45).add(aa(g.sub(BELL)).mul(0.16)).mul(ring);
  const crest = (lag: number, width: number) => {
    const w = clamp(wave.sub(lag).div(1 - lag), 0, 1);
    return aa(abs(g.sub(w.mul(0.92).add(BELL + 0.04))).sub(width)).mul(w.oneMinus().mul(w.oneMinus())).mul(step(1e-3, w));
  };
  const waves = crest(0, 0.022).add(crest(0.3, 0.016).mul(0.7)).mul(0.9);
  const group = new THREE.Group();
  group.add(
    sprite(flat(1.2, 1.2), 2, lit(mix(albedo, rgb(BELL_TINT), ring.mul(0.3).mul(dome))), max(dome, bracket)),
    sprite(flat(2.8, 2.8), 3, rgb(BELL_TINT).mul(halo.add(waves)).mul(fence), float(1), true),
  );
  return group;
}

export function ledger(lit: Lit, values: Slots): THREE.Mesh {
  const q = board(), angle = mod(atan(q.x, q.y).add(TAU), TAU);
  const slot = clamp(floor(float(LEDGER / 2).sub(angle.sub(Math.PI).div(LEDGER_STEP))), 0, LEDGER - 1);
  const p = q.sub(toward(float(Math.PI).add(float(LEDGER / 2 - 0.5).sub(slot).mul(LEDGER_STEP))).mul(R_LEDGER)), rd = length(p), value = values.node.element(int(slot));
  const rim = aa(abs(rd.sub(LAMP + 0.018)).sub(0.014)), socket = aa(rd.sub(LAMP + 0.032));
  const albedo = mix(rgb(0x05070b), rgb(BRASS_DEEP).mul(facing(p).negate().mul(0.4).add(0.8)), rim);
  const shine = exp(length(p.sub(vec2(-0.025, 0.03))).mul(-34)).mul(0.6).add(1);
  const lamp = rgb(GOLD).mul(max(value, 0)).mul(1.15).add(rgb(0x4d6690).mul(max(value.negate(), 0)).mul(0.75)).mul(aa(rd.sub(LAMP))).mul(shine);
  const ring = new THREE.RingGeometry(R_BEZEL - 0.02, R_CASE + 0.02, 160, 1).rotateX(-Math.PI / 2);
  return sprite(ring, 2, lit(albedo).add(lamp), socket);
}
