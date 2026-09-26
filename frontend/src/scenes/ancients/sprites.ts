import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, length, max, min, mix, mod, pow, select, sin, smoothstep, uv, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { FLAG_ATTACK, FLAG_FORTIFY, FLAG_FRENZY, FLAG_SLOW, FLAG_STUN, FLAG_TAUNT, FLAG_WARD } from '@sim/envs/ancients';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Shade = (channel: (name: string) => N<'vec4'>) => { color: N<'vec3'>; opacity: N<'float'> };

export const JADE = 0x3fd49a;
export const JADE_DEEP = 0x0f2c22;
export const JADE_INK = 0xd4fbe8;
export const CRIMSON = 0xf2584f;
export const CRIMSON_DEEP = 0x34100f;
export const CRIMSON_INK = 0xffdcd4;
export const AMBER = 0xe0a646;
export const STUN = 0xffe066;
export const SLOW = 0x86d8ff;
export const FRENZY = 0xff7a2e;
export const FORTIFY = 0xc9bfae;
export const WARD = 0xfff0c2;
export const TAUNT = 0xff3b3b;
export const MANA = 0x5b9dff;

export const HERO_RADIUS = 0.84;
export const HERO_SPAN = 1.62;
export const CREEP_RADIUS = 0.25;
export const TOWER_RADIUS = 0.9;
export const STRUCTURE_SPAN = 3;
export const ANCIENT_RADIUS = 1.5;

const TAU = Math.PI * 2;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const rotate = (p: N<'vec2'>, angle: number) => vec2(p.x.mul(Math.cos(angle)).sub(p.y.mul(Math.sin(angle))), p.x.mul(Math.sin(angle)).add(p.y.mul(Math.cos(angle))));
const has = (flags: N<'float'>, flag: number) => mod(floor(flags.div(flag)), 2);
const pick = (index: N<'float'>, values: N<'float'>[]): N<'float'> => values.slice(0, -1).reduceRight<N<'float'>>((rest, value, i) => select(index.lessThan(i + 0.5), value, rest), values.at(-1)!);
const team = (side: N<'float'>, ours: number, theirs: number) => mix(rgb(ours), rgb(theirs), side);
const pick3 = (index: N<'float'>, a: number, b: number, c: number) => select(index.lessThan(0.5), rgb(a), select(index.lessThan(1.5), rgb(b), rgb(c)));
const ring = (r: N<'float'>, radius: N<'float'> | number, width: N<'float'> | number) => abs(r.sub(radius)).sub(width);
const grey = (color: N<'vec3'>) => vec3(color.x.mul(0.3).add(color.y.mul(0.5)).add(color.z.mul(0.2)));
const shadowed = (color: N<'vec3'>, opacity: N<'float'>, shadow: N<'float'>) => {
  const alpha = opacity.add(shadow.mul(float(1).sub(opacity)));
  return { color: color.mul(opacity).div(max(alpha, 1e-4)), opacity: alpha };
};

export class SpriteBatch {
  readonly mesh: THREE.InstancedMesh;
  private readonly data = new Map<string, Float32Array>();
  private readonly attributes: THREE.InstancedBufferAttribute[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly rotation = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 1, 0);
  private used = 0;

  constructor(name: string, readonly capacity: number, width: number, depth: number, channels: readonly string[], shade: Shade, order: number, additive = false) {
    const geometry = flat(width, depth);
    for (const channel of channels) {
      const array = new Float32Array(capacity * 4), attribute = new THREE.InstancedBufferAttribute(array, 4);
      attribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(channel, attribute);
      this.data.set(channel, array);
      this.attributes.push(attribute);
    }
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    if (additive) material.blending = THREE.AdditiveBlending;
    const { color, opacity } = shade((channel) => attribute<'vec4'>(channel, 'vec4'));
    material.colorNode = color;
    material.opacityNode = opacity;
    this.mesh = instancedMesh(geometry, material, capacity);
    this.mesh.name = name;
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  add(x: number, z: number, angle = 0, sx = 1, sz = sx): number {
    if (this.used >= this.capacity) throw new Error(`${this.mesh.name} holds ${this.capacity} sprites`);
    this.rotation.setFromAxisAngle(this.up, angle);
    this.matrix.compose(this.position.set(x, 0, z), this.rotation, this.scale.set(sx, 1, sz));
    this.mesh.setMatrixAt(this.used, this.matrix);
    return this.used++;
  }

  set(channel: string, index: number, a: number, b = 0, c = 0, d = 0): void {
    const array = this.data.get(channel)!, at = index * 4;
    array[at] = a;
    array[at + 1] = b;
    array[at + 2] = c;
    array[at + 3] = d;
  }

  commit(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const attribute of this.attributes) attribute.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.removeFromParent();
  }
}

const glyphs = (p: N<'vec2'>): N<'float'>[] => {
  const sword = rotate(p, -Math.PI / 4);
  const blade = min(min(sdSegment(sword, vec2(0, -0.28), vec2(0, 0.66), 0.09), sdBox(sword.sub(vec2(0, -0.28)), vec2(0.3, 0.065), 0.03)), min(sdSegment(sword, vec2(0, -0.3), vec2(0, -0.56), 0.06), sdCircle(sword.sub(vec2(0, -0.62)), 0.1)));
  const angle = atan(p.y, p.x), spark = length(p).sub(float(0.2).add(pow(abs(cos(angle.mul(2))), 10).mul(0.52)));
  const bolt = min(min(sdSegment(p, vec2(0.24, 0.66), vec2(-0.2, 0.04), 0.1), sdSegment(p, vec2(-0.2, 0.04), vec2(0.2, 0.02), 0.1)), sdSegment(p, vec2(0.2, 0.02), vec2(-0.24, -0.66), 0.1));
  const lens = p.add(vec2(0, 0.08));
  const shield = max(max(sdCircle(lens.sub(vec2(0.5, 0.36)), 1), sdCircle(lens.add(vec2(0.5, -0.36)), 1)), lens.y.sub(0.56));
  const cross = min(sdBox(p, vec2(0.16, 0.56), 0.05), sdBox(p, vec2(0.56, 0.16), 0.05));
  const crescent = max(sdCircle(p, 0.6), sdCircle(p.sub(vec2(0.28, 0.2)), 0.5).negate());
  return [blade, spark, bolt, shield, cross, crescent];
};

export function heroShade(clock: N<'float'>, accent: number): Shade {
  return (channel) => {
    const hero = channel('aHero'), state = channel('aState');
    const side = hero.x, arch = hero.y, agent = hero.z, alpha = hero.w, flags = state.x, veil = state.y, spirit = state.z, flare = state.w;
    const p = uv().sub(0.5).mul(2 * HERO_SPAN), r = length(p), angle = atan(p.y, p.x);
    const body = coverage(r.sub(1));
    const rimColor = mix(team(side, JADE, CRIMSON), rgb(accent).mul(1.35), agent).mul(float(1).add(has(flags, FLAG_ATTACK).mul(0.5)).add(flare));
    const dashed = select(veil.greaterThan(0.5), smoothstep(0.35, 0.65, abs(fract(angle.mul(8 / TAU)).sub(0.5)).mul(2)), float(1));
    const rim = coverage(ring(r, 0.9, 0.1)).mul(dashed);
    const mark = coverage(pick(arch, glyphs(p.div(0.74))).mul(0.74));
    let color: N<'vec3'> = team(side, JADE_DEEP, CRIMSON_DEEP);
    color = mix(color, mix(team(side, JADE_INK, CRIMSON_INK), rgb(0xffe7ae), agent), mark);
    color = mix(color, rimColor, rim);
    let opacity: N<'float'> = body;
    const halo = coverage(ring(r, 1.16, 0.055)).mul(agent);
    color = mix(color, rgb(accent).mul(1.6), halo.mul(float(1).sub(body)));
    opacity = max(opacity, halo);
    const live = float(1).sub(spirit);
    const orbit = (k: number) => sdCircle(p.sub(vec2(cos(clock.mul(4).add(k * TAU / 3)), sin(clock.mul(4).add(k * TAU / 3))).mul(1.3)), 0.14);
    const layers: [N<'float'>, number][] = [
      [coverage(ring(r, 1.3, 0.05)).mul(smoothstep(0.3, 0.7, fract(angle.mul(10 / TAU).add(clock.mul(0.6))))).mul(has(flags, FLAG_SLOW)), SLOW],
      [coverage(ring(r, float(1.3).add(sin(angle.mul(7).add(clock.mul(11))).mul(0.07)), 0.08)).mul(has(flags, FLAG_FRENZY)), FRENZY],
      [coverage(abs(max(abs(p.x).mul(0.866).add(abs(p.y).mul(0.5)), abs(p.y)).sub(1.2)).sub(0.06)).mul(has(flags, FLAG_FORTIFY)), FORTIFY],
      [coverage(sdCircle(vec2(fract(angle.mul(12 / TAU).add(clock.mul(0.25))).sub(0.5).mul(0.55), r.sub(1.46)), 0.075)).mul(has(flags, FLAG_WARD)), WARD],
      [coverage(ring(r, 1.22, float(0.035).add(sin(clock.mul(14)).mul(0.015)))).mul(has(flags, FLAG_TAUNT)), TAUNT],
      [coverage(min(min(orbit(0), orbit(1)), orbit(2))).mul(has(flags, FLAG_STUN)), STUN],
    ];
    for (const [cover, tint] of layers) {
      const outside = cover.mul(live).mul(float(1).sub(body));
      color = mix(color, rgb(tint).mul(1.5), outside);
      opacity = max(opacity, outside);
    }
    const keys = channel('aKeys'), shown = select(keys.x.greaterThanEqual(0), float(1), float(0));
    const pip = (k: number, level: N<'float'>, tint: N<'vec3'>) => {
      const angle = -Math.PI / 2 + (k - 1) * 0.42, at = vec2(Math.cos(angle) * 1.3, Math.sin(angle) * 1.3);
      const cover = coverage(sdCircle(p.sub(at), 0.13)).mul(shown);
      color = mix(color, mix(rgb(0x2a2824), tint.mul(1.8), level), cover);
      opacity = max(opacity, cover);
    };
    pip(0, keys.x, pick3(keys.y, 0x8ff0c8, 0xff7a6a, 0xffd166));
    pip(1, keys.z, rgb(0x7fb2ff));
    pip(2, keys.w, rgb(0xd08cff));
    color = mix(color, grey(color).mul(0.8), max(spirit, veil.mul(0.6)));
    const out = shadowed(color, opacity, smoothstep(1.25, 0.7, length(p.sub(vec2(0.14, -0.18)))).mul(0.5).mul(live).mul(float(1).sub(veil)));
    return { color: out.color, opacity: out.opacity.mul(alpha) };
  };
}

const SEGMENTS: [number, number, number, number][] = [[-0.3, 0.5, 0.3, 0.5], [0.3, 0.5, 0.3, 0], [0.3, 0, 0.3, -0.5], [-0.3, -0.5, 0.3, -0.5], [-0.3, -0.5, -0.3, 0], [-0.3, 0, -0.3, 0.5], [-0.3, 0, 0.3, 0]];
export const DIGIT_MASKS = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f] as const;

const digit = (p: N<'vec2'>, mask: N<'float'>): N<'float'> => {
  let d: N<'float'> = float(1e3);
  SEGMENTS.forEach(([ax, ay, bx, by], k) => {
    d = min(d, select(has(mask, 2 ** k).greaterThan(0.5), sdSegment(p, vec2(ax * 0.8, ay * 0.82), vec2(bx * 0.8, by * 0.82), 0.1), float(1e3)));
  });
  return d;
};

export const BADGE_WIDTH = 2.1;
const NOTCH = 250;
export const BADGE_DEPTH = 0.6;

export function badgeShade(accent: number): Shade {
  return (channel) => {
    const bar = channel('aBar'), badge = channel('aBadge');
    const hp = bar.x, maxHp = bar.y, mana = bar.z, side = bar.w, tens = badge.x, ones = badge.y, agent = badge.z, alpha = badge.w;
    const p = uv().sub(0.5).mul(vec2(BADGE_WIDTH, BADGE_DEPTH));
    const left = -0.52, right = 0.98, top = 0.1, low = -0.02;
    const frame = coverage(sdBox(p.sub(vec2((left + right) / 2, (top + low) / 2)), vec2((right - left) / 2 + 0.035, (top - low) / 2 + 0.035), 0.03));
    const inner = sdBox(p.sub(vec2((left + right) / 2, (top + low) / 2)), vec2((right - left) / 2, (top - low) / 2), 0.015);
    const fill = coverage(inner).mul(coverage(p.x.sub(hp.div(max(maxHp, 1)).mul(right - left).add(left))));
    const tick = abs(fract(p.x.sub(left).div(right - left).mul(maxHp).div(NOTCH).add(0.5)).sub(0.5)).mul(NOTCH * (right - left)).div(max(maxHp, 1));
    const notch = coverage(tick.sub(0.008)).mul(fill);
    const manaBox = sdBox(p.sub(vec2((left + right) / 2, low - 0.075)), vec2((right - left) / 2, 0.028), 0.01);
    const hasMana = select(mana.greaterThanEqual(0), float(1), float(0));
    const manaFrame = coverage(manaBox.sub(0.03)).mul(hasMana);
    const manaFill = coverage(manaBox).mul(hasMana).mul(coverage(p.x.sub(clamp(mana, 0, 1).mul(right - left).add(left))));
    const disc = sdCircle(p.sub(vec2(-0.76, 0.02)), 0.22);
    const glyphs = min(digit(p.sub(vec2(-0.76 - 0.075, 0.02)).div(0.17), tens), digit(p.sub(vec2(-0.76 + 0.075, 0.02)).div(0.17), ones)).mul(0.17);
    const single = digit(p.sub(vec2(-0.76, 0.02)).div(0.17), ones).mul(0.17);
    const text = coverage(select(tens.greaterThan(0.5), glyphs, single));
    const hpColor = mix(team(side, 0x5fe38f, 0xef5b52), rgb(0xffd479), agent);
    let color: N<'vec3'> = rgb(0x07090c);
    let opacity: N<'float'> = max(max(frame, manaFrame), coverage(disc.sub(0.03)));
    color = mix(color, hpColor.mul(0.28), coverage(inner));
    color = mix(color, hpColor, fill);
    color = mix(color, hpColor.mul(0.35), notch);
    color = mix(color, rgb(MANA), manaFill);
    color = mix(color, mix(team(side, JADE, CRIMSON), rgb(accent), agent).mul(0.9), coverage(ring(length(p.sub(vec2(-0.76, 0.02))), 0.215, 0.028)));
    color = mix(color, rgb(0xf4f1e6), text);
    opacity = max(opacity, text);
    return { color, opacity: opacity.mul(alpha) };
  };
}

export const PLATE_WIDTH = 1.9;
export const PLATE_DEPTH = 0.26;

export const plateShade: Shade = (channel) => {
  const bar = channel('aPlate');
  const fraction = bar.x, side = bar.y, half = bar.z, alpha = bar.w;
  const p = uv().sub(0.5).mul(vec2(PLATE_WIDTH, PLATE_DEPTH));
  const inner = sdBox(p, vec2(half, 0.055), 0.015);
  const frame = coverage(inner.sub(0.035));
  const fill = coverage(inner).mul(coverage(p.x.sub(half.negate().add(fraction.mul(half).mul(2)))));
  const color = mix(mix(rgb(0x07090c), team(side, 0x5fe38f, 0xef5b52).mul(0.25), coverage(inner)), team(side, 0x6fe7a0, 0xf06a5f), fill);
  return { color, opacity: frame.mul(alpha) };
};

export const CREEP_SPAN = 2.1;

export const creepShade: Shade = (channel) => {
  const creep = channel('aCreep');
  const side = creep.x, flash = creep.y, alpha = creep.z, neutral = creep.w;
  const p = uv().sub(0.5).mul(2 * CREEP_SPAN);
  const form = (at: N<'vec2'>) => {
    const q = vec2(abs(at.x).mul(0.72), abs(at.y));
    const nose = max(q.x.add(q.y).sub(0.92).mul(0.62), at.x.negate().sub(0.7));
    const beast = min(sdCircle(at, 0.86), min(sdCircle(at.sub(vec2(0.54, 0.66)), 0.3), sdCircle(at.sub(vec2(-0.54, 0.66)), 0.3)));
    return mix(nose, beast, neutral);
  };
  const shape = form(p), cover = coverage(shape);
  const edge = coverage(abs(shape.add(0.12)).sub(0.1));
  const hue = mix(team(side, JADE, CRIMSON), rgb(AMBER), neutral);
  const deep = mix(team(side, 0x1f5a45, 0x6a1c1a), rgb(0x4a3312), neutral);
  const eyes = coverage(min(sdCircle(p.sub(vec2(0.3, 0.14)), 0.14), sdCircle(p.sub(vec2(-0.3, 0.14)), 0.14))).mul(neutral);
  let color: N<'vec3'> = mix(deep, hue, edge);
  color = mix(color, mix(rgb(0xffe9b0), rgb(0xff4a3a), side), eyes);
  color = mix(color, rgb(0xffffff), flash);
  const out = shadowed(color, cover, smoothstep(0.35, -0.25, form(p.sub(vec2(0.16, -0.2)))).mul(0.5));
  return { color: out.color, opacity: out.opacity.mul(alpha) };
};

export function structureShade(clock: N<'float'>, lit: Lit): Shade {
  return (channel) => {
    const tower = channel('aTower'), fx = channel('aFx');
    const side = tower.x, ancient = tower.y, health = tower.z, standing = tower.w, hit = fx.x;
    const p = uv().sub(0.5).mul(STRUCTURE_SPAN), r = length(p), angle = atan(p.y, p.x);
    const hue = team(side, JADE, CRIMSON);
    const square = sdBox(p, vec2(0.8), 0.14);
    const octagon = max(max(abs(p.x), abs(p.y)), abs(p.x).add(abs(p.y)).mul(0.7071)).sub(0.96);
    const base = mix(square, octagon, ancient);
    const tier = mix(sdBox(p, vec2(0.56), 0.1), max(max(abs(p.x), abs(p.y)), abs(p.x).add(abs(p.y)).mul(0.7071)).sub(0.7), ancient);
    const facing = p.x.negate().add(p.y).mul(0.7071).div(max(r, 0.05));
    const bevel = smoothstep(-0.2, -0.02, base);
    const grain = sin(p.x.mul(31).add(sin(p.y.mul(17)).mul(2))).mul(0.04).add(1);
    const cracks = smoothstep(0.03, 0.0, abs(fract(angle.mul(5 / TAU).add(r.mul(1.7))).sub(0.5)).sub(0.45)).mul(smoothstep(0.75, 0.25, health));
    let stone: N<'vec3'> = mix(rgb(0x8f887d), rgb(0xb3ab9c), smoothstep(0, -0.4, tier)).mul(grain);
    stone = mix(stone, mix(rgb(0x4e4a44), rgb(0xd8d0bf), facing.mul(0.5).add(0.5)), bevel.mul(0.8));
    stone = mix(stone, mix(rgb(0x5a554d), rgb(0xc8c0af), facing.mul(0.5).add(0.5)), coverage(abs(tier).sub(0.035)).mul(0.7));
    stone = stone.mul(float(1).sub(cracks.mul(0.55)));
    const post = (x: number, y: number) => sdCircle(p.sub(vec2(x, y)), 0.12);
    const posts = mix(min(min(post(0.6, 0.6), post(-0.6, 0.6)), min(post(0.6, -0.6), post(-0.6, -0.6))), min(min(post(0.62, 0.62), post(-0.62, 0.62)), min(post(0.62, -0.62), post(-0.62, -0.62))).sub(0.06), ancient);
    const socket = r.sub(mix(float(0.42), float(0.56), ancient));
    const gem = mix(abs(p.x).add(abs(p.y)).sub(0.34).mul(0.7071), max(abs(p.x).mul(0.866).add(abs(p.y).mul(0.5)), abs(p.y)).sub(0.42), ancient);
    const facet = select(p.y.sub(p.x).greaterThan(0), float(1.25), float(0.8));
    const pulse = float(0.85).add(sin(clock.mul(2.2).add(side.mul(1.7))).mul(0.15));
    let color: N<'vec3'> = lit(stone);
    color = mix(color, lit(mix(rgb(0x6d675e), rgb(0xe0d8c8), facing.mul(0.5).add(0.5))), coverage(posts));
    color = mix(color, rgb(0x0c0b0a), coverage(socket));
    color = mix(color, hue.mul(0.5), coverage(ring(r, mix(float(0.42), float(0.56), ancient), 0.03)));
    color = mix(color, hue.mul(pulse.mul(facet).mul(2.4)).add(hit.mul(2.5)), coverage(gem));
    color = mix(color, rgb(0x14120f), coverage(abs(base).sub(0.03)));
    const shards = abs(fract(angle.mul(7 / TAU).add(r.mul(0.9))).sub(0.5)).sub(0.28).negate();
    const rubbleMask = coverage(max(base.add(0.1), shards));
    const rubble = lit(mix(rgb(0x4a4640), rgb(0x8a8378), smoothstep(0.7, 0.1, r)));
    const scorch = smoothstep(1.05, 0.4, r).mul(0.6);
    color = mix(mix(rgb(0x0d0b09), rubble, rubbleMask), color, standing);
    const body = mix(max(rubbleMask, scorch), coverage(base.sub(0.02)), standing);
    const q = p.sub(vec2(0.14, -0.18));
    const cast = smoothstep(0.32, -0.12, mix(sdBox(q, vec2(0.8), 0.14), max(max(abs(q.x), abs(q.y)), abs(q.x).add(abs(q.y)).mul(0.7071)).sub(0.96), ancient)).mul(0.55).mul(standing);
    return shadowed(color, body, cast);
  };
}

export function beamShade(clock: N<'float'>): Shade {
  return (channel) => {
    const beam = channel('aBeam'), span = channel('aSpan');
    const tint = vec3(beam.x, beam.y, beam.z), style = beam.w, extent = span.x, width = span.y, phase = span.z, alpha = span.w;
    const along = uv().x.mul(extent), across = uv().y.sub(0.5).mul(width);
    const wobble = sin(along.mul(9).add(clock.mul(23)).add(phase)).mul(0.35).add(sin(along.mul(23).sub(clock.mul(41))).mul(0.2));
    const offset = select(style.greaterThan(2.5), wobble.mul(width).mul(0.3), float(0));
    const core = exp(abs(across.sub(offset)).div(width).mul(-9));
    const flow = fract(along.mul(1.6).sub(clock.mul(3)).add(phase));
    const dashes = select(style.lessThan(0.5), mix(float(0.55), float(1.4), smoothstep(0.5, 0.9, flow).mul(smoothstep(1, 0.9, flow))), float(1));
    const dots = select(style.greaterThan(0.5).and(style.lessThan(1.5)), smoothstep(0.5, 0.2, abs(fract(along.mul(2.6)).sub(0.5))), float(1));
    const head = uv().x.sub(1).mul(extent).add(0.34);
    const arrow = select(style.greaterThan(1.5).and(style.lessThan(2.5)), coverage(max(abs(across).mul(1.5).add(head.mul(0.9)).sub(0.32), head.negate())), float(0));
    const shaft = select(style.greaterThan(1.5).and(style.lessThan(2.5)), core.mul(smoothstep(0.02, 0.0, head)), core);
    const fade = smoothstep(0, 0.12, uv().x.mul(extent)).mul(max(smoothstep(0, 0.08, float(1).sub(uv().x).mul(extent)), arrow));
    const intensity = max(shaft.mul(dashes).mul(dots).mul(fade), arrow);
    return { color: tint.mul(intensity), opacity: intensity.mul(alpha) };
  };
}

export const effectShade: Shade = (channel) => {
  const fx = channel('aFx'), life = channel('aLife');
  const tint = vec3(fx.x, fx.y, fx.z), kind = fx.w, t = life.x, alpha = life.y;
  const p = uv().sub(0.5).mul(2), r = length(p), angle = atan(p.y, p.x);
  const flash = exp(r.mul(r).mul(-9)).mul(float(1).sub(t).mul(float(1).sub(t)));
  const wave = exp(abs(r.sub(t.mul(0.85).add(0.1))).mul(-26)).mul(float(1).sub(t));
  const bolt = exp(r.mul(r).mul(-26)).mul(1.6).add(exp(r.mul(-7)).mul(0.35));
  const coin = coverage(r.sub(0.34)).mul(mix(float(0.75), float(1.25), smoothstep(0.1, 0.3, abs(p.x.add(p.y).mul(0.7).sub(0.05))))).mul(float(1).sub(smoothstep(0.6, 1, t)));
  const burst = smoothstep(0.35, 0.0, abs(fract(angle.mul(8 / TAU)).sub(0.5))).mul(smoothstep(t.mul(0.6).add(0.3), t.mul(0.6), r)).mul(smoothstep(t.mul(0.45), t.mul(0.45).add(0.1), r)).mul(float(1).sub(t));
  const mark = exp(abs(r.sub(0.42)).mul(-38)).mul(smoothstep(0.12, 0.22, abs(fract(angle.mul(9 / TAU)).sub(0.5)))).add(exp(r.mul(r).mul(-60)).mul(0.6));
  const intensity = pick(kind, [flash, wave, bolt, coin, burst, mark]);
  return { color: tint.mul(intensity), opacity: clamp(intensity, 0, 1).mul(alpha) };
};

