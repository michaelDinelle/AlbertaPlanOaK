import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, float, fract, length, max, min, mix, select, sin, smoothstep, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { crown, ellipse, flame, Layers, ring, shield, smin, turn, type Lit, type N } from './art';

export const MONSTER_QUAD = 2.1;
export const HERO_QUAD = 1.5;

export const PLANS = ['walker', 'runner', 'flier', 'caster', 'growth'] as const;
export type Plan = (typeof PLANS)[number];

type Channel = (name: string) => N<'vec4'>;

const ink = rgb(0x0c0a0c);

function shadowOf(p: N<'vec2'>, rx: N<'float'> | number, ry: N<'float'> | number): N<'float'> {
  return smoothstep(-0.05, 0.1, ellipse(p.sub(vec2(0.05, -0.07)), 0, 0, rx, ry)).oneMinus().mul(0.45);
}

function hpArc(p: N<'vec2'>, radius: N<'float'>, hp: N<'float'>): { track: N<'float'>; fill: N<'float'> } {
  const a = atan(p.x, p.y).div(2 * Math.PI).add(0.5), band = ring(length(p).sub(radius), 0.022);
  const shown = hp.lessThan(0.999);
  return { track: select(shown, coverage(band), float(0)), fill: select(shown, coverage(band).mul(coverage(a.sub(hp).mul(radius).mul(6.3))), float(0)) };
}

function eyes(b: N<'vec2'>, at: N<'vec2'>, spread: number, size: number, awake: N<'float'>): { open: N<'float'>; shut: N<'float'> } {
  const left = at.sub(vec2(spread, 0)), right = at.add(vec2(spread, 0));
  const open = min(sdCircle(b.sub(left), size), sdCircle(b.sub(right), size));
  const shut = min(sdSegment(b, left.sub(vec2(size, 0)), left.add(vec2(size, 0)), size * 0.3), sdSegment(b, right.sub(vec2(size, 0)), right.add(vec2(size, 0)), size * 0.3));
  return { open: coverage(open).mul(awake), shut: coverage(shut).mul(awake.oneMinus()) };
}

interface Body { shape: N<'float'>; skin: N<'vec3'>; eyesAt: N<'vec2'>; eyeSpread: number; eyeSize: number; extra: (layers: Layers) => void }

function body(plan: Plan, b: N<'vec2'>, skin: N<'vec3'>, trim: N<'vec3'>, form: N<'vec4'>, state: N<'vec4'>, time: N<'float'>, lit: Lit): Body {
  const horns = form.y, armour = form.z, stride = sin(time.mul(9).add(state.w.mul(6.28))).mul(state.x);
  const dark = skin.mul(0.62), metal = mix(rgb(0x8d949c), trim, 0.25);
  if (plan === 'walker') {
    const torso = ellipse(b, 0, -0.03, 0.3, 0.2), head = sdCircle(b.sub(vec2(0, 0.1)), 0.14);
    const armL = sdSegment(b, vec2(-0.25, 0), vec2(-0.29, stride.mul(0.08).add(0.17)), 0.075), armR = sdSegment(b, vec2(0.25, 0), vec2(0.29, stride.mul(-0.08).add(0.17)), 0.075);
    const hornL = sdSegment(b, vec2(-0.08, 0.17), vec2(-0.2, 0.3), 0.035), hornR = sdSegment(b, vec2(0.08, 0.17), vec2(0.2, 0.3), 0.035);
    const pad = min(sdCircle(b.sub(vec2(-0.23, 0.01)), 0.1), sdCircle(b.sub(vec2(0.23, 0.01)), 0.1));
    const shape = min(min(smin(torso, head, 0.06), min(armL, armR)), select(horns.greaterThan(0.5), min(hornL, hornR), float(9)));
    return {
      shape, skin, eyesAt: vec2(0, 0.15), eyeSpread: 0.055, eyeSize: 0.028,
      extra: (layers) => {
        layers.over(coverage(min(armL, armR)), lit(dark));
        layers.over(coverage(torso), lit(skin.mul(smoothstep(-0.2, 0.1, b.y).mul(0.25).add(0.8))));
        layers.over(coverage(pad).mul(armour), lit(metal.mul(smoothstep(-0.1, 0.1, b.y.sub(b.x.mul(0.3))).mul(0.3).add(0.8))));
        layers.over(coverage(ring(pad, 0.012)).mul(armour), ink);
        layers.over(select(horns.greaterThan(0.5), coverage(min(hornL, hornR)), float(0)), lit(rgb(0xe8dcc0)));
        layers.over(coverage(head), lit(skin.mul(1.08)));
      },
    };
  }
  if (plan === 'runner') {
    const legs = [[-0.15, 0.14, 1], [0.15, 0.14, -1], [-0.15, -0.2, -1], [0.15, -0.2, 1]] as const;
    let leg: N<'float'> = float(9);
    for (const [x, y, s] of legs) leg = min(leg, sdSegment(b, vec2(x, y), vec2(x * 1.55, stride.mul(0.07 * s).add(y + 0.05)), 0.045));
    const trunk = ellipse(b, 0, -0.04, 0.15, 0.31), head = ellipse(b, 0, 0.29, 0.11, 0.13), snout = ellipse(b, 0, 0.4, 0.06, 0.07);
    const tail = sdSegment(b, vec2(0, -0.33), vec2(stride.mul(0.08), -0.55), 0.03);
    const ears = min(sdCircle(b.sub(vec2(-0.08, 0.34)), 0.04), sdCircle(b.sub(vec2(0.08, 0.34)), 0.04));
    const shape = min(min(smin(smin(trunk, head, 0.05), snout, 0.03), min(leg, tail)), ears);
    return {
      shape, skin, eyesAt: vec2(0, 0.33), eyeSpread: 0.05, eyeSize: 0.024,
      extra: (layers) => {
        layers.over(coverage(min(leg, tail)), lit(dark));
        layers.over(coverage(smin(trunk, head, 0.05)), lit(skin.mul(smoothstep(-0.3, 0.3, b.y).mul(0.2).add(0.85))));
        layers.over(coverage(ellipse(b, 0, -0.04, 0.05, 0.22)).mul(armour), lit(metal));
        layers.over(coverage(snout), lit(skin.mul(1.15)));
        layers.over(coverage(ears), lit(dark));
        layers.over(select(horns.greaterThan(0.5), coverage(min(sdSegment(b, vec2(-0.06, 0.33), vec2(-0.14, 0.44), 0.025), sdSegment(b, vec2(0.06, 0.33), vec2(0.14, 0.44), 0.025))), float(0)), lit(rgb(0xe8dcc0)));
      },
    };
  }
  if (plan === 'flier') {
    const flap = sin(time.mul(26).add(state.w.mul(6.28))).mul(0.5).add(0.5).mul(state.x.mul(0.6).add(0.2));
    const span = flap.mul(0.35).add(0.65);
    const wing = (side: number) => {
      const w = vec2(b.x.mul(side).sub(0.08).div(span), b.y);
      const sail = ellipse(w, 0.2, 0.02, 0.22, 0.13);
      const cut = min(sdCircle(w.sub(vec2(0.1, -0.13)), 0.06), min(sdCircle(w.sub(vec2(0.22, -0.14)), 0.06), sdCircle(w.sub(vec2(0.34, -0.11)), 0.05)));
      return max(sail, cut.negate()).mul(span);
    };
    const wings = min(wing(-1), wing(1)), trunk = ellipse(b, 0, -0.02, 0.1, 0.15), head = sdCircle(b.sub(vec2(0, 0.13)), 0.085);
    const ears = min(sdSegment(b, vec2(-0.04, 0.17), vec2(-0.08, 0.26), 0.022), sdSegment(b, vec2(0.04, 0.17), vec2(0.08, 0.26), 0.022));
    const shape = min(min(wings, smin(trunk, head, 0.04)), ears);
    return {
      shape, skin, eyesAt: vec2(0, 0.15), eyeSpread: 0.035, eyeSize: 0.02,
      extra: (layers) => {
        layers.over(coverage(wings), lit(mix(dark, trim, 0.25).mul(0.9)));
        layers.over(coverage(ring(wings, 0.008)).mul(0.6), lit(dark.mul(0.6)));
        layers.over(coverage(min(smin(trunk, head, 0.04), ears)), lit(skin));
      },
    };
  }
  if (plan === 'caster') {
    const robe = sdCircle(b.sub(vec2(0, -0.04)), 0.25), hood = sdCircle(b.sub(vec2(0, 0.05)), 0.15);
    const hand = vec2(0.2, 0.2), staff = sdSegment(b, vec2(0.21, -0.2), vec2(0.23, 0.36), 0.025), orb = sdCircle(b.sub(vec2(0.23, 0.38)), 0.075);
    const physical = form.w.lessThan(0.5);
    const bow = max(ring(sdCircle(b.sub(vec2(0, 0.08)), 0.3), 0.022), b.y.sub(0.2).negate());
    const sleeve = sdSegment(b, vec2(0.15, 0), hand, 0.06);
    const focus = select(physical, bow, min(staff, orb));
    const shape = min(min(robe, sleeve), focus);
    return {
      shape, skin, eyesAt: vec2(0, 0.1), eyeSpread: 0.045, eyeSize: 0.022,
      extra: (layers) => {
        layers.over(coverage(robe), lit(dark.mul(smoothstep(-0.3, 0.2, b.y).mul(0.3).add(0.75))));
        layers.over(coverage(ring(robe.add(0.06), 0.015)), lit(trim.mul(0.55)));
        layers.over(coverage(sleeve), lit(dark.mul(1.1)));
        layers.over(select(physical, coverage(bow), coverage(staff)), lit(select(physical, rgb(0x9a7a4c), rgb(0x6b4a2e))));
        layers.over(select(physical, float(0), coverage(orb)), trim.mul(1.8));
        layers.over(coverage(hood), lit(skin.mul(0.85)));
        layers.over(coverage(sdCircle(b.sub(vec2(0, 0.1)), 0.09)), rgb(0x0a080c));
      },
    };
  }
  const pulse = sin(time.mul(2.3).add(state.w.mul(6.28))).mul(0.015);
  let cap: N<'float'> = sdCircle(b, float(0.2).add(pulse));
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2 + 0.4;
    cap = smin(cap, sdCircle(b.sub(vec2(Math.cos(a) * 0.17, Math.sin(a) * 0.17)), 0.11), 0.08);
  }
  let tendril: N<'float'> = float(9);
  for (let k = 0; k < 7; k++) {
    const a = (k / 7) * Math.PI * 2, sway = sin(time.mul(1.7).add(k * 1.9)).mul(0.04);
    tendril = min(tendril, sdSegment(b, vec2(Math.cos(a) * 0.22, Math.sin(a) * 0.22), vec2(Math.cos(a) * 0.4, Math.sin(a) * 0.4).add(vec2(sway, sway.negate())), 0.022));
  }
  let spots: N<'float'> = float(9);
  for (const [x, y, r] of [[-0.09, 0.08, 0.045], [0.1, 0.05, 0.035], [0.02, -0.11, 0.05], [-0.13, -0.07, 0.03], [0.13, -0.1, 0.03]] as const) spots = min(spots, sdCircle(b.sub(vec2(x, y)), r));
  return {
    shape: min(cap, tendril), skin, eyesAt: vec2(0, 0.02), eyeSpread: 0.05, eyeSize: 0.026,
    extra: (layers) => {
      layers.over(coverage(tendril), lit(dark.mul(0.9)));
      layers.over(coverage(cap), lit(skin.mul(smoothstep(-0.3, 0.3, b.y.add(b.x.mul(0.4))).mul(0.3).add(0.8))));
      layers.over(coverage(spots), lit(trim.mul(0.9)));
    },
  };
}

const glowing = (lit: Lit): Lit => (albedo) => lit(albedo).mul(0.62).add(albedo.mul(0.5));

export function monsterMaterial(plan: Plan, shaded: Lit, time: N<'float'>) {
  const lit = glowing(shaded);
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const skinCh = channel('aSkin'), trimCh = channel('aTrim'), form = channel('aForm'), state = channel('aState'), mark = channel('aMark');
    const skin = skinCh.xyz, trim = trimCh.xyz, size = form.x, element = skinCh.w;
    const b = p.div(size), awake = state.x, hp = state.y, flash = state.z, up = turn(p, trimCh.w);
    const creature = body(plan, b, skin, trim, form, state, time, lit);
    const layers = new Layers();
    layers.over(shadowOf(b, 0.34, 0.3).mul(mark.z), vec3(0));
    const guard = mark.y, boss = mark.x;
    const ward = sdCircle(p, size.mul(0.52));
    layers.over(coverage(ring(ward, 0.02)).mul(guard).mul(0.8), rgb(0x9fb4ff).mul(1.3));
    layers.over(coverage(creature.shape.sub(0.03)), ink);
    creature.extra(layers);
    const fire = element.equal(1), cold = element.equal(2), poison = element.equal(3), drain = element.equal(4);
    const flicker = sin(time.mul(17).add(state.w.mul(9))).mul(0.5).add(0.5);
    const crest = flame(b.sub(vec2(0, -0.1)), 0.34);
    layers.over(select(fire, coverage(crest).mul(smoothstep(-0.25, 0.1, b.y).mul(0.6)).mul(awake.mul(0.5).add(0.5)), float(0)), mix(trim, rgb(0xfff0b0), flicker.mul(0.4)).mul(1.6));
    let frost: N<'float'> = float(9);
    for (let k = 0; k < 4; k++) frost = min(frost, sdSegment(b, vec2(0, -0.02), vec2(Math.cos(k * 1.57 + 0.78) * 0.2, Math.sin(k * 1.57 + 0.78) * 0.2 - 0.02), 0.012));
    layers.over(select(cold, coverage(frost).mul(0.8), float(0)), trim.mul(1.3));
    const ooze = sdCircle(vec2(b.x.add(0.12), fract(time.mul(0.35).add(state.w)).mul(-0.25).add(b.y.add(0.1))), 0.025);
    layers.over(select(poison, coverage(ooze).mul(0.9), float(0)), trim.mul(1.2));
    const wisp = abs(sin(atan(b.y, b.x).mul(5).add(time.mul(3)))).mul(0.05).add(0.33);
    layers.over(select(drain, smoothstep(wisp, wisp.add(0.08), length(b)).oneMinus().mul(smoothstep(wisp.sub(0.12), wisp, length(b))).mul(0.5), float(0)), trim.mul(1.2));
    const look = eyes(b, creature.eyesAt, creature.eyeSpread, creature.eyeSize, awake);
    layers.over(look.open, mix(rgb(0xffe08a), trim, select(element.greaterThan(0.5), float(0.6), float(0))).mul(2.2));
    layers.over(look.shut, ink);
    layers.over(clamp(flash, 0, 1).mul(coverage(creature.shape)), rgb(0xffffff).mul(2));
    const arc = hpArc(up, size.mul(0.36).add(0.06), hp);
    layers.over(arc.track, rgb(0x1a1113));
    layers.over(arc.fill, mix(rgb(0xe8574a), rgb(0x8fe07a), hp).mul(1.3));
    const tiara = crown(up.sub(vec2(0, size.mul(0.42).add(0.14))), 0.3);
    layers.over(coverage(tiara.sub(0.02)).mul(boss), ink);
    layers.over(coverage(tiara).mul(boss), rgb(0xffd166).mul(1.4));
    const badge = shield(up.sub(vec2(size.mul(0.36).add(0.1), size.mul(0.3).add(0.06))), 0.2);
    layers.over(coverage(badge.sub(0.02)).mul(guard), ink);
    layers.over(coverage(badge).mul(guard), rgb(0x9fb4ff).mul(1.3));
    const bob = sin(time.mul(2).add(state.w.mul(5))).mul(0.03);
    const zAt = up.sub(vec2(size.mul(0.22).add(0.08), size.mul(0.2).add(0.12).add(bob)));
    const zee = (q: N<'vec2'>, s: number) => min(min(sdBox(q.sub(vec2(0, s)), vec2(s, s * 0.17)), sdBox(q.add(vec2(0, s)), vec2(s, s * 0.17))), sdSegment(q, vec2(-s * 0.9, -s), vec2(s * 0.9, s), s * 0.17));
    const z1 = zee(zAt, 0.07), z2 = zee(zAt.sub(vec2(0.15, 0.16)), 0.05);
    layers.over(coverage(min(z1, z2)).mul(awake.oneMinus()).mul(mark.w), rgb(0xcfe0ff).mul(1.4));
    return layers.material(mark.z);
  };
}

export function heroMaterial(accent: number, shaded: Lit, time: N<'float'>) {
  const lit = glowing(shaded);
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const pose = channel('aPose'), status = channel('aStatus'), vital = channel('aVital');
    const stride = pose.x, thrust = pose.y, flash = pose.z, alpha = pose.w;
    const b = p;
    const cloak = rgb(accent), deep = cloak.mul(0.5);
    const layers = new Layers();
    layers.over(shadowOf(b, 0.3, 0.3), vec3(0));
    const swing = sin(stride.mul(Math.PI * 2)).mul(0.05), reach = thrust.mul(0.16);
    const blade = sdSegment(b, vec2(0.21, reach.add(0.16)), vec2(0.23, reach.add(0.47)), 0.026);
    const hilt = min(sdSegment(b, vec2(0.13, reach.add(0.15)), vec2(0.3, reach.add(0.15)), 0.022), sdSegment(b, vec2(0.21, reach.add(0.08)), vec2(0.21, reach.add(0.15)), 0.02));
    const armR = sdSegment(b, vec2(0.17, -0.03), vec2(0.21, reach.add(0.1).add(swing)), 0.062);
    const armL = sdSegment(b, vec2(-0.17, -0.03), vec2(-0.22, float(0.12).sub(swing)), 0.062);
    const handle = sdSegment(b, vec2(-0.22, float(0.05).sub(swing)), vec2(-0.24, float(0.27).sub(swing)), 0.028);
    const cape = smin(ellipse(b, 0, -0.07, 0.25, 0.2), ellipse(b, 0, -0.2, 0.2, 0.13), 0.08);
    const roll = sdSegment(b, vec2(-0.12, -0.12), vec2(0.12, -0.12), 0.055);
    const hood = sdCircle(b.sub(vec2(0, 0.06)), 0.125);
    const face = sdCircle(b.sub(vec2(0, 0.105)), 0.07);
    const silhouette = min(min(min(cape, hood), min(armL, armR)), min(min(blade, hilt), handle));
    layers.over(coverage(silhouette.sub(0.035)), ink);
    layers.over(coverage(blade), lit(mix(rgb(0x9aa6b4), rgb(0xf4f8ff), smoothstep(-0.02, 0.02, b.x.sub(0.22)).mul(0.6).add(0.2))).add(rgb(0x404850).mul(thrust)));
    layers.over(coverage(hilt), lit(rgb(0xb89a5a)));
    layers.over(coverage(handle), lit(rgb(0x5a3a22)));
    layers.over(coverage(cape), lit(deep.mul(abs(sin(b.x.mul(26))).mul(0.25).add(0.8))));
    layers.over(coverage(ring(cape.add(0.04), 0.012)).mul(0.7), lit(cloak.mul(0.75)));
    layers.over(coverage(roll), lit(mix(rgb(0x6b4a2e), rgb(0x9a7248), smoothstep(-0.05, 0.05, b.y.add(0.12)))));
    layers.over(coverage(min(armL, armR)), lit(cloak.mul(0.72)));
    layers.over(coverage(hood), lit(cloak.mul(smoothstep(-0.15, 0.2, b.y.sub(b.x.mul(0.6))).mul(0.35).add(0.85))));
    layers.over(coverage(max(face, hood)).mul(0.85), rgb(0x1a0f0a));
    const flick = sin(time.mul(23)).mul(0.5).add(sin(time.mul(37).add(1.3)).mul(0.5)).mul(0.5).add(0.5);
    const fire = flame(b.sub(vec2(-0.245, float(0.33).sub(swing))), flick.mul(0.025).add(0.2));
    layers.over(smoothstep(0, 0.14, fire).oneMinus().mul(0.35), rgb(0xff9a3c).mul(1.4));
    layers.over(coverage(fire), mix(rgb(0xff8a2c), rgb(0xfff2b8), smoothstep(-0.05, 0.02, fire).oneMinus().mul(0.9)).mul(2.6));
    const poison = status.x, confused = status.y, blind = status.z, ward = status.w;
    const bubble = min(sdCircle(b.sub(vec2(-0.12, fract(time.mul(0.8)).mul(0.25).add(0.12))), 0.03), sdCircle(b.sub(vec2(0.1, fract(time.mul(0.8).add(0.5)).mul(0.25).add(0.08))), 0.025));
    layers.over(coverage(ring(bubble, 0.008)).mul(poison), rgb(0xb4f05a).mul(1.4));
    layers.over(coverage(cape).mul(poison).mul(0.3), rgb(0x8fd43a));
    let stars: N<'float'> = float(9);
    for (let k = 0; k < 3; k++) {
      const a = time.mul(4).add(k * 2.094);
      stars = min(stars, sdCircle(b.sub(vec2(cos(a).mul(0.15), sin(a).mul(0.05).add(0.06))), 0.03));
    }
    layers.over(coverage(stars).mul(confused), rgb(0xffe27a).mul(1.6));
    layers.over(coverage(sdBox(b.sub(vec2(0, 0.1)), vec2(0.1, 0.022), 0.01)).mul(blind), rgb(0xe8e0d0));
    const shimmer = sin(atan(b.y, b.x).mul(6).add(time.mul(3))).mul(0.5).add(0.5);
    const fireWard = select(ward.equal(1).or(ward.equal(3)), float(1), float(0)), frostWard = select(ward.equal(2).or(ward.equal(3)), float(1), float(0));
    layers.over(coverage(abs(length(b).sub(0.4)).sub(0.014)).mul(fireWard).mul(shimmer.mul(0.5).add(0.3)), rgb(0xff8a3a).mul(1.5));
    layers.over(coverage(abs(length(b).sub(0.44)).sub(0.014)).mul(frostWard).mul(shimmer.oneMinus().mul(0.5).add(0.3)), rgb(0x9fe2ff).mul(1.5));
    const hp = vital.x, starving = vital.y;
    const arc = hpArc(turn(b, vital.z), float(0.36), min(hp, 0.9989));
    layers.over(arc.track.mul(0.7), rgb(0x1a1113));
    layers.over(arc.fill, mix(mix(rgb(0xe8574a), rgb(0xf2c14e), smoothstep(0.2, 0.5, hp)), rgb(0x8fe07a), smoothstep(0.5, 0.8, hp)).mul(mix(float(1.2), sin(time.mul(12)).mul(0.5).add(1.3), starving)));
    layers.over(clamp(flash, 0, 1).mul(coverage(silhouette)).mul(0.8), rgb(0xff5a4a).mul(1.8));
    return layers.material(alpha);
  };
}

export const TORCH = new THREE.Vector2(-0.245, 0.33);
