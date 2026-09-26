import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, float, length, max, min, mix, pow, select, sin, smoothstep, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { blaze, blindEye, crown, cross, flame, flask, heart, Layers, loaf, moon, ring, shield, skull, snowflake, spiral, stairs, star, text, tomb, type Lit, type N } from './art';

type Channel = (name: string) => N<'vec4'>;

export const ICON = { heart: 0, heartPlus: 1, cross: 2, flame: 3, snow: 4, skull: 5, spiral: 6, blind: 7, loaf: 8, coin: 9, stairs: 10, moon: 11, flask: 12, crown: 13, shield: 14, star: 15, tomb: 16 } as const;
export const EFFECT_TINTS = [0xff5a6a, 0xff8fc0, 0xe6f7d8, 0xff9a3c, 0x9fe2ff, 0x9ee04e, 0xffe27a, 0xb9a6d9] as const;

const ink = rgb(0x0b0a0c);

function coin(p: N<'vec2'>, size: number): N<'float'> {
  return sdCircle(p, size * 0.34);
}

export function icon(p: N<'vec2'>, code: N<'float'>, size: number): N<'float'> {
  const shapes: N<'float'>[] = [
    heart(p, size * 0.72),
    min(heart(p.sub(vec2(size * -0.08, size * -0.04)), size * 0.6), cross(p.sub(vec2(size * 0.26, size * 0.26)), size * 0.42)),
    cross(p, size * 0.9),
    blaze(p, size * 0.95),
    snowflake(p, size),
    skull(p, size * 0.9),
    spiral(p, size),
    blindEye(p, size),
    loaf(p, size),
    coin(p, size),
    stairs(p, size),
    moon(p, size),
    flask(p, size),
    crown(p, size),
    shield(p, size),
    star(p, size),
    tomb(p, size),
  ];
  let d: N<'float'> = shapes[shapes.length - 1]!;
  for (let k = shapes.length - 2; k >= 0; k--) d = select(abs(code.sub(k)).lessThan(0.5), shapes[k]!, d);
  return d;
}

export const ITEM_QUAD = 0.9;
export const ITEM_SCALE = 1.35;

export function itemMaterial(shaded: Lit, time: N<'float'>) {
  const lit: Lit = (albedo) => shaded(albedo).mul(0.62).add(albedo.mul(0.5));
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const item = channel('aItem'), tint = channel('aTint').xyz;
    const kind = item.x, phase = item.y, alpha = item.z;
    const layers = new Layers();
    const shadow = smoothstep(-0.04, 0.08, length(p.sub(vec2(0.04, -0.06)).div(vec2(1, 0.7))).sub(0.2)).oneMinus().mul(0.5);
    const gold = kind.lessThan(0.5), ration = kind.greaterThan(0.5).and(kind.lessThan(1.5));
    const coins = [[-0.08, -0.05], [0.09, -0.04], [0.0, 0.06]] as const;
    let pile: N<'float'> = float(9);
    for (const [x, y] of coins) pile = min(pile, sdCircle(p.sub(vec2(x, y)), 0.1));
    const goldLayers = (l: Layers) => {
      l.over(coverage(pile.sub(0.025)), ink);
      for (const [x, y] of coins) {
        const c = sdCircle(p.sub(vec2(x, y)), 0.1);
        l.over(coverage(c), lit(mix(rgb(0xb8862c), rgb(0xffd66b), smoothstep(-0.1, 0.1, p.y.sub(y).sub(p.x.sub(x).mul(0.5))))));
        l.over(coverage(ring(c.add(0.03), 0.008)), lit(rgb(0x8a5f1a)));
      }
    };
    const glint = pow(max(sin(time.mul(2.1).add(phase.mul(6.28))), 0), 8);
    const sparkle = min(sdBox(p.sub(vec2(0.05, 0.1)), vec2(0.09, 0.008)), sdBox(p.sub(vec2(0.05, 0.1)), vec2(0.008, 0.09)));
    const bread = sdBox(p, vec2(0.17, 0.1), 0.09), cloth = sdBox(p.sub(vec2(0.02, 0)), vec2(0.035, 0.105), 0.01);
    const body = sdCircle(p.sub(vec2(0, -0.05)), 0.14), neck = sdBox(p.sub(vec2(0, 0.1)), vec2(0.045, 0.06), 0.015), cork = sdBox(p.sub(vec2(0, 0.17)), vec2(0.05, 0.025), 0.01);
    const vessel = min(body, neck);
    const l = new Layers();
    l.over(shadow, vec3(0));
    goldLayers(l);
    l.over(coverage(sparkle).mul(glint), rgb(0xfff6d0).mul(2.5));
    const r = new Layers();
    r.over(shadow, vec3(0));
    r.over(coverage(bread.sub(0.025)), ink);
    r.over(coverage(bread), lit(mix(rgb(0x9a6a36), rgb(0xd29b58), smoothstep(-0.12, 0.1, p.y.sub(p.x.mul(0.3))))));
    r.over(coverage(ring(bread.add(0.05), 0.008)).mul(0.5), lit(rgb(0x6b4520)));
    r.over(coverage(cloth), lit(rgb(0xd8ccb0)));
    const v = new Layers();
    v.over(shadow, vec3(0));
    v.over(coverage(min(vessel, cork).sub(0.025)), ink);
    v.over(coverage(vessel), lit(mix(tint.mul(0.55), tint, smoothstep(-0.14, 0.06, p.y.add(0.05)))).add(tint.mul(0.25)));
    v.over(coverage(ring(vessel, 0.01)), lit(rgb(0xdfe8ee)).mul(0.7));
    v.over(coverage(sdCircle(p.sub(vec2(-0.05, -0.01)), 0.035)), rgb(0xffffff).mul(0.9));
    v.over(coverage(cork), lit(rgb(0x8a6a44)));
    layers.pre = select(gold, l.pre, select(ration, r.pre, v.pre));
    layers.alpha = select(gold, l.alpha, select(ration, r.alpha, v.alpha));
    return layers.material(alpha);
  };
}

export const SCONCE_QUAD = 0.8;

export function sconceMaterial(lit: Lit, time: N<'float'>) {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const state = channel('aSconce');
    const burning = state.x, phase = state.y;
    const layers = new Layers();
    const bracket = sdSegment(p, vec2(0, -0.26), vec2(0, -0.08), 0.03);
    const bowl = max(sdCircle(p.sub(vec2(0, -0.02)), 0.12), p.y.sub(0.02));
    layers.over(coverage(min(bracket, bowl).sub(0.025)), ink);
    layers.over(coverage(bracket), lit(rgb(0x3a3634)));
    layers.over(coverage(bowl), lit(mix(rgb(0x2a2624), rgb(0x5a534c), smoothstep(-0.12, 0.02, p.y))));
    const flick = sin(time.mul(19).add(phase.mul(20))).mul(0.5).add(sin(time.mul(31).add(phase.mul(7))).mul(0.5)).mul(0.5).add(0.5);
    const fire = flame(p.sub(vec2(0, 0.06)), flick.mul(0.03).add(0.2));
    layers.over(coverage(fire).mul(burning), mix(rgb(0xff8a2c), rgb(0xfff0b0), smoothstep(-0.05, 0.03, fire).oneMinus()).mul(2.6));
    const embers = min(sdCircle(p.sub(vec2(-0.04, 0.01)), 0.02), sdCircle(p.sub(vec2(0.035, 0.0)), 0.016));
    layers.over(coverage(embers).mul(burning.oneMinus()), rgb(0x6a4a3a));
    return layers.material();
  };
}

export const SLOT = [4.9, 2.25] as const;
const PLATE = 3.9;
const INSET = SLOT[0] - PLATE;

export function slotMaterial(accent: number, time: N<'float'>) {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const slot = channel('aSlot'), tint = channel('aTint'), codes = channel('aText'), meta = channel('aMeta');
    const kind = slot.x, count = slot.y, selected = slot.z, onStairs = slot.w;
    const w = PLATE, h = SLOT[1], q = p.sub(vec2(INSET / 2, 0));
    const empty = count.lessThan(0.5).and(kind.greaterThan(0.5));
    const dim = select(empty, float(0.22), float(1));
    const layers = new Layers();
    const plate = sdBox(q, vec2(w / 2 - 0.1, h / 2 - 0.1), 0.22);
    const pulse = sin(time.mul(5)).mul(0.5).add(0.5);
    layers.over(smoothstep(-0.1, 0.25, plate.sub(0.05)).oneMinus().mul(0.55), vec3(0));
    layers.over(coverage(plate), mix(rgb(0x1a181c), rgb(0x2a262a), smoothstep(-1, 1, q.y.sub(q.x.mul(0.2)))));
    layers.over(coverage(ring(plate.add(0.05), 0.035)), mix(rgb(0x4a4238), rgb(accent).mul(pulse.mul(0.6).add(1.4)), selected));
    layers.over(smoothstep(0, 0.3, abs(plate)).oneMinus().mul(selected).mul(0.35), rgb(accent).mul(1.4));
    const at = q.sub(vec2(-w / 2 + h / 2, 0));
    const shape = select(kind.lessThan(0.5), select(onStairs.greaterThan(0.5), stairs(at, 1.5), moon(at, 1.35)), select(kind.lessThan(1.5), loaf(at, 1.35), flask(at, 1.45)));
    const colour = select(kind.lessThan(0.5), select(onStairs.greaterThan(0.5), rgb(0xe8dcc0), rgb(0xb8c6e8)), select(kind.lessThan(1.5), rgb(0xd29b58), tint.xyz));
    const shown = select(empty, mix(colour, vec3(0.5), 0.6), colour).mul(dim);
    layers.over(coverage(shape.sub(0.05)), ink);
    layers.over(coverage(shape), shown.mul(select(kind.greaterThan(1.5), smoothstep(-0.4, 0.2, at.y).mul(0.5).add(0.7), float(1))));
    layers.over(select(kind.greaterThan(1.5), coverage(sdCircle(at.sub(vec2(-0.14, -0.1)), 0.07)), float(0)).mul(dim), rgb(0xffffff).mul(0.8));
    const label = text(q.sub(vec2(0.2, 0)).div(0.62), codes, meta.x).mul(0.62);
    layers.over(coverage(label), rgb(0xf2ead8));
    const emblemAt = q.sub(vec2(w / 2 - 0.62, 0));
    const emblem = icon(emblemAt, tint.w, 0.78);
    const known = meta.y;
    layers.over(coverage(sdCircle(emblemAt, 0.42)).mul(known), rgb(0x100e12));
    layers.over(coverage(emblem).mul(known), select(empty, rgb(0x5a5460), effectTint(tint.w)));
    const cursorAt = q.sub(vec2(-w / 2 - 0.3, 0));
    const pointer = max(abs(cursorAt.y).sub(float(0.2).sub(cursorAt.x).mul(0.7)), float(-0.2).sub(cursorAt.x));
    layers.over(coverage(pointer.sub(0.04)).mul(selected), ink);
    layers.over(coverage(pointer).mul(selected), rgb(accent).mul(pulse.mul(0.5).add(1.5)));
    return layers.material();
  };
}

function effectTint(effect: N<'float'>): N<'vec3'> {
  let c: N<'vec3'> = rgb(EFFECT_TINTS[7]);
  for (let k = 6; k >= 0; k--) c = select(abs(effect.sub(k)).lessThan(0.5), rgb(EFFECT_TINTS[k]!), c);
  return c;
}

export const TIER = [4.6, 2.2] as const;

export function tierMaterial(accent: number, time: N<'float'>) {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const tier = channel('aTier'), codes = channel('aText');
    const state = tier.y, mark = tier.z, taper = tier.w;
    const w = TIER[0], h = TIER[1];
    const current = state.greaterThan(1.5), reached = state.greaterThan(0.5);
    const layers = new Layers();
    const half = vec2(float(w / 2 - 0.12).sub(taper), h / 2 - 0.12);
    const slab = sdBox(p, half, 0.16);
    const pulse = sin(time.mul(4)).mul(0.5).add(0.5);
    layers.over(smoothstep(-0.1, 0.3, slab.sub(0.04)).oneMinus().mul(0.5), vec3(0));
    const stone = select(current, rgb(accent).mul(0.5), select(reached, rgb(0x4a4540), rgb(0x1f1d21)));
    layers.over(coverage(slab), stone.mul(smoothstep(-1, 1, p.y).mul(0.3).add(0.8)));
    layers.over(coverage(ring(slab.add(0.04), 0.03)), select(current, rgb(accent).mul(pulse.mul(0.5).add(1.3)), select(reached, rgb(0x7a7064), rgb(0x3a363a))));
    const number = text(p.sub(vec2(half.x.negate().add(0.55), 0)).div(0.9), codes, float(1)).mul(0.9);
    layers.over(coverage(number), select(current, rgb(0xfff4e0), select(reached, rgb(0xd8cfc0), rgb(0x6a6470))));
    const sigil = p.sub(vec2(half.x.sub(0.6), 0));
    const guard = shield(sigil, 0.95), boss = crown(sigil, 1.05);
    layers.over(coverage(select(mark.greaterThan(1.5), boss, guard).sub(0.04)).mul(select(mark.greaterThan(0.5), float(1), float(0))), ink);
    layers.over(coverage(select(mark.greaterThan(1.5), boss, guard)).mul(select(mark.greaterThan(0.5), float(1), float(0))), select(mark.greaterThan(1.5), rgb(0xffd166), rgb(0x9fb4ff)).mul(select(reached, float(1.2), float(0.6))));
    const hood = sdCircle(p.sub(vec2(0.05, 0)), 0.36), cape = sdBox(p.sub(vec2(0.05, -0.18)), vec2(0.34, 0.2), 0.16);
    layers.over(coverage(min(hood, cape).sub(0.04)).mul(select(current, float(1), float(0))), ink);
    layers.over(coverage(min(hood, cape)).mul(select(current, float(1), float(0))), rgb(accent).mul(1.25));
    return layers.material();
  };
}

export const BAR = [8.4, 1.3] as const;

export function barMaterial() {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const bar = channel('aHud'), tint = channel('aTint'), codes = channel('aText'), meta = channel('aMeta');
    const fill = clamp(bar.y, 0, 1), code = bar.z, alert = bar.w;
    const w = BAR[0], h = BAR[1];
    const layers = new Layers();
    const plate = sdBox(p, vec2(w / 2 - 0.05, h / 2 - 0.05), 0.2);
    layers.over(coverage(plate), rgb(0x141216));
    layers.over(coverage(ring(plate.add(0.03), 0.025)), rgb(0x3e3834));
    const at = p.sub(vec2(-w / 2 + h / 2, 0));
    const glyphShape = icon(at, code, 0.95);
    layers.over(coverage(glyphShape.sub(0.04)), ink);
    layers.over(coverage(glyphShape), tint.xyz.mul(alert.mul(sin(meta.w.mul(12)).mul(0.4)).add(1.2)));
    const x0 = -w / 2 + h + 0.05, x1 = w / 2 - 2.4;
    const track = sdBox(p.sub(vec2((x0 + x1) / 2, 0)), vec2((x1 - x0) / 2, 0.2), 0.12);
    layers.over(coverage(track), rgb(0x241f22));
    const level = sdBox(p.sub(vec2((x0 + x1) / 2, 0)), vec2((x1 - x0) / 2, 0.2), 0.12);
    layers.over(coverage(level).mul(coverage(p.x.sub(mix(float(x0), float(x1), fill)))), tint.xyz.mul(smoothstep(-0.2, 0.2, p.y).mul(0.35).add(0.85)));
    const tick = abs(p.x.sub(mix(float(x0), float(x1), meta.y))).sub(0.03);
    layers.over(coverage(max(tick, abs(p.y).sub(0.26))).mul(select(meta.y.greaterThan(0), float(1), float(0))), rgb(0xf2ead8).mul(0.7));
    const label = text(p.sub(vec2(w / 2 - 1.15, 0)).div(0.6), codes, meta.x).mul(0.6);
    layers.over(coverage(label), rgb(0xf2ead8));
    return layers.material();
  };
}

export const BADGE = 1.35;

export function badgeMaterial(time: N<'float'>) {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const badge = channel('aHud'), tint = channel('aTint');
    const left = clamp(badge.y, 0, 1), code = badge.z, active = badge.w;
    const layers = new Layers();
    const disc = sdCircle(p, 0.56);
    layers.over(coverage(disc), mix(rgb(0x141216), tint.xyz.mul(0.22), active));
    const a = atan(p.x, p.y).div(2 * Math.PI).add(0.5);
    const band = ring(length(p).sub(0.56), 0.05);
    layers.over(coverage(band), rgb(0x2e2a2c));
    layers.over(coverage(band).mul(coverage(a.sub(left).mul(3.5))).mul(active), tint.xyz.mul(1.4));
    const glyphShape = icon(p, code, 0.72);
    const beat = sin(time.mul(6)).mul(0.15).add(1);
    layers.over(coverage(glyphShape), mix(rgb(0x4a4450), tint.xyz.mul(beat.mul(1.3)), active));
    return layers.material();
  };
}

export const FX_QUAD = 2.2;
export const FX = { slash: 0, puff: 1, number: 2, sparkle: 3, swirl: 4, rest: 5, dust: 6, tomb: 7, icon: 8, whiff: 9, rise: 10 } as const;

export function fxMaterial() {
  return (p: N<'vec2'>, channel: Channel): THREE.Material => {
    const fx = channel('aFx'), tint = channel('aTint'), codes = channel('aText'), meta = channel('aMeta');
    const kind = fx.x, age = clamp(fx.y, 0, 1), code = fx.z, spin = fx.w;
    const colour = tint.xyz.mul(tint.w);
    const fade = age.oneMinus();
    const is = (k: number) => abs(kind.sub(k)).lessThan(0.5);
    const dir = vec2(meta.z, meta.w);
    const along = p.x.mul(dir.x).add(p.y.mul(dir.y)), side = p.x.mul(dir.y).sub(p.y.mul(dir.x));
    const crescent = max(sdCircle(p.add(dir.mul(0.3)), 0.56), sdCircle(p.add(dir.mul(0.46)), 0.6).negate());
    const reveal = coverage(side.sub(mix(float(-0.7), float(0.7), smoothstep(0, 0.45, age))));
    const slashMask = coverage(crescent).mul(reveal);
    const whiff = coverage(abs(length(vec2(along.add(0.15), side)).sub(0.5)).sub(0.012)).mul(smoothstep(-0.1, 0.3, along)).mul(0.6);
    const puffR = pow(age, 0.5).mul(0.7);
    let motes: N<'float'> = float(9);
    for (let k = 0; k < 7; k++) {
      const a = (k / 7) * Math.PI * 2 + 0.3;
      motes = min(motes, sdCircle(p.sub(vec2(Math.cos(a), Math.sin(a)).mul(puffR.mul(1.1))), fade.mul(0.09).add(0.015)));
    }
    const puff = max(coverage(abs(length(p).sub(puffR)).sub(fade.mul(0.05).add(0.005))), coverage(motes));
    const rise = pow(age, 0.7).mul(0.55), label = text(p.sub(vec2(0, rise.add(0.42))).div(0.34), codes, meta.x).mul(0.34);
    const number = coverage(label);
    const numberInk = coverage(label.sub(0.035));
    let glints: N<'float'> = float(9);
    for (let k = 0; k < 4; k++) {
      const a = (k / 4) * Math.PI * 2 + 0.7, at = vec2(Math.cos(a), Math.sin(a)).mul(age.mul(0.5).add(0.15)).add(vec2(0, age.mul(0.25)));
      const q = p.sub(at), s = fade.mul(0.1).add(0.02);
      glints = min(glints, min(sdBox(q, vec2(s, s.mul(0.15))), sdBox(q, vec2(s.mul(0.15), s))));
    }
    const swirl = coverage(spiral(vec2(p.x.mul(cos(spin.add(age.mul(8)))).sub(p.y.mul(sin(spin.add(age.mul(8))))), p.x.mul(sin(spin.add(age.mul(8)))).add(p.y.mul(cos(spin.add(age.mul(8)))))), age.mul(1.2).add(0.4)));
    const zAt = p.sub(vec2(age.mul(0.3).add(0.1), age.mul(0.8).add(0.25)));
    const zee = min(min(sdBox(zAt.sub(vec2(0, 0.1)), vec2(0.1, 0.018)), sdBox(zAt.add(vec2(0, 0.1)), vec2(0.1, 0.018))), sdSegment(zAt, vec2(-0.09, -0.1), vec2(0.09, 0.1), 0.018));
    let dusts: N<'float'> = float(9);
    for (let k = 0; k < 4; k++) dusts = min(dusts, sdCircle(p.sub(dir.mul(0.45)).sub(vec2(k - 1.5, (k % 2) - 0.5).mul(age.mul(0.25).add(0.06))), fade.mul(0.06).add(0.01)));
    const stone = tomb(p.sub(vec2(0, smoothstep(0, 0.3, age).mul(0.1))), 0.9);
    const grave = coverage(stone).mul(smoothstep(0, 0.15, age));
    const cut = coverage(cross(p.sub(vec2(0, 0.14)), 0.28));
    const token = icon(p.sub(vec2(0, age.mul(0.7).add(0.35))), code, 0.6);
    const layers = new Layers();
    layers.over(select(is(FX.slash), slashMask.mul(fade), float(0)), colour.mul(2));
    layers.over(select(is(FX.whiff), whiff.mul(fade), float(0)), colour);
    layers.over(select(is(FX.puff), puff.mul(fade), float(0)), colour.mul(1.5));
    layers.over(select(is(FX.sparkle), coverage(glints), float(0)), colour.mul(2.2));
    layers.over(select(is(FX.swirl), swirl.mul(fade).mul(0.9), float(0)), colour.mul(1.8));
    layers.over(select(is(FX.rest), coverage(zee).mul(fade), float(0)), colour.mul(1.4));
    layers.over(select(is(FX.dust), coverage(dusts).mul(fade).mul(0.7), float(0)), colour);
    layers.over(select(is(FX.tomb), coverage(stone.sub(0.05)).mul(smoothstep(0, 0.15, age)).mul(pow(fade, 0.4)), float(0)), ink);
    layers.over(select(is(FX.tomb), grave.mul(pow(fade, 0.4)), float(0)), mix(rgb(0x7d7a80), rgb(0xa9a4aa), smoothstep(-0.4, 0.4, p.y)));
    layers.over(select(is(FX.tomb), cut.mul(grave).mul(pow(fade, 0.4)), float(0)), rgb(0x3a363c));
    layers.over(select(is(FX.icon), coverage(token.sub(0.05)).mul(fade), float(0)), ink);
    layers.over(select(is(FX.icon), coverage(token).mul(fade), float(0)), colour.mul(1.6));
    layers.over(select(is(FX.number).or(is(FX.rise)), numberInk.mul(pow(fade, 0.6)), float(0)), ink);
    layers.over(select(is(FX.number).or(is(FX.rise)), number.mul(pow(fade, 0.6)), float(0)), colour.mul(1.6));
    return layers.material();
  };
}
