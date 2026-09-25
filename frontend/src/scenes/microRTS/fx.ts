import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, length, max, min, mix, pow, select, sin, smoothstep, step, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { ACCENT, GOLD, MINT, Stack, hexagon, octagon, sheet, spin, teamOf, triangle, type Level, type N } from './art';

export const BEAM_WIDTH = 0.64;
export const BEAM_PAD = 0.32;
export const SPARK = 1.2;
export const DECAL = 1;
export const ORDER = 2.4;
export const CURSOR = 1.7;

export const IMPACT = 0;
export const COIN = 1;
export const CHIP = 2;
export const REJECT = 3;
export const SPAWN = 4;
export const WAIT = 5;
export const SHATTER = 6;

export const PENDING = 0;
export const SKIPPED = 1;
export const AWAITING = 2;

export const ICON_NONE = 0;
export const ICON_BASE = 1;
export const ICON_BARRACKS = 2;
export const ICON_WORKER = 3;
export const ICON_LIGHT = 4;
export const ICON_HEAVY = 5;
export const ICON_RANGED = 6;
export const ICON_TARGET = 7;

export function beamMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.blending = THREE.AdditiveBlending;
  const state = attribute<'vec4'>('aBeam', 'vec4');
  const span = state.x, known = step(0, state.y), progress = max(state.y, 0), team = state.z, melee = state.w;
  const total = span.add(2 * BEAM_PAD), p = vec2(uv().x.sub(0.5).mul(total), uv().y.sub(0.5).mul(BEAM_WIDTH));
  const from = span.mul(-0.5), to = span.mul(0.5), tint = teamOf(team).mul(1.5).add(vec3(0.15));
  const charge = smoothstep(0, 1, progress);
  const dash = step(0.45, fract(p.x.mul(6).sub(progress.mul(5))));
  const aim = coverage(abs(p.y).sub(0.011)).mul(step(from.add(0.24), p.x)).mul(step(p.x, to.sub(0.2)));
  const thrust = sdSegment(p, vec2(from.add(0.22), 0), vec2(mix(from.add(0.3), to.sub(0.16), charge), 0), mix(float(0.012), float(0.03), charge));
  const meleeAlpha = max(aim.mul(0.3), coverage(thrust).mul(charge.mul(0.75).add(0.25)).mul(known));
  const flight = clamp(progress.sub(0.55).div(0.45), 0, 1), head = mix(from.add(0.2), to, flight);
  const bolt = sdSegment(p, vec2(head.sub(0.16), 0), vec2(head, 0), 0.028);
  const tail = exp(max(head.sub(p.x), 0).mul(-9)).mul(step(p.x, head)).mul(step(from, p.x)).mul(exp(abs(p.y).mul(-60)));
  const loosed = step(0.55, progress).mul(known);
  const rangedAlpha = max(aim.mul(dash).mul(0.45), max(coverage(bolt), tail.mul(0.8)).mul(loosed));
  const lock = abs(sdCircle(p.sub(vec2(to, 0)), mix(float(0.26), float(0.14), smoothstep(0.1, 1, progress).mul(known)))).sub(0.013);
  const ticks = step(0.2, abs(fract(atan(p.y, p.x.sub(to)).div(Math.PI / 2).add(0.125)).sub(0.5))).mul(coverage(lock));
  material.colorNode = mix(tint, vec3(1.4, 1.3, 1.1), coverage(bolt).mul(loosed).mul(melee.oneMinus()));
  material.opacityNode = max(mix(rangedAlpha, meleeAlpha, melee), ticks.mul(mix(float(0.5), smoothstep(0, 0.15, progress), known)).mul(0.85));
  return material;
}

export function sparkMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aSpark', 'vec4');
  const kind = state.x, age = clamp(state.y, 0, 1), team = state.z, extra = state.w;
  const p = uv().sub(0.5).mul(SPARK), r = length(p), tint = teamOf(team), angle = atan(p.y, p.x);
  const fade = pow(age.oneMinus(), 1.6);

  const ring = abs(r.sub(pow(age, 0.5).mul(0.42).add(0.06))).sub(age.oneMinus().mul(0.035).add(0.004));
  const sector = Math.PI / 4, k = floor(angle.div(sector)).add(0.5), dir = vec2(cos(k.mul(sector)), sin(k.mul(sector)));
  const reach = pow(age, 0.6).mul(0.4).add(0.08);
  const ray = sdSegment(p, dir.mul(reach.mul(0.55)), dir.mul(reach), age.oneMinus().mul(0.02).add(0.004));
  const impact = max(coverage(ring), coverage(ray)).mul(fade).add(exp(r.mul(-18)).mul(age.oneMinus().pow(3)).mul(1.5));
  const impactColor = mix(vec3(1.4, 1.1, 0.8), tint.mul(1.5), smoothstep(0.1, 0.4, r));

  const rise = p.sub(vec2(0, age.mul(0.32).sub(0.05))), squash = abs(sin(age.mul(12))).mul(0.6).add(0.4);
  const coin = hexagon(vec2(rise.x.div(squash), rise.y), 0.07).mul(squash);
  const coinFace = mix(rgb(0x0d3a2c), mix(rgb(MINT).mul(1.3), vec3(1.5), coverage(sdCircle(rise.sub(vec2(-0.02, 0.025)), 0.02))), coverage(coin.add(0.012)));
  const coinAlpha = coverage(coin).mul(smoothstep(1, 0.7, age));

  const drift = p.sub(vec2(0, age.mul(0.18)));
  const chip = min(hexagon(spin(drift.sub(vec2(0.06, 0)), age.mul(6)), 0.04), hexagon(spin(drift.sub(vec2(-0.07, 0.03)), age.mul(-5)), 0.03));
  const chipAlpha = coverage(chip).mul(fade);

  const mark = p.sub(vec2(-0.33, 0.33)), stamp = sdCircle(mark, 0.1);
  const cross = min(sdSegment(mark, vec2(-0.045, -0.045), vec2(0.045, 0.045), 0.017), sdSegment(mark, vec2(-0.045, 0.045), vec2(0.045, -0.045), 0.017));
  const pop = smoothstep(0, 0.08, age).mul(smoothstep(1, 0.7, age));
  const crossAlpha = coverage(stamp).mul(0.95).mul(pop);
  const crossColor = mix(mix(rgb(0x1c0606), rgb(0x5a1414), coverage(stamp.add(0.016))), vec3(1.25, 0.3, 0.26), coverage(cross));

  const halo = abs(r.sub(pow(age, 0.7).mul(0.5).add(0.1))).sub(0.018);
  const spawnAlpha = coverage(halo).mul(fade).add(exp(r.mul(-9)).mul(fade).mul(0.35));

  const badge = p.sub(vec2(0.33, 0.33)), hour = spin(badge, extra);
  const disc = sdCircle(badge, 0.105);
  const vessel = min(triangle(hour, [-0.052, 0.066], [0.052, 0.066], [0, 0.004]), triangle(hour, [0, -0.004], [0.052, -0.066], [-0.052, -0.066]));
  const caps = min(sdBox(hour.sub(vec2(0, 0.07)), vec2(0.062, 0.011), 0.004), sdBox(hour.sub(vec2(0, -0.07)), vec2(0.062, 0.011), 0.004));
  const sand = triangle(hour, [0, -0.028], [0.036, -0.062], [-0.036, -0.062]);
  const waitColor = mix(mix(mix(rgb(0x0b1019), rgb(0x2a3342), coverage(disc.add(0.016))), rgb(0xdfe6f2), coverage(min(vessel, caps))), rgb(GOLD).mul(1.3), coverage(sand));
  const waitAlpha = coverage(disc).mul(0.95);

  const shards = Math.PI / 3, s = floor(angle.div(shards)).add(0.5), sdir = vec2(cos(s.mul(shards)), sin(s.mul(shards)));
  const splinter = hexagon(spin(p.sub(sdir.mul(pow(age, 0.5).mul(0.36).add(0.05))), age.mul(5).add(s)), age.oneMinus().mul(0.05).add(0.01));
  const shatterAlpha = max(coverage(splinter), coverage(ring).mul(0.7)).mul(fade);

  const pick = (values: readonly N<'float'>[]) => values.slice(1).reduce((chosen, value, i) => select(kind.lessThan(i + 0.5), chosen, value), values[0]!);
  const pickColor = (values: readonly N<'vec3'>[]) => values.slice(1).reduce((chosen, value, i) => select(kind.lessThan(i + 0.5), chosen, value), values[0]!);
  material.colorNode = pickColor([impactColor, coinFace, rgb(MINT).mul(1.5), crossColor, tint.mul(1.5).add(vec3(0.15)), waitColor, rgb(MINT).mul(1.4)]);
  material.opacityNode = clamp(pick([impact, coinAlpha, chipAlpha, crossAlpha, spawnAlpha, waitAlpha, shatterAlpha]), 0, 1);
  return material;
}

export const WRECK_UNIT = 0;
export const WRECK_HALL = 1;
export const RESERVED = 2;

export function decalMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aDecal', 'vec4');
  const kind = state.x, age = clamp(state.y, 0, 1), seed = state.z, team = state.w;
  const p = uv().sub(0.5).mul(DECAL), r = length(p), angle = atan(p.y, p.x);
  const size = select(kind.lessThan(0.5), float(0.8), float(1.2));
  const jag = sin(angle.mul(7).add(seed)).mul(0.035).add(sin(angle.mul(13).add(seed.mul(2.3))).mul(0.02));
  const burn = smoothstep(0.06, -0.12, r.sub(size.mul(0.3).add(jag)));
  const ember = exp(r.mul(-14)).mul(age.oneMinus().pow(4)).mul(1.2);
  const debris = sdBox(spin(p.sub(vec2(0.12, -0.08)), seed), vec2(0.05, 0.02), 0.01).min(sdBox(spin(p.sub(vec2(-0.1, 0.1)), seed.mul(1.7)), vec2(0.04, 0.018), 0.01));
  const fade = smoothstep(1, 0.55, age);
  const wreckColor = mix(mix(rgb(0x050608), rgb(0x2a2420), coverage(debris)), teamOf(team).mul(1.4).add(vec3(0.6, 0.3, 0.05)), ember.clamp(0, 1));
  const wreckAlpha = max(burn.mul(0.75), coverage(debris).mul(0.8).mul(step(0.5, kind))).mul(fade).max(ember.mul(fade));
  const outline = abs(sdBox(p, vec2(0.36), 0.08)).sub(0.011);
  const dashes = step(0.5, fract(p.x.add(p.y).mul(9)));
  const reservedAlpha = coverage(outline).mul(dashes).mul(age.mul(0.35).add(0.3));
  const isReserved = kind.greaterThan(1.5);
  material.colorNode = select(isReserved, teamOf(team).mul(1.2), wreckColor);
  material.opacityNode = select(isReserved, reservedAlpha, wreckAlpha);
  return material;
}

const icon = (p: N<'vec2'>, code: N<'float'>) => {
  const base = min(abs(sdBox(p, vec2(0.075), 0.015)).sub(0.014), sdCircle(p, 0.025));
  const barracks = min(abs(min(sdBox(p.sub(vec2(0, -0.02)), vec2(0.07, 0.05), 0.005), triangle(p, [-0.09, 0.03], [0.09, 0.03], [0, 0.1]))).sub(0.012), sdBox(p.sub(vec2(0, -0.05)), vec2(0.018, 0.025), 0.004));
  const worker = min(abs(sdCircle(p, 0.07)).sub(0.014), sdCircle(p.sub(vec2(0, 0.015)), 0.035));
  const light = abs(triangle(p, [0, 0.1], [-0.08, -0.07], [0.08, -0.07])).sub(0.013);
  const heavy = abs(octagon(p, 0.075)).sub(0.014);
  const ranged = min(abs(sdCircle(p.sub(vec2(0, -0.03)), 0.1)).sub(0.013).max(p.y.sub(0.0).negate()), sdSegment(p, vec2(0, -0.06), vec2(0, 0.1), 0.01));
  const target = min(abs(sdCircle(p, 0.07)).sub(0.012), min(sdSegment(p, vec2(-0.11, 0), vec2(0.11, 0), 0.009), sdSegment(p, vec2(0, -0.11), vec2(0, 0.11), 0.009)));
  const choose = [base, barracks, worker, light, heavy, ranged, target];
  return choose.slice(1).reduce((chosen, value, i) => select(code.lessThan(i + 1.5), chosen, value), choose[0]!);
};

export function orderMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aOrder', 'vec4');
  const kind = state.x, direction = state.y, code = state.z, age = state.w;
  const p = uv().sub(0.5).mul(ORDER);
  const turn = select(direction.lessThan(0.5), float(0), select(direction.lessThan(1.5), float(-Math.PI / 2), select(direction.lessThan(2.5), float(Math.PI), float(Math.PI / 2))));
  const q = spin(p, turn);
  const shaft = sdSegment(q, vec2(0, 0.3), vec2(0, 0.74), 0.026);
  const tip = triangle(q, [0, 0.9], [-0.1, 0.72], [0.1, 0.72]);
  const march = step(0.4, fract(q.y.mul(6).sub(age.mul(3))));
  const arrow = min(max(shaft, march.oneMinus().mul(0.05)), tip);
  const targetOnly = step(ICON_TARGET - 0.5, code);
  const badgeAt = select(targetOnly.greaterThan(0.5), p.sub(vec2(0.28, 0.28)), p.sub(vec2(cos(turn.add(Math.PI / 2)), sin(turn.add(Math.PI / 2)))));
  const plate = sdCircle(badgeAt, 0.14), glyph = icon(badgeAt, code);
  const hasIcon = step(0.5, code);
  const bright = rgb(ACCENT).mul(1.45);
  const arrowColor = mix(rgb(0x06101f), bright, coverage(arrow));
  const badgeColor = mix(mix(rgb(0x06101f), rgb(0x10203c), coverage(plate.add(0.02))), vec3(1.25), coverage(glyph));
  const onBadge = coverage(plate).mul(hasIcon);
  const pendingColor = mix(arrowColor, badgeColor, onBadge);
  const pendingAlpha = max(coverage(arrow.sub(0.016)).mul(targetOnly.oneMinus()), onBadge);
  const dot = abs(sdCircle(p.sub(vec2(0.3, 0.3)), 0.055)).sub(0.014);
  const skipAlpha = coverage(dot).mul(0.85);
  const bob = sin(age.mul(6)).mul(0.02);
  const awaitShape = triangle(p.sub(vec2(0, bob.add(0.4))), [0, -0.05], [-0.075, 0.06], [0.075, 0.06]);
  const awaitAlpha = coverage(awaitShape.sub(0.012)).mul(0.95);
  const awaitColor = mix(rgb(0x06101f), rgb(ACCENT).mul(1.4), coverage(awaitShape));
  material.colorNode = select(kind.lessThan(0.5), pendingColor, select(kind.lessThan(1.5), rgb(ACCENT).mul(1.2), awaitColor));
  material.opacityNode = select(kind.lessThan(0.5), pendingAlpha, select(kind.lessThan(1.5), skipAlpha, awaitAlpha));
  return material;
}

export interface Cursor {
  material: THREE.MeshBasicNodeMaterial;
  time: Level;
  live: Level;
  code: Level;
  pips: Level;
  pressed: Level;
  refused: Level;
}

export function cursorMaterial(): Cursor {
  const material = sheet(), time = uniform(0), live = uniform(1), code = uniform(0), pips = uniform(0), pressed = uniform(1), refused = uniform(0);
  const p = uv().sub(0.5).mul(CURSOR);
  const breathe = sin(time.mul(4.2)).mul(0.018).add(0.5).add(pressed.oneMinus().mul(0.06).mul(live));
  const a = abs(p), corner = a.sub(breathe);
  const armX = sdBox(vec2(corner.x.add(0.07), corner.y), vec2(0.1, 0.022), 0.01), armY = sdBox(vec2(corner.x, corner.y.add(0.07)), vec2(0.022, 0.1), 0.01);
  const brackets = min(armX, armY);
  const badgeAt = p.sub(vec2(0.56, 0.56)), plate = sdCircle(badgeAt, 0.15), glyph = icon(badgeAt, code);
  const dots = sdCircle(vec2(fract(badgeAt.x.add(0.5).div(0.06)).sub(0.5).mul(0.06), badgeAt.y.add(0.22)), 0.016).max(abs(badgeAt.x).sub(pips.mul(0.03)));
  const hasIcon = step(0.5, code).mul(live);
  const tint = mix(mix(rgb(0x8b93a3), rgb(ACCENT).mul(1.5), live), vec3(1.35, 0.3, 0.26), refused);
  const glow = exp(max(brackets, 0).mul(-40)).mul(0.35).mul(live);
  new Stack()
    .over(tint.mul(0.6), glow)
    .over(mix(rgb(0x05080f), tint, coverage(brackets)), coverage(brackets.sub(0.014)).mul(live.mul(0.55).add(0.35)))
    .over(mix(rgb(0x05080f), rgb(0x0f1a2e), coverage(plate.add(0.02))), coverage(plate).mul(hasIcon))
    .over(rgb(ACCENT).mul(1.6), coverage(glyph).mul(hasIcon))
    .over(rgb(ACCENT).mul(1.6), coverage(dots).mul(hasIcon).mul(step(0.5, pips)))
    .paint(material, float(1));
  return { material, time, live, code, pips, pressed, refused };
}
