import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, pow, select, sin, smoothstep, uniform, uv, vec2, vec3 } from 'three/tsl';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { sheet } from './fleet';
import { coverOf, HALF, type Level, type N } from './sector';

export const BEAM_WIDTH = 0.7;
export const BEAM_PAD = 0.26;
export const SPARK = 3.4;
export const BEACON = 1.6;

export const BOLT = 0;
export const ORDERED = 1;
export const LINK = 2;
export const VOID = 3;

export const IMPACT = 0;
export const BLAST = 1;
export const WARP = 2;
export const RECALL = 3;
export const REVEAL = 4;
export const SPLASH = 5;

export const RELIC = 0;
export const NODE = 1;

const GOLD = 0xffd27a;
const GOLD_DEEP = 0x6b4512;
const CORE = 0xe6fffb;
const CHARGE = 0x3fe0cf;
const WHITE = vec3(1.0, 0.98, 0.95);
const DRAIN_LINK = 0xe2c6ff;

const teamTone = (team: N<'float'>, accent: number, foe: number): N<'vec3'> => select(team.lessThan(0.5), rgb(accent), rgb(foe));

export function beamMaterial(accent: number, foe: number, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.blending = THREE.AdditiveBlending;
  const state = attribute<'vec4'>('aBeam', 'vec4');
  const span = state.x, style = state.y, team = state.z, progress = state.w;
  const along = uv().x.sub(0.5).mul(span.add(2 * BEAM_PAD)), across = uv().y.sub(0.5).mul(BEAM_WIDTH);
  const s = along.add(span.mul(0.5)), t = clamp(s.div(max(span, 1e-3)), 0, 1), aa = fwidth(across);
  const cover = coverOf(aa), tone = teamTone(team, accent, foe);
  const reach = smoothstep(0, 0.25, progress), fade = smoothstep(0.45, 1, progress).oneMinus();
  const crackle = sin(s.mul(9.1).add(time.mul(37))).mul(0.5).add(sin(s.mul(23.7).sub(time.mul(53))).mul(0.3));
  const envelope = sin(t.mul(Math.PI)).mul(0.13);
  const bolt = abs(across.sub(crackle.mul(envelope))).sub(0.028);
  const drawn = select(t.lessThan(reach), float(1), float(0)).mul(select(s.greaterThan(0).and(s.lessThan(span)), float(1), float(0)));
  const core = cover(bolt).mul(drawn).mul(fade);
  const glow = exp(abs(across).mul(-9)).mul(drawn).mul(fade).mul(0.55).mul(smoothstep(BEAM_WIDTH / 4, BEAM_WIDTH / 2, abs(across)).oneMinus());
  const dash = select(fract(s.mul(2.2).sub(time.mul(1.6))).lessThan(0.55), float(1), float(0));
  const line = cover(abs(across).sub(0.022)).mul(dash).mul(select(s.greaterThan(0.3).and(s.lessThan(span.sub(0.26))), float(1), float(0)));
  const mark = cover(abs(length(vec2(s.sub(span), across)).sub(0.2)).sub(0.024));
  const ordered = max(line.mul(0.75), mark).mul(progress);
  const link = line.mul(progress);
  const wave = sin(s.mul(12).sub(time.mul(8))).mul(0.05).mul(sin(t.mul(Math.PI)));
  const tether = cover(abs(across.sub(wave)).sub(0.016)).mul(select(s.greaterThan(0.24).and(s.lessThan(span.sub(0.24))), float(1), float(0))).mul(progress);
  const bright = select(style.lessThan(BOLT + 0.5), core.add(glow), select(style.lessThan(ORDERED + 0.5), ordered, select(style.lessThan(LINK + 0.5), link, tether)));
  material.colorNode = select(style.lessThan(BOLT + 0.5), mix(tone.mul(1.3), WHITE.mul(1.6), core), select(style.lessThan(LINK + 0.5), tone.mul(1.25), rgb(DRAIN_LINK)));
  material.opacityNode = clamp(bright, 0, 1);
  return material;
}

export function sparkMaterial(accent: number, foe: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.blending = THREE.AdditiveBlending;
  const state = attribute<'vec4'>('aSpark', 'vec4');
  const kind = state.x, t = clamp(state.y, 0, 1), team = state.z, seed = state.w;
  const p = uv().sub(0.5).mul(SPARK), cover = coverOf(fwidth(p.x)), r = length(p), angle = atan(p.y, p.x);
  const tone = teamTone(team, accent, foe), fade = pow(t.oneMinus(), 1.5);
  const is = (k: number) => select(abs(kind.sub(k)).lessThan(0.5), float(1), float(0));

  const flash = cover(sdBox(p, vec2(0.46), 0.08)).mul(exp(t.mul(-7)));
  const shock = cover(abs(r.sub(t.mul(1.3).add(0.15))).sub(fade.mul(0.05).add(0.01))).mul(fade);
  const board = select(abs(positionWorld.x).lessThan(HALF).and(abs(positionWorld.z).lessThan(HALF)), float(1), float(0));
  const zone = cover(abs(sdBox(p, vec2(1.46), 0.06)).sub(0.03)).mul(exp(t.mul(-4))).mul(0.8).mul(board);
  const impact = max(max(flash, shock), zone);

  const splash = cover(sdBox(p, vec2(0.4), 0.1)).mul(exp(t.mul(-6))).mul(seed.mul(0.6));

  const shard = floor(angle.div(2 * Math.PI).mul(9).add(seed.mul(9)));
  const tilt = sin(shard.mul(12.9898).add(seed.mul(78.233))).mul(0.5).add(0.5);
  const flight = t.mul(tilt.mul(0.7).add(0.6));
  const streak = sdSegment(vec2(r, fract(angle.div(2 * Math.PI).mul(9).add(seed.mul(9))).sub(0.5).mul(r.mul(0.7))), vec2(flight.mul(0.9), 0), vec2(flight.mul(1.25).add(0.05), 0), fade.mul(0.035).add(0.008));
  const fireball = exp(r.div(t.mul(0.5).add(0.12)).negate()).mul(exp(t.mul(-4))).mul(1.4);
  const blast = max(cover(streak).mul(fade), fireball);

  const iris = cover(abs(r.sub(t.oneMinus().mul(0.9).add(0.05))).sub(0.03)).mul(smoothstep(0, 0.2, t)).mul(fade);
  const warp = max(iris, exp(r.mul(-6)).mul(exp(abs(t.sub(0.75)).mul(-9))).mul(1.2));

  const rise = cover(abs(r.sub(t.mul(0.6).add(0.1))).sub(0.02)).mul(fade).mul(0.8);
  const recall = max(rise, exp(r.mul(-5)).mul(fade).mul(0.5));

  const rim = smoothstep(SPARK / 2 - 0.45, SPARK / 2, r).oneMinus();
  const rays = pow(abs(cos(angle.mul(6).add(t.mul(3)))), 12).mul(exp(r.mul(-1.4))).mul(fade).mul(rim);
  const halo = cover(abs(r.sub(t.mul(1.6).add(0.2))).sub(fade.mul(0.06).add(0.01))).mul(fade).mul(rim);
  const reveal = max(halo, rays.mul(0.8));

  const bright = impact.mul(is(IMPACT)).add(splash.mul(is(SPLASH))).add(blast.mul(is(BLAST))).add(warp.mul(is(WARP))).add(recall.mul(is(RECALL))).add(reveal.mul(is(REVEAL)));
  const hot = mix(tone.mul(1.4), WHITE.mul(1.5), exp(t.mul(-5)).mul(is(IMPACT).add(is(BLAST)).add(is(SPLASH))));
  material.colorNode = select(is(REVEAL).greaterThan(0.5), rgb(GOLD).mul(1.6), hot);
  material.opacityNode = clamp(bright, 0, 1);
  return material;
}

export function beaconMaterial(time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aBeacon', 'vec4');
  const kind = state.x, grown = clamp(state.y, 0, 1), heat = state.z, phase = state.w;
  const p = uv().sub(0.5).mul(BEACON), cover = coverOf(fwidth(p.x)), r = length(p), angle = atan(p.y, p.x);
  const size = smoothstep(0, 1, grown).mul(0.25).add(0.75).mul(select(grown.greaterThan(0.001), float(1), float(0)));
  const q = p.div(max(size, 1e-3));
  const diamond = abs(q.x).add(abs(q.y));
  const crystal = diamond.sub(0.3);
  const facet = select(q.x.greaterThan(0), float(0.75), float(1.2)).mul(select(q.y.greaterThan(0), float(1.1), float(0.8)));
  const inner = abs(diamond.sub(0.16)).sub(0.018);
  const turning = angle.add(time.mul(0.35)).add(phase);
  const crown = max(abs(r.sub(0.52)).sub(0.03), sin(turning.mul(4)).sub(0.25));
  let relic: N<'vec3'> = mix(rgb(GOLD_DEEP), rgb(GOLD).mul(facet), cover(crystal));
  relic = mix(relic, rgb(0xfff4d8).mul(1.6).mul(heat.mul(0.6).add(1)), cover(inner));
  const relicShape = max(cover(crystal.sub(0.035)), cover(crown).mul(0.85));
  const relicColor = select(cover(crystal).greaterThan(0.01), relic, rgb(GOLD).mul(heat.mul(0.8).add(1.1)));
  const relicGlow = exp(max(crystal, 0).mul(-7)).mul(heat.mul(0.4).add(0.25));

  const swirl = sin(angle.mul(3).sub(r.mul(14)).add(time.mul(2.2)).add(phase)).mul(0.5).add(0.5);
  const well = exp(r.mul(-9)).mul(1.8).add(pow(swirl, 3).mul(exp(r.mul(-3.2))).mul(0.55));
  const orbit = cover(abs(r.sub(0.34)).sub(0.012)).mul(sin(angle.mul(2).add(time.mul(-1.3)).add(phase)).mul(0.5).add(0.5)).mul(0.7);
  const nodeColor = mix(rgb(CHARGE).mul(1.1), rgb(CORE).mul(1.5), exp(r.mul(-14)));

  const isRelic = kind.lessThan(RELIC + 0.5);
  material.colorNode = select(isRelic, mix(rgb(GOLD).mul(1.2), relicColor, relicShape), nodeColor);
  const edge = smoothstep(BEACON / 2 - 0.25, BEACON / 2, max(abs(p.x), abs(p.y))).oneMinus();
  material.opacityNode = select(isRelic, max(relicShape, relicGlow.mul(0.8).mul(edge)), clamp(max(well.mul(edge), orbit), 0, 1)).mul(select(grown.greaterThan(0.001), float(1), float(0)));
  return material;
}

export interface Command {
  mesh: THREE.Mesh;
  mode: Level;
  cursor: THREE.UniformNode<'vec2', THREE.Vector2>;
  fence: THREE.UniformNode<'vec4', THREE.Vector4>;
  aim: THREE.UniformNode<'vec2', THREE.Vector2>;
  shown: Level;
  time: Level;
}

export function commandMaterial(accent: number, flat: (w: number, d: number) => THREE.BufferGeometry): Command {
  const material = sheet();
  const mode = uniform(0), cursor = uniform(new THREE.Vector2()), fence = uniform(new THREE.Vector4()), aim = uniform(new THREE.Vector2()), shown = uniform(0), time = uniform(0);
  const w = vec2(positionWorld.x, positionWorld.z.negate()), cover = coverOf(fwidth(w.x));
  const flip = (v: N<'vec2'>) => vec2(v.x, v.y.negate());
  const hue = rgb(accent), bright = hue.mul(1.5).add(0.15);
  const pulse = sin(time.mul(5)).mul(0.5).add(0.5);

  const c = w.sub(flip(cursor)), size = float(0.6).add(pulse.mul(0.02));
  const ac = abs(c);
  const bracket = min(sdBox(ac.sub(vec2(size, size.sub(0.1))), vec2(0.035, 0.12)), sdBox(ac.sub(vec2(size.sub(0.1), size)), vec2(0.12, 0.035)));
  const order = select(mode.greaterThan(0.5).and(mode.lessThan(1.5)), float(1), float(0));

  const half = vec2(fence.z.sub(fence.x), fence.w.sub(fence.y)).mul(0.5);
  const o = w.sub(flip(vec2(fence.x.add(fence.z), fence.y.add(fence.w)).mul(0.5))), box = sdBox(o, half, 0.08);
  const dash = select(fract(o.x.sub(o.y).mul(1.6).sub(time.mul(0.5))).lessThan(0.55), float(1), float(0.3));
  const edge = cover(abs(box).sub(0.03)).mul(dash);
  const field = cover(box).mul(0.07);
  const t = w.sub(flip(aim)), tr = length(t);
  const reticle = min(min(abs(tr.sub(0.36)).sub(0.03), max(abs(tr.sub(0.36)).sub(0.2), min(abs(t.x), abs(t.y)).sub(0.025))), sdCircle(t, 0.05));
  const lock = cover(sdBox(t, vec2(0.5), 0.04).abs().sub(0.02)).mul(0.6);
  const target = select(mode.greaterThan(1.5), float(1), float(0));

  const ink = cover(bracket.sub(0.03)).mul(order).add(cover(reticle.sub(0.03)).mul(target));
  const lines = max(cover(bracket).mul(order), max(max(edge, cover(reticle)), lock).mul(target));
  let color: N<'vec3'> = mix(vec3(0.02, 0.02, 0.05), bright, lines);
  color = mix(color, hue.mul(0.8), field.mul(target).mul(lines.oneMinus()));
  material.colorNode = color;
  material.opacityNode = clamp(max(max(ink.mul(0.7), lines), field.mul(target)), 0, 1).mul(shown);
  const mesh = new THREE.Mesh(flat(HALF * 2 + 16, HALF * 2 + 16), material);
  mesh.renderOrder = 9;
  mesh.frustumCulled = false;
  return { mesh, mode, cursor, fence, aim, shown, time };
}

