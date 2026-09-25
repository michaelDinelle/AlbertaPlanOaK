import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, sin, smoothstep, step, uv, vec2 } from 'three/tsl';
import { coverage, fractal, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = THREE.UniformNode<'float', number>;
type Heading = THREE.UniformNode<'vec2', THREE.Vector2>;

export const PUCK = 1.75;
export const BODY = 0.58;
export const LURE = 1.8;
export const TRAY = 0.52;
export const RIVAL = 1.7;
export const PLATE = 0.5;
export const DAB = 0.8;
export const MARK = 1.2;
export const WEAR = 1.3;
const LEDS = 12;
const REEDS = 7;

export const PALETTE = {
  floor: [0x656b75, 0x747a85, 0x868c97],
  fleck: 0x3a3e45,
  chalk: 0xaebbd0,
  ink: 0x0b0d12,
  bumper: 0x2a2e36,
  face: 0x0e1117,
  dome: 0x1b2230,
  scent: 0xffcf7a,
  gold: 0xffd166,
  glass: 0xcfe3ee,
  liquid: 0xffb54a,
  cork: 0x6b4a2e,
  tray: [0x2c2622, 0x4a3d33],
  reed: 0x9a7650,
  rival: 0xb98cff,
  steel: 0x2c2838,
  plate: 0x2a2833,
  trace: 0x6f86b0,
  polish: 0x98a2b2,
} as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const unit = (value: N<'float'>) => value.mul(0.5).add(0.5);

const settle = (material: THREE.MeshBasicNodeMaterial, color: N<'vec3'>, ink: N<'float'>, shadow: N<'float'>, fade: N<'float'>) => {
  const alpha = max(ink, shadow);
  material.colorNode = color.mul(ink).div(max(alpha, 1e-4));
  material.opacityNode = alpha.mul(fade);
};

export interface Pools { lure: THREE.UniformNode<'vec4', THREE.Vector4>; rival: THREE.UniformNode<'vec4', THREE.Vector4>; on: Level }

const pool = (p: N<'vec2'>, lamp: THREE.UniformNode<'vec4', THREE.Vector4>, reach: number) => lamp.z.div(dot(p.sub(lamp.xy), p.sub(lamp.xy)).div(reach * reach).add(1));

const pooled = (lit: Lit, pools: Pools, albedo: N<'vec3'>) => {
  const p = vec2(positionWorld.x, positionWorld.z.negate());
  return lit(albedo).add(albedo.mul(rgb(PALETTE.liquid).mul(pool(p, pools.lure, 1)).add(rgb(PALETTE.rival).mul(pool(p, pools.rival, 0.9)))).mul(pools.on));
};

export function floorMaterial(lit: Lit, pools: Pools, half: number): THREE.MeshBasicNodeMaterial {
  const p = vec2(positionWorld.x, positionWorld.z.negate());
  const mott = unit(fractal(p.mul(0.07).add(3.1), 4)), cloud = unit(noise(p.mul(0.38).add(7.7))), sweep = unit(fractal(vec2(p.x.mul(0.05), p.y.mul(0.5)).add(1.9), 3));
  let color: N<'vec3'> = mix(rgb(PALETTE.floor[0]), rgb(PALETTE.floor[1]), smoothstep(0.2, 0.8, mott).mul(0.6));
  color = mix(color, rgb(PALETTE.floor[2]), smoothstep(0.55, 0.95, cloud).mul(0.07));
  color = color.mul(smoothstep(0.35, 0.65, sweep).mul(0.035).add(0.98));
  color = color.mul(grain(floor(p.mul(24))).x.mul(0.14).add(0.93));
  const cell = p.mul(3.2), jitter = grain(floor(cell).add(17.3)), fleck = length(fract(cell).sub(jitter.mul(0.6).add(0.2))).sub(jitter.y.mul(0.05).add(0.03)).div(3.2);
  color = mix(color, rgb(PALETTE.fleck), coverage(fleck).mul(step(jitter.x, 0.3)).mul(0.5));
  const edge = sdBox(p, vec2(half + BODY)), rim = abs(edge.sub(0.1)).sub(0.1);
  color = mix(color, rgb(PALETTE.ink).mul(2.2), coverage(edge.negate()).mul(0.85));
  color = mix(color, rgb(PALETTE.bumper).mul(1.8), coverage(rim));
  color = mix(color, rgb(PALETTE.chalk).mul(0.8), coverage(abs(edge.add(BODY)).sub(0.014)).mul(0.35));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = pooled(lit, pools, color);
  return material;
}

export function wearMaterial(lit: Lit, pools: Pools): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aWear', 'vec4');
  const p = uv().sub(0.5).mul(WEAR), w = vec2(positionWorld.x, positionWorld.z.negate());
  const rough = unit(noise(w.mul(4.1).add(2.3))).mul(0.12);
  const patch = smoothstep(0.52, 0.2, length(p).add(rough));
  const streak = unit(noise(vec2(w.x.mul(1.3), w.y.mul(9)).add(5.1))).mul(0.3).add(0.85);
  material.colorNode = pooled(lit, pools, rgb(PALETTE.polish).mul(streak));
  material.opacityNode = patch.mul(state.x);
  return material;
}

export function markMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(MARK), r = length(p);
  const ring = abs(r.sub(0.4)).sub(0.02), inner = abs(r.sub(0.26)).sub(0.009);
  const cross = min(sdBox(p, vec2(0.17, 0.013)), sdBox(p, vec2(0.013, 0.17)));
  const ticks = min(sdBox(abs(p).sub(vec2(0.5, 0)), vec2(0.055, 0.013)), sdBox(abs(p).sub(vec2(0, 0.5)), vec2(0.013, 0.055)));
  const worn = unit(noise(p.mul(9))).mul(0.45).add(0.55);
  material.colorNode = lit(rgb(PALETTE.chalk));
  material.opacityNode = max(max(coverage(ring), coverage(cross)), max(coverage(inner).mul(0.6), coverage(ticks))).mul(worn).mul(0.5);
  return material;
}

export function dabMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.blending = THREE.AdditiveBlending;
  const state = attribute<'vec4'>('aDab', 'vec4');
  const p = uv().sub(0.5).mul(DAB), r2 = dot(p, p);
  const scent = clamp(state.x, 0, 1), rival = clamp(state.y, 0, 1);
  const core = exp(r2.mul(-1 / (2 * 0.05 * 0.05))), bloom = exp(r2.mul(-1 / (2 * 0.15 * 0.15)));
  const glow = rgb(PALETTE.scent).mul(scent).add(rgb(PALETTE.rival).mul(rival));
  material.colorNode = rgb(PALETTE.trace).mul(core.mul(0.26)).add(glow.mul(core.mul(1.15).add(bloom.mul(0.36))));
  material.opacityNode = state.z;
  return material;
}

export interface Lamp { glow: Level; fade: Level; clock: Level; ping: Level }

const ping = (r: N<'float'>, from: number, level: Level) => coverage(abs(r.sub(level.mul(0.86 - from).add(from))).sub(level.oneMinus().mul(0.03).add(0.008))).mul(level.oneMinus().pow(1.5)).mul(step(1e-3, level));

export function lureMaterial(lit: Lit, lamp: Lamp): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(LURE), r = length(p);
  const shadow = smoothstep(-0.12, 0.2, length(p.sub(vec2(0.06, -0.08))).sub(TRAY)).oneMinus().mul(0.5);
  const tray = r.sub(TRAY), bottle = r.sub(0.3), liquid = r.sub(0.235), neck = r.sub(0.1), mouth = r.sub(0.05);
  let reeds: N<'float'> = float(1e3), tips: N<'float'> = float(1e3);
  for (let k = 0; k < REEDS; k++) {
    const base = (k / REEDS) * 2 * Math.PI + 0.35 * Math.sin(k * 2.3), reach = 0.7 + 0.1 * Math.sin(k * 1.7 + 0.4);
    const angle = sin(lamp.clock.mul(0.6 + 0.13 * k).add(k)).mul(0.05).add(base), way = vec2(cos(angle), sin(angle));
    reeds = min(reeds, sdSegment(p, way.mul(0.08), way.mul(reach), 0.019));
    tips = min(tips, sdCircle(p.sub(way.mul(reach)), 0.042));
  }
  const glint = exp(length(p.sub(vec2(-0.12, 0.12))).mul(-12));
  const glow = lamp.glow.mul(0.4).add(0.8);
  const fill = mix(rgb(PALETTE.liquid).mul(0.62), rgb(0xffdc96), exp(r.mul(-16)).mul(0.7)).mul(glow);
  const glaze = mix(rgb(PALETTE.tray[0]), rgb(PALETTE.tray[1]), smoothstep(0.2, TRAY, r)).mul(smoothstep(-0.4, 0.4, p.x.sub(p.y)).mul(-0.25).add(1.1));
  let color: N<'vec3'> = rgb(PALETTE.ink);
  color = mix(color, glaze.add(rgb(PALETTE.liquid).mul(exp(r.mul(-6)).mul(0.25).mul(glow))), coverage(tray.add(0.014)));
  color = mix(color, rgb(PALETTE.tray[1]).mul(1.5), coverage(abs(tray.add(0.04)).sub(0.01)).mul(0.6));
  color = mix(color, rgb(PALETTE.ink), coverage(bottle));
  color = mix(color, rgb(PALETTE.glass).mul(0.32).add(rgb(PALETTE.liquid).mul(0.22)).add(rgb(PALETTE.glass).mul(glint.mul(0.5))), coverage(bottle.add(0.014)));
  color = mix(color, fill, coverage(liquid));
  color = mix(color, rgb(0xfff6e6).mul(0.9), coverage(abs(bottle.add(0.034)).sub(0.011)).mul(0.6));
  color = mix(color, rgb(PALETTE.cork).mul(smoothstep(-0.08, 0.08, p.x.add(p.y)).mul(-0.4).add(1.15)), coverage(neck));
  color = mix(color, rgb(PALETTE.ink), coverage(mouth));
  color = mix(color, rgb(PALETTE.ink), coverage(reeds.sub(0.012)));
  color = mix(color, lit(rgb(PALETTE.reed)).add(rgb(PALETTE.reed).mul(0.3)), coverage(reeds));
  color = mix(color, rgb(PALETTE.scent).mul(lamp.glow.mul(1.2).add(1.1)), coverage(tips));
  const ring = ping(r, TRAY, lamp.ping);
  settle(material, mix(color, rgb(PALETTE.scent).mul(1.6), ring.mul(coverage(tray).oneMinus())), max(coverage(min(tray, min(reeds.sub(0.012), tips))), ring), shadow, lamp.fade);
  return material;
}

export function rivalMaterial(lamp: Lamp): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(RIVAL), r = length(p);
  const shadow = smoothstep(-0.12, 0.2, length(p.sub(vec2(0.06, -0.08))).sub(PLATE)).oneMinus().mul(0.5);
  const plate = r.sub(PLATE), can = sdBox(p, vec2(0.3), 0.1), band = sdBox(p, vec2(0.2), 0.06), vent = r.sub(0.13);
  const stripes = coverage(abs(fract(p.x.add(p.y).mul(5.5)).sub(0.5)).sub(0.25).div(5.5));
  const hazard = mix(rgb(PALETTE.steel).mul(0.8), rgb(PALETTE.rival).mul(0.75), stripes);
  const bolts = sdCircle(vec2(abs(p.x), abs(p.y)).sub(vec2(PLATE * 0.72 * Math.SQRT1_2)), 0.03);
  const slit = sdBox(vec2(p.x, mod(p.y.add(0.04), 0.08).sub(0.04)), vec2(0.09, 0.012), 0.008);
  const glow = lamp.glow.mul(0.5).add(0.85);
  const core = mix(rgb(PALETTE.rival).mul(0.8), rgb(0xf2e6ff), exp(r.mul(-22))).mul(glow);
  const steel = rgb(PALETTE.plate).mul(smoothstep(-0.5, 0.5, p.x.sub(p.y)).mul(-0.3).add(1.1)).add(rgb(PALETTE.rival).mul(exp(r.mul(-5)).mul(0.18).mul(glow)));
  let color: N<'vec3'> = rgb(PALETTE.ink);
  color = mix(color, steel, coverage(plate.add(0.014)));
  color = mix(color, rgb(PALETTE.rival).mul(0.55), coverage(abs(plate.add(0.042)).sub(0.012)).mul(0.8));
  color = mix(color, rgb(PALETTE.ink).mul(1.8), coverage(bolts));
  color = mix(color, rgb(PALETTE.ink), coverage(can));
  color = mix(color, hazard.mul(smoothstep(-0.3, 0.3, p.x.sub(p.y)).mul(-0.3).add(1.1)).mul(0.85), coverage(can.add(0.016)));
  color = mix(color, rgb(PALETTE.steel).mul(1.2), coverage(band));
  color = mix(color, rgb(PALETTE.ink), coverage(abs(band).sub(0.009)));
  color = mix(color, core, coverage(vent));
  color = mix(color, rgb(PALETTE.ink).mul(1.6), coverage(slit).mul(coverage(vent.add(0.012))));
  const ring = ping(r, PLATE, lamp.ping);
  settle(material, mix(color, rgb(PALETTE.rival).mul(1.6), ring.mul(coverage(plate).oneMinus())), max(coverage(plate), ring), shadow, lamp.fade);
  return material;
}

export interface Puck { heading: Heading; scent: Level; rival: Level; sniff: Level; flash: Level }

const LED_RING = BODY * 0.66;

export function puckMaterial(accent: number, competing: boolean, lit: Lit, puck: Puck): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(PUCK), r = length(p);
  const ahead = puck.heading, q = vec2(dot(p, vec2(ahead.y, ahead.x.negate())), dot(p, ahead));
  const shadow = smoothstep(-0.16, 0.2, length(p.sub(vec2(0.07, -0.1))).sub(BODY)).oneMinus().mul(0.55);
  const glowing = (albedo: N<'vec3'>, self: number) => lit(albedo).add(albedo.mul(self));
  const top = r.sub(BODY - 0.1), angle = atan(q.x, q.y);
  const sector = (2 * Math.PI) / LEDS, k = floor(angle.add(Math.PI).div(sector)), centre = k.add(0.5).mul(sector).sub(Math.PI);
  const led = length(q.sub(vec2(sin(centre), cos(centre)).mul(LED_RING))).sub(0.052);
  const amber = competing ? step(0.5, mod(k, 2)) : float(1);
  const level = mix(clamp(puck.rival, 0, 1), clamp(puck.scent, 0, 1), amber);
  const hue = mix(rgb(PALETTE.rival), rgb(PALETTE.scent), amber);
  const face = max(abs(r.sub(BODY - 0.05)).sub(0.052), abs(angle).sub(0.62));
  const sensor = max(abs(r.sub(BODY - 0.05)).sub(0.015), abs(angle).sub(0.4));
  const light = dot(p, vec2(-0.55, 0.83)).div(BODY);
  const shell = rgb(accent).mul(light.mul(0.2).add(1)).mul(r.div(BODY).pow(2).mul(-0.22).add(1.04));
  const glow = hue.mul(level.mul(level).mul(3.2).add(level.mul(1.1)).add(0.07)).add(rgb(0xfff4dc).mul(puck.flash.mul(0.9)));
  let color: N<'vec3'> = rgb(PALETTE.ink);
  color = mix(color, glowing(rgb(PALETTE.bumper).mul(light.mul(0.35).add(1)), 0.25), coverage(r.sub(BODY - 0.014)));
  color = mix(color, glowing(shell, 0.4), coverage(top));
  color = mix(color, glowing(rgb(PALETTE.face), 0.2).add(rgb(0x9fb4d8).mul(smoothstep(0.3, -0.3, q.x).mul(0.1))), coverage(face));
  color = mix(color, rgb(PALETTE.scent).mul(puck.sniff.mul(2.4).add(0.3)), coverage(sensor));
  color = mix(color, rgb(PALETTE.ink).mul(1.3), coverage(led.sub(0.02)));
  color = mix(color, glow, coverage(led));
  color = mix(color, rgb(PALETTE.dome).add(rgb(0xcfe0ff).mul(exp(length(p.sub(vec2(-0.05, 0.065))).mul(-28)).mul(0.7))), coverage(r.sub(0.18)));
  color = mix(color, rgb(PALETTE.bumper).mul(0.8), coverage(abs(r.sub(0.18)).sub(0.015)));
  settle(material, color, coverage(r.sub(BODY)), shadow, float(1));
  return material;
}
