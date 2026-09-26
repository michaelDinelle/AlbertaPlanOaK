import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, sqrt, step, texture, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = N<'float'>;

export const HEDGE = 0.9;
export const PATH = 1;
export const SPLAT = 1.8;
export const PRINT = 0.5;
export const WALKER = 1.7;
export const RIPPLE = 3.6;
export const LANTERN = 1.7;
const SIGMA = 0.36;
const SHORE = 0.5;
const CANOPY = 0.5;
const PANELS = 8;
const SAG = 0.07;

export const PALETTE = {
  lawn: [0x5a8440, 0x3e6230, 0x6f9a4c],
  hedge: [0x1f4221, 0x356b31, 0x69a64f],
  gravel: [0x9a9384, 0x6f6a5f, 0x4a4740],
  soil: 0x14120f,
  night: 0x0b0e14,
  wood: [0xb07a45, 0x86582f],
  seam: 0x3a2412,
  coir: [0x9c7644, 0x6f5130],
  water: [0x7ba2bf, 0x2c506b],
  mud: 0x2a3524,
  sheen: 0xd6eef6,
  cream: 0xf2eee4,
  rib: 0x2b3640,
  ink: 0x12171c,
  boot: 0xf2c14e,
  flame: 0xffc987,
  iron: 0x2a2e34,
  gold: 0xffd166,
} as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const unit = (value: N<'float'>) => value.mul(0.5).add(0.5);

const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

export function groundMaterial(half: number, porch: number, lit: Lit, wet: THREE.Texture, clock: Level): THREE.MeshBasicNodeMaterial {
  const p = vec2(positionWorld.x, positionWorld.z.negate());
  const field = sdBox(p, vec2(half, half));

  const density = texture(wet, vec2(p.x, p.y.negate()).div(2 * half).add(0.5)).r;
  const water = coverage(float(SHORE).sub(density)).mul(coverage(field));
  const deep = smoothstep(SHORE, SHORE * 6, density);

  const mott = unit(fractal(p.mul(0.14).add(3.7), 4)), clump = unit(noise(p.mul(1.6).add(11.3))), blade = grain(floor(p.mul(26))).x;
  let lawn: N<'vec3'> = mix(rgb(PALETTE.lawn[1]), rgb(PALETTE.lawn[0]), smoothstep(0.38, 0.62, mott));
  lawn = mix(lawn, rgb(PALETTE.lawn[2]), smoothstep(0.55, 0.9, clump).mul(0.28));
  lawn = lawn.mul(blade.mul(0.16).add(0.92)).mul(smoothstep(-0.8, 0, field).mul(-0.42).add(1));
  lawn = lawn.mul(smoothstep(SHORE * 0.3, SHORE, density).mul(-0.35).add(1));

  const swell = unit(noise(p.mul(0.9).add(vec2(clock.mul(0.06), clock.mul(-0.045)))));
  let pool: N<'vec3'> = mix(rgb(PALETTE.water[0]), rgb(PALETTE.water[1]), deep.mul(0.7).add(swell.mul(0.3)));
  pool = mix(pool, rgb(PALETTE.sheen), smoothstep(SHORE * 1.5, SHORE, density).mul(0.4));
  const drizzle = p.mul(1.5), h = grain(floor(drizzle).add(21.7)), phase = fract(clock.mul(0.8).add(h.x));
  const ring = abs(length(fract(drizzle).sub(h.mul(0.5).add(0.25))).sub(phase.mul(0.22).add(0.03))).sub(0.03);
  pool = mix(pool, rgb(PALETTE.sheen), coverage(ring).mul(phase.oneMinus().pow(2)).mul(step(h.y, 0.55)).mul(0.6));

  const leaf = unit(fractal(p.mul(1.9).add(5.1), 3)), sprig = unit(noise(p.mul(5.5).add(1.9)));
  const across = clamp(field.div(HEDGE), 0, 1), crown = sqrt(max(sin(across.mul(Math.PI)), 0));
  let hedge: N<'vec3'> = mix(rgb(PALETTE.hedge[0]), rgb(PALETTE.hedge[1]), smoothstep(0.3, 0.7, leaf));
  hedge = mix(hedge, rgb(PALETTE.hedge[2]), smoothstep(0.58, 0.8, sprig).mul(crown).mul(0.75));
  hedge = hedge.mul(crown.mul(0.75).add(0.35));
  const outer = sdBox(p, vec2(half + HEDGE, half + HEDGE), 0.25).add(leaf.sub(0.5).mul(0.2));

  const pebbles = (scale: number, seed: number) => {
    const at = p.mul(scale).add(seed), jitter = grain(floor(at).add(seed)), offset = fract(at).sub(jitter.mul(0.5).add(0.25));
    return [length(offset.mul(vec2(1, jitter.x.mul(0.5).add(0.75)))).sub(jitter.y.mul(0.14).add(0.2)).div(scale), jitter.x, offset] as const;
  };
  const [coarse, tone, lean] = pebbles(3.1, 3.7), [fine, speck] = pebbles(5.3, 11.9);
  const stone = min(coarse, fine), shade = smoothstep(0.3, -0.3, lean.x.sub(lean.y)).mul(0.25).add(0.85);
  let path: N<'vec3'> = mix(rgb(PALETTE.gravel[1]), rgb(PALETTE.gravel[0]), mix(speck, tone, step(coarse, fine))).mul(shade).mul(unit(fractal(p.mul(0.7).add(9.1), 3)).mul(0.2).add(0.88));
  path = mix(rgb(PALETTE.gravel[2]), path, coverage(stone).mul(0.6).add(0.4));
  path = path.mul(smoothstep(-0.1, 0.7, outer).mul(0.5).add(0.5));
  const verge = outer.sub(PATH).add(unit(noise(p.mul(1.3))).sub(0.5).mul(0.25));
  const beyond = mix(rgb(PALETTE.soil), rgb(PALETTE.night), smoothstep(0, 5, verge));
  const pave = mix(path, beyond, smoothstep(-0.04, 0.04, verge));

  const centre = (porch + half) / 2, extent = (half - porch) / 2, deck = sdBox(p.sub(vec2(centre, centre)), vec2(extent, extent));
  const plank = extent / 2.5, board = p.y.sub(porch).div(plank), course = floor(board);
  const seam = abs(fract(board).sub(0.5)).oneMinus().sub(0.5).mul(plank);
  const figure = unit(noise(vec2(p.x.mul(1.2).add(course.mul(7.3)), p.y.mul(14))));
  let wood: N<'vec3'> = mix(rgb(PALETTE.wood[1]), rgb(PALETTE.wood[0]), figure.mul(0.6).add(grain(vec2(course, 2.2)).x.mul(0.4)));
  wood = wood.mul(smoothstep(0, 0.5, fract(board)).mul(0.12).add(0.9));
  wood = mix(wood, rgb(PALETTE.seam), coverage(seam.sub(0.012)));
  const mat = sdBox(p.sub(vec2(centre, centre)), vec2(extent * 0.56, extent * 0.36), extent * 0.06);
  const weave = abs(fract(p.x.div(0.035)).sub(0.5));
  wood = mix(wood, mix(rgb(PALETTE.coir[1]), rgb(PALETTE.coir[0]), smoothstep(0.1, 0.35, weave)), coverage(mat));
  wood = mix(wood, rgb(PALETTE.coir[1]).mul(0.6), coverage(abs(mat.add(0.02)).sub(0.008)).mul(coverage(mat)));
  const drop = sdBox(p.sub(vec2(centre + 0.07, centre - 0.09)), vec2(extent, extent), 0.04);
  lawn = lawn.mul(smoothstep(-0.02, 0.2, drop).mul(0.45).add(0.55));

  let ground: N<'vec3'> = mix(pave, hedge, coverage(outer));
  ground = mix(ground, lawn, coverage(field));
  ground = mix(ground, pool, water);
  ground = mix(ground, wood, coverage(deck));
  const rim = coverage(abs(deck.add(0.04)).sub(0.022)).mul(coverage(deck));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = mix(lit(ground), rgb(PALETTE.gold).mul(1.5), rim.mul(0.9));
  return material;
}

export function splatMaterial(): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });
  const q = uv().sub(0.5).mul(SPLAT);
  material.colorNode = vec3(exp(dot(q, q).mul(-1 / (2 * SIGMA * SIGMA))).mul(attribute<'float'>('aSplat', 'float')), 0, 0);
  material.opacityNode = float(1);
  return material;
}

export function printMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPrint', 'vec4');
  const p = uv().sub(0.5).mul(PRINT).sub(vec2(state.y.mul(0.1), 0));
  const sole = ellipse(p, [0, 0.045], [0.058, 0.095]), heel = ellipse(p, [0, -0.1], [0.048, 0.045]);
  const tread = abs(fract(p.y.div(0.035)).sub(0.5)).mul(0.035).sub(0.006);
  const shape = min(sole, heel);
  const dry = rgb(PALETTE.mud).mul(coverage(tread).mul(0.25).add(0.85)), wet = mix(rgb(PALETTE.water[1]), rgb(PALETTE.water[0]), coverage(tread).mul(0.6));
  material.colorNode = lit(mix(dry, wet, state.z));
  material.opacityNode = coverage(shape).mul(state.x).mul(state.z.mul(0.2).add(0.7));
  return material;
}

export function rippleMaterial(wade: Level, clock: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const r = length(uv().sub(0.5).mul(RIPPLE));
  let rings: N<'float'> = float(0);
  for (let k = 0; k < 3; k++) {
    const phase = fract(clock.mul(0.85).add(k / 3)), radius = phase.mul(1.15).add(0.4);
    rings = rings.add(coverage(abs(r.sub(radius)).sub(phase.oneMinus().mul(0.022).add(0.008))).mul(phase.oneMinus().pow(1.6)));
  }
  material.colorNode = rgb(PALETTE.sheen).mul(0.9);
  material.opacityNode = clamp(rings, 0, 1).mul(wade).mul(0.75);
  return material;
}

export interface Walker {
  facing: THREE.UniformNode<'vec2', THREE.Vector2>;
  spin: THREE.UniformNode<'float', number>;
  stride: THREE.UniformNode<'float', number>;
  alpha: THREE.UniformNode<'float', number>;
  glow: THREE.UniformNode<'float', number>;
  ring: THREE.UniformNode<'float', number>;
}

export function walkerMaterial(accent: number, lit: Lit, walker: Walker): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(WALKER);
  const ahead = walker.facing, side = vec2(ahead.y, ahead.x.negate());
  const body = vec2(dot(p, side), dot(p, ahead));
  const swing = sin(walker.stride.mul(Math.PI)).mul(0.09);
  const boot = (lateral: number, reach: N<'float'>) => sdSegment(body, vec2(lateral, reach.sub(0.065)), vec2(lateral, reach.add(0.065)), 0.055);
  const boots = min(boot(-0.11, swing.add(0.44)), boot(0.11, swing.negate().add(0.44)));

  const r = length(p), theta = atan(p.y, p.x).sub(walker.spin), sector = 2 * Math.PI / PANELS;
  const k = floor(theta.div(sector)), phi = theta.sub(k.mul(sector)).sub(sector / 2);
  const canopy = r.sub(float(CANOPY).mul(cos(phi.mul(Math.PI / sector)).mul(-SAG).add(1)));
  const rib = r.mul(sin(abs(phi).negate().add(sector / 2))).sub(0.006);
  const facet = k.add(0.5).mul(sector).add(walker.spin), light = vec2(-0.55, 0.83);
  const tilt = dot(vec2(cos(facet), sin(facet)), light).mul(r.div(CANOPY)).mul(0.26);
  const cloth = mix(rgb(PALETTE.cream), rgb(accent), mod(k, 2)).mul(tilt.add(1.02).sub(r.div(CANOPY).pow(2).mul(0.12)));
  const hem = coverage(abs(canopy.add(0.02)).sub(0.007)).mul(coverage(canopy));
  const tip = sdCircle(p, 0.036), glint = sdCircle(p.sub(vec2(-0.01, 0.012)), 0.013);
  const shadow = smoothstep(-0.12, 0.14, length(p.sub(vec2(0.08, -0.1))).sub(CANOPY)).oneMinus().mul(0.4);

  const silhouette = min(canopy, boots), ink = coverage(silhouette.sub(0.02));
  let color: N<'vec3'> = rgb(PALETTE.ink);
  color = mix(color, lit(rgb(PALETTE.boot).mul(body.y.mul(0.6).add(0.8))), coverage(boots));
  color = mix(color, rgb(PALETTE.ink), coverage(canopy.sub(0.016)));
  color = mix(color, lit(cloth), coverage(canopy));
  color = mix(color, lit(rgb(accent).mul(0.55)), hem.mul(0.6));
  color = mix(color, lit(rgb(PALETTE.rib)), coverage(rib).mul(coverage(canopy)).mul(smoothstep(0.03, 0.08, r)).mul(0.7));
  color = mix(color, lit(rgb(PALETTE.iron)), coverage(tip));
  color = mix(color, rgb(PALETTE.sheen), coverage(glint).mul(0.8));
  color = mix(color, rgb(0xfff1cf).mul(1.6), walker.glow);
  const shape = max(ink, shadow), figure = shape.mul(walker.alpha);
  const halo = coverage(abs(r.sub(walker.ring.oneMinus().mul(0.3).add(CANOPY))).sub(0.018)).mul(walker.ring);
  material.colorNode = select(halo.greaterThan(figure), rgb(accent).mul(1.5), color.mul(ink).div(max(shape, 1e-4)));
  material.opacityNode = max(figure, halo);
  return material;
}

export function lanternMaterial(flare: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(LANTERN), r = length(p);
  const post = sdBox(p, vec2(0.4, 0.4), 0.06), globe = sdCircle(p, 0.26), collar = abs(sdCircle(p, 0.28)).sub(0.026);
  const stone = mix(rgb(0x928c80), rgb(0x5a564e), smoothstep(-0.4, 0.4, p.x.sub(p.y).mul(0.7))).mul(unit(noise(p.mul(12))).mul(0.2).add(0.9));
  const shine = exp(length(p.sub(vec2(-0.08, 0.09))).mul(-8));
  const bulb = mix(rgb(PALETTE.flame), rgb(0xfff4dc), shine).mul(flare.mul(0.6).add(1.6));
  let color: N<'vec3'> = mix(stone, stone.mul(0.55), coverage(abs(post.add(0.025)).sub(0.01)));
  color = color.add(rgb(PALETTE.flame).mul(exp(max(globe, 0).mul(-12)).mul(flare.mul(0.5).add(0.7))));
  color = mix(color, bulb, coverage(globe));
  color = mix(color, rgb(PALETTE.iron), coverage(collar));
  const halo = exp(r.mul(-3.6)).mul(flare.mul(0.5).add(0.5));
  material.colorNode = mix(rgb(PALETTE.flame).mul(0.9), color, coverage(post));
  material.opacityNode = max(coverage(post), halo.mul(0.45));
  return material;
}

export function rainMaterial(half: number, clock: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.blending = THREE.AdditiveBlending;
  const p = vec2(positionWorld.x, positionWorld.z.negate());
  let streaks: N<'float'> = float(0);
  for (const [scale, speed, seed] of [[0.9, 7.5, 1.3], [1.35, 10, 7.9]] as const) {
    const q = p.mul(scale).add(vec2(0.28, 1).mul(clock.mul(speed))), h = grain(floor(q).add(seed));
    const c = fract(q).sub(vec2(h.y.mul(0.6).add(0.2), 0.5));
    streaks = streaks.add(coverage(sdSegment(c, vec2(-0.03, -0.11), vec2(0.03, 0.11), 0.008)).mul(step(h.x, 0.22)));
  }
  material.colorNode = rgb(PALETTE.sheen);
  material.opacityNode = clamp(streaks, 0, 1).mul(smoothstep(half + HEDGE + PATH + 1.5, half + HEDGE, max(abs(p.x), abs(p.y)))).mul(0.16);
  return material;
}
