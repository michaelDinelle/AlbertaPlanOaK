import * as THREE from 'three/webgpu';
import { Fn, abs, atan, attribute, clamp, dot, exp, float, floor, fract, length, max, min, mix, normalize, positionWorld, select, sin, smoothstep, step, texture, uv, vec2, vec3 } from 'three/tsl';
import { coverage, grain, rgb, sdCircle, sdSegment } from '@kit/shapes2d';
import { CAP, HALF, type Course } from './course';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Layer = readonly [N<'vec3'>, N<'float'>];
export type Level = THREE.UniformNode<'float', number>;
export type Pair = THREE.UniformNode<'vec2', THREE.Vector2>;
export type Quad = THREE.UniformNode<'vec4', THREE.Vector4>;

export interface Maps { mottle: THREE.Texture; grain: THREE.Texture }

const FLOW = 0.42;
export const SWIMMER = 1.2;
const FIGURE = 1.33;
const SHINE = 1.8;
export const BUOY = 0.5;
export const SPRING = 1.6;
export const SWIRL = 0.7;
export const CAMP = 0.5;
export const SPAN = 1.4;
const DROPS = 9;

const GRASS = [0x1d3120, 0x34512e, 0x57763c] as const;
const SAND = [0x6f5b40, 0xb49a6e, 0xd9c296] as const;
const GRAVEL = 0x77716a;
const WATER = [0x0b2a33, 0x184f58, 0x2f7a78, 0x6fae98] as const;
export const FOAM = 0xe2f2ef;
export const LAMP = 0xffc47a;
export const GOLD = 0xffd27a;
const CAP_COLOR = 0xff7a2f;
const SUIT = 0x1b2c47;
const SKIN = 0xc98f6a;
const BUOY_RED = 0xd4512e;
const LEAF = [0x172a1b, 0x2c4a2b, 0x5a7c40] as const;
const STONE = [0x1f2024, 0x4a4b50, 0x8b8780] as const;
const MOSS = 0x4f6a3a;
const WOOD = 0x6b4e34;
const REED = [0x33452a, 0x98a05a, 0x4a2f1c] as const;
const SUN = vec2(-0.6, 0.8);

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);
const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));
const facing = (p: N<'vec2'>, radius: number) => dot(normalize(p.add(vec2(1e-4))), SUN).mul(smoothstep(0, radius, length(p))).mul(0.5).add(0.5);
const dome = (p: N<'vec2'>, radius: number | N<'float'>) => {
  const n = p.div(radius), z = max(n.dot(n).oneMinus(), 0).sqrt();
  return clamp(dot(normalize(vec3(n.x, n.y, z.add(0.05))), normalize(vec3(-0.5, 0.65, 0.58))), 0, 1);
};

function paint(material: THREE.MeshBasicNodeMaterial, ...layers: Layer[]): THREE.MeshBasicNodeMaterial {
  let premultiplied: N<'vec3'> = vec3(0), alpha: N<'float'> = float(0);
  for (const [color, cover] of layers) {
    premultiplied = color.mul(cover).add(premultiplied.mul(cover.oneMinus()));
    alpha = cover.add(alpha.mul(cover.oneMinus()));
  }
  material.colorNode = premultiplied.div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  return material;
}

export function ground(course: Course, maps: Maps, lit: Lit, time: Level): THREE.Mesh {
  const beachAt = course.point(-CAP - 0.55, new THREE.Vector2()), sourceAt = course.point(course.length + CAP, new THREE.Vector2());
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const tex = (map: THREE.Texture, at: N<'vec2'>) => texture(map, at).r;
  material.colorNode = Fn(() => {
    const q = chart(), { s, side, d } = course.locate(q);
    const broad = tex(maps.mottle, q.mul(0.09)), mid = tex(maps.mottle, q.mul(0.31).add(0.37)), fine = tex(maps.grain, q.mul(1.7)), blades = tex(maps.grain, q.mul(6.1).add(0.21));
    const toBeach = length(q.sub(vec2(beachAt.x, beachAt.y))), toSource = length(q.sub(vec2(sourceAt.x, sourceAt.y)));
    const mouth = smoothstep(1.1, -0.35, s), head = smoothstep(course.length - 1.2, course.length + 0.2, s);
    const wobble = tex(maps.mottle, q.mul(0.23).add(0.61)).sub(0.5).mul(0.2).add(tex(maps.grain, q.mul(0.9).add(0.17)).sub(0.5).mul(0.035));
    const edge = d.sub(mouth.mul(0.2).add(HALF).sub(head.mul(0.06))).sub(wobble).toVar();

    let land: N<'vec3'> = mix(rgb(GRASS[0]), rgb(GRASS[1]), smoothstep(0.25, 0.75, broad));
    land = mix(land, rgb(GRASS[2]), smoothstep(0.58, 0.82, mid).mul(0.45));
    land = land.mul(blades.mul(0.36).add(0.82)).mul(fine.mul(0.2).add(0.9));
    land = land.mul(smoothstep(0.05, 0.6, edge).mul(0.25).add(0.75));

    const beach = smoothstep(1.25, 0.45, toBeach.add(mid.sub(0.5).mul(0.7)).add(fine.sub(0.5).mul(0.12)));
    const band = float(0.1).add(fine.sub(0.5).mul(0.06)).add(broad.sub(0.5).mul(0.08)).add(beach.mul(0.95));
    const shore = coverage(edge.sub(band));
    let sand: N<'vec3'> = mix(rgb(SAND[1]), rgb(SAND[2]), smoothstep(0.3, 0.8, mid).mul(beach.mul(0.7).add(0.3))).mul(fine.mul(0.2).add(0.9));
    sand = mix(sand, rgb(GRAVEL).mul(fine.mul(0.3).add(0.75)), smoothstep(1.8, 1.0, toSource));
    const pebble = step(0.76, tex(maps.grain, q.mul(8.7).add(0.5)));
    sand = mix(sand, rgb(GRAVEL).mul(tex(maps.grain, q.mul(19)).mul(0.5).add(0.72)), pebble.mul(beach.oneMinus().mul(0.45).add(0.15)));
    const ripples = smoothstep(0.35, 0.5, fract(edge.mul(9).add(mid.mul(3)))).mul(smoothstep(0.5, 0.7, fract(edge.mul(9).add(mid.mul(3)))).oneMinus()).mul(beach).mul(0.08);
    sand = sand.mul(ripples.oneMinus());
    sand = mix(rgb(SAND[0]), sand, smoothstep(0, 0.07, edge));
    sand = mix(sand, land, smoothstep(band.sub(0.06), band, edge).mul(0.7));

    const depth = clamp(edge.negate().div(0.5), 0, 1);
    const drift = time.mul(FLOW);
    let water: N<'vec3'> = mix(rgb(WATER[3]), rgb(WATER[2]), smoothstep(0, 0.25, depth));
    water = mix(water, rgb(WATER[1]), smoothstep(0.15, 0.6, depth));
    water = mix(water, rgb(WATER[0]), smoothstep(0.55, 1, depth).mul(0.75));
    const swell = tex(maps.mottle, vec2(s.add(drift).mul(0.06), side.mul(0.35)).add(0.5));
    water = water.mul(swell.mul(0.24).add(0.88));
    water = mix(water, mix(rgb(SAND[1]), rgb(WATER[3]), 0.55), mouth.mul(depth.oneMinus().mul(0.4).add(0.35)));
    const lanes = 11, warp = tex(maps.mottle, vec2(s.mul(0.045), 0.3)).sub(0.5).mul(1.6).add(tex(maps.mottle, vec2(s.mul(0.21), 0.7)).sub(0.5).mul(0.9));
    const lane = side.add(sin(s.mul(3.3).add(side.mul(7))).mul(0.02)).mul(lanes).add(warp), laneId = lane.floor();
    const pace = depth.mul(0.6).add(0.7);
    const dash = tex(maps.grain, vec2(s.add(drift.mul(pace)).mul(0.19), laneId.mul(0.173).add(0.05)));
    const current = coverage(abs(fract(lane).sub(0.5)).div(lanes).sub(0.0035)).mul(smoothstep(0.54, 0.7, dash)).mul(smoothstep(0.08, 0.4, depth));
    water = mix(water, rgb(FOAM), current.mul(head.mul(0.25).add(0.24)));
    const lace = exp(edge.div(0.024)).mul(tex(maps.grain, vec2(s.add(drift).mul(0.5), side.mul(1.9))).mul(0.6).add(0.4));
    water = mix(water, rgb(FOAM), clamp(lace, 0, 1).mul(0.7));
    const lap = smoothstep(0.4, 0.48, abs(fract(edge.mul(6).sub(time.mul(0.35))).sub(0.5))).mul(smoothstep(-0.35, -0.04, edge)).mul(mouth).mul(0.45);
    water = mix(water, rgb(FOAM), lap);
    const churn = smoothstep(course.length - 0.7, course.length + 0.3, s).mul(tex(maps.grain, vec2(s.add(drift.mul(1.6)).mul(0.3), side.mul(1.6)).add(0.13)));
    water = mix(water, rgb(FOAM), smoothstep(0.42, 0.7, churn).mul(0.6));

    return lit(mix(mix(land, sand, shore), water, coverage(edge)));
  })();
  const mesh = new THREE.Mesh(flat(120, 120), material);
  mesh.frustumCulled = false;
  return mesh;
}

const lumpy = (maps: Maps, p: N<'vec2'>, seed: N<'float'>, base: number, swing: number, rough = 0) => {
  const dir = normalize(p.add(vec2(1e-4)));
  return length(p).sub(texture(maps.mottle, dir.mul(0.16).add(seed)).r.mul(swing).add(base)).sub(texture(maps.grain, dir.mul(0.22).add(seed.mul(1.7))).r.sub(0.5).mul(rough));
};

export function treeMaterial(maps: Maps, lit: Lit): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aTree', 'vec4');
  const p = uv().sub(0.5).mul(2 * SPAN), canopy = lumpy(maps, p, state.x, 0.5, 0.34, 0.12);
  const shadow = smoothstep(0.1, -0.2, lumpy(maps, p.sub(vec2(0.22, -0.26)), state.x, 0.5, 0.34, 0.12)).mul(0.55);
  const leaves = texture(maps.grain, p.mul(0.55).add(state.x.mul(3.1))).r, clumps = texture(maps.mottle, p.mul(0.3).add(state.x)).r;
  const light = dome(p, 0.86).mul(leaves.mul(0.5).add(0.72)).mul(clumps.mul(0.3).add(0.85));
  let color: N<'vec3'> = mix(rgb(LEAF[0]), rgb(LEAF[1]), smoothstep(0.15, 0.6, light));
  color = mix(color, rgb(LEAF[2]).mul(state.y.mul(0.3).add(0.85)), smoothstep(0.62, 0.95, light).mul(0.85));
  color = color.mul(smoothstep(-0.12, 0, canopy).mul(-0.3).add(1));
  return paint(sheet(), [vec3(0), shadow], [lit(color), coverage(canopy)]);
}

export function stoneMaterial(maps: Maps, lit: Lit): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aStone', 'vec4');
  const p = uv().sub(0.5).mul(2 * SPAN), body = lumpy(maps, p, state.x, 0.5, 0.36, 0.2);
  const shadow = smoothstep(0.12, -0.12, lumpy(maps, p.sub(vec2(0.14, -0.18)), state.x, 0.5, 0.36, 0.2)).mul(0.6);
  const rim = clamp(body.negate().div(0.3), 0, 1);
  const grit = texture(maps.grain, p.mul(0.7).add(state.x.mul(2.3))).r, patch = texture(maps.mottle, p.mul(0.35).add(state.x.mul(1.7))).r;
  const light = dome(p.mul(rim.mul(-0.35).add(1)), 0.8).add(grit.sub(0.5).mul(0.3));
  let color: N<'vec3'> = mix(rgb(STONE[0]), rgb(STONE[1]), smoothstep(0.05, 0.5, light));
  color = mix(color, rgb(STONE[2]), smoothstep(0.55, 0.95, light).mul(0.8));
  color = mix(color, rgb(MOSS).mul(grit.mul(0.4).add(0.7)), smoothstep(0.58, 0.78, patch.add(rim.mul(0.2))).mul(0.6));
  color = color.mul(state.y.mul(0.25).add(0.85)).mul(coverage(abs(grit.sub(0.5)).sub(0.012)).mul(-0.18).add(1));
  return paint(sheet(), [vec3(0), shadow], [rgb(STONE[0]).mul(0.5), coverage(body.sub(0.03))], [lit(color), coverage(body)]);
}

export function buoyMaterial(lit: Lit, clock: Level): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aBuoy', 'vec4');
  const p = uv().sub(0.5).mul(BUOY), level = state.x;
  const sway = sin(clock.mul(1.9).add(state.y)).mul(0.005), c = p.sub(vec2(sway, sway.mul(0.6)));
  const body = sdCircle(c, 0.068), band = abs(sdCircle(c, 0.049)).sub(0.008), housing = sdCircle(c, 0.026), glass = sdCircle(c, 0.016);
  const shadow = smoothstep(0.02, -0.02, sdCircle(p.sub(vec2(0.02, -0.026)), 0.07)).mul(0.45);
  const halo = exp(length(c).mul(-26)).mul(level.mul(level)).mul(0.9);
  const age = fract(clock.mul(0.45).add(state.y));
  const ripple = coverage(abs(sdCircle(p, age.mul(0.07).add(0.085))).sub(0.004)).mul(age.oneMinus()).mul(0.35);
  const wake = coverage(sdSegment(vec2(p.x, abs(p.y)), vec2(-0.07, 0.036), vec2(-0.24, 0.09), 0.007)).mul(smoothstep(-0.25, -0.06, p.x)).mul(0.45);
  const light = dome(c, 0.07);
  let color: N<'vec3'> = lit(rgb(BUOY_RED).mul(light.mul(0.9).add(0.35)));
  color = mix(color, lit(rgb(0xf2ece0).mul(light.mul(0.6).add(0.45))), coverage(band));
  color = mix(color, lit(rgb(0x1c1a1a)), coverage(housing));
  color = mix(color, rgb(LAMP).mul(level.mul(5).add(0.15)), coverage(glass));
  return paint(sheet(), [vec3(0), shadow], [lit(rgb(FOAM)), max(wake, ripple)], [color, coverage(body)], [rgb(LAMP).mul(2), halo.mul(coverage(glass.sub(0.004)).oneMinus())]);
}

export interface SwimmerLevels { arms: Quad; elbows: Quad; dips: Pair; bow: Level; wake: Level; glow: Level; clock: Level }

export function swimmerMaterial(accent: number, maps: Maps, lit: Lit, levels: SwimmerLevels): THREE.MeshBasicNodeMaterial {
  const { arms, elbows, dips, bow, wake, glow, clock } = levels;
  const p = uv().sub(0.5).mul(SWIMMER / FIGURE);
  const flutter = sin(clock.mul(17)).mul(0.022);
  const head = sdCircle(p.sub(vec2(0.158, 0)), 0.054);
  const torso = ellipse(p, [-0.005, 0], [0.125, 0.075]), hips = ellipse(p, [-0.115, 0], [0.058, 0.05]), trunk = min(torso, hips);
  const limb = (shoulder: N<'vec2'>, elbow: N<'vec2'>, hand: N<'vec2'>) => min(sdSegment(p, shoulder, elbow, 0.023), sdSegment(p, elbow, hand, 0.019));
  const armL = limb(vec2(0.065, 0.066), elbows.xy, arms.xy), armR = limb(vec2(0.065, -0.066), elbows.zw, arms.zw);
  const legL = sdSegment(p, vec2(-0.14, 0.028), vec2(-0.3, flutter.add(0.036)), 0.021), legR = sdSegment(p, vec2(-0.14, -0.028), vec2(-0.3, flutter.negate().sub(0.036)), 0.021);
  const dryArms = min(mix(armL, float(1), dips.x), mix(armR, float(1), dips.y));
  const wet = min(min(legL, legR), min(mix(float(1), armL, dips.x), mix(float(1), armR, dips.y)));
  const dry = min(min(head, trunk), dryArms);

  const halo = smoothstep(0.07, -0.02, min(dry, wet)).mul(0.32);
  const churn = texture(maps.grain, p.mul(vec2(2.2, 3.1)).add(vec2(clock.mul(0.9), clock.mul(0.13)))).r.mul(0.6).add(texture(maps.grain, p.mul(5.3).add(vec2(clock.mul(1.7), 0.4))).r.mul(0.4));
  const kick = smoothstep(0.03, -0.03, ellipse(p, [-0.33, 0], [0.08, 0.06])).mul(smoothstep(0.46, 0.62, churn)).mul(0.8);
  const trail = coverage(sdSegment(vec2(p.x, abs(p.y)), vec2(-0.3, 0.05), vec2(-0.45, 0.13), 0.009)).mul(smoothstep(-0.45, -0.3, p.x)).mul(wake).mul(0.7);
  const parting = coverage(abs(dry.sub(0.012)).sub(0.007)).mul(0.55);
  const front = p.sub(vec2(0.158, 0)), fr = length(front);
  const crest = coverage(abs(fr.sub(bow.mul(0.02).add(0.085))).sub(0.009)).mul(smoothstep(1.3, 0.4, abs(atan(front.y, front.x)))).mul(bow);
  const entry = (tip: N<'vec2'>, dip: N<'float'>) => coverage(abs(sdCircle(p.sub(tip), 0.03)).sub(0.006)).mul(dip.mul(dip.oneMinus()).mul(4));
  const foam = clamp(max(max(max(kick, trail), max(parting, crest)), max(entry(arms.xy, dips.x), entry(arms.zw, dips.y))), 0, 1);

  const light = facing(p.sub(vec2(-0.005, 0)), 0.1), shine = min(lit(vec3(1)), vec3(SHINE));
  const skin = rgb(SKIN).mul(light.mul(0.4).add(0.75));
  let suit: N<'vec3'> = rgb(SUIT).mul(light.mul(0.5).add(0.75));
  suit = mix(suit, rgb(accent).mul(light.mul(0.35).add(0.8)), coverage(abs(p.y).sub(0.013)).mul(coverage(torso.add(0.012))));
  let cap: N<'vec3'> = rgb(CAP_COLOR).mul(facing(front, 0.054).mul(0.55).add(0.65));
  cap = mix(cap, rgb(0x121212), coverage(abs(p.x.sub(0.128)).sub(0.006)).mul(coverage(head.add(0.004))));
  cap = mix(cap, rgb(0xfff1dc), smoothstep(0.022, 0, length(front.sub(vec2(-0.014, 0.02)))).mul(0.55));
  let above: N<'vec3'> = mix(skin, suit, coverage(trunk));
  above = mix(above, cap, coverage(head));
  return paint(sheet(),
    [rgb(WATER[0]).mul(0.6), halo],
    [rgb(GOLD).mul(2.2), smoothstep(0.06, 0.01, dry).mul(glow).mul(0.6)],
    [lit(rgb(FOAM)), foam],
    [mix(skin, rgb(WATER[1]), 0.6).mul(shine), coverage(wet).mul(0.62)],
    [rgb(0x0a0d10), coverage(dry.sub(0.009))],
    [above.mul(shine), coverage(dry)],
  );
}

export function springMaterial(maps: Maps, lit: Lit, clock: Level, flare: Level): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(SPRING);
  const cleft = ellipse(p, [0.1, 0], [0.16, 0.12]);
  const core = exp(length(p.sub(vec2(0.1, 0))).mul(-14)).mul(flare.mul(1.6).add(0.8));
  const tongue = max(max(p.x.sub(0.12), abs(p.y).sub(max(float(0.07).sub(p.x.mul(0.35)), 0.02))), p.x.negate().sub(0.62));
  const run = texture(maps.grain, vec2(p.x.add(clock.mul(0.7)).mul(0.55), p.y.mul(4.2)).add(0.3)).r, spray = texture(maps.grain, vec2(p.x.add(clock.mul(1.1)).mul(1.7), p.y.mul(7)).add(0.61)).r;
  const froth = coverage(tongue).mul(max(smoothstep(0.5, 0.64, run), smoothstep(0.58, 0.7, spray).mul(0.7))).mul(smoothstep(-0.62, -0.15, p.x));
  const q = p.mul(DROPS).add(vec2(clock.mul(2.2), 0)), seed = grain(floor(q)), drop = length(fract(q).sub(seed.mul(0.5).add(0.25))).div(DROPS).sub(seed.y.mul(0.006).add(0.006));
  const glint = coverage(drop).mul(step(0.6, seed.x)).mul(smoothstep(0.5, 0.1, length(p.sub(vec2(-0.25, 0))))).mul(flare.mul(0.8).add(0.25));
  return paint(sheet(),
    [rgb(0x0b0a09), coverage(cleft).mul(0.85)],
    [lit(rgb(FOAM)).mul(1.05), froth.mul(0.9)],
    [rgb(GOLD).mul(core.mul(3)), clamp(core, 0, 1).mul(0.9)],
    [rgb(0xfff4da).mul(3), clamp(glint, 0, 1)],
  );
}

export function campMaterial(maps: Maps, lit: Lit, clock: Level, glow: Level): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(CAMP);
  const ash = smoothstep(0.2, 0.06, length(p).add(texture(maps.grain, p.mul(3)).r.sub(0.5).mul(0.08))).mul(0.55);
  let ring: N<'float'> = float(1), logs: N<'float'> = float(1), knots: N<'float'> = float(1), local: N<'vec2'> = vec2(0), size: N<'float'> = float(1);
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2 + 0.3, r = 0.115 + 0.012 * Math.sin(k * 2.7), radius = 0.03 + 0.006 * Math.cos(k * 1.9);
    const offset = p.sub(vec2(Math.cos(a) * r, Math.sin(a) * r)), d = sdCircle(offset, radius), closer = d.lessThan(ring);
    local = select(closer, offset, local);
    size = select(closer, float(radius), size);
    ring = min(ring, d);
  }
  for (const turn of [0.35, 2.4, 4.3]) {
    const d = vec2(Math.cos(turn), Math.sin(turn));
    logs = min(logs, sdSegment(p, d.mul(0.02), d.mul(0.1), 0.017));
    knots = min(knots, sdCircle(p.sub(d.mul(0.1)), 0.012));
  }
  const flicker = sin(clock.mul(11.3)).mul(sin(clock.mul(7.1).add(1.3))).mul(0.5).add(0.5);
  const tongue = texture(maps.grain, p.mul(2.2).add(vec2(clock.mul(0.13), clock.mul(-0.31)))).r;
  const flame = length(p.mul(vec2(1, 0.85))).sub(glow.mul(0.018).add(0.045).add(tongue.sub(0.5).mul(0.05)).add(flicker.mul(0.008)));
  const core = smoothstep(0.03, -0.01, length(p).sub(0.018));
  const stoneColor = lit(mix(rgb(STONE[0]), rgb(STONE[2]), dome(local, size.mul(1.1))).mul(smoothstep(0.1, 0.14, length(p)).mul(0.4).add(0.6)));
  const wood = lit(rgb(WOOD).mul(0.8));
  const fire = mix(rgb(0xff6a1f).mul(2.2), rgb(0xffe6a0).mul(4), core).mul(glow.mul(0.4).add(0.8));
  return paint(sheet(),
    [rgb(0x15110e), ash],
    [wood, coverage(logs)],
    [lit(rgb(0x3a2717)), coverage(knots)],
    [stoneColor, coverage(ring)],
    [fire, coverage(flame).mul(0.95)],
  );
}

export function reedMaterial(lit: Lit, clock: Level): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aReed', 'vec4');
  const p = uv().sub(0.5).mul(2 * SPAN);
  const jitter = (k: number, salt: number) => fract(sin(state.x.mul(12.9898 * (k + 1)).add(salt)).mul(43758.5453));
  let blades: N<'float'> = float(1), shade: N<'float'> = float(1), heads: N<'float'> = float(1), tipness: N<'float'> = float(0);
  for (let k = 0; k < 10; k++) {
    const a = state.x.mul(6.283).add((k / 9 - 0.5) * 2.5).add(jitter(k, 1.7).sub(0.5).mul(0.45)).add(sin(clock.mul(1.3).add(k * 1.7)).mul(0.05));
    const dir = vec2(a.cos(), a.sin()), reach = jitter(k, 4.1).mul(0.45).add(0.4), tip = dir.mul(reach);
    const blade = sdSegment(p, tip.mul(0.08), tip, float(0.05).sub(length(p).mul(0.03)));
    const along = clamp(dot(p, tip).div(reach.mul(reach)), 0, 1);
    tipness = select(blade.lessThan(blades), along, tipness);
    blades = min(blades, blade);
    shade = min(shade, sdSegment(p.sub(vec2(0.07, -0.09)), tip.mul(0.08), tip, 0.05));
    if (k === 3 || k === 6) {
      const q = p.sub(tip.mul(0.72));
      heads = min(heads, ellipse(vec2(dot(q, dir), q.y.mul(dir.x).sub(q.x.mul(dir.y))), [0, 0], [0.12, 0.055]));
    }
  }
  const color = mix(rgb(REED[0]), rgb(REED[1]), tipness.mul(state.y.mul(0.4).add(0.6)));
  return paint(sheet(), [vec3(0), coverage(shade).mul(0.4)], [lit(color), coverage(blades)], [lit(rgb(REED[2])), coverage(heads)]);
}

export function swirlMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aSwirl', 'vec4');
  const age = clamp(state.x, 0, 1), p = uv().sub(0.5).mul(SWIRL), r = length(p), a = atan(p.y, p.x);
  const arms = sin(a.mul(2).sub(r.mul(46)).add(age.mul(9)));
  const band = smoothstep(0.03, 0, abs(r.sub(age.mul(0.12).add(0.1))).sub(age.mul(0.06).add(0.05)));
  const lace = smoothstep(0.55, 0.9, arms).mul(band).mul(age.oneMinus().pow(1.3));
  return paint(sheet(), [lit(rgb(FOAM)), lace.mul(state.y).mul(0.85)]);
}
