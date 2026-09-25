import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, normalize, positionWorld, pow, select, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, grain, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const LIKED = 0xffa94d;
export const CRAVED = TOKENS.cueCyan;
export const GOLD = TOKENS.celebrationGold;
export const DEBT = 0xd05a78;

export const MARGIN = 1;
export const ECHO = 0.3;
export const ITEM = 1;
export const AGENT = 1.6;
export const VIEW = 3.3;
export const THREAD = 0.86;
export const TRAIL = 0.16;
export const GAUGE_W = 1.5;
export const GAUGE_H = 14.8;
export const TUBE = 0.24;

const LIGHT = normalize(vec2(-0.55, 0.7));

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const local = (size: N<'vec2'> | number) => uv().sub(0.5).mul(size);

export function periodic(side: number): N<'float'> {
  const half = side / 2, p = vec2(positionWorld.x, positionWorld.z);
  return mix(float(ECHO), float(1), coverage(sdBox(p, vec2(half)))).mul(coverage(sdBox(p, vec2(half + MARGIN), 0.3)));
}

const over = (below: readonly [N<'vec3'>, N<'float'>], color: N<'vec3'>, alpha: N<'float'>): [N<'vec3'>, N<'float'>] =>
  [below[0].mul(alpha.oneMinus()).add(color.mul(alpha)), below[1].mul(alpha.oneMinus()).add(alpha)];

const finish = (material: THREE.MeshBasicNodeMaterial, [premultiplied, alpha]: readonly [N<'vec3'>, N<'float'>], mask: N<'float'> | number = 1) => {
  material.colorNode = premultiplied.div(max(alpha, 1e-4));
  material.opacityNode = alpha.mul(mask);
  return material;
};

export interface PlateMaps { mottle: THREE.Texture; grain: THREE.Texture }

export function plateMaterial(side: number, maps: PlateMaps, accent: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const half = side / 2, p = vec2(positionWorld.x, positionWorld.z), g = p.add(half), tile = g.div(side);
  const core = coverage(sdBox(p, vec2(half))), field = coverage(sdBox(p, vec2(half + MARGIN), 0.3));
  const mott = texture(maps.mottle, tile).r, drift = texture(maps.mottle, tile.mul(2).add(vec2(0.31, 0.67))).r, fine = texture(maps.grain, tile.mul(4)).r;
  let agar: N<'vec3'> = mix(rgb(0x0a1822), rgb(0x183342), smoothstep(0.2, 0.85, mott)).mul(fine.mul(0.26).add(0.87));
  agar = mix(agar, rgb(0x24404e), smoothstep(0.58, 0.92, drift).mul(0.4));

  const motes = g.mul(5), id = mod(floor(motes).add(side * 50), side * 5), jitter = grain(id), pick = grain(id.add(71.3)).x;
  const mote = coverage(length(fract(motes).sub(jitter.mul(0.6).add(0.2))).div(5).sub(pick.mul(0.25).add(0.012))).mul(step(pick, 0.035));

  const rule = (v: N<'float'>) => {
    const b = floor(v.add(0.5)), d = abs(v.sub(b)), major = step(mod(b.add(side * 4), 4), 0.5);
    return max(coverage(d.sub(0.007)).mul(major.mul(0.4).add(0.6)), coverage(abs(d.sub(0.055)).sub(0.005)).mul(major));
  };
  const lines = max(rule(g.x), rule(g.y));

  const edge = abs(sdBox(p, vec2(half))), along = select(abs(p.x).greaterThan(abs(p.y)), p.y, p.x);
  const dashes = coverage(edge.sub(0.022)).mul(step(fract(along.mul(1.25).add(0.25)), 0.5)).mul(field);
  const chevron = (q: N<'vec2'>) => min(sdSegment(q, vec2(-0.15, 0.08), vec2(0, -0.08), 0.026), sdSegment(q, vec2(0.15, 0.08), vec2(0, -0.08), 0.026));
  const right = (q: N<'vec2'>) => chevron(vec2(q.y, q.x.negate()));
  const marks = min(
    min(chevron(p.sub(vec2(-half, 0))), chevron(p.sub(vec2(half, 0)))),
    min(min(right(p.sub(vec2(-0.1, -half))), right(p.sub(vec2(0.1, -half)))), min(right(p.sub(vec2(-0.1, half))), right(p.sub(vec2(0.1, half))))),
  );
  const halo = coverage(marks.sub(0.05));

  const region = mix(float(ECHO), float(1), core).mul(field);
  let color: N<'vec3'> = mix(rgb(0x05090d).mul(mott.mul(0.4).add(0.8)), agar.mul(mix(vec3(0.62, 0.7, 0.85), vec3(1), core)), field);
  color = mix(color, rgb(0x7fa6ba).mul(0.55), lines.mul(region).mul(0.55));
  color = color.add(rgb(0xbfe6f2).mul(mote).mul(region).mul(pick.mul(8).add(0.12)));
  let shaded = mix(lit(color), rgb(0x05090d), halo.mul(0.85));
  shaded = mix(shaded, rgb(accent).mul(0.8).add(0.1), max(dashes.mul(0.38), coverage(marks).mul(0.9)));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = shaded;
  return material;
}

interface Nugget { bead: N<'float'>; crystal: N<'float'>; color: N<'vec3'>; solid: N<'float'>; glow: N<'vec3'> }

function nugget(q: N<'vec2'>, kind: N<'float'>, spin: N<'float'>, twinkle: N<'float'>): Nugget {
  const craved = step(1.5, kind).mul(step(kind, 2.5)), both = step(2.5, kind);
  const rb = mix(float(0.155), float(0.098), both), n = q.div(rb), bead = sdCircle(q, rb);
  const facing = clamp(dot(n, LIGHT).mul(0.5).add(0.5), 0, 1), rim = smoothstep(0.55, 1, length(n));
  let amber: N<'vec3'> = mix(rgb(0x9a4412), rgb(0xffc56e), facing).mul(rim.mul(-0.35).add(1));
  amber = mix(amber, rgb(0xfff1d2), coverage(sdCircle(q.sub(LIGHT.mul(rb.mul(0.42))), rb.mul(0.3))).mul(0.85));
  amber = mix(amber, rgb(0x5a2208), coverage(abs(bead).sub(0.006)).mul(0.6));

  const c = cos(spin), s = sin(spin), r = vec2(q.x.mul(c).sub(q.y.mul(s)), q.x.mul(s).add(q.y.mul(c)));
  const hc = mix(float(0.132), float(0.172), both), crystal = sdBox(r, vec2(hc), 0.016), table = sdBox(r, vec2(hc.mul(0.46)), 0.008);
  const wide = abs(r.x).greaterThan(abs(r.y));
  const normal = select(wide, select(r.x.greaterThan(0), vec2(c, s.negate()), vec2(c.negate(), s)), select(r.y.greaterThan(0), vec2(s, c), vec2(s.negate(), c.negate())));
  const lightness = clamp(dot(normal, LIGHT).mul(0.42).add(0.6), 0, 1);
  let cyan: N<'vec3'> = mix(rgb(0x0f5577), rgb(0x8eeaff), lightness);
  cyan = mix(cyan, rgb(0xc4f6ff), coverage(table).mul(mix(float(0.7), float(0.25), both)));
  cyan = mix(cyan, rgb(0xeafdff), coverage(abs(crystal).sub(0.007)).mul(0.9));
  cyan = mix(cyan, rgb(0x0e3f57), coverage(abs(abs(r.x).sub(abs(r.y))).sub(0.004)).mul(coverage(table.negate())).mul(0.45));

  const solidCrystal = coverage(crystal), solidBead = coverage(bead);
  const isBead = step(kind, 1.5);
  const color = mix(mix(cyan, amber, solidBead.mul(both)), amber, isBead).mul(twinkle);
  const solid = max(mix(solidCrystal, solidBead, isBead), solidBead.mul(both));
  const radius = length(q);
  const glow = mix(mix(rgb(LIKED), rgb(CRAVED), craved), mix(rgb(LIKED), rgb(CRAVED), smoothstep(0.08, 0.34, radius)), both);
  return { bead, crystal, color, solid, glow };
}

export function itemMaterial(side: number, time: Level): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aItem', 'vec4'), kind = state.x, spin = state.y, grow = state.z;
  const p = local(ITEM), q = p.div(max(grow, 0.02)), radius = length(p);
  const twinkle = sin(time.mul(1.3).add(spin.mul(5.7))).mul(0.05).add(1.12);
  const item = nugget(q, kind, spin, twinkle);
  const haloAlpha = exp(radius.mul(radius).mul(-16)).mul(0.5).mul(min(grow, 1));
  let layer: [N<'vec3'>, N<'float'>] = [item.glow.mul(haloAlpha), haloAlpha];
  layer = over(layer, item.color, item.solid.mul(step(0.02, grow)));
  return finish(sheet(), layer, periodic(side));
}

export function iconMaterial(kind: number): THREE.MeshBasicNodeMaterial {
  const p = local(ITEM), item = nugget(p.div(1.25), float(kind), float(Math.PI / 4 + 0.25), float(1.1));
  const radius = length(p), haloAlpha = exp(radius.mul(radius).mul(-12)).mul(0.45);
  return finish(sheet(), over([item.glow.mul(haloAlpha), haloAlpha], item.color, item.solid));
}

export function foragerArt(side: number, accent: number, time: Level) {
  const stretch = uniform(0), liked = uniform(0), craved = uniform(0), flash = uniform(0), tint = uniform(new THREE.Vector3(1, 1, 1));
  const p = local(AGENT), r = length(p), theta = atan(p.y, p.x);
  const wobble = sin(theta.mul(3).add(time.mul(1.9))).mul(0.045).add(sin(theta.mul(5).sub(time.mul(2.7)).add(1.3)).mul(0.028)).add(sin(theta.mul(2).add(time.mul(0.8)).add(2.1)).mul(0.035));
  const ahead = pow(max(cos(theta), 0), 6), behind = pow(max(cos(theta).negate(), 0), 8);
  const reach = float(0.34).mul(wobble.add(1)).add(ahead.mul(stretch.mul(0.15).add(0.025))).sub(behind.mul(stretch.mul(0.05)));
  const body = r.sub(reach);
  const inner = smoothstep(-0.13, 0, body);
  const flow = p.mul(20).add(vec2(time.mul(-0.35), sin(time.mul(0.4)).mul(0.6))), cell = floor(flow), speck = grain(cell), bright = grain(cell.add(19.7)).x;
  const granule = coverage(length(fract(flow).sub(speck.mul(0.6).add(0.2))).div(20).sub(0.01)).mul(step(bright, 0.35)).mul(coverage(body.add(0.04)));
  let cyto: N<'vec3'> = rgb(accent).mul(inner.mul(0.75).add(0.42));
  cyto = cyto.add(rgb(0xfff0ea).mul(granule).mul(bright.mul(0.8).add(0.25)));

  const nucleusAt = vec2(-0.09, 0.0), nucleus = sdCircle(p.sub(nucleusAt), 0.1);
  cyto = mix(cyto, rgb(accent).mul(0.22).add(rgb(0x2a0c1c).mul(0.5)), coverage(nucleus));
  cyto = mix(cyto, rgb(accent).mul(0.95), coverage(abs(nucleus).sub(0.008)).mul(0.8));
  cyto = mix(cyto, rgb(0xffd6cc), coverage(sdCircle(p.sub(nucleusAt.add(vec2(-0.014, 0.026))), 0.03)).mul(0.9));

  const organelle = (at: readonly [number, number], hue: number, state: Level, phase: number): [N<'float'>, N<'vec3'>] => {
    const d = sdCircle(p.sub(vec2(...at)), 0.06);
    const pulse = sin(time.mul(3.1).add(phase)).mul(0.12).add(1);
    const lit = mix(rgb(hue).mul(0.2).add(rgb(0x2c3440).mul(0.3)), rgb(hue).mul(pulse.mul(1.35)), state);
    const core = mix(lit, rgb(0xffffff).mul(state.mul(0.5).add(0.3)), coverage(sdCircle(p.sub(vec2(at[0] - 0.017, at[1] + 0.019)), 0.017)).mul(state.mul(0.6).add(0.2)));
    return [d, core];
  };
  const [likedD, likedC] = organelle([0.1, 0.14], LIKED, liked, 0);
  const [cravedD, cravedC] = organelle([0.1, -0.14], CRAVED, craved, 1.9);
  cyto = mix(cyto, likedC, coverage(likedD));
  cyto = mix(cyto, cravedC, coverage(cravedD));
  cyto = mix(cyto, rgb(0x14060c), coverage(abs(min(likedD, cravedD)).sub(0.006)).mul(0.5));

  const membrane = coverage(abs(body).sub(0.016));
  const glow = exp(max(body, 0).mul(-11)).mul(0.42).mul(coverage(body.negate()));
  const bloom = exp(r.mul(r).mul(-40)).mul(flash);
  let layer: [N<'vec3'>, N<'float'>] = [rgb(accent).mul(glow), glow];
  layer = over(layer, cyto, coverage(body).mul(0.9));
  layer = over(layer, rgb(accent).mul(0.55).add(rgb(0xfff4ef).mul(0.9)), membrane);
  layer = [layer[0].add(tint.mul(bloom).mul(0.7)), max(layer[1], bloom.mul(0.6))];
  return { material: finish(sheet(), layer, periodic(side)), stretch, liked, craved, flash, tint };
}

export type ForagerArt = ReturnType<typeof foragerArt>;

export function viewMaterial(side: number, accent: number): THREE.MeshBasicNodeMaterial {
  const p = local(VIEW), a = abs(p);
  const box = sdBox(p, vec2(1.5), 0.16);
  const fill = coverage(box).mul(0.035);
  const outline = coverage(abs(box).sub(0.008)).mul(0.22);
  const bracket = coverage(abs(box).sub(0.02)).mul(step(1.08, min(a.x, a.y)));
  let layer: [N<'vec3'>, N<'float'>] = [rgb(accent).mul(fill), fill];
  layer = over(layer, rgb(accent).mul(0.9), outline);
  layer = over(layer, rgb(0xffe8e2), bracket.mul(0.85));
  return finish(sheet(), layer, periodic(side));
}

export function threadMaterial(side: number, time: Level): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aThread', 'vec4'), span = state.x, kind = state.y, alpha = state.z;
  const p = uv().sub(0.5).mul(vec2(span.add(0.9), THREAD)), t = p.x.add(span.mul(0.5));
  const spacing = 0.2, moving = t.sub(time.mul(0.45)), centre = floor(moving.div(spacing)).add(0.5).mul(spacing), at = centre.add(time.mul(0.45));
  const bead = coverage(length(vec2(moving.sub(centre), p.y)).sub(0.023)).mul(step(0.4, at)).mul(step(at, span.sub(0.42)));
  const fade = mix(float(0.95), float(0.35), clamp(at.div(max(span, 0.001)), 0, 1)), dots = bead.mul(fade);
  const q = vec2(p.x.sub(span.mul(0.5)), p.y), fold = abs(q);
  const ring = coverage(abs(length(q).sub(0.4)).sub(0.012));
  const ticks = coverage(min(sdSegment(fold, vec2(0.33, 0), vec2(0.47, 0), 0.012), sdSegment(fold, vec2(0, 0.33), vec2(0, 0.47), 0.012)));
  const cravedOnly = step(1.5, kind).mul(step(kind, 2.5)), both = step(2.5, kind);
  const color = mix(mix(rgb(LIKED), rgb(CRAVED), cravedOnly), mix(rgb(LIKED), rgb(CRAVED), step(0.5, fract(at.mul(2.5)))), both);
  const ringColor = mix(mix(rgb(LIKED), rgb(CRAVED), cravedOnly), mix(rgb(LIKED), rgb(CRAVED), step(0, q.y)), both);
  let layer: [N<'vec3'>, N<'float'>] = [color.mul(dots), dots];
  layer = over(layer, ringColor.mul(1.2), max(ring, ticks).mul(0.75));
  return finish(sheet(), layer, periodic(side).mul(alpha));
}

export function trailMaterial(side: number, accent: number): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aTrail', 'vec4'), span = state.x, alpha = state.y;
  const p = uv().sub(0.5).mul(vec2(span.add(TRAIL), TRAIL));
  const capsule = sdSegment(p, vec2(span.mul(-0.5), 0), vec2(span.mul(0.5), 0), 0.022);
  const a = coverage(capsule).mul(alpha).mul(0.3);
  return finish(sheet(), [rgb(accent).mul(0.9).mul(a), a], periodic(side));
}

export interface GaugeSpec {
  hue: number;
  outward: number;
  zero: number;
  top: number;
  bottom: number;
  window: number;
  step: number;
  decade: number;
  good: readonly [number, number];
  marks: readonly number[];
  hidden: boolean;
}

export function gaugeMaterial(spec: GaugeSpec, level: Level, bump: Level, time: Level): THREE.MeshBasicNodeMaterial {
  const p = local(vec2(GAUGE_W, GAUGE_H)), o = spec.outward, x = p.x.mul(o), y = p.y;
  const centre = (spec.top + spec.bottom) / 2, halfLength = (spec.top - spec.bottom) / 2;
  const tube = sdBox(vec2(p.x, y.sub(centre)), vec2(TUBE, halfLength + TUBE), TUBE);
  const inside = coverage(tube.add(0.04));
  const glass = coverage(abs(tube).sub(0.011));

  const lo = min(float(spec.zero), level), hi = max(float(spec.zero), level), gain = step(spec.zero, level);
  const column = coverage(max(lo.sub(y), y.sub(hi))).mul(inside);
  const rise = clamp(y.sub(lo).div(max(hi.sub(lo), 0.05)), 0, 1);
  const bubbles = vec2(p.x.mul(9), y.mul(3.2).sub(time.mul(0.9))), bcell = floor(bubbles), bpick = grain(bcell), bdot = coverage(length(fract(bubbles).sub(bpick.mul(0.5).add(0.25))).div(9).sub(0.012)).mul(step(bpick.x, 0.3));
  const fluid = rgb(spec.hue).mul(rise.mul(0.7).add(0.45)).add(rgb(0xffffff).mul(bdot.mul(0.35)));
  const hatch = coverage(abs(fract(p.x.add(y).mul(6)).sub(0.5)).mul(1 / 6).sub(0.018));
  const debt = mix(rgb(0x2a0a16), rgb(DEBT).mul(0.8), hatch.mul(0.7));

  const strip = sdBox(vec2(x.sub(TUBE + 0.13), y.sub(centre)), vec2(0.045, halfLength + 0.05), 0.02);
  const good = step(spec.good[0], y).mul(step(y, spec.good[1]));
  const zoneHatch = coverage(abs(fract(y.mul(7)).sub(0.5)).mul(1 / 7).sub(0.02));
  const zone = mix(rgb(0x3a4a5a).mul(zoneHatch.mul(0.6).add(0.4)), rgb(spec.hue).mul(1.05), good);

  const tick = (distance: N<'float'>, size: number) => coverage(max(abs(distance).sub(0.01), max(float(TUBE + 0.06).sub(x.negate()), x.negate().sub(TUBE + 0.06 + size))));
  const fromZero = y.sub(spec.zero), windowTop = spec.zero + spec.window;
  const inWindow = step(spec.zero - 0.01, y).mul(step(y, windowTop + 0.01));
  const windowTicks = tick(fract(fromZero.div(spec.step).add(0.5)).sub(0.5).mul(spec.step), 0.1).mul(inWindow);
  const ruler = (t: N<'float'>) => {
    const f = fract(t);
    let minor: N<'float'> = float(1);
    for (let k = 2; k < 10; k++) minor = min(minor, abs(f.sub(Math.log10(k))));
    return max(tick(min(f, f.oneMinus()).mul(spec.decade), 0.16), tick(minor.mul(spec.decade), 0.07));
  };
  const below = ruler(float(spec.zero).sub(y).div(spec.decade)).mul(step(y, spec.zero - 0.02)).mul(step(spec.bottom - TUBE, y));
  const above = ruler(y.sub(windowTop).div(spec.decade)).mul(step(windowTop + 0.02, y)).mul(step(y, spec.top + TUBE));
  const zeroTick = tick(fromZero, 0.26);
  let thresholds: N<'float'> = float(0);
  for (const mark of spec.marks) thresholds = max(thresholds, coverage(abs(y.sub(mark)).sub(0.012)).mul(step(fract(p.x.mul(9)), 0.55)));

  const meniscus = coverage(abs(y.sub(level)).sub(0.02)).mul(inside);
  const pointer = coverage(max(abs(y.sub(level)).sub(x.sub(TUBE + 0.19).mul(0.55)), x.sub(TUBE + 0.42)));
  const highlight = coverage(sdBox(vec2(p.x.add(TUBE * 0.45), y.sub(centre)), vec2(0.022, halfLength * 0.9), 0.02)).mul(0.14);

  let layer: [N<'vec3'>, N<'float'>] = [rgb(0x9fc3d6).mul(inside.mul(0.06)), inside.mul(0.06)];
  layer = over(layer, mix(debt, fluid, gain), column.mul(mix(float(0.85), float(0.92), gain)));
  layer = over(layer, rgb(0xffffff), highlight.mul(inside));
  layer = over(layer, rgb(spec.hue).mul(0.5).add(0.35), thresholds.mul(inside).mul(0.9));
  layer = over(layer, rgb(0xcfe3ee).mul(0.75), glass);
  layer = over(layer, zone, coverage(strip).mul(0.9));
  layer = over(layer, rgb(0x9fb6c4), max(windowTicks, max(below, above)).mul(0.8));
  layer = over(layer, rgb(0xffffff), zeroTick);
  if (spec.hidden) {
    const veil = coverage(abs(fract(p.x.sub(y).mul(10)).sub(0.5)).mul(0.1).sub(0.012)).mul(inside);
    layer = over(layer, rgb(0xb8c8d8), veil.mul(0.35));
  }
  layer = over(layer, rgb(0xfff6ee).mul(bump.mul(0.6).add(0.9)), max(meniscus, pointer));
  return finish(sheet(), layer);
}

export function eyeMaterial(hidden: boolean): THREE.MeshBasicNodeMaterial {
  const p = local(0.9);
  const lid = max(length(p.sub(vec2(0, -0.13))).sub(0.24), length(p.sub(vec2(0, 0.13))).sub(0.24));
  const outline = coverage(abs(lid).sub(0.018));
  const pupil = coverage(sdCircle(p, 0.055));
  const slash = coverage(sdSegment(p, vec2(-0.2, -0.16), vec2(0.2, 0.16), 0.02));
  const gap = coverage(sdSegment(p, vec2(-0.2, -0.16), vec2(0.2, 0.16), 0.055));
  const shape = hidden ? max(outline.mul(gap.oneMinus()), slash) : max(outline, pupil);
  return finish(sheet(), [rgb(0xdfe9f0).mul(shape.mul(0.85)), shape.mul(0.85)]);
}
