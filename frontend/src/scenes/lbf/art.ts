import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const FRAME = 0.44;
export const ANT = 0.9;
export const BUNCH = 1;
export const SPOT = 0.8;
export const DROP = 0.12;
export const TINTS = [TOKENS.accents.LBF!, 0xf2915c, 0x74c7f2] as const;
export const BERRY = 0xd23c52;
export const GOLD = 0xffc861;

const SOIL = [0x946c4b, 0x76553b] as const;
const GROUT = 0x3a2a1f;
const WOOD = [0xb07e4c, 0x75502f] as const;
const LAWN = [0x51702f, 0x3f5a26] as const;
const FOG = 0x1b2838;
const INK = 0x1e160f;
const LEAF = [0x4b8a41, 0x2d5a2b] as const;
const GROOVE_AT = 0.2;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

const turn = (p: N<'vec2'>, angle: number) => {
  const c = Math.cos(angle), s = Math.sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};

const hidden = (color: N<'vec3'>, seen: N<'float'>) => mix(mix(vec3(dot(color, vec3(0.2126, 0.7152, 0.0722))), rgb(FOG), 0.35).mul(0.55), color, seen);

class Stack {
  color: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(color: N<'vec3'>, alpha: N<'float'>): this {
    this.color = color.mul(alpha).add(this.color.mul(alpha.oneMinus()));
    this.alpha = alpha.add(this.alpha.mul(alpha.oneMinus()));
    return this;
  }

  paint(material: THREE.MeshBasicNodeMaterial, fade: N<'float'>): void {
    material.colorNode = this.color.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha.mul(fade);
  }
}

export interface Ground {
  material: THREE.MeshBasicNodeMaterial;
  window: THREE.UniformNode<'vec4', THREE.Vector4>;
  fog: Level;
  clock: Level;
  time: Level;
  textures: THREE.Texture[];
}

export function groundMaterial(side: number, steps: number, seed: number, lit: Lit): Ground {
  const window = uniform(new THREE.Vector4(0, 0, side, side)), fog = uniform(0), clock = uniform(0), time = uniform(0);
  const mottle = fbm(seed ^ 0x1b5f, { size: 256, frequency: 4, octaves: 5 }), grain = fbm(seed ^ 0x7a13, { size: 256, frequency: 32, octaves: 3 });
  const q = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2)), c = q.sub(side / 2);
  const cell = floor(q), local = fract(q).sub(0.5);
  const mott = texture(mottle, q.mul(0.11)).r, fine = texture(grain, q.mul(0.37)).r, speck = texture(grain, q.mul(2.3).add(0.31)).r;

  const tile = sdBox(local, vec2(0.468), 0.1);
  const parity = mod(cell.x.add(cell.y), 2);
  let soil: N<'vec3'> = mix(rgb(SOIL[1]), rgb(SOIL[0]), parity.mul(0.7).add(mott.sub(0.5).mul(0.6)).clamp(0, 1));
  soil = soil.mul(fine.sub(0.5).mul(0.2).add(1)).mul(smoothstep(0.72, 0.8, speck).mul(0.16).add(1));
  const rim = smoothstep(-0.075, 0, tile), facing = clamp(local.x.add(local.y).mul(-2.4), -1, 1);
  soil = soil.mul(rim.mul(facing).mul(0.2).add(1));
  soil = soil.mul(smoothstep(0, 0.28, min(q.x, q.y)).mul(0.3).add(0.7));
  const bed = mix(rgb(GROUT).mul(fine.mul(0.3).add(0.85)), soil, coverage(tile));

  const horizontal = step(abs(c.x), abs(c.y));
  const along = mix(c.y, c.x, horizontal), across = mix(abs(c.x), abs(c.y), horizontal).sub(side / 2);
  const plank = texture(grain, vec2(along.mul(0.16), across.mul(2.2).add(horizontal.mul(0.5)))).r;
  let wood: N<'vec3'> = mix(rgb(WOOD[1]), rgb(WOOD[0]), smoothstep(0.3, 0.72, plank));
  wood = wood.mul(sin(clamp(across.div(FRAME), 0, 1).mul(Math.PI)).mul(0.26).add(0.8));
  const seams = coverage(abs(fract(along.div(3.1).add(0.5)).sub(0.5)).mul(3.1).sub(0.008));
  const miter = coverage(abs(abs(c.x).sub(abs(c.y))).sub(0.008)).mul(step(side / 2, min(abs(c.x), abs(c.y))));
  wood = mix(wood, rgb(INK), max(seams, miter).mul(0.75));

  const reach = side / 2 + GROOVE_AT, perimeter = 8 * reach;
  const groove = coverage(abs(sdBox(c, vec2(reach), 0)).sub(0.055));
  const u = select(horizontal.greaterThan(0.5), select(c.y.lessThan(0), c.x.add(reach), float(5 * reach).sub(c.x)), select(c.x.greaterThan(0), c.y.add(3 * reach), float(7 * reach).sub(c.y)));
  const spacing = perimeter / steps, done = clock.mul(perimeter);
  const notch = coverage(abs(fract(u.div(spacing).add(0.5)).sub(0.5)).mul(spacing).sub(0.012));
  wood = mix(wood, mix(rgb(0x382617), rgb(0x57402a), notch), groove);

  const blades = texture(grain, q.mul(vec2(1.7, 0.45))).r;
  let lawn: N<'vec3'> = mix(rgb(LAWN[1]), rgb(LAWN[0]), smoothstep(0.2, 0.8, mott).mul(0.5).add(blades.mul(0.5))).mul(smoothstep(0.35, 0.65, texture(grain, q.mul(vec2(3.1, 0.8)).add(0.5)).r).mul(0.22).add(0.86));
  lawn = lawn.mul(smoothstep(-0.05, 0.3, sdBox(c.sub(vec2(0.09, 0.12)), vec2(side / 2 + FRAME), 0.12)).mul(0.4).add(0.6));

  const inBed = coverage(sdBox(c, vec2(side / 2), 0)), inFrame = coverage(sdBox(c, vec2(side / 2 + FRAME), 0.12));
  let albedo: N<'vec3'> = mix(mix(lawn, wood, inFrame), bed, inBed);

  const lo = window.xy, hi = window.zw, gap = sdBox(q.sub(lo.add(hi).mul(0.5)), hi.sub(lo).mul(0.5), 0);
  const mist = texture(mottle, q.mul(0.19).add(vec2(time.mul(0.011), time.mul(-0.007)))).r;
  const outside = coverage(gap.negate()).mul(fog).mul(inBed);
  albedo = mix(albedo, mix(albedo.mul(0.28), rgb(FOG).mul(mist.mul(0.7).add(0.6)), 0.5), outside.mul(0.88));
  const border = coverage(abs(gap).sub(0.022)).mul(fog).mul(inBed);
  const halo = exp(max(gap, 0).mul(-16)).mul(outside).mul(0.3);

  const lead = exp(u.sub(done).div(0.14).pow(2).negate()).mul(step(0.001, clock));
  const glow = rgb(GOLD).mul(step(u, done).mul(0.95).add(lead.mul(1.4))).mul(groove).mul(inFrame).mul(inBed.oneMinus()).mul(notch.mul(-0.55).add(1));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo).add(glow).add(rgb(TINTS[0]).mul(border.mul(1.5).add(halo)));
  return { material, window, fog, clock, time, textures: [mottle, grain] };
}

export function spotMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const fade = attribute<'vec4'>('aSpot', 'vec4').x;
  const p = uv().sub(0.5).mul(SPOT);
  const leaf = min(ellipse(turn(p.sub(vec2(-0.075, 0.01)), 0.5), [0, 0], [0.13, 0.05]), ellipse(turn(p.sub(vec2(0.07, -0.02)), -0.4), [0, 0], [0.12, 0.048]));
  const crumbs = min(min(sdCircle(p.sub(vec2(0.02, 0.07)), 0.017), sdCircle(p.sub(vec2(-0.05, -0.08)), 0.013)), sdCircle(p.sub(vec2(0.11, 0.05)), 0.011));
  new Stack()
    .over(lit(mix(rgb(0x3a2e1d), rgb(0x8c8a55), coverage(leaf.add(0.01)))), coverage(leaf).mul(0.5))
    .over(lit(rgb(0x6e2230)), coverage(crumbs).mul(0.8))
    .paint(material, fade);
  return material;
}

export function dropMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const fade = attribute<'vec4'>('aDrop', 'vec4').x;
  const p = uv().sub(0.5).mul(DROP);
  material.colorNode = lit(rgb(TINTS[0]).mul(0.9)).add(rgb(TINTS[0]).mul(0.25));
  material.opacityNode = coverage(sdCircle(p, 0.034)).mul(fade);
  return material;
}

export function bunchMaterial(): { material: THREE.MeshBasicNodeMaterial; blend: Level } {
  const material = sheet(), blend = uniform(0);
  const state = attribute<'vec4'>('aBunch', 'vec4'), before = attribute<'vec4'>('aRingA', 'vec4'), after = attribute<'vec4'>('aRingB', 'vec4');
  const level = state.x, fade = state.y, lift = state.z, seen = state.w;
  const p = uv().sub(0.5).mul(BUNCH);

  const leaves = (at: N<'vec2'>) => {
    let d: N<'float'> = float(9);
    for (let k = 0; k < 5; k++) d = min(d, ellipse(turn(at, -((k / 5) * 2 * Math.PI + 0.31)), [0.215, 0], [0.175, 0.085]));
    return min(d, sdCircle(at, 0.12));
  };
  const d = 0.142, r = 0.088;
  const places: [number, number, N<'float'>][] = [
    [0, 0, step(0.5, mod(level, 2))],
    [-d, d, step(1.5, level)], [d, -d, step(1.5, level)],
    [d, d, step(3.5, level)], [-d, -d, step(3.5, level)],
    [-d, 0, step(5.5, level)], [d, 0, step(5.5, level)],
  ];
  const grow = lift.mul(0.16).add(1), bp = p.div(grow);
  let nearest: N<'float'> = float(9), second: N<'float'> = float(9), centre: N<'vec2'> = vec2(0, 0);
  for (const [x, y, on] of places) {
    const dist = select(on.greaterThan(0.5), length(bp.sub(vec2(x, y))), float(9));
    second = min(second, max(nearest, dist));
    centre = select(dist.lessThan(nearest), vec2(x, y), centre);
    nearest = min(nearest, dist);
  }
  const berries = nearest.sub(r).mul(grow);
  const split = abs(second.sub(nearest)).mul(grow).sub(0.009);
  const shade = bp.sub(centre).div(r);
  let berry: N<'vec3'> = mix(rgb(BERRY), rgb(0x6f1426), smoothstep(0.2, 1.05, length(shade.sub(vec2(-0.35, 0.35)))));
  berry = mix(berry, rgb(0xffc2c9), coverage(length(shade.sub(vec2(-0.38, 0.4))).sub(0.2)).mul(0.9));
  berry = mix(berry, rgb(0x2c3d1c), coverage(sdCircle(shade.sub(vec2(0.05, 0.64)), 0.16)).mul(0.85));
  berry = berry.add(rgb(GOLD).mul(lift.mul(0.08)));

  const leaf = leaves(p);
  const cast = p.sub(vec2(0.035, -0.045).mul(lift.mul(1.6).add(1)));
  const shadow = smoothstep(-0.04, lift.mul(0.06).add(0.05), min(leaves(cast), length(cast.div(grow)).sub(0.24))).oneMinus().mul(0.36).mul(lift.mul(-0.4).add(1));
  const vein = Array.from({ length: 5 }, (_, k) => turn(p, -((k / 5) * 2 * Math.PI + 0.31))).map((arm) => coverage(abs(arm.y).sub(0.005)).mul(step(0.06, arm.x)).mul(step(arm.x, 0.37))).reduce((u, v) => max(u, v)).mul(0.35);
  const leafColor = mix(mix(rgb(LEAF[1]), rgb(LEAF[0]), smoothstep(0.4, 0.08, length(p))), rgb(0x8cc46a), vein);

  const ring = 0.405, segment = (2 * Math.PI * ring);
  const around = fract(atan(p.x, p.y).div(2 * Math.PI).add(1));
  const slot = floor(around.mul(level)), within = fract(around.mul(level));
  const band = max(abs(length(p).sub(ring)).sub(0.024), float(0.02).sub(min(within, within.oneMinus()).mul(segment).div(level)));
  const tintOf = (load: N<'vec4'>) => select(slot.lessThan(load.x), rgb(TINTS[0]), select(slot.lessThan(load.x.add(load.y)), rgb(TINTS[1]), select(slot.lessThan(load.x.add(load.y).add(load.z)), rgb(TINTS[2]), rgb(0xf4e7c9))));
  const weightOf = (load: N<'vec4'>) => load.w.mul(step(slot.add(0.5), load.x.add(load.y).add(load.z)).mul(0.62).add(0.33));
  const bandColor = hidden(mix(tintOf(before), tintOf(after), blend).mul(lift.mul(0.5).add(1.15)), seen);
  const bandAlpha = coverage(band).mul(mix(weightOf(before), weightOf(after), blend)).mul(seen.mul(0.5).add(0.5));
  const halo = exp(max(abs(length(p).sub(ring)).sub(0.03), 0).mul(-26)).mul(lift).mul(0.55).mul(smoothstep(0.5, 0.44, length(p)));

  new Stack()
    .over(vec3(0), shadow)
    .over(hidden(mix(rgb(INK), leafColor, coverage(leaf.add(0.012))), seen), coverage(leaf))
    .over(hidden(mix(rgb(INK), berry, coverage(split.negate())), seen), coverage(berries.sub(0.013)))
    .over(bandColor, bandAlpha)
    .over(rgb(GOLD).mul(1.6), halo)
    .paint(material, fade);
  return { material, blend };
}

export function antMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aAnt', 'vec4'), pose = attribute<'vec4'>('aPose', 'vec4');
  const tint = state.x, level = state.y, fade = state.z, seen = state.w;
  const walk = pose.x, lean = pose.y, heading = pose.z, wiggle = pose.w;
  const p = uv().sub(0.5).mul(ANT);

  const swing = sin(walk.mul(2 * Math.PI)).mul(0.032);
  const leg = (side: number, attach: readonly [number, number], knee: readonly [number, number], foot: readonly [number, number], group: number) => {
    const s = swing.mul(group), a = vec2(side * attach[0], attach[1]), k = vec2(side * knee[0], s.add(knee[1])), f = vec2(side * foot[0], s.mul(1.6).add(foot[1]));
    return min(sdSegment(p, a, k, 0.0115), sdSegment(p, k, f, 0.0085));
  };
  const legs = [
    leg(-1, [0.03, 0.085], [0.11, 0.135], [0.145, 0.225], 1), leg(1, [0.03, 0.085], [0.11, 0.135], [0.145, 0.225], -1),
    leg(-1, [0.035, 0.05], [0.13, 0.062], [0.19, 0.02], -1), leg(1, [0.035, 0.05], [0.13, 0.062], [0.19, 0.02], 1),
    leg(-1, [0.03, 0.02], [0.11, -0.04], [0.165, -0.15], 1), leg(1, [0.03, 0.02], [0.11, -0.04], [0.165, -0.15], -1),
  ].reduce((u, v) => min(u, v));
  const feeler = (side: number) => {
    const base = vec2(side * 0.035, 0.205), elbow = vec2(side * 0.098, wiggle.mul(side).add(0.262)), tip = vec2(wiggle.mul(side * 0.5).add(side * 0.068), 0.34);
    return min(sdSegment(p, base, elbow, 0.0085), sdSegment(p, elbow, tip, 0.0075));
  };
  const jaw = (side: number) => sdSegment(p, vec2(side * 0.03, 0.212), vec2(lean.mul(0.018).add(0.011).mul(side), 0.258), 0.0105);
  const limbs = min(min(legs, min(feeler(-1), feeler(1))), min(jaw(-1), jaw(1)));

  const gaster = ellipse(p, [0, -0.17], [0.1, 0.13]);
  const petiole = sdCircle(p.sub(vec2(0, -0.028)), 0.024);
  const thorax = ellipse(p, [0, 0.05], [0.05, 0.078]);
  const head = ellipse(p, [0, 0.168], [0.07, 0.062]);
  const body = min(min(gaster, petiole), min(thorax, head));

  const hue = select(tint.lessThan(0.5), rgb(TINTS[0]), select(tint.lessThan(1.5), rgb(TINTS[1]), rgb(TINTS[2])));
  const sheen = (centre: readonly [number, number], radius: number) => hue.mul(smoothstep(radius, 0, length(p.sub(vec2(centre[0] - 0.3 * radius, centre[1] + 0.35 * radius)))).mul(0.45));
  let color: N<'vec3'> = rgb(INK);
  color = mix(color, hue.mul(0.62).add(sheen([0, -0.17], 0.12)), coverage(gaster));
  color = mix(color, hue.mul(0.5), coverage(petiole));
  color = mix(color, hue.mul(0.7).add(sheen([0, 0.05], 0.07)), coverage(thorax));
  color = mix(color, hue.mul(0.55).add(sheen([0, 0.168], 0.07)), coverage(head));
  color = mix(color, rgb(INK), coverage(min(sdCircle(p.sub(vec2(-0.047, 0.19)), 0.014), sdCircle(p.sub(vec2(0.047, 0.19)), 0.014))));
  const pips = select(level.lessThan(1.5), sdCircle(p.sub(vec2(0, -0.17)), 0.025), min(sdCircle(p.sub(vec2(0, -0.122)), 0.024), sdCircle(p.sub(vec2(0, -0.218)), 0.024)));
  color = mix(color, rgb(INK), coverage(pips.sub(0.01)));
  color = mix(color, rgb(0xfff2d2), coverage(pips));
  color = mix(rgb(INK), color, coverage(body.add(0.011)));

  const sx = cos(heading).mul(0.045).sub(sin(heading).mul(0.055)), sy = sin(heading).mul(-0.045).sub(cos(heading).mul(0.055));
  const cast = p.sub(vec2(sx, sy));
  const shadow = smoothstep(-0.06, 0.06, min(ellipse(cast, [0, -0.04], [0.16, 0.25]), sdCircle(cast.sub(vec2(0, 0.17)), 0.09))).oneMinus().mul(0.3);
  const agent = step(tint, 0.5);

  new Stack()
    .over(rgb(TINTS[0]).mul(1.3), exp(length(p).mul(-7)).mul(agent).mul(0.3).mul(smoothstep(0.44, 0.34, length(p))))
    .over(vec3(0), shadow)
    .over(rgb(TINTS[0]).mul(1.3), coverage(abs(length(p).sub(0.3)).sub(0.01)).mul(agent).mul(0.85))
    .over(hidden(color, seen), coverage(min(body, limbs)).mul(seen.mul(0.45).add(0.55)))
    .paint(material, fade);
  return material;
}
