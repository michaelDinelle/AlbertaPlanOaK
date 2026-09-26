import * as THREE from 'three/webgpu';
import { Fn, abs, atan, clamp, cos, dot, exp, float, floor, int, length, max, min, mix, normalize, positionWorld, pow, select, sign, sin, smoothstep, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import { coverage, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Scalar = THREE.UniformNode<'float', number>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

const PITCH = 0.5;
const GROOVE = 0.1;
const RIM = 0.3;
export const PIN = 0.05;
export const LEDGER = 14;
export const COIN_PITCH = 0.44;
export const STAND = new THREE.Vector2(2.6, 3.6);
export const BEAM_END = new THREE.Vector2(-1.08, 1.29);
export const HANG = 0.36;
export const REST = -0.84;
export const BOWL = new THREE.Vector2(0, -0.74);
export const EMBER = 0xff7a2c;
export const GOLD = 0xf2c452;
export const COPPER = 0xc07443;

const TURNS = 5;
const MARGIN = 0.3;
const SUN = new THREE.Vector2(-0.6, 0.8).normalize();
const EMBER_CORE = 0xfff0c8;
const ASH = 0xe4dfd6;
const BED = 0x9d9990;
const BRONZE = 0x5e4429;
const PATINA = 0x4a5446;
const BRASS = 0xd6ab55;
const LINEN = 0xdcd2ba;
const WOOD = 0x6e4a2e;
const CORD = 0x9a2b22;

export interface Arm { ax: number; ay: number; dx: number; dy: number; length: number; start: number }
export interface Seal { arms: Arm[]; total: number; bed: THREE.Vector2; tray: THREE.Vector2; end: THREE.Vector2 }

export function sealLayout(): Seal {
  const arms: Arm[] = [], headings = [[1, 0], [0, 1], [-1, 0], [0, -1]] as const;
  let x = -PITCH / 2, y = 0, start = 0;
  for (let k = 1; k <= 4 * TURNS + 1; k++) {
    const [dx, dy] = headings[(k - 1) % 4]!, span = Math.ceil(k / 2) * PITCH;
    arms.push({ ax: x, ay: y, dx, dy, length: span, start });
    x += dx * span;
    y += dy * span;
    start += span;
  }
  const bed = new THREE.Vector2((TURNS + 0.5) * PITCH + GROOVE + MARGIN, TURNS * PITCH + GROOVE + MARGIN);
  return { arms, total: start, bed, tray: bed.clone().addScalar(RIM), end: new THREE.Vector2(x, y) };
}

function armAt(seal: Seal, s: number): Arm {
  return seal.arms.find((arm) => s <= arm.start + arm.length) ?? seal.arms[seal.arms.length - 1]!;
}

const bend = (jx: number, jy: number, din: readonly [number, number], dout: readonly [number, number], angle: number, out: THREE.Vector2) => {
  const u = GROOVE * (Math.sin(angle) - 1), v = GROOVE * (1 - Math.cos(angle));
  return out.set(jx + u * din[0] + v * dout[0], jy + u * din[1] + v * dout[1]);
};

export function pointAt(seal: Seal, s: number, out: THREE.Vector2): THREE.Vector2 {
  const at = Math.min(Math.max(s, 0), seal.total), arm = armAt(seal, at), t = at - arm.start, k = seal.arms.indexOf(arm);
  const quarter = Math.PI / (4 * GROOVE);
  if (seal.arms[k + 1] && t > arm.length - GROOVE) return bend(arm.ax + arm.dx * arm.length, arm.ay + arm.dy * arm.length, [arm.dx, arm.dy], [-arm.dy, arm.dx], (t - arm.length + GROOVE) * quarter, out);
  if (seal.arms[k - 1] && t < GROOVE) return bend(arm.ax, arm.ay, [arm.dy, -arm.dx], [arm.dx, arm.dy], (t + GROOVE) * quarter, out);
  return out.set(arm.ax + arm.dx * t, arm.ay + arm.dy * t);
}

export function sealPath(seal: Seal): THREE.Vector2[] {
  const points: THREE.Vector2[] = [pointAt(seal, 0, new THREE.Vector2())];
  seal.arms.forEach((arm, k) => {
    if (k > 0) for (let j = 1; j <= 4; j++) points.push(pointAt(seal, arm.start + (GROOVE * j) / 4, new THREE.Vector2()));
    const last = k === seal.arms.length - 1;
    points.push(pointAt(seal, arm.start + arm.length - (last ? 0 : GROOVE), new THREE.Vector2()));
    if (!last) for (let j = 1; j <= 4; j++) points.push(pointAt(seal, arm.start + arm.length - GROOVE + (GROOVE * j) / 4, new THREE.Vector2()));
  });
  return points;
}

export function pinAt(seal: Seal, s: number, out: THREE.Vector2): THREE.Vector2 {
  const arm = armAt(seal, s), reach = GROOVE + PIN + 0.035;
  pointAt(seal, s, out);
  return out.set(out.x - arm.dy * reach, out.y + arm.dx * reach);
}

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);

const board = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (width: number, height: number): N<'vec2'> => uv().sub(0.5).mul(vec2(width, height));
const sun = vec2(SUN.x, SUN.y);
const facing = (x: number, y: number) => x * SUN.x + y * SUN.y;
const gate = (x: N<'float'>): N<'float'> => clamp(x.mul(1e4), 0, 1);
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

function sheet(additive = false): THREE.MeshBasicNodeMaterial {
  return new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
}

function sprite(geometry: THREE.BufferGeometry, material: THREE.Material, order: number): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function table(lit: Lit, seal: Seal, standAt: THREE.Vector2, ledgerX: number): THREE.Mesh {
  const q = board(), material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const streak = noise(vec2(q.x.mul(0.35), q.y.mul(7.5))).mul(0.5).add(noise(vec2(q.x.mul(1.7), q.y.mul(31))).mul(0.25));
  const figure = smoothstep(-0.2, 0.6, noise(vec2(q.x.mul(0.12), q.y.mul(1.4)))).mul(0.18);
  const wood = mix(rgb(0x2c1a12), rgb(0x4a2d1e), streak.mul(0.5).add(0.5).add(figure)).mul(noise(q.mul(23)).mul(0.04).add(1));
  const trayShadow = smoothstep(-0.25, 0.45, sdBox(q.sub(vec2(0.14, -0.2)), vec2(seal.tray.x, seal.tray.y), 0.14)).oneMinus().mul(0.62);
  const standShadow = smoothstep(-0.1, 0.3, ellipse(q, [standAt.x + 0.12, standAt.y - STAND.y / 2 + 0.2], [1.25, 0.2])).oneMinus().mul(0.5);
  const cordShadow = smoothstep(-0.05, 0.2, abs(q.x.sub(ledgerX + 0.05)).sub(0.19)).oneMinus().mul(smoothstep(3.7, 3.3, abs(q.y))).mul(0.28);
  const vignette = smoothstep(4.5, 10, length(q.sub(vec2(0.6, 0)))).mul(0.55).oneMinus();
  material.colorNode = lit(wood.mul(max(max(trayShadow, standShadow), cordShadow).oneMinus()).mul(vignette));
  return sprite(flat(60, 40), material, 0);
}

export interface Burn { front: Scalar; renew: Scalar; glow: Scalar; tail: Scalar }

export function tray(lit: Lit, seal: Seal, steps: number, accent: number, burn: Burn, pins: readonly THREE.Vector2[]): THREE.Mesh {
  const material = sheet(), hw = GROOVE, unit = seal.total / steps;
  const bedHalf = vec2(seal.bed.x, seal.bed.y), trayHalf = vec2(seal.tray.x, seal.tray.y);
  const groove = Fn(() => {
    const q = board(), best = float(1e3).toVar(), arc = float(0).toVar(), side = float(0).toVar();
    const turn = (wx: N<'float'>, wy: N<'float'>, din: readonly [number, number], dout: readonly [number, number], at: number) => {
      const w = vec2(wx, wy), radius = length(w), cross = float(hw).sub(radius);
      const light = wx.mul(facing(...din)).add(wy.mul(facing(...dout))).div(max(radius, 1e-4)).negate();
      const angle = clamp(atan(wx, wy.negate()), 0, Math.PI / 2);
      return { d: abs(cross).sub(hw), s: angle.mul((4 * hw) / Math.PI).add(at - hw), side: cross.div(hw).mul(light) };
    };
    seal.arms.forEach((arm, k) => {
      const r = q.sub(vec2(arm.ax, arm.ay)), along = dot(r, vec2(arm.dx, arm.dy)), across = dot(r, vec2(-arm.dy, arm.dx));
      let d: N<'float'> = sdBox(vec2(along.sub(arm.length / 2), across), vec2(arm.length / 2 + hw, hw));
      let s: N<'float'> = clamp(along, 0, arm.length).add(arm.start);
      let lean: N<'float'> = across.div(hw).mul(facing(-arm.dy, arm.dx));
      const inner = across.lessThan(hw);
      if (seal.arms[k + 1]) {
        const bent = turn(along.sub(arm.length - hw), across.sub(hw), [arm.dx, arm.dy], [-arm.dy, arm.dx], arm.start + arm.length), there = along.greaterThan(arm.length - hw).and(inner);
        d = select(there, bent.d, d);
        s = select(there, bent.s, s);
        lean = select(there, bent.side, lean);
      }
      if (seal.arms[k - 1]) {
        const bent = turn(float(hw).sub(across), along.sub(hw), [arm.dy, -arm.dx], [arm.dx, arm.dy], arm.start), there = along.lessThan(hw).and(inner);
        d = select(there, bent.d, d);
        s = select(there, bent.s, s);
        lean = select(there, bent.side, lean);
      }
      const closer = d.lessThan(best);
      arc.assign(select(closer, s, arc));
      side.assign(select(closer, lean, side));
      best.assign(min(best, d));
    });
    return vec3(best, arc, side);
  });

  material.colorNode = Fn(() => {
    const q = board(), g = groove().toVar(), d = g.x, arc = g.y, side = g.z;
    const inside = coverage(d), depth = clamp(d.negate().div(hw), 0, 1);
    const speck = grain(floor(q.mul(90))), fine = noise(q.mul(41)), broad = noise(q.mul(2.3));

    const dBed = sdBox(q, bedHalf, 0.04);
    const edgeX = float(seal.bed.x).sub(abs(q.x)), edgeY = float(seal.bed.y).sub(abs(q.y));
    const edge = select(edgeX.lessThan(edgeY), vec2(sign(q.x), 0), vec2(0, sign(q.y)));
    const rimShade = exp(min(dBed, 0).div(0.14)).mul(max(dot(edge, sun), 0).mul(0.3).add(0.22));
    const lip = smoothstep(0.045, 0, d).mul(gate(d)).mul(0.1);
    const pressed = rgb(BED).mul(broad.mul(0.05).add(fine.mul(0.035)).add(1)).mul(select(speck.x.greaterThan(0.94), float(0.86), float(1)));
    const bed = pressed.mul(lip.add(1)).mul(rimShade.oneMinus());

    const powderBase = rgb(accent).mul(0.46).mul(fine.mul(0.08).add(broad.mul(0.05)).add(1));
    const powder = mix(mix(powderBase, rgb(accent).mul(0.78), gate(speck.x.sub(0.9)).mul(0.8)), rgb(0x2a3018), gate(float(0.07).sub(speck.y)).mul(0.7));
    const gap = abs(arc.mul(steps / seal.total).add(0.5).fract().sub(0.5)).mul(unit);
    const fresh = steps > 1 ? mix(powder, rgb(0x5c5a4a), coverage(gap.sub(0.009)).mul(0.7 * Math.min(1, Math.max(0.3, (unit - 0.06) / 0.5)))) : powder;
    const ash = rgb(ASH).mul(fine.mul(0.05).add(1)).mul(select(speck.y.greaterThan(0.95), float(0.7), float(1)));
    const burnt = coverage(arc.sub(burn.front)), ashen = burnt.mul(burn.renew.oneMinus());
    const behind = max(burn.front.sub(arc), 0);
    const char = exp(behind.div(0.05).negate()).mul(burnt).mul(burn.glow);
    const wall = smoothstep(0, 0.55, depth).mul(0.3).add(0.7);
    const floorShade = smoothstep(0.25, 1, side).mul(-0.28).add(1).mul(smoothstep(0.4, 1, side.negate()).mul(0.14).add(1));
    const channel = mix(fresh, mix(ash, rgb(0x3a2a22), char.mul(0.8)), ashen).mul(wall).mul(floorShade);

    let albedo: N<'vec3'> = mix(bed, channel, inside);
    for (const pin of pins) {
      const p = q.sub(vec2(pin.x, pin.y)), head = sdCircle(p, PIN), shade = sdCircle(p.sub(vec2(0.022, -0.028)), PIN);
      albedo = albedo.mul(smoothstep(-0.02, 0.03, shade).mul(0.45).add(0.55));
      const shine = exp(length(p.sub(vec2(-0.016, 0.018))).mul(-60)).mul(0.9);
      albedo = mix(albedo, rgb(BRASS).mul(shine.add(0.75)), coverage(head));
    }

    const outward = normalize(max(abs(q).sub(bedHalf), 1e-4).mul(sign(q)));
    const across = clamp(dBed.div(RIM), 0, 1);
    const tilt = dot(outward, sun).mul(cos(across.mul(Math.PI)).negate());
    const metal = mix(rgb(BRONZE), rgb(PATINA), smoothstep(0.15, 0.6, noise(q.mul(5.3))).mul(0.3)).mul(noise(q.mul(47)).mul(0.06).add(1));
    const crest = smoothstep(0.035, 0, abs(dBed.sub(RIM * 0.62))).mul(max(tilt, 0)).mul(0.25);
    const rim = metal.mul(tilt.mul(0.55).add(1)).mul(smoothstep(0, 0.2, across).mul(0.35).add(0.65)).add(rgb(0xffe0a8).mul(crest));
    albedo = mix(rim, albedo, coverage(dBed));

    const glowing = exp(behind.div(burn.tail).negate()).mul(burnt).mul(burn.glow).mul(inside).mul(burn.renew.oneMinus());
    const heat = rgb(EMBER).mul(glowing.mul(0.9)).add(rgb(0xffc27a).mul(pow(glowing, 6).mul(0.9)));
    return lit(albedo).add(heat);
  })();
  material.opacityNode = coverage(sdBox(board(), trayHalf, 0.14));
  return sprite(flat(2 * seal.tray.x + 0.2, 2 * seal.tray.y + 0.2), material, 1);
}

export function emberSprite(heat: Scalar, flicker: Scalar): THREE.Mesh {
  const material = sheet(true), p = local(0.9, 0.9), r = length(p);
  const core = exp(r.div(0.03).pow(2).negate()), halo = exp(r.div(0.075).negate()).mul(0.8).add(exp(r.div(0.22).negate()).mul(0.22));
  material.colorNode = rgb(EMBER_CORE).mul(core.mul(3.2)).add(rgb(EMBER).mul(halo.mul(1.5))).mul(heat).mul(flicker).mul(smoothstep(0.45, 0.3, r));
  return sprite(flat(0.9, 0.9), material, 4);
}

const SMOKE = 2.6;
export const SMOKE_ROOT = new THREE.Vector2(SMOKE / 2 - 0.35, SMOKE / 2 - 0.25);

export function smokeSprite(heat: Scalar, clock: Scalar): THREE.Mesh {
  const material = sheet(), box = local(SMOKE, SMOKE), p = box.add(vec2(SMOKE_ROOT.x, SMOKE_ROOT.y));
  const u = dot(p, vec2(0.34, 0.94)), v = dot(p, vec2(-0.94, 0.34));
  const wobble = sin(u.mul(4.2).sub(clock.mul(1.7))).mul(0.07).mul(u).add(noise(vec2(u.mul(1.6).sub(clock.mul(0.5)), clock.mul(0.13))).mul(0.16).mul(u));
  const width = u.mul(0.11).add(0.012), density = exp(v.sub(wobble).div(width).pow(2).negate());
  const breakup = smoothstep(-0.45, 0.5, noise(p.mul(2.6).add(vec2(0, clock.mul(-0.55)))));
  const fade = smoothstep(0, 0.08, u).mul(exp(u.mul(-1.25)));
  const border = smoothstep(SMOKE / 2, SMOKE / 2 - 0.3, max(abs(box.x), abs(box.y)));
  material.colorNode = mix(rgb(0xffc9a0), rgb(0xb9bec8), smoothstep(0, 0.6, u));
  material.opacityNode = density.mul(breakup).mul(fade).mul(heat).mul(border).mul(0.55);
  return sprite(flat(SMOKE, SMOKE), material, 5);
}

export function puffSprite(age: Scalar, clock: Scalar): THREE.Mesh {
  const material = sheet(), p = local(1.4, 1.4), grow = pow(age, 0.6);
  const radius = grow.mul(0.34).add(0.06), r = length(p.sub(vec2(0, grow.mul(0.12))));
  const cloud = exp(r.div(radius).pow(2).negate()).mul(smoothstep(-0.6, 0.4, noise(p.mul(7).add(clock.mul(0.4)))).mul(0.6).add(0.4));
  material.colorNode = rgb(0xc4c6cc);
  material.opacityNode = cloud.mul(age.oneMinus().pow(1.6)).mul(gate(age)).mul(gate(age.oneMinus())).mul(0.62);
  return sprite(flat(1.4, 1.4), material, 5);
}

export function threadSprite(lit: Lit, span: number, burn: Scalar, alpha: Scalar): THREE.Mesh {
  const material = sheet(), p = local(span, 0.12), along = uv().x;
  const kept = gate(along.sub(burn));
  const scorch = exp(max(along.sub(burn), 0).mul(span / 0.035).negate()).mul(kept).mul(gate(burn.sub(0.001)));
  material.colorNode = mix(lit(rgb(LINEN)), rgb(EMBER).mul(1.6), scorch);
  material.opacityNode = coverage(abs(p.y).sub(0.011)).mul(kept).mul(alpha);
  return sprite(flat(span, 0.12), material, 3);
}

export function flameSprite(heat: Scalar): THREE.Mesh {
  const material = sheet(true), p = local(0.5, 0.5), r = length(p.mul(vec2(1, 0.8)));
  material.colorNode = rgb(EMBER_CORE).mul(exp(r.div(0.02).pow(2).negate()).mul(2.6)).add(rgb(EMBER).mul(exp(r.div(0.05).negate()).mul(1.3))).mul(heat).mul(smoothstep(0.25, 0.15, r));
  return sprite(flat(0.5, 0.5), material, 7);
}

export interface Bell { thread: Scalar; burn: Scalar; armed: Scalar; ball: Scalar; loose: Scalar; ring: Scalar }

export const threadPoint = (burn: number, out: THREE.Vector2): THREE.Vector2 => {
  const across = -BEAM_END.x, drop = BEAM_END.y - HANG - 0.15, u = burn * (across + drop);
  return u <= across ? out.set(BEAM_END.x + u, BEAM_END.y) : out.set(0, BEAM_END.y - (u - across));
};

export function standSprite(lit: Lit, bell: Bell): THREE.Mesh {
  const material = sheet(), p = local(STAND.x, STAND.y);
  const woodTone = rgb(WOOD).mul(noise(vec2(p.x.mul(2.2), p.y.mul(14))).mul(0.12).add(1));
  let color: N<'vec3'> = vec3(0), solid: N<'float'> = float(0);
  const layer = (shape: N<'float'>, tone: N<'vec3'>, alpha: N<'float'> | number = 1) => {
    const c = coverage(shape).mul(alpha);
    color = mix(color, tone, c);
    solid = max(solid, c);
    return c;
  };
  const posts = min(sdBox(p.sub(vec2(-0.8, -0.14)), vec2(0.055, 1.28), 0.02), sdBox(p.sub(vec2(0.8, -0.14)), vec2(0.055, 1.28), 0.02));
  const lift = smoothstep(0.62, 1.08, abs(p.x)).mul(0.06);
  layer(posts, woodTone.mul(abs(abs(p.x).sub(0.8)).mul(-4).add(1.05)));
  layer(sdBox(p.sub(vec2(0, -1.47)), vec2(1.02, 0.08), 0.03), woodTone.mul(p.y.add(1.47).mul(2.5).add(1)));
  layer(sdBox(p.sub(vec2(0, lift.mul(0.5).add(1.2))), vec2(1.08, lift.mul(0.5).add(0.07)), 0.02), woodTone.mul(p.y.sub(1.2).mul(3).add(1.1)));
  layer(sdBox(p.sub(vec2(0, -1.335)), vec2(0.3, 0.055), 0.02), rgb(0x2b1d14));

  const mouth = ellipse(p, [BOWL.x, BOWL.y], [0.62, 0.13]);
  layer(mouth, rgb(0x1a120c).mul(p.y.sub(BOWL.y).mul(2).add(1)));

  const across = -BEAM_END.x, drop = BEAM_END.y - HANG - 0.15;
  const along = select(p.y.greaterThan(BEAM_END.y - 0.02), p.x.sub(BEAM_END.x), float(across).add(float(BEAM_END.y).sub(p.y))).div(across + drop);
  const cord = min(sdSegment(p, vec2(BEAM_END.x, BEAM_END.y), vec2(0, BEAM_END.y), 0.011), sdSegment(p, vec2(0, BEAM_END.y), vec2(0, HANG + 0.15), 0.011));
  layer(cord, rgb(LINEN).mul(1.05), bell.thread.mul(gate(along.sub(bell.burn))));

  const brass = (at: N<'vec2'>) => {
    const shine = exp(length(p.sub(at).sub(vec2(-0.05, 0.055))).mul(-18));
    return rgb(BRASS).mul(shine.mul(0.9).add(0.62)).mul(smoothstep(0.16, -0.05, length(p.sub(at).sub(vec2(0.06, -0.07)))).mul(0.25).add(0.75));
  };
  layer(sdCircle(p.sub(vec2(0, HANG)), 0.15), brass(vec2(0, HANG)), bell.armed);
  const looseAt = vec2(0, bell.ball);
  layer(sdCircle(p.sub(looseAt), 0.15), brass(looseAt), bell.loose);

  const belly = max(ellipse(p, [BOWL.x, BOWL.y], [0.62, 0.56]), p.y.sub(BOWL.y));
  const bronze = rgb(BRONZE).mul(p.x.mul(-0.5).add(1.1)).mul(smoothstep(-0.6, 0.1, p.y.sub(BOWL.y)).mul(0.35).add(0.75)).mul(noise(p.mul(9)).mul(0.08).add(1));
  const front = layer(belly, bronze);
  const lip = layer(abs(mouth).sub(0.018), rgb(0xc99a5a), smoothstep(BOWL.y + 0.2, BOWL.y - 0.02, p.y).max(0.55));

  const age = bell.ring.oneMinus(), waveR = length(p.sub(vec2(BOWL.x, BOWL.y)).mul(vec2(1, 1.6)));
  let waves: N<'float'> = float(0);
  for (const lag of [0, 0.22, 0.44]) {
    const phase = clamp(age.mul(1.5).sub(lag), 0, 1);
    waves = max(waves, coverage(abs(waveR.sub(phase.mul(0.85).add(0.7))).sub(0.016)).mul(phase.oneMinus()).mul(gate(phase)).mul(gate(abs(p.x).sub(0.66))));
  }
  const glow = rgb(0xffd98a).mul(exp(max(mouth, 0).div(0.08).negate()).mul(bell.ring).mul(1.6).mul(max(front.oneMinus(), lip)));
  material.colorNode = lit(color).add(glow).add(rgb(0xffd98a).mul(waves.mul(bell.ring).mul(1.4)));
  material.opacityNode = max(solid, waves.mul(bell.ring).mul(0.9));
  return sprite(flat(STAND.x, STAND.y), material, 6);
}

export function slots(count: number) {
  const values = new Array<number>(count).fill(0);
  return { node: uniformArray(values, 'float' as const), values };
}
type Slots = ReturnType<typeof slots>;

const LEDGER_HEIGHT = (LEDGER + 2) * COIN_PITCH;
export const LEDGER_TOP = LEDGER_HEIGHT / 2 - COIN_PITCH * 1.2;

export function ledgerSprite(lit: Lit, coins: Slots, slide: Scalar): THREE.Mesh {
  const material = sheet(), p = local(0.9, LEDGER_HEIGHT);
  const j = clamp(floor(float(LEDGER_TOP).sub(p.y).div(COIN_PITCH).add(1.5).sub(slide)), 0, LEDGER);
  const centreY = float(LEDGER_TOP).sub(j.sub(1).add(slide).mul(COIN_PITCH));
  const value = coins.node.element(int(j));
  const presence = select(j.lessThan(0.5), slide, select(j.greaterThan(LEDGER - 0.5), slide.oneMinus(), float(1)));
  const gold = gate(value.sub(0.5)), copper = gate(value.negate().sub(0.5)), minted = max(gold, copper).mul(presence);
  const radius = mix(float(0.165), float(0.195), gold).mul(select(j.lessThan(0.5), slide.oneMinus().mul(0.35).add(1), float(1)));
  const c = p.sub(vec2(0, centreY)), r = length(c);
  const face = r.sub(radius), hole = sdBox(c, vec2(radius.mul(0.26)), 0.006);
  const rim = coverage(abs(r.sub(radius.mul(0.86))).sub(radius.mul(0.06)));
  const glyphs = [[0, 1], [0, -1], [1, 0], [-1, 0]].map(([gx, gy]) => sdBox(c.sub(vec2(gx!, gy!).mul(radius.mul(0.56))), vec2(radius.mul(gx === 0 ? 0.09 : 0.03), radius.mul(gx === 0 ? 0.03 : 0.09)), 0.004)).reduce((a, b) => min(a, b));
  const shine = exp(length(c.div(radius).sub(vec2(-0.4, 0.45))).mul(-3.2)).mul(0.55);
  const metal = mix(rgb(COPPER), rgb(GOLD), gold).mul(shine.add(0.72)).mul(rim.mul(0.3).add(1)).mul(coverage(glyphs).mul(0.22).add(1));
  const coin = metal.mul(smoothstep(0, -0.02, hole).mul(0.5).add(0.5));
  const cord = coverage(abs(p.x).sub(0.009)).mul(gate(float(LEDGER_HEIGHT / 2 - 0.12).sub(abs(p.y))));
  const knot = coverage(sdCircle(p.sub(vec2(0, LEDGER_HEIGHT / 2 - 0.18)), 0.045));
  const disk = coverage(face).mul(minted), coinCover = disk.mul(coverage(hole.negate()));
  const shadow = smoothstep(0.06, -0.03, length(c.sub(vec2(0.03, -0.035))).sub(radius)).mul(minted).mul(0.5);
  const strung = max(cord.mul(disk.oneMinus()), knot);
  const color = mix(mix(vec3(0), rgb(CORD), strung), coin, coinCover);
  material.colorNode = lit(color).add(rgb(GOLD).mul(gold.mul(presence).mul(coverage(face)).mul(0.12)));
  material.opacityNode = max(max(strung, coinCover), shadow);
  return sprite(flat(0.9, LEDGER_HEIGHT), material, 6);
}
