import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, smoothstep, step, texture, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
export type Scalar = THREE.UniformNode<'float', number>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

const PITCH = 3.8;
const TRAY_W = 3;
export const CARD_W = 2.6;
export const CARD_H = 0.13;
const TAB_W = 0.46;
const ROW = 0.165;
export const ROWS = 16;
export const SLOT = 3;
export const SIGMA = 0.34;
export const METER_Y = 3.08;
const FACE = 0.95;
export const SLOT_Y = 1.66;
export const RAIL_Y = 0.9;
export const PLATE_Y = 0.37;
export const LABEL_Y = -0.14;
const LABEL_H = 0.32;
export const LABEL_SIGMA = 0.18;
export const LABEL_REACH = 0.9;
export const MOUTH_Y = -0.62;
export const HOUSING_Y = 2.73;
export const HOUSING_H = 1.54;
export const PLUS = TOKENS.cueMagenta;
export const MINUS = TOKENS.cueCyan;
export const GAIN = TOKENS.celebrationGold;
export const LOSS = TOKENS.failureBlue;
export const BIAS = 0xf2e6cc;
export const ZERO = 0x161b26;
export const SPREAD = 0.9;
export const CAP = 0.05;

const GAP = 0.011;
const TAB_GAP = 0.05;
const PAD = 0.008;
const NEEDLE = 0.034;
const OVER = 0.05;
const TRAY_TOP = 0.52;
const BRASS = 0xc39a52;
const CHROME = 0x9aa3b3;

export const rowY = (rank: number): number => MOUTH_Y - (rank + 1) * ROW;
export const TRAY_BOTTOM = rowY(ROWS - 1) - ROW / 2 - 0.13;
const LANE = TAB_W - TAB_GAP - GAP;
export const REACH = LANE - 2 * GAP - 2 * PAD;
export const tabStart = (): number => -CARD_W / 2 + CARD_W - TAB_W + TAB_GAP + GAP + PAD;
export const SHARE_LEFT = -CARD_W / 2 + 0.8;
export const SHARE_WIDTH = CARD_W - 0.8;

export class Layout {
  readonly pockets: number[];
  readonly columns: number[];
  readonly bar: number;
  readonly cell: number;
  readonly face: number;
  readonly divider: number;

  constructor(readonly inputs: number, readonly arms: number) {
    this.pockets = Array.from({ length: arms }, (_, a) => (a - (arms - 1) / 2) * PITCH);
    this.cell = (CARD_W - TAB_W) / inputs;
    const left = -SLOT * CARD_W / 2;
    this.columns = Array.from({ length: inputs }, (_, k) => left + (k + 0.5) * this.cell * SLOT);
    this.columns.push(SLOT * (tabStart() - GAP - PAD + LANE / 2));
    this.bar = 0.5 * this.cell * SLOT;
    this.face = SLOT * CARD_W / 2 + 0.14;
    this.divider = SLOT * (-CARD_W / 2 + CARD_W - TAB_W + TAB_GAP / 2);
  }

  get span(): number {
    return this.pockets[this.pockets.length - 1]! + TRAY_W / 2;
  }
}

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

function sprite(color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  if (additive) material.blending = THREE.AdditiveBlending;
  return material;
}

const heat = (value: N<'float'>): N<'vec3'> => mix(rgb(ZERO), select(value.greaterThanEqual(0), rgb(PLUS), rgb(MINUS)), float(1).sub(exp(abs(value).mul(-SPREAD))));

const payTint = (pay: N<'float'>): N<'vec3'> => select(pay.greaterThanEqual(0), rgb(GAIN), rgb(LOSS)).mul(1.1);

function tab(x: N<'float'>, y: N<'float'>, pay: N<'float'>, shown: N<'float'>, height: number): { lane: N<'float'>; filled: N<'float'> } {
  const tx = x.sub(CARD_W - TAB_W + TAB_GAP);
  const lane = sdBox(vec2(tx.sub(LANE / 2), y), vec2(LANE / 2 - GAP, height / 2 - GAP), 0.014);
  const reach = abs(pay).mul(REACH), along = tx.sub(GAP + PAD);
  const bar = max(max(along.negate(), along.sub(reach)), abs(y).sub(height / 2 - GAP - PAD));
  return { lane, filled: coverage(bar).mul(step(1e-4, reach)).mul(shown) };
}

function sample(values: THREE.DataTexture, layout: Layout, rows: number, x: N<'float'>): { k: N<'float'>; value: N<'float'> } {
  const k = min(floor(x.div(layout.cell)), layout.inputs - 1), row = attribute<'float'>('row', 'float');
  return { k, value: texture(values, vec2(k.add(0.5).div(layout.inputs), row.add(0.5).div(rows))).r };
}

export function cardMaterial(light: Light2D, values: THREE.DataTexture, layout: Layout, rows: number, accent: number): THREE.MeshBasicNodeMaterial {
  const look = attribute<'vec4'>('look', 'vec4');
  const pay = look.x, fade = look.y, glow = look.z, shown = look.w;
  const cell = layout.cell, end = CARD_W - TAB_W;
  const p = uv().sub(0.5).mul(vec2(CARD_W, CARD_H)), x = p.x.add(CARD_W / 2);
  const body = sdBox(p, vec2(CARD_W / 2, CARD_H / 2), 0.03);
  const { k, value } = sample(values, layout, rows, x);
  const box = max(sdBox(vec2(x.sub(k.add(0.5).mul(cell)), p.y), vec2(cell / 2 - GAP, CARD_H / 2 - GAP), 0.014), x.sub(end));
  const { lane, filled } = tab(x, p.y, pay, shown, CARD_H);
  const rim = coverage(abs(body.add(0.011)).sub(0.009)).mul(glow);
  let color: N<'vec3'> = light.lit(rgb(0x2a3140));
  color = mix(color, heat(value).mul(glow.mul(0.45).add(1)), coverage(box));
  color = mix(color, rgb(0x07090e), coverage(lane));
  color = mix(color, payTint(pay), filled);
  color = mix(color, rgb(accent).mul(1.6), rim);
  return sprite(color, coverage(body).mul(fade));
}

export function labelMaterial(values: THREE.DataTexture, layout: Layout, rows: number): THREE.MeshBasicNodeMaterial {
  const look = attribute<'vec4'>('look', 'vec4');
  const pay = look.x, fade = look.y, shown = look.w;
  const cell = layout.cell, end = CARD_W - TAB_W;
  const p = uv().sub(0.5).mul(vec2(CARD_W, 2 * LABEL_REACH)), x = p.x.add(CARD_W / 2);
  const { k, value } = sample(values, layout, rows, x);
  const reach = abs(value).mul(LABEL_SIGMA), s = p.y.mul(select(value.greaterThanEqual(0), float(1), float(-1)));
  const across = abs(x.sub(k.add(0.5).mul(cell))).sub(cell / 2 - 2 * GAP), pixel = max(fwidth(p.x), 1e-6);
  const smooth = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel)), 0, 1);
  const bar = smooth(max(max(across, x.sub(end)), max(s.negate(), s.sub(reach))));
  const needle = smooth(max(max(across.sub(GAP), x.sub(end)), abs(s.sub(reach)).sub(0.012)));
  const zero = coverage(max(abs(p.y).sub(0.005), x.sub(end - GAP))).mul(0.6);
  const { lane, filled } = tab(x, p.y, pay, shown, CARD_H);
  const hue = heat(value);
  let color: N<'vec3'> = rgb(0x3a4458);
  color = mix(color, rgb(0x07090e), coverage(lane));
  color = mix(color, hue.mul(1.15), bar);
  color = mix(color, hue.mul(1.25).add(vec3(0.03)), needle);
  color = mix(color, payTint(pay), filled);
  return sprite(color, max(max(bar, needle), max(zero, max(coverage(lane), filled))).mul(fade));
}

export const HALO = 0.32;

export function haloMaterial(tint: Tint, grow: Scalar, fade: Scalar): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(vec2(CARD_W + 2 * HALO, CARD_H + 2 * HALO));
  const d = sdBox(p, vec2(CARD_W / 2, CARD_H / 2), 0.03).sub(grow);
  return sprite(tint, coverage(abs(d).sub(0.013)).mul(fade), true);
}

export const meterWidth = (bar: number): number => bar + 2 * OVER;

export function meterMaterial(width: number): THREE.MeshBasicNodeMaterial {
  const look = attribute<'vec4'>('look', 'vec4');
  const value = look.x, reach = look.y, bias = look.z;
  const total = reach.add(2 * CAP), p = uv().sub(0.5).mul(vec2(width + 2 * OVER, total));
  const s = p.y.mul(select(value.greaterThanEqual(0), float(1), float(-1))).add(total.div(2)).sub(CAP);
  const body = sdBox(vec2(p.x, s.sub(reach.div(2))), vec2(width / 2, reach.div(2)), min(reach.div(2), 0.03));
  const needle = sdBox(vec2(p.x, s.sub(reach)), vec2(width / 2 + OVER * 0.8, NEEDLE / 2), NEEDLE / 2);
  const hue = mix(heat(value), rgb(BIAS).mul(0.85), bias);
  const rise = clamp(s.div(max(reach, 1e-3)), 0, 1);
  const color = mix(hue.mul(mix(float(0.6), float(1.15), rise)), hue.mul(1.25).add(vec3(0.03)), coverage(needle));
  return sprite(color, max(coverage(body).mul(0.95), coverage(needle)));
}

export function routerMaterial(accent: number, strength: Scalar): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(vec2(2, 2.6)), r = length(p);
  const ring = coverage(abs(r.sub(0.62)).sub(0.08));
  const core = coverage(sdCircle(p, 0.3));
  const beak = coverage(max(abs(p.x).mul(1.6).sub(p.y.add(1.25)).mul(0.53), p.y.add(0.78)));
  const halo = exp(r.mul(-3.6)).mul(0.5).mul(smoothstep(1.2, 0.8, r));
  const tone = rgb(accent).mul(strength);
  return sprite(mix(tone, tone.mul(0.6).add(vec3(0.35)), core), max(max(ring, core), max(beak.mul(0.9), halo)));
}

export function consoleMaterial(light: Light2D, layout: Layout, shares: Scalar[], accent: number): THREE.MeshBasicNodeMaterial {
  const q = vec2(positionWorld.x, positionWorld.z.negate());
  const span = layout.span, top = HOUSING_Y + HOUSING_H;
  const brushed = noise(vec2(q.x.mul(0.7), q.y.mul(16))).mul(0.04).add(noise(q.mul(0.35)).mul(0.05)).add(1);
  const plate = sdBox(q.sub(vec2(0, (TRAY_BOTTOM + top) / 2)), vec2(span + 0.3, (top - TRAY_BOTTOM) / 2 + 0.3), 0.42);
  let albedo: N<'vec3'> = mix(rgb(0x0a0c11), rgb(0x272d38).mul(brushed), coverage(plate));
  albedo = mix(albedo, rgb(0x0f1218), coverage(abs(plate.add(0.05)).sub(0.018)));
  albedo = mix(albedo, rgb(0x5c6576), coverage(abs(plate.add(0.02)).sub(0.008)).mul(0.6));
  for (const [sx, sy] of [[-1, 1], [1, 1], [-1, -1], [1, -1]] as const) {
    const at = q.sub(vec2(sx * (span + 0.05), sy > 0 ? top + 0.05 : TRAY_BOTTOM - 0.05));
    albedo = mix(albedo, rgb(CHROME).mul(0.7), coverage(sdCircle(at, 0.075)));
    albedo = mix(albedo, rgb(0x14171d), coverage(sdSegment(at, vec2(-0.05, 0.05), vec2(0.05, -0.05), 0.012)));
  }

  const housing = sdBox(q.sub(vec2(0, HOUSING_Y)), vec2(layout.face + 0.26, HOUSING_H), 0.24);
  albedo = mix(albedo, rgb(0x1a1f28), coverage(housing));
  albedo = mix(albedo, rgb(CHROME), coverage(abs(housing.add(0.04)).sub(0.022)).mul(0.8));
  const face = sdBox(q.sub(vec2(0, METER_Y)), vec2(layout.face, FACE), 0.1);
  const sheen = smoothstep(0.35, 0, abs(q.x.mul(0.22).add(q.y.sub(METER_Y)).add(0.4))).mul(0.018);
  albedo = mix(albedo, rgb(0x0c1017).add(sheen), coverage(face));
  albedo = mix(albedo, rgb(0x46505f), coverage(abs(face).sub(0.012)).mul(0.8));
  const inFace = coverage(face.add(0.02));
  const lane = layout.cell * SLOT, left = -SLOT * CARD_W / 2;
  const column = clamp(floor(q.x.sub(left).div(lane)), 0, layout.inputs - 1);
  const groove = min(
    sdBox(vec2(q.x.sub(column.add(0.5).mul(lane).add(left)), q.y.sub(METER_Y)), vec2(layout.bar / 2 + 0.03, FACE - 0.07), 0.03),
    sdBox(q.sub(vec2(layout.columns[layout.inputs]!, METER_Y)), vec2(layout.bar / 2 + 0.03, FACE - 0.07), 0.03),
  );
  albedo = mix(albedo, rgb(0x141b26), coverage(groove).mul(inFace));
  const level = q.y.sub(METER_Y).div(SIGMA);
  const tick = abs(fract(level.add(0.5)).sub(0.5)).mul(SIGMA);
  const ticks = coverage(tick.sub(0.005)).mul(step(abs(level), FACE / SIGMA)).mul(inFace);
  const edge = step(layout.face - 0.16, abs(q.x));
  albedo = mix(albedo, rgb(0x283243), ticks);
  albedo = mix(albedo, rgb(0x8793a8), ticks.mul(edge));
  albedo = mix(albedo, rgb(0x6a7890), coverage(abs(q.y.sub(METER_Y)).sub(0.011)).mul(inFace));
  albedo = mix(albedo, rgb(0x46505f), coverage(abs(q.x.sub(layout.divider)).sub(0.008)).mul(inFace).mul(step(abs(q.y.sub(METER_Y)), FACE - 0.05)));

  const slot = sdBox(q.sub(vec2(0, SLOT_Y)), vec2(SLOT * CARD_W / 2 + 0.1, SLOT * CARD_H / 2 + 0.08), 0.07);
  albedo = mix(albedo, rgb(0x040509), coverage(slot));
  albedo = mix(albedo, rgb(CHROME), coverage(abs(slot).sub(0.014)).mul(0.7));

  const first = layout.pockets[0]! - 0.55, last = layout.pockets[layout.arms - 1]! + 0.55;
  const rail = sdSegment(q, vec2(first, RAIL_Y), vec2(last, RAIL_Y), 0.045);
  const across = q.y.sub(RAIL_Y).div(0.045);
  albedo = mix(albedo, mix(rgb(0x3c414d), rgb(0xe3e8f0), float(1).sub(across.mul(across)).mul(0.9)), coverage(rail));
  for (const end of [first, last]) albedo = mix(albedo, rgb(CHROME).mul(0.8), coverage(sdCircle(q.sub(vec2(end, RAIL_Y)), 0.1)));
  for (const x of layout.pockets) albedo = mix(albedo, rgb(0x1a1f28), coverage(abs(sdCircle(q.sub(vec2(x, RAIL_Y)), 0.07)).sub(0.016)));

  const trayCentre = (TRAY_TOP + TRAY_BOTTOM) / 2, trayHalf = (TRAY_TOP - TRAY_BOTTOM) / 2;
  let glow: N<'vec3'> = vec3(0);
  layout.pockets.forEach((x, a) => {
    const tray = sdBox(q.sub(vec2(x, trayCentre)), vec2(TRAY_W / 2, trayHalf), 0.16);
    albedo = mix(albedo, rgb(0x0b0e14), coverage(tray));
    albedo = mix(albedo, rgb(CHROME).mul(0.75), coverage(abs(tray.add(0.02)).sub(0.014)));
    const name = sdBox(q.sub(vec2(x, PLATE_Y)), vec2(TRAY_W / 2 - 0.1, 0.09), 0.07);
    albedo = mix(albedo, rgb(0x343b48).mul(brushed), coverage(name));
    const frame = sdBox(q.sub(vec2(x, LABEL_Y)), vec2(CARD_W / 2 + 0.05, LABEL_H + 0.05), 0.06);
    albedo = mix(albedo, rgb(0x12161e), coverage(frame));
    albedo = mix(albedo, rgb(BRASS), coverage(abs(frame.add(0.02)).sub(0.018)));
    const mouth = sdBox(q.sub(vec2(x, MOUTH_Y)), vec2(CARD_W / 2, CARD_H / 2), 0.03);
    const dash = step(0.5, fract(q.x.sub(x).div(0.1)));
    albedo = mix(albedo, rgb(0x2c3342), coverage(abs(mouth).sub(0.005)).mul(dash));
    for (let n = 0; n <= a; n++) albedo = mix(albedo, rgb(BRASS), coverage(sdCircle(q.sub(vec2(x - CARD_W / 2 + 0.12 + n * 0.16, PLATE_Y)), 0.05)));
    const meter = sdBox(q.sub(vec2(x + SHARE_LEFT + SHARE_WIDTH / 2, PLATE_Y)), vec2(SHARE_WIDTH / 2, 0.035), 0.035);
    albedo = mix(albedo, rgb(0x05070b), coverage(meter));
    const fill = coverage(max(meter.add(0.012), q.x.sub(shares[a]!.mul(SHARE_WIDTH).add(x + SHARE_LEFT))));
    glow = glow.add(rgb(accent).mul(1.35).mul(fill));
  });

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = light.lit(albedo).add(glow);
  return material;
}
