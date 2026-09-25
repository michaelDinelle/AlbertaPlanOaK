import * as THREE from 'three/webgpu';
import { abs, clamp, float, floor, fract, max, min, mix, mod, select, smoothstep, sqrt, texture, uniform, uvec2, vec2 } from 'three/tsl';
import { fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';
import type { EventStream } from '@sim/types';
import { coverage, picture, sheet, type Lit } from './art';
import { COST, FOOD, LIGHT, PRESS, type History, type Totals } from './history';

type N<T extends string> = THREE.Node<T>;

export const REC = { x0: 1.7, x1: 8.1, y0: -3.2, y1: 2.5 };
const FOOT = -3.3;
export const PILOT = new THREE.Vector2(7.62, -2.88);
const DECK = { x0: 2.18, x1: 7.62, y0: -2.58, y1: 1.98 };
export const PAPER = { x0: 2.44, x1: 7.36, y0: -2.48, y1: 1.88 };
export const PEN_X = 6.72;
const TRACE = { y0: -1.62, y1: 1.62 };
export const SWEEP = 36;
const NOTCH = (TRACE.y1 - TRACE.y0) / SWEEP;
export const EVENT_Y = -2.08;
export const EVENT_H = 0.13;
const STEP_W = 0.036;
export const CONTACT = 0.45;
const PIP_T = 0.6;
const PIP_W = 0.07;
const PIP_H = 0.1;
export const WINDOW = Math.ceil((PEN_X - PAPER.x0) / STEP_W) + 2;
const TEXELS = 128;
const INK_HALF = 0.014;
const INK = 0x1c2a52;
const PIP_INK = 0xc2402c;
const LIGHT_INK = 0xb86a12;
const COST_INK = new THREE.Color(TOKENS.failureBlue).multiplyScalar(0.72).getHex();
const GRID = 0x8fbca0;

export const contactOf = (k: number) => k - 1 + CONTACT;
export const levelY = (level: number) => TRACE.y0 + level * NOTCH;
export const paperX = (tau: number, position: number) => PEN_X + (tau - position) * STEP_W;

export class Chart {
  readonly data = new Float32Array(TEXELS * 4);
  readonly texture = new THREE.DataTexture(this.data, TEXELS, 1, THREE.RGBAFormat, THREE.FloatType);
  readonly position = uniform(0);
  readonly pen = uniform(0);
  readonly pip = uniform(0);
  readonly pilot = uniform(1);
  private readonly totals: Totals = { presses: 0, pellets: 0 };
  filledAt = -1;
  complete = false;

  constructor() {
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.needsUpdate = true;
  }

  fill(history: History, stream: EventStream, head: number): void {
    const base = head - WINDOW - 2;
    let count = history.totals(stream, base - 1, this.totals).presses, known = true;
    for (let k = base; k <= head + 1; k++) {
      const o = (((k % TEXELS) + TEXELS) % TEXELS) * 4;
      const bits = k < 0 ? -1 : history.at(k);
      if (bits < 0 || !known) {
        if (k >= 0 && k <= head) known = false;
        this.data[o + 3] = -1;
        continue;
      }
      this.data[o] = count % SWEEP;
      this.data[o + 1] = bits & PRESS ? (bits & FOOD ? 1 : bits & COST ? 2 : 3) : 0;
      this.data[o + 2] = bits & LIGHT ? 1 : 0;
      this.data[o + 3] = k + 1;
      if (bits & PRESS) count++;
    }
    this.texture.needsUpdate = true;
    this.filledAt = head;
    this.complete = known;
  }

  aim(history: History, stream: EventStream, position: number): void {
    const head = Math.floor(position), last = Math.floor(position + 1 - CONTACT);
    let count = history.totals(stream, Math.min(last, head), this.totals).presses;
    if (last > head && history.has(last, PRESS)) count++;
    const level = count % SWEEP, age = position - contactOf(last);
    this.position.value = position;
    this.pen.value = level;
    this.pip.value = level !== 0 && history.has(last, FOOD) && age < PIP_T ? age / PIP_T : 0;
  }
}

const rect = (q: N<'vec2'>, x0: number, y0: number, x1: number, y1: number, r = 0): N<'float'> =>
  sdBox(q.sub(vec2((x0 + x1) / 2, (y0 + y1) / 2)), vec2((x1 - x0) / 2, (y1 - y0) / 2), r);
const ring = (d: N<'float'>, w: number): N<'float'> => abs(d).sub(w);
const vertical = (q: N<'vec2'>, x: N<'float'>, y0: N<'float'>, y1: N<'float'>, r: number): N<'float'> =>
  sdSegment(q, vec2(x, min(y0, y1)), vec2(x, max(y0, y1)), r);

export function recorderMaterial(lit: Lit, chart: Chart): THREE.MeshBasicNodeMaterial {
  const q = picture();
  const p = chart.position;
  const history = texture(chart.texture);
  const load = (k: N<'float'>): N<'vec4'> => history.load(uvec2(vec2(mod(max(k, 0), TEXELS), 0)));
  const known = (t: N<'vec4'>, k: N<'float'>): N<'float'> => select(t.w.equal(k.add(1)).and(k.greaterThanEqual(0)), float(1), float(0));
  const tau = p.add(q.x.sub(PEN_X).div(STEP_W));
  const written = coverage(q.x.sub(PEN_X)).mul(coverage(float(PAPER.x0).sub(q.x)));
  const inTrace = coverage(abs(q.y.sub((TRACE.y0 + TRACE.y1) / 2)).sub((TRACE.y1 - TRACE.y0) / 2 + 0.03));

  const kc = floor(tau.add(1 - CONTACT));
  const here = load(kc);
  const level = mod(here.x.add(select(here.y.greaterThan(0.5), float(1), float(0))), SWEEP);
  const flatD = abs(q.y.sub(level.mul(NOTCH).add(TRACE.y0))).sub(INK_HALF);
  let ink: N<'float'> = coverage(flatD).mul(known(here, kc));
  let pipInk: N<'float'> = float(0), costInk: N<'float'> = float(0);
  const kr = floor(tau.add(1 - CONTACT).add(0.5));
  for (const shift of [-1, 0, 1]) {
    const k = kr.add(shift), t = load(k), valid = known(t, k), kind = t.y;
    const cx = p.sub(k.sub(1 - CONTACT)).mul(-STEP_W).add(PEN_X);
    const bottom = t.x.mul(NOTCH).add(TRACE.y0), top = t.x.add(1).mul(NOTCH).add(TRACE.y0);
    const wraps = t.x.add(1).greaterThanEqual(SWEEP - 0.5);
    const riser = vertical(q, cx, select(wraps, float(TRACE.y0), bottom), select(wraps, float(TRACE.y1), top), INK_HALF);
    const pressed = select(kind.greaterThan(0.5), float(1), float(0)).mul(valid).mul(coverage(cx.sub(PEN_X).sub(0.001)));
    const isCost = select(kind.greaterThan(1.5).and(kind.lessThan(2.5)), float(1), float(0));
    const isFood = select(kind.greaterThan(0.5).and(kind.lessThan(1.5)), float(1), float(0));
    const mark = coverage(riser).mul(pressed);
    ink = max(ink, mark.mul(isCost.oneMinus()));
    costInk = max(costInk, mark.mul(isCost));
    const pipEnd = vec2(cx.add(PIP_W), select(wraps, float(TRACE.y1), top).sub(PIP_H));
    const pip = sdSegment(q, vec2(cx, select(wraps, float(TRACE.y1), top)), pipEnd, INK_HALF * 0.85);
    pipInk = max(pipInk, coverage(pip).mul(pressed).mul(isFood).mul(coverage(q.x.sub(PEN_X))));
    const tick = vertical(q, cx, float(EVENT_Y), float(EVENT_Y - EVENT_H * 0.9), INK_HALF);
    costInk = max(costInk, coverage(tick).mul(pressed).mul(isCost));
  }
  ink = ink.mul(inTrace);

  const ke = floor(tau), lit0 = load(ke), lightNow = lit0.z.mul(known(lit0, ke));
  const eventLine = abs(q.y.sub(lightNow.mul(EVENT_H).add(EVENT_Y))).sub(INK_HALF);
  const ki = floor(tau.add(0.5)), after = load(ki), before = load(ki.sub(1));
  const edgeX = p.sub(ki).mul(-STEP_W).add(PEN_X);
  const changes = abs(after.z.sub(before.z)).mul(known(after, ki)).mul(known(before, ki.sub(1)));
  const edge = vertical(q, edgeX, float(EVENT_Y), float(EVENT_Y + EVENT_H), INK_HALF);
  const eventInk = max(coverage(eventLine).mul(known(lit0, ke)), coverage(edge).mul(changes).mul(coverage(edgeX.sub(PEN_X))));

  const along = tau.div(8), major = tau.div(32);
  const vMinor = abs(fract(along.add(0.5)).sub(0.5)).mul(8 * STEP_W).sub(0.004);
  const vMajor = abs(fract(major.add(0.5)).sub(0.5)).mul(32 * STEP_W).sub(0.007);
  const notch = q.y.sub(TRACE.y0).div(NOTCH);
  const hMinor = abs(fract(notch.div(4).add(0.5)).sub(0.5)).mul(4 * NOTCH).sub(0.004);
  const hMajor = abs(fract(notch.div(12).add(0.5)).sub(0.5)).mul(12 * NOTCH).sub(0.007);
  const gridBand = coverage(abs(q.y.sub((TRACE.y0 + TRACE.y1) / 2)).sub((TRACE.y1 - TRACE.y0) / 2));
  const eventBand = coverage(abs(q.y.sub(EVENT_Y + EVENT_H / 2)).sub(EVENT_H / 2 + 0.12));
  const gridWeight = max(max(coverage(vMinor).mul(0.35), coverage(vMajor).mul(0.7)).mul(max(gridBand, eventBand)), max(coverage(hMinor).mul(0.35), coverage(hMajor).mul(0.7)).mul(gridBand));
  const border = min(abs(q.y.sub(TRACE.y0 - 0.001)), abs(q.y.sub(TRACE.y1 + 0.001))).sub(0.006);
  const fibre = fractal(q.mul(vec2(3, 11)), 2).mul(0.025).add(noise(q.mul(60)).mul(0.012));
  let paper: N<'vec3'> = rgb(0xece3cc).mul(fibre.add(1));
  paper = mix(paper, rgb(GRID), gridWeight.mul(0.8));
  paper = mix(paper, rgb(GRID).mul(0.8), coverage(border).mul(0.7));
  const toRoll = min(q.x.sub(PAPER.x0), float(PAPER.x1).sub(q.x));
  paper = paper.mul(smoothstep(0, 0.35, toRoll).mul(0.18).add(0.82));
  paper = mix(paper, rgb(LIGHT_INK), eventInk.mul(written).mul(0.95));
  paper = mix(paper, rgb(INK), ink.mul(written));
  paper = mix(paper, rgb(COST_INK), costInk.mul(written).mul(0.95));
  paper = mix(paper, rgb(PIP_INK), pipInk.mul(0.95));

  const penY = chart.pen.mul(NOTCH).add(TRACE.y0).sub(chart.pip.mul(PIP_H));
  const carriageY = chart.pen.mul(NOTCH).add(TRACE.y0);
  const shadow = smoothstep(0.16, 0, sdBox(q.sub(vec2(PEN_X + 0.26, carriageY.sub(0.05))), vec2(0.2, 0.13), 0.08)).mul(0.25);
  paper = paper.mul(shadow.oneMinus());
  const paperD = rect(q, PAPER.x0, PAPER.y0, PAPER.x1, PAPER.y1);
  const paperLit = lit(paper).mul(0.25).add(paper.mul(0.75));

  const body = rect(q, REC.x0, REC.y0, REC.x1, REC.y1, 0.3);
  const feet = min(rect(q, REC.x0 + 0.35, FOOT, REC.x0 + 0.85, REC.y0 + 0.05, 0.03), rect(q, REC.x1 - 0.85, FOOT, REC.x1 - 0.35, REC.y0 + 0.05, 0.03));
  const dimple = fractal(q.mul(7), 3);
  const bevel = smoothstep(0, 0.14, body.negate());
  const sheen = smoothstep(REC.y0, REC.y1, q.y).mul(0.2).add(0.85);
  const rim = smoothstep(0.07, 0, abs(body.add(0.05))).mul(smoothstep(-0.5, 0.8, q.y.sub(REC.y1 - 0.8).add(q.x.sub(REC.x0).mul(-0.1))));
  let color: N<'vec3'> = rgb(0x3d4b45).mul(dimple.mul(0.12).add(1)).mul(sheen).mul(mix(float(1.4), float(1), bevel)).add(rgb(0x9fb2aa).mul(rim.mul(0.18)));
  color = mix(color, rgb(0x0e100f), coverage(feet).mul(coverage(body.negate())));
  color = mix(color, rgb(0x121614), coverage(ring(body, 0.012)));
  const deck = rect(q, DECK.x0, DECK.y0, DECK.x1, DECK.y1, 0.1);
  const deckShade = smoothstep(0, 0.3, float(DECK.y1).sub(q.y)).mul(0.4).add(0.6);
  color = mix(color, rgb(0x161b19).mul(deckShade), coverage(deck));
  color = mix(color, rgb(0x5a6a63), coverage(ring(deck.sub(0.02), 0.012)));
  const plate = rect(q, 2.25, -3.02, 3.5, -2.74, 0.04);
  color = mix(color, mix(rgb(0x8a6d3a), rgb(0xd8b870), smoothstep(-3.02, -2.74, q.y)), coverage(plate));
  const engraved = min(rect(q, 2.4, -2.9, 3.35, -2.87), rect(q, 2.4, -2.84, 3.05, -2.81));
  color = mix(color, rgb(0x4a3a1c), coverage(engraved).mul(0.8));
  for (const [x, y] of [[REC.x0 + 0.22, REC.y1 - 0.22], [REC.x1 - 0.22, REC.y1 - 0.22], [REC.x0 + 0.22, REC.y0 + 0.22], [REC.x1 - 0.22, REC.y0 + 0.22]] as const) {
    const screw = sdCircle(q.sub(vec2(x, y)), 0.06);
    color = mix(color, mix(rgb(0x3a3f44), rgb(0xb9c0c6), smoothstep(-0.06, 0.06, q.y.sub(y).sub(q.x.sub(x)))), coverage(screw));
    color = mix(color, rgb(0x1a1d20), coverage(sdSegment(q, vec2(x - 0.04, y - 0.02), vec2(x + 0.04, y + 0.02), 0.009)).mul(coverage(screw)));
  }
  for (const x of [6.2, 6.95]) {
    const knob = sdCircle(q.sub(vec2(x, -2.88)), 0.17);
    color = mix(color, mix(rgb(0x0e0f10), rgb(0x3e4246), smoothstep(-0.17, 0.17, q.y.add(2.88))), coverage(knob));
    color = mix(color, rgb(0xd8d2c0), coverage(sdSegment(q, vec2(x, -2.88), vec2(x + 0.08, -2.8), 0.012)));
  }
  const pilot = sdCircle(q.sub(vec2(PILOT.x, PILOT.y)), 0.065);
  const pilotGlow = rgb(0xff8a3a).mul(chart.pilot);
  color = lit(mix(color, rgb(0x6a7076), coverage(ring(pilot, 0.016))));
  color = mix(color, rgb(0x2a1206).add(pilotGlow.mul(0.9)), coverage(pilot.add(0.016)));
  color = color.add(pilotGlow.mul(smoothstep(0.1, 0, pilot).mul(0.18)).mul(coverage(pilot).oneMinus()));

  const rollers = [[DECK.x0 + 0.02, PAPER.x0], [PAPER.x1, DECK.x1 - 0.02]] as const;
  let rollerCover: N<'float'> = float(0), rollerColor: N<'vec3'> = rgb(0x000000);
  for (const [x0, x1] of rollers) {
    const centre = (x0 + x1) / 2, half = (x1 - x0) / 2;
    const across = clamp(q.x.sub(centre).div(half), -1, 1);
    const round = sqrt(max(float(1).sub(across.mul(across)), 0));
    const drum = rect(q, x0, PAPER.y0 - 0.04, x1, PAPER.y1 + 0.04, 0.03);
    const wrapped = rgb(0xe6dcc4).mul(round.mul(0.55).add(0.35)).add(rgb(0xffffff).mul(smoothstep(0.7, 1, round).mul(0.08)));
    const striped = mix(wrapped, rgb(GRID).mul(round.mul(0.6).add(0.3)), coverage(abs(fract(q.y.sub(TRACE.y0).div(NOTCH * 4)).sub(0.5)).mul(NOTCH * 4).sub(0.004)).mul(0.5));
    rollerColor = mix(rollerColor, striped, coverage(drum));
    rollerCover = max(rollerCover, coverage(drum));
    for (const y of [PAPER.y1 + 0.02, PAPER.y0 - 0.02]) {
      const cap = rect(q, x0 - 0.02, y - 0.05, x1 + 0.02, y + 0.05, 0.03);
      rollerColor = mix(rollerColor, mix(rgb(0x3b4046), rgb(0xc9d0d6), round.mul(round)), coverage(cap));
      rollerCover = max(rollerCover, coverage(cap));
    }
  }

  const railX = PEN_X + 0.36;
  const rail = rect(q, railX - 0.022, DECK.y0 + 0.1, railX + 0.022, DECK.y1 - 0.1, 0.02);
  const railColor = mix(rgb(0x5a6168), rgb(0xe2e7eb), smoothstep(-0.022, 0.012, q.x.sub(railX)));
  const block = sdBox(q.sub(vec2(PEN_X + 0.33, carriageY)), vec2(0.17, 0.1), 0.035);
  const blockColor = mix(rgb(0x7a5a24), rgb(0xe0bf72), smoothstep(-0.1, 0.1, q.y.sub(carriageY)));
  const arm = sdSegment(q, vec2(PEN_X + 0.18, carriageY), vec2(PEN_X + 0.02, penY), 0.016);
  const nib = sdCircle(q.sub(vec2(PEN_X, penY)), 0.024);
  const eventBlockY = EVENT_Y + EVENT_H / 2;
  const eventBlock = rect(q, PEN_X + 0.16, eventBlockY - 0.09, PEN_X + 0.5, eventBlockY + 0.09, 0.03);
  const eventPenY = float(EVENT_Y).add(load(floor(p)).z.mul(known(load(floor(p)), floor(p))).mul(EVENT_H));
  const eventArm = sdSegment(q, vec2(PEN_X + 0.17, eventBlockY), vec2(PEN_X + 0.02, eventPenY), 0.013);
  const eventNib = sdCircle(q.sub(vec2(PEN_X, eventPenY)), 0.02);

  let alpha: N<'float'> = max(coverage(body), coverage(feet));
  color = mix(color, paperLit, coverage(paperD));
  color = mix(color, lit(rollerColor), rollerCover);
  color = mix(color, lit(railColor), coverage(rail));
  color = mix(color, lit(rgb(0x2a2f34)), coverage(eventArm));
  color = mix(color, lit(mix(rgb(0x3c4248), rgb(0x9aa2a9), smoothstep(-0.09, 0.09, q.y.sub(eventBlockY)))), coverage(eventBlock));
  color = mix(color, rgb(LIGHT_INK).mul(1.1), coverage(eventNib));
  color = mix(color, lit(rgb(0x22272c)), coverage(arm));
  color = mix(color, lit(blockColor), coverage(block));
  color = mix(color, rgb(0x2a1a0a), coverage(ring(block, 0.008)).mul(coverage(block)));
  color = mix(color, rgb(INK).mul(1.3), coverage(nib));
  alpha = max(alpha, coverage(rail));
  return sheet(color, alpha);
}
