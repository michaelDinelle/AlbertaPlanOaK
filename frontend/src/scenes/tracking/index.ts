import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, round, smoothstep, step, texture, uniform, uvec2, vec2, vec4 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01 } from '@kit/mix';
import { fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { trackingCall, trackingReading, trackingTruth } from '@sim/envs/tracking';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatRate, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Level = THREE.UniformNode<'float', number>;

const WINDOW = 110;
const RING = WINDOW + 8;
const LEFT = -6.9;
const RIGHT = 5.7;
const NOW = 4.6;
const TOP = 3.9;
const K = 1.5;
const ROLL = 0.42;
const DX = (NOW - LEFT) / WINDOW;
const HALF_X = 7.75;
const HALF_Y = 4.65;
const PAPER_X = (LEFT + RIGHT) / 2;
const GLOW_STEPS = 0.9;

const PAPER = 0xf2ecdc;
const GRID = 0x7fb3aa;
const INK = 0x1c2847;
const INK_TINT = 0x7d98dc;
const GRAPHITE = 0x5f636c;
const GRAPHITE_TINT = 0xb9bcc4;
const CORAL = 0xe0604e;
const ENAMEL = 0x262a31;
const STEEL = 0xa8b0ba;
const ROOM = 0x121418;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const pixel = (): N<'float'> => max(fwidth(positionWorld.x), 1e-5);
const ink = (d: N<'float'>, pw: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pw)), 0, 1);

function sheet(color: N<'vec3'>, opacity: N<'float'> | number): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = typeof opacity === 'number' ? float(opacity) : opacity;
  return material;
}

interface Paper { history: THREE.DataTexture; head: Level; alpha: Level; tick: Level }

function paperMaterial(accent: number, s: Paper): THREE.MeshBasicNodeMaterial {
  const q = chart(), pw = pixel();
  const u = s.alpha.add(q.x.sub(NOW).div(DX)), k0 = floor(u), f = u.sub(k0);
  const at = (k: N<'float'>) => texture(s.history).load(uvec2(vec2(mod(s.head.add(k).add(RING), RING), 0))).mul(vec4(K, K, K, 1));
  const live = (k: N<'float'>) => step(-WINDOW - 3, k).mul(step(k, 1.5));
  const px = (k: N<'float'>) => float(NOW).add(k.sub(s.alpha).mul(DX));
  const ks = [k0.sub(1), k0, k0.add(1), k0.add(2)];
  const cells = ks.map(at);
  let truth: N<'float'> = float(1e3), call: N<'float'> = float(1e3);
  for (let d = 0; d < 3; d++) {
    const a = cells[d]!, b = cells[d + 1]!, ka = ks[d]!, kb = ks[d + 1]!;
    const both = live(ka).mul(live(kb)), xa = px(ka), xb = px(kb);
    const seen = both.mul(step(0.5, a.w)).mul(step(0.5, b.w)), scored = both.mul(step(1.5, a.w)).mul(step(1.5, b.w));
    truth = min(truth, sdSegment(q, vec2(xa, a.x), vec2(xb, b.x), 0).add(seen.oneMinus().mul(1e3)));
    call = min(call, sdSegment(q, vec2(xa, a.z), vec2(xb, b.z), 0).add(scored.oneMinus().mul(1e3)));
  }
  truth = max(truth, q.x.sub(NOW));

  const a = cells[1]!, b = cells[2]!;
  const span = live(k0).mul(live(k0.add(1))).mul(step(1.5, a.w)).mul(step(1.5, b.w));
  const yT = mix(a.x, b.x, f), yP = mix(a.z, b.z, f);
  const gap = max(abs(q.y.sub(yT.add(yP).mul(0.5))).sub(abs(yT.sub(yP)).mul(0.5)), q.x.sub(NOW));
  const fill = ink(gap, pw).mul(span);

  const r = round(u), c = at(r), xr = px(r);
  const shown = live(r).mul(step(xr, NOW + 1e-3));
  const dot = ink(sdCircle(q.sub(vec2(xr, c.y)), 0.036), pw).mul(shown).mul(step(0.5, c.w));
  const tick = ink(sdSegment(q, vec2(xr, c.x), vec2(xr, c.z), 0).sub(0.008), pw).mul(shown).mul(step(1.5, c.w));
  const nib = ink(sdCircle(q.sub(vec2(xr, c.z)), 0.036), pw).mul(live(r)).mul(step(1.5, c.w));

  const tone = fractal(q.mul(vec2(1.2, 6)), 3).mul(0.03).add(fractal(q.mul(30), 2).mul(0.02)).add(0.99);
  let color: N<'vec3'> = rgb(PAPER).mul(tone);
  const yMinor = abs(fract(q.y.mul(2 / K).add(0.5)).sub(0.5)).mul(K / 2), yMajor = abs(fract(q.y.div(K).add(0.5)).sub(0.5)).mul(K);
  const tMinor = abs(fract(u.add(0.5)).sub(0.5)).mul(DX), tMajor = abs(fract(u.add(s.tick).div(10).add(0.5)).sub(0.5)).mul(10 * DX);
  const grid = max(max(ink(yMinor.sub(0.004), pw).mul(0.35), ink(yMajor.sub(0.008), pw).mul(0.7)), max(ink(tMinor.sub(0.003), pw).mul(0.14), ink(tMajor.sub(0.007), pw).mul(0.6)));
  color = mix(color, rgb(GRID), grid);
  color = mix(color, rgb(GRID).mul(0.62), ink(abs(q.y).sub(0.014), pw).mul(0.9));
  const ahead = smoothstep(NOW, NOW + 0.5, q.x);
  color = color.mul(ahead.mul(0.05).add(1)).mul(smoothstep(NOW + 0.05, NOW + 0.45, q.x).mul(0.14).add(0.86).max(step(q.x, NOW)));
  color = mix(color, rgb(CORAL), fill.mul(0.26));
  color = mix(color, rgb(CORAL).mul(0.85), tick.mul(0.5));
  color = mix(color, rgb(GRAPHITE), dot.mul(0.85));
  color = mix(color, rgb(accent).mul(0.5), ink(call.sub(0.016), pw));
  color = mix(color, rgb(accent).mul(0.42), nib);
  color = mix(color, rgb(INK), ink(truth.sub(0.022), pw));

  const roll = (centre: number) => {
    const t = clamp(abs(q.x.sub(centre)).div(ROLL), 0, 1);
    return rgb(PAPER).mul(float(1).sub(t.mul(t)).max(0).sqrt().mul(0.55).add(0.4));
  };
  color = mix(color, roll(LEFT - ROLL), step(q.x, LEFT));
  color = mix(color, roll(RIGHT + ROLL), step(RIGHT, q.x));
  const edge = min(abs(q.x.sub(LEFT)), abs(q.x.sub(RIGHT)));
  color = mix(color, rgb(0x8a8472), ink(edge.sub(0.01), pw).mul(0.6));
  color = color.mul(smoothstep(TOP, TOP - 0.18, abs(q.y)).mul(0.12).add(0.88));
  return sheet(color, 1);
}

function housingMaterial(): THREE.MeshBasicNodeMaterial {
  const q = chart(), pw = pixel();
  const tone = fractal(q.mul(1.6), 3).mul(0.08).add(fractal(q.mul(24), 2).mul(0.03)).add(0.97);
  const window = sdBox(q.sub(vec2(PAPER_X, 0)), vec2((RIGHT - LEFT) / 2 + ROLL + 0.08, TOP + 0.08), 0.12);
  const light = smoothstep(-HALF_Y, HALF_Y, q.y.sub(q.x.mul(0.3))).mul(0.25).add(0.85);
  let color: N<'vec3'> = rgb(ENAMEL).mul(tone).mul(light);
  const recess = smoothstep(0.35, 0, window);
  color = color.mul(recess.mul(-0.45).add(1));
  color = mix(color, rgb(STEEL).mul(0.55), ink(abs(window.sub(0.05)).sub(0.03), pw).mul(0.8));
  const outer = sdBox(q, vec2(HALF_X - 0.05, HALF_Y - 0.05), 0.3);
  color = mix(color, rgb(0x0c0e11), ink(abs(outer).sub(0.03), pw).mul(0.7));
  let screws: N<'float'> = float(0);
  for (const [x, y] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    screws = max(screws, ink(sdCircle(q.sub(vec2(x * (HALF_X - 0.3), y * (HALF_Y - 0.28))), 0.08), pw));
  }
  color = mix(color, rgb(STEEL).mul(0.8), screws);
  return sheet(color, ink(outer, pw));
}

interface Carriage { stylus: Level; pen: Level; penOn: Level; alpha: Level; glow: Level }

function carriageMaterial(accent: number, s: Carriage): THREE.MeshBasicNodeMaterial {
  const q = chart(), pw = pixel();
  const rail = sdBox(q.sub(vec2(NOW, 0)), vec2(0.028, TOP + 0.34), 0.02);
  const shadow = ink(sdBox(q.sub(vec2(NOW + 0.05, -0.05)), vec2(0.05, TOP + 0.3), 0.04).sub(0.04), pw.mul(8)).mul(0.25);
  const gloss = smoothstep(NOW - 0.03, NOW + 0.03, q.x);
  let color: N<'vec3'> = mix(rgb(STEEL), rgb(0x5d6570), gloss);
  let cover = ink(rail, pw);
  const blocks = sdBox(vec2(q.x.sub(NOW), abs(q.y).sub(TOP + 0.3)), vec2(0.13, 0.11), 0.04);
  color = mix(color, rgb(0x3a4049), ink(blocks, pw));
  cover = max(cover, ink(blocks, pw));

  const tipX = float(NOW).add(s.alpha.oneMinus().mul(DX)), tip = vec2(tipX, s.pen);
  const arm = sdSegment(q, vec2(NOW, s.pen), tip, 0.02);
  const pen = sdCircle(q.sub(tip), 0.11);
  const penCover = max(ink(arm.sub(0.012), pw), ink(pen.sub(0.014), pw)).mul(s.penOn);
  color = mix(color, rgb(0x15171c), penCover);
  color = mix(color, rgb(accent).mul(0.9), max(ink(arm, pw), ink(pen, pw)).mul(s.penOn));
  color = mix(color, rgb(accent).mul(0.35), ink(sdCircle(q.sub(tip), 0.04), pw).mul(s.penOn));
  cover = max(cover, penCover);

  const at = vec2(NOW, s.stylus), rad = length(q.sub(at));
  const body = sdCircle(q.sub(at), 0.12);
  const halo = ink(abs(rad.sub(0.19)).sub(0.03), pw).mul(s.glow);
  color = mix(color, rgb(TOKENS.celebrationGold).mul(1.1), halo);
  color = mix(color, rgb(INK), ink(body, pw));
  color = mix(color, rgb(0xe8ecf2), ink(abs(rad.sub(0.08)).sub(0.016), pw));
  color = color.add(rgb(TOKENS.celebrationGold).mul(exp(rad.negate().div(0.12)).mul(s.glow).mul(0.5)));
  cover = max(max(cover, ink(body, pw)), halo);
  return sheet(color, max(cover, shadow));
}

class TrackingScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'tracking';
  override legend: LegendSpec | null = null;

  private eta = 0;
  private innovation = 0;
  private variance = 1;
  private readonly texels = new Float32Array(RING * 4);
  private readonly tags = new Float64Array(RING).fill(NaN);
  private readonly history = new THREE.DataTexture(this.texels, RING, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly head = uniform(0);
  private readonly decade = uniform(0);
  private readonly alpha = uniform(0);
  private readonly stylus = uniform(0);
  private readonly pen = uniform(0);
  private readonly penOn = uniform(0);
  private readonly glow = uniform(0);
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, truth: NaN, reading: NaN, call: NaN, miss: NaN, reward: NaN, rms: NaN, explained: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.eta = parameter(ctx.environment, 'eta');
    this.innovation = parameter(ctx.environment, 'innovation');
    this.variance = (this.innovation * this.innovation) / (1 - this.eta * this.eta);
    return new THREE.Box2(new THREE.Vector2(-HALF_X, -HALF_Y), new THREE.Vector2(HALF_X, HALF_Y));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    const accent = this.accentOf(ctx);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(ROOM);
    pool.group.renderOrder = 10;
    this.history.magFilter = this.history.minFilter = THREE.NearestFilter;
    this.history.needsUpdate = true;
    const place = (mesh: THREE.Mesh, x: number, order: number) => {
      mesh.position.x = x;
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      return mesh;
    };
    const housing = place(new THREE.Mesh(flat(2 * HALF_X, 2 * HALF_Y), housingMaterial()), 0, 0);
    const paper = place(new THREE.Mesh(flat(RIGHT - LEFT + 2 * ROLL, 2 * TOP), paperMaterial(accent, { history: this.history, head: this.head, alpha: this.alpha, tick: this.decade })), PAPER_X, 1);
    const carriage = place(new THREE.Mesh(flat(1.0, 2 * TOP + 1.0), carriageMaterial(accent, { stylus: this.stylus, pen: this.pen, penOn: this.penOn, alpha: this.alpha, glow: this.glow })), NOW + 0.2, 2);
    this.scene.add(housing, paper, carriage);
  }

  protected override drop(): void {
    this.history.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private at(frame: FrameUpdate, s: number): EnvFrame | null {
    if (s < 0) return null;
    const found = frame.frameAt(s);
    return found.step === s ? found : null;
  }

  private slot(s: number): number {
    return ((s % RING) + RING) % RING;
  }

  private sync(frame: FrameUpdate, i: number): void {
    let dirty = false;
    for (let s = i - WINDOW - 3; s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] === s) continue;
      const f = this.at(frame, s);
      if (!f) {
        if (this.texels[o + 3] !== 0) dirty = true;
        this.texels.fill(0, o, o + 4);
        this.tags[slot] = s < 0 ? s : NaN;
        continue;
      }
      const truth = trackingTruth(f), call = trackingCall(f);
      if (s > 0) {
        const want = -(call - truth) * (call - truth);
        if (Math.abs(f.reward - want) > 1e-4 + 1e-3 * Math.abs(want)) throw new Error(`Tracking step ${s} scored ${f.reward} for prediction ${call} of signal ${truth}`);
      }
      this.texels[o] = truth;
      this.texels[o + 1] = trackingReading(f);
      this.texels[o + 2] = call;
      this.texels[o + 3] = s > 0 ? 2 : 1;
      this.tags[slot] = s;
      dirty = true;
    }
    if (dirty) this.history.needsUpdate = true;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const i = frame.state.step, now = frame.frameAt(i);
    this.sync(frame, i);
    const next = this.at(frame, i + 1), alpha = next ? clamp01(frame.position - i) : 0;
    const truth = trackingTruth(now);
    this.head.value = this.slot(i);
    this.decade.value = i % 10;
    this.alpha.value = alpha;
    this.stylus.value = K * (next ? truth + (trackingTruth(next) - truth) * alpha : truth);
    this.penOn.value = next ? 1 : 0;
    if (next) this.pen.value = K * trackingCall(next);
    const hit = lastEvent(frame.eventLog, 'goal', i), age = hit === undefined ? Infinity : frame.position - hit;
    this.glow.value = age >= 0 ? Math.exp(-age / GLOW_STEPS) : 0;
    if (!frame.scrubbed) {
      for (const event of frame.events) if (event.type === 'goal') pool.ripple(this.point.set(NOW, 0.05, -this.stylus.value), TOKENS.celebrationGold, 0.25, time);
    } else pool.clear();
    this.tally(i, now, next);
  }

  private tally(i: number, now: EnvFrame, next: EnvFrame | null): void {
    const l = this.ledger, truth = trackingTruth(now);
    l.step = i;
    l.truth = truth;
    l.reading = trackingReading(now);
    l.call = next ? trackingCall(next) : NaN;
    l.miss = i > 0 ? trackingCall(now) - truth : NaN;
    l.reward = i > 0 ? now.reward : NaN;
    let sum = 0, n = 0;
    for (let s = i - WINDOW + 1; s <= i; s++) {
      const o = this.slot(s) * 4;
      if (this.tags[this.slot(s)] !== s || this.texels[o + 3] !== 2) continue;
      const miss = this.texels[o + 2]! - this.texels[o]!;
      sum += miss * miss;
      n++;
    }
    l.rms = n > 0 ? Math.sqrt(sum / n) : NaN;
    l.explained = n > 0 ? clamp01(1 - sum / n / this.variance) : 0;
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger;
    return {
      title: 'TRACKING',
      rows: [
        { key: 'signal', label: 'hidden signal', tint: INK_TINT, value: () => formatSigned(l.truth) },
        { key: 'reading', label: 'noisy reading', tint: GRAPHITE_TINT, value: () => formatSigned(l.reading) },
        { key: 'call', label: 'prediction for next', tint: accent, value: () => formatSigned(l.call) },
        { key: 'miss', label: 'last miss', tint: CORAL, value: () => formatSigned(l.miss) },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => formatRate(l.reward) },
        { key: 'rms', label: `rms miss, last ${WINDOW}`, tint: CORAL, value: () => formatRate(l.rms) },
      ],
      note: () => `eta ${formatRate(this.eta)} · innovation ${formatRate(this.innovation)} · oracle rms ${formatRate(this.innovation)}`,
      bar: () => l.explained,
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const i = frame.state.step, alpha = this.alpha.value;
    const x = (s: number) => NOW + (s - i - alpha) * DX;
    const signal = kit.curve('signal', WINDOW + 2, TOKENS.uiCyan);
    const calls = kit.marks('calls', WINDOW + 2), oracle = kit.marks('oracle', WINDOW + 2);
    let prev = NaN;
    for (let s = i - WINDOW + 1; s <= i + 1; s++) {
      const o = this.slot(s) * 4;
      if (this.tags[this.slot(s)] !== s) {
        prev = NaN;
        continue;
      }
      const t = this.texels[o]!, xs = x(s);
      if (s <= i) signal.point(xs, 0.06, -K * t);
      if (this.texels[o + 3] === 2) calls.set(xs, 0.06, -K * this.texels[o + 2]!, 0.07, 0xffffff);
      if (Number.isFinite(prev)) oracle.set(xs, 0.06, -K * this.eta * prev, 0.05, 0xffb44a);
      prev = t;
    }
    signal.point(NOW, 0.06, -this.stylus.value);
    const rail = kit.curve('now', 2, TOKENS.uiCyan);
    rail.point(NOW, 0.06, -TOP);
    rail.point(NOW, 0.06, TOP);
  }
}

export default (() => new TrackingScene()) satisfies SceneFactory;
