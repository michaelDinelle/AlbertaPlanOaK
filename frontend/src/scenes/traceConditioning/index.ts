import * as THREE from 'three/webgpu';
import { abs, ceil, clamp, exp, float, fract, fwidth, length, max, min, mix, mod, positionWorld, round, smoothstep, step, texture, uniform, vec2 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { noise as tslNoise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type F = N<'float'>;
type V2 = N<'vec2'>;
type V3 = N<'vec3'>;
type V4 = N<'vec4'>;
type Level = THREE.UniformNode<'float', number>;

const WALL_A = 0x0f141c;
const WALL_B = 0x1b2230;
const PAPER = 0xcfc5ab;
const GRID = 0xb3a78b;
const RULE = 0x6f6652;
const TONE = 0x3a74d8;
const TONE_GLOW = 0x86b6ff;
const ARC_OFF = 0x2a3244;
const PUFF = 0x1f9486;
const MIST = 0xcfe7ec;
const TRACE = 0xffb44a;
const FUSE = 0x3b3629;
const CORAL = 0xff6b5a;
const SCLERA = 0xd2ccbf;
const IRIS = 0x4c7f6c;
const IRIS_RIM = 0x21392f;
const PUPIL = 0x090b0e;
const SKIN = 0x7b6e86;
const LINE = 0x1f1925;
const METAL = 0x8a95a5;
const DARK_METAL = 0x363d48;
const RUBBER = 0x17191d;
const CASE = 0x2c3340;

const X0 = -7.6;
const X1 = 7.6;
const TOP = -0.25;
const BOTTOM = -4.35;
const TRACKS = [-1.45, -2.75, -4.05] as const;
const AMP = 0.72;
const LW = 0.026;
const gridOf = (interval: number): number => Math.max(10, Math.round(interval / 2));
const EYE_Y = 2.35;
const EYE_HALF = 1.3;
const SAG = 0.62;
const EYE_R = (EYE_HALF * EYE_HALF + SAG * SAG) / (2 * SAG);
const SPEAKER_X = -5.0;
const BODY_X = -5.75;
const TIP_X = 4.04;
const STUB_X = 5.15;
const FUSE_Y = 0.9;
const FAST = 0.3;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

interface Knobs {
  head: Level;
  alpha: Level;
  phase: Level;
  tone: Level;
  puff: Level;
  lid: Level;
  hit: Level;
  miss: Level;
  spark: Level;
  sparkOn: Level;
  due: Level;
  dueOn: Level;
  penTone: Level;
  penPuff: Level;
  penBlink: Level;
  clock: Level;
}

interface Layout {
  ring: number;
  colW: number;
  nowX: number;
  interval: number;
  blink: number;
}

function stageMaterial(history: THREE.DataTexture, layout: Layout, k: Knobs): THREE.MeshBasicNodeMaterial {
  const { ring, colW, nowX, interval } = layout;
  const q = vec2(positionWorld.x, positionWorld.z.negate());
  const px = fwidth(positionWorld.x);
  const ink = (d: F): F => clamp(float(0.5).sub(d.div(px)), 0, 1);
  const over = (base: V3, top: V3, amount: F): V3 => mix(base, top, amount);

  let color: V3 = mix(rgb(WALL_A), rgb(WALL_B), smoothstep(-0.5, 4.5, q.y));

  const hose = sdSegment(q, vec2(6.2, EYE_Y), vec2(9.5, EYE_Y), 0.1);
  color = over(color, rgb(RUBBER), ink(hose));

  const np = q.sub(vec2(0, EYE_Y));
  const span = TIP_X - EYE_HALF;
  const t = clamp(float(TIP_X).sub(q.x).div(span), 0, 1);
  const half = t.mul(0.36).add(0.07);
  const inside = clamp(float(1).sub(abs(np.y).div(half)), 0, 1);
  const streak = tslNoise(vec2(q.x.mul(2.6).add(k.clock.mul(7)), np.y.mul(6))).mul(0.5).add(0.6);
  const reach = step(q.x, TIP_X).mul(smoothstep(EYE_HALF - 0.1, EYE_HALF + 0.35, q.x));
  color = over(color, rgb(MIST), clamp(inside.mul(inside).mul(streak).mul(reach).mul(k.puff).mul(0.85), 0, 1));

  const shade = mix(rgb(DARK_METAL), rgb(METAL), smoothstep(-0.22, 0.22, np.y));
  const tube = sdBox(np.sub(vec2(5.4, 0)), vec2(0.85, 0.2), 0.06);
  const collar = sdBox(np.sub(vec2(4.62, 0)), vec2(0.07, 0.27), 0.03);
  const tip = sdBox(np.sub(vec2(4.3, 0)), vec2(0.26, 0.1), 0.03);
  const nozzle = min(min(tube, collar), tip);
  color = over(color, rgb(LINE), ink(nozzle.sub(0.025)));
  color = over(color, shade, ink(nozzle));
  color = over(color, rgb(MIST), ink(sdCircle(np.sub(vec2(TIP_X + 0.02, 0)), 0.05)).mul(k.puff));

  const sp = q.sub(vec2(SPEAKER_X, EYE_Y));
  const body = sdBox(sp.sub(vec2(BODY_X - SPEAKER_X, 0)), vec2(0.28, 0.42), 0.06);
  const cone = max(max(sp.x.negate().sub(0.47), sp.x), abs(sp.y).sub(sp.x.add(0.47).mul(0.95).add(0.2)).mul(0.725));
  color = over(color, rgb(LINE), ink(min(body, cone).sub(0.025)));
  color = over(color, rgb(CASE), ink(body));
  color = over(color, mix(rgb(DARK_METAL), rgb(TONE), k.tone.mul(0.7)), ink(cone));
  const radius = length(sp), aim = abs(sp.y).mul(1.1).sub(sp.x).mul(0.67);
  for (const [r, lag] of [[0.45, 0], [0.75, 0.15], [1.05, 0.3]] as const) {
    const lit = clamp(k.tone.sub(lag).div(1 - lag), 0, 1);
    const arc = max(abs(radius.sub(r)).sub(0.045), aim);
    color = over(color, mix(rgb(ARC_OFF), rgb(TONE_GLOW), lit), ink(arc));
  }

  const fuseA = vec2(BODY_X, FUSE_Y), fuseB = vec2(STUB_X, FUSE_Y);
  const wire = min(min(sdSegment(q, vec2(BODY_X, EYE_Y - 0.42), fuseA, 0.022), sdSegment(q, fuseA, fuseB, 0.022)), sdSegment(q, fuseB, vec2(STUB_X, EYE_Y - 0.2), 0.022));
  const fuseLen = STUB_X - BODY_X, sparkX = k.spark.mul(fuseLen).add(BODY_X);
  const burnt = step(q.x, sparkX).mul(k.sparkOn).mul(step(q.y, FUSE_Y + 0.2));
  color = over(color, mix(rgb(FUSE), rgb(TRACE).mul(0.6), burnt), ink(wire));
  const seg = fuseLen / (interval + 1);
  const dot = Math.min(0.05, seg * 0.24);
  const j = clamp(round(q.x.sub(BODY_X).div(seg)), 0, interval + 1);
  const tick = length(q.sub(vec2(j.mul(seg).add(BODY_X), FUSE_Y))).sub(dot);
  const passed = step(j.mul(seg).add(BODY_X), sparkX.add(1e-3)).mul(k.sparkOn);
  color = over(color, mix(rgb(0x5a5240), rgb(TRACE), passed), ink(tick));
  const sd = length(q.sub(vec2(sparkX, FUSE_Y)));
  color = color.add(rgb(TRACE).mul(exp(sd.mul(-5.5)).mul(0.55).mul(k.sparkOn)));
  color = over(color, rgb(0xffe0a6), ink(sd.sub(0.09)).mul(k.sparkOn));

  const e = q.sub(vec2(0, EYE_Y));
  const upper = sdCircle(e.sub(vec2(0, SAG - EYE_R)), EYE_R);
  const lower = sdCircle(e.sub(vec2(0, EYE_R - SAG)), EYE_R);
  const almond = max(upper, lower);
  const margin = sdCircle(e.sub(vec2(0, k.lid.mul(-2 * SAG).add(SAG - EYE_R))), EYE_R);
  const open = max(almond, margin);
  color = color.add(rgb(TOKENS.celebrationGold).mul(exp(max(almond, 0).mul(-4)).mul(k.hit).mul(0.6)));
  const brow = max(abs(sdCircle(e.sub(vec2(0, -1.25)), 2.35)).sub(0.06), abs(e.x).sub(1.15));
  color = over(color, rgb(SKIN).mul(0.8), ink(brow));
  const skin = mix(rgb(SKIN), rgb(CORAL), k.miss.mul(0.65));
  color = over(color, skin, ink(almond.sub(0.05)));
  let ball: V3 = mix(rgb(SCLERA).mul(0.72), rgb(SCLERA), smoothstep(0, 0.35, margin.negate()));
  const iris = sdCircle(e, 0.52);
  ball = over(ball, rgb(IRIS_RIM), ink(iris));
  ball = over(ball, mix(rgb(IRIS), rgb(IRIS).mul(1.35), smoothstep(0.1, 0.45, length(e))), ink(iris.add(0.06)));
  ball = over(ball, rgb(PUPIL), ink(sdCircle(e, 0.23)));
  ball = over(ball, rgb(0xd6dde4), ink(sdCircle(e.sub(vec2(-0.16, 0.17)), 0.075)));
  color = over(color, ball, ink(open));
  color = over(color, rgb(LINE), ink(max(abs(margin).sub(0.04), almond.sub(0.02))));
  color = over(color, rgb(LINE), ink(abs(almond.sub(0.02)).sub(0.03)));

  const mid = (TOP + BOTTOM) / 2;
  const panel = sdBox(q.sub(vec2(0, mid)), vec2((X1 - X0) / 2, (TOP - BOTTOM) / 2), 0.12);
  let paper: V3 = rgb(PAPER);
  const u = k.alpha.add(q.x.sub(nowX).div(colW));
  const column = (n: F): V4 => texture(history, vec2(mod(k.head.add(n).add(ring * 3), ring).add(0.5).div(ring), 0.5));
  const cur = column(ceil(u)), m = round(u), left = column(m), right = column(m.add(1));
  const fade = smoothstep(X0 + 0.1, X0 + 0.9, q.x);
  const live = step(q.x, nowX).mul(step(0.5, cur.w)).mul(fade);
  const edgeLive = step(m.add(1e-3), k.alpha).mul(step(0.5, left.w)).mul(step(0.5, right.w)).mul(fade);
  const dxe = abs(u.sub(m)).mul(colW);
  const hitOf = (v: V4): F => clamp(float(1).sub(abs(v.w.sub(2))), 0, 1);
  const missOf = (v: V4): F => clamp(v.w.sub(2), 0, 1);

  const every = gridOf(interval);
  const grid = abs(fract(u.add(k.phase).div(every).add(0.5)).sub(0.5)).mul(every * colW);
  paper = over(paper, rgb(GRID), ink(grid.sub(0.01)));
  paper = over(paper, rgb(TOKENS.celebrationGold), hitOf(cur).mul(live).mul(0.55));
  paper = over(paper, rgb(RULE), ink(abs(q.x.sub(nowX)).sub(0.012)));

  const dueL = max(k.due.sub(1).sub(k.alpha).mul(colW).add(nowX), nowX);
  const dueR = max(k.due.sub(k.alpha).mul(colW).add(nowX), dueL);
  const ghost = sdBox(q.sub(vec2(dueL.add(dueR).mul(0.5), TRACKS[1] + AMP / 2)), vec2(dueR.sub(dueL).mul(0.5), AMP / 2));
  const ghostOn = k.dueOn.mul(step(nowX + 1e-4, dueR));
  paper = over(paper, rgb(PUFF), ink(ghost).mul(ghostOn).mul(0.16));
  paper = over(paper, rgb(PUFF), ink(abs(ghost).sub(LW * 0.7)).mul(ghostOn).mul(0.7));

  const blinkInkOf = (v: V4): V3 => mix(mix(rgb(layout.blink).mul(0.75), rgb(0xc2861a), hitOf(v)), rgb(0xd9483a), missOf(v));
  const blinkInk = blinkInkOf(cur);
  const blinkEdge = mix(blinkInkOf(right), blinkInkOf(left), left.z);
  const pens = [k.penTone, k.penPuff, k.penBlink];
  const channels = [(v: V4): F => v.x, (v: V4): F => v.y, (v: V4): F => v.z];
  const inks: V3[] = [rgb(TONE), rgb(PUFF), blinkInk];
  TRACKS.forEach((base, n) => {
    const pick = channels[n]!, tint = inks[n]!;
    const level = pick(cur).mul(live), yl = level.mul(AMP).add(base);
    paper = over(paper, rgb(GRID), ink(abs(q.y.sub(base)).sub(0.01)));
    const tab = sdBox(q.sub(vec2(X0 + 0.22, base + AMP / 2)), vec2(0.06, AMP / 2), 0.05);
    paper = over(paper, n === 2 ? rgb(layout.blink).mul(0.75) : tint, ink(tab));
    const area = ink(max(float(base).sub(q.y), q.y.sub(yl))).mul(live).mul(0.22);
    paper = over(paper, tint, area);
    const top = abs(q.y.sub(yl)).sub(LW).add(float(1).sub(live).mul(10));
    const jump = abs(pick(left).sub(pick(right))).mul(edgeLive);
    const edge = max(dxe.sub(LW), abs(q.y.sub(base + AMP / 2)).sub(AMP / 2)).add(float(1).sub(jump).mul(10));
    paper = over(paper, tint, ink(top));
    paper = over(paper, n === 2 ? blinkEdge : tint, ink(edge));
    const pen = length(q.sub(vec2(nowX, pens[n]!.mul(AMP).add(base))));
    paper = over(paper, rgb(RULE), ink(pen.sub(0.085)));
    paper = over(paper, tint, ink(pen.sub(0.06)));
  });
  paper = over(paper, rgb(RULE), ink(abs(panel).sub(0.018)));
  color = over(color, rgb(0x080a0e), ink(panel.sub(0.06)).mul(0.6));
  color = over(color, paper, ink(panel));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  return material;
}

class TraceConditioningScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.35, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'traceConditioning';
  override legend: LegendSpec | null = null;

  private interval = 1;
  private window = 1;
  private ring = 1;
  private colW = 0.1;
  private nowX = 0;
  private catch = 1;
  private wasted = 0;
  private texels = new Float32Array(4);
  private tags = new Float64Array(1);
  private history = new THREE.DataTexture(this.texels, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly k: Knobs = {
    head: uniform(0), alpha: uniform(0), phase: uniform(0), tone: uniform(0), puff: uniform(0), lid: uniform(0), hit: uniform(0), miss: uniform(0),
    spark: uniform(0), sparkOn: uniform(0), due: uniform(0), dueOn: uniform(0), penTone: uniform(0), penPuff: uniform(0), penBlink: uniform(0), clock: uniform(0),
  };
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, tone: 0, puff: 0, blink: 0, countdown: 0, reward: 0, seen: 0, puffs: 0, hits: 0, wasted: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.interval = parameter(env, 'interval');
    this.catch = constant(env, 'CONDITIONING_BLINK_REWARD');
    this.wasted = constant(env, 'CONDITIONING_FALSE_BLINK');
    this.window = Math.max(120, 3 * (this.interval + 1));
    this.ring = this.window + this.interval + 8;
    this.colW = (X1 - X0 - 0.2) / (this.window + this.interval + 2);
    this.nowX = X1 - 0.1 - (this.interval + 2) * this.colW;
    this.texels = new Float32Array(this.ring * 4);
    this.tags = new Float64Array(this.ring).fill(NaN);
    this.history = new THREE.DataTexture(this.texels, this.ring, 1, THREE.RGBAFormat, THREE.FloatType);
    return new THREE.Box2(new THREE.Vector2(-8, -3.75), new THREE.Vector2(8, 4.6));
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
    this.scene.background = new THREE.Color(WALL_A);
    pool.group.renderOrder = 10;
    this.history.magFilter = this.history.minFilter = THREE.NearestFilter;
    this.history.needsUpdate = true;
    const layout: Layout = { ring: this.ring, colW: this.colW, nowX: this.nowX, interval: this.interval, blink: accent };
    const stage = new THREE.Mesh(flat(60, 40), stageMaterial(this.history, layout, this.k));
    stage.frustumCulled = false;
    this.scene.add(stage);
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
    return ((s % this.ring) + this.ring) % this.ring;
  }

  private check(s: number, before: EnvFrame, now: EnvFrame): void {
    const blink = blinkOf(now), puff = now.words[2] !== 0, tone = now.words[1] !== 0;
    const expected = blink ? (puff ? this.catch : this.wasted) : 0;
    if (Math.abs(now.reward - expected) > 1e-5 || puff !== (before.words[0] === 1) || (tone && now.words[0] !== this.interval + 1)) {
      throw new Error(`TraceConditioning step ${s}: countdown ${before.words[0]} to ${now.words[0]}, tone ${now.words[1]}, puff ${now.words[2]}, blink ${blink}, reward ${now.reward}`);
    }
  }

  private sync(frame: FrameUpdate, i: number): void {
    let dirty = false;
    for (let s = i - this.window; s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] === s) continue;
      const now = this.at(frame, s);
      if (!now) {
        if (this.texels[o + 3] !== 0) dirty = true;
        this.texels.fill(0, o, o + 4);
        this.tags[slot] = s < 0 ? s : NaN;
        continue;
      }
      const before = now.reset ? null : this.at(frame, s - 1);
      if (before) this.check(s, before, now);
      this.texels[o] = now.words[1] !== 0 ? 1 : 0;
      this.texels[o + 1] = now.words[2] !== 0 ? 1 : 0;
      this.texels[o + 2] = blinkOf(now);
      this.texels[o + 3] = now.reward > 0 ? 2 : now.reward < 0 ? 3 : 1;
      this.tags[slot] = s;
      dirty = true;
    }
    if (dirty) this.history.needsUpdate = true;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const i = frame.state.step, now = frame.frameAt(i), k = this.k, span = this.interval + 1;
    this.sync(frame, i);
    const next = this.at(frame, i + 1), alpha = next ? clamp01(frame.position - i) : 0, w = next ? smoothstep01(alpha / FAST) : 0;
    const later = next ?? now, blend = (f: (e: EnvFrame) => number) => f(now) + (f(later) - f(now)) * w;
    k.head.value = this.slot(i);
    k.phase.value = i % gridOf(this.interval);
    k.alpha.value = alpha;
    k.clock.value = time;
    k.tone.value = blend(toneOf);
    k.puff.value = blend(puffOf);
    k.lid.value = blend(blinkOf);
    k.hit.value = blend((e) => (e.reward > 0 ? 1 : 0));
    k.miss.value = blend((e) => (e.reward < 0 ? 1 : 0));
    const writing = alpha > 0 ? later : now;
    k.penTone.value = toneOf(writing);
    k.penPuff.value = puffOf(writing);
    k.penBlink.value = blinkOf(writing);
    const c = now.words[0]!, fresh = next !== null && next.words[0] === span;
    if (c > 0) {
      k.spark.value = (span - c + alpha) / span;
      k.sparkOn.value = 1;
      k.due.value = c;
      k.dueOn.value = 1;
    } else {
      k.spark.value = 0;
      k.sparkOn.value = fresh ? w : 0;
      k.due.value = span + 1;
      k.dueOn.value = fresh ? w : 0;
    }
    this.celebrate(frame, time);
    this.tally(i, now);
  }

  private celebrate(frame: FrameUpdate, time: number): void {
    const pool = this.pool;
    if (!pool) return;
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    for (const event of frame.events) {
      if (event.type !== 'goal') continue;
      pool.ripple(this.point.set(0, 0, -EYE_Y), TOKENS.celebrationGold, 0.6, time);
      pool.burst(this.point.set(0, 40, -EYE_Y), TOKENS.celebrationGold, 1.2, time);
    }
  }

  private tally(i: number, now: EnvFrame): void {
    const l = this.ledger;
    l.step = i;
    l.tone = toneOf(now);
    l.puff = puffOf(now);
    l.blink = blinkOf(now);
    l.countdown = now.words[0]!;
    l.reward = now.reward;
    let seen = 0, puffs = 0, hits = 0, wasted = 0;
    for (let s = Math.max(0, i - this.window + 1); s <= i; s++) {
      const o = this.slot(s) * 4;
      if (this.tags[this.slot(s)] !== s) continue;
      seen++;
      puffs += this.texels[o + 1]!;
      if (this.texels[o + 3] === 2) hits++;
      if (this.texels[o + 3] === 3) wasted++;
    }
    Object.assign(l, { seen, puffs, hits, wasted });
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, span = formatCount(this.window);
    return {
      title: 'TRACE CONDITIONING',
      rows: [
        { key: 'tone', label: 'tone', tint: TONE_GLOW, value: () => (l.tone ? 'sounding' : 'silent') },
        { key: 'puff', label: 'air puff', tint: PUFF, value: () => (l.puff ? 'now' : l.countdown > 0 ? `in ${formatCount(l.countdown)} steps` : 'none pending') },
        { key: 'blink', label: 'eyelid', tint: accent, value: () => (l.blink ? 'blink' : 'open') },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
        { key: 'caught', label: `puffs blinked, last ${span}`, tint: TOKENS.celebrationGold, value: () => (l.puffs > 0 ? `${l.hits} of ${l.puffs}` : '—') },
        { key: 'wasted', label: `false blinks, last ${span}`, tint: CORAL, value: () => (l.seen > 0 ? formatCount(l.wasted) : '—') },
      ],
      note: () => `tone, ${this.interval} silent steps, then the puff`,
      bar: () => (l.puffs > 0 ? l.hits / l.puffs : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const i = frame.state.step, now = frame.frameAt(i), a = this.k.alpha.value, c = now.words[0]!, size = Math.min(0.09, this.colW * 0.6);
    const marks = [kit.marks('tones', this.window + 1), kit.marks('puffs', this.window + 1), kit.marks('blinks', this.window + 1)];
    const tints = [0xffffff, 0xffffff, 0xffffff];
    for (let s = Math.max(0, i - this.window + 1); s <= i; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] !== s) continue;
      const x = this.nowX - (i + a - (s - 0.5)) * this.colW;
      if (x < X0 + 0.5) continue;
      for (let n = 0; n < 3; n++) {
        if (this.texels[o + n] === 1) marks[n]!.set(x, 0.06, -(TRACKS[n]! + AMP + 0.12), size, tints[n]!);
      }
    }
    if (c > 0) kit.marks('due', 1).set(this.nowX + (c - 0.5 - a) * this.colW, 0.06, -(TRACKS[1] + AMP / 2), 0.12, TRACE);
    const span = this.interval + 1;
    if (c > 0) kit.marks('spark', 1).set(BODY_X + ((span - c + a) / span) * (STUB_X - BODY_X), 0.06, -FUSE_Y, 0.16, TRACE);
    kit.marks('lid', 1).set(0, 0.06, -(EYE_Y + SAG - 2 * SAG * blinkOf(now)), 0.12, TOKENS.uiCyan);
    const line = kit.curve('now', 2, TOKENS.uiCyan);
    line.point(this.nowX, 0.06, -TOP);
    line.point(this.nowX, 0.06, -BOTTOM);
  }
}

const toneOf = (f: EnvFrame): number => (f.words[1] !== 0 ? 1 : 0);
const puffOf = (f: EnvFrame): number => (f.words[2] !== 0 ? 1 : 0);
const blinkOf = (f: EnvFrame): number => Math.trunc(f.actions[0]!) & 1;

export default (() => new TraceConditioningScene()) satisfies SceneFactory;
