import * as THREE from 'three/webgpu';
import { Fn, abs, clamp, exp2, float, floor, fwidth, length, max, mix, mod, positionWorld, select, smoothstep, step, texture, uniform, vec2, vec4 } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { subsetBits, subsetCall, subsetMask, subsetParityOf, subsetTruth } from '@sim/envs/subsetparity';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';

type N<T extends string> = THREE.Node<T>;

const ROWS = 10;
const PITCH = 1.4;
const H = 0.56;
const HOLD = 0.35;
const WINDOW = 1000;
const FLASH = 0.25;
const ABSENT = 4;

const BG_TOP = 0x1d2b33;
const BG_BOTTOM = 0x121a20;
const PAPER = 0xefe4c6;
const AGED = 0xcdbf9c;
const INK = 0x4a4234;
const HOLE = 0x0d1418;
const LENS = 0x36d6c3;
const BRASS = 0xd2a95a;
const RIGHT = 0x6fd48a;
const WRONG = 0xe0565c;

class SubsetParityScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.25, fog: { from: ['night', 'hemiSky'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'subsetParity';
  override legend: LegendSpec | null = null;

  private inputs = 8;
  private subset = 1;
  private readonly cardData = new Float32Array((ROWS + 1) * 4);
  private readonly cards = new THREE.DataTexture(this.cardData, ROWS + 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly slide = uniform(0);
  private readonly mask = uniform(0);
  private readonly flash = uniform(0);
  private readonly hits = new Uint8Array(WINDOW);
  private readonly tags = new Float64Array(WINDOW).fill(NaN);
  private readonly point = new THREE.Vector3();
  private filled = -1;
  private filledNext = -1;
  private logged = -1;
  private landed = -Infinity;
  private shown = -1;
  private maskBits = 0;
  private readonly ledger = { bits: 0, call: -1, truth: -1, reward: 0, seen: 0, right: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.inputs = parameter(ctx.environment, 'inputs');
    this.subset = parameter(ctx.environment, 'subset');
    return new THREE.Box2(new THREE.Vector2(-0.9, -H - 0.8), new THREE.Vector2(this.inputs + 4.4, (ROWS - 1) * PITCH + H + 0.25));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 18, sighSize: 12, sighRadius: 0.3, sighHeight: 0.8 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  private stampX(): number { return this.inputs + 2.95; }
  private truthX(): number { return this.inputs + 0.85; }
  private callX(): number { return this.inputs + 1.75; }
  private cardZ(t: number): number { return (t - 1 + this.slide.value) * PITCH; }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(BG_BOTTOM);
    light.setEnabled(false);
    pool.group.renderOrder = 10;
    this.cards.magFilter = this.cards.minFilter = THREE.NearestFilter;
    this.cards.needsUpdate = true;
    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = this.shader(accent);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(400, 400).rotateX(-Math.PI / 2), material);
    plane.frustumCulled = false;
    plane.renderOrder = 0;
    this.scene.add(plane);
  }

  private shader(accent: number): N<'vec4'> {
    const n = this.inputs, slide = this.slide, mask = this.mask, flash = this.flash, cards = this.cards;
    const cx = (n + 3.5) / 2, halfW = (n + 4.1) / 2, span = (ROWS - 1) * PITCH;
    return Fn(() => {
      const p = positionWorld.xz;
      const aa = max(fwidth(p.x), float(1e-5));
      const cov = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(aa)), 0, 1);
      const u = p.y.div(PITCH).sub(slide).add(1);
      const tRaw = floor(u.add(0.5));
      const t = clamp(tRaw, 0, ROWS);
      const dz = u.sub(tRaw).mul(PITCH);
      const row = texture(cards, vec2(t.add(0.5).div(ROWS + 1), 0.5));
      const inRange = step(-0.5, tRaw).mul(step(tRaw, ROWS + 0.5));
      const present = step(row.a, 3.5).mul(inRange);
      const fade = select(t.lessThan(0.5), slide, select(t.greaterThan(ROWS - 0.5), float(1).sub(slide), float(1)));

      const b = floor(p.x);
      const inCols = step(0, p.x).mul(step(p.x, n));
      const lx = p.x.sub(b).sub(0.5);
      const masked = mod(floor(mask.mul(exp2(b.negate()))), 2).mul(inCols);
      const bit = mod(floor(row.r.mul(exp2(b.negate()))), 2).mul(inCols);

      let col: N<'vec3'> = mix(rgb(BG_TOP), rgb(BG_BOTTOM), clamp(p.y.div(span), 0, 1));
      const stack = cov(abs(p.y.sub(span / 2)).sub(span / 2 + H + 0.3));
      col = mix(col, rgb(LENS), cov(abs(lx).sub(0.44)).mul(masked).mul(stack).mul(0.16));
      col = mix(col, rgb(LENS), cov(length(vec2(lx, p.y.add(H + 0.45))).sub(0.13)).mul(masked));
      const reader = sdBox(vec2(p.x.sub(cx), p.y), vec2(halfW + 0.16, H + 0.16), 0.22);
      col = mix(col, rgb(BRASS), cov(abs(reader).sub(0.035)));

      const shadow = float(1).sub(smoothstep(-0.05, 0.3, sdBox(vec2(p.x.sub(cx + 0.07), dz.sub(0.11)), vec2(halfW, H), 0.14)));
      col = col.mul(float(1).sub(shadow.mul(0.5).mul(present).mul(fade)));

      const age = clamp(t.sub(1).div(ROWS), 0, 1);
      let card: N<'vec3'> = mix(rgb(PAPER), rgb(AGED), age.mul(0.7));
      card = mix(card, rgb(LENS), cov(abs(lx).sub(0.46)).mul(masked).mul(cov(abs(dz).sub(H - 0.04))).mul(0.22));
      const hole = sdBox(vec2(lx, dz), vec2(0.2, 0.33), 0.11);
      card = mix(card, rgb(INK), cov(abs(hole).sub(0.018)).mul(float(1).sub(bit)).mul(inCols).mul(mix(float(0.28), float(0.8), masked)));
      const punched = select(masked.greaterThan(0.5), rgb(LENS).mul(1.1), rgb(HOLE));
      card = mix(card, punched, cov(hole).mul(bit));
      card = mix(card, rgb(INK), cov(abs(p.x.sub(n + 0.2)).sub(0.015)).mul(step(abs(dz), H - 0.12)).mul(0.5));

      const known = step(-0.5, row.g);
      const dTruth = length(vec2(p.x.sub(n + 0.85), dz)).sub(0.25);
      const truthFill = select(row.b.greaterThan(0.5), cov(dTruth), cov(abs(dTruth.add(0.03)).sub(0.035)));
      card = mix(card, rgb(LENS).mul(0.8), truthFill.mul(known));
      const dCall = sdBox(vec2(p.x.sub(n + 1.75), dz), vec2(0.24, 0.24), 0.06);
      const callFill = select(row.g.greaterThan(0.5), cov(dCall), cov(abs(dCall.add(0.03)).sub(0.035)));
      card = mix(card, mix(rgb(accent), rgb(0x1a2560), 0.55), callFill.mul(known));
      const pendingRing = cov(abs(dTruth.add(0.03)).sub(0.012)).add(cov(abs(dCall.add(0.03)).sub(0.012)));
      card = mix(card, rgb(INK), clamp(pendingRing, 0, 1).mul(float(1).sub(known)).mul(0.35));

      const sp = vec2(p.x.sub(n + 2.95), dz);
      const dStamp = length(sp).sub(0.36);
      const right = step(0.5, row.a).mul(step(row.a, 1.5));
      const wrong = step(1.5, row.a).mul(step(row.a, 2.5));
      const glow = float(1).add(select(t.equal(2), flash, float(0)).mul(0.9));
      card = mix(card, rgb(RIGHT).mul(glow), cov(dStamp).mul(right));
      card = mix(card, rgb(PAPER), cov(sdSegment(sp, vec2(-0.16, 0.0), vec2(-0.04, 0.13), 0.045).min(sdSegment(sp, vec2(-0.04, 0.13), vec2(0.18, -0.14), 0.045))).mul(right));
      const cross = sdSegment(sp, vec2(-0.2, -0.2), vec2(0.2, 0.2), 0.06).min(sdSegment(sp, vec2(-0.2, 0.2), vec2(0.2, -0.2), 0.06));
      card = mix(card, rgb(WRONG).mul(glow), cov(abs(dStamp).sub(0.035)).max(cov(cross)).mul(wrong));
      card = mix(card, rgb(INK), cov(abs(dStamp.add(0.02)).sub(0.012)).mul(step(row.a, 0.5)).mul(0.3));

      const cardCov = cov(sdBox(vec2(p.x.sub(cx), dz), vec2(halfW, H), 0.14)).mul(present).mul(fade);
      return vec4(mix(col, card, cardCov), 1);
    })() as unknown as N<'vec4'>;
  }

  protected override drop(): void {
    this.cards.dispose();
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

  private check(judged: EnvFrame, judge: EnvFrame): void {
    const truth = subsetTruth(judge), want = subsetParityOf(this.maskBits, subsetBits(judged));
    if ((judge.reward !== 0 && judge.reward !== 1) || subsetMask(judge) !== this.maskBits || subsetMask(judged) !== this.maskBits || truth !== want) {
      throw new Error(`SubsetParity step ${judge.step} paid ${judge.reward} for calling ${subsetCall(judge)} on bits ${subsetBits(judged).toString(2)} under mask ${this.maskBits.toString(2)}`);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0;
    this.slide.value = smoothstep01((f - HOLD) / (1 - HOLD));
    this.maskBits = subsetMask(now);
    this.mask.value = this.maskBits;
    if (k !== this.shown) {
      this.landed = !frame.scrubbed && k === this.shown + 1 ? time : -Infinity;
      this.shown = k;
    }
    this.flash.value = Math.exp(-(time - this.landed) / FLASH);
    this.fill(frame, now, next);
    this.sync(frame, k);
    const prev = this.at(frame, k - 1);
    Object.assign(this.ledger, { bits: subsetBits(now), call: k > 0 ? subsetCall(now) : -1, truth: k > 0 && prev ? subsetTruth(now) : -1, reward: now.reward });
    this.celebrate(frame, pool, time);
  }

  private fill(frame: FrameUpdate, now: EnvFrame, next: EnvFrame | null): void {
    const nextStep = next ? next.step : -1;
    if (this.filled === now.step && this.filledNext === nextStep) return;
    const data = this.cardData, k = now.step;
    const put = (t: number, bits: number, call: number, truth: number, state: number) => {
      data[t * 4] = bits;
      data[t * 4 + 1] = call;
      data[t * 4 + 2] = truth;
      data[t * 4 + 3] = state;
    };
    if (next) put(0, subsetBits(next), -1, -1, 0);
    else put(0, 0, -1, -1, ABSENT);
    put(1, subsetBits(now), -1, -1, 0);
    for (let t = 2; t <= ROWS; t++) {
      const s = k - t + 1, judged = this.at(frame, s), judge = judged ? this.at(frame, s + 1) : null;
      if (!judged) put(t, 0, -1, -1, ABSENT);
      else if (!judge) put(t, subsetBits(judged), -1, -1, 0);
      else {
        this.check(judged, judge);
        put(t, subsetBits(judged), subsetCall(judge), subsetTruth(judge), judge.reward > 0 ? 1 : 2);
      }
    }
    this.cards.needsUpdate = true;
    this.filled = now.step;
    this.filledNext = nextStep;
  }

  private sync(frame: FrameUpdate, k: number): void {
    let fresh = false;
    for (let s = Math.max(1, k - WINDOW + 1); s <= k; s++) {
      const slot = s % WINDOW;
      if (this.tags[slot] === s) continue;
      const f = this.at(frame, s);
      if (!f) continue;
      if (f.reward !== 0 && f.reward !== 1) throw new Error(`SubsetParity step ${s} paid ${f.reward}`);
      this.hits[slot] = f.reward > 0 ? 1 : 0;
      this.tags[slot] = s;
      fresh = true;
    }
    if (!fresh && this.logged === k) return;
    let seen = 0, right = 0;
    for (let s = Math.max(1, k - WINDOW + 1); s <= k; s++) {
      const slot = s % WINDOW;
      if (this.tags[slot] !== s) continue;
      seen++;
      right += this.hits[slot]!;
    }
    this.ledger.seen = seen;
    this.ledger.right = right;
    this.logged = k;
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    const x = this.stampX(), z = PITCH;
    for (const event of frame.events) {
      if (event.type === 'goal') pool.burst(this.point.set(x, 40, z), RIGHT, 0.3, time);
      else if (event.type === 'harm') pool.ripple(this.point.set(x, 0.1, z), WRONG, 0.35, time);
    }
  }

  private columns(): string {
    const cols: number[] = [];
    for (let b = 0; b < this.inputs; b++) if ((this.maskBits >>> b) & 1) cols.push(b);
    return cols.length ? cols.join(' & ') : '—';
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, n = this.inputs;
    const shown = (v: number) => (v < 0 ? '—' : String(v));
    return {
      title: 'SUBSET PARITY',
      rows: [
        { key: 'bits', label: 'input bits', tint: INK, value: () => Array.from({ length: n }, (_, b) => (l.bits >>> b) & 1).join('') },
        { key: 'subset', label: this.subset === 1 ? 'hidden bit' : 'hidden bits', tint: LENS, value: () => this.columns() },
        { key: 'call', label: 'call', tint: accent, value: () => shown(l.call) },
        { key: 'truth', label: 'subset parity', tint: LENS, value: () => shown(l.truth) },
        { key: 'reward', label: 'reward', tint: RIGHT, value: () => (l.call < 0 ? '—' : l.reward > 0 ? '+1' : '0') },
        { key: 'right', label: `right, last ${formatCount(WINDOW)}`, tint: RIGHT, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      ],
      note: () => `${formatCount(n)} inputs · parity of ${formatCount(this.subset)} · chance 50%`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const n = this.inputs, mask = subsetMask(now), next = this.at(frame, k + 1);
    const holes = kit.marks('holes', (ROWS + 1) * n), lens = kit.marks('mask', n);
    const parity = kit.marks('parity', ROWS + 1), verdict = kit.marks('verdict', ROWS + 1);
    for (let b = 0; b < n; b++) if ((mask >>> b) & 1) lens.set(b + 0.5, 0.08, -H - 0.45, 0.18, LENS);
    for (let t = 0; t <= ROWS; t++) {
      const s = k - t + 1, f = t === 0 ? (this.slide.value > 0.01 ? next : null) : t === ROWS && this.slide.value > 0.99 ? null : this.at(frame, s);
      if (!f) continue;
      const z = this.cardZ(t), bits = subsetBits(f);
      for (let b = 0; b < n; b++) {
        if (!((bits >>> b) & 1)) continue;
        const inMask = (mask >>> b) & 1;
        holes.set(b + 0.5, 0.08, z, inMask ? 0.14 : 0.07, inMask ? LENS : 0xffffff);
      }
      const p = subsetParityOf(mask, bits);
      parity.set(this.truthX(), 0.08, z, p ? 0.16 : 0.06, LENS);
      const judge = t >= 2 ? this.at(frame, s + 1) : null;
      if (judge) verdict.set(this.stampX(), 0.08, z, 0.2, judge.reward > 0 ? RIGHT : WRONG);
    }
    kit.marks('call', 1).set(this.callX(), 0.08, this.cardZ(2), k > 0 && subsetCall(now) ? 0.16 : 0.06, this.accentOf(this.ctx!));
  }
}

export default (() => new SubsetParityScene()) satisfies SceneFactory;
