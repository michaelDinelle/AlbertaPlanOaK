import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { repeatCall, repeatSymbol } from '@sim/envs/repeat';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';
import { ACCENT, CHIP_QUAD, GLYPHS, HIT, IVORY, MISS, QUAD, SUITS, TRUTH, cardMaterial, chipMaterial, disc, flat, lampMaterial, lampSize, shoeMaterial, tableMaterial } from './art';
import { CARD, CHIP, RECORD, SCALE, layoutFor, placeX, recordX, type Layout } from './layout';

type Span = readonly [number, number];

const CALL: Span = [0.28, 0.44];
const FILE: Span = [0.3, 0.62];
const DROP: Span = [0.5, 0.94];
const SLIDE: Span = [0.56, 0.9];
const DEAL: Span = [0.64, 0.98];
const VERDICT: Span = [0.9, 1];
const DUE_OUT: Span = [0.5, 0.62];
const DUE_IN: Span = [0.88, 1];
const SEEN_IN: Span = [0.86, 1];
const WINDOW = 1000;
const CACHE = 64;
const FLASH = 0.45;
const BLINK = 0.28;
const MEMORY_SELF = 0.2;
const NEWEST_SELF = 0.55;
const TRAY_SELF = 1;
const HIT_SELF = 0.62;
const MISS_SELF = 0.42;
const TRAY_DIM = 1;
const RECORD_GOLD = 0.8;
const LENS_LIGHT = 22;
const TRAY_LIGHT = 1.4;
const NO_CALL = -1;

const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const ease = (t: number, s: Span) => smoothstep01(span(t, s));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const named = (suit: number) => `${GLYPHS[suit]} ${SUITS[suit]}`;

interface Card { x: number; y: number; size: number; flip: number; suit: number; side: number; alpha: number; self: number; halo: number; due: number; verdict: number; ring: number }
interface Chip { x: number; y: number; size: number; suit: number; alpha: number; gold: number; dim: number }
interface Batch { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[] }

const blankCard = (): Card => ({ x: 0, y: 0, size: 0, flip: 1, suit: 0, side: 1, alpha: 0, self: 0, halo: 0, due: 0, verdict: 0, ring: 0 });
const blankChip = (): Chip => ({ x: 0, y: 0, size: 0, suit: 0, alpha: 0, gold: 0, dim: 0 });

class RepeatScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'repeat';
  override legend: LegendSpec | null = null;

  private layout: Layout = layoutFor(1);
  private symbols = 4;
  private cards: Batch | null = null;
  private chips: Batch | null = null;
  private states: Card[] = [];
  private readonly calls: Chip[] = [blankChip(), blankChip()];
  private readonly lens = uniform(1);
  private readonly lensLight = uniform(LENS_LIGHT);
  private readonly trayLight = uniform(0);
  private readonly truths = new Int8Array(CACHE).fill(-1);
  private readonly hits = new Uint8Array(CACHE);
  private readonly tags = new Float64Array(CACHE).fill(NaN);
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly point = new THREE.Vector3();
  private shown = -1;
  private flashAt = -Infinity;
  private seenAt = -Infinity;
  private readonly rated = { step: -1, count: -1 };
  private readonly ledger = { shown: 0, due: 0, unseen: false, call: NO_CALL, truth: 0, reward: 0, seen: 0, right: 0 };

  private get memory(): number {
    return 1 + this.layout.record.count;
  }

  private get incoming(): number {
    return this.memory + this.layout.places;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.layout = layoutFor(parameter(ctx.environment, 'lag'));
    this.symbols = constant(ctx.environment, 'REPEAT_SYMBOLS');
    return this.layout.extent.clone();
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa8b2cc).multiplyScalar(0.5);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 26, sighSize: 12, sighRadius: 0.45, sighHeight: 0.9 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const layout = this.layout, lit = light.lit;
    this.legend = this.legendFor();
    pool.group.renderOrder = 10;
    const place = <T extends THREE.Object3D>(object: T, order: number, x = 0, z = 0, y = 0): T => {
      object.renderOrder = order;
      object.frustumCulled = false;
      object.position.set(x, y, z);
      this.scene.add(object);
      return object;
    };
    place(new THREE.Mesh(flat(80, 80), tableMaterial(layout, lit)), 0);
    this.states = Array.from({ length: this.incoming + 1 }, blankCard);
    this.cards = this.batch(flat(QUAD.x, QUAD.y), cardMaterial(lit), this.states.length, ['aCard', 'aGlow'], 3);
    this.chips = this.batch(flat(CHIP * CHIP_QUAD, CHIP * CHIP_QUAD), chipMaterial(lit), this.calls.length, ['aChip'], 4);
    place(new THREE.Mesh(flat(layout.shoeSize.x + 0.2, layout.shoeSize.y + 0.2), shoeMaterial(layout, lit)), 5, layout.shoe.x, layout.shoe.y);
    const lamp = lampSize(layout);
    place(new THREE.Mesh(flat(lamp.x, lamp.y), lampMaterial(layout, this.lens, lit)), 6, layout.lamp.x, layout.lamp.y);
    place(light.emitter(disc(0.2), rgb(ACCENT).mul(this.lensLight)), 6, layout.lamp.x, layout.lamp.y, 0.02);
    place(light.emitter(disc(0.55), rgb(HIT).mul(this.trayLight)), 6, layout.due, layout.tray.y, 0.02);
  }

  private batch(geometry: THREE.BufferGeometry, material: THREE.Material, count: number, names: readonly string[], order: number): Batch {
    const attributes = names.map((name) => {
      const attribute = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attribute);
      return attribute;
    });
    const mesh = instancedMesh(geometry, material, count);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return { mesh, attributes };
  }

  protected override drop(): void {
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

  private check(f: EnvFrame): void {
    if (f.step < 1) return;
    const lag = this.layout.lag, call = repeatCall(f), truth = repeatSymbol(f, lag + 1), shown = repeatSymbol(f, 0);
    if (f.reward !== (call === truth ? 1 : 0) || f.observations[shown] !== 1 || f.observations[this.symbols] !== 1) {
      throw new Error(`Repeat step ${f.step} paid ${f.reward} for calling ${SUITS[call]} while the card due ${lag} back was ${SUITS[truth]}`);
    }
  }

  private judged(frame: FrameUpdate, t: number): number {
    const slot = t % CACHE;
    if (this.tags[slot] === t) return this.truths[slot]!;
    const f = this.at(frame, t);
    if (!f) return -1;
    this.check(f);
    this.truths[slot] = repeatSymbol(f, this.layout.lag + 1);
    this.hits[slot] = f.reward > 0 ? 1 : 0;
    this.tags[slot] = t;
    return this.truths[slot]!;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0;
    this.check(now);
    if (next) this.check(next);
    if (k !== this.shown) {
      this.seenAt = !frame.scrubbed && k === this.shown + 1 ? time : -Infinity;
      this.shown = k;
    }
    this.celebrate(frame, pool, time);
    this.deal(frame, now, next, f, k);
    this.call(now, next, f, k);
    this.commit();
    const hit = k >= 1 && now.reward > 0 ? 1 : 0;
    this.trayLight.value = TRAY_LIGHT * hit * (0.35 + 1.4 * Math.exp(-(time - this.flashAt) / FLASH)) * (1 - ease(f, FILE));
    this.lens.value = 1 + 0.6 * Math.exp(-(time - this.seenAt) / BLINK);
    this.lensLight.value = LENS_LIGHT * this.lens.value;
    this.rate(frame, k);
    Object.assign(this.ledger, { shown: repeatSymbol(now, 0), due: repeatSymbol(now, this.layout.lag), unseen: this.layout.lag > k, call: k >= 1 ? repeatCall(now) : NO_CALL, truth: repeatSymbol(now, this.layout.lag + 1), reward: now.reward });
  }

  private deal(frame: FrameUpdate, now: EnvFrame, next: EnvFrame | null, f: number, k: number): void {
    const layout = this.layout, lag = layout.lag, places = layout.places, states = this.states;
    const filed = next ? ease(f, FILE) : 0, slide = next ? ease(f, SLIDE) : 0;

    const tray = states[0]!;
    Object.assign(tray, blankCard());
    if (k >= 1) {
      const hit = now.reward > 0;
      Object.assign(tray, {
        x: lerp(layout.due, recordX(layout, 0), filed), y: lerp(layout.tray.y, layout.record.y, filed), size: lerp(SCALE, SCALE * RECORD, filed),
        suit: repeatSymbol(now, lag + 1), side: 1, alpha: 1, self: lerp(TRAY_SELF, hit ? HIT_SELF : MISS_SELF, filed),
        verdict: hit ? lerp(1, RECORD_GOLD, filed) : lerp(-TRAY_DIM, -1, filed), ring: hit ? 1 - filed : 0,
      });
    }

    const history = layout.record.count;
    for (let r = 0; r < history; r++) {
      const card = states[1 + r]!, t = k - 1 - r;
      Object.assign(card, blankCard());
      const suit = t >= 1 ? this.judged(frame, t) : -1;
      if (suit < 0) continue;
      const hit = this.hits[t % CACHE] === 1;
      Object.assign(card, {
        x: lerp(recordX(layout, r), recordX(layout, r + 1), filed), y: layout.record.y, size: SCALE * RECORD, suit, side: 1,
        alpha: r === history - 1 ? 1 - filed : 1, self: hit ? HIT_SELF : MISS_SELF, verdict: hit ? RECORD_GOLD : -1,
      });
    }

    for (let s = 0; s < places; s++) {
      const card = states[this.memory + s]!, age = lag - s, newest = s === places - 1;
      Object.assign(card, blankCard(), {
        x: placeX(layout, s), y: layout.y, size: SCALE, suit: repeatSymbol(now, age), side: age <= k ? 1 : 0, alpha: 1,
        self: newest ? NEWEST_SELF : MEMORY_SELF, halo: newest ? 1 : 0, due: s === 0 ? 1 : 0,
      });
      if (!next) continue;
      if (s === 0) {
        const d = ease(f, DROP), turn = card.side ? 1 : span(f, DROP), shown = ease(f, VERDICT), hit = next.reward > 0;
        Object.assign(card, {
          x: lerp(placeX(layout, 0), layout.due, d), y: lerp(layout.y, layout.tray.y, d), flip: Math.abs(Math.cos(Math.PI * turn)), side: turn > 0.5 ? 1 : 0,
          self: lerp(MEMORY_SELF, TRAY_SELF, d), due: 1 - ease(f, DUE_OUT), verdict: (hit ? 1 : -TRAY_DIM) * shown, ring: hit ? shown : 0,
        });
        continue;
      }
      card.x = lerp(placeX(layout, s), placeX(layout, s - 1), slide);
      if (s === 1) card.due = ease(f, DUE_IN);
      if (newest) {
        card.halo = 1 - slide;
        card.self = lerp(NEWEST_SELF, MEMORY_SELF, slide);
      }
    }

    const dealt = states[this.incoming]!;
    Object.assign(dealt, blankCard());
    if (next && f >= DEAL[0]) {
      const d = ease(f, DEAL);
      Object.assign(dealt, {
        x: lerp(layout.shoe.x, placeX(layout, places - 1), d), y: layout.y, size: lerp(SCALE * RECORD, SCALE, d), suit: repeatSymbol(next, 0), side: 1,
        alpha: 1, self: lerp(MEMORY_SELF, NEWEST_SELF, d), halo: ease(f, SEEN_IN),
      });
    }
  }

  private call(now: EnvFrame, next: EnvFrame | null, f: number, k: number): void {
    const layout = this.layout, [current, incoming] = this.calls as [Chip, Chip], shift = next ? ease(f, CALL) : 0;
    Object.assign(current, blankChip());
    if (k >= 1) {
      const hit = now.reward > 0 ? 1 : 0;
      Object.assign(current, { x: layout.chip.x, y: layout.chip.y, size: 1 - 0.3 * shift, suit: repeatCall(now), alpha: 1 - shift, gold: hit, dim: 1 - hit });
    }
    Object.assign(incoming, blankChip());
    if (next) {
      const hit = next.reward > 0 ? 1 : 0, verdict = ease(f, VERDICT), pop = span(f, CALL);
      Object.assign(incoming, { x: layout.chip.x, y: layout.chip.y, size: 0.72 + 0.28 * smoothstep01(pop) + 0.12 * Math.sin(Math.PI * pop), suit: repeatCall(next), alpha: smoothstep01(pop), gold: hit * verdict, dim: (1 - hit) * verdict });
    }
  }

  private commit(): void {
    const cards = this.cards, chips = this.chips;
    if (!cards || !chips) return;
    const [face, look] = cards.attributes as [THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute];
    this.states.forEach((card, i) => {
      this.position.set(card.x, 0.01, card.y);
      this.scale.set(card.size * Math.max(card.flip, 1e-3), 1, card.size);
      cards.mesh.setMatrixAt(i, card.alpha > 0 ? this.matrix.compose(this.position, this.quaternion, this.scale) : this.matrix.makeScale(0, 0, 0));
      face.setXYZW(i, card.suit, card.side, card.alpha, card.self);
      look.setXYZW(i, card.halo, card.due, card.verdict, card.ring);
    });
    const [state] = chips.attributes as [THREE.InstancedBufferAttribute];
    this.calls.forEach((chip, i) => {
      this.position.set(chip.x, 0.02, chip.y);
      chips.mesh.setMatrixAt(i, chip.alpha > 0 ? this.matrix.compose(this.position, this.quaternion, this.scale.setScalar(chip.size)) : this.matrix.makeScale(0, 0, 0));
      state.setXYZW(i, chip.suit, chip.alpha, chip.gold, chip.dim);
    });
    for (const batch of [cards, chips]) {
      batch.mesh.instanceMatrix.needsUpdate = true;
      for (const attribute of batch.attributes) attribute.needsUpdate = true;
    }
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
      return;
    }
    const x = this.layout.due, z = this.layout.tray.y, last = frame.events.at(-1);
    if (last?.type === 'goal') {
      this.flashAt = time;
      pool.burst(this.point.set(x, 42, z), HIT, 0.45, time);
    } else if (last?.type === 'harm') pool.sigh(0.35, this.point.set(x, 47.4, z), time);
  }

  private rate(frame: FrameUpdate, k: number): void {
    const log = frame.eventLog;
    if (k === this.rated.step && log.count === this.rated.count) return;
    const right = countEvents(log, 'goal', k - WINDOW, k), wrong = countEvents(log, 'harm', k - WINDOW, k);
    this.ledger.right = right;
    this.ledger.seen = right + wrong;
    this.rated.step = k;
    this.rated.count = log.count;
  }

  private legendFor(): LegendSpec {
    const l = this.ledger, lag = this.layout.lag;
    return {
      title: 'REPEAT',
      rows: [
        { key: 'shown', label: 'newest card', tint: ACCENT, value: () => named(l.shown) },
        { key: 'due', label: 'due next', tint: ACCENT, value: () => (l.unseen ? `${named(l.due)} · never shown` : named(l.due)) },
        { key: 'call', label: 'call', tint: ACCENT, value: () => (l.call === NO_CALL ? '—' : named(l.call)) },
        { key: 'was', label: 'was', tint: IVORY, value: () => (l.call === NO_CALL ? '—' : named(l.truth)) },
        { key: 'reward', label: 'reward', tint: HIT, value: () => (l.call === NO_CALL ? '—' : l.reward > 0 ? '+1' : '0') },
        { key: 'right', label: `right, last ${formatCount(WINDOW)}`, tint: HIT, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      ],
      note: () => `${formatCount(lag)}-step delay · chance ${formatPercent(1 / this.symbols, 0)}`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const layout = this.layout, states = this.states, lift = 0.08;
    const index = (card: Card, out: THREE.Vector3) => out.set(card.x - 0.33 * card.size * card.flip, lift, card.y - 0.24 * card.size);
    const suits = kit.marks('suits', states.length);
    for (const card of states) if (card.alpha > 0.01) {
      index(card, this.point);
      suits.set(this.point.x, lift, this.point.z, 0.1 * (card.size / SCALE), TRUTH[card.suit]!);
    }
    const verdicts = kit.marks('verdicts', this.memory);
    for (const card of states.slice(0, this.memory)) if (card.alpha > 0.01) {
      verdicts.set(card.x, lift, card.y + CARD.y * card.size / 2 + 0.16, 0.07, card.verdict > 0 ? HIT : MISS);
    }
    const chips = kit.marks('calls', this.calls.length);
    for (const chip of this.calls) if (chip.alpha > 0.01) chips.set(chip.x, lift, chip.y + CHIP * 0.66, 0.1, TRUTH[chip.suit]!);
    const w = CARD.x * SCALE / 2 + 0.1, h = CARD.y * SCALE / 2 + 0.1;
    const box = (name: string, x: number, z: number, color: number) => {
      const curve = kit.curve(name, 5, color);
      for (const [dx, dz] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) curve.point(x + dx * w, lift, z + dz * h);
    };
    box('seen', placeX(layout, layout.places - 1), layout.y, ACCENT);
    box('due', placeX(layout, 0), layout.y, TOKENS.cueMagenta);
    const ruler = kit.curve('lag', 3 * layout.places, TOKENS.uiCyan), base = layout.y + h + 0.12;
    for (let s = 0; s < layout.places; s++) {
      const x = placeX(layout, s);
      ruler.point(x, lift, base);
      ruler.point(x, lift, base + (s === 0 || s === layout.places - 1 ? 0.3 : 0.14));
      ruler.point(x, lift, base);
    }
    const judge = kit.curve('judge', 2, TOKENS.cueMagenta);
    judge.point(placeX(layout, 0), lift, layout.y);
    judge.point(layout.due, lift, layout.tray.y);
  }
}

export default (() => new RepeatScene()) satisfies SceneFactory;
