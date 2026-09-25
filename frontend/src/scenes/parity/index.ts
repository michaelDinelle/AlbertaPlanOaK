import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { parityBit, parityCall, parityTruth } from '@sim/envs/parity';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';
import {
  BENCH, HIT, MISS, SIGNAL, boardMaterial, chipMaterial, chipSpecs, disc, flat, ledMaterial, lineGeometry, lineLight, lineMaterial, logBounds, logMaterial,
  padMaterial, resistorMaterial, verdictLight, verdictMaterial, wireGeometry, wireLight, wireMaterial, type Signal,
} from './art';
import { BAND, LAMP, LED, LOG, PAD, RESISTOR, TRACE, along, layoutFor, pathArcs, type Layout, type Point, type Station } from './layout';

const HOLD = 0.4;
const WINDOW = 1000;
const FLASH = 0.22;
const COOL = 0.45;
const TAIL = 0.3;
const LINE_LIGHT = 0.45;
const WIRE_LIGHT = 0.5;
const LED_LIGHT = 2.2;
const LAMP_LIGHT = 2;
const PAD_LIGHT = 1.6;
const HEAT_LIGHT = 1.6;

type Wire = 'feed' | 'probe' | 'output' | 'stubA' | 'stubB' | 'truth' | 'judged' | 'called' | 'verdict';

const signal = (): Signal => ({ from: uniform(0), to: uniform(0), front: uniform(0) });
const hold = (s: Signal, value: number): void => { s.from.value = value; s.to.value = value; s.front.value = 0; };
const travel = (s: Signal, from: number, to: number, front: number): void => { s.from.value = from; s.to.value = to; s.front.value = front; };

class ParityScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.32, fog: { from: ['dusk', 'hemiSky'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'parity';
  override legend: LegendSpec | null = null;

  private layout: Layout = layoutFor(4);
  private bitsData = new Float32Array(4);
  private bits = new THREE.DataTexture(this.bitsData, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly logData = new Float32Array(LOG * 4);
  private readonly verdicts = new THREE.DataTexture(this.logData, LOG, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly offset = uniform(0);
  private readonly shift = uniform(0);
  private readonly padGlow = uniform(0);
  private readonly heat = uniform(0);
  private readonly agentGlow = uniform(0);
  private readonly callLevel = uniform(0);
  private readonly truthLevel = uniform(0);
  private readonly verdict = { hit: uniform(0), shown: uniform(0), flash: uniform(0) };
  private readonly wires: Record<Wire, Signal> = { feed: signal(), probe: signal(), output: signal(), stubA: signal(), stubB: signal(), truth: signal(), judged: signal(), called: signal(), verdict: signal() };
  private readonly hits = new Uint8Array(WINDOW);
  private readonly tags = new Float64Array(WINDOW).fill(NaN);
  private readonly point = new THREE.Vector3();
  private readonly station: Station = { x: 0, z: 0, tx: 1, tz: 0 };
  private filled = -1;
  private filledNext = -1;
  private logged = -1;
  private landed = -Infinity;
  private shown = -1;
  private readonly ledger = { step: 0, bit: 0, call: -1, truth: 0, reward: 0, seen: 0, right: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.layout = layoutFor(parameter(ctx.environment, 'delay'));
    this.bitsData = new Float32Array((this.layout.cells + 1) * 4);
    this.bits = new THREE.DataTexture(this.bitsData, this.layout.cells + 1, 1, THREE.RGBAFormat, THREE.FloatType);
    return this.layout.board.clone();
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb3a6d6).multiplyScalar(0.42);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 20, sighSize: 14, sighRadius: 0.32, sighHeight: 0.9 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const layout = this.layout, lit = light.lit, accent = this.accentOf(ctx);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(BENCH);
    pool.group.renderOrder = 10;
    for (const texture of [this.bits, this.verdicts]) {
      texture.magFilter = texture.minFilter = THREE.NearestFilter;
      texture.needsUpdate = true;
    }
    const place = <T extends THREE.Object3D>(object: T, order: number, x = 0, z = 0, y = 0): T => {
      object.renderOrder = order;
      object.frustumCulled = false;
      object.position.set(x, y, z);
      this.scene.add(object);
      return object;
    };
    const centre = layout.board.getCenter(new THREE.Vector2());
    place(new THREE.Mesh(flat(400, 400), boardMaterial(layout, lit)), 0, centre.x, centre.y);

    const line = { offset: this.offset, bits: this.bits };
    place(new THREE.Mesh(lineGeometry(layout, BAND, TAIL), lineMaterial(layout, line, lit)), 1);
    place(light.emitter(lineGeometry(layout, TRACE, TAIL), lineLight(layout, line, LINE_LIGHT)), 1, 0, 0, 0.02);

    const routes: [Wire, Point[], number][] = [
      ['feed', layout.wires.feed, SIGNAL], ['probe', layout.wires.probe, SIGNAL], ['output', layout.wires.output, accent],
      ['stubA', layout.wires.stubs[0], SIGNAL], ['stubB', layout.wires.stubs[1], SIGNAL], ['truth', layout.wires.truth, SIGNAL],
      ['judged', layout.wires.judged, SIGNAL], ['called', layout.wires.called, accent], ['verdict', layout.wires.verdict, HIT],
    ];
    for (const [name, points, color] of routes) {
      place(new THREE.Mesh(wireGeometry(points), wireMaterial(color, this.wires[name], lit)), 2);
      place(light.emitter(wireGeometry(points), wireLight(color, this.wires[name], WIRE_LIGHT)), 2, 0, 0, 0.02);
    }

    place(new THREE.Mesh(flat(PAD * 2 + 0.04, PAD * 2 + 0.04), padMaterial(layout, this.padGlow, lit)), 3, layout.pad.x, layout.pad.y);
    place(light.emitter(disc(PAD * 0.8), rgb(SIGNAL).mul(this.padGlow).mul(PAD_LIGHT)), 3, layout.pad.x, layout.pad.y, 0.03);
    place(new THREE.Mesh(flat(RESISTOR.x * 2 + 0.04, RESISTOR.y * 2 + 0.04), resistorMaterial(layout, this.heat, lit)), 3, layout.resistor.x, layout.resistor.y);
    place(light.emitter(flat(RESISTOR.x * 1.2, RESISTOR.y * 1.6), rgb(0xff8a4a).mul(this.heat).mul(HEAT_LIGHT)), 3, layout.resistor.x, layout.resistor.y, 0.03);

    for (const spec of chipSpecs(layout, this.agentGlow)) {
      const reach = Math.max(spec.half.x, spec.half.y) * 2 + 0.4;
      place(new THREE.Mesh(flat(reach, reach), chipMaterial(spec, lit)), 4, spec.centre.x, spec.centre.y);
      place(light.occluder(flat(spec.half.x * 2, spec.half.y * 2)), 4, spec.centre.x, spec.centre.y, 0.01);
    }
    place(light.emitter(flat(0.3, 0.3), rgb(accent).mul(this.agentGlow).mul(LED_LIGHT * 0.35)), 4, layout.agent.x, layout.agent.y, 0.03);

    for (const [at, color, level] of [[layout.truth, SIGNAL, this.truthLevel], [layout.call, accent, this.callLevel]] as const) {
      place(new THREE.Mesh(flat(LED * 2 + 0.04, LED * 2 + 0.04), ledMaterial(at, LED, color, level, lit)), 4, at.x, at.y);
      place(light.emitter(disc(LED * 0.72), rgb(color).mul(level).mul(LED_LIGHT)), 4, at.x, at.y, 0.03);
    }
    place(new THREE.Mesh(flat(LAMP * 2 + 0.36, LAMP * 2 + 0.36), verdictMaterial(layout, this.verdict, lit)), 4, layout.verdict.x, layout.verdict.y);
    place(light.emitter(disc(LAMP * 0.76), verdictLight(this.verdict, LAMP_LIGHT)), 4, layout.verdict.x, layout.verdict.y, 0.03);

    const strip = logBounds(layout);
    place(new THREE.Mesh(flat(strip.size.x, strip.size.y), logMaterial(layout, { verdicts: this.verdicts, shift: this.shift }, lit)), 4, strip.centre.x, strip.centre.y);
  }

  protected override drop(): void {
    this.bits.dispose();
    this.verdicts.dispose();
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

  private pairOf(f: EnvFrame): number {
    return parityBit(f, this.layout.delay + 1) ^ parityBit(f, this.layout.delay + 2);
  }

  private check(f: EnvFrame): void {
    const call = parityCall(f), pair = this.pairOf(f);
    if ((f.reward !== 0 && f.reward !== 1) || f.actions[0] !== call || parityTruth(f) !== pair || f.observations[0] !== parityBit(f, 0)) {
      throw new Error(`Parity step ${f.step} paid ${f.reward} for calling ${f.actions[0]} while bits ${this.layout.delay} and ${this.layout.delay + 1} back ${pair ? 'differ' : 'agree'}`);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0;
    const slide = smoothstep01((f - HOLD) / (1 - HOLD)), layout = this.layout, cells = layout.cells;
    if (k !== this.shown) {
      if (!frame.scrubbed && k === this.shown + 1) this.landed = time;
      else this.landed = -Infinity;
      this.shown = k;
    }
    this.fill(now, next);
    this.offset.value = slide;

    const bit = parityBit(now, 0), incoming = next ? parityBit(next, 0) : bit;
    travel(this.wires.feed, bit, incoming, Math.min(1, slide * 2));
    this.padGlow.value = bit + (incoming - bit) * smoothstep01((slide - 0.4) / 0.2);
    travel(this.wires.probe, bit, incoming, clamp01(slide * 2 - 1));

    const call = k > 0 ? parityCall(now) : 0, called = next ? parityCall(next) : call;
    travel(this.wires.output, call, called, slide);
    this.agentGlow.value = slide > 0 ? called : call;

    const tapA = this.bitsData[(Math.floor(cells - 1.5 - slide) + 1) * 4]!, tapB = this.bitsData[(Math.floor(cells - 0.5 - slide) + 1) * 4]!;
    hold(this.wires.stubA, tapA);
    hold(this.wires.stubB, tapB);

    if (k > 0) this.check(now);
    const pair = this.pairOf(now), hit = k > 0 && now.reward > 0 ? 1 : 0;
    this.truthLevel.value = pair;
    this.callLevel.value = call;
    hold(this.wires.truth, pair);
    hold(this.wires.judged, pair);
    hold(this.wires.called, call);
    hold(this.wires.verdict, hit);
    this.verdict.hit.value = hit;
    this.verdict.shown.value = k > 0 ? 1 : 0;
    this.verdict.flash.value = Math.exp(-(time - this.landed) / FLASH);
    this.heat.value = Math.max(parityBit(now, cells - 1) * slide, parityBit(now, cells) * Math.exp(-(time - this.landed) / COOL) * (1 - slide));

    this.sync(frame, k);
    this.shift.value = slide;
    this.celebrate(frame, pool, time);
    Object.assign(this.ledger, { step: k, bit, call: k > 0 ? call : -1, truth: pair, reward: now.reward });
  }

  private fill(now: EnvFrame, next: EnvFrame | null): void {
    const nextStep = next ? next.step : -1;
    if (this.filled === now.step && this.filledNext === nextStep) return;
    const data = this.bitsData, cells = this.layout.cells;
    data[0] = next ? parityBit(next, 0) : 0;
    for (let age = 0; age < cells; age++) data[(age + 1) * 4] = parityBit(now, age);
    this.bits.needsUpdate = true;
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
      this.check(f);
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
    for (let i = 0; i < LOG; i++) {
      const s = k - i, slot = s % WINDOW;
      this.logData[i * 4] = s >= 1 && this.tags[slot] === s ? (this.hits[slot] ? 1 : 2) : 0;
    }
    this.verdicts.needsUpdate = true;
    this.logged = k;
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    const at = this.layout.verdict;
    for (const event of frame.events) {
      if (event.type === 'goal') pool.burst(this.point.set(at.x, 42, at.y), HIT, 0.3, time);
      else if (event.type === 'harm') pool.sigh(0.5, this.point.set(at.x, 47.4, at.y), time);
    }
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, d = this.layout.delay, judged = (value: number) => (l.call < 0 ? '—' : value ? 'differ' : 'same');
    return {
      title: 'PARITY',
      rows: [
        { key: 'bit', label: 'newest bit', tint: SIGNAL, value: () => String(l.bit) },
        { key: 'call', label: 'call', tint: accent, value: () => judged(l.call) },
        { key: 'pair', label: `bits ${d} & ${d + 1} back`, tint: SIGNAL, value: () => judged(l.truth) },
        { key: 'reward', label: 'reward', tint: HIT, value: () => (l.call < 0 ? '—' : l.reward > 0 ? '+1' : '0') },
        { key: 'right', label: `right, last ${formatCount(WINDOW)}`, tint: HIT, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      ],
      note: () => `${formatCount(d)}-step delay · chance 50%`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const layout = this.layout, cells = layout.cells, offset = this.offset.value, station = this.station, next = this.at(frame, k + 1);
    const arcs = pathArcs(layout, TAIL), path = kit.curve('path', arcs.length, SIGNAL);
    for (const arc of arcs) {
      along(layout, arc, station);
      path.point(station.x, 0.06, station.z);
    }
    const marks = kit.marks('cells', cells + 1);
    const mark = (arc: number, value: number) => {
      if (arc < 0 || arc > cells) return;
      along(layout, arc, station);
      marks.set(station.x, 0.08, station.z, value ? 0.13 : 0.05, value ? 0xffffff : 0x6b6480);
    };
    for (let age = 0; age < cells; age++) mark(age + 0.5 + offset, parityBit(now, age));
    if (next) mark(offset - 0.5, parityBit(next, 0));
    const taps = kit.marks('taps', 2);
    for (const tap of layout.taps) taps.set(tap.x, 0.08, tap.y, 0.07, 0xff5ec8);
    const pair = this.pairOf(now), call = k > 0 ? parityCall(now) : -1, reward = now.reward;
    kit.marks('pair', 1).set(layout.truth.x, 0.08, layout.truth.y, pair ? 0.16 : 0.06, SIGNAL);
    kit.marks('call', 1).set(layout.call.x, 0.08, layout.call.y, call === 1 ? 0.16 : 0.06, this.accentOf(this.ctx!));
    kit.marks('verdict', 1).set(layout.verdict.x, 0.08, layout.verdict.y, k > 0 ? 0.2 : 0.06, k === 0 ? 0x6b6480 : reward > 0 ? HIT : MISS);
    const log = kit.marks('log', LOG), { right, z, pitch } = layout.log;
    for (let i = 0; i < LOG; i++) {
      const s = k - i, f = s >= 1 ? this.at(frame, s) : null;
      if (!f) continue;
      log.set(right - (i + this.shift.value) * pitch, 0.08, z, 0.035, f.reward > 0 ? HIT : MISS);
    }
  }
}

export default (() => new ParityScene()) satisfies SceneFactory;
