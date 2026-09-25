import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, clampRange, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { covariateAcross, covariateAlong, covariateGuess, covariateRule, type CovariateRule } from '@sim/envs/covariate';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatRate, formatSigned, formatStep } from '../../format';
import {
  BIN, BINS, CLASS, GAP, GOLD, HEADER, LABEL_MID, METER_LIFT, METER_MID, MISS, PANEL, PANEL_MID, REACH, RISE, ROW_BASE, ROW_FLOOR, ROW_TOP, SPAN, STRIP, STRIP_MID, TICK, TICKS, TRACK_MID,
  backdrop, band, cross, crosshair, dashes, disc, dot, flat, halo, label, plate, sparkle, table,
  type Scalar, type Tint,
} from './art';

const WINDOW = 1024;
const RING = WINDOW + 2;
const DOT = 0.15;
const MARK = 0.24;
const INSET = 0.3;
const CORE = 0.36;
const PING_SECONDS = 0.24;
const GROW_SECONDS = 0.08;
const REVEAL_SECONDS = [0.02, 0.1] as const;
const GLINT_SECONDS = 0.3;
const FLASH_SECONDS = 0.34;
const FAST = 12;
const LAMP = 6;
const RULE_GLOW = 1.2;
const INK = 0x71879b;
const KERNEL = [0.135, 0.607, 1, 0.607, 0.135];
const PAST = 0.4;

const easeOut = (u: number) => 1 - (1 - u) ** 3;
const within = (u: number, v: number, out: THREE.Vector3) => out.set(clampRange(u, -REACH + INSET, REACH - INSET), 0, -clampRange(v, -RISE + INSET, RISE - INSET));

class Samples {
  readonly across = new Float32Array(RING);
  readonly along = new Float32Array(RING);
  readonly inputs: Float32Array;
  readonly guess = new Int8Array(RING);
  readonly hit = new Int8Array(RING);
  private readonly seen = new Int32Array(RING).fill(-1);
  private readonly judged = new Int32Array(RING).fill(-1);

  constructor(readonly width: number) {
    this.inputs = new Float32Array(RING * width);
  }

  has(s: number): boolean {
    return s >= 0 && this.seen[s % RING] === s;
  }

  known(s: number): boolean {
    return s >= 0 && this.judged[s % RING] === s;
  }

  clear(): void {
    this.seen.fill(-1);
    this.judged.fill(-1);
  }

  sync(rule: CovariateRule, first: number, last: number, at: (s: number) => EnvFrame | null): boolean {
    let fresh = false;
    for (let s = Math.max(0, first); s <= last; s++) {
      const slot = s % RING;
      if (this.seen[slot] !== s) {
        const frame = at(s);
        if (!frame) continue;
        const x = frame.observations;
        this.across[slot] = covariateAcross(rule, x);
        this.along[slot] = covariateAlong(rule, x);
        this.inputs.set(x.subarray(0, this.width), slot * this.width);
        this.seen[slot] = s;
        fresh = true;
      }
      if (this.judged[slot] !== s) {
        const next = at(s + 1);
        if (!next) continue;
        this.guess[slot] = covariateGuess(next);
        this.hit[slot] = next.reward === 1 ? 1 : 0;
        this.judged[slot] = s;
        fresh = true;
      }
    }
    return fresh;
  }
}

interface Marker {
  group: THREE.Group;
  core: THREE.Mesh;
  coreTint: Tint;
  ringTint: Tint;
  ringFade: Scalar;
  ping: THREE.Mesh;
  pingFade: Scalar;
  dash: THREE.Mesh;
  dashFade: Scalar;
  lamp: Tint;
}

class CovariateScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'covariate';

  private rule: CovariateRule | null = null;
  private samples: Samples | null = null;
  private shift = 0;
  private period = 1;
  private rows = 0;
  private readonly margins = new Float32Array(BINS * 4);
  private readonly marginTexture = new THREE.DataTexture(this.margins, BINS, 1, THREE.RGBAFormat, THREE.FloatType);
  private inputs = new Float32Array(4);
  private inputTexture: THREE.DataTexture | null = null;
  private readonly textures: THREE.Texture[] = [];
  private readonly smooth = new Float32Array(TICKS);
  private weighed = false;
  private dots: THREE.InstancedMesh | null = null;
  private misses: THREE.InstancedMesh | null = null;
  private marker: Marker | null = null;
  private glint: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private flash: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private centroid: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private readonly bandMarker = uniform(0);
  private readonly bandShown = uniform(0);
  private readonly classes = CLASS.map((hex) => new THREE.Color(hex));
  private readonly miss = new THREE.Color(MISS);
  private readonly white = new THREE.Color(1, 1, 1);
  private readonly color = new THREE.Color();
  private readonly matrix = new THREE.Matrix4();
  private readonly spot = new THREE.Vector3();
  private readonly glintSpot = new THREE.Vector3();
  private readonly flashSpot = new THREE.Vector3();
  private readonly mean = new THREE.Vector2();
  private arrived = -Infinity;
  private glintAt = -Infinity;
  private flashAt = -Infinity;
  private lastStep = -1;
  private lastPosition = 0;
  private pace = 0;
  private readonly ledger = { step: 0, guess: -1, truth: -1, margin: NaN, reward: 0, hits: 0, judged: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, inputs = constant(env, 'COVARIATE_INPUTS');
    if (env.obs !== inputs + 1 || env.words !== inputs + 1) throw new Error(`${env.envId} shows ${env.obs} observations and keeps ${env.words} words for ${inputs} inputs`);
    this.shift = parameter(env, 'shift');
    this.period = constant(env, 'COVARIATE_PERIOD');
    this.rows = env.obs;
    this.samples = new Samples(env.obs);
    this.inputs = new Float32Array((TICKS + 1) * this.rows * 4);
    this.inputTexture = new THREE.DataTexture(this.inputs, TICKS + 1, this.rows, THREE.RGBAFormat, THREE.FloatType);
    return new THREE.Box2(new THREE.Vector2(PANEL_MID - PANEL / 2 - 0.35, -RISE - 0.35), new THREE.Vector2(REACH + 0.35, STRIP_MID + STRIP / 2 + 1.6));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa6b8cc).multiplyScalar(0.6);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    return {
      debug: (frame, kit) => {
        const samples = this.samples;
        if (!samples) return;
        const i = frame.state.step, rule = kit.curve('rule', 2, TOKENS.uiCyan), outline = kit.curve('plate', 5, 0x8fa6c4);
        rule.point(0, 0.1, -RISE);
        rule.point(0, 0.1, RISE);
        for (const [x, z] of [[-REACH, -RISE], [REACH, -RISE], [REACH, RISE], [-REACH, RISE], [-REACH, -RISE]] as const) outline.point(x, 0.1, z);
        const truth = kit.marks('truth', WINDOW);
        for (let s = Math.max(0, i - WINDOW); s < i; s++) {
          if (!samples.has(s) || !samples.known(s)) continue;
          const slot = s % RING, label = samples.hit[slot] === 1 ? samples.guess[slot]! : 1 - samples.guess[slot]!;
          within(samples.across[slot]!, samples.along[slot]!, at);
          truth.set(at.x, 0.1, at.z, 0.075, CLASS[label]!);
        }
        if (samples.has(i)) {
          const slot = i % RING;
          within(samples.across[slot]!, samples.along[slot]!, at);
          kit.marks('sample', 1).set(at.x, 0.1, at.z, 0.12, 0xffffff);
          const margin = kit.curve('margin', 2, TOKENS.celebrationGold);
          margin.point(at.x, 0.1, at.z);
          margin.point(0, 0.1, at.z);
        }
        if (this.ledger.judged > 0) {
          within(this.mean.x, this.mean.y, at);
          kit.marks('centroid', 1).set(at.x, 0.1, at.z, 0.1, TOKENS.cueMagenta);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const accent = new THREE.Color(this.accentOf(ctx)), inputTexture = this.inputTexture!;
    this.scene.background = new THREE.Color(0x04070b);
    this.marginTexture.needsUpdate = true;
    inputTexture.needsUpdate = true;
    this.textures.push(this.marginTexture, inputTexture);
    const rows = [
      { key: 'guess', label: 'guess', tint: 0xdce6f0, value: () => (this.ledger.guess < 0 ? '—' : `class ${this.ledger.guess}`) },
      { key: 'truth', label: 'truth', tint: accent.getHex(), value: () => (this.ledger.truth < 0 ? '—' : `class ${this.ledger.truth} · ${formatRate(Math.abs(this.ledger.margin))} from the rule`) },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => (this.ledger.step < 1 ? '—' : formatSigned(this.ledger.reward)) },
      { key: 'accuracy', label: 'recent accuracy', tint: accent.getHex(), value: () => (this.ledger.judged > 0 ? `${formatPercent(this.ledger.hits / this.ledger.judged)} over ${formatCount(this.ledger.judged)}` : '—') },
    ];
    if (this.shift !== 0) rows.push({ key: 'period', label: 'period', tint: 0x8fa6c4, value: () => `${formatCount(Math.floor(this.ledger.step / this.period) + 1)} · shifts in ${formatCount(this.period - (this.ledger.step % this.period))}` });
    this.legend = {
      title: 'COVARIATE',
      rows,
      note: () => (this.shift !== 0 ? `shift ${Number(this.shift.toPrecision(3))} every ${formatStep(this.period)} steps` : 'stationary inputs'),
      bar: () => (this.ledger.judged > 0 ? this.ledger.hits / this.ledger.judged : 0),
    };

    const lit = light.lit;
    this.scene.add(backdrop(lit), plate(lit, accent), band(lit, accent, this.marginTexture, this.bandMarker, this.bandShown), table(lit, inputTexture, this.rows));
    this.scene.add(light.emitter(flat(0.08, 2 * RISE - 0.3), rgb(accent.clone().lerp(this.white, 0.35)).mul(RULE_GLOW)));

    const recent = formatCount(WINDOW);
    this.put('class 0', 0.28, CLASS[0], -REACH + 0.35, -RISE + 0.42, -1);
    this.put('class 1', 0.28, CLASS[1], REACH - 0.35, -RISE + 0.42, 1);
    this.put('rule', 0.24, accent.getHex(), 0.16, -RISE + 0.42, -1);
    this.put(`inputs, last ${recent}`, 0.22, INK, PANEL_MID + TRACK_MID, -(RISE - HEADER / 2));
    this.put('weight', 0.22, INK, PANEL_MID + METER_MID, -(RISE - HEADER / 2));
    this.put(`distance from the rule, last ${recent} guesses`, 0.2, INK, -REACH + 0.12, RISE + GAP / 2, -1);
    this.put('wrong guesses', 0.2, MISS, REACH - 0.12, RISE + GAP / 2, 1);
    for (let r = 0; r < this.rows; r++) this.put(r < this.rows - 1 ? `in ${r + 1}` : 'bias', 0.22, INK, PANEL_MID + LABEL_MID, -this.meterUp(r));

    const dots = instancedMesh(flat(1, 1), dot(), WINDOW), misses = instancedMesh(flat(1, 1), cross(), WINDOW);
    for (const [mesh, order] of [[dots, 3], [misses, 4]] as const) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(WINDOW * 3), 3);
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.renderOrder = order;
      this.scene.add(mesh);
    }
    this.dots = dots;
    this.misses = misses;

    const coreTint: Tint = uniform(new THREE.Vector3(1, 1, 1)), ringTint: Tint = uniform(new THREE.Vector3(1, 1, 1)), ringFade: Scalar = uniform(0);
    const pingFade: Scalar = uniform(0), dashFade: Scalar = uniform(0), lamp: Tint = uniform(new THREE.Vector3());
    const group = new THREE.Group(), ring = halo(ringTint, ringFade, 0.8, 0.09, 6), core = disc(coreTint), ping = halo(uniform(new THREE.Vector3(1, 1, 1)), pingFade, 0.9, 0.05, 6);
    ring.scale.setScalar(0.62);
    core.scale.setScalar(CORE);
    group.add(ring, core, ping, light.emitter(new THREE.CircleGeometry(0.1, 24).rotateX(-Math.PI / 2), lamp));
    group.visible = false;
    const dash = dashes(uniform(new THREE.Vector3(0.85, 0.9, 0.95)), dashFade);
    this.scene.add(dash, group);
    this.marker = { group, core, coreTint, ringTint, ringFade, ping, pingFade, dash, dashFade, lamp };

    const glintFade: Scalar = uniform(0), glint = sparkle(uniform(new THREE.Vector3().setFromColor(new THREE.Color(GOLD).multiplyScalar(1.6))), glintFade);
    glint.visible = false;
    const centroidFade: Scalar = uniform(0), mark = crosshair(uniform(new THREE.Vector3(0.9, 0.95, 1)), centroidFade);
    mark.scale.setScalar(0.62);
    mark.visible = false;
    const flashFade: Scalar = uniform(0), flash = halo(uniform(new THREE.Vector3().setFromColor(this.miss.clone().multiplyScalar(1.4))), flashFade, 0.86, 0.1, 8);
    flash.visible = false;
    this.scene.add(glint, flash, mark);
    this.glint = { mesh: glint, fade: glintFade };
    this.flash = { mesh: flash, fade: flashFade };
    this.centroid = { mesh: mark, fade: centroidFade };
  }

  private put(text: string, height: number, tint: number, x: number, z: number, align: -1 | 0 | 1 = 0): void {
    const made = label(text, height, tint);
    made.mesh.position.set(x - (align * made.width) / 2, 0, z);
    this.textures.push(made.texture);
    this.scene.add(made.mesh);
  }

  private meterUp(r: number): number {
    return ROW_TOP - (r + ROW_BASE) * ((ROW_TOP - ROW_FLOOR) / this.rows) + METER_LIFT;
  }

  private weigh(rule: CovariateRule): void {
    this.weighed = true;
    const values = [...rule.weights, rule.bias];
    for (let r = 0; r < this.rows; r++) {
      const w = values[r]!;
      this.put(`${w < 0 ? '−' : '+'}${Math.abs(w).toFixed(2)}`, 0.2, w < 0 ? CLASS[0] : CLASS[1], PANEL_MID + METER_MID, -(this.meterUp(r) + 0.36));
    }
  }

  private rebuild(i: number): void {
    const samples = this.samples!, dots = this.dots!, misses = this.misses!, rule = this.rule!, width = samples.width, inputs = this.inputs, margins = this.margins;
    margins.fill(0);
    inputs.fill(0);
    let shown = 0, crossed = 0, hits = 0, judged = 0, sumU = 0, sumV = 0;
    const regime = this.shift !== 0 ? Math.floor(i / this.period) * this.period : 0;
    for (let s = Math.max(0, i - WINDOW); s < i; s++) {
      if (!samples.has(s) || !samples.known(s)) continue;
      const slot = s % RING, u = samples.across[slot]!, v = samples.along[slot]!, guess = samples.guess[slot]!, hit = samples.hit[slot] === 1;
      const fade = (0.3 + 0.7 * (1 - (i - 1 - s) / WINDOW) ** 2) * (s < regime ? PAST : 1);
      within(u, v, this.spot);
      this.color.copy(hit ? this.classes[guess]! : this.miss).multiplyScalar(fade);
      const size = hit ? DOT : MARK, mesh = hit ? dots : misses, n = hit ? shown++ : crossed++;
      mesh.setMatrixAt(n, this.matrix.makeScale(size, 1, size).setPosition(this.spot));
      mesh.setColorAt(n, this.color);
      if (hit) hits++;
      judged++;
      sumU += u;
      sumV += v;
      const k = Math.floor((u + REACH) / BIN);
      if (k >= 0 && k < BINS) margins[k * 4 + (hit ? 0 : 1)]!++;
      for (let j = 0; j < width; j++) {
        const b = Math.round((samples.inputs[slot * width + j]! + SPAN) / TICK);
        if (b >= 0 && b < TICKS) inputs[(j * (TICKS + 1) + b) * 4]!++;
      }
    }
    let peak = 0;
    for (let k = 0; k < BINS; k++) peak = Math.max(peak, margins[k * 4]! + margins[k * 4 + 1]!);
    for (let k = 0; k < BINS && peak > 0; k++) {
      const right = margins[k * 4]!, all = right + margins[k * 4 + 1]!;
      margins[k * 4] = right / peak;
      margins[k * 4 + 1] = all / peak;
    }
    const scale = Math.max(...rule.weights.map(Math.abs), Math.abs(rule.bias)), current = samples.has(i) ? (i % RING) * width : -1;
    const smooth = this.smooth, reach = (KERNEL.length - 1) / 2;
    for (let j = 0; j < width; j++) {
      let top = 0;
      for (let b = 0; b < TICKS; b++) {
        let sum = 0;
        for (let o = -reach; o <= reach; o++) if (b + o >= 0 && b + o < TICKS) sum += KERNEL[o + reach]! * inputs[(j * (TICKS + 1) + b + o) * 4]!;
        smooth[b] = sum;
        top = Math.max(top, sum);
      }
      for (let b = 0; b < TICKS; b++) inputs[(j * (TICKS + 1) + b) * 4] = top > 0 ? smooth[b]! / top : 0;
      const info = (j * (TICKS + 1) + TICKS) * 4;
      inputs[info] = current < 0 ? Number.NaN : samples.inputs[current + j]!;
      inputs[info + 1] = (j < rule.weights.length ? rule.weights[j]! : rule.bias) / scale;
    }
    this.marginTexture.needsUpdate = true;
    this.inputTexture!.needsUpdate = true;
    dots.count = shown;
    misses.count = crossed;
    for (const mesh of [dots, misses]) {
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor!.needsUpdate = true;
    }
    this.ledger.hits = hits;
    this.ledger.judged = judged;
    if (judged > 0) this.mean.set(sumU / judged, sumV / judged);
  }

  protected tick(frame: FrameUpdate, { time }: Stage2D): void {
    const samples = this.samples, marker = this.marker, glint = this.glint, flash = this.flash, centroid = this.centroid;
    if (!samples || !marker || !glint || !flash || !centroid) return;
    const i = frame.state.step;
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > FAST, at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    if (frame.scrubbed || !this.rule) {
      this.rule = covariateRule(frame.frameAt(i).words);
      samples.clear();
      this.arrived = this.glintAt = this.flashAt = -Infinity;
    }
    if (!this.weighed) this.weigh(this.rule);
    const fresh = samples.sync(this.rule, i - WINDOW, i, at), advanced = !frame.scrubbed && i !== this.lastStep;
    if (advanced) {
      this.arrived = fast ? -Infinity : time;
      if (!fast && samples.has(i - 1) && samples.known(i - 1)) {
        const slot = (i - 1) % RING;
        within(samples.across[slot]!, samples.along[slot]!, this.spot);
        if (samples.hit[slot] === 1) {
          this.glintAt = time;
          this.glintSpot.copy(this.spot);
        } else if (frame.events.some((event) => event.type === 'harm')) {
          this.flashAt = time;
          this.flashSpot.copy(this.spot);
        }
      }
    }
    if (fresh || advanced || frame.scrubbed) this.rebuild(i);
    this.lastStep = i;

    const slot = i % RING, has = samples.has(i), known = samples.known(i);
    marker.group.visible = has;
    marker.dash.visible = has;
    this.bandShown.value = has ? 1 : 0;
    this.ledger.step = i;
    this.ledger.reward = frame.state.reward;
    this.ledger.guess = known ? samples.guess[slot]! : -1;
    this.ledger.truth = known ? (samples.hit[slot] === 1 ? samples.guess[slot]! : 1 - samples.guess[slot]!) : -1;
    this.ledger.margin = has ? samples.across[slot]! : Number.NaN;
    if (has) {
      within(samples.across[slot]!, samples.along[slot]!, this.spot);
      marker.group.position.copy(this.spot);
      const age = time - this.arrived, live = Number.isFinite(age);
      const reveal = known ? (live ? smoothstep01((age - REVEAL_SECONDS[0]) / (REVEAL_SECONDS[1] - REVEAL_SECONDS[0])) : 1) : 0;
      const hue = known ? this.classes[samples.guess[slot]!]! : this.white;
      this.color.copy(this.white).lerp(hue, reveal).multiplyScalar(1.25);
      marker.coreTint.value.set(this.color.r, this.color.g, this.color.b);
      marker.core.scale.setScalar(CORE * (live ? 0.55 + 0.45 * easeOut(clamp01(age / GROW_SECONDS)) : 1));
      this.color.copy(hue).multiplyScalar(1.3);
      marker.ringTint.value.set(this.color.r, this.color.g, this.color.b);
      marker.ringFade.value = reveal;
      const ping = live ? age / PING_SECONDS : 1;
      marker.ping.visible = ping < 1;
      if (marker.ping.visible) {
        marker.ping.scale.setScalar(0.34 + 0.7 * easeOut(ping));
        marker.pingFade.value = 0.6 * (1 - ping) ** 2;
      }
      this.color.copy(this.white).lerp(hue, reveal).multiplyScalar(LAMP);
      marker.lamp.value.set(this.color.r, this.color.g, this.color.b);
      marker.dash.position.set(this.spot.x / 2, 0, this.spot.z);
      marker.dash.scale.set(Math.max(Math.abs(this.spot.x), 1e-3), 1, 1);
      marker.dashFade.value = 0.5;
      this.bandMarker.value = this.spot.x;
    }

    const shine = (time - this.glintAt) / GLINT_SECONDS;
    glint.mesh.visible = shine >= 0 && shine < 1;
    if (glint.mesh.visible) {
      glint.mesh.position.copy(this.glintSpot);
      glint.mesh.scale.setScalar(0.3 + 0.35 * easeOut(shine));
      glint.fade.value = (1 - shine) ** 2;
    }
    const burst = (time - this.flashAt) / FLASH_SECONDS;
    flash.mesh.visible = burst >= 0 && burst < 1;
    if (flash.mesh.visible) {
      flash.mesh.position.copy(this.flashSpot);
      flash.mesh.scale.setScalar(0.3 + 0.45 * easeOut(burst));
      flash.fade.value = (1 - burst) ** 2;
    }
    centroid.mesh.visible = this.ledger.judged > 0;
    if (centroid.mesh.visible) {
      centroid.mesh.position.copy(within(this.mean.x, this.mean.y, this.spot));
      centroid.fade.value = 0.75;
    }
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.textures.length = 0;
  }
}

export default (() => new CovariateScene()) satisfies SceneFactory;
