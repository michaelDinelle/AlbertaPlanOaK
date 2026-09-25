import * as THREE from 'three/webgpu';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { RATE_LATENT, RATE_SIGMA, SUPERVISED_SHOWN } from '@sim/envs/supervised';
import { wordFloat } from '@sim/words';
import { parameter } from '../../data/envs';
import { formatCount, formatRate, formatSigned } from '../../format';
import {
  CORAL, GAUGE, GOLD, HISTORY, INK, KEY, LANE_X, METER, MID_Y, NOW_X, SCALE_X, SEA, SLOT, START, STEEL, SWEEP, Text, UNIT,
  buoy, disc, gauge, lantern, level, panel, sea, type GaugeLevels, type LanternLevels, type SeaLevels,
} from './art';

const SPAN = HISTORY + 3;
const HOLD = 0.6;
const FLASH = 0.55;
const TAU = 10;
const MARKS = 12;
const TOP = 4.3;

const levelText = (value: number) => Number.isFinite(value) ? value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';
const markText = (value: number) => `${value < 0 ? '−' : ''}${Math.abs(value).toLocaleString('en-US')}`;

class RateTrackingScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'rateTracking';
  override legend: LegendSpec | null = null;

  private scale = 1;
  private drift = 1;
  private period = 1;
  private accent = 0xffffff;
  private readonly ok = new Uint8Array(SPAN);
  private readonly shown = new Float64Array(SPAN);
  private readonly hidden = new Float64Array(SPAN);
  private readonly sigma = new Float64Array(SPAN);
  private readonly act = new Float64Array(SPAN);
  private readonly paid = new Float64Array(SPAN);
  private readonly guess = new Float64Array(SPAN);
  private readonly guessed = new Uint8Array(SPAN);
  private readonly filled = new Float64Array(SPAN);
  private readonly cells = new Float32Array((HISTORY + 1) * 2 * 4);
  private readonly store = new THREE.DataTexture(this.cells, HISTORY + 1, 2, THREE.RGBAFormat, THREE.FloatType);
  private readonly seaLevels: SeaLevels = { scroll: level(), center: level(), five: level(), flash: level(), time: level() };
  private readonly lanternLevels: LanternLevels = { railY: level(), railA: level(), hangY: level(), hangA: level(), capHigh: level(), capLow: level(), landX: level(LANE_X), landY: level(), landA: level(), landS: level(1) };
  private readonly gaugeLevels: GaugeLevels = { needle: level(START), period: level(), agent: level(), oracle: level() };
  private readonly floatY = level(MID_Y);
  private readonly lamps = { float: level(5), hang: level(), land: level(), flash: level() };
  private float: THREE.Mesh | null = null;
  private readonly glows: Record<keyof RateTrackingScene['lamps'], THREE.Mesh | null> = { float: null, hang: null, land: null, flash: null };
  private readonly texts: Text[] = [];
  private marks: Text[] = [];
  private reach: Text | null = null;
  private front: Text | null = null;
  private sigmaText: Text | null = null;
  private agentText: Text | null = null;
  private oracleText: Text | null = null;
  private center = NaN;
  private step = -1;
  private readonly view = { s: 0, base: 0, center: 0, floatY: MID_Y, railY: MID_Y, hangY: 0, hangA: 0, landX: LANE_X, landY: 0, landA: 0, needle: 0, frontX: NaN };
  private readonly ledger = { step: 0, reading: NaN, forecast: NaN, level: NaN, reward: NaN, sigma: NaN, agent: NaN, oracle: NaN };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    if (env.obs !== 2 || env.words !== 3 || env.actions !== 1 || env.box !== 1 || parameter(env, 'inputs') !== 1) throw new Error(`${env.envId} shows ${env.obs} observations, keeps ${env.words} words and takes ${env.actions} actions`);
    this.scale = parameter(env, 'scale');
    this.drift = parameter(env, 'drift');
    this.period = parameter(env, 'period');
    this.accent = this.accentOf(ctx);
    return new THREE.Box2(new THREE.Vector2(-8.1, -4.55), new THREE.Vector2(8.1, 4.55));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa9b7cc).multiplyScalar(0.72);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 40, sighSize: 20, sighRadius: 0.6, sighHeight: 1 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (_frame, kit) => this.truth(kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit, accent = this.accent;
    this.scene.background = new THREE.Color(0x03060c);
    pool.group.renderOrder = 11;
    this.store.needsUpdate = true;
    this.scene.add(sea(this.store, this.seaLevels, this.drift, this.scale, accent, lit));
    this.scene.add(lantern(this.lanternLevels, this.scale * UNIT, accent));
    this.scene.add(panel(lit), gauge(this.gaugeLevels, 2 * Math.round(this.drift), accent, lit));
    this.float = buoy(this.floatY, lit);
    this.scene.add(this.float);

    const glow = (key: keyof RateTrackingScene['lamps'], radius: number, tint: number) => {
      const mesh = light.emitter(disc(radius), rgb(tint).mul(this.lamps[key]));
      mesh.renderOrder = 1;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.glows[key] = mesh;
    };
    glow('float', 0.05, GOLD);
    glow('hang', 0.045, accent);
    glow('land', 0.04, accent);
    glow('flash', 0.05, CORAL);
    const hub = light.emitter(disc(0.06), rgb(accent).mul(1.6));
    hub.position.set(GAUGE.x, 0, -GAUGE.y);
    this.scene.add(hub);

    const put = (text: Text): Text => {
      this.texts.push(text);
      this.scene.add(text.mesh);
      return text;
    };
    put(new Text('NOW', 0.13, INK, 0, 700, 0.14).set('NOW').place(NOW_X, TOP));
    put(new Text('NEXT', 0.13, accent, 0, 700, 0.14).set('NEXT').place(LANE_X + 0.05, TOP));
    put(new Text('DRIFT σ', 0.15, INK, 0, 700, 0.14).set('DRIFT σ').place(GAUGE.x, GAUGE.y + GAUGE.r + 0.42));
    const redrawn = `redrawn every ${formatCount(this.period)} steps`;
    put(new Text(redrawn, 0.1, 0x8b99aa, 0, 500, 0.04).set(redrawn).place(GAUGE.x, GAUGE.y - GAUGE.r - 0.34));
    put(new Text('level per step', 0.1, 0x8b99aa, 0, 500, 0.04).set('level per step').place(GAUGE.x, GAUGE.y - 0.76));
    const ticks = 2 * Math.round(this.drift);
    for (let i = 0; i <= ticks; i += 2) {
      const angle = START - (SWEEP * i) / ticks, value = (this.drift * i) / ticks;
      const text = Number.isInteger(value) ? String(value) : formatRate(value);
      put(new Text(text, 0.12, INK, 0, 600).set(text).place(GAUGE.x + Math.cos(angle) * GAUGE.r * 0.47, GAUGE.y + Math.sin(angle) * GAUGE.r * 0.47));
    }
    for (const [text, column, row] of [['reading', 0, 0], ['forecast', 1, 0], ['miss', 0, 1], ['hidden level', 1, 1]] as const) put(new Text(text, 0.1, 0x9aa8b8, -1, 600, 0.04).set(text).place(KEY.x[column] + 0.14, KEY.y[row]));
    put(new Text('TYPICAL MISS', 0.13, INK, 0, 700, 0.14).set('TYPICAL MISS').place(METER.x, METER.top));
    const window = `root mean square, last ${HISTORY} steps`;
    put(new Text(window, 0.1, 0x8b99aa, 0, 500, 0.04).set(window).place(METER.x, METER.top - 0.22));
    put(new Text('agent', 0.11, accent, 0, 600, 0.06).set('agent').place(METER.x - METER.gap / 2, METER.base - 0.2));
    put(new Text('oracle', 0.11, SEA, 0, 600, 0.06).set('oracle').place(METER.x + METER.gap / 2, METER.base - 0.2));
    this.sigmaText = put(new Text('0.00', 0.24, 0xf2f6fa, 0, 700));
    this.agentText = put(new Text('000.00', 0.12, accent, 0, 700));
    this.oracleText = put(new Text('000.00', 0.12, SEA, 0, 700));
    this.reach = put(new Text(`±${this.scale}`, 0.1, accent, -1, 600).set(`±${this.scale}`));
    this.front = put(new Text('NEW DRIFT', 0.11, 0xdcecf8, 1, 700, 0.1).set('NEW DRIFT'));
    this.marks = Array.from({ length: MARKS }, () => put(new Text('−00,000,000', 0.13, 0xa9b8c8, -1, 600)));

    const l = this.ledger;
    this.legend = {
      title: 'RATE TRACKING',
      rows: [
        { key: 'reading', label: 'reading', tint: GOLD, value: () => levelText(l.reading) },
        { key: 'forecast', label: 'next forecast', tint: accent, value: () => levelText(l.forecast) },
        { key: 'level', label: 'hidden level', tint: SEA, value: () => levelText(l.level) },
        { key: 'reward', label: 'reward', tint: CORAL, value: () => Number.isFinite(l.reward) ? formatSigned(l.reward).replace('-', '−') : '—' },
        { key: 'drift', label: 'drift σ', tint: STEEL, value: () => formatRate(l.sigma) },
        { key: 'miss', label: `mean miss², last ${HISTORY}`, tint: CORAL, value: () => `${formatRate(l.agent)} · oracle ${formatRate(l.oracle)}` },
      ],
      note: () => {
        const left = this.period - (l.step % this.period);
        return `new drift in ${formatCount(left)} step${left === 1 ? '' : 's'}`;
      },
      bar: () => (l.step % this.period) / this.period,
    };
  }

  protected override drop(): void {
    this.store.dispose();
    for (const text of this.texts) text.dispose();
    this.texts.length = 0;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private read(frame: FrameUpdate, t: number): void {
    for (let j = 0; j < SPAN; j++) {
      const k = t + 2 - j, f = k >= 0 ? frame.frameAt(k) : null;
      this.ok[j] = f !== null && f.step === k ? 1 : 0;
      if (!f || !this.ok[j]) continue;
      this.shown[j] = wordFloat(f.words[SUPERVISED_SHOWN]!);
      this.hidden[j] = wordFloat(f.words[RATE_LATENT]!);
      this.sigma[j] = wordFloat(f.words[RATE_SIGMA]!);
      this.act[j] = f.actions[0]!;
      this.paid[j] = f.reward;
      if (f.observations[0] !== 1 || f.observations[1] !== this.shown[j]) throw new Error(`RateTracking step ${k} observes ${f.observations[0]}, ${f.observations[1]} though it shows ${this.shown[j]}`);
    }
    for (let j = 0; j < SPAN - 1; j++) {
      const k = t + 2 - j;
      this.guessed[j] = this.ok[j] && this.ok[j + 1] && k >= 1 ? 1 : 0;
      if (!this.guessed[j]) continue;
      const g = this.shown[j + 1]! + this.scale * Math.max(-1, Math.min(1, this.act[j]!)), miss = g - this.shown[j]!;
      if (!(Math.abs(this.paid[j]! + miss * miss) <= 1e-3 * (1 + miss * miss))) throw new Error(`RateTracking step ${k} paid ${this.paid[j]} for forecasting ${g} against ${this.shown[j]}`);
      this.guess[j] = g;
    }
    this.guessed[SPAN - 1] = 0;
    let carry = NaN;
    for (let j = SPAN - 1; j >= 0; j--) {
      if (this.ok[j]) carry = this.hidden[j]!;
      this.filled[j] = carry;
    }
    carry = NaN;
    for (let j = 0; j < SPAN; j++) {
      if (this.ok[j]) carry = this.hidden[j]!;
      else if (Number.isNaN(this.filled[j]!)) this.filled[j] = carry;
    }
  }

  private average(from: number): number {
    let sum = 0, weight = 0;
    for (let i = 0; i < HISTORY; i++) {
      const j = from + i;
      if (j >= SPAN || !this.ok[j]) continue;
      const w = Math.exp(-i / TAU);
      sum += w * this.shown[j]!;
      weight += w;
    }
    return weight > 0 ? sum / weight : NaN;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const t = frame.state.step, alpha = clamp01(frame.position - t);
    this.read(frame, t);
    if (!this.ok[2]) return;
    const later = this.ok[1] === 1, s = later ? smoothstep01(clamp01((alpha - HOLD) / (1 - HOLD))) : 0;
    const now = this.average(2), next = later ? this.average(1) : now;
    if (Number.isFinite(now)) this.center = now + ((Number.isFinite(next) ? next : now) - now) * s;
    const base = Math.round(Number.isFinite(now) ? now : this.center), center = this.center;
    const screen = (value: number) => MID_Y + (value - center) * UNIT;
    const v = this.view;
    v.s = s;
    v.base = base;
    v.center = center;

    const cells = this.cells, row = (HISTORY + 1) * 4, x0 = this.filled[2]!, x1 = later ? this.hidden[1]! : x0;
    let frontX = NaN;
    for (let c = -1; c < HISTORY; c++) {
      const j = c + 2, o = (c + 1) * 4, k = t - c, seen = c >= 0 && this.ok[j] === 1, guessed = c >= 0 && this.guessed[j] === 1;
      cells[o] = seen ? this.shown[j]! - base : 0;
      cells[o + 1] = (c === -1 ? x0 + (x1 - x0) * s : this.filled[j]!) - base;
      cells[o + 2] = guessed ? this.guess[j]! - base : 0;
      cells[o + 3] = (seen ? 1 : 0) + (guessed ? 2 : 0) + (this.ok[c === -1 ? 2 : j] ? 4 : 0);
      const newer = this.ok[j + 1] ? this.sigma[j + 1]! : NaN, older = j + 2 < SPAN && this.ok[j + 2] ? this.sigma[j + 2]! : NaN;
      const front = newer !== older && Number.isFinite(newer) && Number.isFinite(older);
      cells[row + o] = Number.isFinite(newer) ? newer : this.ok[j] ? this.sigma[j]! : 0;
      cells[row + o + 1] = seen && guessed ? Math.abs(this.guess[j]! - this.shown[j]!) : 0;
      cells[row + o + 2] = front ? 1 : 0;
      cells[row + o + 3] = seen && k % 10 === 0 ? 1 : 0;
      if (front && Number.isNaN(frontX)) frontX = NOW_X - (c + s) * SLOT - SLOT / 2;
    }
    this.store.needsUpdate = true;
    v.frontX = frontX;

    const flash = alpha < FLASH && this.guessed[2] ? (1 - alpha / FLASH) ** 2 : 0;
    const sea = this.seaLevels;
    sea.scroll.value = s;
    sea.center.value = center - base;
    sea.five.value = ((base % 5) + 5) % 5;
    sea.flash.value = flash;
    sea.time.value = time;

    const floatLevel = this.shown[2]! + ((later ? this.shown[1]! : this.shown[2]!) - this.shown[2]!) * s;
    v.floatY = screen(floatLevel);
    this.floatY.value = v.floatY;
    this.float?.position.set(NOW_X, 0, -v.floatY);

    const clampAct = (a: number) => Math.max(-1, Math.min(1, a));
    const lan = this.lanternLevels, half = this.scale * UNIT;
    const hangAct = s > 0 ? this.act[0]! : this.act[1]!, hanging = s > 0 ? this.guessed[0] === 1 : this.guessed[1] === 1;
    v.railY = v.floatY;
    v.hangY = screen(floatLevel + this.scale * clampAct(hangAct));
    v.hangA = hanging ? (s > 0 ? smoothstep01(clamp01((s - 0.35) / 0.65)) : 1) : 0;
    v.landX = LANE_X + (NOW_X - LANE_X) * s;
    v.landY = this.guessed[1] ? screen(this.guess[1]!) : 0;
    v.landA = s > 0 && this.guessed[1] ? 1 - s ** 3 : 0;
    lan.railY.value = v.railY;
    lan.railA.value = this.guessed[0] || this.guessed[1] ? 1 : 0;
    lan.hangY.value = v.hangY;
    lan.hangA.value = v.hangA;
    lan.capHigh.value = hangAct >= 1 ? v.hangA : 0;
    lan.capLow.value = hangAct <= -1 ? v.hangA : 0;
    lan.landX.value = v.landX;
    lan.landY.value = v.landY;
    lan.landA.value = v.landA;
    lan.landS.value = 1 - 0.45 * s;

    const sigmaNow = this.sigma[2]!, sigmaShown = sigmaNow + ((later ? this.sigma[1]! : sigmaNow) - sigmaNow) * s;
    v.needle = sigmaShown;
    const g = this.gaugeLevels;
    g.needle.value = START - (sigmaShown / this.drift) * SWEEP;
    g.period.value = (((t + s) % this.period) + this.period) % this.period / this.period;

    let agent = 0, oracle = 0, n = 0, m = 0;
    for (let j = 2; j < HISTORY + 2; j++) {
      if (this.guessed[j]) {
        const miss = this.guess[j]! - this.shown[j]!;
        agent += miss * miss;
        n++;
      }
      if (this.ok[j] && this.ok[j + 1] && t + 2 - j >= 1) {
        const miss = this.hidden[j + 1]! - this.shown[j]!;
        oracle += miss * miss;
        m++;
      }
    }
    const l = this.ledger;
    l.step = t;
    l.reading = this.shown[2]!;
    l.forecast = this.guessed[1] ? this.guess[1]! : NaN;
    l.level = this.hidden[2]!;
    l.reward = t >= 1 ? this.paid[2]! : NaN;
    l.sigma = sigmaNow;
    l.agent = n > 0 ? agent / n : NaN;
    l.oracle = m > 0 ? oracle / m : NaN;
    g.agent.value = Number.isFinite(l.agent) ? Math.sqrt(l.agent) : 0;
    g.oracle.value = Number.isFinite(l.oracle) ? Math.sqrt(l.oracle) : 0;

    this.lamps.hang.value = 4 * v.hangA;
    this.lamps.land.value = 4 * v.landA;
    const severity = this.guessed[2] ? Math.min(1, Math.abs(this.guess[2]! - this.shown[2]!) / this.scale) : 0;
    this.lamps.flash.value = 9 * flash * (0.35 + 0.65 * severity);
    const colX = NOW_X - s * SLOT;
    this.glows.float?.position.set(NOW_X, 0.01, -v.floatY);
    this.glows.hang?.position.set(LANE_X, 0.01, -v.hangY);
    this.glows.land?.position.set(v.landX, 0.01, -v.landY);
    if (this.guessed[2]) this.glows.flash?.position.set(colX, 0.01, -(screen(this.guess[2]!) + screen(this.shown[2]!)) / 2);

    this.label(center, sigmaShown);
    this.celebrate(frame, t, pool, time);
    this.step = t;
  }

  private label(center: number, sigma: number): void {
    const low = Math.ceil((center + (-4.4 - MID_Y) / UNIT) / 5), high = Math.floor((center + (TOP - 0.25 - MID_Y) / UNIT) / 5);
    for (const mark of this.marks) mark.mesh.visible = false;
    for (let n = low; n <= high; n++) {
      const value = 5 * n, mark = this.marks[((n % MARKS) + MARKS) % MARKS]!;
      mark.set(markText(value)).place(SCALE_X, MID_Y + (value - center) * UNIT).mesh.visible = true;
    }
    const v = this.view;
    this.reach?.place(LANE_X + 0.075, v.railY + this.scale * UNIT + 0.09);
    if (this.reach) this.reach.mesh.visible = this.lanternLevels.railA.value > 0;
    if (this.front) {
      this.front.mesh.visible = Number.isFinite(v.frontX) && v.frontX > NOW_X - HISTORY * SLOT * 0.7;
      if (this.front.mesh.visible) this.front.place(v.frontX - 0.06, TOP - 0.3);
    }
    this.sigmaText?.set(sigma.toFixed(2)).place(GAUGE.x, GAUGE.y - 0.54);
    const g = this.gaugeLevels;
    const l = this.ledger;
    if (this.agentText) this.agentText.set(formatRate(g.agent.value)).place(METER.x - METER.gap / 2, METER.base + g.agent.value * UNIT + 0.14).mesh.visible = Number.isFinite(l.agent);
    if (this.oracleText) this.oracleText.set(formatRate(g.oracle.value)).place(METER.x + METER.gap / 2, METER.base + g.oracle.value * UNIT + 0.14).mesh.visible = Number.isFinite(l.oracle);
  }

  private celebrate(frame: FrameUpdate, t: number, pool: Stage2D['pool'], time: number): void {
    if (frame.scrubbed || this.step < 0 || t <= this.step) return;
    for (let k = this.step + 1; k <= t; k++) {
      const j = t + 2 - k;
      if (j + 1 >= SPAN || !this.ok[j] || !this.ok[j + 1] || this.sigma[j] === this.sigma[j + 1]) continue;
      pool.burst({ x: GAUGE.x, y: 38, z: -GAUGE.y }, this.accent, 0.55, time);
      pool.ripple({ x: GAUGE.x, y: 0, z: -GAUGE.y }, this.accent, 0.34, time);
      return;
    }
  }

  private truth(kit: DebugKit): void {
    if (this.step < 0) return;
    const v = this.view, accent = this.accent, screen = (value: number) => MID_Y + (value - v.center) * UNIT;
    const hidden = kit.curve('hidden level', HISTORY + 1, SEA), readings = kit.marks('readings', HISTORY + 1), forecasts = kit.marks('forecasts', HISTORY + 2);
    for (let c = HISTORY - 1; c >= -1; c--) {
      const j = c + 2, x = NOW_X - (c + v.s) * SLOT, cell = this.cells[(c + 1) * 4 + 1]! + v.base;
      if (this.cells[(c + 1) * 4 + 3]! >= 4) hidden.point(x, 0.1, -screen(cell));
      if (c < 0) continue;
      if (this.ok[j]) readings.set(x, 0.1, -screen(this.shown[j]!), 0.05, GOLD);
      if (this.guessed[j]) forecasts.set(x, 0.1, -screen(this.guess[j]!), 0.05, accent);
    }
    readings.set(NOW_X, 0.1, -v.floatY, 0.07, GOLD);
    if (v.hangA > 0) forecasts.set(LANE_X, 0.1, -v.hangY, 0.07, accent);
    if (v.landA > 0) forecasts.set(v.landX, 0.1, -v.landY, 0.06, accent);
    const reach = kit.curve('reach', 2, accent), half = this.scale * UNIT;
    if (this.lanternLevels.railA.value > 0) {
      reach.point(LANE_X - 0.1, 0.1, -(v.railY - half));
      reach.point(LANE_X - 0.1, 0.1, -(v.railY + half));
    }
    const aim = screen(this.cells[1]! + v.base);
    kit.marks('oracle', 1).set(LANE_X + 0.16, 0.1, -aim, 0.05, 0xffffff);
    const band = kit.curve('drift', 2, STEEL);
    band.point(LANE_X + 0.26, 0.1, -(aim - v.needle * UNIT));
    band.point(LANE_X + 0.26, 0.1, -(aim + v.needle * UNIT));
    const front = kit.curve('front', 2, 0xdcecf8);
    if (Number.isFinite(v.frontX)) {
      front.point(v.frontX, 0.1, -4.6);
      front.point(v.frontX, 0.1, 4.6);
    }
  }
}

export default (() => new RateTrackingScene()) satisfies SceneFactory;
