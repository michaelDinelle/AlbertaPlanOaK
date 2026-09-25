import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { contextualAction } from '@sim/envs/contextual';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatRate, formatSigned } from '../../format';
import {
  BIAS, CAP, CARD_H, CARD_W, GAIN, HALO, HOUSING_H, HOUSING_Y, LABEL_REACH, LABEL_SIGMA, LABEL_Y, LOSS, Layout, METER_Y, MINUS, MOUTH_Y, PLATE_Y, PLUS, RAIL_Y, REACH, ROWS, SHARE_LEFT, SHARE_WIDTH, SIGMA, SLOT, SLOT_Y, SPREAD, TRAY_BOTTOM, ZERO,
  cardMaterial, consoleMaterial, flat, haloMaterial, labelMaterial, meterMaterial, meterWidth, routerMaterial, rowY, tabStart, type Scalar, type Tint,
} from './art';

const RING = 4096;
const MASK = RING - 1;
const WINDOW = 1024;
const GLIDE = 0.18;
const LIFT = 0.2;
const LAND = 0.56;
const SETTLE = 0.74;
const SWING_FROM = 0.6;
const SWING_TO = 0.88;
const FLASH = 0.16;
const AGE_DIM = 0.5;

const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const setTint = (target: Tint, color: THREE.Color, gain: number) => target.value.set(color.r * gain, color.g * gain, color.b * gain);

class Pulls {
  readonly arm = new Int32Array(RING);
  readonly pay = new Float32Array(RING);
  readonly context: Float32Array;
  private readonly held = new Int32Array(RING).fill(-1);

  constructor(private readonly width: number, private readonly arms: number) {
    this.context = new Float32Array(RING * width);
  }

  has(s: number): boolean {
    return s >= 0 && this.held[s & MASK] === s;
  }

  pulled(s: number): boolean {
    return s >= 1 && this.has(s) && this.has(s - 1);
  }

  sync(last: number, at: (s: number) => EnvFrame | null, block: number): boolean {
    let filled = false;
    for (let s = last; s >= Math.max(0, last - RING + 1); s--) {
      const slot = s & MASK;
      if (this.held[slot] === s) continue;
      const frame = at(s);
      if (!frame) {
        s -= s % block;
        continue;
      }
      this.arm[slot] = contextualAction(frame) % this.arms;
      this.pay[slot] = frame.reward;
      this.context.set(frame.observations.subarray(0, this.width), slot * this.width);
      this.held[slot] = s;
      filled = true;
    }
    return filled;
  }
}

class Summary {
  readonly stacks: Int32Array;
  readonly counts: Int32Array;
  readonly payouts: Float64Array;
  readonly sums: Float64Array;
  total = 0;
  paid = 0;
  extreme = 0;

  constructor(private readonly inputs: number, private readonly arms: number) {
    this.stacks = new Int32Array(arms * (ROWS + 1));
    this.counts = new Int32Array(arms);
    this.payouts = new Float64Array(arms);
    this.sums = new Float64Array(arms * inputs);
  }

  compute(pulls: Pulls, newest: number): void {
    const { inputs, arms } = this, depth = new Int32Array(arms);
    this.stacks.fill(-1);
    this.counts.fill(0);
    this.payouts.fill(0);
    this.sums.fill(0);
    this.total = this.paid = this.extreme = 0;
    let open = arms;
    for (let s = newest; s >= Math.max(1, newest - RING + 2) && open > 0; s--) {
      if (!pulls.pulled(s)) continue;
      const a = pulls.arm[s & MASK]!;
      if (depth[a]! > ROWS) continue;
      this.stacks[a * (ROWS + 1) + depth[a]!] = s;
      this.extreme = Math.max(this.extreme, Math.abs(pulls.pay[s & MASK]!));
      if (++depth[a]! > ROWS) open--;
    }
    for (let s = newest; s >= Math.max(1, newest - WINDOW + 1); s--) {
      if (!pulls.pulled(s)) continue;
      const a = pulls.arm[s & MASK]!, pay = pulls.pay[s & MASK]!, from = ((s - 1) & MASK) * (inputs + 1);
      this.counts[a]!++;
      this.payouts[a]! += pay;
      this.total++;
      this.paid += pay;
      this.extreme = Math.max(this.extreme, Math.abs(pay));
      for (let k = 0; k < inputs; k++) this.sums[a * inputs + k]! += pulls.context[from + k]!;
    }
  }
}

interface Lamp {
  emitter: THREE.Mesh;
  tint: Tint;
}

class ContextualScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 2;
  protected readonly stageName = 'contextual';

  private layout: Layout | null = null;
  private drift = 0;
  private pulls: Pulls | null = null;
  private summary: Summary | null = null;
  private values: THREE.DataTexture | null = null;
  private cards: THREE.InstancedMesh | null = null;
  private labels: THREE.InstancedMesh | null = null;
  private meters: THREE.InstancedMesh | null = null;
  private lamps: Lamp[] = [];
  private router: { mesh: THREE.Mesh; strength: Scalar; light: Tint } | null = null;
  private landing: { mesh: THREE.Mesh; tint: Tint } | null = null;
  private halo: { mesh: THREE.Mesh; tint: Tint; grow: Scalar; fade: Scalar } | null = null;
  private shares: Scalar[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private readonly accent = new THREE.Color();
  private readonly gain = new THREE.Color(GAIN);
  private readonly loss = new THREE.Color(LOSS);
  private readonly plus = new THREE.Color(PLUS);
  private readonly minus = new THREE.Color(MINUS);
  private readonly zero = new THREE.Color(ZERO);
  private readonly bias = new THREE.Color(BIAS);
  private summarized = -1;
  private scale = 0;
  private routerX = 0;
  private pace = 0;
  private lastPosition = 0;
  private readonly ledger = { arm: -1, pay: 0, mean: NaN, shares: '—', total: 0 };

  override legend: LegendSpec = {
    title: 'CONTEXTUAL',
    rows: [
      { key: 'arm', label: 'pulled', tint: TOKENS.accents.Contextual!, value: () => (this.ledger.arm < 0 ? '—' : `arm ${this.ledger.arm + 1}`) },
      { key: 'payout', label: 'payout', tint: GAIN, value: () => (this.ledger.arm < 0 ? '—' : formatSigned(this.ledger.pay)) },
      { key: 'mean', label: 'mean payout', tint: GAIN, value: () => formatSigned(this.ledger.mean) },
      { key: 'shares', label: 'arm shares', tint: TOKENS.accents.Contextual!, value: () => this.ledger.shares },
    ],
    note: () => `${this.layout?.inputs ?? 0} features + bias · ${this.drift > 0 ? `weights drift σ ${formatRate(this.drift)} per step` : 'fixed weights'} · last ${formatCount(this.ledger.total)} pulls`,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, inputs = parameter(env, 'inputs'), arms = constant(env, 'CONTEXTUAL_ARMS');
    this.drift = parameter(env, 'drift');
    if (env.obs !== inputs + 1 || env.actions !== arms) throw new Error(`${env.envId} observes ${env.obs} values and acts on ${env.actions} arms`);
    const layout = new Layout(inputs, arms);
    this.layout = layout;
    const half = layout.span + 0.45;
    return new THREE.Box2(new THREE.Vector2(-half, -(HOUSING_Y + HOUSING_H + 0.4)), new THREE.Vector2(half, -(TRAY_BOTTOM - 0.5)));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x6a7288).multiplyScalar(0.55);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const layout = this.layout;
        if (!layout) return;
        const step = frame.state.step, head = frame.frameAt(step), bars = kit.bars('context', layout.inputs + 1);
        for (let k = 0; k <= layout.inputs; k++) {
          const value = head.observations[k]!;
          bars.set(layout.columns[k]!, 0.1, -(METER_Y + value * SIGMA / 2), 0.02, k === layout.inputs ? GAIN : TOKENS.uiCyan, 0.05, Math.max(1e-3, Math.abs(value) * SIGMA));
        }
        const zero = kit.curve('zero', 2, 0x8fa6c4);
        zero.point(-layout.face, 0.1, -METER_Y);
        zero.point(layout.face, 0.1, -METER_Y);
        const summary = this.summary, shares = kit.bars('shares', layout.arms), means = kit.bars('means', layout.arms * layout.inputs);
        if (summary && summary.total > 0) layout.pockets.forEach((x, a) => {
          const n = summary.counts[a]!, width = SHARE_WIDTH * n / summary.total;
          if (width > 0) shares.set(x + SHARE_LEFT + width / 2, 0.1, -PLATE_Y, 0.02, TOKENS.uiCyan, width, 0.02);
          if (n > 0) for (let k = 0; k < layout.inputs; k++) {
            const mean = summary.sums[a * layout.inputs + k]! / n;
            means.set(x - CARD_W / 2 + (k + 0.5) * layout.cell, 0.1, -(LABEL_Y + mean * LABEL_SIGMA / 2), 0.02, TOKENS.uiCyan, 0.02, Math.max(1e-3, Math.abs(mean) * LABEL_SIGMA));
          }
        });
        if (step < 1) return;
        const arm = contextualAction(head) % layout.arms, x = layout.pockets[arm]!;
        kit.marks('routed', 1).set(x, 0.1, -RAIL_Y, 0.12, TOKENS.uiCyan);
        const length = this.scale > 0 ? Math.abs(head.reward) / this.scale * REACH : 0, start = x + tabStart();
        kit.bars('payout', 1).set(start + length / 2, 0.1, -rowY(0), 0.02, head.reward >= 0 ? GAIN : LOSS, Math.max(1e-3, length), 0.03);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const layout = this.layout!, accent = this.accentOf(ctx), inputs = layout.inputs, arms = layout.arms;
    this.accent.set(accent);
    this.scene.background = new THREE.Color(0x05070b);
    this.pulls = new Pulls(inputs + 1, arms);
    this.summary = new Summary(inputs, arms);
    this.shares = layout.pockets.map(() => uniform(0));

    const floorMesh = new THREE.Mesh(flat(120, 80), consoleMaterial(light, layout, this.shares, accent));
    floorMesh.renderOrder = 0;
    this.scene.add(floorMesh);

    const count = arms * (ROWS + 2) + 2;
    const values = new THREE.DataTexture(new Float32Array(inputs * count * 4), inputs, count, THREE.RGBAFormat, THREE.FloatType);
    values.magFilter = values.minFilter = THREE.NearestFilter;
    values.needsUpdate = true;
    this.values = values;
    const cardGeometry = flat(CARD_W, CARD_H);
    cardGeometry.setAttribute('look', new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage));
    cardGeometry.setAttribute('row', new THREE.InstancedBufferAttribute(Float32Array.from({ length: count }, (_, j) => j), 1));
    const cards = instancedMesh(cardGeometry, cardMaterial(light, values, layout, count, accent), count);
    cards.frustumCulled = false;
    cards.renderOrder = 3;
    for (let j = 0; j < count; j++) cards.setMatrixAt(j, this.matrix.makeScale(0, 0, 0));
    this.cards = cards;
    const labelGeometry = flat(CARD_W, 2 * LABEL_REACH);
    labelGeometry.setAttribute('look', new THREE.InstancedBufferAttribute(new Float32Array(arms * 4), 4).setUsage(THREE.DynamicDrawUsage));
    labelGeometry.setAttribute('row', new THREE.InstancedBufferAttribute(Float32Array.from({ length: arms }, (_, a) => arms * (ROWS + 1) + a), 1));
    const labels = instancedMesh(labelGeometry, labelMaterial(values, layout, count), arms);
    labels.frustumCulled = false;
    labels.renderOrder = 2;
    layout.pockets.forEach((x, a) => labels.setMatrixAt(a, this.matrix.makeTranslation(x, 0, -LABEL_Y)));
    labels.instanceMatrix.needsUpdate = true;
    this.labels = labels;
    this.scene.add(cards, labels);

    const meterGeometry = flat(meterWidth(layout.bar), 1);
    meterGeometry.setAttribute('look', new THREE.InstancedBufferAttribute(new Float32Array((inputs + 1) * 4), 4).setUsage(THREE.DynamicDrawUsage));
    const meters = instancedMesh(meterGeometry, meterMaterial(layout.bar), inputs + 1);
    meters.frustumCulled = false;
    meters.renderOrder = 4;
    this.meters = meters;
    this.scene.add(meters);
    this.lamps = layout.columns.map(() => {
      const tint: Tint = uniform(new THREE.Vector3());
      const emitter = light.emitter(flat(1, 1), tint);
      this.scene.add(emitter);
      return { emitter, tint };
    });

    const strength: Scalar = uniform(1.3), routerLight: Tint = uniform(new THREE.Vector3());
    const router = new THREE.Mesh(flat(0.58, 0.754), routerMaterial(accent, strength));
    router.renderOrder = 6;
    const routerEmitter = light.emitter(new THREE.CircleGeometry(0.1, 24).rotateX(-Math.PI / 2), routerLight);
    router.add(routerEmitter);
    this.router = { mesh: router, strength, light: routerLight };
    const landingTint: Tint = uniform(new THREE.Vector3());
    const landing = light.emitter(flat(CARD_W * 0.9, 0.07), landingTint);
    this.landing = { mesh: landing, tint: landingTint };
    const haloTint: Tint = uniform(new THREE.Vector3()), grow: Scalar = uniform(0), fade: Scalar = uniform(0);
    const halo = new THREE.Mesh(flat(CARD_W + 2 * HALO, CARD_H + 2 * HALO), haloMaterial(haloTint, grow, fade));
    halo.renderOrder = 5;
    this.halo = { mesh: halo, tint: haloTint, grow, fade };
    this.scene.add(router, landing, halo);
  }

  protected override drop(): void {
    const pool = this.pool?.group;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || object.parent === pool) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
      if (object instanceof THREE.InstancedMesh) object.dispose();
    });
    this.values?.dispose();
  }

  private heat(value: number, out: THREE.Color): THREE.Color {
    return out.copy(this.zero).lerp(value >= 0 ? this.plus : this.minus, 1 - Math.exp(-SPREAD * Math.abs(value)));
  }

  private card(j: number, x: number, y: number, size: number, look: THREE.InstancedBufferAttribute, pay: number, fade: number, glow: number, shown: number): void {
    this.cards!.setMatrixAt(j, this.matrix.makeScale(size, 1, size).setPosition(x, 0, -y));
    look.setXYZW(j, pay, fade, glow, shown);
  }

  private hide(j: number): void {
    this.cards!.setMatrixAt(j, this.matrix.makeScale(0, 0, 0));
  }

  private write(j: number, source: ArrayLike<number>, from: number, divide = 1): void {
    const data = this.values!.image.data as Float32Array, inputs = this.layout!.inputs;
    for (let k = 0; k < inputs; k++) data[(j * inputs + k) * 4] = source[from + k]! / divide;
  }

  protected tick(frame: FrameUpdate): void {
    const layout = this.layout, pulls = this.pulls, summary = this.summary, cards = this.cards, labels = this.labels, meters = this.meters, router = this.router, landing = this.landing, halo = this.halo, values = this.values;
    if (!layout || !pulls || !summary || !cards || !labels || !meters || !router || !landing || !halo || !values) return;
    const dt = frame.dtReal, inputs = layout.inputs, arms = layout.arms, width = inputs + 1;
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    const fast = this.pace > this.controlHz * 5, i = frame.state.step, head = frame.frameAt(i), next = fast ? null : at(i + 1);
    const t = next ? clamp01(frame.position - i) : 0;
    const filled = pulls.sync(next ? i + 1 : i, at, frame.eventLog.blockFrames);
    const toward = next ? contextualAction(next) % arms : -1, last = i >= 1 ? contextualAction(head) % arms : -1;
    const landed = next !== null && t >= LAND && pulls.pulled(i + 1), newest = landed ? i + 1 : i;
    if (filled || newest !== this.summarized) {
      summary.compute(pulls, newest);
      this.summarized = newest;
      this.ledger.shares = summary.total > 0 ? `${Array.from(summary.counts, (n) => Math.round(100 * n / summary.total)).join(' · ')} %` : '—';
    }
    const target = summary.extreme;
    this.scale = frame.scrubbed || target >= this.scale ? target : this.scale + (target - this.scale) * (1 - Math.exp(-1.5 * dt));
    const scale = this.scale > 0 ? this.scale : 1;

    const look = cards.geometry.getAttribute('look') as THREE.InstancedBufferAttribute, plates = labels.geometry.getAttribute('look') as THREE.InstancedBufferAttribute;
    const receiving = landed ? toward : -1, settle = landed ? smoothstep01(span(t, LAND, SETTLE)) : 1;
    for (let a = 0; a < arms; a++) {
      const x = layout.pockets[a]!;
      for (let r = 0; r <= ROWS; r++) {
        const j = a * (ROWS + 1) + r, s = summary.stacks[j]!;
        if (s < 0 || (r === ROWS && (a !== receiving || settle >= 1))) {
          this.hide(j);
          continue;
        }
        const position = a === receiving ? r - 1 + settle : r, fade = r === ROWS ? (1 - AGE_DIM) * clamp01(1 - 3 * settle) : 1 - AGE_DIM * clamp01(position / ROWS);
        this.write(j, pulls.context, ((s - 1) & MASK) * width);
        this.card(j, x, rowY(position), 1, look, pulls.pay[s & MASK]! / scale, fade, s === newest ? 1 : 0, 1);
      }
      const label = arms * (ROWS + 1) + a, n = summary.counts[a]!;
      this.write(label, summary.sums, a * inputs, Math.max(1, n));
      plates.setXYZW(a, n > 0 ? summary.payouts[a]! / n / scale : 0, n > 0 ? 1 : 0.5, 0, n > 0 ? 1 : 0);
      this.shares[a]!.value = summary.total > 0 ? n / summary.total : 0;
    }

    const slot = arms * (ROWS + 2), flyer = slot + 1;
    const lift = next ? smoothstep01(span(t, LIFT, LAND)) : 0, swing = next ? smoothstep01(span(t, SWING_FROM, SWING_TO)) : 0;
    const flying = next !== null && t >= LIFT && t < LAND && pulls.pulled(i + 1);
    const incoming = next !== null && t >= SWING_FROM;
    if (incoming) {
      this.write(slot, next.observations, 0);
      this.card(slot, 0, SLOT_Y, SLOT, look, 0, swing, 0, 0);
    } else if (next !== null && t >= LIFT) this.hide(slot);
    else {
      this.write(slot, head.observations, 0);
      this.card(slot, 0, SLOT_Y, SLOT, look, 0, 1, next ? smoothstep01(span(t, 0, LIFT)) * 0.6 : 0, 0);
    }
    if (flying) {
      const x2 = layout.pockets[toward]!, u = lift, v = 1 - u;
      const x = 2 * v * u * x2 + u * u * x2, y = v * v * SLOT_Y + 2 * v * u * (RAIL_Y + 0.7) + u * u * MOUTH_Y;
      this.write(flyer, head.observations, 0);
      this.card(flyer, x, y, 1 + (SLOT - 1) * v * v, look, 0, 1, 0.6, 0);
    } else this.hide(flyer);
    look.needsUpdate = plates.needsUpdate = true;
    cards.instanceMatrix.needsUpdate = true;
    values.needsUpdate = true;

    const meterLook = meters.geometry.getAttribute('look') as THREE.InstancedBufferAttribute;
    for (let k = 0; k <= inputs; k++) {
      const from = head.observations[k]!, value = next ? lerp(from, next.observations[k]!, swing) : from, reach = Math.abs(value) * SIGMA;
      const y = METER_Y + value * SIGMA / 2, x = layout.columns[k]!;
      meters.setMatrixAt(k, this.matrix.makeScale(1, 1, reach + 2 * CAP).setPosition(x, 0, -y));
      meterLook.setXYZW(k, value, reach, k === inputs ? 1 : 0, 0);
      const lamp = this.lamps[k]!;
      lamp.emitter.position.set(x, 0, -y);
      lamp.emitter.scale.set(layout.bar * 0.7, 1, Math.max(0.02, reach));
      if (k === inputs) setTint(lamp.tint, this.bias, 6);
      else setTint(lamp.tint, this.heat(value, this.color), 4 + 16 * (1 - Math.exp(-SPREAD * Math.abs(value))));
    }
    meterLook.needsUpdate = true;
    meters.instanceMatrix.needsUpdate = true;

    const fromX = last >= 0 ? layout.pockets[last]! : 0;
    if (next) this.routerX = lerp(fromX, layout.pockets[toward]!, smoothstep01(span(t, 0, GLIDE)));
    else this.routerX = frame.scrubbed ? fromX : this.routerX + (fromX - this.routerX) * (1 - Math.exp(-18 * dt));
    router.mesh.position.set(this.routerX, 0, -RAIL_Y);
    const through = flying ? Math.exp(-(((t - (LIFT + LAND) / 2) / 0.12) ** 2)) : 0;
    const flash = landed ? Math.exp(-(t - LAND) / FLASH) : 0;
    router.strength.value = 1.25 + 0.9 * through + 0.6 * flash;
    setTint(router.light, this.accent, 10 + 14 * through + 8 * flash);

    const landedPay = next ? next.reward : 0;
    landing.mesh.visible = landed && flash > 0.02;
    if (landing.mesh.visible) {
      landing.mesh.position.set(layout.pockets[receiving]!, 0, -rowY(0));
      setTint(landing.tint, landedPay >= 0 ? this.gain : this.loss, 16 * flash * (0.35 + 0.65 * Math.abs(landedPay) / scale));
    }

    const spread = landed ? span(t, LAND, LAND + 0.4) : 1;
    halo.mesh.visible = spread < 1;
    if (halo.mesh.visible) {
      halo.mesh.position.set(layout.pockets[receiving]!, 0, -rowY(settle - 1));
      halo.grow.value = 0.02 + 0.26 * (1 - (1 - spread) ** 3);
      halo.fade.value = (1 - spread) ** 2 * (0.45 + 0.55 * Math.abs(landedPay) / scale);
      setTint(halo.tint, landedPay >= 0 ? this.gain : this.loss, 1.8);
    }

    this.ledger.arm = last;
    this.ledger.pay = head.reward;
    this.ledger.total = summary.total;
    this.ledger.mean = summary.total > 0 ? summary.paid / summary.total : NaN;
  }
}

export default (() => new ContextualScene()) satisfies SceneFactory;
