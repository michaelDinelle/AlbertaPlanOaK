import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { banditArm, banditBest } from '@sim/envs/bandit';
import type { EnvFrame } from '@sim/types';
import { wordFloat } from '@sim/words';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';
import {
  CABINET, CABINET_TALL, CROWN, DIGITS, FLOOR, KNOB, LED, LEVER, LEVER_X, PANEL_Y, PITCH, PIVOT, REACH, TOP, TUBE,
  cabinetMaterial, coinCoverage, coinShade, flat, hallMaterial, knobMaterial, needleMaterial, orbMaterial, ringMaterial, rodMaterial, sparkleMaterial, sprite, trailMaterial, tubeMaterial,
  type Scalar, type Tint,
} from './art';

const WINDOW = 256;
const RING = WINDOW + 2;
const BIN = 0.16;
const COIN = 0.9;
const SPREAD = 0.62;
const EDGE = 0.12;
const STEP_SPAN = 0.5;
const AGE_DIM = 0.5;
const GLIDE = 0.22;
const PULL = 0.42;
const RETURN = 0.64;
const LAUNCH = 0.34;
const LAND = 0.72;
const LEAVE_FROM = 0.3;
const LEAVE_TO = 0.62;
const RING_LIFE = 0.8;
const FLASH_LIFE = 0.45;
const THROW = (150 * Math.PI) / 180;
const HOP = 0.45;
const TAIL = 0.07;
const SPIN = 3;
const GAIN = 0xffd166;
const LOSS = TOKENS.failureBlue;
const TRUTH = 0x9fe6ff;

const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const easeOut = (u: number) => 1 - (1 - u) ** 3;
const sideOf = (s: number) => (Math.imul(s, 0x9e3779b1) >>> 31 === 1 ? 1 : -1);
const binOf = (pay: number) => Math.floor(pay / BIN);

class Pulls {
  readonly arm = new Int16Array(RING);
  readonly pay = new Float32Array(RING);
  readonly now = new Int16Array(RING);
  readonly next = new Int16Array(RING);
  readonly counts: Int32Array;
  total = 0;
  widestNow = 0;
  widestNext = 0;
  extreme = 0;
  private readonly held = new Int32Array(RING).fill(-1);
  private readonly stacksNow = new Map<number, number>();
  private readonly stacksNext = new Map<number, number>();

  constructor(arms: number) {
    this.counts = new Int32Array(arms);
  }

  has(s: number): boolean {
    return s >= 1 && this.held[s % RING] === s;
  }

  sync(i: number, ahead: boolean, at: (s: number) => EnvFrame | null): void {
    const first = Math.max(1, i - WINDOW + 1), last = ahead ? i + 1 : i;
    this.stacksNow.clear();
    this.stacksNext.clear();
    this.counts.fill(0);
    this.total = this.widestNow = this.widestNext = this.extreme = 0;
    for (let s = first; s <= last; s++) {
      const slot = s % RING;
      if (this.held[slot] !== s) {
        const frame = at(s);
        if (!frame) continue;
        this.arm[slot] = banditArm(frame);
        this.pay[slot] = frame.reward;
        this.held[slot] = s;
      }
      const arm = this.arm[slot]!, bin = binOf(this.pay[slot]!), key = (bin * 64 + arm) * 2 + (sideOf(s) > 0 ? 1 : 0);
      this.extreme = Math.max(this.extreme, Math.max(-bin, bin + 1) * BIN);
      if (s <= i) {
        const rank = this.stacksNow.get(key) ?? 0;
        this.stacksNow.set(key, rank + 1);
        this.now[slot] = rank;
        this.widestNow = Math.max(this.widestNow, rank + 1);
        this.counts[arm]!++;
        this.total++;
      }
      if (ahead && s > i - WINDOW + 1) {
        const rank = this.stacksNext.get(key) ?? 0;
        this.stacksNext.set(key, rank + 1);
        this.next[slot] = rank;
        this.widestNext = Math.max(this.widestNext, rank + 1);
      }
    }
  }
}

interface Machine {
  x: number;
  glow: Scalar;
  light: Tint;
  needle: THREE.Mesh;
  needleTint: Tint;
  rod: THREE.Mesh;
  knob: THREE.Mesh;
}

class BanditScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 2;
  protected readonly stageName = 'bandit';

  private arms = 0;
  private drift = 0;
  private rowHalf = 0;
  private reach = 0;
  private scale = 1;
  private readonly scaleUniform = uniform(1);
  private pulls: Pulls | null = null;
  private machines: Machine[] = [];
  private coins: THREE.InstancedMesh | null = null;
  private flyer: { mesh: THREE.Mesh; tint: Tint; light: Tint; trail: THREE.Mesh; streak: Tint } | null = null;
  private ring: { mesh: THREE.Mesh; tint: Tint; fade: Scalar } | null = null;
  private orb: { mesh: THREE.Mesh; light: Tint; strength: Scalar } | null = null;
  private crown: { mesh: THREE.Mesh; light: THREE.Mesh } | null = null;
  private readonly means = new Float64Array(16);
  private readonly flashes = new Float64Array(16);
  private readonly matrix = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private readonly gain = new THREE.Color(GAIN);
  private readonly loss = new THREE.Color(LOSS);
  private readonly led = new THREE.Color(LED);
  private readonly accent = new THREE.Color();
  private readonly point = new THREE.Vector3();
  private readonly behind = new THREE.Vector3();
  private orbX = 0;
  private crownX = 0;
  private flare = 0;
  private pace = 0;
  private lastPosition = 0;
  private readonly ledger = { step: 0, arm: 0, reward: 0, mean: 0, best: 0, top: 0, share: NaN, total: 0 };

  override legend: LegendSpec = {
    title: 'BANDIT',
    rows: [
      { key: 'arm', label: 'pulled', tint: 0xffc857, value: () => (this.ledger.step > 0 ? `arm ${this.ledger.arm + 1}` : '—') },
      { key: 'payout', label: 'payout', tint: GAIN, value: () => (this.ledger.step > 0 ? formatSigned(this.ledger.reward) : '—') },
      { key: 'mean', label: 'its true mean', tint: TRUTH, value: () => (this.ledger.step > 0 ? formatSigned(this.ledger.mean) : '—') },
      { key: 'best', label: 'best', tint: TRUTH, value: () => `arm ${this.ledger.best + 1} · ${formatSigned(this.ledger.top)}` },
      { key: 'share', label: 'best-arm pulls', tint: 0xffc857, value: () => formatPercent(this.ledger.share) },
    ],
    note: () => `${this.drift > 0 ? `drift σ ${Number(this.drift.toPrecision(3))}` : 'stationary'} · last ${formatCount(this.ledger.total)} pulls`,
    bar: () => (Number.isFinite(this.ledger.share) ? this.ledger.share : 0),
  };

  private xOf(arm: number): number {
    return (arm - (this.arms - 1) / 2) * PITCH;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.arms = parameter(env, 'arms');
    this.drift = parameter(env, 'drift');
    if (env.words !== this.arms || env.actions !== this.arms) throw new Error(`${env.envId} carries ${env.words} means for ${this.arms} arms`);
    if (this.arms > DIGITS.length - 1) throw new Error(`${env.envId} has ${this.arms} arms; the machines number 1 to ${DIGITS.length - 1}`);
    this.rowHalf = (this.arms * PITCH) / 2;
    return new THREE.Box2(new THREE.Vector2(-this.rowHalf - 0.2, -CROWN - 0.8), new THREE.Vector2(this.rowHalf + 0.2, -FLOOR + 0.75));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x7a6a8c).multiplyScalar(0.45);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const step = frame.state.step, head = frame.frameAt(step), best = banditBest(head.words), marks = kit.marks('means', this.arms);
        const zero = kit.curve('zero', 2, 0x8fa6c4);
        zero.point(-this.rowHalf, 0.1, 0);
        zero.point(this.rowHalf, 0.1, 0);
        for (let a = 0; a < this.arms; a++) {
          const x = this.xOf(a), z = -wordFloat(head.words[a]!) * this.scale, line = kit.curve(`mean ${a + 1}`, 2, TOKENS.uiCyan);
          line.point(x - TUBE, 0.1, z);
          line.point(x + TUBE, 0.1, z);
          marks.set(x, 0.1, z, a === best ? 0.16 : 0.1, a === best ? TOKENS.celebrationGold : TOKENS.uiCyan);
        }
        const pulls = this.pulls, shares = kit.bars('share', this.arms);
        if (pulls && pulls.total > 0) for (let a = 0; a < this.arms; a++) {
          const width = (2 * CABINET * pulls.counts[a]!) / pulls.total;
          if (width > 0) shares.set(this.xOf(a) - CABINET + width / 2, 0.1, -FLOOR + 0.4, 0.05, a === best ? TOKENS.celebrationGold : TOKENS.uiCyan, width, 0.12);
        }
        if (step < 1) return;
        const arm = banditArm(head), x = this.xOf(arm), z = -head.reward * this.scale, payout = kit.curve('payout', 2, TOKENS.uiCyan);
        payout.point(x, 0.1, REACH);
        payout.point(x, 0.1, z);
        kit.marks('payout', 1).set(x, 0.1, z, 0.1, head.reward >= 0 ? GAIN : LOSS);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    for (const child of pool.group.children) child.renderOrder = 11;
    this.accent.set(this.accentOf(ctx));
    this.scene.background = new THREE.Color(0x06040a);
    this.pulls = new Pulls(this.arms);

    const hall = new THREE.Mesh(flat(this.rowHalf * 6 + 40, 40), hallMaterial(light, this.rowHalf));
    hall.position.set(0, 0, 2);
    hall.renderOrder = 0;
    this.scene.add(hall);

    const tube = tubeMaterial(light, this.scaleUniform), rod = rodMaterial(light), knob = knobMaterial(light);
    const tubeGeometry = flat(tube.width, tube.height), leverRod = flat(rod.width, 1), knobGeometry = flat(KNOB, KNOB);
    this.machines = [];
    for (let a = 0; a < this.arms; a++) {
      const x = this.xOf(a), glow = uniform(0.3), mask = uniform(DIGITS[a + 1]!);
      const cabinet = cabinetMaterial(light, glow, mask);
      const body = new THREE.Mesh(flat(cabinet.width, CABINET_TALL), cabinet.material);
      body.position.set(x, 0, -(TOP - CABINET_TALL / 2));
      body.renderOrder = 2;
      const back = new THREE.Mesh(tubeGeometry, tube.back);
      back.position.set(x, 0, 0);
      back.renderOrder = 3;
      const front = new THREE.Mesh(tubeGeometry, tube.front);
      front.position.set(x, 0, 0);
      front.renderOrder = 6;
      const needleTint: Tint = uniform(new THREE.Vector3());
      const needleArt = needleMaterial(needleTint);
      const needle = new THREE.Mesh(flat(needleArt.width, needleArt.height), needleArt.material);
      needle.position.set(x, 0, 0);
      needle.renderOrder = 5;
      const leverRodMesh = new THREE.Mesh(leverRod, rod.material);
      leverRodMesh.renderOrder = 7;
      const knobMesh = new THREE.Mesh(knobGeometry, knob);
      knobMesh.renderOrder = 7;
      const lamp: Tint = uniform(new THREE.Vector3());
      const emitter = light.emitter(flat(2 * (CABINET - 0.3), 0.4), lamp);
      emitter.position.set(x, 0, -PANEL_Y);
      this.scene.add(body, back, front, needle, leverRodMesh, knobMesh, emitter);
      this.machines.push({ x, glow, light: lamp, needle, needleTint, rod: leverRodMesh, knob: knobMesh });
    }

    const coinMaterial = sprite(light.lit(coinShade()).mul(0.45).add(coinShade().mul(0.55)), coinCoverage());
    const coins = instancedMesh(flat(1, 1), coinMaterial, RING);
    coins.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(RING * 3), 3);
    coins.count = 0;
    coins.frustumCulled = false;
    coins.renderOrder = 4;
    this.coins = coins;
    this.scene.add(coins);

    const flyerTint: Tint = uniform(new THREE.Vector3()), flyerLight: Tint = uniform(new THREE.Vector3());
    const flyer = new THREE.Mesh(flat(1, 1), sprite(coinShade().mul(flyerTint), coinCoverage()));
    flyer.renderOrder = 8;
    flyer.visible = false;
    const flyerEmitter = light.emitter(new THREE.CircleGeometry(0.07, 20).rotateX(-Math.PI / 2), flyerLight);
    flyer.add(flyerEmitter);
    const streak: Tint = uniform(new THREE.Vector3());
    const trail = new THREE.Mesh(flat(1, 1), trailMaterial(streak));
    trail.renderOrder = 8;
    trail.visible = false;
    this.flyer = { mesh: flyer, tint: flyerTint, light: flyerLight, trail, streak };

    const ringTint: Tint = uniform(new THREE.Vector3()), ringFade: Scalar = uniform(0);
    const ringMesh = new THREE.Mesh(flat(1, 1), ringMaterial(ringTint, ringFade));
    ringMesh.renderOrder = 8;
    ringMesh.visible = false;
    this.ring = { mesh: ringMesh, tint: ringTint, fade: ringFade };

    const orbLight: Tint = uniform(new THREE.Vector3()), strength: Scalar = uniform(1.4);
    const orb = new THREE.Mesh(flat(1.05, 1.05), orbMaterial(rgb(this.accent), strength));
    orb.renderOrder = 9;
    const orbEmitter = light.emitter(new THREE.CircleGeometry(0.11, 24).rotateX(-Math.PI / 2), orbLight);
    orb.add(orbEmitter);
    this.orb = { mesh: orb, light: orbLight, strength };

    const crown = new THREE.Mesh(flat(0.9, 0.9), sparkleMaterial(rgb(TRUTH).mul(2.2)));
    crown.renderOrder = 10;
    const crownLight = light.emitter(new THREE.CircleGeometry(0.09, 20).rotateX(-Math.PI / 2), rgb(TRUTH).mul(24));
    this.crown = { mesh: crown, light: crownLight };
    this.scene.add(trail, flyer, ringMesh, orb, crown, crownLight);
  }

  private knobY(angle: number): number {
    return PIVOT + LEVER * Math.cos(angle);
  }

  private setLever(machine: Machine, angle: number): void {
    const knob = this.knobY(angle), x = machine.x + LEVER_X;
    machine.rod.position.set(x, 0, -(PIVOT + knob) / 2);
    machine.rod.scale.set(1, 1, Math.max(1e-3, Math.abs(knob - PIVOT)));
    machine.knob.position.set(x, 0, -knob);
    machine.knob.scale.setScalar(1 + 0.35 * Math.sin(angle));
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const pulls = this.pulls, coins = this.coins, flyer = this.flyer, ring = this.ring, orb = this.orb, crown = this.crown;
    if (!pulls || !coins || !flyer || !ring || !orb || !crown) return;
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    const fast = this.pace > this.controlHz * 5, i = frame.state.step, head = frame.frameAt(i), next = fast ? null : at(i + 1);
    const t = next ? clamp01(frame.position - i) : 0, dt = frame.dtReal;
    pulls.sync(i, next !== null, at);

    const best = banditBest(head.words);
    let extreme = pulls.extreme;
    for (let a = 0; a < this.arms; a++) {
      this.means[a] = wordFloat(head.words[a]!);
      extreme = Math.max(extreme, Math.abs(this.means[a]!));
    }
    const reach = Math.max(STEP_SPAN, Math.ceil((extreme * 1.04) / STEP_SPAN) * STEP_SPAN);
    this.reach = frame.scrubbed || this.reach === 0 ? reach : this.reach + (reach - this.reach) * (1 - Math.exp(-3 * dt));
    const scale = (REACH - EDGE) / this.reach;
    this.scale = scale;
    this.scaleUniform.value = scale;

    const d = COIN * BIN * scale, room = TUBE - d / 2 - 0.04;
    const gapOf = (widest: number) => Math.min(SPREAD * d, room / Math.max(0.5, widest - 0.5));
    const gapNow = gapOf(pulls.widestNow), gapNext = gapOf(pulls.widestNext), slide = next ? smoothstep01(span(t, LEAVE_FROM, LEAVE_TO)) : 0;
    const first = Math.max(1, i - WINDOW + 1), newest = next && t >= LAND ? i + 1 : i;
    let n = 0, ringX = 0, ringY = 0, ringPay = 0, ringFound = false, flyX = 0, flyY = 0;
    for (let s = first; s <= (next ? i + 1 : i); s++) {
      if (!pulls.has(s)) continue;
      const slot = s % RING, arm = pulls.arm[slot]!, pay = pulls.pay[slot]!, side = sideOf(s);
      const inNow = s <= i, inNext = next !== null && s > i - WINDOW + 1;
      const offset = inNow && inNext ? lerp((pulls.now[slot]! + 0.5) * gapNow, (pulls.next[slot]! + 0.5) * gapNext, slide) : inNow ? (pulls.now[slot]! + 0.5) * gapNow : (pulls.next[slot]! + 0.5) * gapNext;
      const x = this.xOf(arm) + side * offset, y = (binOf(pay) + 0.5) * BIN * scale;
      if (s === i + 1) {
        flyX = x;
        flyY = y;
        if (t < LAND) continue;
      }
      if (s === newest) {
        ringX = x;
        ringY = y;
        ringPay = pay;
        ringFound = true;
      }
      const going = inNow && !inNext ? slide : 0, size = d * (1 - 0.7 * going);
      if (size <= 1e-4) continue;
      coins.setMatrixAt(n, this.matrix.makeScale(size, 1, size).setPosition(x, 0, -y));
      const age = i - s + t;
      coins.setColorAt(n, this.color.copy(pay >= 0 ? this.gain : this.loss).multiplyScalar((1 - AGE_DIM * clamp01(age / WINDOW)) * (1 - going)));
      n++;
    }
    coins.count = n;
    coins.instanceMatrix.needsUpdate = true;
    coins.instanceColor!.needsUpdate = true;

    flyer.mesh.visible = flyer.trail.visible = next !== null && t >= LAUNCH && t < LAND && pulls.has(i + 1);
    if (next && flyer.mesh.visible) {
      const x0 = this.xOf(banditArm(next)), y0 = -REACH + 0.05;
      const along = (u: number, out: THREE.Vector3) => out.set((1 - u * u) * x0 + u * u * flyX, 0, -((1 - u) * (1 - u) * y0 + (1 - (1 - u) * (1 - u)) * flyY));
      const u = easeOut(span(t, LAUNCH, LAND)), head = along(u, this.point), grow = 1.9 - 0.9 * u;
      flyer.mesh.position.copy(head);
      flyer.mesh.scale.set(d * grow * (0.2 + 0.8 * Math.abs(Math.cos(SPIN * Math.PI * u))), 1, d * grow);
      const tail = along(easeOut(span(t - TAIL, LAUNCH, LAND)), this.behind), dx = head.x - tail.x, dz = head.z - tail.z, length = Math.hypot(dx, dz);
      flyer.trail.position.set((head.x + tail.x) / 2, 0, (head.z + tail.z) / 2);
      flyer.trail.rotation.y = Math.atan2(-dz, dx);
      flyer.trail.scale.set(Math.max(1e-3, length), 1, d * grow * 0.7);
      this.color.copy(next.reward >= 0 ? this.gain : this.loss).multiplyScalar(1.5);
      flyer.tint.value.set(this.color.r, this.color.g, this.color.b);
      flyer.streak.value.set(this.color.r * 0.8, this.color.g * 0.8, this.color.b * 0.8);
      flyer.light.value.set(this.color.r * 12, this.color.g * 12, this.color.b * 12);
    } else flyer.light.value.set(0, 0, 0);

    const ringAge = newest === i + 1 ? t - LAND : t + 1 - LAND;
    ring.mesh.visible = !fast && ringFound && newest >= 1 && ringAge >= 0;
    if (ring.mesh.visible) {
      const pulse = (1 - clamp01(ringAge / RING_LIFE)) ** 2;
      ring.mesh.position.set(ringX, 0, -ringY);
      ring.mesh.scale.setScalar(d * (1.7 + 1.8 * pulse));
      this.color.copy(ringPay >= 0 ? this.gain : this.loss).multiplyScalar(1.4);
      ring.tint.value.set(this.color.r, this.color.g, this.color.b);
      ring.fade.value = 0.55 + 0.45 * pulse;
    }

    let favored = 0;
    for (let a = 1; a < this.arms; a++) if (pulls.counts[a]! > pulls.counts[favored]!) favored = a;
    const a0 = fast ? favored : i >= 1 ? banditArm(head) : 0, a1 = next ? banditArm(next) : a0;
    let pull = 0;
    if (next) pull = t < GLIDE ? 0 : t < PULL ? smoothstep01(span(t, GLIDE, PULL)) : 1 - easeOut(span(t, PULL, RETURN));
    this.flashes.fill(0);
    if (i >= 1 && !fast) this.flashes[a0] = Math.exp(-(t + 1 - PULL) / FLASH_LIFE);
    if (next && t >= PULL) this.flashes[a1] = Math.max(this.flashes[a1]!, Math.exp(-(t - PULL) / FLASH_LIFE));
    for (let a = 0; a < this.arms; a++) {
      const machine = this.machines[a]!, share = pulls.total > 0 ? pulls.counts[a]! / pulls.total : 0, flash = this.flashes[a]!;
      this.setLever(machine, a === a1 ? THROW * pull : 0);
      machine.glow.value = 0.3 + 0.8 * share + 1.6 * flash;
      const lamp = 2 + 12 * share + 30 * flash;
      machine.light.value.set(this.led.r * lamp, this.led.g * lamp, this.led.b * lamp);
      machine.needle.position.z = -this.means[a]! * scale;
      const bright = a === best ? 1.5 : 0.9;
      this.color.set(TRUTH).multiplyScalar(bright);
      machine.needleTint.value.set(this.color.r, this.color.g, this.color.b);
    }

    const rest = this.knobY(0), glide = smoothstep01(span(t, 0, GLIDE)), fromX = i >= 1 || fast ? this.xOf(a0) + LEVER_X : 0, toX = this.xOf(a1) + LEVER_X;
    if (next) this.orbX = lerp(fromX, toX, glide);
    else this.orbX = frame.scrubbed ? fromX : this.orbX + (fromX - this.orbX) * (1 - Math.exp(-18 * dt));
    const hop = next && fromX !== toX && t < GLIDE ? Math.sin(Math.PI * glide) * HOP : 0;
    const orbAngle = next && t >= GLIDE ? THROW * pull : 0;
    orb.mesh.position.set(this.orbX, 0, -(t >= GLIDE && next ? this.knobY(orbAngle) : rest + hop));
    orb.mesh.scale.setScalar(1 + 0.35 * Math.sin(orbAngle));
    this.color.copy(this.accent).multiplyScalar(24 + 16 * pull);
    orb.light.value.set(this.color.r, this.color.g, this.color.b);
    orb.strength.value = 1.3 + 0.8 * pull;

    const crownTarget = this.xOf(best);
    this.crownX = frame.scrubbed ? crownTarget : this.crownX + (crownTarget - this.crownX) * (1 - Math.exp(-14 * dt));
    this.flare = frame.scrubbed ? 0 : this.flare * Math.exp(-4 * dt);
    if (frame.scrubbed) pool.clear();
    else for (const event of frame.events) {
      if (event.type !== 'cue' || event.value < 0 || event.value >= this.arms) continue;
      pool.ripple(this.point.set(this.xOf(event.value), 0, -this.means[event.value]! * scale), TRUTH, 0.4, time);
      pool.burst(this.point.set(this.xOf(event.value), 0, -CROWN), TRUTH, 0.6, time);
      this.flare = 1;
    }
    crown.mesh.position.set(this.crownX, 0, -CROWN - 0.25 * clamp01(Math.abs(crownTarget - this.crownX) / PITCH));
    crown.mesh.scale.setScalar(1 + 0.5 * this.flare);
    crown.light.position.copy(crown.mesh.position);

    const pulled = banditArm(head);
    this.ledger.step = i;
    this.ledger.arm = pulled;
    this.ledger.reward = head.reward;
    const before = i >= 1 ? at(i - 1) : null;
    this.ledger.mean = before ? wordFloat(before.words[pulled]!) : this.means[pulled]!;
    this.ledger.best = best;
    this.ledger.top = this.means[best]!;
    this.ledger.total = pulls.total;
    this.ledger.share = pulls.total > 0 ? pulls.counts[best]! / pulls.total : NaN;
  }
}

export default (() => new BanditScene()) satisfies SceneFactory;
