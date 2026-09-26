import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { nextingCall, nextingPhase } from '@sim/envs/nexting';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';
import {
  AMBER, BEAM_AT, BODY, CORAL, HUB, LAMP, LAMP_SIZE, LAPS, PEN, PITCH, PLATE, R0, ROBOT, RT, SENSE, TAU, TRACK, TRAIL, UNIT, VOID, WALL,
  disc, flat, groundMaterial, hubMaterial, lampMaterial, penFrame, recordMaterial, robotMaterial,
} from './art';

const PILOT = 7;
const FLASH = 42;
const FLASH_STEPS = 1.6;
const BEAM = 24;
const BEAM_SECONDS = 0.04;
const SWING = 0.55;

const lampTint = new THREE.Color(AMBER);

class NextingScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'nexting';
  override legend: LegendSpec | null = null;

  private period = 1;
  private horizon = 1;
  private ring = 1;
  private window = 1;
  private texels = new Float32Array(4);
  private tags = new Float64Array(1);
  private history = new THREE.DataTexture(this.texels, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly marks = new Float32Array(TRAIL * 4);
  private readonly trail = new THREE.DataTexture(this.marks, TRAIL, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly head = uniform(0);
  private readonly phase = uniform(0);
  private readonly alpha = uniform(0);
  private readonly tip = uniform(new THREE.Vector2(0, 1));
  private readonly pulse = uniform(0);
  private readonly beam = uniform(0);
  private readonly glow = uniform(0);
  private readonly lampLight = uniform(new THREE.Vector3());
  private readonly beamLight = uniform(new THREE.Vector3());
  private readonly rover = new THREE.Group();
  private readonly accent = new THREE.Color();
  private readonly point = new THREE.Vector3();
  private beamLevel = 0;
  private statsAt = -1;
  private statsDone = false;
  private readonly ledger = { step: 0, phase: 0, call: -1, reward: 0, seen: 0, right: 0, pulses: 0, caught: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.period = constant(ctx.environment, 'NEXTING_PERIOD');
    this.horizon = parameter(ctx.environment, 'horizon');
    this.window = LAPS * this.period;
    this.ring = this.window + 4;
    this.texels = new Float32Array(this.ring * 4);
    this.tags = new Float64Array(this.ring).fill(NaN);
    this.history = new THREE.DataTexture(this.texels, this.ring, 1, THREE.RGBAFormat, THREE.FloatType);
    const half = PEN + WALL + 0.1;
    return new THREE.Box2(new THREE.Vector2(-half, -half), new THREE.Vector2(half, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb4c0d8).multiplyScalar(0.62);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 44, sighSize: 26, sighRadius: 0.62, sighHeight: 1.1 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit;
    this.accent.set(accent);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(VOID);
    pool.group.renderOrder = 10;
    for (const texture of [this.history, this.trail]) {
      texture.magFilter = texture.minFilter = THREE.NearestFilter;
      texture.needsUpdate = true;
    }
    const place = (mesh: THREE.Mesh, order: number) => {
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      return mesh;
    };
    const ground = place(new THREE.Mesh(flat(90, 90), groundMaterial(lit, this.period, this.horizon)), 0);
    const record = place(new THREE.Mesh(disc(PLATE + 0.02), recordMaterial(lit, accent, this.period, this.horizon, { head: this.head, phase: this.phase, alpha: this.alpha, history: this.history, ring: this.ring })), 1);
    const hub = place(new THREE.Mesh(disc(HUB + 0.02), hubMaterial(lit, accent, { tip: this.tip, pulse: this.pulse, trail: this.trail })), 2);
    const lamp = place(new THREE.Mesh(flat(LAMP_SIZE, LAMP_SIZE), lampMaterial(lit, this.glow)), 3);
    lamp.position.set(0, 0, -LAMP);
    const robot = place(new THREE.Mesh(flat(ROBOT, ROBOT), robotMaterial(lit, accent, this.beam)), 4);
    const body = light.occluder(disc(BODY - 0.01));
    body.position.y = 0.01;
    const headlight = light.emitter(flat(0.07, 0.34), this.beamLight);
    headlight.position.set(BEAM_AT, 0.02, 0);
    this.rover.add(robot, body, headlight);
    const bulb = light.emitter(disc(LAMP_SIZE * 0.31), this.lampLight);
    bulb.position.set(0, 0.02, -LAMP);
    const walls = light.occluder(penFrame());
    walls.position.y = 0.01;
    this.scene.add(ground, record, hub, lamp, this.rover, bulb, walls);
  }

  protected override drop(): void {
    this.history.dispose();
    this.trail.dispose();
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

  private due(phase: number): number {
    return phase + this.horizon >= this.period ? 1 : 0;
  }

  private sync(frame: FrameUpdate, i: number): void {
    let dirty = false;
    for (let s = i - this.window; s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] === s) continue;
      const now = s >= 1 ? this.at(frame, s) : null, before = now ? this.at(frame, s - 1) : null;
      if (!now || !before) {
        if (this.texels[o + 2] !== 0) dirty = true;
        this.texels.fill(0, o, o + 4);
        this.tags[slot] = s < 1 ? s : NaN;
        continue;
      }
      const q = nextingPhase(before), call = nextingCall(now), right = now.reward > 0 ? 1 : 0, p = nextingPhase(now);
      if (p !== (q + 1) % this.period || right !== (call === this.due(q) ? 1 : 0) || (now.observations[2] === 1) !== (p === 0)) {
        throw new Error(`Nexting step ${s} went from phase ${q} to ${p} calling ${call} for reward ${now.reward} with pulse ${now.observations[2]}`);
      }
      this.texels[o] = call;
      this.texels[o + 1] = right;
      this.texels[o + 2] = 1;
      this.texels[o + 3] = q;
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
    const p = nextingPhase(now), angle = TAU * (p + alpha) / this.period;
    this.rover.position.set(RT * Math.sin(angle), 0, -RT * Math.cos(angle));
    this.rover.rotation.y = -angle;
    this.head.value = i % this.ring;
    this.phase.value = p;
    this.alpha.value = alpha;

    const call = next ? nextingCall(next) : 0;
    this.beamLevel = frame.scrubbed ? call : call + (this.beamLevel - call) * Math.exp(-frame.dtReal / BEAM_SECONDS);
    this.beam.value = this.beamLevel;
    const beam = BEAM * this.beamLevel;
    this.beamLight.value.set(this.accent.r * beam, this.accent.g * beam, this.accent.b * beam);
    const flash = Math.exp(-(p + alpha) / FLASH_STEPS), lamp = PILOT + FLASH * flash;
    this.glow.value = flash;
    this.lampLight.value.set(lampTint.r * lamp, lampTint.g * lamp, lampTint.b * lamp);
    this.pulse.value = now.observations[2]! * (1 - smoothstep01(alpha));

    this.sense(frame, i, now, next, alpha);
    this.celebrate(frame, pool, time);
    this.tally(i, now, next);
  }

  private sense(frame: FrameUpdate, i: number, now: EnvFrame, next: EnvFrame | null, alpha: number): void {
    const k = next ? smoothstep01((alpha - SWING) / (1 - SWING)) : 0, a = now.observations, b = next ? next.observations : a;
    this.tip.value.set(a[0]! + (b[0]! - a[0]!) * k, a[1]! + (b[1]! - a[1]!) * k);
    for (let m = 0; m < TRAIL; m++) {
      const seen = this.at(frame, i - m), o = m * 4;
      if (!seen) {
        this.marks.fill(0, o, o + 4);
        continue;
      }
      this.marks[o] = seen.observations[0]!;
      this.marks[o + 1] = seen.observations[1]!;
      this.marks[o + 2] = m + alpha;
      this.marks[o + 3] = 1;
    }
    this.trail.needsUpdate = true;
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    for (const event of frame.events) {
      if (event.type === 'goal') {
        pool.burst(this.point.set(0, 38, -LAMP), TOKENS.celebrationGold, 1.1, time);
        pool.ripple(this.point.set(0, 0, -LAMP), TOKENS.celebrationGold, 0.34, time);
      } else if (event.type === 'harm') pool.sigh(0.45, this.point.set(0, 47.4, -LAMP), time);
    }
  }

  private tally(i: number, now: EnvFrame, next: EnvFrame | null): void {
    const l = this.ledger;
    l.step = i;
    l.phase = nextingPhase(now);
    l.call = next ? nextingCall(next) : -1;
    l.reward = now.reward;
    if (this.statsAt === i && this.statsDone) return;
    let seen = 0, right = 0, pulses = 0, caught = 0;
    for (let s = Math.max(1, i - this.window + 1); s <= i; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] !== s) continue;
      seen++;
      right += this.texels[o + 1]!;
      if (this.texels[o + 3] === this.period - 1) {
        pulses++;
        caught += this.texels[o + 1]!;
      }
    }
    Object.assign(l, { seen, right, pulses, caught });
    this.statsAt = i;
    this.statsDone = seen === Math.min(this.window, i);
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, span = `${this.horizon} step${this.horizon === 1 ? '' : 's'}`;
    return {
      title: 'NEXTING',
      rows: [
        { key: 'call', label: 'prediction', tint: accent, value: () => (l.call < 0 ? '—' : `${l.call ? 'pulse' : 'none'} within ${span}`) },
        { key: 'pulse', label: 'next pulse', tint: AMBER, value: () => (l.phase === 0 ? 'now' : `in ${formatCount(this.period - l.phase)} steps`) },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
        { key: 'right', label: `right, last ${formatCount(this.window)}`, tint: accent, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
        { key: 'caught', label: 'pulses caught', tint: TOKENS.celebrationGold, value: () => (l.pulses > 0 ? `${l.caught} of ${l.pulses}` : '—') },
      ],
      note: () => `horizon ${this.horizon} · pulse every ${this.period}`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private spot(phase: number, radius: number): THREE.Vector3 {
    const angle = TAU * phase / this.period;
    return this.point.set(radius * Math.sin(angle), 0.06, -radius * Math.cos(angle));
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const i = frame.state.step, now = frame.frameAt(i), p = nextingPhase(now), x = p + this.alpha.value, accent = this.accent.getHex();
    const robot = this.spot(p, RT);
    kit.marks('robot', 1).set(robot.x, robot.y, robot.z, 0.14, 0xffffff);
    kit.marks('pulse', 1).set(0, 0.06, -LAMP, now.observations[2] === 1 ? 0.26 : 0.08, AMBER);
    const zone = kit.curve('zone', 64, AMBER), edge = RT + TRACK, from = this.period - this.horizon;
    const inward = this.spot(from, HUB);
    zone.point(inward.x, inward.y, inward.z);
    for (let k = 0; k <= 32; k++) {
      const out = this.spot(from + (this.horizon * k) / 32, edge);
      zone.point(out.x, out.y, out.z);
    }
    const back = this.spot(this.period, HUB);
    zone.point(back.x, back.y, back.z);
    const groove = kit.curve('groove', this.window + 1, TOKENS.uiCyan);
    for (let age = 0; age <= this.window; age++) {
      const at = this.spot(x - age, R0 - (PITCH * age) / this.period);
      groove.point(at.x, at.y, at.z);
    }
    const calls = kit.marks('calls', this.window + 2);
    for (let s = Math.max(1, i + 1 - this.window); s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] !== s) continue;
      const call = this.texels[o]!, right = this.texels[o + 1]!, age = i + this.alpha.value - (s - 0.5);
      if ((call === 0 && right === 1) || age < 0) continue;
      const at = this.spot(this.texels[o + 3]! + 0.5, R0 - (PITCH * age) / this.period);
      calls.set(at.x, at.y, at.z, 0.05, right ? accent : CORAL);
    }
    const reading = kit.curve('reading', 2, SENSE), o = now.observations;
    reading.point(0, 0.06, 0);
    reading.point(o[0]! * UNIT, 0.06, -o[1]! * UNIT);
    const exact = this.spot(p, UNIT);
    kit.marks('truth', 1).set(exact.x, exact.y, exact.z, 0.06, AMBER);
  }
}

export default (() => new NextingScene()) satisfies SceneFactory;
