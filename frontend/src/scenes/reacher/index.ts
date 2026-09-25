import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugCurve, DebugKit, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { REACHER_ELBOW, REACHER_SHOULDER, REACHER_TARGET_X, REACHER_TARGET_Y } from '@sim/envs/planar';
import type { EnvFrame } from '@sim/types';
import { constantRecord } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { PALETTE, SCALE, SWEEP, armMaterial, dims, disc, flat, markMaterial, ribbon, tableMaterial, targetMaterial, tipMaterial, type ArmRig, type Dims, type Ribbon, type TargetRig } from './art';

const PACE = 20;
const SUB = 6;
const TRAIL = 64;
const CAPACITY = (TRAIL + 2) * SUB;
const GHOST = 8;
const SPAWN = 4;
const BURSTS = 2;
const INK = 0.0018;
const GLOW = 5;
const LIFT = 30;
const COUNTER = 30;
const ROUND = 31;
const GAP = 8;
const ACCENT = TOKENS.accents.Reacher!;
const TRUTH = { arm: 0x101418, target: 0xb87800, sensed: 0xe0561f, reach: 0x3f6488, stop: 0xd11f1f, torque: 0xc2410c, path: 0x0069a8 } as const;

interface Pose { elbow: THREE.Vector2; end: THREE.Vector2; tip: THREE.Vector2; heading: THREE.Vector2 }

const pose = (): Pose => ({ elbow: new THREE.Vector2(), end: new THREE.Vector2(), tip: new THREE.Vector2(), heading: new THREE.Vector2() });
const pin = (): THREE.UniformNode<'vec2', THREE.Vector2> => uniform(new THREE.Vector2());
const clampUnit = (value: number): number => Math.max(-1, Math.min(1, value));
const gapOf = (frame: EnvFrame): number => Math.hypot(frame.observations[GAP]!, frame.observations[GAP + 1]!);
const centimetres = (metres: number): string => Number.isFinite(metres) ? `${(100 * metres).toFixed(1)} cm` : '—';
const torque = (value: number): string => Number.isFinite(value) ? `${value >= 0 ? '↺' : '↻'} ${Math.abs(value).toFixed(1)} N·m` : '—';
const signed = (value: number): string => formatSigned(value).replace('-', '−');

function circle(curve: DebugCurve, x: number, y: number, radius: number, from = 0, sweep = 2 * Math.PI): void {
  const count = curve.capacity - 1;
  for (let i = 0; i <= count; i++) {
    const a = from + sweep * i / count;
    curve.point((x + radius * Math.cos(a)) * SCALE, 0.06, -(y + radius * Math.sin(a)) * SCALE);
  }
}

class ReacherScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = PACE;
  protected readonly stageName = 'reacher';

  private d!: Dims;
  private gear = 1;
  private steps = 0;
  private step = -1;
  private clock = 0;
  private ink: Ribbon | null = null;
  private fading: Ribbon | null = null;
  private readonly rig = new THREE.Group();
  private readonly shades: THREE.Mesh[] = [];
  private lamp: THREE.Mesh | null = null;
  private readonly dpr = uniform(1);
  private readonly glow = uniform(0);
  private readonly live = pose();
  private readonly was = pose();
  private readonly probe = pose();
  private readonly trail = new Float32Array(2 * CAPACITY);
  private readonly faded = new Float32Array(2 * CAPACITY);
  private trailCount = 0;
  private readonly spot = new THREE.Vector3();
  private readonly arm: ArmRig = { elbow: pin(), end: pin(), tip: pin(), was: [pin(), pin(), pin()], ghost: uniform(0), torque: pin(), heading: pin(), stop: uniform(0) };
  private readonly target: TargetRig = { at: pin(), old: pin(), tip: pin(), spawn: uniform(1), fade: uniform(0), won: uniform(0), round: uniform(0), contact: uniform(0), clock: uniform(0) };
  private readonly ledger = { gap: 0, closest: 0, shoulder: NaN, elbow: NaN, stop: false, reward: 0, round: 0, count: 0, progress: 0 };

  override legend: LegendSpec = {
    title: 'REACHER',
    rows: [
      { key: 'gap', label: 'to target', tint: PALETTE.target, value: () => centimetres(this.ledger.gap) },
      { key: 'closest', label: 'closest', tint: ACCENT, value: () => centimetres(this.ledger.closest) },
      { key: 'shoulder', label: 'shoulder', tint: PALETTE.torque, value: () => torque(this.ledger.shoulder) },
      { key: 'elbow', label: 'elbow', tint: PALETTE.torque, value: () => `${torque(this.ledger.elbow)}${this.ledger.stop ? ' · at its stop' : ''}` },
      { key: 'reward', label: 'reward', tint: PALETTE.gold, value: () => signed(this.ledger.reward) },
    ],
    note: () => `round ${formatCount(this.ledger.round)}${this.steps > 0 ? ` · ${this.ledger.count}/${this.steps}` : ''}`,
    bar: () => this.ledger.progress,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const arm = constantRecord(ctx.environment, 'PLANAR_ARM');
    const scalar = (key: string): number => {
      const value = arm[key];
      if (typeof value !== 'number') throw new Error(`${ctx.envId} PLANAR_ARM lacks ${key}`);
      return value;
    };
    const centre = arm.center;
    if (!Array.isArray(centre) || centre.length !== 3) throw new Error(`${ctx.envId} PLANAR_ARM lacks its three body centres`);
    this.d = dims(scalar('link'), 2 * centre[1]!, centre[2]!, scalar('range'));
    this.gear = scalar('gear');
    const half = (this.d.plate + this.d.rail + 0.1 * this.d.link) * SCALE;
    return new THREE.Box2(new THREE.Vector2(-half, -half), new THREE.Vector2(half, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff4e8).multiplyScalar(0.96);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const d = this.d, lit = light.lit, span = 2 * (d.plate + d.rail) * SCALE, wide = span + 8 * d.ring * SCALE;
    this.dpr.value = ctx.size.dpr;
    this.scene.background = new THREE.Color(PALETTE.desk);
    this.scene.add(this.rig);
    const place = (mesh: THREE.Mesh, order: number): void => {
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.rig.add(mesh);
    };
    place(new THREE.Mesh(flat(400, 400), tableMaterial(d, this.dpr, lit)), 0);
    this.fading = ribbon(CAPACITY, PALETTE.ink, uniform(0), lit);
    this.ink = ribbon(CAPACITY, PALETTE.ink, uniform(1), lit);
    place(this.fading.mesh, 1);
    place(this.ink.mesh, 2);
    place(new THREE.Mesh(flat(wide, wide), markMaterial(d, this.target, this.dpr)), 3);
    place(new THREE.Mesh(flat(span, span), armMaterial(d, this.arm, this.dpr)), 4);
    place(new THREE.Mesh(flat(wide, wide), targetMaterial(d, this.target, this.dpr)), 5);
    place(new THREE.Mesh(flat(span, span), tipMaterial(d, this.arm, ACCENT, this.dpr)), 6);

    const bar = (length: number) => flat(length * SCALE, 2 * d.rod * SCALE);
    for (const geometry of [bar(d.link), bar(d.forearm), disc(d.hub * SCALE), disc(d.base * SCALE)]) {
      const shade = light.occluder(geometry);
      shade.renderOrder = 1;
      this.shades.push(shade);
      this.rig.add(shade);
    }
    this.shades[3]!.position.set(0, 0.01, 0);
    const lamp = light.emitter(disc(d.target * SCALE), rgb(PALETTE.target).mul(this.glow));
    lamp.renderOrder = 2;
    this.lamp = lamp;
    this.rig.add(lamp);
    pool.group.traverse((object) => { object.renderOrder = 20; });
  }

  private solve(shoulder: number, elbow: number, out: Pose): Pose {
    const d = this.d, turn = shoulder + elbow, c = Math.cos(turn), s = Math.sin(turn);
    out.elbow.set(d.link * Math.cos(shoulder), d.link * Math.sin(shoulder));
    out.end.set(out.elbow.x + d.forearm * c, out.elbow.y + d.forearm * s);
    out.tip.set(out.elbow.x + d.reach * c, out.elbow.y + d.reach * s);
    out.heading.set(shoulder, turn);
    return out;
  }

  private world(x: number, y: number, lift: number): THREE.Vector3 {
    return this.spot.set(x * SCALE, lift, -y * SCALE);
  }

  private lay(mesh: THREE.Mesh, ax: number, ay: number, bx: number, by: number): void {
    mesh.position.set((ax + bx) / 2 * SCALE, 0.01, -(ay + by) / 2 * SCALE);
    mesh.rotation.y = Math.atan2(by - ay, bx - ax);
  }

  private trace(frame: FrameUpdate, from: number, to: number, tail: Float32Array | null, out: Float32Array): number {
    let count = 0, last0 = NaN, last1 = NaN;
    const put = (shoulder: number, elbow: number): void => {
      if (count >= CAPACITY) return;
      const tip = this.solve(shoulder, elbow, this.probe).tip;
      out[2 * count] = tip.x;
      out[2 * count + 1] = tip.y;
      count++;
    };
    const walk = (shoulder: number, elbow: number): void => {
      if (!Number.isNaN(last0)) for (let k = 1; k < SUB; k++) put(last0 + (shoulder - last0) * k / SUB, last1 + (elbow - last1) * k / SUB);
      put(shoulder, elbow);
      last0 = shoulder;
      last1 = elbow;
    };
    for (let s = Math.max(from, to - TRAIL); s <= to; s++) {
      if (frame.frameAt(s).step !== s) {
        count = 0;
        last0 = last1 = NaN;
        continue;
      }
      const q = frame.poseAt(s);
      walk(q[REACHER_SHOULDER]!, q[REACHER_ELBOW]!);
    }
    if (tail && !Number.isNaN(last0)) walk(tail[REACHER_SHOULDER]!, tail[REACHER_ELBOW]!);
    return count;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const ink = this.ink, fading = this.fading, lamp = this.lamp;
    if (!ink || !fading || !lamp) return;
    pool.update(frame.dtReal, time);
    this.clock += frame.dtReal;
    const d = this.d, state = frame.state, step = state.step, fraction = Math.min(1, Math.max(0, frame.position - step));
    const now = frame.frameAt(step), count = now.words[COUNTER]!, start = step - count, since = count + fraction;
    if (this.steps === 0 && start >= 1) {
      const last = frame.frameAt(start - 1);
      if (last.step === start - 1) this.steps = last.words[COUNTER]! + 1;
    }

    const live = this.solve(state.pose[REACHER_SHOULDER]!, state.pose[REACHER_ELBOW]!, this.live);
    const tx = state.pose[REACHER_TARGET_X]!, ty = state.pose[REACHER_TARGET_Y]!;
    const arm = this.arm, target = this.target;
    arm.elbow.value.copy(live.elbow);
    arm.end.value.copy(live.end);
    arm.tip.value.copy(live.tip);
    arm.heading.value.copy(live.heading);
    const next = frame.frameAt(step + 1), acting = next.step === step + 1;
    const shoulderTorque = acting ? clampUnit(next.actions[0]!) : 0, elbowTorque = acting ? clampUnit(next.actions[1]!) : 0;
    arm.torque.value.set(shoulderTorque, elbowTorque);
    const stop = Math.abs(state.pose[REACHER_ELBOW]!) > d.range;
    arm.stop.value = stop ? 1 : 0;

    this.lay(this.shades[0]!, 0, 0, live.elbow.x, live.elbow.y);
    this.lay(this.shades[1]!, live.elbow.x, live.elbow.y, live.end.x, live.end.y);
    this.shades[2]!.position.copy(this.world(live.elbow.x, live.elbow.y, 0.01));

    const before = start >= 1 ? frame.frameAt(start - 1) : null, known = before !== null && before.step === start - 1;
    const fade = known && since < GHOST ? (1 - since / GHOST) ** 2 : 0;
    target.won.value = known && gapOf(before) < d.touch ? 1 : 0;
    arm.ghost.value = fade;
    target.fade.value = fade;
    fading.level.value = fade;
    if (known) {
      const q = frame.poseAt(start - 1), was = this.solve(q[REACHER_SHOULDER]!, q[REACHER_ELBOW]!, this.was);
      arm.was[0].value.copy(was.elbow);
      arm.was[1].value.copy(was.end);
      arm.was[2].value.copy(was.tip);
      target.old.value.set(q[REACHER_TARGET_X]!, q[REACHER_TARGET_Y]!);
      if (fade > 0) fading.draw(this.faded, this.trace(frame, start - 1 - before.words[COUNTER]!, start - 1, null, this.faded), INK);
    }
    fading.mesh.visible = fade > 0;
    this.trailCount = this.trace(frame, start, step, state.pose, this.trail);
    ink.draw(this.trail, this.trailCount, INK);

    const reach = Math.hypot(live.tip.x - tx, live.tip.y - ty);
    const contact = Math.min(1, Math.max(0, (d.touch * 1.1 - reach) / (d.touch * 0.2)));
    target.at.value.set(tx, ty);
    target.tip.value.copy(live.tip);
    target.spawn.value = Math.min(1, since / SPAWN);
    target.contact.value = contact;
    target.round.value = this.steps > 0 ? Math.min(1, since / this.steps) : 0;
    target.clock.value = this.clock;
    lamp.position.copy(this.world(tx, ty, 0.02));
    this.glow.value = GLOW * (1 + 1.6 * contact);

    if (frame.scrubbed || this.step < 0 || step < this.step) pool.clear();
    else this.celebrate(frame, this.step, step, pool, time);
    this.step = step;

    const ledger = this.ledger;
    ledger.gap = gapOf(now);
    ledger.closest = ledger.gap;
    for (let s = Math.max(start, step - TRAIL); s < step; s++) {
      const f = frame.frameAt(s);
      if (f.step === s) ledger.closest = Math.min(ledger.closest, gapOf(f));
    }
    ledger.shoulder = acting ? this.gear * shoulderTorque : NaN;
    ledger.elbow = acting ? this.gear * elbowTorque : NaN;
    ledger.stop = stop;
    ledger.reward = state.reward;
    ledger.round = now.words[ROUND]!;
    ledger.count = count;
    ledger.progress = target.round.value;
  }

  private celebrate(frame: FrameUpdate, from: number, to: number, pool: CelebrationPool, time: number): void {
    const d = this.d;
    let fired = 0;
    for (let s = from + 1; s <= to && fired < BURSTS; s++) {
      const f = frame.frameAt(s), b = frame.frameAt(s - 1);
      if (f.step !== s || b.step !== s - 1) continue;
      if (f.done) {
        const q = frame.poseAt(s - 1), x = q[REACHER_TARGET_X]!, y = q[REACHER_TARGET_Y]!;
        if (gapOf(b) < d.touch) {
          pool.burst(this.world(x, y, LIFT), PALETTE.amber, 0.7, time);
          fired++;
        }
      } else if (gapOf(f) < d.touch && gapOf(b) >= d.touch) {
        const q = frame.poseAt(s), x = q[REACHER_TARGET_X]!, y = q[REACHER_TARGET_Y]!;
        pool.burst(this.world(x, y, LIFT), PALETTE.amber, 0.45, time);
        fired++;
      }
    }
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const d = this.d, state = frame.state, live = this.solve(state.pose[REACHER_SHOULDER]!, state.pose[REACHER_ELBOW]!, this.probe), y = 0.06;
    const at = (x: number, z: number, curve: { point(x: number, y: number, z: number): void }): void => curve.point(x * SCALE, y, -z * SCALE);
    const arm = kit.curve('arm', 4, TRUTH.arm);
    at(0, 0, arm);
    at(live.elbow.x, live.elbow.y, arm);
    at(live.end.x, live.end.y, arm);
    at(live.tip.x, live.tip.y, arm);
    const joints = kit.marks('joints', 3);
    joints.set(0, y, 0, 0.1, TRUTH.arm);
    joints.set(live.elbow.x * SCALE, y, -live.elbow.y * SCALE, 0.1, TRUTH.arm);
    joints.set(live.tip.x * SCALE, y, -live.tip.y * SCALE, 0.08, TRUTH.arm);
    const tx = state.pose[REACHER_TARGET_X]!, ty = state.pose[REACHER_TARGET_Y]!, sx = tx + state.observations[GAP]!, sy = ty + state.observations[GAP + 1]!;
    const marks = kit.marks('target', 2);
    marks.set(tx * SCALE, y, -ty * SCALE, 0.1, TRUTH.target);
    marks.set(sx * SCALE, y, -sy * SCALE, 0.07, TRUTH.sensed);
    const gap = kit.curve('gap', 2, TRUTH.sensed);
    at(sx, sy, gap);
    at(tx, ty, gap);
    circle(kit.curve('touch', 49, TRUTH.target), tx, ty, d.touch);
    circle(kit.curve('reach', 145, TRUTH.reach), 0, 0, d.link + d.reach);
    const stops = kit.curve('stops', 3, TRUTH.stop);
    at(live.elbow.x + d.forearm * Math.cos(live.heading.x + d.range), live.elbow.y + d.forearm * Math.sin(live.heading.x + d.range), stops);
    at(live.elbow.x, live.elbow.y, stops);
    at(live.elbow.x + d.forearm * Math.cos(live.heading.x - d.range), live.elbow.y + d.forearm * Math.sin(live.heading.x - d.range), stops);
    const push = this.arm.torque.value;
    circle(kit.curve('shoulderTorque', 33, TRUTH.torque), 0, 0, d.shoulderArc, live.heading.x, push.x * SWEEP);
    circle(kit.curve('elbowTorque', 33, TRUTH.torque), live.elbow.x, live.elbow.y, d.elbowArc, live.heading.y, push.y * SWEEP);
    const path = kit.curve('path', CAPACITY, TRUTH.path);
    for (let i = 0; i < this.trailCount; i++) at(this.trail[2 * i]!, this.trail[2 * i + 1]!, path);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    this.dpr.value = size.dpr;
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.shades.length = 0;
    this.ink = this.fading = null;
    this.lamp = null;
  }
}

export default (() => new ReacherScene()) satisfies SceneFactory;
