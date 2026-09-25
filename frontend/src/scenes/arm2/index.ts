import * as THREE from 'three/webgpu';
import { float, mix, uniform } from 'three/tsl';
import type { DebugCurve, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendRow, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { lastEvent } from '@sim/events';
import { ARM2_DISTANCE, ARM2_Q1, ARM2_Q2, ARM2_REGIME, ARM2_TARGET_X, ARM2_TARGET_Y } from '@sim/envs/arm2';
import { wordFloat } from '@sim/words';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { DPR, PALETTE, arc, backdrop, claw, flat, halo, hub, link, payload, reticle, ribbon, sprite, tether, type ArcControl } from './sprites';

const TRAIL = 40;
const GHOST_STEPS = 45;
const SWAP_STEPS = 40;
const SHOULDER = 0.14;
const ELBOW = 0.11;
const HALF = 0.07;
const WRIST = 0.1;
const PLATE = 0.16;
const SWEEP = 1.5 * Math.PI;
const HEAD = 0.055;
const MARGIN = 0.36;

interface Gauge {
  mesh: THREE.Mesh;
  control: ArcControl;
  radius: number;
  width: number;
}

const toWorld = (x: number, y: number, out: THREE.Vector3): THREE.Vector3 => out.set(x, 0, -y);

function lay(mesh: THREE.Object3D, ax: number, ay: number, bx: number, by: number): void {
  mesh.position.set((ax + bx) / 2, 0, -(ay + by) / 2);
  mesh.rotation.y = Math.atan2(by - ay, bx - ax);
}

function gauge(radius: number, thick: number, order: number): Gauge {
  const control: ArcControl = { axis: new THREE.Vector2(0, 1), aperture: new THREE.Vector2(0, 1), head: [new THREE.Vector2(), new THREE.Vector2(), new THREE.Vector2()] };
  const size = 2 * (radius + 0.07);
  return { mesh: sprite(size, size, order, arc(radius, thick, control)), control, radius, width: thick * 2.6 };
}

function aim(g: Gauge, x: number, y: number, start: number, fraction: number): void {
  g.mesh.visible = Math.abs(fraction) > 0.02;
  if (!g.mesh.visible) return;
  const sweep = fraction * SWEEP, middle = start + sweep / 2, end = start + sweep, turn = Math.sign(sweep) * HEAD / g.radius;
  g.mesh.position.set(x, 0, -y);
  g.control.axis.set(Math.sin(middle), Math.cos(middle));
  g.control.aperture.set(Math.sin(Math.abs(sweep) / 2), Math.cos(Math.abs(sweep) / 2));
  g.control.head[0].set(Math.cos(end), Math.sin(end)).multiplyScalar(g.radius - g.width);
  g.control.head[1].set(Math.cos(end), Math.sin(end)).multiplyScalar(g.radius + g.width);
  g.control.head[2].set(Math.cos(end + turn), Math.sin(end + turn)).multiplyScalar(g.radius);
}

function circle(curve: DebugCurve, x: number, y: number, radius: number, from = 0, sweep = 2 * Math.PI): void {
  const count = curve.capacity - 1;
  for (let i = 0; i <= count; i++) {
    const a = from + sweep * i / count;
    curve.point(x + radius * Math.cos(a), 0.05, -(y + radius * Math.sin(a)));
  }
}

const torque = (value: number): string => `${value >= 0 ? '↺' : '↻'} ${Math.abs(value).toFixed(1)} N·m`;
const clampUnit = (value: number): number => Math.max(-1, Math.min(1, value));

class WorkcellScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  controlHz = 50;
  protected readonly stageName = 'workcell';

  private l1 = 1;
  private l2 = 1;
  private reach = 2;
  private arrival = 0.1;
  private torques: readonly [number, number] = [1, 1];
  private masses: number[] = [0];
  private heaviest = 1;
  private regimes = 1;
  private readonly rig = new THREE.Group();
  private readonly parts = new Map<string, THREE.Mesh>();
  private gauges: [Gauge, Gauge] | null = null;
  private trail: ReturnType<typeof ribbon> | null = null;
  private readonly path = new Float32Array(TRAIL * 2);
  private readonly elbow = new THREE.Vector2();
  private readonly tip = new THREE.Vector2();
  private readonly bend = new THREE.Vector2();
  private readonly scratch = new THREE.Vector2();
  private readonly spot = new THREE.Vector3();
  private readonly effort = [uniform(0), uniform(0)] as const;
  private readonly closing = uniform(0.5);
  private readonly span = uniform(1);
  private readonly ghostAlpha = uniform(0);
  private readonly swapGlow = uniform(0);
  private readonly ledger = { gap: 0, shoulder: 0, elbow: 0, mass: 0, reward: 0, since: null as number | null };

  private forward(q1: number, q2: number, elbow: THREE.Vector2, tip: THREE.Vector2): void {
    elbow.set(this.l1 * Math.sin(q1), -this.l1 * Math.cos(q1));
    tip.set(elbow.x + this.l2 * Math.sin(q1 + q2), elbow.y - this.l2 * Math.cos(q1 + q2));
  }

  private massAt(regime: number): number {
    return this.masses[this.regimes > 1 ? regime : 0]!;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.controlHz = 1 / constant(env, 'ARM2_DT');
    this.l1 = constant(env, 'ARM2_LINK1');
    this.l2 = constant(env, 'ARM2_LINK2');
    this.reach = constant(env, 'ARM2_REACH');
    this.arrival = constant(env, 'ARM2_ARRIVAL');
    this.torques = [constant(env, 'ARM2_TORQUE1'), constant(env, 'ARM2_TORQUE2')];
    this.masses = constantList(env, 'ARM2_PAYLOAD');
    this.heaviest = Math.max(...this.masses);
    this.regimes = parameter(env, 'regimes');
    const half = this.reach + MARGIN;
    return new THREE.Box2(new THREE.Vector2(-half, -half), new THREE.Vector2(half, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa9bde3).multiplyScalar(0.95);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const elbow = new THREE.Vector2(), tip = new THREE.Vector2(), cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold;
    return {
      debug: (frame, kit) => {
        const pose = frame.state.pose, step = frame.state.step, q1 = pose[ARM2_Q1]!, q2 = pose[ARM2_Q2]!;
        this.forward(q1, q2, elbow, tip);
        const arm = kit.curve('arm', 3, cyan);
        arm.point(0, 0.05, 0);
        arm.point(elbow.x, 0.05, -elbow.y);
        arm.point(tip.x, 0.05, -tip.y);
        const joints = kit.marks('joints', 3);
        joints.set(0, 0.05, 0, 0.035, cyan);
        joints.set(elbow.x, 0.05, -elbow.y, 0.035, cyan);
        joints.set(tip.x, 0.05, -tip.y, 0.025, gold);
        const tx = pose[ARM2_TARGET_X]!, ty = pose[ARM2_TARGET_Y]!;
        circle(kit.curve('arrival', 49, gold), tx, ty, this.arrival);
        circle(kit.curve('workspace', 129, 0x8fa6c4), 0, 0, this.reach);
        const gap = kit.curve('gap', 2, gold);
        gap.point(tip.x, 0.05, -tip.y);
        gap.point(tx, 0.05, -ty);
        const actions = frame.frameAt(step + 1).actions;
        const start1 = Math.atan2(elbow.y, elbow.x), start2 = Math.atan2(tip.y - elbow.y, tip.x - elbow.x);
        circle(kit.curve('shoulderTorque', 33, 0xff9a4a), 0, 0, SHOULDER + 0.07, start1, clampUnit(actions[0]!) * SWEEP);
        circle(kit.curve('elbowTorque', 33, 0xff9a4a), elbow.x, elbow.y, ELBOW + 0.055, start2, clampUnit(actions[1]!) * SWEEP);
        const mass = this.massAt(frame.frameAt(step).words[ARM2_REGIME]!);
        if (mass > 0) circle(kit.curve('payload', 33, 0xd8a94f), tip.x, tip.y, PLATE * Math.sqrt(mass / this.heaviest));
        const trace = kit.curve('trace', TRAIL, cyan);
        for (let i = 0; i < TRAIL; i++) {
          const words = frame.frameAt(step - i).words;
          this.forward(wordFloat(words[ARM2_Q1]!), wordFloat(words[ARM2_Q2]!), elbow, tip);
          trace.point(tip.x, 0.05, -tip.y);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    DPR.value = ctx.size.dpr;
    this.scene.background = new THREE.Color(0x05080f);
    this.scene.add(this.rig);
    const accent = this.accentOf(ctx), lit = light.lit, size = 2 * (this.reach + MARGIN) * 3;
    const add = (name: string, mesh: THREE.Mesh): THREE.Mesh => {
      this.parts.set(name, mesh);
      this.rig.add(mesh);
      return mesh;
    };

    add('backdrop', sprite(size, size, 0, backdrop(this.reach, this.l1, lit)));
    const trail = ribbon(TRAIL, accent);
    trail.mesh.renderOrder = 1;
    this.trail = trail;
    this.rig.add(trail.mesh);
    add('ghost', sprite(0.5, 0.5, 2, reticle(this.arrival, false, this.ghostAlpha)));
    add('target', sprite(0.5, 0.5, 2, reticle(this.arrival, true, float(1))));
    const tetherMesh = add('tether', sprite(1, 0.08, 3, tether(this.span, this.arrival, mix(rgb(PALETTE.opening), rgb(PALETTE.closing), this.closing))));
    tetherMesh.geometry.translate(0.5, 0, 0);
    add('link1', sprite(this.l1 + 2 * HALF + 0.04, 2 * HALF + 0.04, 4, link(this.l1, HALF, lit)));
    add('shoulder', sprite(2 * SHOULDER + 0.04, 2 * SHOULDER + 0.04, 5, hub(SHOULDER, accent, lit)));
    add('payload', sprite(2.2, 2.2, 6, payload(lit, this.swapGlow)));
    add('swap', sprite(2.2, 2.2, 11, halo(PALETTE.brass, this.swapGlow)));
    add('link2', sprite(this.l2 - WRIST + 2 * HALF + 0.04, 2 * HALF + 0.04, 7, link(this.l2 - WRIST, HALF, lit)));
    add('claw', sprite(0.42, 0.22, 8, claw(WRIST, accent, lit)));
    add('elbow', sprite(2 * ELBOW + 0.04, 2 * ELBOW + 0.04, 9, hub(ELBOW, accent, lit)));
    this.gauges = [gauge(SHOULDER + 0.07, 0.018, 10), gauge(ELBOW + 0.055, 0.015, 10)];
    for (const g of this.gauges) this.rig.add(g.mesh);

    const disc = (radius: number) => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);
    const occluder = (name: string, geometry: THREE.BufferGeometry) => {
      const mesh = light.occluder(geometry);
      mesh.renderOrder = 1;
      add(name, mesh);
    };
    occluder('link1Shadow', flat(this.l1, 2 * HALF));
    occluder('link2Shadow', flat(this.l2 - WRIST, 2 * HALF));
    occluder('payloadShadow', disc(1));
    const emitter = (name: string, geometry: THREE.BufferGeometry, tint: THREE.Node<'vec3'>) => {
      const mesh = light.emitter(geometry, tint);
      mesh.renderOrder = 2;
      add(name, mesh);
    };
    emitter('shoulderGlow', disc(SHOULDER), rgb(PALETTE.effort).mul(this.effort[0].mul(3.5)));
    emitter('elbowGlow', disc(ELBOW), rgb(PALETTE.effort).mul(this.effort[1].mul(3.5)));
    emitter('targetGlow', disc(0.06), rgb(PALETTE.gold).mul(30));

    pool.group.traverse((object) => { object.renderOrder = 20; });
    this.legend = { title: 'WORKCELL', rows: this.rows(accent) };
  }

  private rows(accent: number): LegendRow[] {
    const ledger = this.ledger;
    const rows: LegendRow[] = [
      { key: 'gap', label: 'to target', tint: PALETTE.gold, value: () => `${ledger.gap.toFixed(2)} m` },
      { key: 'shoulder', label: 'shoulder', tint: PALETTE.effort, value: () => torque(ledger.shoulder) },
      { key: 'elbow', label: 'elbow', tint: PALETTE.effort, value: () => torque(ledger.elbow) },
    ];
    if (this.regimes > 1) rows.push({ key: 'payload', label: 'payload', tint: PALETTE.brass, value: () => `${ledger.mass.toFixed(2)} kg` });
    rows.push(
      { key: 'reach', label: 'last reach', tint: accent, value: () => ledger.since === null ? 'none yet' : `${formatCount(ledger.since)} steps ago` },
      { key: 'reward', label: 'reward', tint: PALETTE.gold, value: () => formatSigned(ledger.reward) },
    );
    return rows;
  }

  private part(name: string): THREE.Mesh {
    return this.parts.get(name)!;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const gauges = this.gauges, trail = this.trail;
    if (!gauges || !trail) return;
    const state = frame.state, pose = state.pose, step = state.step, fraction = frame.position - step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    const now = frame.frameAt(step), actions = frame.frameAt(step + 1).actions;
    const { elbow, tip } = this;
    this.forward(pose[ARM2_Q1]!, pose[ARM2_Q2]!, elbow, tip);
    const toward1 = Math.atan2(elbow.y, elbow.x), toward2 = Math.atan2(tip.y - elbow.y, tip.x - elbow.x);
    const wristX = elbow.x + (this.l2 - WRIST) * Math.cos(toward2), wristY = elbow.y + (this.l2 - WRIST) * Math.sin(toward2);

    lay(this.part('link1'), 0, 0, elbow.x, elbow.y);
    lay(this.part('link1Shadow'), 0, 0, elbow.x, elbow.y);
    lay(this.part('link2'), elbow.x, elbow.y, wristX, wristY);
    lay(this.part('link2Shadow'), elbow.x, elbow.y, wristX, wristY);
    for (const name of ['shoulder', 'shoulderGlow']) this.part(name).rotation.y = toward1;
    for (const name of ['elbow', 'elbowGlow']) {
      const mesh = this.part(name);
      toWorld(elbow.x, elbow.y, mesh.position);
      mesh.rotation.y = toward2;
    }
    for (const name of ['claw', 'payload', 'payloadShadow']) {
      const mesh = this.part(name);
      toWorld(tip.x, tip.y, mesh.position);
      mesh.rotation.y = toward2;
    }

    const regime = now.words[ARM2_REGIME]!, mass = this.massAt(regime), plate = mass > 0 ? PLATE * Math.sqrt(mass / this.heaviest) : 0;
    for (const name of ['payload', 'payloadShadow']) {
      const mesh = this.part(name);
      mesh.visible = plate > 0;
      mesh.scale.setScalar(Math.max(plate, 1e-3));
    }

    const shoulder = clampUnit(actions[0]!), elbowTorque = clampUnit(actions[1]!);
    aim(gauges[0], 0, 0, toward1, shoulder);
    aim(gauges[1], elbow.x, elbow.y, toward2, elbowTorque);
    this.effort[0].value = Math.abs(shoulder);
    this.effort[1].value = Math.abs(elbowTorque);

    const tx = pose[ARM2_TARGET_X]!, ty = pose[ARM2_TARGET_Y]!;
    const reached = lastEvent(frame.eventLog, 'goal', step), age = reached === undefined ? Infinity : step - reached + fraction;
    toWorld(tx, ty, this.part('target').position);
    toWorld(tx, ty, this.part('targetGlow').position);
    const ghost = this.part('ghost');
    ghost.visible = age < GHOST_STEPS;
    if (ghost.visible) {
      const words = frame.frameAt(reached! - 1).words, progress = age / GHOST_STEPS;
      toWorld(wordFloat(words[ARM2_TARGET_X]!), wordFloat(words[ARM2_TARGET_Y]!), ghost.position);
      ghost.scale.setScalar(1 + 2.2 * progress);
      this.ghostAlpha.value = (1 - progress) ** 2;
    }

    const gapX = tx - tip.x, gapY = ty - tip.y, span = Math.hypot(gapX, gapY), tetherMesh = this.part('tether');
    toWorld(tip.x, tip.y, tetherMesh.position);
    tetherMesh.rotation.y = Math.atan2(gapY, gapX);
    tetherMesh.scale.x = Math.max(span, 1e-3);
    this.span.value = span;
    this.closing.value = state.reward > 0 ? 1 : state.reward < 0 ? 0 : 0.5;

    const swapped = lastEvent(frame.eventLog, 'cue', step), swapAge = swapped === undefined ? Infinity : step - swapped + fraction, swap = this.part('swap');
    this.swapGlow.value = swapAge < SWAP_STEPS ? (1 - swapAge / SWAP_STEPS) ** 2 : 0;
    swap.visible = swapAge < SWAP_STEPS;
    toWorld(tip.x, tip.y, swap.position);
    swap.scale.setScalar((Math.max(plate, PLATE / 2) + 0.03) * (1 + 1.6 * Math.min(1, swapAge / SWAP_STEPS)));

    for (const event of frame.events) {
      if (event.type === 'goal' && ghost.visible) pool.burst(ghost.position, PALETTE.gold, 0.8, time);
      if (event.type === 'cue') pool.burst(toWorld(tip.x, tip.y, this.spot), PALETTE.brass, 0.5, time);
    }

    this.stroke(frame, trail);
    const ledger = this.ledger;
    ledger.gap = wordFloat(now.words[ARM2_DISTANCE]!);
    ledger.shoulder = shoulder * this.torques[0];
    ledger.elbow = elbowTorque * this.torques[1];
    ledger.mass = mass;
    ledger.reward = state.reward;
    ledger.since = reached === undefined ? null : step - reached;
  }

  private stroke(frame: FrameUpdate, trail: ReturnType<typeof ribbon>): void {
    const path = this.path, step = frame.state.step, out = trail.positions;
    path[0] = this.tip.x;
    path[1] = this.tip.y;
    for (let i = 1; i < TRAIL; i++) {
      const words = frame.frameAt(step - i + 1).words;
      this.forward(wordFloat(words[ARM2_Q1]!), wordFloat(words[ARM2_Q2]!), this.bend, this.scratch);
      path[2 * i] = this.scratch.x;
      path[2 * i + 1] = this.scratch.y;
    }
    let nx = 0, ny = 1;
    for (let i = TRAIL - 1; i >= 0; i--) {
      const a = Math.max(0, i - 1), b = Math.min(TRAIL - 1, i + 1);
      const dx = path[2 * a]! - path[2 * b]!, dy = path[2 * a + 1]! - path[2 * b + 1]!, chord = Math.hypot(dx, dy);
      if (chord > 1e-5) {
        nx = -dy / chord;
        ny = dx / chord;
      }
      const width = 0.022 * (1 - 0.6 * i / (TRAIL - 1)), x = path[2 * i]!, y = path[2 * i + 1]!, o = 6 * i;
      out[o] = x + nx * width;
      out[o + 2] = -(y + ny * width);
      out[o + 3] = x - nx * width;
      out[o + 5] = -(y - ny * width);
    }
    trail.attribute.needsUpdate = true;
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    DPR.value = size.dpr;
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.parts.clear();
    this.gauges = null;
    this.trail = null;
  }
}

export default (() => new WorkcellScene()) satisfies SceneFactory;
