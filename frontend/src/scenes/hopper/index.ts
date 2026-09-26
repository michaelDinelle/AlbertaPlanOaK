import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { HOPPER_ROOT_ANGLE, HOPPER_ROOT_X, HOPPER_ROOT_Z } from '@sim/envs/planar';
import { SIM_EVENT_TYPES, type EventStream } from '@sim/types';
import { wordsDouble } from '@sim/words';
import { formatCount } from '../../format';
import { GAUGE, HALO, LAMP, PALETTE, PIN, RULE, SPARK, arc, disc, flat, foot, gauge, ground, hub, lamp, limb, pinMaterial, shade, slab, sparkMaterial, sprite, strokeMaterial, torso, wall, type Ink } from './art';
import { BODIES, JOINTS, forward, hopperModel, locate, solve, type HopperModel, type Point } from './model';

const RESETS = 36;
const STRIDE = 3;
const EXPOSURES = 10;
const SPAN = STRIDE * EXPOSURES;
const GHOST_STEPS = 24;
const FLASH_STEPS = 16;
const FALLS = 24;
const SEARCH = 64;
const FOLLOW = 0.5;
const HOME = 0.36;
const SWEEP = 0.8 * Math.PI;
const LINE = 0.0075;
const DOT = 0.017;
const ENDS = BODIES * 2;
const ARC = 10;
const LAMPS = 3;
const LAMP_GAP = 2.5;
const LAMP_HEIGHT = 1.9;
const DONE = SIM_EVENT_TYPES.indexOf('done');

interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute; used: number }

const signed = (value: number, digits: number): string => !Number.isFinite(value) ? '—' : `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`;
const smooth = (from: number, to: number, x: number): number => { const t = Math.min(1, Math.max(0, (x - from) / (to - from))); return t * t * (3 - 2 * t); };

class GaitLabScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  controlHz = 1;
  protected readonly stageName = 'gaitLab';

  private model!: HopperModel;
  private measured = false;
  private readonly rig = new THREE.Group();
  private readonly set = new THREE.Group();
  private readonly bodies: THREE.Group[] = [];
  private readonly arcs: THREE.Mesh[] = [];
  private readonly leds: THREE.Mesh[] = [];
  private readonly touches: THREE.Mesh[] = [];
  private readonly lamps: THREE.Group[] = [];
  private gaugeMesh: THREE.Mesh | null = null;
  private warning: THREE.Mesh | null = null;
  private flashLight: THREE.Mesh | null = null;
  private shadow: THREE.Mesh | null = null;
  private strokes: Batch | null = null;
  private pins: Batch | null = null;
  private sparks: Batch | null = null;

  private readonly dpr = uniform(1);
  private readonly effort = Array.from({ length: JOINTS }, () => uniform(0));
  private readonly sweep = Array.from({ length: JOINTS }, () => uniform(0));
  private readonly glow = Array.from({ length: JOINTS }, () => uniform(0));
  private readonly pitch = uniform(0);
  private readonly danger = uniform(0);
  private readonly sink = uniform(0);
  private readonly heat = uniform(0);
  private readonly flash = uniform(0);
  private readonly dark = uniform(0);

  private readonly live = new Float64Array(3 * BODIES);
  private readonly past = new Float64Array(3 * BODIES);
  private readonly truth = new Float64Array(3 * BODIES);
  private readonly ghost = new Float64Array(3 * BODIES);
  private readonly record = new Float64Array(3 * BODIES);
  private readonly a: Point = { x: 0, y: 0 };
  private readonly b: Point = { x: 0, y: 0 };
  private readonly spot = new THREE.Vector3();
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly falls: number[] = [];
  private previous = -1;
  private readonly ledger = { tilt: 0, height: 0, speed: NaN, distance: 0, reward: 0, attempt: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const model = hopperModel(ctx.environment), torso = model.capsules[0]!;
    this.model = model;
    this.controlHz = 1 / model.step;
    return new THREE.Box2(new THREE.Vector2(HOME - 2.3, -(model.rest + Math.max(torso.az, torso.bz) + GAUGE.outer)), new THREE.Vector2(HOME + 2.3, 0.62));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xd2d8e6).multiplyScalar(0.9);
  }

  protected override poolOptions() {
    return { burstSize: 36 };
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    return { debug: (frame, kit) => this.drawTruth(frame, kit, ctx) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const model = this.model, accent = this.accentOf(ctx), ink: Ink = { lit: light.lit, dpr: this.dpr };
    this.dpr.value = ctx.size.dpr;
    this.scene.background = new THREE.Color(PALETTE.night);
    this.scene.add(this.rig);
    this.rig.add(this.set);
    this.set.add(sprite(-40, -12, 40, 14, 0, wall(ink, model.height, this.sink), true));
    this.set.add(sprite(-40, -12, 40, 0.02, 1, ground(ink), true));
    const slabShade = light.occluder(flat(80, 12).translate(0, 0, 6));
    slabShade.renderOrder = 1;
    this.set.add(slabShade);

    const shadow = sprite(-0.34, -0.03, 0.34, 0.03, 2, shade(this.dark));
    this.shadow = shadow;
    this.rig.add(shadow);
    this.pins = this.batch(flat(PIN, PIN), 'aPin', pinMaterial(accent), FALLS + 1, 2);

    const { outer } = GAUGE;
    const gaugeMesh = sprite(-0.24, 0.02, 0.24, outer + 0.05, 3, gauge(ink, model.angle, this.pitch, this.danger));
    this.gaugeMesh = gaugeMesh;
    this.rig.add(gaugeMesh);
    const warning = light.emitter(flat(0.016, outer - GAUGE.inner).translate(0, 0, -(outer + GAUGE.inner) / 2), rgb(PALETTE.danger).mul(this.heat));
    warning.renderOrder = 2;
    this.warning = warning;
    this.rig.add(warning);

    this.strokes = this.batch(flat(1, 1), 'aStroke', strokeMaterial(), (EXPOSURES + 1) * (BODIES + JOINTS), 4);
    for (let i = 0; i < LAMPS; i++) {
      const fixture = new THREE.Group(), glow = light.emitter(disc(LAMP.bulb * 1.4), rgb(PALETTE.contact).mul(60));
      glow.renderOrder = 2;
      fixture.add(sprite(-LAMP.radius - 0.02, -LAMP.bulb - 0.02, LAMP.radius + 0.02, LAMP.drop, 5, lamp(ink)), glow);
      this.lamps.push(fixture);
      this.rig.add(fixture);
    }

    const caps = model.capsules;
    caps.forEach((c, g) => {
      const group = new THREE.Group(), pad = c.radius + 0.012;
      const paint = g === 0 ? torso(ink, c, accent, this.danger, this.sink) : g === BODIES - 1 ? foot(ink, c, accent) : limb(ink, c, g === 1 ? accent : new THREE.Color(accent).multiplyScalar(0.9).getHex(), 0.075);
      const box = [Math.min(c.ax, c.bx) - pad, Math.min(c.az, c.bz) - pad, Math.max(c.ax, c.bx) + pad, Math.max(c.az, c.bz) + pad] as const;
      group.add(sprite(...box, 6 + g * 0.1, paint));
      const block = light.occluder(slab(c));
      block.renderOrder = 1;
      group.add(block);
      this.bodies.push(group);
      this.rig.add(group);
    });
    for (let j = 0; j < JOINTS; j++) {
      const radius = Math.max(caps[j]!.radius, caps[j + 1]!.radius) + 0.012, reach = radius + 0.028, frame = reach + 0.05;
      this.bodies[j + 1]!.add(sprite(-radius - 0.01, -radius - 0.01, radius + 0.01, radius + 0.01, 6.5, hub(ink, radius, accent, this.effort[j]!)));
      const gaugeArc = sprite(-frame, -frame, frame, frame, 6.6, arc(ink, reach, this.sweep[j]!));
      this.arcs.push(gaugeArc);
      const led = light.emitter(disc(radius * 0.8), rgb(PALETTE.effort).mul(this.glow[j]!));
      led.renderOrder = 2;
      this.leds.push(led);
      this.rig.add(gaugeArc, led);
    }

    this.sparks = this.batch(flat(SPARK, SPARK), 'aSpark', sparkMaterial(), ENDS, 7);
    for (let i = 0; i < ENDS; i++) {
      const touch = light.emitter(disc(0.02), rgb(PALETTE.contact).mul(7));
      touch.renderOrder = 2;
      touch.visible = false;
      this.touches.push(touch);
      this.rig.add(touch);
    }
    const flashLight = light.emitter(disc(0.07), rgb(PALETTE.flash).mul(this.flash));
    flashLight.renderOrder = 2;
    this.flashLight = flashLight;
    this.rig.add(flashLight);
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const ledger = this.ledger, degrees = model.angle * 180 / Math.PI;
    this.legend = {
      title: 'GAIT LAB',
      rows: [
        { key: 'tilt', label: 'tilt', tint: PALETTE.danger, value: () => ledger.tilt === 0 ? '0.0°' : `${Math.abs(ledger.tilt).toFixed(1)}° ${ledger.tilt > 0 ? 'forward' : 'back'}` },
        { key: 'height', label: 'torso height', tint: PALETTE.strobe, value: () => `${ledger.height.toFixed(2)} m` },
        { key: 'speed', label: 'speed', tint: accent, value: () => `${signed(ledger.speed, 2)} m/s` },
        { key: 'distance', label: 'from start', tint: PALETTE.gold, value: () => `${signed(ledger.distance, 2)} m` },
        { key: 'reward', label: 'reward', tint: ctx.tokens.celebrationGold, value: () => signed(ledger.reward, 2) },
        { key: 'attempt', label: 'attempt', tint: ctx.tokens.failureBlue, value: () => `#${formatCount(ledger.attempt)}` },
      ],
      note: () => `falls past ±${degrees.toFixed(1)}° or under ${+model.height.toPrecision(3)} m`,
      bar: () => Math.min(1, Math.abs(ledger.tilt) / degrees),
    };
  }

  private batch(geometry: THREE.BufferGeometry, name: string, material: THREE.Material, count: number, order: number): Batch {
    const state = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, state);
    const mesh = instancedMesh(geometry, material, count);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.rig.add(mesh);
    return { mesh, state, used: 0 };
  }

  private put(batch: Batch, x: number, z: number, angle: number, sx: number, sz: number, a: number, b: number, c: number, d: number): void {
    const i = batch.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    batch.mesh.setMatrixAt(i, this.matrix.compose(this.spot.set(x, 0, -z), this.quaternion, this.scale.set(sx, 1, sz)));
    batch.state.setXYZW(i, a, b, c, d);
  }

  private commit(batch: Batch): void {
    batch.mesh.count = batch.used;
    batch.mesh.instanceMatrix.needsUpdate = true;
    batch.state.needsUpdate = true;
    batch.used = 0;
  }

  private stroke(batch: Batch, radius: number, alpha: number, ghost: number): void {
    const { a, b } = this, dx = b.x - a.x, dz = b.y - a.y, span = Math.hypot(dx, dz);
    this.put(batch, (a.x + b.x) / 2, (a.y + b.y) / 2, Math.atan2(dz, dx), span + 2 * (radius + HALO), 2 * (radius + HALO), span, radius, alpha, ghost);
  }

  private figure(batch: Batch, bodies: Float64Array, alpha: number, ghost: boolean): void {
    this.model.capsules.forEach((c, g) => {
      locate(bodies, g, c.ax, c.az, this.a);
      locate(bodies, g, c.bx, c.bz, this.b);
      this.stroke(batch, LINE, alpha, ghost ? 1 : 0);
    });
    for (let j = 1; j <= JOINTS; j++) {
      this.a.x = this.b.x = bodies[3 * j]!;
      this.a.y = this.b.y = bodies[3 * j + 1]!;
      this.stroke(batch, DOT, alpha, ghost ? 1 : 0);
    }
  }

  private measure(frame: FrameUpdate): void {
    const model = this.model, at = frame.state.step;
    let moved = 0, substeps = 0;
    for (let i = Math.max(1, at - SEARCH); i <= at + SEARCH; i++) {
      const now = frame.frameAt(i), before = frame.frameAt(i - 1);
      if (now.step !== i || before.step !== i - 1 || now.done) continue;
      const shift = wordsDouble(now.words, 2 * HOPPER_ROOT_X) - wordsDouble(before.words, 2 * HOPPER_ROOT_X);
      if (Math.abs(shift) <= moved) continue;
      moved = Math.abs(shift);
      substeps = shift / (forward(model, now) * model.step);
    }
    const whole = Math.round(substeps);
    if (!(whole >= 1 && Math.abs(substeps - whole) <= 1e-3)) return;
    this.controlHz = 1 / (model.step * whole);
    this.measured = true;
  }

  private start(frame: FrameUpdate, step: number): number {
    const floor = Math.max(0, step - Math.max(SPAN, GHOST_STEPS));
    for (let s = step; s > floor; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) return s + 1;
      if (f.reset) return s;
    }
    return floor;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const { strokes, pins, sparks, gaugeMesh, warning, flashLight, shadow } = this;
    if (!strokes || !pins || !sparks || !gaugeMesh || !warning || !flashLight || !shadow) return;
    if (!this.measured) this.measure(frame);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    const model = this.model, state = frame.state, pose = state.pose, step = state.step, fraction = frame.position - step;
    const live = solve(model, pose, this.live);
    this.bodies.forEach((group, g) => {
      group.position.set(live[3 * g]!, 0, -live[3 * g + 1]!);
      group.rotation.y = -live[3 * g + 2]!;
    });

    const next = frame.frameAt(step + 1), acting = next.step === step + 1 ? next.actions : null;
    for (let j = 0; j < JOINTS; j++) {
      const u = acting ? Math.max(-1, Math.min(1, acting[j]!)) : 0;
      this.effort[j]!.value = Math.abs(u);
      this.sweep[j]!.value = u * SWEEP;
      this.glow[j]!.value = 0.4 + 7 * Math.abs(u);
      this.arcs[j]!.position.set(live[3 * j + 3]!, 0, -live[3 * j + 4]!);
      this.leds[j]!.position.copy(this.arcs[j]!.position);
    }

    const tilt = pose[HOPPER_ROOT_ANGLE]!, danger = Math.abs(tilt) / model.angle;
    this.pitch.value = tilt;
    this.danger.value = danger;
    gaugeMesh.position.set(live[0]!, 0, -live[1]!);
    warning.position.copy(gaugeMesh.position);
    warning.rotation.y = tilt >= 0 ? -model.angle : model.angle;
    this.heat.value = 16 * smooth(0.55, 1, danger) ** 2;
    warning.visible = this.heat.value > 0;
    this.sink.value = smooth(0.4, 0.05, pose[HOPPER_ROOT_Z]! - model.height);

    this.touch(live, solve(model, frame.poseAt(step), this.past));
    const begun = this.start(frame, step), fell = begun > 0 && frame.frameAt(begun).reset && frame.frameAt(begun - 1).step === begun - 1;
    const age = fell ? step - begun + fraction : Infinity, ghost = age < GHOST_STEPS ? solve(model, frame.poseAt(begun - 1), this.ghost) : null;
    this.expose(frame, step, fraction, begun, ghost, age);
    this.mark(frame, step, live);

    const sole = model.capsules[BODIES - 1]!, heel = locate(live, BODIES - 1, sole.ax, sole.az, this.a), toe = locate(live, BODIES - 1, sole.bx, sole.bz, this.b);
    shadow.position.set((heel.x + toe.x) / 2, 0, 0.004);
    this.dark.value = Math.exp(-6 * Math.max(0, Math.min(heel.y, toe.y) - sole.radius));

    this.flash.value = ghost && age < FLASH_STEPS ? 18 * (1 - age / FLASH_STEPS) ** 2 : 0;
    flashLight.visible = this.flash.value > 0;
    if (ghost) flashLight.position.set(ghost[0]!, 0, -ghost[1]!);

    const x = pose[HOPPER_ROOT_X]!;
    this.camera.position.x = HOME + Math.max(0, x - FOLLOW) + Math.min(0, x + FOLLOW);
    this.set.position.x = this.camera.position.x;
    const first = Math.ceil((this.camera.position.x - LAMP_GAP * LAMPS / 2) / LAMP_GAP - 0.5);
    this.lamps.forEach((fixture, i) => fixture.position.set((first + i + 0.5) * LAMP_GAP, 0, -LAMP_HEIGHT));

    if (!frame.scrubbed) this.celebrate(frame, step, ghost, pool, time);
    this.previous = step;

    const now = frame.frameAt(step), ledger = this.ledger;
    ledger.tilt = tilt * 180 / Math.PI;
    ledger.height = pose[HOPPER_ROOT_Z]!;
    ledger.speed = step > 0 && now.step === step ? forward(model, now) : NaN;
    ledger.distance = x;
    ledger.reward = state.reward;
    ledger.attempt = now.words[RESETS]!;
  }

  private touch(live: Float64Array, recorded: Float64Array): void {
    const model = this.model, sparks = this.sparks!;
    let i = 0;
    model.capsules.forEach((c, g) => {
      for (const [u, w] of [[c.ax, c.az], [c.bx, c.bz]] as const) {
        const contact = locate(recorded, g, u, w, this.b).y - c.radius < model.margin, light = this.touches[i++]!;
        light.visible = contact;
        if (!contact) continue;
        const end = locate(live, g, u, w, this.a);
        light.position.set(end.x, 0, 0.002);
        this.put(sparks, end.x, 0, 0, 1, 1, 1, 0, 0, 0);
      }
    });
    this.commit(sparks);
  }

  private expose(frame: FrameUpdate, step: number, fraction: number, begun: number, ghost: Float64Array | null, age: number): void {
    const strokes = this.strokes!, model = this.model;
    for (let s = step - (step % STRIDE), k = 0; k < EXPOSURES && s >= begun; s -= STRIDE, k++) {
      const since = step - s + fraction;
      if (since < 0.5) continue;
      this.figure(strokes, solve(model, frame.poseAt(s), this.past), 0.62 * (1 - since / (SPAN + STRIDE)) ** 1.4, false);
    }
    if (ghost) this.figure(strokes, ghost, (1 - age / GHOST_STEPS) ** 1.5, true);
    this.commit(strokes);
  }

  private recent(log: EventStream, step: number): number[] {
    const out = this.falls;
    out.length = 0;
    for (let b = Math.min(log.blocks.length - 1, Math.floor(step / log.blockFrames)); b >= 0 && out.length < FALLS; b--) {
      const block = log.blocks[b];
      if (!block) break;
      for (let i = block.steps.length - 1; i >= 0 && out.length < FALLS; i--) if (block.kinds[i] === DONE && block.steps[i]! <= step) out.push(block.steps[i]!);
    }
    return out;
  }

  private mark(frame: FrameUpdate, step: number, live: Float64Array): void {
    const pins = this.pins!;
    this.recent(frame.eventLog, step).forEach((d, rank) => {
      const last = frame.frameAt(d - 1);
      if (last.step !== d - 1) return;
      this.put(pins, frame.poseAt(d - 1)[HOPPER_ROOT_X]!, RULE.pins, 0, 1, 1, 0.85 * (1 - rank / FALLS) ** 1.5, 0, 0, 0);
    });
    this.put(pins, live[0]!, RULE.pins + 0.012, 0, 1.45, 1.45, 1, 1, 0, 0);
    this.commit(pins);
  }

  private celebrate(frame: FrameUpdate, step: number, ghost: Float64Array | null, pool: Stage2D['pool'], time: number): void {
    for (const event of frame.events) {
      if (event.type === 'done' && ghost) {
        const top = this.model.capsules[0]!;
        locate(ghost, 0, top.ax, top.az, this.a);
        pool.burst(this.spot.set(this.a.x, 38, -this.a.y), PALETTE.ghost, 0.35, time);
      }
      if (event.type === 'teleport') pool.ripple(this.spot.set(this.live[3 * (BODIES - 1)]!, 0, 0), PALETTE.strobe, 0.12, time);
    }
    if (this.previous < 0 || step - this.previous > 12) return;
    for (let s = this.previous + 1; s <= step; s++) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1 || now.reset) continue;
      const from = frame.poseAt(s - 1)[HOPPER_ROOT_X]!, to = frame.poseAt(s)[HOPPER_ROOT_X]!, mark = Math.floor(to);
      if (mark >= 1 && mark > Math.floor(from)) {
        pool.burst(this.spot.set(mark, 38, -RULE.top), PALETTE.gold, 0.7, time);
        pool.ripple(this.spot.set(mark, 0, 0), PALETTE.gold, 0.2, time);
      }
    }
  }

  private drawTruth(frame: FrameUpdate, kit: DebugKit, ctx: SceneContext): void {
    const model = this.model, cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, red = PALETTE.ghost, slate = 0x8fa6c4, y = 0.06;
    const recorded = solve(model, frame.poseAt(frame.state.step), this.record), bodies = solve(model, frame.state.pose, this.truth), x = bodies[0]!, z = bodies[1]!;
    const contacts = kit.marks('contacts', ENDS);
    model.capsules.forEach((c, g) => {
      for (const [u, w] of [[c.ax, c.az], [c.bx, c.bz]] as const) {
        const end = locate(recorded, g, u, w, this.a);
        if (end.y - c.radius < model.margin) contacts.set(end.x, y, -(end.y - c.radius), 0.03, gold);
      }
      locate(bodies, g, c.ax, c.az, this.a);
      locate(bodies, g, c.bx, c.bz, this.b);
      const length = Math.hypot(this.b.x - this.a.x, this.b.y - this.a.y), ux = (this.b.x - this.a.x) / length, uz = (this.b.y - this.a.y) / length;
      const outline = kit.curve(`capsule${g}`, 2 * ARC + 3, cyan), r = c.radius;
      for (const [cx, cz, from] of [[this.b.x, this.b.y, Math.atan2(-ux, uz)], [this.a.x, this.a.y, Math.atan2(ux, -uz)]] as const) {
        for (let k = 0; k <= ARC; k++) outline.point(cx + r * Math.cos(from - Math.PI * k / ARC), y, -(cz + r * Math.sin(from - Math.PI * k / ARC)));
      }
      outline.point(this.b.x + r * Math.cos(Math.atan2(-ux, uz)), y, -(this.b.y + r * Math.sin(Math.atan2(-ux, uz))));
    });
    const joints = kit.marks('joints', BODIES);
    for (let g = 0; g < BODIES; g++) joints.set(bodies[3 * g]!, y, -bodies[3 * g + 1]!, 0.02, g === 0 ? gold : cyan);
    const floorLine = kit.curve('ground', 2, gold), fall = kit.curve('fallLine', 2, red);
    floorLine.point(x - 6, y, 0);
    floorLine.point(x + 6, y, 0);
    fall.point(x - 6, y, -model.height);
    fall.point(x + 6, y, -model.height);
    for (const [name, side] of [['tiltBack', -1], ['tiltForward', 1]] as const) {
      const ray = kit.curve(name, 2, red);
      ray.point(x, y, -z);
      ray.point(x + 0.75 * Math.sin(side * model.angle), y, -(z + 0.75 * Math.cos(side * model.angle)));
    }
    for (let j = 0; j < JOINTS; j++) {
      const c = model.capsules[j + 1]!, span = Math.hypot(c.bx - c.ax, c.bz - c.az), ux = (c.bx - c.ax) / span, uz = (c.bz - c.az) / span;
      const cx = bodies[3 * j + 3]!, cz = bodies[3 * j + 4]!, parent = bodies[3 * j + 2]!, range = kit.curve(`range${j}`, 25, slate), radius = 0.13;
      const direction = (a: number) => [ux * Math.cos(a) + uz * Math.sin(a), -ux * Math.sin(a) + uz * Math.cos(a)] as const;
      for (let k = 0; k <= 24; k++) {
        const [dx, dz] = direction(parent - model.high[j]! + (model.high[j]! - model.low[j]!) * k / 24);
        range.point(cx + radius * dx, y, -(cz + radius * dz));
      }
      const [dx, dz] = direction(bodies[3 * j + 5]!), now = kit.curve(`angle${j}`, 2, gold);
      now.point(cx, y, -cz);
      now.point(cx + (radius + 0.04) * dx, y, -(cz + (radius + 0.04) * dz));
    }
    const step = frame.state.step, next = frame.frameAt(step + 1);
    for (let j = 0; j < JOINTS; j++) {
      const u = next.step === step + 1 ? next.actions[j]! : 0, torque = kit.curve(`torque${j}`, 2, u >= 0 ? gold : cyan), at = x + 0.4 + 0.1 * j;
      torque.point(at, y, -(z - 0.3));
      torque.point(at, y, -(z - 0.3 + 0.25 * u));
    }
    const trail = kit.trail('root', 240, cyan);
    if (frame.scrubbed || frame.state.reset) trail.clear();
    trail.push(x, y, -z);
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
    this.bodies.length = this.arcs.length = this.leds.length = this.touches.length = this.lamps.length = 0;
    this.gaugeMesh = this.warning = this.flashLight = this.shadow = null;
    this.strokes = this.pins = this.sparks = null;
  }
}

export default (() => new GaitLabScene()) satisfies SceneFactory;
