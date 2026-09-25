import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugCurve, DebugKit, GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { BLOCKING_X, BLOCKING_Y } from '@sim/envs/blocking';
import type { EnvFrame, GridCell } from '@sim/types';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { groundMaterial, layoutOf, SHADOW, texelOf, type Ground, type Layout, type Lit } from './canal';
import {
  BOAT, CABIN, HULL, LANDING, LANTERN, MOOR, PLANK_X, PONTOON, RIDER, SIGNAL, SWING, TAIL_AT, TAILLAMP, LAMP_AT, WINDOWS,
  HEADLAMP, flat, houseboatMaterial, landingMaterial, riderMaterial, signalMaterial, swingMaterial, type Houseboat, type Rider, type Swing, type Tint,
} from './figures';

type Shift = readonly [number, number];
type Kind = 'rest' | 'move' | 'bump' | 'goal';

interface Bridge { cell: GridCell; index: number; mesh: THREE.Mesh; swing: Swing; tint: Tint; across: number }

const ACCENT = TOKENS.accents.ShortcutMaze!;
const GOLD = TOKENS.celebrationGold;
const AMBIENT = 0x9aaee0;
const AMBIENT_LEVEL = 0.9;
const MARGIN = 0.85;
const RIDER_SCALE = 1.25;
const SWING_STEPS = 6;
const GHOST_STEPS = 5;
const CHEER_SECONDS = 0.6;
const FLARE_SECONDS = 0.7;
const TURN = 18;
const NUDGE = 0.2;
const WEAR_TAU = 400;
const WEAR_SPAN = 2400;
const WEAR_SCALE = 6;
const SCAN = 8192;
const ROUTE = 256;
const LAMP_RADIUS = 0.045;
const BEAM_LIGHT = 7;
const TAIL_LIGHT = 2.4;
const WINDOW_LIGHT = 14;
const LANTERN_LIGHT = 10;
const SIGNAL_LIGHT = 5;
const RED = new THREE.Color(0xff3b30);
const GREEN = new THREE.Color(0x3dff88);
const WIRE = new THREE.Color(0x8fa6c4);
const COMPASS: Record<string, string> = { '0,-1': '↑ north', '0,1': '↓ south', '-1,0': '← west', '1,0': 'east →' };

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const ease = (u: number) => u * u * (3 - 2 * u);
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const angleOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
const level = (count: number) => 1 - Math.exp(-count / WEAR_SCALE);

class ShortcutMazeScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'shortcutMaze';

  private layout: Layout | null = null;
  private period = 1;
  private start: GridCell | null = null;
  private readonly crossings = new Map<number, 'bridge' | 'gate'>();
  private readonly delta: (Shift | null)[] = [null, null, null, null];

  private ground: Ground | null = null;
  private lit: Lit | null = null;
  private rider: Rider | null = null;
  private ghost: Rider | null = null;
  private boat: Houseboat | null = null;
  private readonly bridges: Bridge[] = [];
  private readonly owned: THREE.Mesh[] = [];
  private readonly riderGroup = new THREE.Group();
  private readonly ghostGroup = new THREE.Group();
  private readonly landingGroup = new THREE.Group();
  private readonly beam = uniform(BEAM_LIGHT);
  private readonly tail = uniform(TAIL_LIGHT);

  private clock = 0;
  private step = -1;
  private phase = -1;
  private kind: Kind = 'rest';
  private outcome: Kind = 'rest';
  private aim = 0;
  private heading = 0;
  private pedal = 0;
  private ghostStep = -Infinity;
  private ghostAim = 0;
  private cheerAt = -Infinity;
  private flareAt = -Infinity;
  private tripAt = -Infinity;
  private tripLabel = 'none yet';
  private counts = new Float32Array(0);
  private wearAt = -Infinity;
  private wearStale = true;
  private wearDirty = false;
  private log: FrameUpdate['eventLog'] | null = null;

  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly last = new THREE.Vector3();
  private readonly push = new THREE.Vector3();
  private readonly goalAt = new THREE.Vector3();
  private readonly startAt = new THREE.Vector3();
  private readonly doorAt = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly truth = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly ledger = { x: 0, y: 0, step: 0, reward: 0, action: -1 };

  override legend: LegendSpec = {
    title: 'SHORTCUT MAZE',
    rows: [
      { key: 'at', label: 'cyclist', tint: ACCENT, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionLabel() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'bridge', label: 'swing bridge', tint: GREEN.getHex(), value: () => this.bridgeLabel() },
      { key: 'trip', label: 'last trip', tint: 0xffd99c, value: () => this.tripLabel },
    ],
    note: () => this.countdown(),
    bar: () => (this.ledger.step % this.period) / this.period,
  };

  private get grid() {
    return this.layout!.grid;
  }

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private phaseAt(step: number): number {
    return Math.floor(step / this.period) % this.layout!.phases.length;
  }

  private open(bridge: Bridge, phase: number): number {
    const phases = this.layout!.phases;
    return phases[phase % phases.length]![bridge.cell.y * this.grid.w + bridge.cell.x] ? 0 : 1;
  }

  private acrossAt(bridge: Bridge, position: number): number {
    const k = Math.floor(Math.floor(position) / this.period), now = this.open(bridge, k), since = position - k * this.period;
    if (k === 0 || since >= SWING_STEPS) return now;
    const was = this.open(bridge, k - 1);
    return was + (now - was) * ease(since / SWING_STEPS);
  }

  private shoreOf(cell: GridCell): Shift {
    const { w, h } = this.grid;
    for (const [dx, dy] of [[0, -1], [0, 1], [1, 0], [-1, 0]] as const) {
      const x = cell.x + dx, y = cell.y + dy;
      if (x < 0 || y < 0 || x >= w || y >= h) return [dx, dy];
    }
    throw new Error(`ShortcutMaze has no open water beside (${cell.x}, ${cell.y})`);
  }

  private moor(group: THREE.Group, cell: GridCell, depth: number, reach: number): number {
    const [dx, dy] = this.shoreOf(cell), turn = Math.atan2(-dx, -dy), ax = Math.cos(turn), az = -Math.sin(turn);
    this.place(cell.x, cell.y, this.scratch);
    const inward = (this.grid.w - 1) / 2 - cell.x, down = (this.grid.h - 1) / 2 - cell.y, mirror = inward * ax + down * az < 0 ? 1 : -1;
    const offset = (0.5 + MOOR + depth / 2) * CELL;
    group.position.set(this.scratch.x + dx * offset - ax * mirror * reach * CELL, 0, this.scratch.z + dy * offset - az * mirror * reach * CELL);
    group.rotation.y = turn;
    return mirror;
  }

  private local(offset: THREE.Vector2, turn: number, out: THREE.Vector2): THREE.Vector2 {
    const c = Math.cos(turn), s = Math.sin(turn);
    return out.set(offset.x * c - offset.y * s, -(offset.x * s + offset.y * c));
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const layout = layoutOf(ctx.environment);
    this.layout = layout;
    this.period = constant(ctx.environment, 'BLOCKING_PERIOD');
    this.counts = new Float32Array(layout.grid.w * layout.grid.h * 2);
    for (const cell of layout.bridges) this.crossings.set(cell.y * layout.grid.w + cell.x, 'bridge');
    for (const cell of layout.gates) this.crossings.set(cell.y * layout.grid.w + cell.x, 'gate');
    const halfW = layout.grid.w / 2 + MARGIN, halfH = layout.grid.h / 2 + MARGIN;
    return new THREE.Box2(new THREE.Vector2(-halfW, -halfH).multiplyScalar(CELL), new THREE.Vector2(halfW, halfH).multiplyScalar(CELL));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(AMBIENT).multiplyScalar(AMBIENT_LEVEL);
  }

  protected override poolOptions() {
    return { burstSize: 44 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const { w, h } = this.grid;
    return {
      cellCapacity: Math.max(1, ...this.layout!.phases.map((p) => p.reduce((n, v) => n + v, 0))),
      lattice: { w, h, origin: [-w / 2 * CELL, 0.02, -h / 2 * CELL], u: [CELL, 0, 0], v: [0, 0, CELL] },
      frame: (state) => ({ originX: 0, originY: 0, originZ: 0, agent: this.place(state.pose[BLOCKING_X]!, state.pose[BLOCKING_Y]!, this.truth), cone: this.noCone, coneCount: 0 }),
      debug: (frame, kit) => this.groundTruth(frame, kit),
    };
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number, parent: THREE.Object3D = this.scene): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.owned.push(mesh);
    parent.add(mesh);
    return mesh;
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const layout = this.layout!, lit = light.lit, accent = this.accentOf(ctx);
    const ground = groundMaterial(layout, lit, this.ambient());
    this.ground = ground;
    ground.glint.value = this.debugOn ? 0 : 1;
    this.add(new THREE.Mesh(flat(60, 60), ground.material), 0);
    this.place(layout.goal.x, layout.goal.y, this.goalAt);

    layout.gates.forEach((cell, index) => {
      const swing = swingMaterial(lit), tint = uniform(new THREE.Vector3(RED.r, RED.g, RED.b));
      const mesh = this.add(new THREE.Mesh(flat(SWING, SWING), swing.material), 3);
      this.place(cell.x, cell.y, mesh.position);
      for (const [dx, dy] of [[0.4, -0.58], [-0.4, 0.58]] as const) {
        const post = this.add(new THREE.Mesh(flat(SIGNAL, SIGNAL), signalMaterial(lit, tint)), 3.5);
        this.place(cell.x + dx, cell.y + dy, post.position);
        const glow = this.add(light.emitter(new THREE.CircleGeometry(0.045, 20).rotateX(-Math.PI / 2), tint.mul(SIGNAL_LIGHT)), 2);
        glow.position.copy(post.position).setZ(post.position.z - 0.012);
      }
      this.bridges.push({ cell, index, mesh, swing, tint, across: -1 });
    });

    const dock = new THREE.Group();
    const mirror = this.moor(dock, layout.goal, HULL.y, PLANK_X);
    const drop = this.local(SHADOW, dock.rotation.y, new THREE.Vector2());
    const boat = houseboatMaterial(lit, mirror, drop);
    this.boat = boat;
    this.add(new THREE.Mesh(flat(BOAT.x, BOAT.y), boat.material), 2, dock);
    const glass = rgb(0xffc98a).mul(boat.flare.mul(1.2).add(1));
    const panes = WINDOWS.flatMap((x) => [1, -1].map((side) => flat(0.16, 0.04).translate((CABIN.x + x) * mirror, 0, -side * (CABIN.half.y + 0.022))));
    this.add(light.emitter(mergeGeometries(panes), glass.mul(WINDOW_LIGHT)), 2, dock);
    for (const pane of panes) pane.dispose();
    this.add(light.occluder(flat(CABIN.half.x * 2, CABIN.half.y * 2).translate(CABIN.x * mirror, 0, 0)), 1, dock);
    this.add(light.emitter(new THREE.CircleGeometry(0.034, 20).rotateX(-Math.PI / 2).translate(LANTERN.x * mirror, 0, -LANTERN.y), glass.mul(LANTERN_LIGHT)), 2, dock);
    this.scene.add(dock);
    this.doorAt.set(PLANK_X * mirror, 0, HULL.y / 2 + MOOR / 2).applyAxisAngle(THREE.Object3D.DEFAULT_UP, dock.rotation.y).add(dock.position).setY(36);

    this.landingGroup.visible = false;
    this.scene.add(this.landingGroup);

    const rider = riderMaterial(accent, lit);
    this.rider = rider;
    this.add(new THREE.Mesh(flat(RIDER, RIDER), rider.material), 5, this.riderGroup);
    this.add(light.emitter(new THREE.CircleGeometry(LAMP_RADIUS, 20).rotateX(-Math.PI / 2).translate(0, 0, -LAMP_AT), rgb(HEADLAMP).mul(this.beam)), 2, this.riderGroup);
    this.add(light.emitter(new THREE.CircleGeometry(0.03, 16).rotateX(-Math.PI / 2).translate(0, 0, -TAIL_AT), rgb(TAILLAMP).mul(this.tail)), 2, this.riderGroup);
    this.riderGroup.scale.setScalar(RIDER_SCALE);
    this.scene.add(this.riderGroup);

    const ghost = riderMaterial(accent, lit);
    this.ghost = ghost;
    this.add(new THREE.Mesh(flat(RIDER, RIDER), ghost.material), 4.8, this.ghostGroup);
    this.ghostGroup.visible = false;
    this.scene.add(this.ghostGroup);

    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 4 : 6; });
    this.lit = lit;
  }

  private meet(frame: FrameUpdate): void {
    const layout = this.layout!, scored = lastEvent(frame.eventLog, 'goal', frame.state.step);
    let origin: EnvFrame | null = frame.frameAt(0);
    if (origin.step !== 0) origin = scored === undefined ? null : frame.frameAt(scored);
    if (!origin || (origin.step !== 0 && !origin.done)) return;
    const start = { x: origin.words[0]!, y: origin.words[1]! };
    if (!layout.land[start.y * this.grid.w + start.x]) throw new Error('ShortcutMaze starts off the quays');
    this.start = start;
    this.place(start.x, start.y, this.startAt);
    this.ground?.start.value.set(start.x, start.y);
    this.moor(this.landingGroup, start, PONTOON.y, 0);
    this.add(new THREE.Mesh(flat(LANDING.x, LANDING.y), landingMaterial(this.lit!, this.local(SHADOW, this.landingGroup.rotation.y, new THREE.Vector2()))), 2, this.landingGroup);
    this.landingGroup.visible = true;
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    if (this.ground) this.ground.glint.value = on ? 0 : 1;
  }

  protected override drop(): void {
    this.ground?.map.dispose();
    this.ground?.wear.dispose();
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
    this.bridges.length = 0;
  }

  protected tick(frame: FrameUpdate, { pool }: Stage2D): void {
    const dt = frame.dtReal, step = frame.state.step;
    this.clock += dt;
    pool.update(dt, this.clock);
    this.log = frame.eventLog;
    if (!this.start) this.meet(frame);
    if (step !== this.step || frame.scrubbed) this.enter(frame, step, pool);
    this.swingBridges(frame, pool);
    if (this.wearStale && this.clock - this.wearAt > 0.4) this.rebuildWear(frame, step);
    if (this.wearDirty) this.paintWear();
    if (this.clock - this.tripAt > 0.4) this.refreshTrip(frame, step);
    this.ride(frame, step);
    if (this.ground) this.ground.time.value = this.clock;
    if (this.boat) this.boat.flare.value = Math.exp(-Math.max(0, this.clock - this.flareAt) / FLARE_SECONDS);
  }

  private learn(before: EnvFrame, now: EnvFrame): void {
    const action = Math.trunc(now.actions[0]!) & 3;
    if (this.delta[action]) return;
    const goal = this.layout!.goal, x = now.done ? goal.x : now.words[0]!, y = now.done ? goal.y : now.words[1]!;
    const dx = x - before.words[0]!, dy = y - before.words[1]!;
    if (Math.abs(dx) + Math.abs(dy) === 1) this.delta[action] = [dx, dy];
  }

  private recall(frame: FrameUpdate, step: number): void {
    for (let s = step; s > Math.max(0, step - SCAN) && this.delta.some((d) => !d); s--) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1) return;
      this.learn(before, now);
    }
  }

  private enter(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null, goal = this.layout!.goal;
    const known = now.step === step && before !== null && before.step === step - 1;
    if (frame.scrubbed) {
      this.recall(frame, step);
      this.wearStale = true;
      this.wearAt = -Infinity;
      this.ghostStep = -Infinity;
    } else {
      for (let s = Math.max(1, this.step + 1); s <= step; s++) {
        const later = frame.frameAt(s), earlier = frame.frameAt(s - 1);
        if (later.step !== s || earlier.step !== s - 1) { this.wearStale = true; continue; }
        this.learn(earlier, later);
        this.wear(earlier, later);
      }
    }
    const x = now.words[0]!, y = now.words[1]!, bx = known ? before.words[0]! : x, by = known ? before.words[1]! : y;
    this.step = step;
    this.outcome = !known ? 'rest' : now.done ? 'goal' : bx === x && by === y ? 'bump' : 'move';
    this.kind = frame.scrubbed ? 'rest' : this.outcome;
    this.place(x, y, this.to);
    this.place(bx, by, this.from);
    const action = step > 0 ? Math.trunc(now.actions[0]!) & 3 : -1, d = action >= 0 ? this.delta[action] : null;
    this.aim = this.outcome === 'goal' ? angleOf(goal.x - bx, goal.y - by) : this.outcome === 'move' ? angleOf(x - bx, y - by) : d ? angleOf(d[0], d[1]) : this.aim;
    this.push.set(d ? d[0] : 0, 0, d ? d[1] : 0);
    Object.assign(this.ledger, { x, y, step, reward: now.reward, action });
    if (frame.scrubbed || !frame.events.some((event) => event.type === 'goal')) return;
    const scored = this.log ? lastEvent(this.log, 'goal', step) ?? step : step, prior = frame.frameAt(scored - 1);
    this.ghostStep = scored;
    this.ghostAim = prior.step === scored - 1 ? angleOf(goal.x - prior.words[0]!, goal.y - prior.words[1]!) : this.aim;
    this.flareAt = this.clock;
    this.tripAt = -Infinity;
    if (this.clock - this.cheerAt < CHEER_SECONDS) return;
    this.cheerAt = this.clock;
    pool.burst(this.doorAt, GOLD, 0.7, this.clock);
    pool.ripple(this.goalAt, GOLD, 0.45, this.clock);
    pool.ripple(this.startAt, ACCENT, 0.3, this.clock);
  }

  private swingBridges(frame: FrameUpdate, pool: CelebrationPool): void {
    const phase = this.phaseAt(frame.state.step), blink = Math.sin(this.clock * 14) > 0 ? 1 : 0.25, w = this.grid.w;
    for (const bridge of this.bridges) {
      const across = this.acrossAt(bridge, frame.position), turn = (1 - across) * Math.PI / 2;
      bridge.across = across;
      bridge.mesh.rotation.y = turn;
      this.local(SHADOW, turn, bridge.swing.shadow.value);
      const { x, y } = bridge.cell;
      bridge.swing.north.value = level(this.counts[((y - 1) * w + x) * 2 + 1] ?? 0);
      bridge.swing.south.value = level(this.counts[(y * w + x) * 2 + 1] ?? 0);
      const open = this.open(bridge, phase) === 1, color = open ? GREEN : RED, k = across > 0 && across < 1 ? blink : 1;
      bridge.tint.value.set(color.r * k, color.g * k, color.b * k);
    }
    if (phase === this.phase) return;
    if (this.phase >= 0 && !frame.scrubbed) pool.pulse.fire(0.4);
    this.phase = phase;
  }

  private edge(ax: number, ay: number, bx: number, by: number): number {
    const w = this.grid.w;
    if (by === ay && bx === ax + 1) return (ay * w + ax) * 2;
    if (by === ay && bx === ax - 1) return (by * w + bx) * 2;
    if (bx === ax && by === ay + 1) return (ay * w + ax) * 2 + 1;
    if (bx === ax && by === ay - 1) return (by * w + bx) * 2 + 1;
    return -1;
  }

  private traversed(before: EnvFrame, now: EnvFrame): number {
    const goal = this.layout!.goal;
    return this.edge(before.words[0]!, before.words[1]!, now.done ? goal.x : now.words[0]!, now.done ? goal.y : now.words[1]!);
  }

  private wear(before: EnvFrame, now: EnvFrame): void {
    const decay = Math.exp(-1 / WEAR_TAU), counts = this.counts;
    for (let i = 0; i < counts.length; i++) counts[i]! *= decay;
    const e = this.traversed(before, now);
    if (e >= 0) counts[e]! += 1;
    this.wearDirty = true;
  }

  private rebuildWear(frame: FrameUpdate, step: number): void {
    this.counts.fill(0);
    this.wearAt = this.clock;
    this.wearStale = false;
    for (let i = 0; i < WEAR_SPAN && step - i >= 1; i++) {
      const now = frame.frameAt(step - i), before = frame.frameAt(step - i - 1);
      if (now.step !== step - i || before.step !== step - i - 1) {
        this.wearStale = true;
        continue;
      }
      const e = this.traversed(before, now);
      if (e >= 0) this.counts[e]! += Math.exp(-i / WEAR_TAU);
    }
    this.paintWear();
  }

  private paintWear(): void {
    const texture = this.ground?.wear;
    if (!texture) return;
    const data = texture.image.data as Uint8Array, { w, h } = this.grid;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const at = texelOf(this.grid, x, y), i = (y * w + x) * 2;
      data[at] = Math.round(255 * level(this.counts[i]!));
      data[at + 1] = Math.round(255 * level(this.counts[i + 1]!));
    }
    texture.needsUpdate = true;
    this.wearDirty = false;
  }

  private refreshTrip(frame: FrameUpdate, step: number): void {
    this.tripAt = this.clock;
    const log = frame.eventLog, end = lastEvent(log, 'goal', step);
    if (end === undefined) {
      this.tripLabel = 'none yet';
      return;
    }
    const begin = lastEvent(log, 'goal', end - 1) ?? 0, w = this.grid.w;
    let bridge = false, gate = false, known = end - begin <= SCAN;
    for (let s = begin + 1; known && s < end; s++) {
      const f = frame.frameAt(s);
      if (f.step !== s) {
        known = false;
        break;
      }
      const kind = this.crossings.get(f.words[1]! * w + f.words[0]!);
      if (kind === 'bridge') bridge = true;
      if (kind === 'gate') gate = true;
    }
    const route = !known ? '' : gate && bridge ? ' · both bridges' : gate ? ' · shortcut' : bridge ? ' · long way' : '';
    this.tripLabel = `${formatCount(end - begin)} ${end - begin === 1 ? 'step' : 'steps'}${route}`;
  }

  private ride(frame: FrameUpdate, step: number): void {
    const rider = this.rider, ghost = this.ghost;
    if (!rider || !ghost) return;
    const u = this.kind === 'rest' ? 1 : clamp01(frame.position - step), at = this.at;
    let fade = 1;
    if (this.kind === 'move') at.lerpVectors(this.from, this.to, u);
    else if (this.kind === 'bump') at.copy(this.to).addScaledVector(this.push, NUDGE * CELL * Math.sin(Math.PI * u));
    else if (this.kind === 'goal' && u < 0.5) at.lerpVectors(this.from, this.goalAt, u * 2);
    else if (this.kind === 'goal') {
      at.copy(this.to);
      fade = ease((u - 0.5) * 2);
    } else at.copy(this.to);
    const turn = wrap(this.aim - this.heading);
    this.heading = frame.scrubbed ? this.aim : this.heading + turn * (1 - Math.exp(-TURN * frame.dtReal));
    const travelled = at.distanceTo(this.last);
    if (!frame.scrubbed && travelled < 1.5 * CELL) this.pedal += travelled * 11;
    this.last.copy(at);
    this.riderGroup.position.copy(at);
    this.riderGroup.rotation.y = this.heading;
    const steer = frame.scrubbed ? 0 : Math.max(-0.45, Math.min(0.45, turn * 1.2));
    rider.steer.value = steer;
    rider.lean.value = steer;
    rider.pedal.value = this.pedal;
    rider.fade.value = fade;
    this.local(SHADOW, this.heading, rider.shadow.value);
    this.beam.value = BEAM_LIGHT * fade;
    this.tail.value = TAIL_LIGHT * fade;

    const t = (frame.position - this.ghostStep - 0.5) / GHOST_STEPS;
    this.ghostGroup.visible = t >= 0 && t < 1;
    if (!this.ghostGroup.visible) return;
    const lift = ease(t);
    this.ghostGroup.position.copy(this.goalAt).lerp(this.scratch.copy(this.doorAt).setY(0), 0.45 * lift);
    this.ghostGroup.rotation.y = this.ghostAim;
    this.ghostGroup.scale.setScalar(RIDER_SCALE * (1 - 0.35 * lift));
    ghost.fade.value = (1 - t) ** 1.5;
    ghost.pedal.value = this.pedal;
    this.local(SHADOW, this.ghostAim, ghost.shadow.value);
  }

  private actionLabel(): string {
    const { step, action } = this.ledger;
    if (step === 0 || action < 0) return '—';
    const d = this.delta[action], name = d ? COMPASS[`${d[0]},${d[1]}`]! : `action ${action}`;
    return this.outcome === 'bump' ? `${name} · blocked` : this.outcome === 'goal' ? `${name} · home` : name;
  }

  private bridgeLabel(): string {
    if (!this.bridges.length) return 'none';
    const across = this.bridges.map((bridge) => bridge.across);
    return across.every((a) => a >= 1) ? 'open' : across.every((a) => a <= 0) ? 'closed' : 'swinging';
  }

  private countdown(): string {
    const bridge = this.bridges[0];
    if (!bridge) return '';
    const step = this.ledger.step, k = Math.floor(step / this.period), now = this.open(bridge, k);
    for (let j = k + 1; j <= k + this.layout!.phases.length; j++) {
      if (this.open(bridge, j) === now) continue;
      const left = j * this.period - step;
      return `${now ? 'swings away' : 'swings across'} in ${formatCount(left)} ${left === 1 ? 'step' : 'steps'}`;
    }
    return now ? 'always open' : 'always closed';
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit): void {
    const { w } = this.grid, step = frame.state.step, walls = this.layout!.phases[this.phaseAt(step)]!, boxes = this.overlay?.cellBoxes();
    if (boxes) {
      boxes.begin();
      for (let i = 0; i < walls.length; i++) {
        if (!walls[i]) continue;
        this.place(i % w, Math.floor(i / w), this.scratch);
        boxes.box(this.scratch.x, this.scratch.z, 0, 0, 0, 0, WIRE.r, WIRE.g, WIRE.b);
      }
      boxes.commit();
    }
    const opened = kit.curve('gate open', 5 * this.bridges.length, GREEN), closed = kit.curve('gate closed', 5 * this.bridges.length, RED);
    for (const bridge of this.bridges) this.outline(walls[bridge.cell.y * w + bridge.cell.x] ? closed : opened, bridge.cell, 0.3, false);
    this.outline(kit.curve('goal', 5, GOLD), this.layout!.goal, 0.42, true);
    if (this.start) this.outline(kit.curve('start', 5, ACCENT), this.start, 0.36, false);
    const route = kit.curve('trip', ROUTE, TOKENS.uiCyan), cells: EnvFrame[] = [];
    for (let s = step; s >= 0 && cells.length < ROUTE; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) break;
      cells.push(f);
      if (f.done) break;
    }
    for (let i = cells.length - 1; i >= 0; i--) {
      const f = cells[i]!;
      this.place(f.words[0]!, f.words[1]!, this.scratch);
      route.point(this.scratch.x, 0.08, this.scratch.z);
    }
    const moved = kit.curve('moved', 5, TOKENS.uiCyan), blocked = kit.curve('blocked', 5, TOKENS.failureBlue), scored = kit.curve('scored', 5, GOLD);
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null;
    if (!before || before.step !== step - 1 || now.step !== step) return;
    const d = this.delta[Math.trunc(now.actions[0]!) & 3], bx = before.words[0]!, by = before.words[1]!;
    if (!d) return;
    const curve = now.done ? scored : bx === now.words[0] && by === now.words[1] ? blocked : moved;
    this.place(bx, by, this.scratch);
    const cx = this.scratch.x, cz = this.scratch.z, tip = 0.62 * CELL, back = 0.44 * CELL, side = 0.14 * CELL;
    curve.point(cx + d[0] * 0.08 * CELL, 0.1, cz + d[1] * 0.08 * CELL);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back - d[1] * side, 0.1, cz + d[1] * back + d[0] * side);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back + d[1] * side, 0.1, cz + d[1] * back - d[0] * side);
  }

  private outline(curve: DebugCurve, cell: GridCell, r: number, diamond: boolean): void {
    this.place(cell.x, cell.y, this.scratch);
    const corners = diamond ? [[0, -r], [r, 0], [0, r], [-r, 0], [0, -r]] : [[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]];
    for (const [x, z] of corners) curve.point(this.scratch.x + x! * CELL, 0.1, this.scratch.z + z! * CELL);
  }
}

export default (() => new ShortcutMazeScene()) satisfies SceneFactory;
