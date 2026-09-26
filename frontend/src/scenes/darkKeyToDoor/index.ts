import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugCurve, DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { DARKKEY_DOOR_X, DARKKEY_DOOR_Y, DARKKEY_FOUND, DARKKEY_KEY_X, DARKKEY_KEY_Y, DARKKEY_X, DARKKEY_Y, darkKeyFound, darkKeyToDoorDecoder } from '@sim/envs/darkkey';
import type { EnvFrame, GridCell } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { BACKDROP, CANDLE, DEEP, figureMaterial, flat, FIGURE, FLAME, GOLD, HATCH, hatchMaterial, HOLE, KEY, keyMaterial, RIM, routeMaterial, startMaterial, boardMaterial, type FigureArt, type HatchArt, type KeyArt, type Level } from './art';

type Kind = 'rest' | 'move' | 'bump' | 'reset';

const ACCENT = TOKENS.accents.DarkKeyToDoor!;
const COMPASS: Record<string, string> = { '0,-1': '↑ north', '0,1': '↓ south', '-1,0': '← west', '1,0': 'east →' };
const QUEST = ['find the key', 'find the door', 'door open'] as const;
const SCAN = 4096;
const DOORS = 20;
const OPEN = 2.5;
const TURN = 14;
const CANDLE_LIGHT = 9;
const HATCH_LIGHT = 7;
const KEY_LIGHT = 5;
const AMBIENT = 0.38;
const SIZE = 1.4;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const ease = (u: number) => u * u * (3 - 2 * u);
const rise = (u: number, from: number, to: number) => ease(clamp01((u - from) / (to - from)));
const angleOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
const cellOf = (packed: number): GridCell => ({ x: packed & 15, y: packed >>> 4 });

interface View { task: number; found: number }

class DarkKeyToDoorScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'darkKeyToDoor';

  private side = 1;
  private horizon = 1;
  private period = 1;
  private grid: GridSize = { w: 1, h: 1 };
  private readonly owned: THREE.Mesh[] = [];
  private figure: THREE.Mesh | null = null;
  private figureArt: FigureArt | null = null;
  private key: THREE.Mesh | null = null;
  private keyArt: KeyArt | null = null;
  private hatch: THREE.Mesh | null = null;
  private hatchArt: HatchArt | null = null;
  private route: { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute } | null = null;
  private startRing: THREE.Mesh | null = null;
  private startLevel: Level | null = null;
  private candle: THREE.Mesh | null = null;
  private hatchLight: THREE.Mesh | null = null;
  private keyLight: THREE.Mesh | null = null;
  private readonly candleGlow = uniform(0);
  private readonly hatchGlow = uniform(0);
  private readonly keyGlow = uniform(0);

  private clock = 0;
  private step = -1;
  private kind: Kind = 'rest';
  private heading = 0;
  private aim = 0;
  private readonly moves: (readonly [number, number] | undefined)[] = [];
  private readonly cells: number[] = [];
  private episodeStart = 0;
  private keyStep = -1;
  private doorStep = -1;
  private keyAt = -Infinity;
  private openedAt = -Infinity;
  private readonly was: View = { task: 0, found: 0 };
  private readonly now: View = { task: 0, found: 0 };
  private wasFound = 0;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly push = new THREE.Vector3();
  private readonly final = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3(1, 1, 1);
  private log: FrameUpdate['eventLog'] | null = null;
  private readonly ledger = { x: 0, y: 0, action: -1, blocked: false, reward: 0, found: 0, step: 0 };

  override legend: LegendSpec = {
    title: 'DARK KEY-TO-DOOR',
    rows: [
      { key: 'at', label: 'wanderer', tint: ACCENT, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionLabel() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'quest', label: 'quest', tint: DEEP, value: () => QUEST[this.ledger.found] ?? '—' },
      { key: 'layout', label: 'layout', tint: ACCENT, value: () => formatCount(Math.floor(this.ledger.step / (this.horizon * this.period)) + 1) },
      { key: 'episode', label: 'episode', tint: ACCENT, value: () => `${formatCount(Math.floor((this.ledger.step % (this.horizon * this.period)) / this.horizon) + 1)} of ${formatCount(this.period)}` },
      { key: 'doors', label: 'doors opened', tint: FLAME, value: () => this.doorsLabel() },
    ],
    note: () => {
      const left = this.horizon - (this.ledger.step - this.episodeStart);
      return `${left} ${left === 1 ? 'move' : 'moves'} left`;
    },
    bar: () => (this.ledger.step - this.episodeStart) / this.horizon,
  };

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.side = parameter(env, 'side');
    this.horizon = parameter(env, 'horizon');
    this.period = parameter(env, 'period');
    this.grid = { w: this.side, h: this.side };
    const half = this.side / 2 + RIM;
    return new THREE.Box2(new THREE.Vector2(-half - 0.3, -half - 0.35), new THREE.Vector2(half + 0.3, half + 1.3));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(ACCENT).lerp(new THREE.Color(0x7f7f7f), 0.45).multiplyScalar(AMBIENT);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 }, door: GridCell = { x: 0, y: 0 }, keys: GridCell[] = [{ x: 0, y: 0 }], none: GridCell[] = [];
    return {
      agentCell: (state) => {
        agent.x = state.pose[DARKKEY_X]!;
        agent.y = state.pose[DARKKEY_Y]!;
        return agent;
      },
      goalCell: (state) => {
        door.x = state.pose[DARKKEY_DOOR_X]!;
        door.y = state.pose[DARKKEY_DOOR_Y]!;
        return door;
      },
      keyCells: (state) => {
        if (state.pose[DARKKEY_FOUND] !== 0) return none;
        keys[0]!.x = state.pose[DARKKEY_KEY_X]!;
        keys[0]!.y = state.pose[DARKKEY_KEY_Y]!;
        return keys;
      },
      debug: (frame, kit) => this.groundTruth(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), span = this.side + 2 * (RIM + 0.8);
    this.scene.background = new THREE.Color(BACKDROP);
    this.add(new THREE.Mesh(flat(span, span), boardMaterial(this.side, span, light.lit)), 0);

    const start = startMaterial(accent);
    this.startLevel = start.level;
    this.startRing = this.add(new THREE.Mesh(flat(0.5, 0.5), start.material), 1);

    const geometry = flat(1, 0.2), state = new THREE.InstancedBufferAttribute(new Float32Array(this.horizon * 2), 2).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aRoute', state);
    const route = instancedMesh(geometry, routeMaterial(accent), this.horizon);
    route.renderOrder = 1.5;
    route.frustumCulled = false;
    this.scene.add(route);
    this.route = { mesh: route, state };

    this.hatchArt = hatchMaterial(accent, light.lit);
    this.hatch = this.add(new THREE.Mesh(flat(HATCH, HATCH), this.hatchArt.material), 2);
    this.keyArt = keyMaterial(accent);
    this.key = this.add(new THREE.Mesh(flat(KEY, KEY), this.keyArt.material), 3);
    this.key.scale.setScalar(1.3);
    this.figureArt = figureMaterial(accent, light.lit);
    this.figure = this.add(new THREE.Mesh(flat(FIGURE, FIGURE), this.figureArt.material), 4);

    this.candle = this.add(light.emitter(new THREE.CircleGeometry(0.16, 32).rotateX(-Math.PI / 2), rgb(FLAME).mul(this.candleGlow)), 0);
    this.hatchLight = this.add(light.emitter(flat(2 * HOLE, 2 * HOLE), rgb(DEEP).mul(this.hatchGlow)), 0);
    this.keyLight = this.add(light.emitter(new THREE.CircleGeometry(0.14, 32).rotateX(-Math.PI / 2), rgb(GOLD).mul(this.keyGlow)), 0);
    pool.group.traverse((object) => { object.renderOrder = 5; });
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.owned.push(mesh);
    this.scene.add(mesh);
    return mesh;
  }

  protected override drop(): void {
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
    if (this.route) {
      this.route.mesh.geometry.dispose();
      (this.route.mesh.material as THREE.Material).dispose();
      this.route = null;
    }
  }

  protected tick(frame: FrameUpdate, { pool }: Stage2D): void {
    const step = frame.state.step;
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, this.clock);
    this.log = frame.eventLog;
    if (step !== this.step || frame.scrubbed) this.enter(frame, step, pool);
    this.animate(frame, step);
  }

  private enter(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null;
    const known = before !== null && before.step === step - 1 && now.step === step;
    if (frame.scrubbed) this.recall(frame, step);
    else for (let s = Math.max(1, this.step + 1); s <= step; s++) this.celebrate(frame, s, pool);
    const previous = this.step;
    this.scan(frame, step);
    if (frame.scrubbed) this.keyAt = this.openedAt = -Infinity;
    if (!frame.scrubbed && this.keyStep > previous) this.keyAt = this.keyStep;
    if (!frame.scrubbed && this.doorStep > previous) this.openedAt = this.doorStep;
    const action = step > 0 ? Math.trunc(now.actions[0]!) : -1, move = action >= 0 ? this.moves[action] : undefined;
    const x = now.words[0]!, y = now.words[1]!;
    this.now.task = now.words[2]!;
    this.now.found = now.words[3]!;
    this.was.task = known ? before.words[2]! : this.now.task;
    this.was.found = known ? before.words[3]! : this.now.found;
    this.wasFound = known ? darkKeyFound(before, now) : this.now.found;
    this.place(x, y, this.to);
    const bx = known ? before.words[0]! : x, by = known ? before.words[1]! : y;
    this.place(bx, by, this.from);
    const moved = bx !== x || by !== y;
    this.kind = frame.scrubbed || !known ? 'rest' : now.done ? 'reset' : moved ? 'move' : move ? 'bump' : 'rest';
    if (this.kind === 'reset') {
      const sought = this.was.found === 0 ? this.was.task & 255 : this.was.task >>> 8;
      const target = now.reward > 0 ? cellOf(sought) : { x: bx, y: by };
      this.place(target.x, target.y, this.final);
      if (target.x !== bx || target.y !== by) this.aim = angleOf(target.x - bx, target.y - by);
      else if (move) this.aim = angleOf(move[0], move[1]);
    } else if (moved) this.aim = angleOf(x - bx, y - by);
    else if (move) {
      this.aim = angleOf(move[0], move[1]);
      this.push.set(move[0], 0, move[1]);
    }
    if (frame.scrubbed) this.heading = this.aim;
    this.step = step;
    Object.assign(this.ledger, { x, y, action, blocked: known && !now.done && !moved && move !== undefined, reward: now.reward, found: this.now.found, step });
  }

  private celebrate(frame: FrameUpdate, s: number, pool: CelebrationPool): void {
    const before = frame.frameAt(s - 1), now = frame.frameAt(s);
    if (before.step !== s - 1 || now.step !== s) return;
    this.learn(before, now);
    const task = before.words[2]!;
    for (const event of darkKeyToDoorDecoder.deriveEvents(before, now)) {
      if (event.type === 'key') pool.burst(this.place(task & 15, (task >>> 4) & 15, this.scratch).setY(30), GOLD, 0.55, this.clock);
      if (event.type === 'door') pool.burst(this.place((task >>> 8) & 15, task >>> 12, this.scratch).setY(30), DEEP, 1, this.clock);
    }
    const next = now.words[2]!;
    if (now.done && next !== task) {
      for (const packed of [next & 255, next >>> 8]) {
        const cell = cellOf(packed);
        pool.burst(this.place(cell.x, cell.y, this.scratch).setY(30), ACCENT, 0.3, this.clock);
      }
    }
  }

  private learn(before: EnvFrame, now: EnvFrame): void {
    if (now.done) return;
    const action = Math.trunc(now.actions[0]!), dx = now.words[0]! - before.words[0]!, dy = now.words[1]! - before.words[1]!;
    if (action >= 0 && this.moves[action] === undefined && Math.abs(dx) + Math.abs(dy) === 1) this.moves[action] = [dx, dy];
  }

  private learned(): number {
    return this.moves.filter(Boolean).length;
  }

  private recall(frame: FrameUpdate, step: number): void {
    for (let s = step; s > Math.max(0, step - SCAN) && this.learned() < 4; s--) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1) return;
      this.learn(before, now);
    }
  }

  private scan(frame: FrameUpdate, step: number): void {
    const cells = this.cells;
    cells.length = 0;
    this.keyStep = this.doorStep = -1;
    let first = step;
    for (let s = step; s >= 0 && step - s <= this.horizon; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) break;
      cells.push(f.words[0]! | f.words[1]! << 4);
      first = s;
      if (f.done || s === 0) break;
      if (f.words[3]! >= 1) this.keyStep = s;
      if (f.words[3]! >= 2) this.doorStep = s;
    }
    cells.reverse();
    this.episodeStart = first;
  }

  private animate(frame: FrameUpdate, step: number): void {
    const figure = this.figure, figureArt = this.figureArt, keyArt = this.keyArt, hatchArt = this.hatchArt;
    if (!figure || !figureArt || !keyArt || !hatchArt || !this.key || !this.hatch) return;
    const dt = frame.dtReal, u = this.kind === 'rest' ? 1 : clamp01(frame.position - step), t = step + u;
    const reset = this.kind === 'reset', late = !reset || u >= 0.5;
    let fade = 1;
    if (this.kind === 'move') this.at.lerpVectors(this.from, this.to, ease(u));
    else if (this.kind === 'bump') this.at.copy(this.to).addScaledVector(this.push, 0.16 * Math.sin(Math.PI * u));
    else if (reset && !late) {
      this.at.lerpVectors(this.from, this.final, ease(clamp01(u / 0.3)));
      fade = 1 - rise(u, 0.3, 0.5);
    } else {
      this.at.copy(this.to);
      if (reset) fade = rise(u, 0.55, 0.95);
    }
    const turn = Math.atan2(Math.sin(this.aim - this.heading), Math.cos(this.aim - this.heading));
    this.heading += turn * (1 - Math.exp(-TURN * dt));
    figure.position.set(this.at.x, 0, this.at.z);
    figure.rotation.y = this.heading;
    figure.scale.setScalar(SIZE * (0.75 + 0.25 * fade));
    const view = late ? this.now : this.was, found = late ? this.now.found : this.wasFound;
    const picking = !reset && this.keyAt === step;
    figureArt.carry.value = reset ? (late ? 0 : this.wasFound === 1 ? 1 : 0) : found === 1 ? (picking ? rise(u, 0.35, 0.8) : 1) : 0;
    figureArt.fade.value = fade;
    const flicker = 1 + 0.07 * Math.sin(this.clock * 13.1) * Math.sin(this.clock * 5.3 + 1.1) + 0.04 * Math.sin(this.clock * 23.7);
    figureArt.flame.value = flicker * fade;
    this.candleGlow.value = CANDLE_LIGHT * flicker * fade;
    if (this.candle) this.candle.visible = fade > 0;
    if (this.candle) this.candle.position.copy(this.scratch.set(CANDLE.x, 0, -CANDLE.y).applyAxisAngle(this.axis, this.heading).multiplyScalar(figure.scale.x).add(this.at)).setY(0.01);

    const key = cellOf(view.task & 255), door = cellOf(view.task >>> 8);
    this.place(key.x, key.y, this.key.position);
    this.place(door.x, door.y, this.hatch.position);
    const swap = reset && this.was.task !== this.now.task ? (late ? rise(u, 0.55, 0.95) : 1 - rise(u, 0.1, 0.45)) : 1;
    const flash = this.keyStep < 0 || reset || t < this.keyAt + 0.55 ? 0 : Math.exp(-(t - this.keyAt - 0.55) * 1.4);
    const hidden = reset ? (this.wasFound === 0 ? 1 : late ? rise(u, 0.55, 0.95) : 0) : found === 0 ? 1 : picking ? 1 - rise(u, 0.4, 0.7) : 0;
    keyArt.ghost.value = hidden * swap;
    keyArt.flash.value = flash;
    keyArt.time.value = this.clock;
    this.keyGlow.value = KEY_LIGHT * flash;
    if (this.keyLight) {
      this.keyLight.visible = flash > 0.002;
      this.keyLight.position.copy(this.key.position).setY(0.01);
    }

    const open = reset ? (late ? 0 : this.was.found === 2 ? 1 - rise(u, 0, 0.45) : 0) : this.doorStep < 0 ? 0 : rise(t - this.openedAt, 0.55, 0.55 + OPEN);
    hatchArt.open.value = open;
    hatchArt.ghost.value = swap;
    hatchArt.seek.value = found === 1 ? 1 : 0;
    hatchArt.time.value = this.clock;
    this.hatchGlow.value = HATCH_LIGHT * open;
    if (this.hatchLight) {
      this.hatchLight.visible = open > 0;
      this.hatchLight.position.copy(this.hatch.position).setY(0.01);
    }

    this.paintRoute(reset && !late ? 0 : 1);
  }

  private paintRoute(level: number): void {
    const route = this.route, cells = this.cells;
    if (!route || !this.startRing || !this.startLevel) return;
    const first = cells[0];
    if (first !== undefined) {
      const cell = cellOf(first);
      this.place(cell.x, cell.y, this.startRing.position);
    }
    this.startLevel.value = first === undefined ? 0 : level;
    let used = 0;
    for (let i = 1; i < cells.length && used < this.horizon; i++) {
      const a = cellOf(cells[i - 1]!), b = cellOf(cells[i]!);
      if (a.x === b.x && a.y === b.y) continue;
      this.place((a.x + b.x) / 2, (a.y + b.y) / 2, this.scratch);
      this.quaternion.setFromAxisAngle(this.axis, Math.atan2(-(b.y - a.y), b.x - a.x));
      route.mesh.setMatrixAt(used, this.matrix.compose(this.scratch, this.quaternion, this.scale));
      route.state.setXY(used, level * (0.16 + 0.44 * (i / Math.max(1, cells.length - 1)) ** 1.5), 0);
      used++;
    }
    for (let i = used; i < this.horizon; i++) {
      route.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
      route.state.setXY(i, 0, 0);
    }
    route.mesh.instanceMatrix.needsUpdate = true;
    route.state.needsUpdate = true;
  }

  private actionLabel(): string {
    const { action, blocked, step } = this.ledger;
    if (step === 0 || action < 0) return '—';
    const move = this.moves[action], name = move ? COMPASS[`${move[0]},${move[1]}`]! : this.learned() === 4 ? 'stay' : `action ${action}`;
    return blocked ? `${name} · blocked` : name;
  }

  private doorsLabel(): string {
    const span = this.horizon * this.period, layout = this.ledger.step - this.ledger.step % span;
    const episodes = Math.min(DOORS, Math.floor((this.episodeStart - layout) / this.horizon));
    if (!this.log || episodes <= 0) return 'new layout';
    return `${countEvents(this.log, 'door', this.episodeStart - episodes * this.horizon, this.episodeStart)} of last ${episodes}`;
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit): void {
    const step = frame.state.step, pose = frame.state.pose, found = pose[DARKKEY_FOUND]!;
    const route = kit.curve('route', this.horizon + 1, TOKENS.uiCyan);
    for (const packed of this.cells) {
      const cell = cellOf(packed);
      this.place(cell.x, cell.y, this.scratch);
      route.point(this.scratch.x, 0.1, this.scratch.z);
    }
    const sought = kit.curve('sought', 5, 0xffffff);
    if (found === 0) this.outline(sought, pose[DARKKEY_KEY_X]!, pose[DARKKEY_KEY_Y]!, 0.3);
    else if (found === 1) this.outline(sought, pose[DARKKEY_DOOR_X]!, pose[DARKKEY_DOOR_Y]!, 0.3);
    const start = cellOf(this.cells[0] ?? (pose[DARKKEY_X]! | pose[DARKKEY_Y]! << 4));
    this.outline(kit.curve('start', 5, ACCENT), start.x, start.y, 0.24);
    const marks = kit.marks('found', 2);
    for (let i = 0; i < found; i++) {
      this.place(pose[DARKKEY_X]!, pose[DARKKEY_Y]!, this.scratch);
      marks.set(this.scratch.x - 0.3 + 0.16 * i, 0.12, this.scratch.z - 0.34, 0.06, i === 0 ? GOLD : DEEP);
    }
    const arrow = kit.curve('action', 5, 0xffa05a), now = frame.frameAt(step), before = frame.frameAt(step - 1);
    if (step < 1 || now.step !== step || before.step !== step - 1) return;
    const move = this.moves[Math.trunc(now.actions[0]!)];
    if (!move) return;
    this.place(before.words[0]!, before.words[1]!, this.scratch);
    const cx = this.scratch.x, cz = this.scratch.z, [dx, dz] = move;
    arrow.point(cx + dx * 0.08, 0.1, cz + dz * 0.08);
    arrow.point(cx + dx * 0.62, 0.1, cz + dz * 0.62);
    arrow.point(cx + dx * 0.46 - dz * 0.12, 0.1, cz + dz * 0.46 + dx * 0.12);
    arrow.point(cx + dx * 0.62, 0.1, cz + dz * 0.62);
    arrow.point(cx + dx * 0.46 + dz * 0.12, 0.1, cz + dz * 0.46 - dx * 0.12);
  }

  private outline(curve: DebugCurve, x: number, y: number, r: number): void {
    this.place(x, y, this.scratch);
    for (const [dx, dz] of [[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]] as const) curve.point(this.scratch.x + dx, 0.1, this.scratch.z + dz);
  }
}

export default (() => new DarkKeyToDoorScene()) satisfies SceneFactory;
