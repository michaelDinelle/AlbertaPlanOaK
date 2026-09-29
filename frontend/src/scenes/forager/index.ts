import * as THREE from 'three/webgpu';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { FORAGER_X, FORAGER_Y, FW, STEP_X, STEP_Y, deltaOf, foragerDecoder, foragerShape, itemAt, kindOf, phaseOf, valueOf, type ForagerShape } from '@sim/envs/forager';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { AMBER, HALO, LANTERN, VIOLET, floorMaterial, type Floor } from './art';

const NAMES = ['↑ north', '↓ south', '← west', '→ east'] as const;
const SPECIES = ['', 'amber caps', 'violet parasols'] as const;
const RECENT = 1000;
const signed = (value: number) => formatSigned(value).replace('-', '−');

class ForagerScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.2, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'forager';

  private shape: ForagerShape | null = null;
  private grid: GridSize = { w: 16, h: 16 };
  private margin = 2;
  private floor: Floor | null = null;
  private pair = '';
  private step = -1;
  private clock = 0;
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { x: 0, y: 0, act: -1, count: 0, reward: 0, step: 0, good: 1, gains: 0, losses: 0, flip: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const shape = foragerShape(ctx.environment);
    this.shape = shape;
    this.grid = { w: shape.side, h: shape.side };
    this.margin = shape.fov + 1;
    const half = shape.side / 2 + this.margin + 0.2;
    return new THREE.Box2(new THREE.Vector2(-half, -half), new THREE.Vector2(half, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[FORAGER_X]!;
        agent.y = state.pose[FORAGER_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(_ctx: SceneContext, { pool }: Stage2D): void {
    const shape = this.shape!, { side, fov, target, switching } = shape, size = side + 2 * this.margin;
    const floor = floorMaterial(side, fov, this.margin, Math.max(1.35, side / 30));
    this.floor = floor;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size).rotateX(-Math.PI / 2), floor.material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 0;
    this.scene.add(mesh);
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 2 : 8; });

    const ledger = this.ledger;
    const species = (item: number) => `${SPECIES[item]} ${signed(valueOf(shape, ledger.step, item))}`;
    this.legend = {
      title: 'FORAGER',
      rows: [
        { key: 'forager', label: 'forager', tint: LANTERN, value: () => `(${ledger.x}, ${ledger.y}) · ${ledger.act < 0 ? '—' : NAMES[ledger.act]}` },
        { key: 'amber', label: 'amber', tint: AMBER, value: () => species(1) },
        { key: 'violet', label: 'violet', tint: VIOLET, value: () => species(2) },
        { key: 'field', label: 'on the field', tint: HALO, value: () => `${ledger.count} of ${target}` },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(ledger.reward) },
        { key: 'recent', label: `last ${formatCount(RECENT)}`, tint: TOKENS.uiCyan, value: () => `${ledger.gains} good · ${ledger.losses} bad` },
      ],
      note: () => switching ? `step ${formatCount(ledger.step)} · species swap in ${formatCount(ledger.flip)}` : `step ${formatCount(ledger.step)} · continuing torus ${side}×${side} · sees ${2 * fov + 1}×${2 * fov + 1}`,
      ...(switching ? { bar: () => 1 - ledger.flip / shape.period } : {}),
    } satisfies LegendSpec;
  }

  protected override drop(): void {
    this.floor?.map.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, time);
    const at = frame.state.step, A = frame.frameAt(at);
    if (A.step !== at) return;
    if (frame.scrubbed || at < this.step || this.step < 0 || at - this.step > 64) pool.clear();
    else for (let s = this.step + 1; s <= at; s++) this.effects(frame, s, pool, time);
    this.step = at;
    const after = frame.frameAt(at + 1), B = after.step === at + 1 && !after.reset ? after : A;
    this.compose(frame, A, B, B === A ? 0 : clamp01(frame.position - at));
    this.record(frame, A);
  }

  private compose(frame: FrameUpdate, A: EnvFrame, B: EnvFrame, t: number): void {
    const floor = this.floor, shape = this.shape;
    if (!floor || !shape) return;
    const { side } = shape, next = B !== A;
    const key = `${A.step}:${B.step}`;
    if (key !== this.pair) {
      this.pair = key;
      const data = floor.data, eaten = next ? B.words[FW.Y]! * side + B.words[FW.X]! : -1;
      for (let cell = 0; cell < side * side; cell++) {
        const a = itemAt(A.view, cell), b = next ? itemAt(B.view, cell) : a, changed = cell === eaten || a !== b;
        data[4 * cell] = 64 * a;
        data[4 * cell + 1] = 64 * b;
        data[4 * cell + 2] = a && changed ? 64 * (cell === eaten ? 1 : 2) : 0;
        data[4 * cell + 3] = b && changed ? 64 : 0;
      }
      floor.map.needsUpdate = true;
    }
    const ax = A.words[FW.X]!, ay = A.words[FW.Y]!, dx = next ? deltaOf(ax, B.words[FW.X]!, side) : 0, dy = next ? deltaOf(ay, B.words[FW.Y]!, side) : 0;
    const e = smoothstep01(clamp01(t / 0.85));
    floor.agent.value.set(ax + dx * e, ay + dy * e);
    const last = this.moveOf(frame, A), to = next ? Math.atan2(dy, dx) : last < 0 ? Math.PI / 2 : Math.atan2(STEP_Y[last]!, STEP_X[last]!);
    const from = last < 0 ? to : Math.atan2(STEP_Y[last]!, STEP_X[last]!);
    const angle = from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * smoothstep01(clamp01(t / 0.35));
    floor.dir.value.set(Math.cos(angle), Math.sin(angle));
    const goodA = valueOf(shape, A.step, 1) > 0 ? 1 : 0, goodB = next ? (valueOf(shape, B.step, 1) > 0 ? 1 : 0) : goodA;
    floor.good.value = THREE.MathUtils.lerp(goodA, goodB, e);
    floor.t.value = next ? t : 0;
    floor.time.value = this.clock;
  }

  private moveOf(frame: FrameUpdate, A: EnvFrame): number {
    const before = frame.frameAt(A.step - 1), side = this.grid.w;
    if (A.step === 0 || before.step !== A.step - 1 || A.reset) return -1;
    const dx = deltaOf(before.words[FW.X]!, A.words[FW.X]!, side), dy = deltaOf(before.words[FW.Y]!, A.words[FW.Y]!, side);
    return dy < 0 ? 0 : dy > 0 ? 1 : dx < 0 ? 2 : dx > 0 ? 3 : -1;
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private effects(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const shape = this.shape, A = frame.frameAt(s - 1), B = frame.frameAt(s);
    if (!shape || A.step !== s - 1 || B.step !== s || B.reset) return;
    const spot = this.centre(B.words[FW.X]!, B.words[FW.Y]!, this.at);
    for (const event of foragerDecoder.deriveEvents(A, B)) {
      if (event.type === 'goal') {
        pool.ripple(this.other.copy(spot).setY(0), TOKENS.celebrationGold, 0.3, time);
        pool.burst(this.other.copy(spot).setY(38), TOKENS.celebrationGold, 0.5, time);
      }
      if (event.type === 'harm') {
        pool.ripple(this.other.copy(spot).setY(0), TOKENS.failureBlue, 0.3, time);
        pool.burst(this.other.copy(spot).setY(38), TOKENS.failureBlue, 0.35, time);
        pool.pulse.fireSigh(0.6);
      }
    }
    if (phaseOf(shape, s) !== phaseOf(shape, s - 1)) {
      pool.ripple(this.other.set(0, 0, 0), HALO, shape.side * 0.35, time);
      pool.pulse.fire(0.6);
    }
  }

  private record(frame: FrameUpdate, A: EnvFrame): void {
    const shape = this.shape!, ledger = this.ledger;
    ledger.x = A.words[FW.X]!;
    ledger.y = A.words[FW.Y]!;
    ledger.act = this.moveOf(frame, A);
    ledger.count = A.words[FW.COUNT]!;
    ledger.reward = A.reward;
    ledger.step = A.step;
    ledger.flip = shape.period - (A.step % shape.period);
    const after = Math.max(-1, A.step - RECENT);
    ledger.gains = countEvents(frame.eventLog, 'goal', after, A.step);
    ledger.losses = countEvents(frame.eventLog, 'harm', after, A.step);
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const shape = this.shape, A = frame.frameAt(frame.state.step);
    if (!shape || A.step !== frame.state.step) return;
    const { side, fov, span } = shape, at = this.at;
    const diamond = (name: string, x: number, z: number, r: number, color: number) => {
      const curve = kit.curve(name, 5, color);
      for (const [u, v] of [[0, -1], [1, 0], [0, 1], [-1, 0], [0, -1]] as const) curve.point(x + u * r, 0.06, z + v * r);
    };
    let good = 0, bad = 0;
    for (let cell = 0; cell < side * side; cell++) {
      const item = itemAt(A.view, cell);
      if (!item) continue;
      this.centre(cell % side, Math.floor(cell / side), at);
      const pays = valueOf(shape, A.step, item) > 0;
      diamond(pays ? `good${good++}` : `bad${bad++}`, at.x, at.z, 0.3, pays ? TOKENS.celebrationGold : TOKENS.failureBlue);
    }
    const ax = A.words[FW.X]!, ay = A.words[FW.Y]!;
    for (let i = 0; i < span * span; i++) {
      const kind = kindOf(shape, A.observations[i]!);
      if (kind <= 0) continue;
      this.centre(ax + (i % span) - fov, ay + Math.floor(i / span) - fov, at);
      diamond(`seen${kind}-${i}`, at.x + 0.3, at.z - 0.3, 0.12, kind === 1 ? AMBER : VIOLET);
    }
    const box = kit.curve('window', 5, TOKENS.uiCyan);
    for (const [cx, cy] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) {
      this.centre(ax + cx * (fov + 0.5), ay + cy * (fov + 0.5), at);
      box.point(at.x, 0.06, at.z);
    }
    const arrow = kit.curve('action', 5, TOKENS.uiCyan), act = this.moveOf(frame, A);
    if (act >= 0) {
      const dx = STEP_X[act]!, dz = STEP_Y[act]!;
      const from = this.centre(ax - dx, ay - dz, this.other), tipX = from.x + 0.62 * dx, tipZ = from.z + 0.62 * dz;
      arrow.point(from.x, 0.07, from.z);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.14 * dx - 0.1 * dz, 0.07, tipZ - 0.14 * dz + 0.1 * dx);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.14 * dx + 0.1 * dz, 0.07, tipZ - 0.14 * dz - 0.1 * dx);
    }
  }
}

export default (() => new ForagerScene()) satisfies SceneFactory;
