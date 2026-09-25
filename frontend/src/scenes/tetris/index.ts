import * as THREE from 'three/webgpu';
import { exp, float, fract, fwidth, instancedDynamicBufferAttribute, length, max, min, mix, positionWorld, select, smoothstep, uniform, uv, vec2 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01 } from '@kit/mix';
import { coverage, rgb, sdBox } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { TETRIS_H, TETRIS_W, tetrisByte, tetrisCleared, tetrisClearedRow, tetrisFilled, tetrisGame, tetrisLines, tetrisPiece, tetrisScore, type TetrisPiece } from '@sim/envs/tetris';
import type { EnvFrame } from '@sim/types';
import { constantList, constantRecords } from '../../data/envs';
import { formatCount } from '../../format';

type N<T extends string> = THREE.Node<T>;

const CELLS = TETRIS_W * TETRIS_H;
const GHOST = CELLS;
const LIVE = 2 * CELLS;
const NEXT = LIVE + 4;
const COUNT = NEXT + 4;
const HALF_W = TETRIS_W / 2;
const HALF_H = TETRIS_H / 2;
const RIM = 0.42;
const BOX = new THREE.Vector2(HALF_W + 3.3, -HALF_H + 2.2);
const BOX_HALF = new THREE.Vector2(2.3, 1.75);
const WIPE = 0.7;
const GHOST_SECONDS = 0.9;
const FLASH_SECONDS = 0.5;
const LOOKBACK = 64;
const NAMES = ['T', 'J', 'Z', 'O', 'S', 'L', 'I'] as const;
const STYLE = [0, 1, 2, 0, 1, 2, 0] as const;
const PALETTE: readonly (readonly [number, number])[] = [
  [0x3d78f2, 0x62c8f5], [0x2fae55, 0xa3d943], [0xb946c9, 0xf07fd9], [0x2b8fd6, 0x5fe0b0], [0xd9407a, 0x58d68f],
  [0x3fc79a, 0x7686f2], [0xd95236, 0x8f93a6], [0x7d46d9, 0xb82e63], [0x3563e6, 0xd9502f], [0xd9502f, 0xf0a93a],
];
const CORE = 0xcfd3de;
const DEAD = 0x5a6f9e;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const easeOut = (t: number) => 1 - (1 - t) ** 3;

interface Offset { dx: number; dy: number }

class TetrisScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.25, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 60;
  protected readonly stageName = 'tetris';

  private shapes: Offset[][] = [];
  private spawn: number[] = [];
  private cells: THREE.InstancedMesh | null = null;
  private lookAttr: THREE.InstancedBufferAttribute | null = null;
  private tintAttr: THREE.InstancedBufferAttribute | null = null;
  private readonly lip = uniform(new THREE.Color(PALETTE[0]![0]));
  private readonly glow = uniform(0);
  private readonly chill = uniform(0);
  private readonly danger = uniform(0);
  private readonly ghost = new Uint8Array(CELLS);
  private ghostAt = -Infinity;
  private flashAt = -Infinity;
  private flashGain = 0;
  private step = -1;
  private readonly shown = new Uint8Array(CELLS);
  private readonly sink = new Float32Array(CELLS);
  private readonly wiped = new Float32Array(CELLS);
  private readonly colors = PALETTE.map(([a, b]) => ({ a: new THREE.Color(a), b: new THREE.Color(b), stack: new THREE.Color(a).lerp(new THREE.Color(b), 0.3).lerp(new THREE.Color(0x2c3140), 0.55).multiplyScalar(0.85) }));
  private readonly deadTint = new THREE.Color(DEAD);
  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private readonly ledger = { score: 0, lines: 0, level: 0, next: 0, reward: 0, stack: 0 };

  override legend: LegendSpec = {
    title: 'Tetris',
    rows: [
      { key: 'score', label: 'score', tint: 0xffd166, value: () => formatCount(this.ledger.score) },
      { key: 'lines', label: 'lines', tint: 0x9fd4ff, value: () => formatCount(this.ledger.lines) },
      { key: 'level', label: 'level', tint: PALETTE[this.ledger.level % 10]![1], value: () => String(this.ledger.level) },
      { key: 'next', label: 'next', tint: 0xdfe3ee, value: () => NAMES[this.ledger.next] ?? '—' },
      { key: 'reward', label: 'reward', tint: 0xf2d27e, value: () => this.ledger.reward > 0 ? `+${formatCount(this.ledger.reward)}` : formatCount(this.ledger.reward) },
    ],
    note: () => `stack ${this.ledger.stack} of ${TETRIS_H} rows`,
    bar: () => this.ledger.stack / TETRIS_H,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const records = constantRecords(ctx.environment, 'TETRIS_SHAPE');
    this.shapes = records.map((record) => (record.cell as number[]).map((o) => {
      const dy = Math.round(o / 10);
      return { dx: o - 10 * dy, dy };
    }));
    this.spawn = constantList(ctx.environment, 'TETRIS_SPAWN');
    if (this.spawn.length !== NAMES.length) throw new Error(`${ctx.environment.envId} spawns ${this.spawn.length} pieces`);
    return new THREE.Box2(new THREE.Vector2(-HALF_W - RIM - 0.5, -HALF_H - RIM - 0.5), new THREE.Vector2(BOX.x + BOX_HALF.x + RIM + 0.5, HALF_H + RIM + 0.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      agentCell: (state) => ({ x: state.pose[0]!, y: state.pose[1]! }),
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step), marks = kit.marks('occupied', CELLS + 4), sky = kit.curve('heights', 2 * TETRIS_W, 0x5ec1ff);
        for (let c = 0; c < CELLS; c++) if (tetrisFilled(now.words, c)) marks.set(c % TETRIS_W - HALF_W + 0.5, 0.05, Math.floor(c / TETRIS_W) - HALF_H + 0.5, 0.16, 0x5ec1ff);
        const p = tetrisPiece(now.words);
        if (p.delay === 0) for (const o of this.shapes[p.orient] ?? []) if (p.y + o.dy >= 0) marks.set(p.x + o.dx - HALF_W + 0.5, 0.05, p.y + o.dy - HALF_H + 0.5, 0.2, 0xffd166);
        for (let c = 0; c < TETRIS_W; c++) {
          const top = HALF_H - now.observations[c]! * TETRIS_H;
          sky.point(c - HALF_W, 0.08, top);
          sky.point(c + 1 - HALF_W, 0.08, top);
        }
      },
    };
  }

  protected build(): void {
    this.scene.background = new THREE.Color(0x07080d);
    const q = vec2(positionWorld.x, positionWorld.z), px = max(fwidth(q.x), 1e-4);
    const well = sdBox(q, vec2(HALF_W, HALF_H)), frame = sdBox(q, vec2(HALF_W + RIM, HALF_H + RIM), 0.18);
    const box = q.sub(vec2(BOX.x, BOX.y)), inner = sdBox(box, vec2(BOX_HALF.x, BOX_HALF.y)), outer = sdBox(box, vec2(BOX_HALF.x + RIM, BOX_HALF.y + RIM), 0.18);
    const g = fract(q).sub(0.5).abs().sub(0.5).abs();
    const dots = smoothstep(px.mul(1.2), float(0), length(g).sub(0.03));
    const floorColor = mix(rgb(0x04050a), rgb(0x0a0d18), smoothstep(HALF_H, -HALF_H, q.y)).add(rgb(0x1b2236).mul(dots.mul(0.8)));
    const alarm = smoothstep(-HALF_H + 7, -HALF_H, q.y).mul(this.danger);
    const lit = floorColor.add(this.lip.mul(this.glow.mul(0.22))).add(rgb(DEAD).mul(this.chill.mul(0.25))).add(rgb(0xb8343a).mul(alarm.mul(0.16)));
    const rimShade = (d: N<'float'>) => {
      const t = min(max(d.div(RIM), 0), 1);
      const steel = mix(rgb(0x7d869c), rgb(0x262b39), smoothstep(0.15, 1, t)).mul(q.y.mul(-0.018).add(1));
      return steel.add(this.lip.mul(smoothstep(0.3, 0.05, t).mul(0.45)));
    };
    const inWell = coverage(well), inBox = coverage(inner), onFrame = coverage(frame), onBox = coverage(outer);
    const shadowA = smoothstep(0.6, 0, sdBox(q.sub(vec2(0.12, 0.18)), vec2(HALF_W + RIM, HALF_H + RIM), 0.18)).mul(0.5);
    const shadowB = smoothstep(0.6, 0, sdBox(box.sub(vec2(0.12, 0.18)), vec2(BOX_HALF.x + RIM, BOX_HALF.y + RIM), 0.18)).mul(0.5);
    const room = rgb(0x0b0d15).mul(exp(length(q.sub(vec2(1.5, 0))).mul(-0.06)).mul(0.9).add(0.2)).mul(float(1).sub(max(shadowA, shadowB)));
    let color = mix(room, rimShade(well), onFrame);
    color = mix(color, rimShade(inner), onBox);
    color = mix(color, lit, max(inWell, inBox));
    const backdrop = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    backdrop.colorNode = color;
    const plane = new THREE.Mesh(flat(400, 400), backdrop);
    plane.renderOrder = 0;
    this.scene.add(plane);

    this.lookAttr = new THREE.InstancedBufferAttribute(new Float32Array(COUNT * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.tintAttr = new THREE.InstancedBufferAttribute(new Float32Array(COUNT * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const look = instancedDynamicBufferAttribute(this.lookAttr, 'vec4') as N<'vec4'>;
    const tint = instancedDynamicBufferAttribute(this.tintAttr, 'vec3') as N<'vec3'>;
    const p = uv().sub(0.5), d = sdBox(p, vec2(0.455), 0.06), face = coverage(d);
    const bevel = p.y.sub(p.x).mul(0.38).add(1);
    const edge = smoothstep(-0.13, -0.09, d);
    const glint = coverage(sdBox(p.sub(vec2(-0.27, 0.27)), vec2(0.06), 0.015));
    const core = coverage(sdBox(p, vec2(0.21), 0.03));
    const body = tint.mul(bevel).mul(edge.mul(-0.4).add(1));
    const solid = mix(body, rgb(CORE), glint);
    const hollow = mix(mix(body, rgb(CORE).mul(p.y.mul(0.2).add(0.95)), core), rgb(CORE), glint);
    const stack = mix(body.mul(0.92), rgb(CORE).mul(0.55), glint.mul(0.6));
    const kind = look.x;
    const shade = select(kind.lessThan(0.5), hollow, select(kind.lessThan(2.5), solid, stack));
    const cell = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    cell.colorNode = mix(shade, rgb(CORE), look.z);
    cell.opacityNode = face.mul(look.y);
    const cells = instancedMesh(flat(1, 1), cell, COUNT);
    cells.renderOrder = 2;
    cells.frustumCulled = false;
    this.cells = cells;
    this.scene.add(cells);
  }

  protected override drop(): void {
    this.cells?.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const step = frame.state.step, now = frame.frameAt(step), piece = tetrisPiece(now.words);
    const forward = !frame.scrubbed && step > this.step;
    if (frame.scrubbed) this.ghostAt = this.flashAt = -Infinity;
    if (forward) for (const event of frame.events) {
      if (event.type === 'goal') {
        const lines = tetrisCleared(now.words);
        this.flashAt = time;
        this.flashGain = lines / 4;
        pool.pulse.fire(0.12 + 0.12 * lines);
        if (lines === 4) pool.ripple({ x: 0, y: 0, z: tetrisClearedRow(now.words, 0) + 1 - HALF_H }, 0xffd166, 2.5);
      }
      if (event.type === 'done') {
        const last = frame.frameAt(step - 1);
        if (last.step === step - 1) {
          for (let c = 0; c < CELLS; c++) this.ghost[c] = tetrisFilled(last.words, c) ? 1 : 0;
          this.ghostAt = time;
        }
      }
    }
    this.step = step;

    const frac = clamp01(frame.position - step), palette = this.colors[piece.level % 10]!;
    this.lip.value.copy(palette.a);
    const flash = (time - this.flashAt) / FLASH_SECONDS;
    this.glow.value = flash >= 0 && flash < 1 ? (0.3 + 0.7 * this.flashGain) * (1 - flash) ** 2 : 0;
    const ghostAge = (time - this.ghostAt) / GHOST_SECONDS;
    this.chill.value = ghostAge >= 0 && ghostAge < 1 ? (1 - ghostAge) ** 2 : 0;

    const clearing = this.clearing(frame, step, now, piece, frac);
    for (let c = 0; c < CELLS; c++) {
      const row = Math.floor(c / TETRIS_W), col = c % TETRIS_W;
      this.at.set(col - HALF_W + 0.5, 0, row - HALF_H + 0.5);
      if (clearing) {
        this.at.z += this.sink[c]!;
        this.place(c, this.shown[c] ? 3 : -1, 1, this.wiped[c]!, palette.stack);
      } else this.place(c, tetrisFilled(now.words, c) ? 3 : -1, 1, 0, palette.stack);
      this.at.set(col - HALF_W + 0.5, 0, row - HALF_H + 0.5);
      const fade = ghostAge >= 0 && ghostAge < 1 && this.ghost[c] ? 1 - clamp01((ghostAge - (row / TETRIS_H) * 0.35) / 0.65) : 0;
      this.place(GHOST + c, fade > 0 ? 3 : -1, fade, 0.15 * fade, this.deadTint);
    }

    const offsets = this.shapes[piece.orient] ?? [];
    let x = piece.x, y = piece.y;
    const ahead = frame.frameAt(step + 1);
    if (ahead.step === step + 1 && !ahead.done && tetrisGame(ahead.words) === tetrisGame(now.words)) {
      const next = tetrisPiece(ahead.words);
      if (next.delay === 0 && next.dealt === piece.dealt && next.orient === piece.orient) {
        x += (next.x - x) * frac;
        y += (next.y - y) * frac;
      }
    }
    for (let k = 0; k < 4; k++) {
      const o = offsets[k];
      const visible = piece.delay === 0 && o !== undefined && piece.y + o.dy >= 0;
      if (o) this.at.set(x + o.dx - HALF_W + 0.5, 0, y + o.dy - HALF_H + 0.5);
      this.place(LIVE + k, visible ? STYLE[piece.type]! : -1, 1, 0, STYLE[piece.type] === 2 ? palette.b : palette.a);
    }
    const preview = this.shapes[this.spawn[piece.next]!] ?? [];
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const o of preview) {
      minX = Math.min(minX, o.dx); maxX = Math.max(maxX, o.dx); minY = Math.min(minY, o.dy); maxY = Math.max(maxY, o.dy);
    }
    for (let k = 0; k < 4; k++) {
      const o = preview[k];
      if (o) this.at.set(BOX.x + o.dx - (minX + maxX) / 2, 0, BOX.y + o.dy - (minY + maxY) / 2);
      this.place(NEXT + k, o ? STYLE[piece.next]! : -1, 1, 0, STYLE[piece.next] === 2 ? palette.b : palette.a);
    }
    this.cells!.instanceMatrix.needsUpdate = true;
    this.lookAttr!.needsUpdate = true;
    this.tintAttr!.needsUpdate = true;

    let stack = 0;
    for (let c = 0; c < TETRIS_W; c++) stack = Math.max(stack, tetrisByte(now.words, 67 + c));
    this.danger.value = clamp01((stack - 12) / 6);
    Object.assign(this.ledger, { score: tetrisScore(now.words), lines: tetrisLines(now.words), level: piece.level, next: piece.next, reward: frame.state.reward, stack });
  }

  private clearing(frame: FrameUpdate, step: number, now: EnvFrame, piece: TetrisPiece, frac: number): boolean {
    const lines = tetrisCleared(now.words);
    if (piece.delay === 0 || lines === 0) return false;
    let lock = -1, total = 0;
    for (let s = step; s > step - LOOKBACK && s > 0; s--) {
      const f = frame.frameAt(s), b = frame.frameAt(s - 1);
      if (f.step !== s || b.step !== s - 1 || f.done) return false;
      if (tetrisByte(b.words, 57) === 0) {
        lock = s;
        total = tetrisByte(f.words, 57);
        for (let c = 0; c < CELLS; c++) this.shown[c] = tetrisFilled(b.words, c) ? 1 : 0;
        break;
      }
    }
    if (lock < 0 || total === 0) return false;
    for (const o of this.shapes[piece.orient] ?? []) {
      const row = piece.y + o.dy, col = piece.x + o.dx;
      if (row >= 0 && row < TETRIS_H && col >= 0 && col < TETRIS_W) this.shown[row * TETRIS_W + col] = 1;
    }
    const rows: number[] = [];
    for (let i = 0; i < lines; i++) rows.push(tetrisClearedRow(now.words, i));
    const t = clamp01((step - lock + frac) / total), wipe = clamp01(t / WIPE), fall = easeOut(clamp01((t - WIPE) / (1 - WIPE)));
    for (let c = 0; c < CELLS; c++) {
      const row = Math.floor(c / TETRIS_W), col = c % TETRIS_W;
      this.wiped[c] = 0;
      this.sink[c] = 0;
      if (rows.includes(row)) {
        const reach = Math.abs(col + 0.5 - HALF_W) / HALF_W;
        if (t >= WIPE || reach < wipe) this.shown[c] = 0;
        else this.wiped[c] = clamp01(1 - (reach - wipe) * 4);
      } else {
        let below = 0;
        for (const r of rows) if (r > row) below++;
        this.sink[c] = below * fall;
      }
    }
    return true;
  }

  private place(i: number, kind: number, alpha: number, flash: number, tint: THREE.Color): void {
    this.matrix.compose(this.at, this.quaternion, this.scale.setScalar(kind >= 0 && alpha > 0 ? 1 : 0));
    this.cells!.setMatrixAt(i, this.matrix);
    this.lookAttr!.setXYZW(i, Math.max(kind, 0), alpha, flash, 0);
    this.tintAttr!.setXYZ(i, tint.r, tint.g, tint.b);
  }
}

export default (() => new TetrisScene()) satisfies SceneFactory;
