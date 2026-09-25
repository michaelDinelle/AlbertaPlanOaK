import * as THREE from 'three/webgpu';
import { abs, atan, clamp, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, normalize, positionWorld, select, smoothstep, step, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import type { Light2D } from '@kit/light2d';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, fractal, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import {
  C4_AGENT, C4_CELLS, C4_COLUMNS, C4_CURSOR_SLOT, C4_DRAWN, C4_DROP, C4_LEFT, C4_LOST, C4_RIVAL, C4_ROWS, C4_WON,
  c4Action, c4Cursor, c4Game, c4Height, c4LastColumn, c4LastWho, c4Member, c4Owner, c4Result, c4Slot, c4Think,
} from '@sim/envs/connect4';
import type { EnvFrame, GridCell } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;

const GRID: GridSize = { w: C4_COLUMNS, h: C4_ROWS };
const HALF = C4_CELLS / 2;
const HOLE = 0.39;
const DISC = 0.44;
const SPRITE = 1;
const RIM = 0.34;
const HALF_W = C4_COLUMNS / 2 + RIM;
const TOP = -C4_ROWS / 2 - RIM;
const BOTTOM = C4_ROWS / 2 + RIM;
const FLOOR = C4_ROWS / 2 + 0.02;
const HOVER_ROWS = 1.42;
const HOVER_Z = -(C4_ROWS - 1) / 2 - HOVER_ROWS;
const TABLE = BOTTOM + 0.78;
const SEAT_X = HALF_W + 1.45;
const STACK_W = 0.9;
const STACK_T = 0.13;
const WALL = 0.1;
const SHELF = 0.08;
const DRAIN = C4_ROWS + 1;
const RING_R = 0.64;
const RING_SPRITE = 1.8;
const LAMPS = 16;
const CHIPS = 2 * C4_CELLS + 1;
const WAIT = 0.34;
const RIVAL = 0xe5413b;
const BLUE = 0x2156d8;
const GOLD = TOKENS.celebrationGold;
const BLOCKED = TOKENS.failureBlue;
const OPPONENTS = ['random', 'greedy', 'search', 'league'] as const;
const DIRECTIONS = [[1, 0], [0, 1], [1, 1], [1, -1]] as const;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const disc = (radius: number) => new THREE.CircleGeometry(radius, 28).rotateX(-Math.PI / 2);
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const cellAt = (column: number, row: number) => cellToWorldVec(column, C4_ROWS - 1 - row, GRID, new THREE.Vector3());
const columnX = (column: number) => cellAt(column, 0).x;
const rowZ = (row: number) => cellAt(0, row).z;

interface Finish { board: Uint8Array; column: number; row: number; who: number; result: number; line: number[]; runs: number[][] }
interface Strike { mesh: THREE.Mesh; from: THREE.Vector2; to: THREE.Vector2; tint: THREE.Color; alpha: { value: number } }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }
interface Falling { column: number; row: number; who: number; fall: number; x: number }

function readBoard(words: ArrayLike<number>, out: Uint8Array): Uint8Array {
  for (let column = 0; column < C4_COLUMNS; column++) for (let row = 0; row < C4_ROWS; row++) out[c4Slot(column, row)] = c4Owner(words, column, row);
  return out;
}

function fours(board: Uint8Array, column: number, row: number): number[][] {
  const who = board[c4Slot(column, row)]!, runs: number[][] = [];
  for (const [dc, dr] of DIRECTIONS) {
    const run = [c4Slot(column, row)];
    for (const sign of [1, -1]) for (let k = 1; ; k++) {
      const c = column + sign * dc * k, r = row + sign * dr * k;
      if (c < 0 || c >= C4_COLUMNS || r < 0 || r >= C4_ROWS || board[c4Slot(c, r)] !== who) break;
      if (sign > 0) run.push(c4Slot(c, r)); else run.unshift(c4Slot(c, r));
    }
    if (run.length >= 4) runs.push(run);
  }
  return runs;
}

function finishOf(before: EnvFrame, end: EnvFrame): Finish {
  const column = c4LastColumn(end.words), who = c4LastWho(end.words), result = c4Result(end.words);
  if (column < 0 || result === 0) throw new Error(`Connect Four step ${end.step} ends a game without recording its final move`);
  const board = readBoard(before.words, new Uint8Array(C4_CELLS)), row = c4Height(before.words, column);
  if (row >= C4_ROWS) throw new Error(`Connect Four step ${end.step} drops into full column ${column + 1}`);
  board[c4Slot(column, row)] = who;
  const runs = result === C4_DRAWN ? [] : fours(board, column, row);
  if (result !== C4_DRAWN && runs.length === 0) throw new Error(`Connect Four step ${end.step} records a win without four in a row`);
  return { board, column, row, who, result, line: [...new Set(runs.flat())], runs };
}

const composite = (color: N<'vec3'>, face: N<'float'>, under: N<'float'>) => {
  const alpha = face.add(face.oneMinus().mul(under));
  return { color: color.mul(face).div(max(alpha, 1e-4)), alpha };
};

class ConnectFourScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'connectFour';

  private accent = 0xffd84d;
  private rival: string = OPPONENTS[0];
  private league = false;
  private budget = 0;
  private perStep = 0;
  private readonly disposables: { dispose(): void }[] = [];

  private chips: THREE.InstancedMesh | null = null;
  private chipLook: THREE.InstancedBufferAttribute | null = null;
  private chipSeen: THREE.InstancedBufferAttribute | null = null;
  private hands: THREE.InstancedMesh | null = null;
  private handLook: THREE.InstancedBufferAttribute | null = null;
  private stacks: THREE.InstancedMesh | null = null;
  private stackLook: THREE.InstancedBufferAttribute | null = null;
  private shelves: THREE.Mesh[] = [];
  private lamps: Lamp[] = [];
  private strikes: Strike[] = [];
  private shaftLamp: Lamp | null = null;
  private seatLamp: Lamp | null = null;
  private flash: THREE.Mesh | null = null;
  private readonly ringProgress = uniform(0);
  private readonly ringAlpha = uniform(0);
  private readonly flashPower = uniform(0);

  private readonly live = new Uint8Array(C4_CELLS);
  private readonly tops = new Int8Array(C4_COLUMNS);
  private readonly finishes = new Map<number, Finish>();
  private celebrated = -1;
  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly ledger = { think: 0, cursor: 0, action: '—', reward: 0, member: 0, discs: 0, step: 0, game: 0 };

  private legendFor(): LegendSpec {
    return {
      title: 'CONNECT FOUR',
      rows: [
        { key: 'turn', label: 'turn', tint: this.accent, value: () => this.turnText() },
        { key: 'rival', label: 'rival', tint: RIVAL, value: () => (this.league ? `league member ${this.ledger.member + 1}` : this.rival) },
        { key: 'cursor', label: 'cursor', tint: this.accent, value: () => `column ${this.ledger.cursor + 1}` },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
        { key: 'last', label: 'last game', tint: GOLD, value: () => this.lastGame() },
      ],
      note: () => this.record(),
      bar: () => this.ledger.discs / C4_CELLS,
    };
  }

  private turnText(): string {
    const think = this.ledger.think;
    if (think === 0) return 'agent';
    if (this.budget === 0) return 'rival';
    return `rival thinking · ${formatCount(this.perStep * (think - 1))} of ${formatCount(this.budget)} nodes`;
  }

  private lastGame(): string {
    const log = this.eventLog;
    if (!log) return '—';
    const ended = lastEvent(log, 'done', this.ledger.step);
    if (ended === undefined) return '—';
    return lastEvent(log, 'goal', ended) === ended ? 'won' : lastEvent(log, 'harm', ended) === ended ? 'lost' : 'drawn';
  }

  private record(): string {
    const log = this.eventLog, step = this.ledger.step;
    if (!log) return '';
    const won = countEvents(log, 'goal', -1, step), lost = countEvents(log, 'harm', -1, step), drawn = countEvents(log, 'done', -1, step) - won - lost;
    return `game ${formatCount(this.ledger.game + 1)} · won ${formatCount(won)} · lost ${formatCount(lost)}${drawn ? ` · drawn ${formatCount(drawn)}` : ''}`;
  }

  private eventLog: FrameUpdate['eventLog'] | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, opponent = parameter(env, 'opponent'), name = OPPONENTS[opponent];
    if (name === undefined) throw new Error(`${env.envId} seats opponent ${opponent}, which Connect Four does not name`);
    this.accent = this.accentOf(ctx);
    this.rival = name;
    this.league = name === 'league';
    this.budget = parameter(env, 'nodesPerMove');
    this.perStep = constant(env, 'CONNECT4_NODES_PER_STEP');
    this.legend = this.legendFor();
    return new THREE.Box2(new THREE.Vector2(-SEAT_X - 1.05, HOVER_Z - 0.75), new THREE.Vector2(SEAT_X + 1.05, TABLE + 1.1));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff3e6).multiplyScalar(0.86);
  }

  protected override poolOptions() {
    return { burstSize: 64, sighRadius: 2.6, sighHeight: 1.2 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const cursor: GridCell = { x: 0, y: 0 }, spot = new THREE.Vector3();
    return {
      agentCell: (state) => {
        cursor.x = state.pose[C4_CURSOR_SLOT]!;
        cursor.y = -HOVER_ROWS;
        return cursor;
      },
      debug: (frame, kit) => {
        const step = frame.state.step, now = frame.frameAt(step), words = now.words, obs = now.observations;
        const discs = kit.marks('discs', C4_CELLS), ghosts = kit.marks('finished', C4_CELLS), tops = kit.marks('observed', C4_COLUMNS), last = kit.marks('last', 1);
        const skyline = kit.curve('skyline', 2 * C4_COLUMNS, 0xffffff), search = kit.curve('search', 65, RIVAL);
        for (let column = 0; column < C4_COLUMNS; column++) for (let row = 0; row < C4_ROWS; row++) {
          const owner = c4Owner(words, column, row);
          if (owner) discs.set(columnX(column), 0.05, rowZ(row), 0.14, owner === C4_AGENT ? TOKENS.uiCyan : TOKENS.cueMagenta);
        }
        for (let column = 0; column < C4_COLUMNS; column++) {
          const height = Math.round(obs[column]! * C4_ROWS), top = obs[C4_COLUMNS + column]!, z = rowZ(height) + 0.5;
          skyline.point(columnX(column) - 0.5, 0.05, z);
          skyline.point(columnX(column) + 0.5, 0.05, z);
          if (top !== 0) tops.set(columnX(column), 0.05, z, 0.07, top > 0 ? TOKENS.uiCyan : TOKENS.cueMagenta);
        }
        const lastColumn = c4LastColumn(words);
        if (!now.done && lastColumn >= 0) {
          const height = c4Height(words, lastColumn);
          if (height > 0) last.set(columnX(lastColumn), 0.05, rowZ(height - 1), 0.24, GOLD);
        }
        const finish = now.done ? this.finishAt(frame, step) : null;
        if (finish) for (let slot = 0; slot < C4_CELLS; slot++) {
          if (!finish.board[slot]) continue;
          ghosts.set(columnX(Math.floor(slot / C4_ROWS)), 0.05, rowZ(slot % C4_ROWS), finish.line.includes(slot) ? 0.2 : 0.07, finish.line.includes(slot) ? GOLD : 0x8fa6c4);
        }
        const think = c4Think(words);
        if (this.budget > 0 && think > 0) {
          const progress = clamp01(this.perStep * (think - 1) / this.budget);
          for (let k = 0; k <= 64; k++) {
            const angle = 2 * Math.PI * progress * k / 64;
            spot.set(SEAT_X + Math.sin(angle) * (RING_R + 0.2), 0.05, HOVER_Z - Math.cos(angle) * (RING_R + 0.2));
            search.point(spot.x, spot.y, spot.z);
          }
        }
      },
    };
  }

  private finishAt(frame: FrameUpdate, step: number): Finish | null {
    const cached = this.finishes.get(step);
    if (cached) return cached;
    const end = frame.frameAt(step), before = frame.frameAt(step - 1);
    if (step < 1 || end.step !== step || before.step !== step - 1 || !end.done) return null;
    const finish = finishOf(before, end);
    if (this.finishes.size > 64) this.finishes.clear();
    this.finishes.set(step, finish);
    return finish;
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    this.scene.background = new THREE.Color(0x07080d);
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const room = new THREE.Mesh(flat(80, 80), this.roomMaterial(light));
    room.renderOrder = 0;
    const face = new THREE.Mesh(flat(2 * HALF_W + 2.4, TABLE - TOP + 1), this.faceMaterial());
    face.position.z = (TOP + TABLE) / 2;
    face.renderOrder = 2;
    this.scene.add(room, face);
    this.disposables.push(room.geometry, room.material as THREE.Material, face.geometry, face.material as THREE.Material);

    this.chipLook = new THREE.InstancedBufferAttribute(new Float32Array(CHIPS * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.chipSeen = new THREE.InstancedBufferAttribute(new Float32Array(CHIPS), 1).setUsage(THREE.DynamicDrawUsage);
    const chips = instancedMesh(flat(SPRITE, SPRITE), this.discMaterial(light, this.chipLook, this.chipSeen, true), CHIPS);
    chips.renderOrder = 1;
    chips.frustumCulled = false;
    this.chips = chips;

    this.handLook = new THREE.InstancedBufferAttribute(new Float32Array(2 * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const hands = instancedMesh(flat(SPRITE, SPRITE), this.discMaterial(light, this.handLook, null, false), 2);
    hands.renderOrder = 4;
    hands.frustumCulled = false;
    this.hands = hands;

    this.stackLook = new THREE.InstancedBufferAttribute(new Float32Array(C4_CELLS), 1).setUsage(THREE.DynamicDrawUsage);
    const stacks = instancedMesh(flat(STACK_W + 0.1, STACK_T + 0.1), this.stackMaterial(light, this.stackLook), C4_CELLS);
    stacks.renderOrder = 3;
    stacks.frustumCulled = false;
    this.stacks = stacks;

    const ring = new THREE.Mesh(flat(RING_SPRITE, RING_SPRITE), this.ringMaterial());
    ring.position.set(SEAT_X, 0, HOVER_Z);
    ring.renderOrder = 5;
    const flash = new THREE.Mesh(flat(1, 0.5), this.flashMaterial());
    flash.renderOrder = 5;
    this.flash = flash;
    this.scene.add(chips, hands, stacks, ring, flash);
    for (const mesh of [chips, hands, stacks, ring, flash]) this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
    this.disposables.push(chips, hands, stacks);

    const wallGeometry = flat(WALL, C4_ROWS / 2 + 0.1 - TOP);
    for (let k = 0; k <= C4_COLUMNS; k++) {
      const wall = light.occluder(wallGeometry);
      wall.position.set(columnX(k) - 0.5, 0, (TOP + C4_ROWS / 2 + 0.1) / 2);
      this.scene.add(wall);
      this.disposables.push(wall.material as THREE.Material);
    }
    const sill = light.occluder(flat(C4_COLUMNS + WALL, 0.14));
    sill.position.set(0, 0, C4_ROWS / 2 + 0.07);
    this.scene.add(sill);
    this.disposables.push(wallGeometry, sill.geometry, sill.material as THREE.Material);

    const shelfGeometry = flat(1 - WALL + 0.02, SHELF), shelf = light.occluder(shelfGeometry);
    this.disposables.push(shelfGeometry, shelf.material as THREE.Material);
    for (let i = 0; i <= C4_CELLS; i++) {
      const mesh = i === 0 ? shelf : shelf.clone();
      mesh.visible = false;
      this.shelves.push(mesh);
      this.scene.add(mesh);
    }

    const lamp = (geometry: THREE.BufferGeometry): Lamp => {
      const tint = new THREE.Color(), mesh = light.emitter(geometry, uniform(tint) as unknown as N<'vec3'>);
      mesh.visible = false;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return { mesh, tint };
    };
    this.shaftLamp = lamp(flat(1 - WALL - 0.04, 0.12));
    this.seatLamp = lamp(disc(0.16));
    this.seatLamp.mesh.position.set(SEAT_X, 0, HOVER_Z);
    for (let i = 0; i < LAMPS; i++) this.lamps.push(lamp(disc(0.17)));

    const span = flat(C4_COLUMNS + 1, C4_ROWS + 1);
    this.disposables.push(span);
    for (let i = 0; i < DIRECTIONS.length; i++) {
      const from = new THREE.Vector2(), to = new THREE.Vector2(), tint = new THREE.Color(), alpha = uniform(0);
      const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
      const d = sdSegment(positionWorld.xz, uniform(from), uniform(to), 0.06), core = coverage(d), color = uniform(tint) as unknown as N<'vec3'>;
      material.colorNode = mix(color.mul(1.25), mix(color, vec3(1), 0.55).mul(1.6), core);
      material.opacityNode = min(core.add(exp(max(d, 0).mul(-11)).mul(0.5)), 1).mul(alpha);
      const mesh = new THREE.Mesh(span, material);
      mesh.renderOrder = 6;
      mesh.visible = false;
      this.scene.add(mesh);
      this.disposables.push(material);
      this.strikes.push({ mesh, from, to, tint, alpha });
    }
  }

  private roomMaterial(light: Light2D): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const p = positionWorld.xz, z = p.y;
    const plaster = fractal(p.mul(0.8), 4).mul(0.5).add(0.5);
    const wall = mix(rgb(0x3a4166), rgb(0x1f2440), smoothstep(HOVER_Z - 1, TABLE, z)).mul(plaster.mul(0.14).add(0.93));
    const frame = sdBox(p.sub(vec2(0, (TOP + BOTTOM) / 2)), vec2(HALF_W, (BOTTOM - TOP) / 2), 0.26);
    const cast = sdBox(p.sub(vec2(0.24, (TOP + BOTTOM) / 2 + 0.34)), vec2(HALF_W + 0.1, (BOTTOM - TOP) / 2 + 0.2), 0.4);
    const shadow = smoothstep(0.9, -0.3, cast).mul(smoothstep(-0.05, 0.08, frame)).mul(0.5);
    const grain = fractal(vec2(p.x.mul(0.3), z.mul(6.5)), 4).mul(0.5).add(0.5);
    const wood = mix(rgb(0x4a2d18), rgb(0x7a4c2a), grain).mul(smoothstep(TABLE, TABLE + 2.5, z).mul(-0.45).add(1));
    const lip = smoothstep(TABLE + 0.07, TABLE + 0.01, z).mul(0.3);
    const ax = abs(p.x), under = smoothstep(TABLE + 0.45, TABLE, z);
    const feet = exp(abs(ax.sub(HALF_W + 0.05)).div(0.75).pow(2).negate()).mul(under).mul(0.55);
    const piles = exp(abs(ax.sub(SEAT_X)).div(0.6).pow(2).negate()).mul(under).mul(0.5);
    const table = wood.add(rgb(0xffd9a8).mul(lip)).mul(max(feet, piles).oneMinus());
    const interior = mix(rgb(0x6b76a3), rgb(0x3c4468), smoothstep(TOP, BOTTOM, z));
    const backdrop = mix(interior, wall.mul(shadow.oneMinus()), smoothstep(-0.02, 0.02, frame));
    material.colorNode = light.lit(mix(backdrop, table, coverage(float(TABLE).sub(z))));
    return material;
  }

  private faceMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = positionWorld.xz, z = p.y, centre = vec2(0, (TOP + BOTTOM) / 2), half = vec2(HALF_W, (BOTTOM - TOP) / 2);
    const frameAt = (q: N<'vec2'>) => sdBox(q.sub(centre), half, 0.26);
    const frame = frameAt(p), e = 0.015;
    const normal = normalize(vec2(frameAt(p.add(vec2(e, 0))).sub(frameAt(p.sub(vec2(e, 0)))), frameAt(p.add(vec2(0, e))).sub(frameAt(p.sub(vec2(0, e))))));
    const origin = vec2(-(C4_COLUMNS - 1) / 2, -(C4_ROWS - 1) / 2);
    const q = p.sub(origin), cell = clamp(floor(q.add(0.5)), vec2(0), vec2(C4_COLUMNS - 1, C4_ROWS - 1)), local = q.sub(cell);
    const radius = length(local), hole = radius.sub(HOLE), face = max(frame, hole.negate());
    const ax = abs(p.x);
    const post = sdBox(vec2(ax, z).sub(vec2(HALF_W + 0.05, (0.8 + TABLE - 0.1) / 2)), vec2(0.18, (TABLE - 0.1 - 0.8) / 2), 0.1);
    const foot = sdBox(vec2(ax, z).sub(vec2(HALF_W + 0.05, TABLE - 0.13)), vec2(0.62, 0.13), 0.1);
    const slider = sdBox(p.sub(vec2(0, BOTTOM + 0.06)), vec2(HALF_W - 0.22, 0.16), 0.07);
    const legs = min(min(post, foot), slider);

    const blue = rgb(BLUE), depth = smoothstep(TOP, BOTTOM, z);
    const bevel = smoothstep(-0.11, 0, frame).mul(normal.y.negate()).mul(0.3);
    const gloss = exp(abs(frame.add(0.045)).mul(-70)).mul(max(normal.y.negate(), 0)).mul(0.45);
    const lip = smoothstep(0.08, 0, hole).mul(local.y.div(max(radius, 1e-4))).mul(0.34);
    const slot = coverage(sdBox(vec2(local.x, z.sub(TOP + 0.13)), vec2(0.33, 0.035), 0.03)).mul(step(z, TOP + 0.3));
    const sheen = exp(abs(p.x.mul(0.34).add(z.mul(0.5)).add(1.3)).mul(-2.2)).mul(0.07);
    const plastic = blue.mul(depth.mul(-0.22).add(1.12).add(bevel).add(lip).add(sheen)).add(vec3(gloss)).mul(slot.mul(-0.7).add(1));
    const legTone = blue.mul(0.72).mul(smoothstep(-0.1, 0, legs).mul(-0.3).add(1)).mul(select(slider.lessThan(min(post, foot)), float(0.82), float(1)));

    const inside = step(frame, 0).mul(step(hole, 0));
    const crescent = smoothstep(-0.03, 0.05, length(local.sub(vec2(0, 0.075))).sub(HOLE)).mul(0.55);
    const rim = smoothstep(-0.07, 0, hole).mul(0.28);
    const under = max(crescent, rim).mul(inside);
    const solid = max(coverage(face), coverage(legs));
    const color = mix(legTone, plastic, coverage(face));
    const result = composite(color, solid, under);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private discMaterial(light: Light2D, look: THREE.InstancedBufferAttribute, seen: THREE.InstancedBufferAttribute | null, board: boolean): THREE.MeshBasicNodeMaterial {
    const v = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, observed = seen ? instancedDynamicBufferAttribute(seen, 'float') as N<'float'> : float(1);
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(SPRITE), r = length(t), rival = v.x, glow = v.y, alpha = v.z, marker = v.w;
    const body = mix(rgb(this.accent), rgb(RIVAL), rival);
    const dir = t.div(max(r, 1e-4)), facing = dir.x.mul(-0.6).add(dir.y.mul(0.8));
    const ridge = smoothstep(0.265, 0.29, r).mul(smoothstep(0.37, 0.345, r));
    const boss = smoothstep(0.125, 0.105, r);
    const well = smoothstep(0.29, 0.265, r).mul(boss.oneMinus());
    const dome = clamp(t.x.mul(-0.6).add(t.y.mul(0.8)).div(0.125), -1, 1);
    const shade = float(1.04).add(ridge.mul(facing).mul(0.24)).sub(well.mul(0.1)).add(boss.mul(dome).mul(0.1)).sub(r.mul(0.12));
    const sheen = exp(length(t.sub(vec2(-0.12, 0.15))).mul(-10)).mul(0.2);
    const ring = exp(abs(r.sub(0.2)).mul(-55)).mul(marker);
    const crest = smoothstep(0.24, 0.36, r).mul(smoothstep(HOLE, 0.35, r)).mul(max(dir.y, 0).pow(1.5));
    const exposure = mix(float(0.8), float(1), observed).add(crest.mul(observed).mul(0.3));
    const albedo = mix(body.mul(shade).mul(exposure).add(vec3(sheen)), vec3(1, 0.97, 0.9), ring.mul(0.85));
    const glowing = albedo.mul(glow.mul(0.5).add(1));
    const shaded = board ? min(light.lit(albedo), albedo.mul(1.5)) : albedo.mul(alpha.mul(0.36).add(0.6));
    material.colorNode = mix(shaded, glowing, clamp(glow.mul(4), 0, 1));
    const edge = coverage(r.sub(DISC)).mul(alpha);
    material.opacityNode = board ? edge.mul(step(positionWorld.z, FLOOR)) : edge;
    return material;
  }

  private stackMaterial(light: Light2D, look: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
    const rival = instancedDynamicBufferAttribute(look, 'float') as N<'float'>;
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(vec2(STACK_W + 0.1, STACK_T + 0.1));
    const d = sdBox(t, vec2(STACK_W / 2, STACK_T / 2 - 0.008), 0.055);
    const body = mix(rgb(this.accent), rgb(RIVAL), rival);
    const band = smoothstep(-STACK_T / 2, STACK_T / 2, t.y);
    const ends = t.x.div(STACK_W / 2).pow(4).mul(-0.35).add(1);
    const seam = exp(abs(t.y.sub(STACK_T / 2 - 0.028)).mul(-90)).mul(0.25);
    material.colorNode = light.lit(body.mul(mix(float(0.55), float(1.08), band)).mul(ends).add(vec3(seam)));
    material.opacityNode = coverage(d);
    return material;
  }

  private ringMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(RING_SPRITE), r = length(t);
    const turn = fract(atan(t.x, t.y).div(2 * Math.PI).add(1));
    const band = abs(r.sub(RING_R));
    const track = coverage(band.sub(0.028));
    const filled = step(turn, this.ringProgress);
    const head = this.ringProgress.mul(2 * Math.PI);
    const tip = vec2(head.sin(), head.cos()).mul(RING_R);
    const spark = exp(length(t.sub(tip)).mul(-14)).mul(step(0.002, this.ringProgress));
    const ticks = coverage(abs(fract(turn.mul(16)).sub(0.5)).mul(-1).add(0.5).mul(RING_R * 2 * Math.PI / 16).sub(0.018)).mul(coverage(abs(r.sub(RING_R + 0.12)).sub(0.035)));
    const glow = exp(band.mul(-26)).mul(filled).mul(0.35);
    const intensity = track.mul(filled.mul(0.72).add(0.28)).add(glow).add(spark.mul(0.9)).add(ticks.mul(0.35));
    material.colorNode = mix(rgb(0x6b2a2a), rgb(0xff6a4f).mul(1.7), clamp(filled.add(spark), 0, 1));
    material.opacityNode = clamp(intensity, 0, 1).mul(this.ringAlpha);
    return material;
  }

  private flashMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });
    const t = uv().sub(0.5).mul(vec2(1, 0.5)), d = sdBox(t, vec2(0.36, 0.05), 0.05);
    material.colorNode = mix(rgb(BLOCKED), vec3(1), 0.35).mul(2.6);
    material.opacityNode = min(coverage(d).add(exp(max(d, 0).mul(-16)).mul(0.8)), 1).mul(this.flashPower);
    return material;
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.shelves = [];
    this.lamps = [];
    this.strikes = [];
    this.finishes.clear();
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.eventLog = frame.eventLog;
    const t = frame.state.step, now = frame.frameAt(t), peek = frame.frameAt(t + 1), ahead = peek.step === t + 1 ? peek : null;
    const a = ahead ? clamp01(frame.position - t) : 0;
    if (frame.events.some((event) => event.type === 'done')) this.celebrate(frame, pool, time);
    this.ledgerOf(frame, now);

    readBoard(now.words, this.live);
    for (let column = 0; column < C4_COLUMNS; column++) this.tops[column] = Math.round(now.observations[column]! * C4_ROWS) - 1;
    const cursor = c4Cursor(now.words), think = c4Think(now.words), thinking = think > 0 && this.budget > 0;
    let hoverX = columnX(cursor), hoverZ = HOVER_Z, hoverAlpha = think === 0 ? 1 : WAIT, seatAlpha = think === 0 ? WAIT : 1, beam = think === 0 ? 1 : 0, beamX = hoverX;
    let flash = 0, falling: Falling | null = null, line: Set<number> | null = null, lineGlow = 0, struck: Finish | null = null, strike = 0;
    let ring = thinking ? this.searched(think) : 0, ringAlpha = thinking ? 1 : 0;

    if (ahead) {
      const after = c4Think(ahead.words), moved = ahead.done || (think === 0 ? after > 0 : after === 0);
      const opens = ahead.done && after === 0, rests = ahead.done && after > 0;
      if (moved) {
        const column = c4LastColumn(ahead.words), who = c4LastWho(ahead.words), row = c4Height(now.words, column);
        if (column < 0 || row >= C4_ROWS || who !== (think === 0 ? C4_AGENT : C4_RIVAL)) throw new Error(`Connect Four step ${ahead.step} records a move the board cannot hold`);
        if (who === C4_AGENT) {
          const arrive = smoothstep01((a - 0.55) / 0.45);
          falling = { column, row, who, fall: a, x: columnX(column) };
          hoverX = columnX(c4Cursor(ahead.words));
          hoverAlpha = (opens ? 1 : WAIT) * arrive;
          beam = a < 0.55 ? 1 - a / 0.55 : opens ? arrive : 0;
          beamX = a < 0.55 ? columnX(column) : hoverX;
          seatAlpha = lerp(WAIT, opens ? WAIT : 1, a);
          if (this.budget > 0 && !ahead.done) ringAlpha = a;
        } else {
          falling = { column, row, who, fall: clamp01((a - 0.4) / 0.6), x: lerp(SEAT_X, columnX(column), smoothstep01(a / 0.4)) };
          seatAlpha = (rests ? 1 : WAIT) * smoothstep01((a - 0.5) / 0.5);
          hoverX = lerp(columnX(cursor), columnX(c4Cursor(ahead.words)), smoothstep01(a));
          hoverAlpha = lerp(WAIT, rests ? WAIT : 1, a);
          beam = rests ? 0 : a;
          beamX = hoverX;
          if (this.budget > 0) {
            ring = lerp(ring, this.searched(think + 1), a);
            ringAlpha = 1 - smoothstep01((a - 0.3) / 0.5);
          }
        }
        const finish = ahead.done ? this.finishAt(frame, t + 1) : null;
        if (finish) {
          line = new Set(finish.line);
          lineGlow = strike = smoothstep01((a - 0.6) / 0.4);
          struck = finish;
        }
      } else if (think === 0) {
        const action = c4Action(ahead.actions), aim = c4Cursor(ahead.words), swing = Math.sin(Math.PI * a);
        if (action === C4_DROP) {
          flash = swing;
          hoverZ += 0.2 * swing;
        } else if (aim !== cursor) hoverX = beamX = lerp(columnX(cursor), columnX(aim), smoothstep01(a));
        else hoverX += (action === C4_LEFT ? -1 : 1) * 0.14 * swing;
      } else if (this.budget > 0) ring = lerp(ring, this.searched(after), a);
    }

    const lastColumn = c4LastColumn(now.words), top = lastColumn >= 0 && c4Result(now.words) === 0 ? c4Height(now.words, lastColumn) - 1 : -1;
    const marked = top >= 0 && this.live[c4Slot(lastColumn, top)] === c4LastWho(now.words) ? c4Slot(lastColumn, top) : -1;
    const ended = now.done ? this.finishAt(frame, t) : null;
    if (ended) {
      struck = ended;
      strike = 1 - smoothstep01(a / 0.7);
    }
    this.drawChips(falling, line, lineGlow, marked, ended, a);
    this.drawStrikes(struck, strike);
    this.drawHands(hoverX, hoverZ, hoverAlpha, seatAlpha, ringAlpha);
    this.drawStacks(falling);
    this.ringProgress.value = ring;
    this.ringAlpha.value = ringAlpha;
    this.flashPower.value = flash;
    this.flash!.position.set(columnX(cursor), 0, TOP + 0.14);
    const shaft = this.shaftLamp!, seat = this.seatLamp!;
    shaft.mesh.visible = seat.mesh.visible = true;
    shaft.mesh.position.set(beamX, 0, -C4_ROWS / 2 - 0.1);
    shaft.tint.set(this.accent).multiplyScalar(9 * beam);
    seat.tint.set(RIVAL).multiplyScalar(1.2 + 4 * ring * ringAlpha + 2.5 * Math.max(0, seatAlpha - WAIT));
  }

  private searched(think: number): number {
    return clamp01(this.perStep * (think - 1) / this.budget);
  }

  private drawChips(falling: Falling | null, line: Set<number> | null, lineGlow: number, marked: number, finish: Finish | null, a: number): void {
    const chips = this.chips!, look = this.chipLook!;
    let i = 0, shelf = 0, lamp = 0;
    const seen = this.chipSeen!;
    const place = (x: number, z: number, who: number, glow: number, alpha: number, marker: number, observed: number, occlude: boolean) => {
      this.matrix.makeTranslation(x, 0, z);
      chips.setMatrixAt(i, this.matrix);
      look.setXYZW(i, who === C4_RIVAL ? 1 : 0, glow, alpha, marker);
      seen.setX(i, observed);
      i++;
      if (occlude) {
        const mesh = this.shelves[shelf++]!;
        mesh.visible = true;
        mesh.position.set(x, 0, z + DISC + SHELF / 2);
      }
      if (glow > 0.02 && lamp < LAMPS) {
        const item = this.lamps[lamp++]!;
        item.mesh.visible = true;
        item.mesh.position.set(x, 0, z);
        item.tint.set(who === C4_RIVAL ? RIVAL : this.accent).multiplyScalar(1.8 * glow);
      }
    };
    const covered = falling ? smoothstep01((falling.fall - 0.7) / 0.3) : 0;
    for (let column = 0; column < C4_COLUMNS; column++) for (let row = 0; row < C4_ROWS; row++) {
      const slot = c4Slot(column, row), who = this.live[slot]!;
      if (!who) continue;
      const top = row === this.tops[column] ? (falling?.column === column ? 1 - covered : 1) : 0;
      place(columnX(column), rowZ(row), who, line?.has(slot) ? lineGlow : 0, 1, slot === marked ? 1 - (falling ? a : 0) : 0, top, true);
    }
    if (falling) {
      const s = falling.fall, z = lerp(HOVER_Z, rowZ(falling.row), s * s);
      place(falling.x, z, falling.who, line?.has(c4Slot(falling.column, falling.row)) ? lineGlow : 0, 1, covered, 1, true);
    }
    if (finish) {
      const drop = DRAIN * a * a, fade = 1 - smoothstep01((a - 0.75) / 0.25);
      for (let slot = 0; slot < C4_CELLS; slot++) {
        const who = finish.board[slot]!;
        if (who) place(columnX(Math.floor(slot / C4_ROWS)), rowZ(slot % C4_ROWS) + drop, who, finish.line.includes(slot) ? 1 - 0.6 * a : 0, fade, 0, 0, false);
      }
    }
    chips.count = i;
    chips.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
    seen.needsUpdate = true;
    for (; shelf < this.shelves.length; shelf++) this.shelves[shelf]!.visible = false;
    for (; lamp < LAMPS; lamp++) this.lamps[lamp]!.mesh.visible = false;
  }

  private drawStrikes(finish: Finish | null, alpha: number): void {
    this.strikes.forEach((strike, i) => {
      const run = finish?.runs[i];
      strike.mesh.visible = run !== undefined && alpha > 0.01;
      if (!run || !strike.mesh.visible) return;
      const first = run[0]!, last = run[run.length - 1]!;
      strike.from.set(columnX(Math.floor(first / C4_ROWS)), rowZ(first % C4_ROWS));
      strike.to.set(columnX(Math.floor(last / C4_ROWS)), rowZ(last % C4_ROWS));
      strike.tint.set(finish!.who === C4_RIVAL ? RIVAL : this.accent);
      strike.alpha.value = alpha;
    });
  }

  private drawHands(hoverX: number, hoverZ: number, hoverAlpha: number, seatAlpha: number, ringAlpha: number): void {
    const hands = this.hands!, look = this.handLook!;
    this.matrix.makeTranslation(hoverX, 0, hoverZ);
    hands.setMatrixAt(0, this.matrix);
    look.setXYZW(0, 0, 0, hoverAlpha, 0);
    this.matrix.makeTranslation(SEAT_X, 0, HOVER_Z);
    hands.setMatrixAt(1, this.matrix);
    look.setXYZW(1, 1, 0.12 * ringAlpha, seatAlpha, 0);
    hands.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
  }

  private drawStacks(falling: Falling | null): void {
    const stacks = this.stacks!, look = this.stackLook!, placed = [0, 0];
    let i = 0;
    for (const who of this.live) if (who) placed[who === C4_AGENT ? 0 : 1]!++;
    if (falling) placed[falling.who === C4_AGENT ? 0 : 1]!++;
    for (let side = 0; side < 2; side++) for (let k = 0; k < HALF - placed[side]! - 1; k++) {
      this.matrix.makeTranslation(side ? SEAT_X : -SEAT_X, 0, TABLE - (k + 0.5) * STACK_T);
      stacks.setMatrixAt(i, this.matrix);
      look.setX(i, side);
      i++;
    }
    stacks.count = i;
    stacks.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
  }

  private ledgerOf(frame: FrameUpdate, now: EnvFrame): void {
    const words = now.words, ledger = this.ledger;
    ledger.step = now.step;
    ledger.think = c4Think(words);
    ledger.cursor = c4Cursor(words);
    ledger.member = c4Member(words);
    ledger.game = c4Game(words);
    ledger.reward = now.reward;
    ledger.discs = 0;
    for (let column = 0; column < C4_COLUMNS; column++) ledger.discs += c4Height(words, column);
    const before = now.step > 0 ? frame.frameAt(now.step - 1) : null;
    if (!before || before.step !== now.step - 1) ledger.action = '—';
    else if (c4Think(before.words) > 0) ledger.action = 'waiting';
    else {
      const action = c4Action(now.actions), column = c4Cursor(before.words);
      if (action === C4_DROP) ledger.action = `drop · column ${column + 1}${now.done || c4Think(words) > 0 ? '' : ' full'}`;
      else ledger.action = `${action === C4_LEFT ? '← left' : 'right →'}${c4Cursor(words) === column ? ' · at edge' : ''}`;
    }
  }

  private celebrate(frame: FrameUpdate, pool: Stage2D['pool'], time: number): void {
    const ended = lastEvent(frame.eventLog, 'done', frame.state.step);
    if (ended === undefined || ended === this.celebrated) return;
    this.celebrated = ended;
    const finish = this.finishAt(frame, ended);
    if (!finish) return;
    const cells = finish.line.length ? finish.line : [c4Slot(finish.column, finish.row)];
    this.at.set(0, 0, 0);
    for (const slot of cells) this.at.add(cellAt(Math.floor(slot / C4_ROWS), slot % C4_ROWS));
    this.at.divideScalar(cells.length);
    if (finish.result === C4_WON) {
      pool.burst(this.at, this.accent, 2.2, time);
      pool.burst(this.at, GOLD, 1.4, time);
      pool.pulse.fire(0.7);
    } else if (finish.result === C4_LOST) {
      pool.burst(this.at, RIVAL, 1.6, time);
      pool.sigh(0.6, this.at, time);
    } else pool.burst(this.at.set(0, 0, 0), 0xdcf4ff, 2.4, time);
  }
}

export default (() => new ConnectFourScene()) satisfies SceneFactory;
