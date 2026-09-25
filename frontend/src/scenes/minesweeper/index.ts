import * as THREE from 'three/webgpu';
import { instancedDynamicBufferAttribute, uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { FLAGGED, MINED, MINESWEEPER_FEED, MINESWEEPER_FEED_X, MINESWEEPER_FEED_Y, MINESWEEPER_FLAGS, MINESWEEPER_GAME, MINESWEEPER_OPENED, MINESWEEPER_X, MINESWEEPER_Y, SHOWN, minesweeperPlane } from '@sim/envs/minesweeper';
import type { EnvFrame, GridCell } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { BAR, BLAST, INK, LAMP, LED_HEIGHT, PING, RING, SEVEN, cellMaterial, cursorMaterial, deskMaterial, flat, frameMaterial, ledLayout, ledMaterial, linkMaterial, numerals } from './look';

type N<T extends string> = THREE.Node<T>;

interface Board { mine: Uint8Array; shown: Uint8Array; flag: Uint8Array; count: Uint8Array }
interface Ending { step: number; won: boolean; pre: Board; kind: Uint8Array; order: Float32Array; reach: Float32Array; blast: GridCell }

const PRESS = 4;
const FLAG = 5;
const DX = [0, 0, -1, 1] as const;
const DY = [-1, 1, 0, 0] as const;
const ARROWS = ['↑ up', '↓ down', '← left', 'right →'] as const;
const PLAIN = 0, KEPT = 1, OPEN = 2, BURIED = 3, STRUCK = 4, WRONG = 5, PLANT = 6, FLOOD = 7;

const HOP = 0.62;
const FLIP_U = 0.42;
const PLANT_U = 0.5;
const PINCH_U = 0.5;
const FRESH_STEPS = 4;
const PING_STEPS = 3;
const CHAIN_FROM = 0.04;
const CHAIN = 0.3;
const FLIP = 0.08;
const COVER_FROM = 0.75;
const COVER = 0.5;
const FADE = 0.08;
const FLASH = 0.55;
const SETTLE = COVER_FROM + COVER + FADE;
const CURSOR = 3.6;
const MARGIN = 0.9;
const TABLE = 400;
const ACCENT = TOKENS.accents.Minesweeper!;

const easeBack = (t: number) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2;
const INKS = INK.map((hex) => new THREE.Color(hex));
const minus = (value: number) => value < 0 ? `−${formatCount(-value)}` : formatCount(value);

class MinesweeperScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.32, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'minesweeper';

  private grid: GridSize = { w: 9, h: 9 };
  private mines = 10;
  private safe = 71;
  private digits = 3;
  private actions = 6;
  private around = new Int32Array(0);
  private cells: THREE.InstancedMesh | null = null;
  private shape: THREE.InstancedBufferAttribute | null = null;
  private mark: THREE.InstancedBufferAttribute | null = null;
  private ink: THREE.InstancedBufferAttribute | null = null;
  private cursor: THREE.Mesh | null = null;
  private lamp: THREE.Mesh | null = null;
  private beacon: THREE.Mesh | null = null;
  private flash: THREE.Mesh | null = null;
  private link: THREE.Mesh | null = null;
  private textures: THREE.Texture[] = [];
  private readonly leds: { value: number }[][] = [];
  private readonly heat = uniform(0);
  private readonly pinch = uniform(0);
  private readonly tone = uniform(new THREE.Color(ACCENT));
  private readonly lampLevel = uniform(0.62);
  private readonly beaconLevel = uniform(0);
  private readonly flashLevel = uniform(0);
  private readonly flashTint = uniform(new THREE.Color(BLAST));
  private readonly linkStrength = uniform(0);
  private readonly linkLength = uniform(1);

  private readonly boards = new Map<number, Board>();
  private ending: Ending | null = null;
  private shown: Ending | null = null;
  private floodStep = -1;
  private floodOrder = new Float32Array(0);
  private born = new Int32Array(0);
  private ping = new Float32Array(0);
  private depth = new Int32Array(0);
  private queue = new Int32Array(0);
  private step = -1;
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly spot = { x: 0, y: 0 };
  private readonly accent = new THREE.Color(ACCENT).multiplyScalar(2.4);
  private readonly white = new THREE.Color(0xffffff).multiplyScalar(3);
  private readonly red = new THREE.Color(0xff5a40).multiplyScalar(2.6);
  private readonly ledger = { x: 0, y: 0, flags: 0, opened: 0, game: 1, reward: 0, action: '—', feed: '—', step: 0, won: undefined as number | undefined, lost: undefined as number | undefined, looked: -Infinity, clock: 0 };

  override legend: LegendSpec = {
    title: 'MINESWEEPER',
    rows: [
      { key: 'cursor', label: 'cursor', tint: ACCENT, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
      { key: 'mines', label: 'mines left', tint: 0xe0392c, value: () => minus(this.mines - this.ledger.flags) },
      { key: 'opened', label: 'opened', tint: 0xdde2e8, value: () => `${formatCount(this.ledger.opened)} of ${formatCount(this.safe)}` },
      { key: 'feed', label: 'feed', tint: PING, value: () => this.ledger.feed },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => formatSigned(this.ledger.reward) },
      { key: 'last', label: 'last game', tint: 0xffe2a8, value: () => this.lastGame() },
    ],
    note: () => `game ${formatCount(this.ledger.game)}`,
    bar: () => this.ledger.opened / this.safe,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, w = parameter(env, 'width'), h = parameter(env, 'height');
    this.grid = { w, h };
    this.mines = parameter(env, 'mines');
    this.actions = env.actions;
    this.safe = w * h - this.mines;
    this.digits = Math.max(3, String(this.mines - w * h).length, String(this.safe).length);
    return new THREE.Box2(new THREE.Vector2(-w / 2 - RING - 0.45, -h / 2 - RING - BAR - 0.45), new THREE.Vector2(w / 2 + RING + 0.45, h / 2 + RING + 1.7));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xe8ecf2).multiplyScalar(0.62);
  }

  protected override poolOptions() {
    return { burstSize: 72 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: () => {
        agent.x = this.ledger.x;
        agent.y = this.ledger.y;
        return agent;
      },
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step), board = this.boardOf(now), { w, h } = this.grid;
        const mined = kit.marks('mines', w * h), flagged = kit.marks('flags', w * h);
        for (let c = 0; c < w * h; c++) {
          this.centre(c % w, Math.floor(c / w), this.at);
          if (board.mine[c]) mined.set(this.at.x, 0.05, this.at.z, 0.16, 0xff4a3a);
          if (board.flag[c]) flagged.set(this.at.x + 0.3, 0.05, this.at.z - 0.3, 0.08, board.mine[c] ? 0xffc34d : 0xc86bff);
        }
        const ended = kit.marks('ended', w * h), last = this.shown;
        if (last) for (let c = 0; c < w * h; c++) {
          if (!last.pre.mine[c]) continue;
          this.centre(c % w, Math.floor(c / w), this.at);
          ended.set(this.at.x - 0.3, 0.05, this.at.z + 0.3, 0.09, last.kind[c] === STRUCK ? 0xffffff : 0xff9a3a);
        }
        const x = now.words[MINESWEEPER_X]!, y = now.words[MINESWEEPER_Y]!, window = kit.curve('window', 5, TOKENS.uiCyan);
        for (const [dx, dy] of [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]] as const) {
          this.centre(x + dx, y + dy, this.at);
          window.point(this.at.x, 0.06, this.at.z);
        }
        const fed = this.feedOf(now), beam = kit.curve('feed', 2, PING), mark = kit.marks('feed', 1);
        if (!fed) return;
        this.centre(fed.x, fed.y, this.at);
        mark.set(this.at.x, 0.05, this.at.z, 0.12, PING);
        beam.point(this.at.x, 0.06, this.at.z);
        this.centre(x, y, this.at);
        beam.point(this.at.x, 0.06, this.at.z);
      },
    };
  }

  private act(frame: EnvFrame): number {
    const control = frame.actions[0]!;
    return control >= 0 && control < this.actions ? Math.trunc(control) : 0;
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  protected build(_ctx: SceneContext, { light }: Stage2D): void {
    const { w, h } = this.grid, cells = w * h, lit = light.lit;
    this.scene.background = new THREE.Color(0x0d1116);
    this.around = new Int32Array(cells * 8).fill(-1);
    for (let c = 0; c < cells; c++) {
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const x = c % w + dx, y = Math.floor(c / w) + dy;
        if ((dx || dy) && x >= 0 && y >= 0 && x < w && y < h) this.around[c * 8 + k++] = y * w + x;
      }
    }
    this.floodOrder = new Float32Array(cells);
    this.born = new Int32Array(cells);
    this.ping = new Float32Array(cells);
    this.depth = new Int32Array(cells);
    this.queue = new Int32Array(cells);

    const desk = new THREE.Mesh(flat(TABLE, TABLE), deskMaterial(lit));
    desk.renderOrder = -4;
    desk.frustumCulled = false;

    const width = w + 2 * RING + 2 * MARGIN, depth = h + 2 * RING + BAR + 2 * MARGIN;
    const frame = new THREE.Mesh(flat(width, depth), frameMaterial({ w, h, digits: this.digits }, width, depth, BAR / 2, lit));
    frame.position.set(0, 0, -BAR / 2);
    frame.renderOrder = -3;
    frame.frustumCulled = false;

    const atlas = numerals();
    this.textures.push(atlas);
    const count = 2 * cells;
    this.shape = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.mark = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.ink = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const material = cellMaterial(atlas, instancedDynamicBufferAttribute(this.shape, 'vec4') as N<'vec4'>, instancedDynamicBufferAttribute(this.mark, 'vec4') as N<'vec4'>, instancedDynamicBufferAttribute(this.ink, 'vec4') as N<'vec4'>, this.heat, lit);
    const tiles = instancedMesh(flat(1, 1), material, count);
    for (let i = 0; i < count; i++) {
      const c = i % cells;
      this.centre(c % w, Math.floor(c / w), this.at);
      tiles.setMatrixAt(i, new THREE.Matrix4().makeTranslation(this.at.x, 0, this.at.z));
    }
    tiles.renderOrder = -2;
    tiles.frustumCulled = false;
    this.cells = tiles;

    const layout = ledLayout(w, h, this.digits);
    for (const side of [-1, 1]) {
      const masks = Array.from({ length: this.digits }, () => uniform(0));
      this.leds.push(masks);
      const led = new THREE.Mesh(flat(layout.width, LED_HEIGHT), ledMaterial(masks as unknown as N<'float'>[], layout.width));
      led.position.set(side * layout.x, 0, -layout.y);
      led.renderOrder = -1;
      this.scene.add(led);
    }

    this.link = new THREE.Mesh(flat(1, 0.09), linkMaterial(this.linkStrength, this.linkLength));
    this.link.renderOrder = 4;
    this.link.visible = false;
    this.cursor = new THREE.Mesh(flat(CURSOR, CURSOR), cursorMaterial(this.pinch, this.tone as unknown as N<'vec3'>, CURSOR));
    this.cursor.renderOrder = 6;
    this.lamp = light.emitter(flat(2.9, 2.9), rgb(LAMP).mul(this.lampLevel));
    this.beacon = light.emitter(new THREE.CircleGeometry(0.16, 24).rotateX(-Math.PI / 2), rgb(PING).mul(this.beaconLevel));
    this.flash = light.emitter(new THREE.CircleGeometry(0.6, 32).rotateX(-Math.PI / 2), (this.flashTint as unknown as N<'vec3'>).mul(this.flashLevel));
    this.beacon.visible = this.flash.visible = false;
    this.scene.add(desk, frame, tiles, this.link, this.cursor, this.lamp, this.beacon, this.flash);
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.cells?.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.ledger.clock += frame.dtReal;
    const hz = frame.controlHz, step = frame.state.step, now = frame.frameAt(step), ahead = frame.frameAt(step + 1);
    const next = ahead.step === step + 1 ? ahead : null, u = next ? clamp01(frame.position - step) : 0;
    if (!frame.scrubbed && this.step >= 0 && step > this.step) for (let s = this.step + 1; s <= step; s++) this.celebrate(frame, s, pool);
    if (step !== this.step || frame.scrubbed) this.record(frame, step, now);
    this.step = step;
    const ending = this.endingNear(frame, step, next), tau = ending ? (frame.position - ending.step) / hz : Infinity;
    this.shown = ending && tau < SETTLE ? ending : null;
    this.paintLive(frame, step, now, next && !next.done ? next : null, u);
    this.paintGhost(ending, tau, hz);
    this.placeCursor(now, next, u);
    this.placeFeed(now, u);
    this.blast(ending, tau);
    for (const attribute of [this.shape, this.mark, this.ink]) if (attribute) attribute.needsUpdate = true;
  }

  private boardOf(frame: EnvFrame): Board {
    const known = this.boards.get(frame.step);
    if (known) return known;
    if (this.boards.size > 32) this.boards.clear();
    const { w, h } = this.grid, cells = w * h;
    const board: Board = { mine: new Uint8Array(cells), shown: new Uint8Array(cells), flag: new Uint8Array(cells), count: new Uint8Array(cells) };
    for (let c = 0; c < cells; c++) {
      board.mine[c] = minesweeperPlane(frame, MINED, cells, c);
      board.shown[c] = minesweeperPlane(frame, SHOWN, cells, c);
      board.flag[c] = minesweeperPlane(frame, FLAGGED, cells, c);
    }
    for (let c = 0; c < cells; c++) {
      let n = 0;
      for (let k = 0; k < 8; k++) { const o = this.around[c * 8 + k]!; if (o >= 0) n += board.mine[o]!; }
      board.count[c] = n;
    }
    this.boards.set(frame.step, board);
    return board;
  }

  private feedOf(frame: EnvFrame): GridCell | null {
    if (!frame.words[MINESWEEPER_FEED]) return null;
    const { w, h } = this.grid;
    return { x: Math.round(frame.observations[MINESWEEPER_FEED_X]! * (w - 1)), y: Math.round(frame.observations[MINESWEEPER_FEED_Y]! * (h - 1)) };
  }

  private spread(fresh: (c: number) => boolean, origin: number, out: Float32Array): void {
    const depth = this.depth, queue = this.queue;
    depth.fill(-1);
    out.fill(0);
    depth[origin] = 0;
    let head = 0, tail = 0, deepest = 0;
    queue[tail++] = origin;
    while (head < tail) {
      const c = queue[head++]!;
      for (let k = 0; k < 8; k++) {
        const o = this.around[c * 8 + k]!;
        if (o < 0 || depth[o]! >= 0 || !fresh(o)) continue;
        depth[o] = depth[c]! + 1;
        deepest = Math.max(deepest, depth[o]!);
        queue[tail++] = o;
      }
    }
    for (let c = 0; c < out.length; c++) if (fresh(c)) out[c] = (depth[c]! < 0 ? deepest : depth[c]!) / (deepest + 1);
  }

  private endingNear(frame: FrameUpdate, step: number, next: EnvFrame | null): Ending | null {
    if (next?.done) return this.endingAt(frame, next.step);
    for (let s = step; s > step - Math.ceil(SETTLE * frame.controlHz) && s > 0; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) return null;
      if (f.done) return this.endingAt(frame, s);
    }
    return null;
  }

  private endingAt(frame: FrameUpdate, step: number): Ending | null {
    if (this.ending?.step === step) return this.ending;
    const last = frame.frameAt(step), before = frame.frameAt(step - 1);
    if (last.step !== step || before.step !== step - 1 || !last.done) return null;
    const { w, h } = this.grid, cells = w * h, pre = this.boardOf(before), won = last.reward > 0;
    const x = last.words[MINESWEEPER_X]!, y = last.words[MINESWEEPER_Y]!, pressed = y * w + x;
    if (this.act(last) !== PRESS || x !== before.words[MINESWEEPER_X] || y !== before.words[MINESWEEPER_Y]) throw new Error(`Minesweeper step ${step} ended a game without a press in place`);
    const struck: number[] = [];
    if (!won && !pre.shown[pressed]) struck.push(pressed);
    if (!won && pre.shown[pressed]) for (let k = 0; k < 8; k++) {
      const o = this.around[pressed * 8 + k]!;
      if (o >= 0 && !pre.shown[o] && !pre.flag[o] && pre.mine[o]) struck.push(o);
    }
    if (!won && (!struck.length || struck.some((c) => !pre.mine[c] || pre.flag[c]))) throw new Error(`Minesweeper step ${step} lost without pressing onto a mine`);
    const kind = new Uint8Array(cells), order = new Float32Array(cells), reach = new Float32Array(cells);
    for (let c = 0; c < cells; c++) kind[c] = pre.shown[c] ? OPEN : won ? (pre.mine[c] ? (pre.flag[c] ? KEPT : PLANT) : FLOOD) : pre.flag[c] ? (pre.mine[c] ? KEPT : WRONG) : pre.mine[c] ? BURIED : PLAIN;
    for (const c of struck) kind[c] = STRUCK;
    if (won) this.spread((c) => kind[c] === FLOOD, pressed, order);
    const blast = won ? { x, y } : { x: struck.reduce((sum, c) => sum + c % w, 0) / struck.length, y: struck.reduce((sum, c) => sum + Math.floor(c / w), 0) / struck.length };
    let far = 1;
    for (let c = 0; c < cells; c++) far = Math.max(far, Math.hypot(c % w - blast.x, Math.floor(c / w) - blast.y));
    for (let c = 0; c < cells; c++) reach[c] = Math.hypot(c % w - blast.x, Math.floor(c / w) - blast.y) / far;
    this.ending = { step, won, pre, kind, order, reach, blast };
    return this.ending;
  }

  private celebrate(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    const f = frame.frameAt(step);
    if (f.step !== step || !f.done) return;
    const ending = this.endingAt(frame, step);
    if (!ending) return;
    const born = step / frame.controlHz, size = Math.max(this.grid.w, this.grid.h);
    this.centre(ending.blast.x, ending.blast.y, this.at);
    if (ending.won) {
      pool.burst(this.other.copy(this.at).setY(38), TOKENS.celebrationGold, 1.3, born);
      pool.ripple(this.at, TOKENS.celebrationGold, size / 4, born);
      pool.pulse.fire(0.6);
    } else {
      pool.burst(this.other.copy(this.at).setY(38), BLAST, 0.9, born);
      pool.ripple(this.at, 0xff6a3d, size / 5, born);
      pool.pulse.fire(0.4);
    }
  }

  private freshness(frame: FrameUpdate, step: number, board: Board): Int32Array {
    const born = this.born;
    for (let c = 0; c < born.length; c++) born[c] = board.shown[c] ? step : -1;
    let s = step - 1;
    for (; s >= step - FRESH_STEPS && s >= 0; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s || frame.frameAt(s + 1).done) break;
      const earlier = this.boardOf(f);
      for (let c = 0; c < born.length; c++) if (born[c] === s + 1 && earlier.shown[c]) born[c] = s;
    }
    if (s < step - FRESH_STEPS) for (let c = 0; c < born.length; c++) if (born[c] === step - FRESH_STEPS) born[c] = -1;
    return born;
  }

  private paintLive(frame: FrameUpdate, step: number, now: EnvFrame, next: EnvFrame | null, u: number): void {
    const shape = this.shape!, mark = this.mark!, ink = this.ink!, a = this.boardOf(now), b = next ? this.boardOf(next) : null;
    if (b && this.floodStep !== step) {
      this.floodStep = step;
      const origin = next!.words[MINESWEEPER_Y]! * this.grid.w + next!.words[MINESWEEPER_X]!;
      this.spread((c) => !a.shown[c] && b.shown[c] === 1, origin, this.floodOrder);
    }
    const born = this.freshness(frame, step, a), ping = this.ping, at = frame.position;
    ping.fill(0);
    for (let s = step; s > step - PING_STEPS && s >= 0; s--) {
      const f = frame.frameAt(s), fed = f.step === s ? this.feedOf(f) : null;
      if (fed) ping[fed.y * this.grid.w + fed.x] = Math.max(ping[fed.y * this.grid.w + fed.x]!, (1 - (at - s) / PING_STEPS) ** 1.5);
    }
    const plant = easeBack(clamp01(u / PLANT_U));
    for (let c = 0; c < a.shown.length; c++) {
      let raise = 1, flag = a.flag[c]!, count = 0, fresh = 0;
      if (a.shown[c]) {
        raise = 0;
        count = a.count[c]!;
        fresh = born[c]! >= 0 ? Math.max(0, 1 - (at - born[c]!) / FRESH_STEPS) ** 2 : 0;
      } else if (b?.shown[c]) {
        const t = clamp01((u - this.floodOrder[c]! * (1 - FLIP_U)) / FLIP_U);
        raise = 1 - smoothstep01(t);
        count = b.count[c]!;
        fresh = t;
        flag = 0;
      } else if (b) flag = Math.max(0, a.flag[c]! + (b.flag[c]! - a.flag[c]!) * plant);
      const hue = INKS[count]!;
      shape.setXYZW(c, raise, flag, 0, count);
      mark.setXYZW(c, 0, 0, 1, fresh);
      ink.setXYZW(c, hue.r, hue.g, hue.b, ping[c]!);
    }
  }

  private paintGhost(ending: Ending | null, tau: number, hz: number): void {
    const shape = this.shape!, mark = this.mark!, ink = this.ink!, cells = this.grid.w * this.grid.h;
    for (let c = 0; c < cells; c++) {
      const i = cells + c, k = ending ? ending.kind[c]! : PLAIN;
      if (!ending || tau >= SETTLE) {
        mark.setXYZW(i, 0, 0, 0, 0);
        continue;
      }
      const chain = clamp01((tau - CHAIN_FROM - CHAIN * ending.reach[c]!) / FLIP), press = clamp01((tau * hz + 1 - ending.order[c]! * (1 - FLIP_U)) / FLIP_U);
      let raise = 1, flag = 0, mine = 0, count = 0, struck = 0, cross = 0;
      if (k === KEPT) flag = 1;
      else if (k === OPEN) { raise = 0; count = ending.pre.count[c]!; }
      else if (k === BURIED) { raise = 1 - smoothstep01(chain); mine = chain > 0 ? easeBack(chain) : 0; }
      else if (k === STRUCK) { raise = 1 - smoothstep01(press); mine = press > 0 ? easeBack(press) : 0; struck = press; }
      else if (k === WRONG) { flag = 1; cross = chain; }
      else if (k === PLANT) flag = chain > 0 ? easeBack(chain) : 0;
      else if (k === FLOOD) { raise = 1 - smoothstep01(press); count = ending.pre.count[c]!; }
      const hue = INKS[count]!;
      shape.setXYZW(i, raise, flag, mine, count);
      const wipe = clamp01((tau - COVER_FROM - COVER * ending.reach[c]!) / FADE);
      mark.setXYZW(i, struck, cross, 1 - wipe, Math.sin(Math.PI * wipe));
      ink.setXYZW(i, hue.r, hue.g, hue.b, 0);
    }
  }

  private placeCursor(now: EnvFrame, next: EnvFrame | null, u: number): void {
    const cursor = this.cursor, lamp = this.lamp;
    if (!cursor || !lamp) return;
    const ax = now.words[MINESWEEPER_X]!, ay = now.words[MINESWEEPER_Y]!, bx = next ? next.words[MINESWEEPER_X]! : ax, by = next ? next.words[MINESWEEPER_Y]! : ay;
    const act = next ? this.act(next) : -1, hop = clamp01(u / HOP), glide = smoothstep01(hop);
    let x = ax + (bx - ax) * glide, y = ay + (by - ay) * glide;
    if (act >= 0 && act < 4 && ax === bx && ay === by) {
      const bump = 0.12 * Math.sin(Math.PI * hop);
      x += DX[act]! * bump;
      y += DY[act]! * bump;
    }
    const pinch = act === PRESS || act === FLAG ? Math.sin(Math.PI * clamp01(u / PINCH_U)) : 0;
    this.pinch.value = pinch;
    this.tone.value.copy(this.accent).lerp(act === FLAG ? this.red : this.white, 0.55 * pinch);
    this.lampLevel.value = 0.62 * (1 + 0.4 * pinch);
    this.spot.x = x;
    this.spot.y = y;
    this.centre(x, y, cursor.position);
    lamp.position.copy(cursor.position);
  }

  private placeFeed(now: EnvFrame, u: number): void {
    const link = this.link, beacon = this.beacon, fed = this.feedOf(now);
    if (!link || !beacon) return;
    link.visible = beacon.visible = fed !== null;
    if (!fed) return;
    this.centre(fed.x, fed.y, this.at);
    this.centre(this.spot.x, this.spot.y, this.other);
    beacon.position.copy(this.at);
    this.beaconLevel.value = 1.1 * (1 - 0.5 * u);
    const dx = this.at.x - this.other.x, dz = this.at.z - this.other.z, length = Math.hypot(dx, dz);
    link.position.set((this.at.x + this.other.x) / 2, 0, (this.at.z + this.other.z) / 2);
    link.rotation.y = Math.atan2(-dz, dx);
    link.scale.set(Math.max(length, 1e-3), 1, 1);
    this.linkLength.value = length;
    this.linkStrength.value = 0.42 * (1 - 0.5 * u);
  }

  private blast(ending: Ending | null, tau: number): void {
    const flash = this.flash;
    if (!flash) return;
    const k = ending && tau >= 0 && tau < FLASH ? (1 - tau / FLASH) ** 2 : 0;
    flash.visible = k > 0;
    this.heat.value = ending && !ending.won && tau < SETTLE ? 0.25 + 0.75 * k : 0;
    if (!ending || !flash.visible) return;
    this.flashTint.value.set(ending.won ? TOKENS.celebrationGold : BLAST);
    this.flashLevel.value = k * (ending.won ? 1.6 : 1.8);
    this.centre(ending.blast.x, ending.blast.y, flash.position);
  }

  private record(frame: FrameUpdate, step: number, now: EnvFrame): void {
    const ledger = this.ledger;
    ledger.step = step;
    ledger.x = now.words[MINESWEEPER_X]!;
    ledger.y = now.words[MINESWEEPER_Y]!;
    ledger.flags = now.words[MINESWEEPER_FLAGS]!;
    ledger.opened = now.words[MINESWEEPER_OPENED]!;
    ledger.game = now.words[MINESWEEPER_GAME]! + 1;
    ledger.reward = now.reward;
    ledger.action = this.describe(frame, step, now);
    const fed = this.feedOf(now);
    ledger.feed = fed ? `column ${fed.x + 1} · row ${fed.y + 1}` : '—';
    if (frame.scrubbed || ledger.clock - ledger.looked > 1) {
      ledger.won = lastEvent(frame.eventLog, 'goal', step);
      ledger.lost = lastEvent(frame.eventLog, 'harm', step);
      ledger.looked = ledger.clock;
    }
    if (now.done && now.reward > 0) ledger.won = step;
    if (now.done && now.reward < 0) ledger.lost = step;
    this.show(this.leds[0], this.mines - ledger.flags);
    this.show(this.leds[1], this.safe - ledger.opened);
  }

  private show(masks: { value: number }[] | undefined, value: number): void {
    if (!masks) return;
    const text = value < 0 ? `-${String(-value).padStart(masks.length - 1, '0')}` : String(value).padStart(masks.length, '0');
    masks.forEach((mask, k) => { mask.value = SEVEN[text[text.length - masks.length + k]!] ?? 0; });
  }

  private describe(frame: FrameUpdate, step: number, now: EnvFrame): string {
    const before = step > 0 ? frame.frameAt(step - 1) : null;
    if (!before || before.step !== step - 1) return '—';
    const act = this.act(now), x = before.words[MINESWEEPER_X]!, y = before.words[MINESWEEPER_Y]!, c = y * this.grid.w + x;
    if (act < PRESS) return now.words[MINESWEEPER_X] === x && now.words[MINESWEEPER_Y] === y ? `${ARROWS[act]} · edge` : ARROWS[act]!;
    const pre = this.boardOf(before);
    if (act === FLAG) return pre.shown[c] ? 'flag · no effect' : pre.flag[c] ? 'unflag' : 'flag';
    if (pre.flag[c]) return 'press · flagged';
    const chord = pre.shown[c] === 1;
    if (now.done) return `${chord ? 'chord' : 'reveal'} · ${now.reward > 0 ? 'cleared' : 'mine'}`;
    const opened = now.words[MINESWEEPER_OPENED]! - before.words[MINESWEEPER_OPENED]!, cells = `${formatCount(opened)} cell${opened === 1 ? '' : 's'}`;
    if (chord) return opened > 0 ? `chord · ${cells}` : 'chord · no effect';
    return before.words[MINESWEEPER_OPENED] === 0 && pre.mine[c] ? `reveal · ${cells} · mine moved` : `reveal · ${cells}`;
  }

  private lastGame(): string {
    const { won, lost, step } = this.ledger;
    if (won === undefined && lost === undefined) return 'none yet';
    const latest = Math.max(won ?? -1, lost ?? -1), which = latest === won ? 'won' : 'lost';
    return latest === step ? `${which} now` : `${which} · ${formatCount(step - latest)} steps ago`;
  }
}

export default (() => new MinesweeperScene()) satisfies SceneFactory;
