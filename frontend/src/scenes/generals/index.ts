import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  CAPITAL, CITY, GEN_CURSOR_X, GEN_CURSOR_Y, GW, JUMP_KEY, NOBODY, PEAK, STEP_X, STEP_Y,
  armyOf, cellX, cellY, checkPatch, generalsShape, heeded, keyOf, orderOf, ownerOf, patchCell, sightOf, terrainOf, type GeneralsShape, type Order,
} from '@sim/envs/generals';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { FLOOD_STEPS, INKS, NEUTRAL, RIM, SWEEP_STEPS, boardMaterial, tableMesh, type Board } from './board';
import { numerals, type Numerals } from './glyphs';
import { MARCH_STEPS, SHOCK_STEPS, marksMaterial, type Marks } from './marks';
import { PANEL_WIDTH, panelHeight, panelMaterial, rowTop, type Panel } from './panel';

const ACCENT = TOKENS.accents.Generals!;
const KEY_NAMES = ['↑', '↓', '←', '→', 'mode', 'jump'] as const;
const MODE_NAMES = ['look', 'move all', 'move half'] as const;
const CAPITAL_LIGHT = 4.5;
const CITY_LIGHT = 2;
const CURSOR_LIGHT = 0.8;
const FOLLOW = 9;

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}`;
const signed = (value: number) => (value > 0 ? `+${formatCount(value)}` : value < 0 ? `−${formatCount(-value)}` : formatSigned(value));

interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }
interface Standing { player: number; land: number; army: number; alive: boolean }
interface Shock { index: number; tint: number }

class GeneralsScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 16;
  protected readonly stageName = 'generals';

  private shape: GeneralsShape = { side: 1, players: 2, cities: 0, substeps: 1, turn: 1, round: 1, reach: 0, codes: [] };
  private grid: GridSize = { w: 1, h: 1 };
  private numerals: Numerals | null = null;
  private board: Board | null = null;
  private marks: Marks | null = null;
  private panel: Panel | null = null;
  private table: THREE.Mesh | null = null;
  private shade: THREE.Mesh | null = null;
  private lamps: Lamp[] = [];
  private cursorLamp: Lamp | null = null;
  private readonly origin = new THREE.Vector2();
  private scale = 1;
  private rowY = new Float32Array(0);
  private rowGoal = new Float32Array(0);
  private settled = false;

  private tickAt = -1;
  private arrived = -Infinity;
  private relieved = -1;
  private sightNow = new Uint8Array(0);
  private sightWas = new Uint8Array(0);
  private shock: Shock | null = null;
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { turn: 0, land: 0, army: 0, rank: 1, alive: 2, key: -1, mode: 0, reward: 0, game: 0, share: 0 };

  override legend: LegendSpec = {
    title: 'GENERALS',
    rows: [
      { key: 'realm', label: 'your realm', tint: ACCENT, value: () => `${formatCount(this.ledger.land)} land · ${formatCount(this.ledger.army)} army` },
      { key: 'rank', label: 'rank', tint: 0xdcf4ff, value: () => `${ordinal(this.ledger.rank)} of ${this.ledger.alive}` },
      { key: 'key', label: 'key', tint: 0x9fd8ff, value: () => `${this.ledger.key < 0 ? '—' : KEY_NAMES[this.ledger.key]} · ${MODE_NAMES[this.ledger.mode]}` },
      { key: 'turn', label: 'turn', tint: 0xf4c95e, value: () => formatCount(this.ledger.turn) },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(this.ledger.reward) },
    ],
    note: () => `game ${formatCount(this.ledger.game + 1)} · ${this.ledger.alive} of ${this.shape.players} generals stand`,
    bar: () => this.ledger.share,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.shape = generalsShape(ctx.environment);
    const side = this.shape.side, players = this.shape.players;
    this.grid = { w: side, h: side };
    this.sightNow = new Uint8Array(side * side);
    this.sightWas = new Uint8Array(side * side);
    this.scale = side / 32;
    this.rowY = new Float32Array(players);
    this.rowGoal = new Float32Array(players);
    const left = side / 2 + RIM + (1.1 + PANEL_WIDTH) * this.scale, half = side / 2 + 0.6;
    this.origin.set(-left, -side / 2 + Math.min(0.12 * side, side - panelHeight(players) * this.scale));
    return new THREE.Box2(new THREE.Vector2(-left - 0.5 * this.scale, -half), new THREE.Vector2(left + 0.5 * this.scale, half));
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { sighRadius: 0.1 * this.shape.side, sighHeight: 2 };
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff3e2).multiplyScalar(0.66);
  }

  private centre(cell: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(cellX(cell), cellY(cell), this.grid, out);
  }

  private spot(index: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(index % this.shape.side, Math.floor(index / this.shape.side), this.grid, out);
  }

  private inkOf(owner: number, seat: number): number {
    if (owner === NOBODY) return NEUTRAL;
    if (owner === seat) return ACCENT;
    return INKS[owner < seat ? owner : owner - 1]!;
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[GEN_CURSOR_X]!;
        agent.y = state.pose[GEN_CURSOR_Y]!;
        return agent;
      },
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step);
        if (now.step !== frame.state.step) return;
        const side = this.shape.side, seat = now.words[GW.SEAT]!, sight = sightOf(now.view, side, seat, new Uint8Array(side * side));
        const owners = kit.bars('owners', side * side), terrain = kit.marks('terrain', side * side), fog = kit.marks('fog', side * side);
        for (let i = 0; i < side * side; i++) {
          const word = now.view[i]!, owner = ownerOf(word), kind = terrainOf(word), army = armyOf(word), size = 0.06 + 0.04 * Math.min(6, Math.log2(1 + army));
          this.spot(i, this.at);
          if (owner !== NOBODY || army > 0) owners.set(this.at.x - 0.44 + size / 2, 0.02, this.at.z - 0.44 + size / 2, 0.01, this.inkOf(owner, seat), size, size);
          if (kind !== 0) terrain.set(this.at.x - 0.34, 0.05, this.at.z + 0.34, kind === CAPITAL ? 0.12 : 0.08, kind === PEAK ? 0x8a857c : kind === CITY ? 0xffffff : TOKENS.celebrationGold);
          if (!sight[i]) fog.set(this.at.x + 0.34, 0.05, this.at.z + 0.34, 0.06, 0x2a3346);
        }
        const span = 2 * this.shape.reach + 1, patch = kit.marks('patch', span * span), cursor = now.words[GW.CURSOR]!, cell: GridCell = { x: 0, y: 0 };
        for (let i = 0; i < span * span; i++) {
          const { x: u, y: v } = patchCell(cursor, this.shape.reach, i, cell);
          if (u < 0 || v < 0 || u >= side || v >= side) continue;
          cellToWorldVec(u, v, this.grid, this.at);
          patch.set(this.at.x + 0.34, 0.06, this.at.z - 0.34, 0.07, TOKENS.uiCyan);
        }
        const order = orderOf(now.words), arrow = kit.curve('order', 5, ACCENT);
        if (!order) return;
        this.centre(order.cell, this.at);
        const dx = STEP_X[order.dir]!, dz = STEP_Y[order.dir]!, tip = { x: this.at.x + dx * 0.9, z: this.at.z + dz * 0.9 };
        arrow.point(this.at.x, 0.08, this.at.z);
        arrow.point(tip.x, 0.08, tip.z);
        arrow.point(tip.x - dx * 0.2 - dz * 0.14, 0.08, tip.z - dz * 0.2 + dx * 0.14);
        arrow.point(tip.x, 0.08, tip.z);
        arrow.point(tip.x - dx * 0.2 + dz * 0.14, 0.08, tip.z - dz * 0.2 - dx * 0.14);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const side = this.shape.side, set = numerals();
    this.numerals = set;
    this.scene.background = new THREE.Color(0x0b0f16);
    this.table = tableMesh(side, light.lit);
    this.board = boardMaterial(side, set, light.lit, ctx.seed);
    this.marks = marksMaterial(side, ACCENT);
    this.panel = panelMaterial(this.shape.players, this.shape.substeps, set, ACCENT, this.origin, this.scale);
    this.scene.add(this.table, this.board.mesh, this.marks.mesh, this.panel.mesh);
    const disc = (radius: number, tint: THREE.Color): Lamp => {
      const mesh = light.emitter(new THREE.CircleGeometry(radius, 24).rotateX(-Math.PI / 2), uniform(tint) as unknown as THREE.Node<'vec3'>);
      this.scene.add(mesh);
      return { mesh, tint };
    };
    for (let i = this.shape.players + this.shape.cities; i > 0; i--) this.lamps.push(disc(0.2, new THREE.Color()));
    this.cursorLamp = disc(0.16, new THREE.Color(0xfff4e0).multiplyScalar(CURSOR_LIGHT));
  }

  protected override drop(): void {
    this.board?.dispose();
    this.marks?.dispose();
    this.panel?.dispose();
    this.numerals?.texture.dispose();
    for (const mesh of [this.shade, this.table, ...this.lamps.map((lamp) => lamp.mesh), this.cursorLamp?.mesh]) {
      if (!mesh) continue;
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const board = this.board, marks = this.marks;
    if (!board || !marks) return;
    const step = frame.state.step, now = frame.frameAt(step);
    if (now.step !== step) return;
    const at = step - now.words[GW.PHASE]!;
    if (at !== this.tickAt) this.land(frame, at, now, pool, time);
    const age = frame.position - at;
    const settle = this.arrived === -Infinity ? 1e3 : Math.max(age, (time - this.arrived) * this.controlHz);
    board.uniforms.age.value = settle;
    board.uniforms.front.value = marks.front.value = board.uniforms.sweep.value ? settle * (2 * this.shape.side + 2) / SWEEP_STEPS - 1 : -1e3;
    board.uniforms.glow.value = age;
    board.uniforms.time.value = marks.time.value = time;
    marks.fall.value = this.shock && age < SHOCK_STEPS ? age : -1;
    for (const event of frame.events) if (event.type === 'harm' || event.type === 'goal') this.finish(frame, event.type === 'goal', pool, time);
    this.steer(frame, step, now, age);
    const ledger = this.ledger, tick = now.words[GW.TICK]!;
    ledger.reward = now.reward;
    ledger.key = step > 0 ? keyOf(now) : -1;
    ledger.mode = now.words[GW.MODE]!;
    ledger.turn = Math.floor(tick / this.shape.turn);
    ledger.game = now.words[GW.GAME]!;
    const panel = this.panel!;
    panel.phase.value = now.words[GW.PHASE]!;
    panel.round.value = (tick % this.shape.round) / this.shape.round;
    panel.turn.value = ledger.turn;
    panel.key.value = ledger.key;
    panel.pressed.value = frame.position - step;
    panel.mode.value = ledger.mode;
    panel.sweep.value = now.words[GW.SWEEP]!;
    const follow = this.settled && !frame.scrubbed ? 1 - Math.exp(-frame.dtReal * FOLLOW) : 1, rows = panel.rows.image.data as Float32Array;
    for (let p = 0; p < this.shape.players; p++) rows[p * 4 + 3] = this.rowY[p]! += (this.rowGoal[p]! - this.rowY[p]!) * follow;
    this.settled = true;
    panel.rows.needsUpdate = true;
  }

  private land(frame: FrameUpdate, at: number, now: EnvFrame, pool: CelebrationPool, time: number): void {
    const board = this.board!, side = this.shape.side, cells = side * side, seat = now.words[GW.SEAT]!;
    const before = at > 0 ? frame.frameAt(at - 1) : null, known = before !== null && before.step === at - 1;
    const fresh = frame.frameAt(at), ended = fresh.step === at && fresh.done !== 0, newGame = at === 0 || ended || (known && before.words[GW.GAME] !== now.words[GW.GAME]);
    const live = !frame.scrubbed && at > this.tickAt && at - this.tickAt <= 4 * this.shape.substeps;
    const was = known ? before : now, formerSeat = was.words[GW.SEAT]!;
    this.paint(board.uniforms.palette, seat);
    this.paint(board.uniforms.former, formerSeat);
    this.paint(this.panel!.palette, seat);
    sightOf(now.view, side, seat, this.sightNow);
    checkPatch(now, this.shape, this.sightNow);
    sightOf(was.view, side, formerSeat, this.sightWas);
    const nowData = board.now.image.data as Float32Array, wasData = board.was.image.data as Float32Array, wipe = board.wipe.image.data as Float32Array;
    let fallen = -1, victim = NOBODY, victor = NOBODY;
    for (let i = 0; i < cells; i++) {
      const a = now.view[i]!, b = was.view[i]!;
      nowData[4 * i] = armyOf(a);
      nowData[4 * i + 1] = ownerOf(a);
      nowData[4 * i + 2] = terrainOf(a);
      nowData[4 * i + 3] = this.sightNow[i]!;
      wasData[4 * i] = armyOf(b);
      wasData[4 * i + 1] = ownerOf(b);
      wasData[4 * i + 2] = terrainOf(b);
      wasData[4 * i + 3] = this.sightWas[i]!;
      if (!newGame && terrainOf(b) === CAPITAL && terrainOf(a) === CITY) {
        fallen = i;
        victim = ownerOf(b);
        victor = ownerOf(a);
      }
    }
    for (let i = 0; i < cells; i++) {
      const a = ownerOf(now.view[i]!), b = ownerOf(was.view[i]!), flooded = fallen >= 0 && b === victim && a === victor;
      wipe[4 * i] = flooded ? FLOOD_STEPS * (Math.abs((i % side) - (fallen % side)) + Math.abs(Math.floor(i / side) - Math.floor(fallen / side))) : 0;
      wipe[4 * i + 1] = !newGame && a !== b ? 1 : 0;
    }
    board.now.needsUpdate = board.was.needsUpdate = board.wipe.needsUpdate = true;
    board.uniforms.sweep.value = newGame && known && live ? 1 : 0;
    this.arrived = live ? time : -Infinity;
    board.uniforms.muster.value = !newGame && now.words[GW.TICK]! % this.shape.round === 0 ? 1 : 0;
    this.shock = fallen >= 0 ? { index: fallen, tint: victor === seat ? TOKENS.celebrationGold : this.inkOf(victor, seat) } : ended && known && live ? this.finale(was, fresh.reward) : null;
    if (this.shock) {
      this.spot(this.shock.index, this.at);
      this.marks!.fallen.value.set(this.at.x, this.at.z);
      this.marks!.shock.set(this.shock.tint);
    }
    if (now.words[GW.GAME] !== this.relieved) this.relief(now);
    this.illuminate(now, seat);
    this.tally(now, seat);
    if (live) this.cheer(now, was, seat, fallen, victor, newGame, pool, time);
    this.tickAt = at;
  }

  private finale(last: EnvFrame, reward: number): Shock | null {
    if (reward === 0) return null;
    const seat = last.words[GW.SEAT]!;
    for (let i = 0; i < last.view.length; i++) {
      const word = last.view[i]!;
      if (terrainOf(word) === CAPITAL && (ownerOf(word) === seat) === reward < 0) return { index: i, tint: reward < 0 ? TOKENS.failureBlue : TOKENS.celebrationGold };
    }
    throw new Error(`Generals game ${last.words[GW.GAME]} ends with reward ${reward} though its last board holds no general to fall`);
  }

  private cheer(now: EnvFrame, was: EnvFrame, seat: number, fallen: number, victor: number, newGame: boolean, pool: CelebrationPool, time: number): void {
    if (fallen >= 0) {
      this.spot(fallen, this.at);
      pool.ripple(this.at, victor === seat ? TOKENS.celebrationGold : this.inkOf(victor, seat), 0.9, time);
      if (victor === seat) pool.burst(this.at.setY(38), TOKENS.celebrationGold, 1.1, time);
    }
    if (newGame) return;
    for (let i = 0; i < now.view.length; i++) {
      const a = now.view[i]!, b = was.view[i]!;
      if (terrainOf(a) !== CITY || ownerOf(a) !== seat || ownerOf(b) === seat || i === fallen) continue;
      pool.burst(this.spot(i, this.at).setY(38), TOKENS.celebrationGold, 0.55, time);
    }
  }

  private finish(frame: FrameUpdate, won: boolean, pool: CelebrationPool, time: number): void {
    const step = frame.state.step;
    for (let s = step; s > step - 2 * this.shape.substeps && s > 0; s--) {
      const end = frame.frameAt(s), last = frame.frameAt(s - 1);
      if (end.step !== s || last.step !== s - 1 || !end.done) continue;
      const shock = this.finale(last, end.reward);
      if (!shock) return;
      this.spot(shock.index, this.at);
      if (won) {
        pool.burst(this.at.setY(38), TOKENS.celebrationGold, 1.4, time);
        pool.pulse.fire(1);
      } else pool.sigh(1, this.at.setY(2), time);
      return;
    }
  }

  private paint(palette: THREE.Color[], seat: number): void {
    for (let p = 0; p < NOBODY; p++) palette[p]!.set(p < this.shape.players ? this.inkOf(p, seat) : 0x000000);
  }

  private relief(now: EnvFrame): void {
    const positions: number[] = [], index: number[] = [], r = 0.36;
    for (let i = 0; i < now.view.length; i++) {
      if (terrainOf(now.view[i]!) !== PEAK) continue;
      const base = positions.length / 3, { x, z } = this.spot(i, this.at);
      positions.push(x - r, 0.01, z - r, x + r, 0.01, z - r, x + r, 0.01, z + r, x - r, 0.01, z + r);
      index.push(base, base + 2, base + 1, base, base + 3, base + 2);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(index);
    if (this.shade) {
      this.shade.geometry.dispose();
      this.shade.geometry = geometry;
    } else {
      this.shade = this.light!.occluder(geometry);
      this.scene.add(this.shade);
    }
    this.relieved = now.words[GW.GAME]!;
  }

  private illuminate(now: EnvFrame, seat: number): void {
    const marks = this.marks!, warm = new THREE.Color(0xffd9a0);
    let used = 0, home = -1;
    for (let i = 0; i < now.view.length; i++) {
      const word = now.view[i]!, kind = terrainOf(word), owner = ownerOf(word);
      if (kind === CAPITAL && owner === seat) home = i;
      if ((kind !== CAPITAL && kind !== CITY) || owner === NOBODY) continue;
      const lamp = this.lamps[used++];
      if (!lamp) throw new Error(`Generals holds more owned structures than its ${this.lamps.length} generals and cities`);
      this.spot(i, lamp.mesh.position).setY(0.02);
      lamp.tint.set(this.inkOf(owner, seat)).lerp(warm, kind === CITY ? 0.55 : 0.2).multiplyScalar(kind === CAPITAL ? CAPITAL_LIGHT : CITY_LIGHT);
      lamp.mesh.visible = true;
    }
    for (let i = used; i < this.lamps.length; i++) this.lamps[i]!.mesh.visible = false;
    marks.homed.value = home >= 0 ? 1 : 0;
    if (home >= 0) {
      this.spot(home, this.at);
      marks.home.value.set(this.at.x, this.at.z);
    }
  }

  private tally(now: EnvFrame, seat: number): void {
    const players = this.shape.players, rows: Standing[] = Array.from({ length: players }, (_, player) => ({ player, land: 0, army: 0, alive: false }));
    let open = 0;
    for (const word of now.view) {
      const owner = ownerOf(word);
      if (terrainOf(word) !== PEAK) open++;
      if (owner === NOBODY) continue;
      const row = rows[owner]!;
      row.land++;
      row.army += armyOf(word);
      if (terrainOf(word) === CAPITAL) row.alive = true;
    }
    const data = this.panel!.rows.image.data as Float32Array;
    [...rows].sort((a, b) => Number(b.alive) - Number(a.alive) || b.army - a.army || a.player - b.player).forEach((row, rank) => {
      this.rowGoal[row.player] = rowTop(rank);
      data.set([row.army, row.land, row.player + (row.alive ? 8 : 0) + (row.player === seat ? 16 : 0)], row.player * 4);
    });
    const alive = rows.filter((row) => row.alive), mine = rows[seat]!;
    this.ledger.land = mine.land;
    this.ledger.army = mine.army;
    this.ledger.alive = alive.length;
    this.ledger.rank = 1 + alive.filter((row) => row.army > mine.army).length;
    this.ledger.share = open > 0 ? mine.land / open : 0;
  }

  private steer(frame: FrameUpdate, step: number, now: EnvFrame, age: number): void {
    const marks = this.marks!, u = clamp01(frame.position - step), here = now.words[GW.CURSOR]!;
    const next = frame.frameAt(step + 1), ahead = next.step === step + 1 && !next.done ? next.words[GW.CURSOR]! : here;
    this.centre(here, this.at).lerp(this.centre(ahead, this.other), this.span(here, ahead) <= 1 ? u : 0);
    marks.cursor.value.set(this.at.x, this.at.z);
    this.cursorLamp?.mesh.position.set(this.at.x, 0.02, this.at.z);
    marks.mode.value = now.words[GW.MODE]!;
    const before = frame.frameAt(step - 1), from = before.words[GW.CURSOR]!;
    const jumped = step > 0 && before.step === step - 1 && !now.done && keyOf(now) === JUMP_KEY && this.span(from, here) > 1;
    marks.jumped.value = jumped ? u * 4 : -1;
    if (jumped) marks.jump.value.set(this.centre(from, this.at).x, this.at.z, this.centre(here, this.other).x, this.other.z);
    this.show(marks.order, marks.ordered, orderOf(now.words));
    const at = step - now.words[GW.PHASE]!, last = frame.frameAt(at - 1);
    const resolved = age < MARCH_STEPS && last.step === at - 1 && !frame.frameAt(at).done ? orderOf(last.words) : null;
    marks.marched.value = resolved ? age : -1;
    if (!resolved) return;
    marks.held.value = heeded(last.view, this.shape.side, last.words[GW.SEAT]!, resolved) ? 0 : 1;
    marks.march.value.set(this.centre(resolved.cell, this.at).x, this.at.z, this.at.x + STEP_X[resolved.dir]!, this.at.z + STEP_Y[resolved.dir]!);
  }

  private span(a: number, b: number): number {
    return Math.abs(cellX(a) - cellX(b)) + Math.abs(cellY(a) - cellY(b));
  }

  private show(target: THREE.UniformNode<'vec4', THREE.Vector4>, on: THREE.UniformNode<'float', number>, order: Order | null): void {
    on.value = order ? 1 : 0;
    if (order) target.value.set(this.centre(order.cell, this.at).x, this.at.z, order.dir, order.split ? 1 : 0);
  }
}

export default (() => new GeneralsScene()) satisfies SceneFactory;
