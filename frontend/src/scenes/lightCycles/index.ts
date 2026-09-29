import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { hash2f } from '@kit/rng';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import {
  LC_AGENT, LC_AGENT_X, LC_AGENT_Y, LC_DRAWN, LC_EMPTY, LC_LOST, LC_RIVAL, LC_WON,
  lcAction, lcFacing, lcGame, lcHead, lcOwner, lcResult, lcTarget, lcTick,
} from '@sim/envs/lightcycles';
import type { EnvFrame, GridCell } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { arenaMaterial, CYCLE, cycleMaterial, flat, NOSE, PAD, RIM, SHARD, shardMaterial, Strips, WALL, wallMaterial, type Level, type Palette } from './look';

const RIVAL = 0xff7a2f;
const GOLD = TOKENS.celebrationGold;
const BOTS = ['rambler', 'hoarder', 'tactician'] as const;
const TURNS = ['← left', '↑ straight', 'right →'] as const;
const OWNERS = [LC_AGENT, LC_RIVAL] as const;
const GHOST_LIFE = 0.45;
const SHARD_LIFE = 0.9;
const SPAWN_LIFE = 0.35;
const VANISH = 0.25;
const SHARDS = 18;
const RIDERS = 4;
const RETRY = 0.25;
const GLOW_HALF = 0.07;
const TRAIL_LIGHT = 1.35;
const CYCLE_LIGHT = 5;
const BORDER_LIGHT = 0.55;

const angleOf = (dx: number, dy: number) => Math.atan2(-dy, dx);
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => formatSigned(value).replace('-', '−');
const compass = (x: number, y: number) => (Math.abs(x) > Math.abs(y) ? (x > 0 ? 'east' : 'west') : y > 0 ? 'south' : 'north');

interface Chain { game: number; cells: [Int16Array, Int16Array]; low: number; high: number; opened: boolean; start: number; retry: number }
interface Mover { fx: number; fy: number; tx: number; ty: number; dx: number; dy: number; crashed: boolean; stop: number }
interface Crash { step: number; result: number; movers: [Mover, Mover] }
interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute; used: number }
interface Glow { mesh: THREE.Mesh; level: Level }

class LightCyclesScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 12;
  protected readonly stageName = 'lightCycles';

  private grid: GridSize = { w: 16, h: 16 };
  private palette: Palette = { agent: 0x5ef0ff, rival: RIVAL };
  private bot: string = BOTS[0];
  private reach = 15;
  private readonly time = uniform(0);
  private readonly ghostLevel = uniform(0);
  private claims: THREE.DataTexture | null = null;
  private claimKey = '';
  private walls: Batch | null = null;
  private spans: THREE.InstancedBufferAttribute | null = null;
  private riders: Batch | null = null;
  private shards: Batch | null = null;
  private trails: [Strips, Strips] | null = null;
  private trailGlows: Glow[] = [];
  private cycleGlows: Glow[] = [];
  private readonly disposables: { dispose(): void }[] = [];

  private readonly chains = new Map<number, Chain>();
  private readonly crashes = new Map<number, Crash>();
  private celebrated = -1;
  private clock = 0;
  private log: FrameUpdate['eventLog'] | null = null;

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly spot = new THREE.Vector3();
  private readonly tip = new THREE.Vector3();
  private readonly cell = new THREE.Vector3();
  private readonly ledger = { x: 0, y: 0, facing: 'east', action: '—', clear: '—', reward: 0, result: 0, game: 0, tick: 0, step: 0 };

  private legendFor(): LegendSpec {
    return {
      title: 'LIGHT CYCLES',
      rows: [
        { key: 'agent', label: 'agent', tint: this.palette.agent, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1} · ${this.ledger.facing}` },
        { key: 'rival', label: 'rival', tint: RIVAL, value: () => this.bot },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
        { key: 'clear', label: 'clear', tint: TOKENS.uiCyan, value: () => this.ledger.clear },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(this.ledger.reward) },
        { key: 'last', label: 'last game', tint: GOLD, value: () => ['—', 'won', 'lost', 'drawn'][this.ledger.result]! },
      ],
      note: () => this.record(),
      bar: () => (2 * (this.ledger.tick + 1)) / (this.grid.w * this.grid.h),
    };
  }

  private record(): string {
    const log = this.log, step = this.ledger.step;
    if (!log) return '';
    const won = countEvents(log, 'goal', -1, step), lost = countEvents(log, 'harm', -1, step), drawn = countEvents(log, 'done', -1, step) - won - lost;
    return `game ${formatCount(this.ledger.game + 1)} · won ${formatCount(won)} · lost ${formatCount(lost)} · drawn ${formatCount(drawn)}`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, bot = BOTS[parameter(env, 'bot')];
    if (bot === undefined) throw new Error(`${env.envId} seats rival ${parameter(env, 'bot')}, which LightCycles does not name`);
    this.grid = { w: parameter(env, 'width'), h: parameter(env, 'height') };
    this.bot = bot;
    this.reach = Math.max(this.grid.w, this.grid.h) - 1;
    this.palette = { agent: this.accentOf(ctx), rival: RIVAL };
    this.legend = this.legendFor();
    const x = this.grid.w / 2 + RIM + 0.3, z = this.grid.h / 2 + RIM + 0.3;
    return new THREE.Box2(new THREE.Vector2(-x, -z), new THREE.Vector2(x, z));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x7d93c8).multiplyScalar(0.3);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighRadius: 2.4, sighHeight: 1.2 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[LC_AGENT_X]!;
        agent.y = state.pose[LC_AGENT_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const { w, h } = this.grid, step = frame.state.step, now = frame.frameAt(step), words = now.words, obs = now.observations;
    const cells = kit.marks('lit cells', w * h), heads = kit.marks('heads', 2), diagonals = kit.marks('diagonals', 2), targets = kit.marks('crash targets', 2);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const owner = lcOwner(words, x, y);
      if (owner === LC_EMPTY) continue;
      this.centre(x, y, this.at);
      cells.set(this.at.x, 0.05, this.at.z, 0.11, owner === LC_AGENT ? this.palette.agent : RIVAL);
    }
    const chain = this.chains.get(lcGame(words)), tick = lcTick(words);
    OWNERS.forEach((owner, k) => {
      const head = lcHead(words, owner), facing = lcFacing(obs, owner), left = { x: facing.y, y: -facing.x };
      this.centre(head.x, head.y, this.at);
      if (owner === LC_RIVAL) heads.set(this.at.x, 0.06, this.at.z, 0.2, RIVAL);
      const color = owner === LC_AGENT ? TOKENS.uiCyan : RIVAL, base = owner === LC_AGENT ? 0 : 13;
      const rays = kit.curve(`${owner === LC_AGENT ? 'agent' : 'rival'} rays`, 6, color);
      for (const [dx, dy, value] of [[facing.x, facing.y, obs[base]!], [left.x, left.y, obs[base + 1]!], [-left.x, -left.y, obs[base + 2]!]] as const) {
        const length = Math.round(value * this.reach) + 0.5;
        rays.point(this.at.x, 0.07, this.at.z);
        rays.point(this.at.x + dx * length, 0.07, this.at.z + dy * length);
      }
      if (owner === LC_AGENT) for (const [side, value] of [[1, obs[3]!], [-1, obs[4]!]] as const) {
        diagonals.set(this.at.x + facing.x + side * left.x, 0.06, this.at.z + facing.y + side * left.y, 0.09, value > 0.5 ? 0xff5a5a : 0x7dffa8);
      }
      const trail = kit.curve(`${owner === LC_AGENT ? 'agent' : 'rival'} chain`, w * h, owner === LC_AGENT ? 0xffffff : 0xffd0a8);
      if (!chain) return;
      for (let i = 0; i <= tick; i++) {
        const cell = chain.cells[k]![i]!;
        if (cell < 0) continue;
        this.centre(cell % w, Math.floor(cell / w), this.other);
        trail.point(this.other.x, 0.08, this.other.z);
      }
    });
    if (now.done) for (const owner of OWNERS) {
      const target = lcTarget(words, owner);
      this.centre(target.x, target.y, this.at);
      targets.set(this.at.x, 0.07, this.at.z, 0.16, owner === LC_AGENT ? TOKENS.uiCyan : RIVAL);
    }
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private inside(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < this.grid.w && y < this.grid.h;
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    const { w, h } = this.grid, palette = this.palette, cells = w * h;
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const claims = new THREE.DataTexture(new Uint8Array(cells * 4), w, h, THREE.RGBAFormat);
    claims.magFilter = claims.minFilter = THREE.NearestFilter;
    claims.needsUpdate = true;
    this.claims = claims;
    const floor = new THREE.Mesh(flat(120, 120), arenaMaterial(w, h, claims, this.ghostLevel, palette, light.lit));
    floor.renderOrder = 0;
    floor.frustumCulled = false;
    this.scene.add(floor);
    this.disposables.push(claims, floor.geometry, floor.material as THREE.Material);

    const make = (geometry: THREE.BufferGeometry, name: string, material: THREE.Material, count: number, order: number): Batch => {
      const state = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, state);
      const mesh = instancedMesh(geometry, material, count);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      mesh.count = 0;
      this.scene.add(mesh);
      this.disposables.push(geometry, material, mesh);
      return { mesh, state, used: 0 };
    };
    const runs = 4 * cells, wallGeometry = flat(1, 1);
    this.spans = new THREE.InstancedBufferAttribute(new Float32Array(runs * 4), 4).setUsage(THREE.DynamicDrawUsage);
    wallGeometry.setAttribute('aSpan', this.spans);
    this.walls = make(wallGeometry, 'aWall', wallMaterial(palette, this.time), runs, 2);
    this.shards = make(flat(SHARD, SHARD), 'aShard', shardMaterial(palette), 2 * SHARDS, 5);
    this.riders = make(flat(CYCLE, CYCLE), 'aRider', cycleMaterial(palette, light.lit), RIDERS, 4);

    const glow = (geometry: THREE.BufferGeometry, color: number): Glow => {
      const level = uniform(0), mesh = light.emitter(geometry, rgb(color).mul(level));
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.disposables.push(geometry, mesh.material as THREE.Material);
      return { mesh, level };
    };
    this.trails = [new Strips(cells), new Strips(cells)];
    this.trailGlows = this.trails.map((strips, i) => glow(strips.geometry, i ? RIVAL : palette.agent));
    for (const trail of this.trailGlows) trail.level.value = TRAIL_LIGHT;
    this.cycleGlows = OWNERS.map((owner) => glow(new THREE.CircleGeometry(0.17, 24).rotateX(-Math.PI / 2), owner === LC_AGENT ? palette.agent : RIVAL));

    const ring = (inset: number, half: number) => {
      const strips = new Strips(4), x = w / 2 + inset, z = h / 2 + inset;
      strips.add(-x, -z, x, -z, half);
      strips.add(-x, z, x, z, half);
      strips.add(-x, -z, -x, z, half);
      strips.add(x, -z, x, z, half);
      strips.commit();
      return strips.geometry;
    };
    glow(ring(0.03, 0.03), 0xcdf3ff).level.value = BORDER_LIGHT;
    const wall = light.occluder(ring(RIM / 2 + 0.04, RIM / 2 - 0.02));
    wall.frustumCulled = false;
    this.scene.add(wall);
    this.disposables.push(wall.geometry, wall.material as THREE.Material);
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.chains.clear();
    this.crashes.clear();
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.clock += frame.dtReal;
    this.time.value = time;
    this.log = frame.eventLog;
    const hz = frame.controlHz, t = frame.state.step, now = frame.frameAt(t);
    const peek = frame.frameAt(t + 1), ahead = peek.step === t + 1 ? peek : null;
    const a = ahead ? clamp01(frame.position - t) : 0;
    if (frame.events.some((event) => event.type === 'done')) this.celebrate(frame, pool, time);

    const walls = this.walls!, riders = this.riders!, shards = this.shards!, trails = this.trails!;
    walls.used = riders.used = shards.used = 0;
    for (const strips of trails) strips.begin();

    const chain = this.chainOf(frame, t, now.words), tick = lcTick(now.words);
    const crash = ahead?.done ? this.crashAt(frame, t + 1) : null;
    const spawn = chain.start >= 0 ? clamp01((time - chain.start / hz) / SPAWN_LIFE) : 1;
    const recent = crash ?? this.recentCrash(frame, t, hz);
    let ghost = 0;
    if (recent) {
      const age = time - recent.step / hz;
      if (recent.step <= t && age < GHOST_LIFE) {
        const before = frame.frameAt(recent.step - 1), old = this.chainOf(frame, recent.step - 1, before.words), fade = Math.max(1e-3, age / GHOST_LIFE);
        ghost = (1 - fade) ** 2;
        OWNERS.forEach((owner, k) => {
          const mover = recent.movers[k]!, travel = mover.crashed ? mover.stop : 1;
          this.centre(mover.fx, mover.fy, this.tip).add(this.scale.set(mover.dx * travel, 0, mover.dy * travel));
          this.lay(old, owner, lcTick(before.words), before.words, this.tip, fade, null);
          if (!mover.crashed && age < VANISH) this.ride(this.tip, angleOf(mover.dx, mover.dy), owner, 1 - age / VANISH, 1, age / VANISH);
        });
        this.paintClaims(now.words, before.words, `${t}:${recent.step}`);
      }
      OWNERS.forEach((owner, k) => {
        const mover = recent.movers[k]!;
        if (mover.crashed) this.shatter(recent, mover, owner, time - (recent.step - 1 + mover.stop) / hz);
      });
    }
    OWNERS.forEach((owner, k) => {
      const head = lcHead(now.words, owner), facing = lcFacing(now.observations, owner);
      const rest = angleOf(facing.x, facing.y);
      this.centre(head.x, head.y, this.spot);
      let angle = rest, alpha = 1, tip: THREE.Vector3 | null = null;
      if (crash) {
        const mover = crash.movers[k]!, u = mover.crashed ? Math.min(a, mover.stop) : a;
        tip = this.tip.set(this.spot.x + mover.dx * u, 0, this.spot.z + mover.dy * u);
        angle = turn(rest, angleOf(mover.dx, mover.dy), smoothstep01(a / 0.3));
        if (mover.crashed && a >= mover.stop) alpha = 0;
      } else if (ahead) {
        const next = lcHead(ahead.words, owner), dx = next.x - head.x, dy = next.y - head.y;
        if (Math.abs(dx) + Math.abs(dy) !== 1) throw new Error(`LightCycles step ${t + 1} moves a cycle from (${head.x}, ${head.y}) to (${next.x}, ${next.y})`);
        tip = this.tip.set(this.spot.x + dx * a, 0, this.spot.z + dy * a);
        angle = turn(rest, angleOf(dx, dy), smoothstep01(a / 0.3));
      }
      this.lay(chain, owner, tick, now.words, tip, 0, trails[k]!);
      const shown = tip ?? this.spot;
      this.ride(shown, angle, owner, alpha, spawn, spawn < 1 ? (1 - spawn) ** 2 : 0);
      const lamp = this.cycleGlows[k]!;
      lamp.mesh.position.set(shown.x, 0, shown.z);
      lamp.mesh.visible = alpha > 0;
      lamp.level.value = CYCLE_LIGHT * (0.35 + 0.65 * spawn);
    });

    if (ghost === 0) this.paintClaims(now.words, null, `${t}`);
    this.ghostLevel.value = ghost;
    this.trailGlows.forEach((glow, i) => { glow.mesh.visible = trails[i]!.count > 0; });

    for (const batch of [walls, riders, shards]) {
      batch.mesh.count = batch.used;
      batch.mesh.visible = batch.used > 0;
      batch.mesh.instanceMatrix.needsUpdate = true;
      batch.state.needsUpdate = true;
    }
    this.spans!.needsUpdate = true;
    for (const strips of trails) strips.commit();
    this.account(frame, now, t);
  }

  private chainOf(frame: FrameUpdate, t: number, words: ArrayLike<number>): Chain {
    const game = lcGame(words), cells = this.grid.w * this.grid.h;
    let chain = this.chains.get(game);
    if (!chain) {
      chain = { game, cells: [new Int16Array(cells).fill(-1), new Int16Array(cells).fill(-1)], low: -1, high: -1, opened: false, start: -1, retry: -Infinity };
      this.chains.set(game, chain);
      const stale = [...this.chains.keys()].sort((x, y) => Math.abs(y - game) - Math.abs(x - game));
      while (this.chains.size > 4) this.chains.delete(stale.shift()!);
    }
    if (chain.high >= 0 && t > chain.high) {
      let s = chain.high + 1;
      for (; s <= t; s++) {
        const f = frame.frameAt(s);
        if (f.step !== s || lcGame(f.words) !== game) break;
        this.note(chain, f);
      }
      if (s > t) chain.high = t;
      else this.restart(frame, chain, t);
    } else if (chain.high < 0 || t < chain.low) this.restart(frame, chain, t);
    if (!chain.opened && this.clock - chain.retry >= RETRY) {
      let s = chain.low - 1;
      for (; s >= 0; s--) {
        const f = frame.frameAt(s);
        if (f.step !== s) break;
        if (lcGame(f.words) !== game) {
          chain.opened = true;
          chain.start = chain.low;
          break;
        }
        this.note(chain, f);
        chain.low = s;
      }
      if (s < 0) {
        chain.opened = true;
        chain.start = 0;
      }
      if (!chain.opened) chain.retry = this.clock;
    }
    return chain;
  }

  private restart(frame: FrameUpdate, chain: Chain, t: number): void {
    this.note(chain, frame.frameAt(t));
    chain.low = chain.high = t;
    chain.opened = false;
    chain.retry = -Infinity;
  }

  private note(chain: Chain, f: EnvFrame): void {
    const tick = lcTick(f.words);
    OWNERS.forEach((owner, k) => {
      const head = lcHead(f.words, owner), cell = head.y * this.grid.w + head.x, cells = chain.cells[k]!, known = cells[tick]!;
      if (known >= 0 && known !== cell) throw new Error(`LightCycles step ${f.step} puts cycle ${owner} of game ${chain.game} at two cells on tick ${tick}`);
      cells[tick] = cell;
    });
  }

  private lay(chain: Chain, owner: number, tick: number, words: ArrayLike<number>, tip: THREE.Vector3 | null, fade: number, strips: Strips | null): void {
    const { w, h } = this.grid, cells = chain.cells[owner === LC_AGENT ? 0 : 1]!, pieces: number[][] = [];
    let points: number[] = [], complete = true, px = -9, py = -9;
    for (let i = 0; i <= tick; i++) {
      const cell = cells[i]!;
      if (cell < 0) {
        complete = false;
        if (points.length) pieces.push(points);
        points = [];
        continue;
      }
      const x = cell % w, y = Math.floor(cell / w);
      if (lcOwner(words, x, y) !== owner) throw new Error(`LightCycles game ${chain.game} lights (${x}, ${y}) on tick ${i} for cycle ${owner}, which the arena does not show`);
      if (points.length && Math.abs(x - px) + Math.abs(y - py) !== 1) throw new Error(`LightCycles game ${chain.game} jumps cycle ${owner} from (${px}, ${py}) to (${x}, ${y}) on tick ${i}`);
      this.centre(x, y, this.cell);
      points.push(this.cell.x, this.cell.z);
      px = x;
      py = y;
    }
    if (points.length && complete && tip) points.push(tip.x, tip.z);
    if (points.length) pieces.push(points);
    for (const piece of pieces) this.runs(piece, owner, fade, strips);
    if (complete) return;
    const known = new Set<number>();
    for (let i = 0; i <= tick; i++) if (cells[i]! >= 0) known.add(cells[i]!);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (known.has(y * w + x) || lcOwner(words, x, y) !== owner) continue;
      this.centre(x, y, this.cell);
      this.runs([this.cell.x, this.cell.z], owner, fade, strips);
    }
  }

  private runs(points: number[], owner: number, fade: number, strips: Strips | null): void {
    const n = points.length / 2, x = (i: number) => points[2 * i]!, z = (i: number) => points[2 * i + 1]!;
    let total = 0;
    for (let i = 1; i < n; i++) total += Math.abs(x(i) - x(i - 1)) + Math.abs(z(i) - z(i - 1));
    let from = 0, last = 0, dx = 0, dz = 0, along = 0;
    const emit = () => {
      this.wall(x(from), z(from), x(last), z(last), along, total, owner, fade);
      strips?.add(x(from), z(from), x(last), z(last), GLOW_HALF);
      along += Math.abs(x(last) - x(from)) + Math.abs(z(last) - z(from));
      from = last;
    };
    for (let i = 1; i < n; i++) {
      const sx = Math.sign(x(i) - x(last)), sz = Math.sign(z(i) - z(last));
      if (sx === 0 && sz === 0) continue;
      if ((dx !== 0 || dz !== 0) && (sx !== dx || sz !== dz)) emit();
      dx = sx;
      dz = sz;
      last = i;
    }
    emit();
  }

  private wall(ax: number, az: number, bx: number, bz: number, along: number, total: number, owner: number, fade: number): void {
    const walls = this.walls!, i = walls.used++;
    if (i >= walls.mesh.instanceMatrix.count) throw new Error(`LightCycles holds ${walls.mesh.instanceMatrix.count} wall runs`);
    this.matrix.makeScale(Math.abs(bx - ax) + 2 * PAD, 1, Math.abs(bz - az) + 2 * PAD).setPosition((ax + bx) / 2, 0, (az + bz) / 2);
    walls.mesh.setMatrixAt(i, this.matrix);
    this.spans!.setXYZW(i, ax, az, bx, bz);
    walls.state.setXYZW(i, along, total, owner === LC_AGENT ? 0 : 1, fade);
  }

  private ride(at: THREE.Vector3, angle: number, owner: number, alpha: number, spawn: number, heat: number): void {
    const riders = this.riders!, i = riders.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    riders.mesh.setMatrixAt(i, this.matrix.compose(this.other.set(at.x, 0, at.z), this.quaternion, this.scale.setScalar(0.82 + 0.18 * smoothstep01(spawn * 1.6))));
    riders.state.setXYZW(i, owner === LC_AGENT ? 0 : 1, alpha, spawn, heat);
  }

  private shatter(crash: Crash, mover: Mover, owner: number, age: number): void {
    if (!(age >= 0 && age < SHARD_LIFE)) return;
    const shards = this.shards!, seed = crash.step * 2 + owner, heading = Math.atan2(mover.dy, mover.dx);
    const contact = mover.stop + NOSE;
    this.centre(mover.fx, mover.fy, this.at).add(this.scale.set(mover.dx * contact, 0, mover.dy * contact));
    const drag = (1 - Math.exp(-age * 4.5)) / 4.5, life = age / SHARD_LIFE;
    for (let n = 0; n < SHARDS; n++) {
      const spread = hash2f(n, 1, seed), speed = 1.1 + 2.6 * hash2f(n, 2, seed), spin = (hash2f(n, 3, seed) - 0.5) * 14;
      const angle = heading + Math.PI + (spread - 0.5) * 3.4, back = 0.12 * hash2f(n, 4, seed);
      const x = this.at.x - mover.dx * back + Math.cos(angle) * speed * drag, z = this.at.z - mover.dy * back + Math.sin(angle) * speed * drag;
      this.quaternion.setFromAxisAngle(this.axis, spin * age + 6 * hash2f(n, 5, seed));
      const i = shards.used++;
      shards.mesh.setMatrixAt(i, this.matrix.compose(this.other.set(x, 0, z), this.quaternion, this.scale.setScalar(1 - 0.5 * life)));
      shards.state.setXYZW(i, owner === LC_AGENT ? 0 : 1, 1 - life * life, Math.exp(-age * 7), hash2f(n, 6, seed) > 0.55 ? 1 : 0);
    }
  }

  private recentCrash(frame: FrameUpdate, t: number, hz: number): Crash | null {
    const back = Math.ceil(Math.max(GHOST_LIFE, SHARD_LIFE) * hz) + 1;
    for (let s = t; s >= Math.max(1, t - back); s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) return null;
      if (f.done) return this.crashAt(frame, s);
    }
    return null;
  }

  private crashAt(frame: FrameUpdate, d: number): Crash | null {
    const cached = this.crashes.get(d);
    if (cached) return cached;
    const end = frame.frameAt(d), before = frame.frameAt(d - 1);
    if (d < 1 || end.step !== d || before.step !== d - 1 || !end.done) return null;
    const result = lcResult(end.words);
    if (result !== LC_WON && result !== LC_LOST && result !== LC_DRAWN) throw new Error(`LightCycles step ${d} ends a game without recording who crashed`);
    const movers = OWNERS.map((owner): Mover => {
      const from = lcHead(before.words, owner), to = lcTarget(end.words, owner), dx = to.x - from.x, dy = to.y - from.y;
      if (Math.abs(dx) + Math.abs(dy) !== 1) throw new Error(`LightCycles step ${d} records cycle ${owner} leaving (${from.x}, ${from.y}) for (${to.x}, ${to.y})`);
      const crashed = owner === LC_AGENT ? result !== LC_WON : result !== LC_LOST;
      return { fx: from.x, fy: from.y, tx: to.x, ty: to.y, dx, dy, crashed, stop: 1 };
    }) as [Mover, Mover];
    const [agent, rival] = movers, meet = agent.tx === rival.tx && agent.ty === rival.ty;
    for (const mover of movers) {
      const other = mover === agent ? rival : agent, off = !this.inside(mover.tx, mover.ty);
      const blocked = off || lcOwner(before.words, mover.tx, mover.ty) !== LC_EMPTY;
      if (mover.crashed !== (blocked || meet)) throw new Error(`LightCycles step ${d} ${mover.crashed ? 'crashes' : 'spares'} a cycle entering (${mover.tx}, ${mover.ty})`);
      const swap = mover.tx === other.fx && mover.ty === other.fy && other.tx === mover.fx && other.ty === mover.fy;
      if (mover.crashed) mover.stop = off || swap ? 0.5 - NOSE : blocked ? 1 - WALL - NOSE : 1 - NOSE;
    }
    const crash: Crash = { step: d, result, movers };
    if (this.crashes.size > 32) this.crashes.clear();
    this.crashes.set(d, crash);
    return crash;
  }

  private paintClaims(words: ArrayLike<number>, ghost: ArrayLike<number> | null, key: string): void {
    const claims = this.claims!;
    if (key === this.claimKey) return;
    this.claimKey = key;
    const { w, h } = this.grid, data = claims.image.data as Uint8Array;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4, owner = lcOwner(words, x, y), old = ghost ? lcOwner(ghost, x, y) : LC_EMPTY;
      data[i] = owner === LC_AGENT ? 255 : 0;
      data[i + 1] = owner === LC_RIVAL ? 255 : 0;
      data[i + 2] = old === LC_AGENT ? 255 : 0;
      data[i + 3] = old === LC_RIVAL ? 255 : 0;
    }
    claims.needsUpdate = true;
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    const ended = lastEvent(frame.eventLog, 'done', frame.state.step);
    if (ended === undefined || ended === this.celebrated) return;
    this.celebrated = ended;
    const crash = this.crashAt(frame, ended);
    if (!crash) return;
    const [agent, rival] = crash.movers, spot = (mover: Mover) => {
      const contact = mover.stop + (mover.crashed ? NOSE : 0);
      return this.centre(mover.fx, mover.fy, new THREE.Vector3()).add(new THREE.Vector3(mover.dx * contact, 0, mover.dy * contact));
    };
    if (crash.result === LC_WON) {
      const at = spot(rival);
      pool.burst(at.clone().setY(30), GOLD, 1.3, time);
      pool.ripple(at, GOLD, 0.42, time);
    } else if (crash.result === LC_LOST) {
      const at = spot(agent);
      pool.sigh(0.55, at, time);
      pool.ripple(at, TOKENS.failureBlue, 0.3, time);
    } else {
      const at = spot(agent).add(spot(rival)).multiplyScalar(0.5);
      pool.burst(at.clone().setY(30), 0xdcf4ff, 1, time);
      pool.ripple(at, 0xdcf4ff, 0.3, time);
    }
  }

  private account(frame: FrameUpdate, now: EnvFrame, t: number): void {
    const ledger = this.ledger, words = now.words, obs = now.observations, agent = lcHead(words, LC_AGENT), facing = lcFacing(obs, LC_AGENT);
    ledger.step = t;
    ledger.x = agent.x;
    ledger.y = agent.y;
    ledger.facing = compass(facing.x, facing.y);
    ledger.reward = now.reward;
    ledger.result = lcResult(words);
    ledger.game = lcGame(words);
    ledger.tick = lcTick(words);
    ledger.clear = `ahead ${Math.round(obs[0]! * this.reach)} · left ${Math.round(obs[1]! * this.reach)} · right ${Math.round(obs[2]! * this.reach)}`;
    const before = t > 0 ? frame.frameAt(t - 1) : null;
    ledger.action = !before || before.step !== t - 1 ? '—' : TURNS[lcAction(now.actions)]!;
  }
}

export default (() => new LightCyclesScene()) satisfies SceneFactory;
