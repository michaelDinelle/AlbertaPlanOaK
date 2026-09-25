import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import {
  ACHIEVEMENTS, ACTION, BLOCK, CX_X, CX_Y, CXW, FACING, ITEMS, MOB_SLOTS, PRESS, alive, arrowDirection, blockAt, cellOf, craftaxShape, cursorOf, facingOf, healthOf, itemCount,
  loadingOf, perished, pressOf, readoutOf, restOf, sleepingOf, stageOf, vital, type CraftaxShape,
} from '@sim/envs/craftax';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { arrowShade, cowShade, Figures, playerShade, skeletonShade, zombieShade } from './figures';
import { ACHIEVEMENT_GLYPHS, ACTION_GLYPHS, ACTION_NAMES, GLYPH, Icons, ITEM_GLYPHS, MODE, TIER, VITAL_GLYPHS } from './glyphs';
import { groundMaterial, lightProxy, makeNoise, minimap, WorldMap, worldUniforms, type Minimap, type Noise, type WorldUniforms } from './world';

const ACCENT = TOKENS.accents.Craftax!;
const FACING_ANGLE = [Math.PI / 2, -Math.PI / 2, 0, Math.PI] as const;
const ARROWS = ['←', '→', '↑', '↓'] as const;
const KIND = { ZOMBIE: 0, COW: 1, SKELETON: 2, ARROW: 3 } as const;
const BLOCK_NAMES: Record<number, string> = {
  [BLOCK.OUT]: 'the world’s edge', [BLOCK.GRASS]: 'grass', [BLOCK.WATER]: 'water', [BLOCK.STONE]: 'stone', [BLOCK.TREE]: 'a tree', [BLOCK.WOOD]: 'wood', [BLOCK.PATH]: 'path',
  [BLOCK.COAL]: 'coal', [BLOCK.IRON]: 'iron', [BLOCK.DIAMOND]: 'a diamond', [BLOCK.TABLE]: 'a table', [BLOCK.FURNACE]: 'a furnace', [BLOCK.SAND]: 'sand', [BLOCK.LAVA]: 'lava',
  [BLOCK.PLANT]: 'a sapling', [BLOCK.RIPE]: 'a ripe plant',
};
const MOB_NAMES = ['a zombie', 'a cow', 'a skeleton', 'an arrow'] as const;
const DEBRIS: Record<number, number> = { [BLOCK.TREE]: 0x3f8a3a, [BLOCK.STONE]: 0x9a9ca3, [BLOCK.COAL]: 0x3a3940, [BLOCK.IRON]: 0xd49468, [BLOCK.DIAMOND]: 0xdffbff, [BLOCK.TABLE]: 0xb07a45, [BLOCK.FURNACE]: 0xff9a3c, [BLOCK.PLANT]: 0x8fd46a, [BLOCK.RIPE]: 0xe0453a, [BLOCK.GRASS]: 0x7ab04a, [BLOCK.SAND]: 0xd8c08a, [BLOCK.PATH]: 0x8a7866, [BLOCK.WATER]: 0x8cc8e6, [BLOCK.LAVA]: 0xffa040 };

const HEIGHT = 50;
const CONTROL_HZ = 8;
const FLY = 0.75;
const MENU_ROW = 0.55;
const VITAL_ROW = 0.55;
const ITEM_ROW = 1.27;
const GUTTER = 0.4;
const MAP_SIZE = 3.6;

const ease = (t: number) => t * t * (3 - 2 * t);
const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
const signed = (value: number) => formatSigned(value).replace('-', '−');
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const kindOf = (slot: number) => (slot < CXW.COW - CXW.ZOMBIE ? KIND.ZOMBIE : slot < CXW.SKELETON - CXW.ZOMBIE ? KIND.COW : slot < CXW.ARROW - CXW.ZOMBIE ? KIND.SKELETON : KIND.ARROW);

interface Flyer { glyph: number; tier: number; from: THREE.Vector2; target: 'item' | 'badge'; index: number; age: number }

class CraftaxScene extends Stage2DScene {
  post: ScenePost = { bloom: 0.5, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = CONTROL_HZ;
  protected readonly stageName = 'craftax';

  private shape: CraftaxShape = { side: 64, readout: [], menu: [], moves: [], presses: 6 };
  private grid: GridSize = { w: 64, h: 64 };
  private half = new THREE.Vector2(4.5, 3.5);
  private health = { zombie: 5, cow: 3, skeleton: 3 };
  private map: WorldMap | null = null;
  private noise: Noise | null = null;
  private u: WorldUniforms | null = null;
  private sun = uniform(new THREE.Vector3());
  private readonly clockNode = uniform(0);
  private hero: Figures | null = null;
  private zombies: Figures | null = null;
  private cows: Figures | null = null;
  private skeletons: Figures | null = null;
  private arrows: Figures | null = null;
  private icons: Icons | null = null;
  private mini: Minimap | null = null;
  private pixels = 720;
  private step = -1;
  private clock = 0;
  private daylight = 1;
  private debugView = false;
  private hurt = 0;
  private veil = 0;
  private readonly mobHurt = new Float32Array(MOB_SLOTS);
  private readonly itemFlash = new Float32Array(ITEMS.length);
  private readonly badgeFlash = new Float32Array(ACHIEVEMENTS.length);
  private readonly menuFlash = new Float32Array(13);
  private readonly flyers: Flyer[] = [];
  private readonly origin = new THREE.Vector2();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { game: 0, time: 0, reward: 0, action: '—', achieved: 0, daylight: 1, note: '', step: 0, log: null as FrameUpdate['eventLog'] | null };

  override legend: LegendSpec = {
    title: 'CRAFTAX',
    rows: [
      { key: 'world', label: 'world', tint: ACCENT, value: () => `${formatCount(this.ledger.game + 1)} · tick ${formatCount(this.ledger.time)} · ${this.ledger.daylight >= 0.5 ? 'day' : 'night'}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => signed(this.ledger.reward) },
      { key: 'achieved', label: 'achievements', tint: TOKENS.celebrationGold, value: () => `${this.ledger.achieved} of ${ACHIEVEMENTS.length} this life` },
      { key: 'death', label: 'last death', tint: TOKENS.failureBlue, value: () => this.since() },
    ],
    note: () => this.ledger.note,
    bar: () => this.ledger.daylight,
  };

  private since(): string {
    const log = this.ledger.log;
    let found = log ? lastEvent(log, 'done', this.ledger.step) : undefined;
    while (log && found !== undefined && countEvents(log, 'harm', found - 1, found) === 0) found = found > 0 ? lastEvent(log, 'done', found - 1) : undefined;
    return found === undefined ? 'none yet' : found === this.ledger.step ? 'now' : `${formatCount(this.ledger.step - found)} steps ago`;
  }

  private centre(cell: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(cell & 63, cell >>> 6, this.grid, out);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.shape = craftaxShape(env);
    this.grid = { w: this.shape.side, h: this.shape.side };
    const rows = this.shape.readout.map(([dy]) => Math.abs(dy)), cols = this.shape.readout.map(([, dx]) => Math.abs(dx));
    this.half.set(Math.max(...cols) + 0.5, Math.max(...rows) + 0.5);
    this.health = { zombie: parameter(env, 'zombieHealth'), cow: parameter(env, 'cowHealth'), skeleton: parameter(env, 'skeletonHealth') };
    this.pixels = ctx.size.height * ctx.size.dpr;
    const { left, right, top, bottom } = this.console();
    return new THREE.Box2(new THREE.Vector2(left - 1, top - 1.3), new THREE.Vector2(right + 1, bottom + 1.3));
  }

  private console() {
    const h = this.half;
    return { left: -h.x - GUTTER - MAP_SIZE, right: h.x + GUTTER + 1.6, top: -h.y - MENU_ROW - 0.45, bottom: h.y + ITEM_ROW + 0.42 };
  }

  protected ambient(): THREE.Color {
    return new THREE.Color().setRGB(0.1, 0.13, 0.24);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    this.pixels = size.height * size.dpr;
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    this.debugView = on;
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[CX_X]!;
        agent.y = state.pose[CX_Y]!;
        return agent;
      },
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step), colors = [0x9bff6a, 0xf3efe6, 0xe9e4d0, 0xffb347];
        const marks = kit.marks('mobs', MOB_SLOTS);
        for (let k = 0; k < MOB_SLOTS; k++) {
          const word = now.words[CXW.ZOMBIE + k]!;
          if (!alive(word)) continue;
          this.centre(cellOf(word), this.at);
          marks.set(this.at.x, 0.06, this.at.z, kindOf(k) === KIND.ARROW ? 0.1 : 0.18, colors[kindOf(k)]!);
        }
        const heads = kit.curve('arrows', 3 * 5, 0xffb347);
        for (let k = CXW.ARROW - CXW.ZOMBIE; k < MOB_SLOTS; k++) {
          const word = now.words[CXW.ZOMBIE + k]!;
          if (!alive(word)) continue;
          this.centre(cellOf(word), this.at);
          const [dx, dy] = FACING[arrowDirection(word)]!;
          heads.point(this.at.x - dx * 0.3, 0.07, this.at.z - dy * 0.3);
          heads.point(this.at.x + dx * 0.4, 0.07, this.at.z + dy * 0.4);
          heads.point(this.at.x + dx * 0.25 - dy * 0.12, 0.07, this.at.z + dy * 0.25 + dx * 0.12);
          heads.point(this.at.x + dx * 0.4, 0.07, this.at.z + dy * 0.4);
          heads.point(this.at.x + dx * 0.25 + dy * 0.12, 0.07, this.at.z + dy * 0.25 - dx * 0.12);
        }
        const player = cellOf(now.words[CXW.PLAYER]!);
        this.centre(player, this.at);
        const view = kit.curve('view', 5, TOKENS.uiCyan), touch = kit.curve('touch', 5, TOKENS.uiCyan), h = this.half;
        for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) {
          view.point(this.at.x + sx * h.x, 0.07, this.at.z + sz * h.y);
          touch.point(this.at.x + sx * 1.5, 0.07, this.at.z + sz * 1.5);
        }
        const slot = readoutOf(now.words), target = kit.marks('readout', 1), facing = kit.marks('facing', 1);
        if (slot >= ITEMS.length && !loadingOf(now.words)) {
          const [dy, dx] = this.shape.readout[slot - ITEMS.length]!;
          target.set(this.at.x + dx, 0.06, this.at.z + dy, 0.12, 0x7fe8ff);
        }
        const [fx, fy] = FACING[facingOf(now.words)]!, x = (player & 63) + fx, y = (player >>> 6) + fy;
        if (x >= 0 && y >= 0 && x < this.grid.w && y < this.grid.h) facing.set(this.at.x + fx, 0.06, this.at.z + fy, 0.1, 0xffffff);
        const trail = kit.curve('trail', 97, ACCENT), game = now.words[CXW.GAME];
        const cells: number[] = [];
        for (let s = frame.state.step; s >= Math.max(0, frame.state.step - 96); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s || f.words[CXW.GAME] !== game) break;
          cells.push(cellOf(f.words[CXW.PLAYER]!));
        }
        for (let i = cells.length - 1; i >= 0; i--) {
          this.centre(cells[i]!, this.other);
          trail.point(this.other.x, 0.05, this.other.z);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const side = this.shape.side;
    this.scene.background = new THREE.Color(0x0b0f14);
    const map = new WorldMap(side), noise = makeNoise(ctx.seed), u = worldUniforms(this.half);
    this.map = map;
    this.noise = noise;
    this.u = u;
    u.clock = this.clockNode;
    const shade = (albedo: THREE.Node<'vec3'>) => light.lit(albedo).add(albedo.mul(this.sun));
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(side + 60, side + 60).rotateX(-Math.PI / 2), groundMaterial(side, map.texture, noise, u, shade));
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground, lightProxy(side, map.texture, u, noise, light));
    this.zombies = new Figures('craftaxZombies', 2 * 3, zombieShade(shade, this.health.zombie), side, u, 6, 1.4);
    this.cows = new Figures('craftaxCows', 2 * 3, cowShade(shade, this.health.cow), side, u, 5, 1.3);
    this.skeletons = new Figures('craftaxSkeletons', 2 * 2, skeletonShade(shade, this.health.skeleton), side, u, 6, 1.4);
    this.hero = new Figures('craftaxPlayer', 1, playerShade(this.accentOf(ctx), shade, this.clockNode), side, u, 8, 1.4);
    this.arrows = new Figures('craftaxArrows', 2 * 3, arrowShade(shade), side, u, 9, 1.3, 1);
    this.icons = new Icons(96, this.accentOf(ctx), this.clockNode, 20);
    this.mini = minimap(side, map.texture, MAP_SIZE, this.half, this.accentOf(ctx));
    this.mini.mesh.renderOrder = 21;
    this.scene.add(this.zombies.mesh, this.cows.mesh, this.skeletons.mesh, this.hero.mesh, this.arrows.mesh, this.icons.mesh, this.mini.mesh);
    pool.group.traverse((object) => { object.renderOrder = 15; });
  }

  protected override drop(): void {
    this.map?.dispose();
    this.noise?.mottle.dispose();
    this.noise?.grain.dispose();
    for (const figures of [this.hero, this.zombies, this.cows, this.skeletons, this.arrows]) figures?.dispose();
    this.icons?.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || object instanceof THREE.InstancedMesh) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal;
    this.clock += dt;
    this.clockNode.value = this.clock;
    pool.update(dt, time);
    const step = frame.state.step;
    if (frame.scrubbed || step < this.step || this.step < 0) this.settle();
    else for (let s = this.step + 1; s <= step; s++) this.advance(frame, s, pool, time);
    this.step = step;
    this.decay(dt);
    const now = frame.frameAt(step), ahead = frame.frameAt(step + 1), next = ahead.step === step + 1 ? ahead : now;
    const alpha = next === now ? 0 : clamp01(frame.position - step);
    this.map!.paint(now.view);
    this.record(frame, now, step);
    const hero = this.placeHero(now, next, alpha);
    this.follow(hero);
    this.brighten(frame, now, next, alpha);
    this.placeMobs(frame, now, next, alpha, hero);
    this.drawHud(now, next, alpha, hero);
    this.u!.veil.value = Math.max(this.veil * 0.85, loadingOf(now.words) ? 0.4 : 0);
  }

  private settle(): void {
    this.hurt = 0;
    this.veil = 0;
    this.mobHurt.fill(0);
    this.itemFlash.fill(0);
    this.badgeFlash.fill(0);
    this.menuFlash.fill(0);
    this.flyers.length = 0;
  }

  private decay(dt: number): void {
    this.hurt = Math.max(0, this.hurt - dt / 0.45);
    this.veil = Math.max(0, this.veil - dt / 0.7);
    for (const list of [this.mobHurt, this.itemFlash, this.badgeFlash, this.menuFlash]) for (let i = 0; i < list.length; i++) list[i] = Math.max(0, list[i]! - dt / 0.6);
    for (const flyer of this.flyers) flyer.age += dt / FLY;
    for (let i = this.flyers.length - 1; i >= 0; i--) if (this.flyers[i]!.age >= 1) this.flyers.splice(i, 1);
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const before = frame.frameAt(s - 1), now = frame.frameAt(s);
    if (before.step !== s - 1 || now.step !== s) return;
    const player = cellOf(now.words[CXW.PLAYER]!);
    this.centre(player, this.at);
    if (before.words[CXW.GAME] !== now.words[CXW.GAME]) {
      this.veil = 1;
      pool.ripple(this.at.setY(0), ACCENT, 0.4, time);
      this.flyers.length = 0;
      return;
    }
    if (now.done && perished(before, now)) {
      pool.ripple(this.at.setY(0), TOKENS.failureBlue, 0.6, time);
      pool.burst(this.other.copy(this.at).setY(30), 0xff5a44, 0.5, time);
    }
    if (loadingOf(before.words)) return;
    const gained = now.words[CXW.ACHIEVED]! & ~before.words[CXW.ACHIEVED]!;
    for (let bit = 0; bit < ACHIEVEMENTS.length; bit++) {
      if (!((gained >>> bit) & 1)) continue;
      this.badgeFlash[bit] = 1;
      const [glyph, tier] = ACHIEVEMENT_GLYPHS[bit]!;
      this.flyers.push({ glyph, tier, from: new THREE.Vector2(this.at.x, this.at.z - 0.7), target: 'badge', index: bit, age: 0 });
      pool.burst(this.other.copy(this.at).setY(30), TOKENS.celebrationGold, 0.8, time);
    }
    if (vital(now.words, 0) < vital(before.words, 0)) {
      this.hurt = 1;
      pool.burst(this.other.copy(this.at).setY(30), 0xff5a44, 0.22, time);
    }
    const facing = facingOf(before.words), origin = cellOf(before.words[CXW.PLAYER]!), [fx, fy] = FACING[facing]!;
    const tx = (origin & 63) + fx, ty = (origin >>> 6) + fy, target = tx >= 0 && ty >= 0 && tx < this.grid.w && ty < this.grid.h ? ty * this.grid.w + tx : -1;
    const collected = target >= 0 && pressOf(now, this.shape) === PRESS.CONFIRM && !sleepingOf(before.words) && this.shape.menu[cursorOf(before.words)] === ACTION.DO;
    for (let i = 0; i < ITEMS.length; i++) {
      const diff = itemCount(now.words, i) - itemCount(before.words, i);
      if (diff === 0) continue;
      this.itemFlash[i] = 1;
      if (diff > 0) {
        const source = collected ? target : origin;
        this.centre(source, this.other);
        const [glyph, tier] = ITEM_GLYPHS[i]!;
        this.flyers.push({ glyph, tier, from: new THREE.Vector2(this.other.x, this.other.z), target: 'item', index: i, age: 0 });
      }
    }
    for (let w = 0; w < now.view.length; w++) {
      if (now.view[w] === before.view[w]) continue;
      for (let j = 0; j < 8; j++) {
        const cell = w * 8 + j, was = blockAt(before.view, cell), is = blockAt(now.view, cell);
        if (was === is) continue;
        this.centre(cell, this.other).setY(30);
        const tint = DEBRIS[was === BLOCK.GRASS || was === BLOCK.SAND || was === BLOCK.PATH ? is : was] ?? 0xd8d8d8;
        pool.burst(this.other, tint, was === BLOCK.PLANT && is === BLOCK.RIPE ? 0.5 : 0.4, time);
      }
    }
    if (vital(now.words, 2) > vital(before.words, 2) && target >= 0 && blockAt(before.view, target) === BLOCK.WATER) {
      this.centre(target, this.other);
      pool.ripple(this.other.setY(0), 0x8cc8e6, 0.25, time);
    }
    for (let k = 0; k < MOB_SLOTS; k++) {
      const was = before.words[CXW.ZOMBIE + k]!, is = now.words[CXW.ZOMBIE + k]!;
      if (!alive(was) || kindOf(k) === KIND.ARROW) continue;
      if (healthOf(is) < healthOf(was)) this.mobHurt[k] = 1;
      if (!alive(is) && healthOf(is) === 0) {
        this.centre(cellOf(was), this.other).setY(30);
        pool.burst(this.other, kindOf(k) === KIND.COW ? 0xf3efe6 : kindOf(k) === KIND.ZOMBIE ? 0x7fb069 : 0xe9e3d0, 0.55, time);
      }
    }
    if (pressOf(now, this.shape) === PRESS.CONFIRM && !sleepingOf(before.words)) this.menuFlash[cursorOf(before.words)] = 1;
  }

  private record(frame: FrameUpdate, now: EnvFrame, step: number): void {
    const ledger = this.ledger, words = now.words;
    ledger.step = step;
    ledger.log = frame.eventLog;
    ledger.game = words[CXW.GAME]!;
    ledger.time = words[CXW.TIME]!;
    ledger.reward = now.reward;
    let count = 0;
    for (let bits = words[CXW.ACHIEVED]!; bits; bits &= bits - 1) count++;
    ledger.achieved = count;
    const before = step > 0 ? frame.frameAt(step - 1) : null;
    const press = pressOf(now, this.shape);
    if (loadingOf(words)) {
      ledger.action = 'waiting for the next world';
      ledger.note = `generating the next world · stage ${stageOf(words)}`;
      return;
    }
    if (!before || before.step !== step - 1 || before.words[CXW.GAME] !== words[CXW.GAME]) ledger.action = step > 0 ? 'born into a new world' : '—';
    else if (sleepingOf(before.words) && press !== PRESS.NEXT) ledger.action = 'asleep';
    else if (press === PRESS.NEXT) ledger.action = `menu → ${ACTION_NAMES[this.shape.menu[cursorOf(words)]!]}`;
    else if (press === PRESS.CONFIRM) ledger.action = `confirm · ${ACTION_NAMES[this.shape.menu[cursorOf(before.words)]!]}`;
    else {
      const facing = this.shape.moves[press]! - ACTION.LEFT, moved = cellOf(words[CXW.PLAYER]!) !== cellOf(before.words[CXW.PLAYER]!);
      ledger.action = `move ${ARROWS[facing]}${moved ? '' : ' · blocked'}`;
    }
    const cell = cellOf(words[CXW.PLAYER]!), [fx, fy] = FACING[facingOf(words)]!, slot = readoutOf(words);
    const reading = slot < ITEMS.length ? `${ITEMS[slot]} ×${itemCount(words, slot)}` : this.named(now, cell, this.shape.readout[slot - ITEMS.length]!);
    ledger.note = `${sleepingOf(words) ? 'asleep · ' : ''}facing ${this.named(now, cell, [fy, fx])} · reading ${reading}`;
  }

  private named(now: EnvFrame, cell: number, [dy, dx]: readonly [number, number]): string {
    const x = (cell & 63) + dx, y = (cell >>> 6) + dy;
    if (x < 0 || y < 0 || x >= this.grid.w || y >= this.grid.h) return BLOCK_NAMES[BLOCK.OUT]!;
    const target = y * this.grid.w + x;
    for (let k = 0; k < MOB_SLOTS; k++) if (alive(now.words[CXW.ZOMBIE + k]!) && cellOf(now.words[CXW.ZOMBIE + k]!) === target) return MOB_NAMES[kindOf(k)]!;
    return BLOCK_NAMES[blockAt(now.view, target)]!;
  }

  private placeHero(now: EnvFrame, next: EnvFrame, alpha: number): THREE.Vector3 {
    const hero = this.hero!, same = now.words[CXW.GAME] === next.words[CXW.GAME] && !loadingOf(now.words);
    const from = cellOf(now.words[CXW.PLAYER]!), to = same ? cellOf(next.words[CXW.PLAYER]!) : from;
    const position = this.centre(from, new THREE.Vector3()), goal = this.centre(to, this.other);
    const moving = from !== to, k = ease(alpha);
    position.lerp(goal, k);
    let angle = FACING_ANGLE[facingOf(now.words)]!;
    const press = pressOf(next, this.shape), asleep = sleepingOf(now.words);
    let strike = 0, phase = 0;
    if (same && next !== now) {
      angle = turn(angle, FACING_ANGLE[facingOf(next.words)]!, ease(clamp01(alpha * 2.5)));
      if (moving) phase = alpha;
      else if (press < PRESS.CONFIRM && !asleep) {
        const [fx, fy] = FACING[facingOf(next.words)]!;
        position.x += fx * 0.13 * Math.sin(Math.PI * alpha);
        position.z += fy * 0.13 * Math.sin(Math.PI * alpha);
      }
      if (press === PRESS.CONFIRM && !asleep && this.shape.menu[cursorOf(now.words)] !== ACTION.NOOP && this.shape.menu[cursorOf(now.words)] !== ACTION.SLEEP) strike = Math.sin(Math.PI * alpha);
    }
    const sleep = clamp01(asleep ? (same && next !== now && !sleepingOf(next.words) ? 1 - alpha * 2 : 1) : same && next !== now && sleepingOf(next.words) ? alpha * 2 : 0);
    const fatal = next !== now && next.done && perished(now, next);
    const dying = loadingOf(now.words) && vital(now.words, 0) === 0 ? 1 : fatal ? ease(alpha) : 0;
    hero.begin();
    hero.add(position.x, position.z, [angle, 1, Math.max(this.hurt, dying * 0.35), phase], [strike, sleep, dying, 0]);
    hero.commit();
    return position;
  }

  private follow(hero: THREE.Vector3): void {
    const { left, right, top, bottom } = this.console(), unit = (this.camera.top - this.camera.bottom) / Math.max(1, this.pixels);
    const x = Math.round((hero.x + (left + right) / 2) / unit) * unit, z = Math.round((hero.z + (top + bottom) / 2) / unit) * unit;
    this.camera.position.set(x, HEIGHT, z);
    this.camera.updateMatrixWorld();
    this.origin.set(x - (left + right) / 2, z - (top + bottom) / 2);
    this.u!.view.value.set(this.origin.x + this.grid.w / 2, this.origin.y + this.grid.h / 2);
  }

  private brighten(frame: FrameUpdate, now: EnvFrame, next: EnvFrame, alpha: number): void {
    if (loadingOf(now.words)) {
      for (let s = now.step - 1; s >= Math.max(0, now.step - 64); s--) {
        const earlier = frame.frameAt(s);
        if (earlier.step !== s) break;
        if (loadingOf(earlier.words)) continue;
        this.daylight = earlier.observations[13]!;
        break;
      }
    } else {
      const a = now.observations[13]!, b = next.words[CXW.GAME] === now.words[CXW.GAME] && !loadingOf(next.words) ? next.observations[13]! : a;
      this.daylight = a + (b - a) * alpha;
    }
    this.ledger.daylight = this.daylight;
    const on = this.debugView ? 0 : 1, day = this.daylight * on;
    this.sun.value.set(0.86 * day, 0.82 * day, 0.7 * day);
  }

  private mobFacing(frame: FrameUpdate, slot: number, step: number, fallback: number): number {
    for (let s = step; s > Math.max(0, step - 12); s--) {
      const a = frame.frameAt(s - 1), b = frame.frameAt(s);
      if (a.step !== s - 1 || b.step !== s || a.words[CXW.GAME] !== b.words[CXW.GAME]) break;
      const wa = a.words[CXW.ZOMBIE + slot]!, wb = b.words[CXW.ZOMBIE + slot]!;
      if (!alive(wa) || !alive(wb)) break;
      const ca = cellOf(wa), cb = cellOf(wb);
      if (ca !== cb) return Math.atan2(-((cb & 63) - (ca & 63)), -((cb >>> 6) - (ca >>> 6)));
    }
    return fallback;
  }

  private placeMobs(frame: FrameUpdate, now: EnvFrame, next: EnvFrame, alpha: number, hero: THREE.Vector3): void {
    const same = now.words[CXW.GAME] === next.words[CXW.GAME] && next !== now && !loadingOf(now.words);
    const batches = [this.zombies!, this.cows!, this.skeletons!, this.arrows!];
    for (const batch of batches) batch.begin();
    const night = 1 - clamp01((this.daylight - 0.2) / 0.4), k = ease(alpha);
    for (let slot = 0; slot < MOB_SLOTS; slot++) {
      const kind = kindOf(slot), batch = batches[kind]!, wa = now.words[CXW.ZOMBIE + slot]!, wb = same ? next.words[CXW.ZOMBIE + slot]! : wa;
      const aliveA = alive(wa), aliveB = alive(wb), ca = cellOf(wa), cb = cellOf(wb);
      const near = Math.abs((ca & 63) - (cb & 63)) + Math.abs((ca >>> 6) - (cb >>> 6)) <= 1;
      const draw = (from: number, to: number, t: number, fade: number, word: number, angle: number, act: number) => {
        const a = this.centre(from, this.at), b = this.centre(to, this.other);
        a.lerp(b, t);
        batch.add(a.x, a.z, [angle, fade, this.mobHurt[slot]!, from !== to ? alpha : 0], [act, 0, kind === KIND.ARROW ? 0 : healthOf(word), night]);
      };
      const toward = (cell: number) => {
        const dx = hero.x - this.centre(cell, this.at).x, dz = hero.z - this.at.z;
        return Math.abs(dz) > Math.abs(dx) ? (dz < 0 ? 0 : Math.PI) : dx < 0 ? Math.PI / 2 : -Math.PI / 2;
      };
      const adjacent = (cell: number) => Math.abs(hero.x - this.centre(cell, this.at).x) + Math.abs(hero.z - this.at.z) < 1.5;
      if (kind === KIND.ARROW) {
        const angle = FACING_ANGLE[arrowDirection(aliveA ? wa : wb)]!;
        if (aliveA && aliveB && near) draw(ca, cb, alpha, 1, wa, angle, 0);
        else {
          if (aliveA) {
            const [dx, dy] = FACING[arrowDirection(wa)]!, x = (ca & 63) + dx, y = (ca >>> 6) + dy;
            draw(ca, x >= 0 && y >= 0 && x < this.grid.w && y < this.grid.h ? y * this.grid.w + x : ca, alpha * 0.5, 1 - alpha, wa, angle, 0);
          }
          if (aliveB && same) draw(cb, cb, 0, alpha, wb, angle, 0);
        }
        continue;
      }
      const attack = same && restOf(wb) > restOf(wa) ? Math.sin(Math.PI * alpha) : 0;
      const facing = aliveA && aliveB && ca !== cb && near ? Math.atan2(-((cb & 63) - (ca & 63)), -((cb >>> 6) - (ca >>> 6))) : this.mobFacing(frame, slot, frame.state.step, kind === KIND.COW ? Math.PI : toward(ca));
      const angle = (kind === KIND.ZOMBIE && (attack > 0 || adjacent(ca))) || (kind === KIND.SKELETON && restOf(wa) > 0) ? toward(ca) : facing;
      if (aliveA && aliveB && near) draw(ca, cb, k, 1, wa, angle, attack);
      else {
        if (aliveA) draw(ca, ca, 0, 1 - alpha, wa, angle, 0);
        if (aliveB && same) draw(cb, cb, 0, alpha, wb, kind === KIND.COW ? Math.PI : toward(cb), 0);
      }
    }
    for (const batch of batches) batch.commit();
    const mini = this.mini!;
    for (let slot = 0; slot < MOB_SLOTS; slot++) {
      const word = now.words[CXW.ZOMBIE + slot]!, cell = cellOf(word);
      mini.mobData[slot]!.set((cell & 63) + 0.5, (cell >>> 6) + 0.5, kindOf(slot), alive(word) ? 1 : 0);
    }
    mini.player.value.set(hero.x + this.grid.w / 2, hero.z + this.grid.h / 2);
  }

  private drawHud(now: EnvFrame, next: EnvFrame, alpha: number, hero: THREE.Vector3): void {
    const icons = this.icons!, h = this.half, ox = this.origin.x, oz = this.origin.y, words = now.words, loading = loadingOf(words);
    icons.begin();
    const panel = (x0: number, z0: number, x1: number, z1: number) => icons.add(ox + (x0 + x1) / 2, oz + (z0 + z1) / 2, x1 - x0, z1 - z0, [0, 0, 0, MODE.PANEL], [1, 0, 0, 1], [(x1 - x0) / (z1 - z0), 0, 0.26 / ((z1 - z0) / 2), 0]);
    const menuTop = -h.y - MENU_ROW - 0.42, pitch = (2 * h.x) / this.shape.menu.length;
    panel(-h.x - 0.12, menuTop - 0.03, h.x + 0.12, -h.y - 0.12);
    panel(-h.x - 0.12, h.y + 0.12, h.x + 0.12, h.y + ITEM_ROW + 0.45);
    panel(h.x + GUTTER - 0.12, -h.y - 0.05, h.x + GUTTER + 1.62, h.y + 0.05);
    icons.add(ox, oz, 2 * h.x, 2 * h.y, [0, 0, 0, MODE.FRAME], [1, 0, 0, loading ? 0.3 : 0.9], [h.x / h.y, 0, 0.18 / h.y, 0]);
    if (!loading) icons.add(ox, oz, 3, 3, [0, 0, 0, MODE.FRAME], [1, 0, 0, 0.3], [1, 0, 0.2 / 1.5, 0]);
    const cursor = cursorOf(words), press = pressOf(next, this.shape), stepping = next !== now && press === PRESS.NEXT;
    this.shape.menu.forEach((action, i) => {
      const [glyph, tier] = ACTION_GLYPHS[action]!, x = -h.x + pitch * (i + 0.5);
      const ring = i === cursor ? (stepping ? 1 - alpha : 1) : stepping && i === cursorOf(next.words) ? alpha : 0;
      icons.add(ox + x, oz - h.y - MENU_ROW, pitch * 0.9, pitch * 0.9, [glyph, tier, -1, MODE.SLOT], [1, ring, this.menuFlash[i]!, 1]);
    });
    const vitalPitch = (2 * h.x) / VITAL_GLYPHS.length;
    VITAL_GLYPHS.forEach(([glyph], i) => {
      icons.add(ox - h.x + vitalPitch * (i + 0.5), oz + h.y + VITAL_ROW, vitalPitch * 0.94, 0.62, [glyph, TIER.NONE, vital(words, i), MODE.VITAL], [1, 0, 0, 1], [(vitalPitch * 0.94) / 0.62, 0, 0, 0]);
    });
    const itemPitch = (2 * h.x) / ITEMS.length, readout = readoutOf(words);
    ITEM_GLYPHS.forEach(([glyph, tier], i) => {
      const count = itemCount(words, i);
      icons.add(ox - h.x + itemPitch * (i + 0.5), oz + h.y + ITEM_ROW, itemPitch * 0.92, itemPitch * 0.92, [glyph, tier, count, MODE.SLOT], [count > 0 ? 1 : 0, 0, this.itemFlash[i]!, 1], [1, !loading && readout === i ? 1 : 0, 0, 0]);
    });
    const achieved = words[CXW.ACHIEVED]!, rows = Math.ceil(ACHIEVEMENTS.length / 2), rowPitch = (2 * h.y) / rows;
    ACHIEVEMENT_GLYPHS.forEach(([glyph, tier], i) => {
      const x = h.x + GUTTER + 0.37 + (i % 2) * 0.76, z = -h.y + rowPitch * (Math.floor(i / 2) + 0.5), lit = (achieved >>> i) & 1;
      icons.add(ox + x, oz + z, rowPitch * 0.9, rowPitch * 0.9, [glyph, tier, -1, MODE.SLOT], [lit, lit ? 0.6 : 0, this.badgeFlash[i]!, 1], [1, 0, 0, 1]);
    });
    if (!loading) {
      const player = cellOf(words[CXW.PLAYER]!), px = hero.x - this.centre(player, this.at).x, pz = hero.z - this.at.z;
      if (readout >= ITEMS.length) {
        const [dy, dx] = this.shape.readout[readout - ITEMS.length]!;
        icons.add(this.at.x + dx + px, this.at.z + dy + pz, 1, 1, [GLYPH.RETICLE, 0, -1, MODE.BARE], [1, 0, 0, 0.95]);
      }
      const [fx, fy] = FACING[facingOf(words)]!, tx = (player & 63) + fx, ty = (player >>> 6) + fy;
      if (tx >= 0 && ty >= 0 && tx < this.grid.w && ty < this.grid.h) icons.add(this.at.x + fx + px, this.at.z + fy + pz, 1, 1, [GLYPH.CORNERS, 0, -1, MODE.BARE], [1, 0, 0, 0.8]);
    }
    for (const flyer of this.flyers) {
      const t = ease(flyer.age), goal = flyer.target === 'item'
        ? new THREE.Vector2(ox - h.x + itemPitch * (flyer.index + 0.5), oz + h.y + ITEM_ROW)
        : new THREE.Vector2(ox + h.x + GUTTER + 0.37 + (flyer.index % 2) * 0.76, oz - h.y + rowPitch * (Math.floor(flyer.index / 2) + 0.5));
      const x = flyer.from.x + (goal.x - flyer.from.x) * t, z = flyer.from.y + (goal.y - flyer.from.y) * t - Math.sin(Math.PI * t) * 1.2, size = 0.62 + 0.25 * Math.sin(Math.PI * t);
      icons.add(x, z, size, size, [flyer.glyph, flyer.tier, -1, MODE.BARE], [1, 0, 0.3 * (1 - t), Math.min(1, (1 - flyer.age) * 4)]);
    }
    icons.commit();
    const mini = this.mini!;
    mini.mesh.position.set(ox - h.x - GUTTER - MAP_SIZE / 2, 0.1, oz - h.y + MAP_SIZE / 2 - 0.05);
    mini.pulse.value = (this.clock * 0.8) % 1;
  }
}


export default (() => new CraftaxScene()) satisfies SceneFactory;
