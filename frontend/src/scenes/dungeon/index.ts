import * as THREE from 'three/webgpu';
import { uniform, vec3 } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { LIGHT_LAYER } from '@kit/light2d';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  ACT_NEXT, ACT_USE, AWAKE, CORRIDOR, DOOR, DW_BLIND, DW_COLD, DW_CONFUSED, DW_DEEPEST, DW_DEPTH, DW_EVENT, DW_FIRE, DW_FOOD, DW_GOLD, DW_HP, DW_LEVEL, DW_MAGNITUDE, DW_POISON,
  DW_RATIONS, DW_RUN, DW_SLOT, DW_STAIRS, DW_TURN, DW_X, DW_XP, DW_Y, EFFECTS, effectOf, EV_BUMP, EV_DESCEND, EV_EAT, EV_GOLD, EV_HIT, EV_HURT, EV_KILL, EV_MISS, EV_NONE,
  EV_POTION, EV_QUAFF, EV_RATION, EV_REST, EV_RUN, EV_SICK, EV_STARVE, EV_WIN, floorKey, GLYPH_MONSTER, GLYPH_UNSEEN, GOLD, GUARD, itemsOf, monstersOf, most, open, POTION,
  potions, RATION, REACH, SIDE, SPAN, STAIRS, STEP_X, STEP_Y, terrainAt, turned, WALL, type Item, type Monster,
} from '@sim/envs/dungeon';
import type { EnvFrame, GridCell } from '@sim/types';
import { constantList, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { Batch, clamp01, codesOf, EARTH_TINTS, ease, ELEMENT_TINTS, flat, headingOf, POTION_NAMES, POTION_TINTS } from './art';
import { Board, EMPTY, groundMaterial, HALF, occluderMask, veilMaterial, type VeilUniforms } from './board';
import { HERO_QUAD, heroMaterial, MONSTER_QUAD, monsterMaterial, PLANS, TORCH, type Plan } from './beings';
import { BADGE, badgeMaterial, BAR, barMaterial, EFFECT_TINTS, FX, FX_QUAD, fxMaterial, ICON, ITEM_QUAD, ITEM_SCALE, itemMaterial, SCONCE_QUAD, sconceMaterial, SLOT, slotMaterial, TIER, tierMaterial } from './things';

const ACCENT = TOKENS.accents.Dungeon!;
const COMPASS = ['↑ north', '↓ south', '← west', 'east →'] as const;
const HOP = 0.11;
const STRIKE = 0.18;
const CLOSE = 0.3;
const OPEN = 0.65;
const FADE = 0.35;
const DEATH = 0.55;
const FX_SLOTS = 48;
const MAP_EDGE = 0.6;
const LEFT_X = -21;
const RIGHT_X = 17.85;
const PANEL_TOP = -14.2;
const HURT = 0xff5a4a;
const HEAL = 0x8fe07a;
const GOLDEN = TOKENS.celebrationGold;
const LAG = 24;
const GHOSTS = 4;
const POTION_COLOURS = POTION_TINTS.map((tint) => new THREE.Color(tint));
const WHITE = new THREE.Color(0xffffff);
const HERO_SCALE = 1.7;

interface Kind { plan: Plan; hp: number; size: number; skin: THREE.Color; trim: THREE.Color; horns: number; armour: number; damage: number; fire: boolean }

interface Snap {
  step: number;
  words: Uint32Array;
  view: Uint32Array;
  obs: Float32Array;
  monsters: Monster[];
  ids: number[];
  items: Item[];
  key: number;
  action: number;
  reward: number;
  done: boolean;
}

interface Actor {
  from: THREE.Vector2;
  via: THREE.Vector2 | null;
  to: THREE.Vector2;
  t: number;
  heading: number;
  aim: number;
  flash: number;
  phase: number;
  spawn: number;
}

interface Ghost { type: number; at: THREE.Vector2; heading: number; t: number; phase: number; flag: number }

interface Effect { kind: number; x: number; z: number; t: number; life: number; tint: THREE.Color; strength: number; code: number; spin: number; codes: number[]; count: number; dx: number; dy: number; scale: number }

type HeroKind = 'rest' | 'move' | 'bump' | 'strike' | 'use';

interface Pending { snap: Snap; at: number; kind: 'descend' | 'death' | 'win' }

const world = (x: number, y: number, out: THREE.Vector2): THREE.Vector2 => out.set(x - HALF + 0.5, y - HALF + 0.5);
const signed = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace('-', '−');

class DungeonScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.6, fog: { from: ['cave', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'dungeon';

  private floors = 6;
  private radius = 1;
  private boss = 14;
  private guardian = 0;
  private kinds: Kind[] = [];
  private glyphs: number[] = [];
  private board: Board | null = null;
  private readonly time = uniform(0);
  private readonly owned: THREE.Mesh[] = [];
  private batches: Batch[] = [];
  private items: Batch | null = null;
  private sconces: Batch | null = null;
  private monsters = new Map<Plan, Batch>();
  private hero: Batch | null = null;
  private fx: Batch | null = null;
  private slots: Batch | null = null;
  private tiers: Batch | null = null;
  private bars: Batch | null = null;
  private badges: Batch | null = null;
  private torch: THREE.Mesh | null = null;
  private readonly torchPower = uniform(10);
  private lamps: THREE.InstancedMesh | null = null;
  private embers: THREE.InstancedMesh | null = null;
  private veil: VeilUniforms | null = null;

  private shown: Snap | null = null;
  private pending: Pending | null = null;
  private readonly actors = new Map<number, Actor>();
  private ghosts: Ghost[] = [];
  private effects: Effect[] = [];
  private nextId = 1;
  private step = -1;
  private clock = 0;
  private cursor = 0;
  private iris = { from: 999, to: 999, at: -9, span: 1, x: 0, y: 0 };
  private sightAt = -9;
  private readonly heroState = { from: new THREE.Vector2(), to: new THREE.Vector2(), t: 1, kind: 'rest' as HeroKind, dx: 0, dy: -1, heading: 0, aim: 0, flash: 0, sink: 0, fall: 0 };
  private readonly scratch = new THREE.Vector2();
  private readonly other = new THREE.Vector2();
  private readonly lampMatrix = new THREE.Matrix4();
  private readonly tone = new THREE.Color();
  private readonly codes = [-1, -1, -1, -1];

  override legend: LegendSpec = {
    title: 'DUNGEON',
    rows: [
      { key: 'floor', label: 'floor', tint: ACCENT, value: () => this.floorLabel() },
      { key: 'hero', label: 'hero', tint: HURT, value: () => this.heroLabel() },
      { key: 'food', label: 'food', tint: 0xd29b58, value: () => this.foodLabel() },
      { key: 'slot', label: 'slot', tint: 0xdcf4ff, value: () => this.slotLabel() },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionLabel() },
      { key: 'turn', label: 'turn', tint: GOLDEN, value: () => this.eventLabel() },
      { key: 'status', label: 'status', tint: 0x9ee04e, value: () => this.statusLabel() },
      { key: 'reward', label: 'reward', tint: GOLDEN, value: () => signed(this.shown?.reward ?? 0) },
    ],
    note: () => this.shown ? `run ${formatCount(this.shown.words[DW_RUN]! + 1)} · turn ${formatCount(this.shown.words[DW_TURN]!)}` : '',
    bar: () => this.shown ? this.shown.words[DW_HP]! / most(this.shown.words[DW_LEVEL]!) : 0,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.floors = parameter(env, 'floors');
    this.radius = parameter(env, 'radius');
    this.boss = parameter(env, 'boss');
    this.guardian = parameter(env, 'guardian');
    const hp = constantList(env, 'DUNGEON_HP'), attack = constantList(env, 'DUNGEON_ATTACK'), defence = constantList(env, 'DUNGEON_DEFENCE'), speed = constantList(env, 'DUNGEON_SPEED');
    const damage = constantList(env, 'DUNGEON_DAMAGE'), behaviour = constantList(env, 'DUNGEON_BEHAVIOUR');
    this.glyphs = constantList(env, 'DUNGEON_GLYPH');
    const low = Math.log(Math.min(...hp)), high = Math.log(Math.max(...hp)), plain = damage.flatMap((d, t) => (d === 0 ? [t] : []));
    this.kinds = hp.map((life, t) => {
      const b = behaviour[t]!, d = damage[t]!;
      const plan: Plan = b === 1 ? 'flier' : b === 2 ? 'caster' : b === 3 ? 'growth' : speed[t]! > Math.min(...speed) ? 'runner' : 'walker';
      const skin = new THREE.Color(d === 0 ? EARTH_TINTS[plain.indexOf(t) % EARTH_TINTS.length]! : [0, 0xc4502a, 0x7fb8d8, 0x6f9a3a, 0x5a4278][d]!);
      return { plan, hp: life, size: 1.05 + 0.62 * (Math.log(life) - low) / Math.max(1e-6, high - low), skin, trim: new THREE.Color(ELEMENT_TINTS[d]!), horns: attack[t]! >= 6 ? 1 : 0, armour: clamp01((defence[t]! - 1) / 3), damage: d, fire: d === 1 };
    });
    const bottom = HALF - 1 + MAP_EDGE, top = -HALF + 1 - MAP_EDGE;
    return new THREE.Box2(new THREE.Vector2(LEFT_X - BAR[0] / 2 - 0.4, top), new THREE.Vector2(RIGHT_X + SLOT[0] / 2 + 0.4, bottom - 1));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8391b8).multiplyScalar(0.74);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 }, goal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[0]!;
        agent.y = state.pose[1]!;
        return agent;
      },
      goalCell: () => {
        const stairs = this.shown?.words[DW_STAIRS] ?? 0;
        if (!stairs) return null;
        goal.x = stairs & 31;
        goal.y = stairs >>> 5;
        return goal;
      },
      debug: (_frame, kit) => this.groundTruth(kit),
    };
  }

  protected override poolOptions() {
    return { burstSize: 18, sighRadius: 2.5, sighHeight: 2 };
  }

  protected async build(ctx: SceneContext, { light, pool }: Stage2D): Promise<void> {
    const board = new Board(ctx.seed);
    this.board = board;
    this.scene.background = new THREE.Color(0x050608);
    const lit = light.lit, accent = this.accentOf(ctx), time = this.time;

    const ground = this.own(new THREE.Mesh(flat(120, 80), groundMaterial(board, lit)), 0);
    ground.position.set(-4, 0, 0);
    const occluder = light.occluder(flat(SIDE + 8, SIDE + 8));
    const shade = occluder.material as THREE.MeshBasicNodeMaterial;
    shade.opacityNode = occluderMask(board);
    shade.alphaTest = 0.5;
    this.own(occluder, 0);

    const torch = light.emitter(new THREE.CircleGeometry(0.15, 24).rotateX(-Math.PI / 2), rgb(0xffb45a).mul(this.torchPower));
    this.torch = this.own(torch, 0);
    const lamp = light.emitter(new THREE.CircleGeometry(0.1, 20).rotateX(-Math.PI / 2), vec3(1));
    this.lamps = this.ownInstanced(instancedMesh(lamp.geometry, lamp.material, 48));
    const ember = light.emitter(new THREE.CircleGeometry(0.16, 20).rotateX(-Math.PI / 2), vec3(1));
    this.embers = this.ownInstanced(instancedMesh(ember.geometry, ember.material, 24));

    this.sconces = this.batch(new Batch(48, SCONCE_QUAD, ['aSconce'], sconceMaterial(lit, time), 1));
    this.items = this.batch(new Batch(16, ITEM_QUAD, ['aItem', 'aTint'], itemMaterial(lit, time), 2));
    for (const plan of PLANS) this.monsters.set(plan, this.batch(new Batch(28, MONSTER_QUAD, ['aSkin', 'aTrim', 'aForm', 'aState', 'aMark'], monsterMaterial(plan, lit, time), 3)));
    const veil = veilMaterial();
    this.veil = veil.uniforms;
    this.own(new THREE.Mesh(flat(140, 90), veil.material), 4).position.set(-4, 0, 0);
    this.hero = this.batch(new Batch(1, HERO_QUAD, ['aPose', 'aStatus', 'aVital'], heroMaterial(accent, lit, time), 5));
    this.fx = this.batch(new Batch(FX_SLOTS, FX_QUAD, ['aFx', 'aTint', 'aText', 'aMeta'], fxMaterial(), 6));
    this.slots = this.batch(new Batch(10, SLOT, ['aSlot', 'aTint', 'aText', 'aMeta'], slotMaterial(accent, time), 7));
    this.tiers = this.batch(new Batch(12, TIER, ['aTier', 'aText'], tierMaterial(accent, time), 7));
    this.bars = this.batch(new Batch(3, BAR, ['aHud', 'aTint', 'aText', 'aMeta'], barMaterial(), 7));
    this.badges = this.batch(new Batch(5, BADGE, ['aHud', 'aTint'], badgeMaterial(time), 7));
    pool.group.traverse((object) => { object.renderOrder = 6.5; });
  }

  private own<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.owned.push(mesh);
    this.scene.add(mesh);
    return mesh;
  }

  private ownInstanced(mesh: THREE.InstancedMesh): THREE.InstancedMesh {
    mesh.layers.set(LIGHT_LAYER);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(mesh.count * 3), 3);
    mesh.count = 0;
    return this.own(mesh, 0);
  }

  private batch(batch: Batch): Batch {
    this.batches.push(batch);
    this.scene.add(batch.mesh);
    return batch;
  }

  protected override drop(): void {
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
    for (const batch of this.batches) batch.dispose();
    this.batches = [];
    this.board?.dispose();
    this.board = null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal;
    this.clock += dt;
    this.time.value = this.clock;
    pool.update(dt, time);
    const step = frame.state.step;
    if (frame.scrubbed || !this.shown || step < this.step) this.settle(frame, step);
    else for (let s = this.step + 1; s <= step; s++) {
      if (this.pending) {
        if (this.clock - this.pending.at < CLOSE && step - this.step <= LAG) break;
        this.swap(pool, time);
      }
      this.advance(frame, s, pool, time);
      if (this.step < s) this.step = s;
    }
    if (this.pending && this.clock - this.pending.at >= CLOSE) this.swap(pool, time);
    this.animate(dt);
    this.draw();
  }

  private snapshot(f: EnvFrame): Snap {
    const monsters = monstersOf(f);
    return {
      step: f.step,
      words: Uint32Array.from(f.words),
      view: Uint32Array.from(f.view),
      obs: Float32Array.from(f.observations),
      monsters,
      ids: monsters.map(() => 0),
      items: itemsOf(f),
      key: floorKey(f),
      action: f.step > 0 ? Math.trunc(f.actions[0]!) : -1,
      reward: f.reward,
      done: f.done !== 0,
    };
  }

  private settle(frame: FrameUpdate, step: number): void {
    const f = frame.frameAt(step);
    const snap = this.snapshot(f);
    this.pending = null;
    this.install(snap, true);
    this.effects = [];
    this.ghosts = [];
    this.iris = { from: 999, to: 999, at: -9, span: 1, x: 0, y: 0 };
    this.veil?.tint.value.set(0x03050c);
    let back = step;
    while (back > Math.max(0, step - 64)) {
      const g = frame.frameAt(back);
      if (g.step !== back) break;
      const a = Math.trunc(g.actions[0]!);
      if (a >= 0 && a < 4) {
        this.heroState.dx = STEP_X[a]!;
        this.heroState.dy = STEP_Y[a]!;
        break;
      }
      back--;
    }
    this.heroState.heading = this.heroState.aim = headingOf(this.heroState.dx, this.heroState.dy);
    this.cursor = snap.words[DW_SLOT]!;
    this.sightKey.key = -1;
    this.step = step;
  }

  private install(snap: Snap, fresh: boolean): void {
    const board = this.board!;
    if (board.key !== snap.key || fresh) board.paint(snap.view, snap.key);
    this.actors.clear();
    snap.ids = snap.monsters.map((m) => {
      const id = this.nextId++;
      this.actors.set(id, this.actor(m, m, null, 1));
      return id;
    });
    this.shown = snap;
    const hero = this.heroState;
    world(snap.words[DW_X]!, snap.words[DW_Y]!, hero.to);
    hero.from.copy(hero.to);
    hero.t = 1;
    hero.kind = 'rest';
    hero.sink = 0;
    hero.fall = 0;
    hero.flash = 0;
    this.sightAt = -9;
    this.placeSconces();
  }

  private actor(from: Monster, to: Monster, via: THREE.Vector2 | null, spawn: number): Actor {
    return { from: world(from.x, from.y, new THREE.Vector2()), via, to: world(to.x, to.y, new THREE.Vector2()), t: 0, heading: ((to.x * 7 + to.y * 3) % 4) * Math.PI / 2, aim: 0, flash: 0, phase: ((to.x * 13 + to.y * 7 + to.type * 5) % 17) / 17, spawn };
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const next = frame.frameAt(s), prev = frame.frameAt(s - 1);
    if (next.step !== s || prev.step !== s - 1) {
      this.settle(frame, s);
      return;
    }
    const was = this.shown!, now = this.snapshot(next);
    if (!turned(prev, next)) {
      now.ids = was.ids;
      this.shown = now;
      return;
    }
    if (now.key !== was.key || now.done) {
      this.transition(was, now, pool, time);
      return;
    }
    this.turn(was, now, pool, time);
  }

  private turn(was: Snap, now: Snap, pool: CelebrationPool, time: number): void {
    const board = this.board!, event = now.words[DW_EVENT]!, magnitude = now.words[DW_MAGNITUDE]!;
    const hx = was.words[DW_X]!, hy = was.words[DW_Y]!, nx = now.words[DW_X]!, ny = now.words[DW_Y]!;
    const killed = event === EV_KILL ? this.killedIndex(was, now, hx, hy) : -1;
    now.ids = this.match(was, now, killed);
    const hero = this.heroState, action = now.action;
    hero.from.copy(this.shownHero());
    world(nx, ny, hero.to);
    hero.t = 0;
    hero.kind = 'rest';
    const confused = was.words[DW_CONFUSED]! > 0;
    if (nx !== hx || ny !== hy) {
      hero.kind = 'move';
      hero.dx = nx - hx;
      hero.dy = ny - hy;
    } else if (event === EV_HIT || event === EV_MISS || event === EV_KILL) {
      const target = this.target(was, now, hx, hy, action, confused, killed);
      hero.kind = 'strike';
      hero.dx = target[0];
      hero.dy = target[1];
      const tx = hx + target[0], ty = hy + target[1];
      world(tx, ty, this.scratch);
      const dir: [number, number] = [target[0], -target[1]];
      if (event === EV_MISS) this.emit(FX.whiff, this.scratch.x, this.scratch.y, 0.4, 0xb8c0cc, { dir });
      else this.emit(FX.slash, this.scratch.x, this.scratch.y, 0.32, 0xf4f8ff, { dir });
      if (event === EV_HIT) this.number(this.scratch.x, this.scratch.y, magnitude, '-', 0xf4f0e8);
    } else if (event === EV_BUMP && action >= 0 && action < 4) {
      const dx = STEP_X[action]!, dy = STEP_Y[action]!, solid = !board.open(hx + dx, hy + dy);
      hero.kind = 'bump';
      hero.dx = solid ? dx : hero.dx;
      hero.dy = solid ? dy : hero.dy;
      this.emit(FX.dust, hero.to.x, hero.to.y, 0.4, 0x9a8e7e, { dir: [hero.dx, -hero.dy] });
    } else if (action === ACT_USE) hero.kind = 'use';
    hero.aim = headingOf(hero.dx, hero.dy);
    if (killed >= 0) this.slay(was, killed, pool, time, now.reward);
    for (let j = 0; j < now.monsters.length; j++) {
      const m = now.monsters[j]!, id = now.ids[j]!, i = was.ids.indexOf(id), actor = this.actors.get(id)!;
      const before = i >= 0 ? was.monsters[i]! : null;
      const fromAt = this.actorAt(actor, this.scratch).clone();
      actor.from.copy(fromAt);
      world(m.x, m.y, actor.to);
      actor.via = null;
      if (before && Math.abs(m.x - before.x) + Math.abs(m.y - before.y) === 2 && m.x !== before.x && m.y !== before.y) {
        const viaX = board.open(m.x, before.y) ? m.x : before.x, viaY = viaX === m.x ? before.y : m.y;
        actor.via = world(viaX, viaY, new THREE.Vector2());
      }
      actor.t = 0;
      if (before && (m.x !== before.x || m.y !== before.y)) actor.aim = headingOf(Math.sign(m.x - before.x), Math.sign(m.y - before.y));
      else if (m.flag & AWAKE && Math.abs(m.x - nx) + Math.abs(m.y - ny) <= 6) actor.aim = headingOf(Math.sign(nx - m.x) || 0, Math.sign(ny - m.y) || (nx === m.x ? 1 : 0));
      if (before && m.hp < before.hp) actor.flash = 1;
      if (!before) {
        actor.spawn = 0;
        this.emit(FX.puff, actor.to.x, actor.to.y, 0.5, 0x8a93b0, { strength: 0.6 });
      }
    }
    const lost = was.words[DW_HP]! - now.words[DW_HP]!;
    if (lost > 0) {
      const tint = event === EV_STARVE ? 0xf2a93b : event === EV_SICK || event === EV_QUAFF + 5 ? 0x9ee04e : HURT;
      hero.flash = 1;
      this.number(hero.to.x, hero.to.y, lost, '-', tint);
    }
    if (now.words[DW_LEVEL]! > was.words[DW_LEVEL]!) {
      this.emit(FX.icon, hero.to.x, hero.to.y, 1.2, GOLDEN, { code: ICON.star });
      pool.ripple({ x: hero.to.x, y: 0, z: hero.to.y }, GOLDEN, 0.3, time);
    }
    this.outcome(was, now, event, magnitude, pool, time);
    this.shown = now;
  }

  private outcome(was: Snap, now: Snap, event: number, magnitude: number, pool: CelebrationPool, time: number): void {
    const hero = this.heroState, x = hero.to.x, z = hero.to.y;
    if (event === EV_GOLD) {
      this.emit(FX.sparkle, x, z, 0.7, GOLDEN);
      this.number(x, z, magnitude, '+', GOLDEN);
      pool.burst({ x, y: 38, z }, GOLDEN, 0.3, time);
    } else if (event === EV_RATION) this.emit(FX.icon, x, z, 1, 0xd29b58, { code: ICON.loaf });
    else if (event === EV_POTION) this.emit(FX.icon, x, z, 1, POTION_TINTS[(magnitude - 1) & 7]!, { code: ICON.flask });
    else if (event === EV_EAT) {
      this.emit(FX.icon, x, z, 1, 0xd29b58, { code: ICON.loaf });
      this.emit(FX.puff, x, z, 0.45, 0xc9a26a, { strength: 0.7 });
    } else if (event >= EV_QUAFF && event < EV_QUAFF + 8) {
      const colour = was.words[DW_SLOT]! - 2, effect = event - EV_QUAFF;
      this.emit(FX.swirl, x, z, 0.85, POTION_TINTS[colour & 7]!, { spin: colour });
      this.emit(FX.icon, x, z, 1.1, EFFECT_TINTS[effect]!, { code: effect });
      const gained = now.words[DW_HP]! - was.words[DW_HP]!;
      if (gained > 0) this.number(x, z, gained, '+', HEAL);
    } else if (event === EV_REST) this.emit(FX.rest, x, z, 1, 0xb8c6e8);
    if (now.reward > 0 && event === EV_KILL) pool.pulse.fire(Math.min(1, now.reward / 200));
  }

  private target(was: Snap, now: Snap, hx: number, hy: number, action: number, confused: boolean, killed: number): [number, number] {
    if (!confused && action >= 0 && action < 4) return [STEP_X[action]!, STEP_Y[action]!];
    if (killed >= 0) return [was.monsters[killed]!.x - hx, was.monsters[killed]!.y - hy];
    for (let j = 0; j < now.monsters.length; j++) {
      const i = was.ids.indexOf(now.ids[j]!), m = now.monsters[j]!;
      if (i >= 0 && m.hp < was.monsters[i]!.hp) return [was.monsters[i]!.x - hx, was.monsters[i]!.y - hy];
    }
    for (const m of was.monsters) if (Math.abs(m.x - hx) + Math.abs(m.y - hy) === 1) return [m.x - hx, m.y - hy];
    return action >= 0 && action < 4 ? [STEP_X[action]!, STEP_Y[action]!] : [this.heroState.dx, this.heroState.dy];
  }

  private killedIndex(was: Snap, now: Snap, hx: number, hy: number): number {
    const n0 = was.monsters.length, last = n0 - 1;
    for (let k = 0; k < n0; k++) {
      const m = was.monsters[k]!;
      if (Math.abs(m.x - hx) + Math.abs(m.y - hy) !== 1) continue;
      let fits = now.monsters.length >= n0 - 1;
      for (let j = 0; j < n0 - 1 && fits; j++) {
        const source = j === k ? was.monsters[last]! : was.monsters[j]!, target = now.monsters[j]!;
        fits = target.type === source.type && target.hp <= source.hp && Math.abs(target.x - source.x) + Math.abs(target.y - source.y) <= 2;
      }
      if (fits) return k;
    }
    return -1;
  }

  private match(was: Snap, now: Snap, killed: number): number[] {
    const n0 = was.monsters.length, last = n0 - 1;
    return now.monsters.map((m, j) => {
      const i = killed < 0 ? (j < n0 ? j : -1) : j < n0 - 1 ? (j === killed ? last : j) : -1;
      if (i >= 0 && was.monsters[i]!.type === m.type && this.actors.has(was.ids[i]!)) return was.ids[i]!;
      const id = this.nextId++;
      this.actors.set(id, this.actor(m, m, null, 0));
      return id;
    });
  }

  private slay(was: Snap, k: number, pool: CelebrationPool, time: number, reward: number): void {
    const m = was.monsters[k]!, id = was.ids[k]!, actor = this.actors.get(id);
    const at = actor ? this.actorAt(actor, new THREE.Vector2()) : world(m.x, m.y, new THREE.Vector2());
    if (this.ghosts.length >= GHOSTS) this.ghosts.shift();
    this.ghosts.push({ type: m.type, at, heading: actor?.heading ?? 0, t: 0, phase: actor?.phase ?? 0, flag: m.flag });
    this.actors.delete(id);
    const kind = this.kinds[m.type]!;
    this.emit(FX.puff, at.x, at.y, 0.6, kind.trim.getHex(), { strength: 1 });
    this.number(at.x, at.y, reward, '+', GOLDEN);
    if (reward >= 1000) {
      pool.burst({ x: at.x, y: 38, z: at.y }, GOLDEN, 0.7, time);
      pool.ripple({ x: at.x, y: 0, z: at.y }, GOLDEN, 0.6, time);
    }
  }

  private transition(was: Snap, now: Snap, pool: CelebrationPool, time: number): void {
    const event = now.words[DW_EVENT]!, hero = this.heroState, at = this.shownHero().clone();
    const kind: Pending['kind'] = now.done ? (event === EV_WIN ? 'win' : 'death') : 'descend';
    this.pending = { snap: now, at: this.clock, kind };
    hero.from.copy(at);
    hero.to.copy(at);
    hero.t = 1;
    hero.kind = 'rest';
    this.iris = { from: 26, to: 0.6, at: this.clock, span: CLOSE, x: at.x, y: at.y };
    this.veil!.tint.value.set(kind === 'death' ? 0x1a0306 : 0x03050c);
    if (kind === 'death') {
      this.emit(FX.tomb, at.x, at.y, CLOSE * 1.6, 0xffffff, { scale: 0.9 });
      pool.sigh(0.8, { x: at.x, y: 0, z: at.y }, time);
    } else if (kind === 'win') {
      const bossAt = was.monsters.findIndex((m) => m.type === this.boss);
      if (bossAt >= 0) this.slay(was, bossAt, pool, time, now.reward);
      pool.burst({ x: at.x, y: 38, z: at.y }, GOLDEN, 1, time);
      pool.ripple({ x: at.x, y: 0, z: at.y }, GOLDEN, 0.9, time);
    } else if (now.reward > 0) pool.burst({ x: at.x, y: 38, z: at.y }, GOLDEN, 0.45, time);
  }

  private swap(pool: CelebrationPool, time: number): void {
    const pending = this.pending;
    if (!pending) return;
    this.pending = null;
    this.install(pending.snap, false);
    const hero = this.heroState;
    this.iris = { from: 0.6, to: 48, at: this.clock, span: OPEN, x: hero.to.x, y: hero.to.y };
    this.ghosts = [];
    this.effects = [];
    if (pending.kind === 'descend' && pending.snap.reward > 0) {
      this.number(hero.to.x, hero.to.y, pending.snap.reward, '+', GOLDEN);
      pool.ripple({ x: hero.to.x, y: 0, z: hero.to.y }, GOLDEN, 0.4, time);
    }
  }

  private shownHero(): THREE.Vector2 {
    const hero = this.heroState;
    return this.other.lerpVectors(hero.from, hero.to, ease(clamp01(hero.t)));
  }

  private actorAt(actor: Actor, out: THREE.Vector2): THREE.Vector2 {
    const k = ease(clamp01(actor.t));
    if (!actor.via) return out.lerpVectors(actor.from, actor.to, k);
    return k < 0.5 ? out.lerpVectors(actor.from, actor.via, k * 2) : out.lerpVectors(actor.via, actor.to, k * 2 - 1);
  }

  private emit(kind: number, x: number, z: number, life: number, tint: number, extra: { dir?: [number, number]; code?: number; spin?: number; strength?: number; scale?: number } = {}): Effect {
    if (this.effects.length >= FX_SLOTS) this.effects.shift();
    const effect: Effect = { kind, x, z, t: 0, life, tint: new THREE.Color(tint), strength: extra.strength ?? 1, code: extra.code ?? 0, spin: extra.spin ?? 0, codes: [-1, -1, -1, -1], count: 0, dx: extra.dir?.[0] ?? 0, dy: extra.dir?.[1] ?? 1, scale: extra.scale ?? 1 };
    this.effects.push(effect);
    return effect;
  }

  private number(x: number, z: number, value: number, sign: '+' | '-', tint: number): void {
    if (value <= 0) return;
    const effect = this.emit(FX.number, x, z, 0.95, tint);
    effect.count = codesOf(value, sign, effect.codes);
  }

  private animate(dt: number): void {
    const hero = this.heroState;
    hero.t = Math.min(1, hero.t + dt / (hero.kind === 'strike' ? STRIKE : HOP));
    hero.flash = Math.max(0, hero.flash - dt / 0.35);
    const turnRate = 1 - Math.exp(-18 * dt);
    hero.heading += Math.atan2(Math.sin(hero.aim - hero.heading), Math.cos(hero.aim - hero.heading)) * turnRate;
    const pending = this.pending;
    if (pending?.kind === 'descend') hero.sink = Math.min(1, hero.sink + dt / CLOSE);
    if (pending?.kind === 'death') hero.fall = Math.min(1, hero.fall + dt / CLOSE);
    for (const actor of this.actors.values()) {
      actor.t = Math.min(1, actor.t + dt / HOP);
      actor.flash = Math.max(0, actor.flash - dt / 0.3);
      actor.spawn = Math.min(1, actor.spawn + dt / FADE);
      actor.heading += Math.atan2(Math.sin(actor.aim - actor.heading), Math.cos(actor.aim - actor.heading)) * turnRate;
    }
    for (const ghost of this.ghosts) ghost.t += dt / DEATH;
    this.ghosts = this.ghosts.filter((g) => g.t < 1);
    for (const effect of this.effects) effect.t += dt / effect.life;
    this.effects = this.effects.filter((e) => e.t < 1);
    this.cursor += (this.shown ? this.shown.words[DW_SLOT]! - this.cursor : 0) * (1 - Math.exp(-20 * dt));
  }

  private draw(): void {
    const snap = this.shown, board = this.board;
    if (!snap || !board) return;
    this.drawSight(snap);
    this.drawItems(snap);
    this.drawMonsters(snap);
    this.drawHero(snap);
    this.drawEffects();
    this.drawPanels(snap);
  }

  private placeSconces(): void {
    const board = this.board!, sconces = this.sconces!, lamps = this.lamps!;
    sconces.begin();
    let lit = 0;
    for (const room of board.rooms) {
      const mx = Math.floor((room.x0 + room.x1) / 2), my = Math.floor((room.y0 + room.y1) / 2);
      const spots: [number, number, number, number][] = [[mx, room.y0 - 1, 0, 1], [mx, room.y1 + 1, 0, -1], [room.x0 - 1, my, 1, 0], [room.x1 + 1, my, -1, 0]];
      for (const [sx, sy, ix, iy] of spots) {
        let x = sx, y = sy;
        if (board.at(x, y) !== WALL) { x += iy !== 0 ? 1 : 0; y += ix !== 0 ? 1 : 0; }
        if (board.at(x, y) !== WALL) continue;
        world(x, y, this.scratch);
        const i = sconces.put(this.scratch.x + ix * 0.28, this.scratch.y + iy * 0.28, 1, headingOf(ix, iy));
        sconces.set('aSconce', i, room.lit ? 1 : 0, (x * 7 + y * 3) % 11 / 11);
        if (room.lit && lit < 48) {
          this.lampMatrix.makeTranslation(this.scratch.x + ix * 0.36, 0.01, this.scratch.y + iy * 0.36);
          lamps.setMatrixAt(lit, this.lampMatrix);
          lamps.setColorAt(lit, this.tone.setRGB(1, 0.62, 0.3).multiplyScalar(16));
          lit++;
        }
      }
    }
    sconces.end();
    lamps.count = lit;
    lamps.instanceMatrix.needsUpdate = true;
    if (lamps.instanceColor) lamps.instanceColor.needsUpdate = true;
  }

  private sightRects(snap: Snap, out: [THREE.Vector4, THREE.Vector4]): void {
    const x = snap.words[DW_X]!, y = snap.words[DW_Y]!, r = this.radius;
    if (snap.words[DW_BLIND]! > 0) {
      out[0].set(x - 1, y, x + 2, y + 1);
      out[1].set(x, y - 1, x + 1, y + 2);
      return;
    }
    out[0].set(x - r, y - r, x + r + 1, y + r + 1);
    const room = this.board!.bright[y * SIDE + x] ? this.board!.roomAt(x, y) : null;
    if (room) out[1].set(room.x0 - 1, room.y0 - 1, room.x1 + 2, room.y1 + 2);
    else out[1].copy(EMPTY);
  }

  private readonly sightNow: [THREE.Vector4, THREE.Vector4] = [new THREE.Vector4(), new THREE.Vector4()];
  private readonly sightKey = { step: -1, key: -1 };

  private drawSight(snap: Snap): void {
    const veil = this.veil!;
    if (this.sightKey.step !== snap.step || this.sightKey.key !== snap.key) {
      const same = this.sightKey.key === snap.key;
      this.sightRects(snap, this.sightNow);
      for (let k = 0; k < 2; k++) {
        veil.was[k]!.value.copy(same ? veil.now[k]!.value : this.sightNow[k]!);
        veil.now[k]!.value.copy(this.sightNow[k]!);
      }
      this.sightAt = this.clock;
      this.sightKey.step = snap.step;
      this.sightKey.key = snap.key;
    }
    veil.blend.value = ease(clamp01((this.clock - this.sightAt) / HOP));
    const iris = this.iris, k = clamp01((this.clock - iris.at) / iris.span);
    const radius = iris.from + (iris.to - iris.from) * (iris.to < iris.from ? ease(k) : k * k * (0.35 + 0.65 * k));
    veil.iris.value.set(iris.x + HALF, iris.y + HALF, radius, iris.at > 0 && (k < 1 || iris.to < iris.from) ? 1 : 0);
    if (k >= 1 && iris.to > iris.from) veil.tint.value.set(0x03050c);
  }

  private drawItems(snap: Snap): void {
    const items = this.items!;
    items.begin();
    for (const item of snap.items) {
      world(item.x, item.y, this.scratch);
      const i = items.put(this.scratch.x, this.scratch.y, ITEM_SCALE);
      const kind = item.kind === GOLD ? 0 : item.kind === RATION ? 1 : 2;
      const tint = item.kind >= POTION ? POTION_COLOURS[(item.kind - POTION) & 7]! : WHITE;
      items.set('aItem', i, kind, ((item.x * 5 + item.y * 11) % 13) / 13, 1);
      items.set('aTint', i, tint.r, tint.g, tint.b);
    }
    items.end();
  }

  private drawMonsters(snap: Snap): void {
    for (const batch of this.monsters.values()) batch.begin();
    const embers = this.embers!;
    let glow = 0;
    const put = (type: number, x: number, z: number, heading: number, awake: number, hp: number, flash: number, phase: number, alpha: number, flag: number, scale: number) => {
      const kind = this.kinds[type]!, batch = this.monsters.get(kind.plan)!;
      const i = batch.put(x, z, scale, heading);
      batch.set('aSkin', i, kind.skin.r, kind.skin.g, kind.skin.b, kind.damage);
      batch.set('aTrim', i, kind.trim.r, kind.trim.g, kind.trim.b, heading);
      batch.set('aForm', i, kind.size, kind.horns, kind.armour, kind.damage);
      batch.set('aState', i, awake, hp, flash, phase);
      batch.set('aMark', i, type === this.boss && !(flag & GUARD) ? 1 : 0, flag & GUARD ? 1 : 0, alpha, awake > 0.5 ? 0 : 1);
      if (kind.fire && glow < 24 && alpha > 0.05) {
        this.lampMatrix.makeTranslation(x, 0.01, z);
        embers.setMatrixAt(glow, this.lampMatrix);
        embers.setColorAt(glow, this.tone.setRGB(1, 0.45, 0.15).multiplyScalar((awake ? 7 : 3) * alpha));
        glow++;
      }
    };
    snap.monsters.forEach((m, j) => {
      const actor = this.actors.get(snap.ids[j]!);
      if (!actor) return;
      const at = this.actorAt(actor, this.scratch);
      const hop = actor.t < 1 ? Math.sin(Math.PI * actor.t) * 0.06 : 0;
      put(m.type, at.x, at.y, actor.heading, m.flag & AWAKE ? 1 : 0, m.hp / this.kinds[m.type]!.hp, actor.flash, actor.phase, ease(actor.spawn), m.flag, 1 + hop);
    });
    for (const ghost of this.ghosts) {
      const k = ease(clamp01(ghost.t));
      put(ghost.type, ghost.at.x, ghost.at.y, ghost.heading + k * 1.4, 1, 0.001, 1 - k, ghost.phase, 1 - k, ghost.flag, 1 - 0.5 * k);
    }
    for (const batch of this.monsters.values()) batch.end();
    embers.count = glow;
    embers.instanceMatrix.needsUpdate = true;
    if (embers.instanceColor) embers.instanceColor.needsUpdate = true;
  }

  private drawHero(snap: Snap): void {
    const hero = this.heroState, batch = this.hero!, words = snap.words;
    const at = this.shownHero();
    let x = at.x, z = at.y, thrust = 0;
    const k = clamp01(hero.t), wave = Math.sin(Math.PI * k);
    if (hero.kind === 'bump') { x += hero.dx * 0.2 * wave; z += hero.dy * 0.2 * wave; }
    if (hero.kind === 'strike') { x += hero.dx * 0.26 * wave; z += hero.dy * 0.26 * wave; thrust = wave; }
    const scale = HERO_SCALE * (1 - 0.6 * hero.sink) * (1 - 0.25 * hero.fall) * (hero.kind === 'move' ? 1 + 0.05 * wave : 1);
    const heading = hero.heading + hero.fall * 1.3;
    batch.begin();
    const i = batch.put(x, z, scale, heading);
    const ward = (words[DW_FIRE]! > 0 ? 1 : 0) + (words[DW_COLD]! > 0 ? 2 : 0);
    batch.set('aPose', i, hero.kind === 'move' ? k : 0, thrust, hero.flash, 1 - 0.9 * Math.max(hero.sink, hero.fall * 0.7));
    batch.set('aStatus', i, words[DW_POISON]! > 0 ? 1 : 0, words[DW_CONFUSED]! > 0 ? 1 : 0, words[DW_BLIND]! > 0 ? 1 : 0, ward);
    batch.set('aVital', i, Math.max(0, words[DW_HP]! | 0) / most(words[DW_LEVEL]!), words[DW_FOOD]! === 0 ? 1 : 0, heading);
    batch.end();
    const torch = this.torch!, c = Math.cos(heading), s = Math.sin(heading);
    const tx = TORCH.x * scale, ty = TORCH.y * scale;
    torch.position.set(x + (tx * c - ty * s), 0.01, z + (-tx * s - ty * c));
    const flicker = 0.85 + 0.1 * Math.sin(this.clock * 23) + 0.05 * Math.sin(this.clock * 37 + 1.3);
    this.torchPower.value = 11 * flicker * (1 - hero.sink) * (1 - hero.fall);
  }

  private drawEffects(): void {
    const fx = this.fx!;
    fx.begin();
    for (const e of this.effects) {
      const i = fx.put(e.x, e.z, e.scale);
      fx.set('aFx', i, e.kind, e.t, e.code, e.spin);
      fx.set('aTint', i, e.tint.r, e.tint.g, e.tint.b, e.strength);
      fx.set('aText', i, e.codes[0]!, e.codes[1]!, e.codes[2]!, e.codes[3]!);
      fx.set('aMeta', i, e.count, 1, e.dx, e.dy);
    }
    fx.end();
  }

  private drawPanels(snap: Snap): void {
    const words = snap.words, slots = this.slots!, tiers = this.tiers!, bars = this.bars!, badges = this.badges!;
    const pitch = SLOT[1] + 0.42, top = -(9 * pitch) / 2 - 0.5;
    slots.begin();
    for (let slot = 0; slot < 10; slot++) {
      const i = slots.put(RIGHT_X, top + slot * pitch);
      const count = slot === 0 ? (terrainAt(snap.view, words[DW_X]!, words[DW_Y]!) === STAIRS ? 1 : 0) : slot === 1 ? words[DW_RATIONS]! : potions(words, slot - 2);
      const colour = slot >= 2 ? POTION_COLOURS[slot - 2]! : WHITE;
      const selected = Math.max(0, 1 - Math.abs(this.cursor - slot) * 1.6);
      slots.set('aSlot', i, slot, count, selected, slot === 0 ? count : 0);
      slots.set('aTint', i, colour.r, colour.g, colour.b, slot >= 2 ? effectOf(words, slot - 2) : 0);
      const n = slot === 0 || count === 0 ? 0 : codesOf(count, '', this.codes);
      slots.set('aText', i, this.codes[0]!, this.codes[1]!, this.codes[2]!, this.codes[3]!);
      slots.set('aMeta', i, n, slot >= 2 ? 1 : 0);
    }
    slots.end();
    const depth = words[DW_DEPTH]!, deepest = words[DW_DEEPEST]!;
    const shaftTop = PANEL_TOP + 7.2, room = HALF - 1 + MAP_EDGE - 1.4 - shaftTop, step = Math.min(TIER[1] + 0.5, room / this.floors);
    tiers.begin();
    for (let d = 1; d <= this.floors; d++) {
      const i = tiers.put(LEFT_X, shaftTop + (d - 0.5) * step, Math.min(1, step / (TIER[1] + 0.3)));
      const state = d === depth ? 2 : d <= deepest ? 1 : 0;
      const mark = d === this.floors ? 2 : d === this.guardian ? 1 : 0;
      tiers.set('aTier', i, d, state, mark, (d - 1) * 0.12);
      codesOf(d, '', this.codes);
      tiers.set('aText', i, this.codes[0]!, -1, -1, -1);
    }
    tiers.end();
    bars.begin();
    const hp = Math.max(0, words[DW_HP]! | 0), level = words[DW_LEVEL]!, food = words[DW_FOOD]!, xp = words[DW_XP]!;
    const floorXp = level > 1 ? 10 << (level - 2) : 0, nextXp = 10 << (level - 1);
    const rows: [number, number, number, number, number, number][] = [
      [ICON.heart, hp / most(level), HURT, hp, 0, hp * 4 < most(level) ? 1 : 0],
      [ICON.loaf, food / 2000, 0xd29b58, food, 0.5, food === 0 ? 1 : 0],
      [ICON.star, clamp01((xp - floorXp) / Math.max(1, nextXp - floorXp)), GOLDEN, level, 0, 0],
    ];
    rows.forEach(([code, fill, tint, value, tick, alert], r) => {
      const i = bars.put(LEFT_X, PANEL_TOP + r * (BAR[1] + 0.3));
      const colour = this.tone.set(tint);
      bars.set('aHud', i, 0, fill, code, alert);
      bars.set('aTint', i, colour.r, colour.g, colour.b);
      const n = codesOf(value, '', this.codes);
      bars.set('aText', i, this.codes[0]!, this.codes[1]!, this.codes[2]!, this.codes[3]!);
      bars.set('aMeta', i, n, tick, 0, this.clock);
    });
    bars.end();
    badges.begin();
    const status: [number, number, number, number][] = [
      [ICON.skull, words[DW_POISON]!, 20, 0x9ee04e],
      [ICON.spiral, words[DW_CONFUSED]!, 12, 0xffe27a],
      [ICON.blind, words[DW_BLIND]!, 40, 0xb9a6d9],
      [ICON.flame, words[DW_FIRE]!, 300, 0xff9a3c],
      [ICON.snow, words[DW_COLD]!, 300, 0x9fe2ff],
    ];
    status.forEach(([code, left, full, tint], k) => {
      const i = badges.put(LEFT_X + (k - 2) * 1.62, PANEL_TOP + 3 * (BAR[1] + 0.3) + 0.35);
      const colour = this.tone.set(tint);
      badges.set('aHud', i, 1, left / full, code, left > 0 ? 1 : 0);
      badges.set('aTint', i, colour.r, colour.g, colour.b);
    });
    badges.end();
  }

  private floorLabel(): string {
    const w = this.shown?.words;
    return w ? `${w[DW_DEPTH]} of ${this.floors} · deepest ${w[DW_DEEPEST]}` : '—';
  }

  private heroLabel(): string {
    const w = this.shown?.words;
    return w ? `HP ${w[DW_HP]! | 0}/${most(w[DW_LEVEL]!)} · level ${w[DW_LEVEL]} · gold ${formatCount(w[DW_GOLD]!)}` : '—';
  }

  private foodLabel(): string {
    const w = this.shown?.words;
    if (!w) return '—';
    const rations = w[DW_RATIONS]!;
    return `${formatCount(w[DW_FOOD]!)} · ${rations} ration${rations === 1 ? '' : 's'}`;
  }

  private potionLabel(words: Uint32Array, colour: number): string {
    return `${POTION_NAMES[colour]} (${EFFECTS[effectOf(words, colour)]})`;
  }

  private slotLabel(): string {
    const snap = this.shown;
    if (!snap) return '—';
    const w = snap.words, slot = w[DW_SLOT]!;
    if (slot === 0) return terrainAt(snap.view, w[DW_X]!, w[DW_Y]!) === STAIRS ? 'stairs · use descends' : 'rest';
    if (slot === 1) return `ration ×${w[DW_RATIONS]}`;
    return `${this.potionLabel(w, slot - 2)} ×${potions(w, slot - 2)}`;
  }

  private actionLabel(): string {
    const snap = this.shown;
    if (!snap || snap.action < 0) return '—';
    const a = snap.action;
    return a < 4 ? COMPASS[a]! : a === ACT_USE ? 'use slot' : a === ACT_NEXT ? 'next slot' : `action ${a}`;
  }

  private eventLabel(): string {
    const snap = this.shown;
    if (!snap) return '—';
    const w = snap.words, e = w[DW_EVENT]!, m = w[DW_MAGNITUDE]!;
    if (e === EV_NONE) return snap.action >= 0 && snap.action < 4 ? 'moved' : '—';
    if (e === EV_HIT) return `struck for ${m}`;
    if (e === EV_MISS) return 'missed';
    if (e === EV_KILL) return `slew a foe · ${m} xp`;
    if (e === EV_HURT) return `wounded for ${m}`;
    if (e === EV_GOLD) return `+${m} gold`;
    if (e === EV_RATION) return 'found a ration';
    if (e === EV_POTION) return `found a ${POTION_NAMES[(m - 1) & 7]} potion`;
    if (e === EV_EAT) return `ate · +${m} food`;
    if (e >= EV_QUAFF && e < EV_QUAFF + 8) return `quaffed ${EFFECTS[e - EV_QUAFF]}${m ? ` · ${m}` : ''}`;
    if (e === EV_DESCEND) return `descended to ${m}`;
    if (e === EV_REST) return 'rested';
    if (e === EV_BUMP) return 'bumped a wall';
    if (e === EV_STARVE) return 'starving';
    if (e === EV_SICK) return 'poison';
    if (e === EV_RUN) return snap.step === 0 ? 'entered the dungeon' : 'died · new run';
    if (e === EV_WIN) return 'won · new run';
    return `event ${e}`;
  }

  private statusLabel(): string {
    const w = this.shown?.words;
    if (!w) return '—';
    const parts: string[] = [];
    if (w[DW_POISON]) parts.push(`poisoned ${w[DW_POISON]}`);
    if (w[DW_CONFUSED]) parts.push(`confused ${w[DW_CONFUSED]}`);
    if (w[DW_BLIND]) parts.push(`blind ${w[DW_BLIND]}`);
    if (w[DW_FIRE]) parts.push(`fire ward ${w[DW_FIRE]}`);
    if (w[DW_COLD]) parts.push(`frost ward ${w[DW_COLD]}`);
    if (w[DW_FOOD] === 0) parts.push('starving');
    return parts.length ? parts.join(' · ') : 'well';
  }

  private groundTruth(kit: DebugKit): void {
    const snap = this.shown, board = this.board;
    if (!snap || !board) return;
    const y = 0.1;
    board.rooms.forEach((room, r) => {
      const curve = kit.curve(`room${r}`, 5, room.lit ? GOLDEN : 0x8fa6c4);
      const x0 = room.x0 - 1 - HALF, x1 = room.x1 + 2 - HALF, z0 = room.y0 - 1 - HALF, z1 = room.y1 + 2 - HALF;
      for (const [px, pz] of [[x0, z0], [x1, z0], [x1, z1], [x0, z1], [x0, z0]] as const) curve.point(px, y, pz);
    });
    for (let r = board.rooms.length; r < 9; r++) kit.curve(`room${r}`, 5, 0x8fa6c4);
    const ways = kit.marks('ways', SIDE * SIDE);
    for (let cy = 0; cy < SIDE; cy++) for (let cx = 0; cx < SIDE; cx++) {
      const t = board.at(cx, cy);
      if (t !== CORRIDOR && t !== DOOR) continue;
      world(cx, cy, this.scratch);
      ways.set(this.scratch.x, y, this.scratch.y, t === DOOR ? 0.16 : 0.07, t === DOOR ? 0xc08850 : 0x8a8fa0);
    }
    const foes = kit.marks('monsters', 32);
    snap.monsters.forEach((m) => {
      world(m.x, m.y, this.scratch);
      foes.set(this.scratch.x, y, this.scratch.y, 0.1 + 0.2 * m.hp / this.kinds[m.type]!.hp, m.flag & GUARD ? 0x9fb4ff : m.flag & AWAKE ? 0xff6a5a : 0x5e7cc4);
    });
    const loot = kit.marks('items', 16);
    for (const item of snap.items) {
      world(item.x, item.y, this.scratch);
      loot.set(this.scratch.x, y, this.scratch.y, 0.14, item.kind === GOLD ? GOLDEN : item.kind === RATION ? 0xd29b58 : POTION_TINTS[(item.kind - POTION) & 7]!);
    }
    const rects: [THREE.Vector4, THREE.Vector4] = [new THREE.Vector4(), new THREE.Vector4()];
    this.sightRects(snap, rects);
    rects.forEach((r, k) => {
      const curve = kit.curve(`sight${k}`, 5, TOKENS.uiCyan);
      if (r.x < -50) return;
      for (const [px, pz] of [[r.x, r.y], [r.z, r.y], [r.z, r.w], [r.x, r.w], [r.x, r.y]] as const) curve.point(px - HALF, y + 0.01, pz - HALF);
    });
    const hx = snap.words[DW_X]!, hy = snap.words[DW_Y]!, patch = kit.marks('patch', SPAN * SPAN);
    for (let i = 0; i < SPAN * SPAN; i++) {
      const glyph = this.glyphs.indexOf(snap.obs[i]!);
      if (glyph < 0 || glyph === GLYPH_UNSEEN) continue;
      world(hx + (i % SPAN) - REACH, hy + Math.floor(i / SPAN) - REACH, this.scratch);
      const tint = glyph >= GLYPH_MONSTER ? 0xff6a5a : glyph > STAIRS ? 0x9ee04e : glyph === STAIRS ? GOLDEN : open(glyph) ? TOKENS.uiCyan : 0x5e6a80;
      patch.set(this.scratch.x, y + 0.02, this.scratch.y, glyph > STAIRS ? 0.14 : 0.06, tint);
    }
  }
}

export default (() => new DungeonScene()) satisfies SceneFactory;
