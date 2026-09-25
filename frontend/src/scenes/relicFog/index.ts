import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  ASTEROID, CENTER, LEAGUE, MATCH_STEPS, MATCHES, MAX_ENERGY, NEBULA, RF_CURSOR_X, RF_CURSOR_Y, RW, SAP, SIDE, TARGET, TILES, UNITS, WAIT,
  driftOf, driftsBy, engaged, fired, focusOf, keyOf, live, makeWorld, pendingOf, readWorld, reborn, relicShape, scoredBy, shifted,
  type Order, type RelicShape, type World,
} from '@sim/envs/relicfog';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { DECK_HEIGHT, DECK_WIDTH, deckMaterial, type Deck } from './deck';
import { AIM, ARROW, GLYPH, HOLD, LABEL, POINT, SHIP, TOKEN, labelMaterial, orderMaterial, shipMaterial } from './fleet';
import { BEACON, BEAM_PAD, BEAM_WIDTH, BLAST, BOLT, IMPACT, LINK, NODE, ORDERED, RECALL, RELIC, REVEAL, SPARK, SPLASH, VOID, WARP, beaconMaterial, beamMaterial, commandMaterial, sparkMaterial, type Command } from './fx';
import { glyphs, type Glyphs } from './glyphs';
import { HALF, flat, sectorMaterial, type N, type Sector } from './sector';

const ACCENT = TOKENS.accents.RelicFog!;
const FOE = 0xff8a4c;
const GOLD = TOKENS.celebrationGold;
const KEY_NAMES = ['hold', '↑', '→', '↓', '←', 'sap'] as const;
const RIVALS = ['scout bot', 'sapper bot', 'league'] as const;
const GAP = 1.0;
const GLIDE = 6;
const BOLT_STEPS = 10;
const IMPACT_STEPS = 13;
const BLAST_STEPS = 15;
const WARP_STEPS = 12;
const RECALL_STEPS = 12;
const SCORE_STEPS = 16;
const REVEAL_STEPS = 20;
const VOID_STEPS = 14;
const POP_STEPS = 0.6;
const VERDICT_SECONDS = 2.2;
const WORLDS = 24;
const DEPTH = 10;
const FLASHES = 10;
const CROWD = 5;
const CHEERS = 6;
const SHIP_LIGHT = 2.4;
const NODE_LIGHT = 7;
const RELIC_LIGHT = 3.2;

const signed = (value: number) => formatSigned(value).replace('-', '−');
const headingOf = (dx: number, dy: number) => Math.atan2(dx, -dy);
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

interface Sheet { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[]; used: number }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }
interface Spot { x: number; z: number; scale: number; shown: number }
interface Crowd { count: number; x: number; z: number; scale: number }
interface Layout { spots: Spot[][]; crowds: Map<number, Crowd>[] }
interface Glow { x: number; z: number; team: number; level: number }
interface Tick { at: number; world: World; before: World | null; fresh: boolean; cleared: boolean; drift: boolean; ticks: number }

class RelicFogScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 24;
  protected readonly stageName = 'relicFog';

  private shape: RelicShape | null = null;
  private readonly grid: GridSize = { w: SIDE, h: SIDE };
  private set: Glyphs | null = null;
  private sector: Sector | null = null;
  private deck: Deck | null = null;
  private command: Command | null = null;
  private ships: Sheet | null = null;
  private orders: Sheet | null = null;
  private labels: Sheet | null = null;
  private beams: Sheet | null = null;
  private sparks: Sheet | null = null;
  private beacons: Sheet | null = null;
  private shade: THREE.Mesh | null = null;
  private shadeKey = -1;
  private readonly lamps: Lamp[] = [];
  private readonly shipLamps: Lamp[] = [];
  private readonly flashLamps: Lamp[] = [];
  private readonly time = uniform(0);
  private readonly worlds = new Map<number, World>();
  private readonly headings = new Map<number, Float32Array>();
  private readonly layouts = new Map<number, Layout>();
  private readonly scored = [new Uint8Array(TILES), new Uint8Array(TILES)];
  private readonly flashes: { x: number; z: number; strength: number; tint: number }[] = [];
  private readonly glows: Glow[] = [];
  private painted = -1;
  private clock = 0;
  private verdictAt = -Infinity;
  private verdictWon = false;
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly mid = new THREE.Vector3();
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly ledger = {
    game: 0, match: 0, step: 0, ours: 0, theirs: 0, ourWins: 0, theirWins: 0, fleet: 0, energy: 0, rivals: 0, member: 0,
    mode: WAIT, cursor: -1, key: -1, fx: 0, fy: 0, clock: 0, reward: 0, pairs: 0, moveCost: 0, sapCost: 0, sapRange: 0, sensorRange: 0,
  };

  override legend: LegendSpec = {
    title: 'RELIC FOG',
    rows: [
      { key: 'match', label: 'match', tint: 0xd8dcff, value: () => `${this.ledger.match + 1} of ${MATCHES} · step ${this.ledger.step} of ${MATCH_STEPS}` },
      { key: 'points', label: 'relic points', tint: ACCENT, value: () => `${formatCount(this.ledger.ours)} to ${formatCount(this.ledger.theirs)}` },
      { key: 'wins', label: 'matches won', tint: GOLD, value: () => `${this.ledger.ourWins} to ${this.ledger.theirWins}` },
      { key: 'fleet', label: 'your fleet', tint: ACCENT, value: () => `${this.ledger.fleet} ship${this.ledger.fleet === 1 ? '' : 's'} · ${formatCount(this.ledger.energy)} energy` },
      { key: 'rival', label: 'rival', tint: FOE, value: () => `${this.rival()} · ${this.ledger.rivals} ship${this.ledger.rivals === 1 ? '' : 's'}` },
      { key: 'command', label: 'command', tint: TOKENS.uiCyan, value: () => this.orderText() },
      { key: 'rules', label: 'this game', tint: 0x7ff5e6, value: () => `move ${this.ledger.moveCost} · sap ${this.ledger.sapCost} within ${this.ledger.sapRange} · sensor ${this.ledger.sensorRange}` },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(this.ledger.reward) },
    ],
    note: () => `game ${formatCount(this.ledger.game + 1)} · ${this.ledger.pairs} relic pair${this.ledger.pairs === 1 ? '' : 's'} surfaced`,
    bar: () => this.ledger.step / MATCH_STEPS,
  };

  private rival(): string {
    const opponent = this.shape?.opponent ?? 0;
    return opponent === LEAGUE ? `league member ${this.ledger.member + 1}` : RIVALS[opponent]!;
  }

  private orderText(): string {
    const { mode, cursor, key, fx, fy, clock } = this.ledger, pressed = key < 0 ? '' : ` · pressed ${KEY_NAMES[key]}`;
    if (mode === WAIT) return `all orders given · key ${clock} of ${this.shape?.window ?? 0}${pressed}`;
    if (mode === TARGET) return `aiming ship ${cursor + 1} at ${fx >= 0 ? '+' : '−'}${Math.abs(fx)}, ${fy >= 0 ? '+' : '−'}${Math.abs(fy)}${pressed}`;
    return `ordering ship ${cursor + 1}${pressed}`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.shape = relicShape(ctx.environment);
    const left = HALF + GAP + DECK_WIDTH;
    return new THREE.Box2(new THREE.Vector2(-left - 0.3, -HALF - 0.45), new THREE.Vector2(HALF + 0.45, HALF + 0.45));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xe4e2ff).multiplyScalar(0.62);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { sighRadius: 3.2, sighHeight: 2 };
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        if (state.pose[RF_CURSOR_X]! < 0) return null;
        agent.x = state.pose[RF_CURSOR_X]!;
        agent.y = state.pose[RF_CURSOR_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const lit = light.lit;
    this.scene.background = new THREE.Color(0x05060d);
    this.set = glyphs();
    this.sector = sectorMaterial(ctx.seed, ACCENT, FOE, lit);
    this.scene.add(this.sector.mesh);
    this.beacons = this.sheet(flat(BEACON, BEACON), beaconMaterial(this.time), 8, ['aBeacon'], 3);
    this.orders = this.sheet(flat(GLYPH, GLYPH), orderMaterial(ACCENT), UNITS, ['aOrder'], 5);
    this.ships = this.sheet(flat(SHIP, SHIP), shipMaterial(ACCENT, FOE, lit), 4 * UNITS, ['aShip', 'aState'], 6);
    this.beams = this.sheet(flat(1, 1), beamMaterial(ACCENT, FOE, this.time), 7 * UNITS + 1, ['aBeam'], 7);
    this.sparks = this.sheet(flat(SPARK, SPARK), sparkMaterial(ACCENT, FOE), 24 * UNITS, ['aSpark'], 8);
    this.labels = this.sheet(flat(LABEL, LABEL), labelMaterial(this.set, ACCENT, FOE), 6 * UNITS, ['aLabel'], 10);
    this.command = commandMaterial(ACCENT, flat);
    this.scene.add(this.command.mesh);
    this.deck = deckMaterial(this.set, ACCENT, FOE, new THREE.Vector2(-HALF - GAP - DECK_WIDTH, -DECK_HEIGHT / 2));
    this.scene.add(this.deck.mesh);
    const lamp = (radius: number): Lamp => {
      const tint = new THREE.Color();
      const mesh = light.emitter(new THREE.CircleGeometry(radius, 20).rotateX(-Math.PI / 2), uniform(tint) as unknown as N<'vec3'>);
      mesh.visible = false;
      this.scene.add(mesh);
      return { mesh, tint };
    };
    for (let i = 0; i < 8; i++) this.lamps.push(lamp(i < 2 ? 0.26 : 0.2));
    for (let i = 0; i < 2 * UNITS; i++) this.shipLamps.push(lamp(0.12));
    for (let i = 0; i < FLASHES; i++) this.flashLamps.push(lamp(0.22));
  }

  private sheet(geometry: THREE.BufferGeometry, material: THREE.Material, count: number, names: readonly string[], order: number): Sheet {
    const attributes = names.map((name) => {
      const attribute = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
      attribute.name = name;
      geometry.setAttribute(name, attribute);
      return attribute;
    });
    const mesh = instancedMesh(geometry, material, count);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    return { mesh, attributes, used: 0 };
  }

  private put(sheet: Sheet, x: number, z: number, scale: number, ...values: readonly (readonly number[])[]): void {
    if (sheet.used >= sheet.attributes[0]!.count) throw new Error(`RelicFog sheet ${sheet.attributes.map((a) => a.name).join()} holds ${sheet.attributes[0]!.count} marks`);
    const i = sheet.used++;
    sheet.mesh.setMatrixAt(i, this.matrix.compose(this.at.set(x, 0, z), this.quaternion.identity(), this.scale.setScalar(scale)));
    values.forEach((row, k) => sheet.attributes[k]!.setXYZW(i, row[0]!, row[1]!, row[2]!, row[3]!));
  }

  private segment(sheet: Sheet, ax: number, az: number, bx: number, bz: number, style: number, team: number, progress: number): void {
    const dx = bx - ax, dz = bz - az, span = Math.hypot(dx, dz);
    if (span < 1e-3) return;
    if (sheet.used >= sheet.attributes[0]!.count) throw new Error(`RelicFog beams hold ${sheet.attributes[0]!.count} marks`);
    const i = sheet.used++;
    this.quaternion.setFromAxisAngle(this.axis, Math.atan2(-dz, dx));
    sheet.mesh.setMatrixAt(i, this.matrix.compose(this.mid.set((ax + bx) / 2, 0, (az + bz) / 2), this.quaternion, this.scale.set(span + 2 * BEAM_PAD, 1, BEAM_WIDTH)));
    sheet.attributes[0]!.setXYZW(i, span, style, team, progress);
  }

  private commit(sheet: Sheet): void {
    sheet.mesh.count = sheet.used;
    sheet.mesh.instanceMatrix.needsUpdate = true;
    for (const attribute of sheet.attributes) attribute.needsUpdate = true;
    sheet.used = 0;
  }

  protected override drop(): void {
    this.sector?.dispose();
    this.deck?.dispose();
    this.set?.texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || object === this.sector?.mesh || object === this.deck?.mesh) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.worlds.clear();
    this.headings.clear();
    this.layouts.clear();
  }

  private worldAt(frame: FrameUpdate, s: number, turn: EnvFrame | null = null): World | null {
    const cached = this.worlds.get(s);
    if (cached) return cached;
    if (s < 0) return null;
    const source = frame.frameAt(s), view = source.step === s ? source.view : turn?.view;
    if (!view) return null;
    if (this.worlds.size >= WORLDS) this.worlds.delete(this.worlds.keys().next().value!);
    const world = readWorld(view, this.shape!, makeWorld());
    this.worlds.set(s, world);
    return world;
  }

  private tickOf(frame: FrameUpdate, step: number): Tick | null {
    const source = frame.frameAt(step);
    if (source.step !== step) return null;
    const at = step - source.words[RW.CLOCK]!, world = this.worldAt(frame, at, source)!;
    const ticks = source.words[RW.STEP]!, fresh = ticks === 0, cleared = !fresh && source.words[RW.MATCH] === 1;
    const before = at > 0 ? this.worldAt(frame, at - 1) : null;
    let drift = false;
    if (before && !fresh) {
      const [dx, dy] = driftOf(world);
      drift = driftsBy(world, ticks) !== driftsBy(world, ticks - 1);
      if (!shifted(before, world, drift ? dx : 0, drift ? dy : 0)) throw new Error(`RelicFog step ${at} ${drift ? 'should roll its nebulae and asteroids' : 'moves its nebulae and asteroids off the drift schedule'}`);
    }
    return { at, world, before, fresh, cleared, drift, ticks };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    this.time.value = this.clock;
    pool.update(frame.dtReal, time);
    const step = frame.state.step, current = frame.frameAt(step);
    if (current.step !== step || !this.shape) return;
    const tick = this.tickOf(frame, step);
    if (!tick) return;
    const age = Math.max(0, frame.position - tick.at);
    this.flashes.length = 0;
    this.paint(tick);
    this.drift(tick, age);
    this.fleet(frame, tick, age);
    this.volleys(tick, age);
    this.voids(tick, age);
    this.beaconsAt(tick, age);
    this.steer(frame, tick, current);
    this.panel(frame, tick, current);
    this.illuminate(tick, age);
    this.celebrate(frame, tick, pool, time);
    for (const sheet of [this.ships, this.orders, this.labels, this.beams, this.sparks, this.beacons]) if (sheet) this.commit(sheet);
    const sector = this.sector!;
    sector.age.value = age;
    sector.time.value = this.clock;
    this.command!.time.value = this.clock;
    this.deck!.time.value = this.clock;
  }

  private paint(tick: Tick): void {
    if (tick.at === this.painted) return;
    this.painted = tick.at;
    const sector = this.sector!, now = sector.now.image.data as Float32Array, was = sector.was.image.data as Float32Array, marks = sector.marks.image.data as Float32Array;
    const world = tick.world, before = tick.before ?? world;
    scoredBy(world, 0, this.scored[0]!);
    scoredBy(world, 1, this.scored[1]!);
    for (let i = 0; i < TILES; i++) {
      now.set([world.kind[i]!, world.energy[i]!, world.weight[i]!, world.sight[0][i]! + 2 * world.sight[1][i]!], 4 * i);
      was.set([before.kind[i]!, before.energy[i]!, before.weight[i]!, before.sight[0][i]! + 2 * before.sight[1][i]!], 4 * i);
      const shown = live(world, i);
      marks.set([this.scored[0]![i]!, this.scored[1]![i]!, shown ? 1 : 0, shown && tick.before && !tick.fresh && !live(tick.before, i) ? 1 : 0], 4 * i);
    }
    sector.now.needsUpdate = sector.was.needsUpdate = sector.marks.needsUpdate = true;
    sector.fresh.value = tick.fresh && tick.before ? 1 : 0;
    const [dx, dy] = driftOf(world), drifts = driftsBy(world, tick.ticks);
    sector.identity.value.set(drifts * dx, drifts * dy);
    this.occlude(world, tick.at);
  }

  private occlude(world: World, at: number): void {
    if (this.shadeKey === at) return;
    this.shadeKey = at;
    const positions: number[] = [], index: number[] = [], r = 0.34;
    for (let i = 0; i < TILES; i++) {
      if (world.kind[i] !== ASTEROID) continue;
      const base = positions.length / 3, { x, z } = this.centre(i % SIDE, Math.floor(i / SIDE), this.at);
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
  }

  private drift(tick: Tick, age: number): void {
    const slide = this.sector!.slide.value, [dx, dy] = driftOf(tick.world), rest = tick.drift ? 1 - smoothstep01(clamp01(age / GLIDE)) : 0;
    slide.set(dx * rest, dy * rest);
    this.shade?.position.set(-slide.x, 0, -slide.y);
  }

  private moveHeading(order: Order): number | null {
    if (order.type < 1 || order.type > 4) return null;
    const [dx, dy] = this.shape!.steps[order.type]!;
    return headingOf(dx, dy);
  }

  private headingsAt(frame: FrameUpdate, turn: EnvFrame, depth: number): Float32Array | null {
    const at = turn.step - turn.words[RW.CLOCK]!, cached = this.headings.get(at);
    if (cached) return cached;
    const world = this.worldAt(frame, at, turn);
    if (!world) return null;
    const previous = at > 0 ? frame.frameAt(at - 1) : null, known = previous !== null && previous.step === at - 1;
    const before = known ? this.worldAt(frame, at - 1) : null;
    const sameGame = before !== null && turn.words[RW.STEP] !== 0, cleared = sameGame && previous!.words[RW.MATCH] === 0;
    const prior = sameGame && depth > 0 ? this.headingsAt(frame, previous!, depth - 1) : null;
    const out = new Float32Array(2 * UNITS);
    for (let team = 0; team < 2; team++) for (let i = 0; i < UNITS; i++) {
      const unit = world.units[team]![i]!, fallback = team === 0 ? headingOf(1, 1) : headingOf(-1, -1), k = team * UNITS + i;
      const spawned = !sameGame || cleared || !before!.units[team]![i]!.alive || reborn(before!, world, team, i, this.shape!.steps);
      const move = this.moveHeading(unit);
      if (spawned) out[k] = fallback;
      else if (move !== null) out[k] = move;
      else if (unit.type === SAP && (unit.dx !== 0 || unit.dy !== 0)) out[k] = headingOf(unit.dx, unit.dy);
      else out[k] = prior?.[k] ?? fallback;
    }
    if (this.headings.size > 4 * WORLDS) this.headings.delete(this.headings.keys().next().value!);
    this.headings.set(at, out);
    return out;
  }

  private layoutOf(world: World, at: number): Layout {
    const cached = this.layouts.get(at);
    if (cached) return cached;
    const spots: Spot[][] = [[], []], crowds = [new Map<number, Crowd>(), new Map<number, Crowd>()];
    for (let team = 0; team < 2; team++) {
      const groups = new Map<number, number[]>();
      world.units[team]!.forEach((unit, i) => {
        if (!unit.alive) return;
        const tile = unit.y * SIDE + unit.x, members = groups.get(tile);
        if (members) members.push(i);
        else groups.set(tile, [i]);
      });
      spots[team] = world.units[team]!.map(() => ({ x: 0, z: 0, scale: 1, shown: 0 }));
      for (const [tile, members] of groups) {
        const count = members.length, shared = world.units[1 - team]!.some((unit) => unit.alive && unit.y * SIDE + unit.x === tile);
        const nudge = shared ? (team === 0 ? -0.17 : 0.17) : 0, squeeze = shared ? 0.66 : 1;
        if (count >= CROWD) crowds[team]!.set(tile, { count, x: nudge, z: nudge, scale: squeeze });
        members.forEach((i, j) => {
          let x = 0, z = 0, scale = 0.92, shown = 1;
          if (count === 2) [x, z, scale] = [j ? 0.19 : -0.19, j ? 0.19 : -0.19, 0.64];
          else if (count === 3) [x, z, scale] = [[0, -0.21, 0.21][j]!, [-0.2, 0.15, 0.15][j]!, 0.56];
          else if (count === 4) [x, z, scale] = [j % 2 ? 0.2 : -0.2, j < 2 ? -0.2 : 0.2, 0.5];
          else if (count >= CROWD) [scale, shown] = [0.7, 0];
          spots[team]![i] = { x: x * squeeze + nudge, z: z * squeeze + nudge, scale: scale * squeeze, shown };
        });
      }
    }
    const layout = { spots, crowds };
    if (this.layouts.size > 4 * WORLDS) this.layouts.delete(this.layouts.keys().next().value!);
    this.layouts.set(at, layout);
    return layout;
  }

  private fleet(frame: FrameUpdate, tick: Tick, age: number): void {
    const sparks = this.sparks!, world = tick.world, before = tick.before, recalled = tick.fresh || tick.cleared, steps = this.shape!.steps;
    const glide = smoothstep01(clamp01(age / GLIDE)), bend = smoothstep01(clamp01(age / (GLIDE * 0.8)));
    const heads = this.headingsAt(frame, frame.frameAt(frame.state.step), DEPTH)!, layout = this.layoutOf(world, tick.at);
    const previous = frame.frameAt(tick.at - 1), priorHeads = before && previous.step === tick.at - 1 ? this.headingsAt(frame, previous, DEPTH) : null;
    const prior = before ? this.layoutOf(before, tick.at - 1) : null;
    this.glows.length = 0;
    for (let team = 0; team < 2; team++) for (let i = 0; i < UNITS; i++) {
      const unit = world.units[team]![i]!, was = before?.units[team]![i], k = team * UNITS + i, old = priorHeads?.[k] ?? heads[k]!;
      const from = prior?.spots[team]![i], renewed = !!before && !recalled && reborn(before, world, team, i, steps);
      if (was?.alive && from && (recalled || !unit.alive || renewed)) {
        this.centre(was.x, was.y, this.at);
        const ax = this.at.x + from.x, az = this.at.z + from.z;
        if (recalled || (renewed && was.energy >= 0)) {
          const t = age / RECALL_STEPS, fade = 1 - clamp01(age / (RECALL_STEPS * 0.5));
          if (t < 1 && from.shown && recalled) this.put(sparks, ax, az, 0.5, [RECALL, t, team, 0]);
          if (fade > 0 && from.shown) this.drawShip(team, ax, az, from.scale, was.energy, old, fade, before!, was.x, was.y);
        } else {
          const end = renewed ? was : unit;
          this.centre(end.x, end.y, this.other);
          const bx = this.other.x, bz = this.other.z, fade = 1 - clamp01((age - GLIDE * 0.7) / 3), t = (age - GLIDE * 0.7) / BLAST_STEPS;
          if (fade > 0 && from.shown) this.drawShip(team, lerp(ax, bx, glide), lerp(az, bz, glide), from.scale, was.energy, turn(old, this.moveHeading(unit) ?? old, bend), fade, world, end.x, end.y);
          if (t >= 0 && t < 1) {
            this.put(sparks, bx, bz, 0.8, [BLAST, t, team, (i * 0.618) % 1]);
            this.flashes.push({ x: bx, z: bz, strength: 1 - t, tint: team === 0 ? ACCENT : FOE });
          }
        }
      }
      if (!unit.alive) continue;
      const to = layout.spots[team]![i]!, stays = !recalled && !!was?.alive && !!from && !renewed;
      this.centre(unit.x, unit.y, this.other);
      let x = this.other.x + to.x, z = this.other.z + to.z, scale = to.scale, heading = heads[k]!, alpha = to.shown;
      if (stays) {
        this.centre(was!.x, was!.y, this.at);
        const sx = this.at.x + from!.x, sz = this.at.z + from!.z;
        x = lerp(sx, x, glide);
        z = lerp(sz, z, glide);
        scale = lerp(from!.scale, scale, glide);
        heading = turn(old, heading, bend);
        alpha = lerp(from!.shown, to.shown, glide);
      } else {
        const t = age / WARP_STEPS;
        if (t < 1) this.put(sparks, this.other.x, this.other.z, 0.6, [WARP, t, team, 0]);
        alpha *= smoothstep01(clamp01((age - WARP_STEPS * 0.45) / (WARP_STEPS * 0.4)));
      }
      if (alpha < 0.002) continue;
      this.drawShip(team, x, z, scale, unit.energy, heading, alpha, world, unit.x, unit.y);
      if (unit.energy > 0 && (team === 0 || world.sight[0][unit.y * SIDE + unit.x])) this.glows.push({ x, z, team, level: alpha * (0.35 + 0.65 * Math.min(1, unit.energy / 200)) });
    }
    this.tokens(world, layout, prior, recalled, age, glide);
    this.scores(tick, age);
  }

  private tokens(world: World, layout: Layout, prior: Layout | null, recalled: boolean, age: number, glide: number): void {
    const fade = recalled ? 1 - clamp01(age / (RECALL_STEPS * 0.5)) : 1 - glide;
    const same = (a: Crowd | undefined, b: Crowd) => !!a && a.count === b.count && a.x === b.x && a.scale === b.scale;
    for (let team = 0; team < 2; team++) {
      const now = layout.crowds[team]!, was = prior?.crowds[team];
      for (const [tile, crowd] of now) this.token(world, team, tile, crowd, !recalled && same(was?.get(tile), crowd) ? 1 : glide);
      if (was) for (const [tile, crowd] of was) if (recalled || !same(now.get(tile), crowd)) this.token(world, team, tile, crowd, fade);
    }
  }

  private token(world: World, team: number, tile: number, crowd: Crowd, alpha: number): void {
    if (alpha <= 0.002) return;
    const hidden = team === 1 && world.sight[0][tile] === 0;
    this.centre(tile % SIDE, Math.floor(tile / SIDE), this.at);
    const x = this.at.x + crowd.x, z = this.at.z + crowd.z;
    this.put(this.labels!, x, z, crowd.scale, [crowd.count, team, alpha * (hidden ? 0.6 : 1), TOKEN]);
    if (!hidden) this.glows.push({ x, z, team, level: alpha });
  }

  private drawShip(team: number, x: number, z: number, scale: number, energy: number, heading: number, alpha: number, world: World, cx: number, cy: number): void {
    const ghost = team === 1 && world.sight[0][cy * SIDE + cx] === 0 ? 1 : 0;
    this.put(this.ships!, x, z, scale, [team, Math.max(0, energy) / MAX_ENERGY, heading, 0], [alpha, ghost, energy < world.params.moveCost ? 1 : 0, energy < 0 ? 1 : 0]);
  }

  private scores(tick: Tick, age: number): void {
    if (tick.fresh || !tick.before) return;
    const t = (age - GLIDE) / SCORE_STEPS;
    if (t < 0 || t >= 1) return;
    for (let team = 0; team < 2; team++) {
      const tiles = this.scored[team]!;
      for (let i = 0; i < TILES; i++) {
        if (!tiles[i]) continue;
        this.centre(i % SIDE, Math.floor(i / SIDE), this.at);
        const hidden = team === 1 && tick.world.sight[0][i] === 0;
        this.put(this.labels!, this.at.x, this.at.z - 0.38 - 0.6 * smoothstep01(t), 0.85, [1, team, (1 - t * t) * (hidden ? 0.4 : 1), POINT]);
      }
    }
  }

  private volleys(tick: Tick, age: number): void {
    if (!tick.before || tick.fresh) return;
    const { dropoff } = tick.before.params;
    for (let team = 0; team < 2; team++) for (let i = 0; i < UNITS; i++) {
      if (!fired(tick.before, tick.world, team, i)) continue;
      const unit = tick.world.units[team]![i]!, was = tick.before.units[team]![i]!, tx = was.x + unit.dx, ty = was.y + unit.dy;
      this.centre(was.x, was.y, this.at);
      this.centre(tx, ty, this.other);
      const shot = age / BOLT_STEPS;
      if (shot < 1) this.segment(this.beams!, this.at.x, this.at.z, this.other.x, this.other.z, BOLT, team, shot);
      const hit = (age - 1.5) / IMPACT_STEPS;
      if (hit < 0 || hit >= 1) continue;
      this.put(this.sparks!, this.other.x, this.other.z, 1, [IMPACT, hit, team, 0]);
      this.flashes.push({ x: this.other.x, z: this.other.z, strength: 1 - hit, tint: team === 0 ? ACCENT : FOE });
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if ((dx === 0 && dy === 0) || tx + dx < 0 || ty + dy < 0 || tx + dx >= SIDE || ty + dy >= SIDE) continue;
        this.put(this.sparks!, this.other.x + dx, this.other.z + dy, 1, [SPLASH, hit, team, dropoff / 4]);
      }
    }
  }

  private voids(tick: Tick, age: number): void {
    const t = (age - GLIDE) / VOID_STEPS, before = tick.before, world = tick.world, steps = this.shape!.steps;
    if (!before || tick.fresh || tick.cleared || t < 0 || t >= 1) return;
    const strength = Math.sin(Math.PI * Math.sqrt(t)), drawn = new Set<number>();
    const resolved = (team: number, i: number) => engaged(before, team, i) && !reborn(before, world, team, i, steps);
    for (let i = 0; i < UNITS; i++) {
      if (!resolved(0, i)) continue;
      const ours = world.units[0][i]!;
      for (let j = 0; j < UNITS; j++) {
        const theirs = world.units[1][j]!, key = (ours.y * SIDE + ours.x) * TILES + theirs.y * SIDE + theirs.x;
        if (!resolved(1, j) || Math.abs(ours.x - theirs.x) + Math.abs(ours.y - theirs.y) !== 1 || drawn.has(key)) continue;
        drawn.add(key);
        this.centre(ours.x, ours.y, this.at);
        this.centre(theirs.x, theirs.y, this.other);
        this.segment(this.beams!, this.at.x, this.at.z, this.other.x, this.other.z, VOID, 0, strength);
      }
    }
  }

  private beaconsAt(tick: Tick, age: number): void {
    const beacons = this.beacons!, world = tick.world, before = tick.fresh ? null : tick.before;
    const glide = smoothstep01(clamp01(age / GLIDE));
    const place = (x: number, y: number, from: { x: number; y: number } | null, kind: number, grown: number, heat: number, phase: number) => {
      this.centre(x, y, this.other);
      if (from) {
        this.centre(from.x, from.y, this.at);
        this.other.lerpVectors(this.at, this.other, glide);
      }
      this.put(beacons, this.other.x, this.other.z, 1, [kind, grown, heat, phase]);
    };
    const mirror = (node: { x: number; y: number }) => ({ x: SIDE - 1 - node.y, y: SIDE - 1 - node.x });
    const wasNode = before ? before.node : null;
    place(world.node.x, world.node.y, wasNode, NODE, 1, 0, 0);
    const m = mirror(world.node);
    place(m.x, m.y, wasNode ? mirror(wasNode) : null, NODE, 1, 0, 2.1);
    world.relics.forEach((relic, r) => {
      if (!relic.revealed) return;
      const fresh = !!before && !before.relics[r]!.revealed, grown = fresh ? clamp01(age / REVEAL_STEPS) : 1;
      if (fresh && age < REVEAL_STEPS) {
        this.centre(relic.x, relic.y, this.at);
        this.put(this.sparks!, this.at.x, this.at.z, 1, [REVEAL, age / REVEAL_STEPS, 2, r * 0.3]);
      }
      const heat = this.relicHeat(world, r % 3) * Math.exp(-Math.max(0, age - GLIDE) / 8) * (age >= GLIDE ? 1 : 0);
      place(relic.x, relic.y, null, RELIC, grown, heat, r * 1.3);
    });
  }

  private relicHeat(world: World, pair: number): number {
    for (let i = 0; i < TILES; i++) if (world.weight[i] === pair + 1 && (this.scored[0]![i] || this.scored[1]![i])) return 1;
    return 0;
  }

  private spotOf(source: EnvFrame, world: World, aim: boolean): THREE.Vector2 | null {
    const mode = source.words[RW.MODE]!, cursor = source.words[RW.CURSOR]!;
    if (mode === WAIT || cursor >= UNITS) return null;
    const unit = world.units[0][cursor]!, focus = aim && mode === TARGET ? focusOf(source.words) : { dx: 0, dy: 0 };
    this.centre(unit.x + focus.dx, unit.y + focus.dy, this.other);
    return new THREE.Vector2(this.other.x, this.other.z);
  }

  private steer(frame: FrameUpdate, tick: Tick, current: EnvFrame): void {
    const command = this.command!, orders = this.orders!, world = tick.world, step = current.step, u = clamp01(frame.position - step), k = smoothstep01(clamp01(u / POP_STEPS));
    const next = frame.frameAt(step + 1), ahead = next.step === step + 1 && next.done === 0 ? next : null;
    const aheadWorld = ahead ? (ahead.words[RW.CLOCK] === 0 ? this.worldAt(frame, step + 1) : world) : null;
    const here = this.spotOf(current, world, false), there = ahead && aheadWorld ? this.spotOf(ahead, aheadWorld, false) : null;
    const mode = current.words[RW.MODE]!;
    command.mode.value = mode === WAIT && ahead ? ahead.words[RW.MODE]! : mode;
    command.shown.value = here ? (there ? 1 : 1 - k) : there ? k : 0;
    const spot = here && there ? here.clone().lerp(there, k) : here ?? there;
    if (spot) command.cursor.value.copy(spot);
    if (mode === TARGET && here) {
      const cursor = current.words[RW.CURSOR]!, unit = world.units[0][cursor]!, reach = world.params.sapRange;
      this.centre(Math.max(unit.x - reach, 0), Math.max(unit.y - reach, 0), this.at);
      this.centre(Math.min(unit.x + reach, SIDE - 1), Math.min(unit.y + reach, SIDE - 1), this.other);
      command.fence.value.set(this.at.x - 0.5, this.at.z - 0.5, this.other.x + 0.5, this.other.z + 0.5);
      const aim = this.spotOf(current, world, true)!, later = ahead && ahead.words[RW.MODE] === TARGET && ahead.words[RW.CLOCK]! > current.words[RW.CLOCK]! ? this.spotOf(ahead, world, true)! : aim;
      aim.lerp(later, k);
      command.aim.value.copy(aim);
      this.segment(this.beams!, here.x, here.y, aim.x, aim.y, LINK, 0, 1);
    }
    const full = this.rosterOf(world, current.words), roster = current.words[RW.ROSTER]!, sameTurn = ahead !== null && ahead.words[RW.CLOCK]! > current.words[RW.CLOCK]!;
    const coming = sameTurn ? roster & ~ahead!.words[RW.ROSTER]! : 0, spots = this.layoutOf(world, tick.at).spots[0]!;
    for (let i = 0; i < UNITS; i++) {
      const given = ((full >> i) & 1) === 1 && ((roster >> i) & 1) === 0, fresh = ((coming >> i) & 1) === 1;
      if (!given && !fresh) continue;
      const unit = world.units[0][i]!, order = pendingOf((fresh ? ahead! : current).view, i), place = spots[i]!, alpha = fresh ? k : 1;
      this.centre(unit.x, unit.y, this.at);
      if (order.type === SAP) {
        this.centre(unit.x + order.dx, unit.y + order.dy, this.other);
        this.segment(this.beams!, this.at.x, this.at.z, this.other.x, this.other.z, ORDERED, 0, alpha);
      }
      if (!place.shown) continue;
      const x = this.at.x + place.x, z = this.at.z + place.z, size = Math.max(0.55, place.scale) * (fresh ? 1 + 0.35 * (1 - alpha) : 1);
      if (order.type === SAP) this.put(orders, x, z, size, [AIM, 0, 0, alpha]);
      else if (order.type === CENTER) this.put(orders, x, z, size, [HOLD, 0, 0, alpha]);
      else {
        const [dx, dy] = this.shape!.steps[order.type]!;
        this.put(orders, x + dx * 0.34 * size, z + dy * 0.34 * size, size, [ARROW, dx, -dy, alpha]);
      }
    }
  }

  private rosterOf(world: World, words: Uint32Array): number {
    let roster = 0;
    if (words[RW.MATCH] !== 0) world.units[0].forEach((unit, i) => { if (unit.alive && unit.energy >= 0) roster |= 1 << i; });
    return roster;
  }

  private panel(frame: FrameUpdate, tick: Tick, current: EnvFrame): void {
    const deck = this.deck!, world = tick.world, words = current.words, rows = deck.rows.image.data as Float32Array, roster = words[RW.ROSTER]!, full = this.rosterOf(world, words);
    const mode = words[RW.MODE]!, step = current.step;
    for (let i = 0; i < UNITS; i++) {
      const unit = world.units[0][i]!, ordered = ((full >> i) & 1) === 1 && ((roster >> i) & 1) === 0, order = ordered ? pendingOf(current.view, i) : { type: 0, dx: 0, dy: 0 };
      const [dx, dy] = order.type >= 1 && order.type <= 4 ? this.shape!.steps[order.type]! : [0, 0];
      rows.set([(unit.alive ? 1 : 0) + (ordered ? 2 : 0) + (unit.energy < world.params.moveCost ? 4 : 0), unit.energy, order.type, dx + 8 + 16 * (dy + 8)], 4 * i);
    }
    deck.rows.needsUpdate = true;
    deck.ours.value = words[RW.OUR_POINTS]!;
    deck.theirs.value = words[RW.THEIR_POINTS]!;
    deck.ourWins.value = words[RW.OUR_WINS]!;
    deck.theirWins.value = words[RW.THEIR_WINS]!;
    deck.clock.value = words[RW.MATCH]!;
    deck.cursor.value = mode === WAIT ? -1 : words[RW.CURSOR]!;
    deck.mode.value = mode;
    const key = step > 0 && !current.done ? keyOf(current) : -1;
    deck.key.value = key;
    deck.wait.value = mode === WAIT ? Math.min(1, Math.max(1e-3, words[RW.CLOCK]! / this.shape!.window)) : 0;
    const since = this.clock - this.verdictAt;
    deck.verdict.value = since < VERDICT_SECONDS ? (this.verdictWon ? 1 : -1) * (1 - since / VERDICT_SECONDS) : 0;
    const ledger = this.ledger, params = world.params;
    ledger.game = words[RW.GAME]!;
    ledger.match = Math.floor(words[RW.STEP]! / (MATCH_STEPS + 1));
    ledger.step = words[RW.MATCH]!;
    ledger.ours = words[RW.OUR_POINTS]!;
    ledger.theirs = words[RW.THEIR_POINTS]!;
    ledger.ourWins = words[RW.OUR_WINS]!;
    ledger.theirWins = words[RW.THEIR_WINS]!;
    ledger.member = words[RW.MEMBER]!;
    ledger.mode = mode;
    ledger.cursor = words[RW.CURSOR]!;
    ledger.key = key;
    ledger.clock = words[RW.CLOCK]!;
    ledger.reward = current.reward;
    ledger.pairs = world.pairs;
    ledger.moveCost = params.moveCost;
    ledger.sapCost = params.sapCost;
    ledger.sapRange = params.sapRange;
    ledger.sensorRange = params.sensorRange;
    const focus = focusOf(words);
    ledger.fx = focus.dx;
    ledger.fy = focus.dy;
    ledger.fleet = world.units[0].filter((unit) => unit.alive).length;
    ledger.energy = world.units[0].reduce((sum, unit) => sum + (unit.alive ? Math.max(0, unit.energy) : 0), 0);
    ledger.rivals = world.units[1].filter((unit) => unit.alive).length;
  }

  private illuminate(tick: Tick, age: number): void {
    const world = tick.world, glide = smoothstep01(clamp01(age / GLIDE)), before = tick.fresh ? null : tick.before;
    const light = (lamp: Lamp, x: number, z: number, color: number, level: number) => {
      lamp.mesh.position.set(x, 0.02, z);
      lamp.tint.set(color).multiplyScalar(level);
      lamp.mesh.visible = level > 0.01;
    };
    const nodes = [[world.node, before?.node], [{ x: SIDE - 1 - world.node.y, y: SIDE - 1 - world.node.x }, before ? { x: SIDE - 1 - before.node.y, y: SIDE - 1 - before.node.x } : undefined]] as const;
    nodes.forEach(([node, from], n) => {
      this.centre(node.x, node.y, this.other);
      if (from) this.other.lerpVectors(this.centre(from.x, from.y, this.at), this.other, glide);
      light(this.lamps[n]!, this.other.x, this.other.z, 0x7ff5e6, NODE_LIGHT);
    });
    world.relics.forEach((relic, r) => {
      this.centre(relic.x, relic.y, this.other);
      light(this.lamps[2 + r]!, this.other.x, this.other.z, 0xffcf70, relic.revealed ? RELIC_LIGHT : 0);
    });
    this.shipLamps.forEach((lamp, i) => {
      const glow = this.glows[i];
      if (glow) light(lamp, glow.x, glow.z, glow.team === 0 ? ACCENT : FOE, SHIP_LIGHT * glow.level);
      else lamp.mesh.visible = false;
    });
    this.flashLamps.forEach((lamp, i) => {
      const flash = this.flashes[i];
      if (flash) light(lamp, flash.x, flash.z, flash.tint, 9 * flash.strength);
      else lamp.mesh.visible = false;
    });
  }

  private celebrate(frame: FrameUpdate, tick: Tick, pool: CelebrationPool, time: number): void {
    for (const event of frame.events) {
      if (event.type === 'goal' || event.type === 'harm') {
        this.verdictAt = this.clock;
        this.verdictWon = event.type === 'goal';
        if (event.type === 'harm') pool.sigh(0.8, this.at.set(0, 2, 0), time);
        else {
          const fleet = (tick.fresh ? tick.before : tick.world)?.units[0].filter((unit) => unit.alive && unit.energy >= 0).sort((a, b) => b.energy - a.energy) ?? [];
          for (const unit of fleet.slice(0, CHEERS)) pool.burst(this.centre(unit.x, unit.y, this.at).setY(38), GOLD, 0.5, time);
        }
      }
      if (event.type === 'cue') for (const relic of [tick.world.relics[event.value]!, tick.world.relics[event.value + 3]!]) pool.burst(this.centre(relic.x, relic.y, this.at).setY(38), GOLD, 0.4, time);
    }
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const tick = this.tickOf(frame, frame.state.step);
    if (!tick) return;
    const world = tick.world, current = frame.frameAt(frame.state.step);
    const field = kit.marks('field', TILES), terrain = kit.marks('terrain', TILES), fog = kit.marks('fog', TILES), runes = kit.marks('scoring', TILES);
    for (let i = 0; i < TILES; i++) {
      this.centre(i % SIDE, Math.floor(i / SIDE), this.at);
      const energy = world.energy[i]!;
      if (energy !== 0) field.set(this.at.x - 0.28, 0.02, this.at.z + 0.28, 0.03 + 0.012 * Math.abs(energy), energy > 0 ? 0x3fe0cf : 0xd0406a);
      if (world.kind[i] === ASTEROID) terrain.set(this.at.x, 0.05, this.at.z, 0.2, 0x9a8f80);
      if (world.kind[i] === NEBULA) terrain.set(this.at.x, 0.05, this.at.z, 0.12, 0xa070e0);
      if (!world.sight[0][i]) fog.set(this.at.x + 0.32, 0.05, this.at.z - 0.32, 0.07, 0x9aa3d9);
      if (world.weight[i]) runes.set(this.at.x - 0.3, 0.06, this.at.z - 0.3, live(world, i) ? 0.1 : 0.05, live(world, i) ? (this.scored[0]![i] ? ACCENT : this.scored[1]![i] ? FOE : GOLD) : 0x6b5a2a);
    }
    const units = kit.marks('units', 2 * UNITS), beacons = kit.marks('beacons', 8);
    for (let team = 0; team < 2; team++) for (const unit of world.units[team]!) {
      if (!unit.alive) continue;
      this.centre(unit.x, unit.y, this.at);
      units.set(this.at.x, 0.1, this.at.z, 0.08 + 0.16 * Math.min(1, Math.max(0, unit.energy) / MAX_ENERGY), unit.energy < 0 ? 0xff3b3b : team === 0 ? ACCENT : FOE);
    }
    this.centre(world.node.x, world.node.y, this.at);
    beacons.set(this.at.x, 0.1, this.at.z, 0.3, 0x7ff5e6);
    this.centre(SIDE - 1 - world.node.y, SIDE - 1 - world.node.x, this.at);
    beacons.set(this.at.x, 0.1, this.at.z, 0.3, 0x7ff5e6);
    for (const relic of world.relics) {
      this.centre(relic.x, relic.y, this.at);
      beacons.set(this.at.x, 0.1, this.at.z, relic.revealed ? 0.28 : 0.12, relic.revealed ? GOLD : 0x6b5a2a);
    }
    const mode = current.words[RW.MODE]!, cursor = current.words[RW.CURSOR]!;
    const range = kit.curve('range', 5, ACCENT), aim = kit.curve('aim', 2, 0xffffff);
    if (mode === TARGET && cursor < UNITS) {
      const unit = world.units[0][cursor]!, reach = world.params.sapRange, focus = focusOf(current.words);
      const x0 = Math.max(unit.x - reach, 0) - 0.5, x1 = Math.min(unit.x + reach, SIDE - 1) + 0.5, y0 = Math.max(unit.y - reach, 0) - 0.5, y1 = Math.min(unit.y + reach, SIDE - 1) + 0.5;
      for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] as const) {
        this.centre(x, y, this.at);
        range.point(this.at.x, 0.08, this.at.z);
      }
      this.centre(unit.x, unit.y, this.at);
      aim.point(this.at.x, 0.08, this.at.z);
      aim.point(this.at.x + focus.dx, 0.08, this.at.z + focus.dy);
    }
    const pending = kit.marks('orders', UNITS), full = this.rosterOf(world, current.words), roster = current.words[RW.ROSTER]!;
    for (let i = 0; i < UNITS; i++) {
      if (((full >> i) & 1) === 0 || ((roster >> i) & 1) === 1) continue;
      const unit = world.units[0][i]!, order = pendingOf(current.view, i), [dx, dy] = order.type >= 1 && order.type <= 4 ? this.shape!.steps[order.type]! : [0, 0];
      this.centre(unit.x, unit.y, this.at);
      pending.set(this.at.x + dx * 0.4, 0.12, this.at.z + dy * 0.4, 0.07, order.type === SAP ? 0xffffff : order.type === CENTER ? 0x8a90b8 : TOKENS.uiCyan);
    }
    let shot = 0;
    if (tick.before && !tick.fresh) for (let team = 0; team < 2; team++) for (let i = 0; i < UNITS; i++) {
      if (!fired(tick.before, world, team, i)) continue;
      const unit = world.units[team]![i]!, was = tick.before.units[team]![i]!, line = kit.curve(`sap${shot++}`, 2, team === 0 ? ACCENT : FOE);
      this.centre(was.x, was.y, this.at);
      line.point(this.at.x, 0.09, this.at.z);
      line.point(this.at.x + unit.dx, 0.09, this.at.z + unit.dy);
    }
  }
}

export default (() => new RelicFogScene()) satisfies SceneFactory;
