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
  ATTACK, BARRACKS, BASE, DOCTRINE, HARVEST, HEAVY, IDLE, KILLED, LIGHT, LOST, MODE_KEY, MOVE, MRTS_CURSOR, MW, NEAREST_AT, NEUTRAL, NOBODY, NONE, OFF_MAP_GLYPH, ORE_GLYPH, PATCH_AT,
  PRODUCE, PRODUCED, RANGED, REACH, REJECTED, RESOURCE, RETURN, RETURNED, SPAN, STEP_X, STEP_Y, SWEEP_AT, UNSEEN_GLYPH, WORKER, aimOf, etaOf, keyOf, makeOf, microShape, timeOf, type MicroShape,
} from '@sim/envs/microrts';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { ACCENT, FOE, GOLD, MINT, RIM, TEAMS, TOKEN, TRAY_WIDTH, TROOP_QUAD, boardMaterial, treasuryMaterial, type Treasury, flat, fogMaterial, mineralMaterial, structureMaterial, troopMaterial, type Board, type Fog, type Level, type N } from './art';
import { Chronicle, type Fetch, type Piece, type Turn } from './chronicle';
import {
  AWAITING, BEAM_PAD, BEAM_WIDTH, CHIP, COIN, CURSOR, DECAL, RESERVED, WRECK_HALL, WRECK_UNIT, ICON_BASE, ICON_BARRACKS, ICON_HEAVY, ICON_LIGHT, ICON_NONE, ICON_RANGED, ICON_TARGET, ICON_WORKER, IMPACT, ORDER, PENDING, REJECT,
  SHATTER, SKIPPED, SPARK, SPAWN, WAIT, beamMaterial, cursorMaterial, decalMaterial, orderMaterial, sparkMaterial, type Cursor,
} from './fx';

const KIND_NAMES = ['resource', 'base', 'barracks', 'worker', 'light', 'heavy', 'ranged'] as const;
const PLURALS = ['resources', 'bases', 'barracks', 'workers', 'lights', 'heavies', 'ranged'] as const;
const SCRIPTS = ['random bot', 'worker rush', 'light rush', 'heavy rush', 'ranged rush', 'league doctrine'] as const;
const SENSED: Readonly<Record<number, string>> = { [LOST]: 'unit lost', [KILLED]: 'enemy felled', [PRODUCED]: 'unit trained', [REJECTED]: 'order refused', [RETURNED]: 'resources returned' };
const RECALL = 900;
const KEY_NAMES = ['skip', '↑', '→', '↓', '←', 'mode'] as const;
const ECHO = 10;
const WRECK = 9;
const IMPACT_LIFE = 0.7;
const COIN_LIFE = 1.4;
const CHIP_LIFE = 0.9;
const SPAWN_LIFE = 1.3;
const REFUSE_LIFE = 2.2;
const SHATTER_LIFE = 1.1;
const PRESS_LIFE = 0.4;
const HOLD = 0.85;
const DUSK = 0.55;
const DAWN = 1.5;
const BASE_LIGHT = 1.5;
const ORE_LIGHT = 1.1;
const FLASH_LIGHT = 7;
const LAMPS = 8;
const FLASHES = 8;

const ease = (t: number) => smoothstep01(t);
const signed = (value: number) => formatSigned(value).replace('-', '−');

interface Sheet { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[]; used: number }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color; level: Level }
interface Flash { x: number; z: number; strength: number; tint: number }
interface Plan { x: number; y: number; key: number }

class MicroRtsScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 30;
  protected readonly stageName = 'microRTS';

  private shape: MicroShape | null = null;
  private grid: GridSize = { w: 8, h: 8 };
  private chronicle: Chronicle | null = null;
  private board: Board | null = null;
  private fog: Fog | null = null;
  private cursor: Cursor | null = null;
  private cursorMesh: THREE.Mesh | null = null;
  private minerals: Sheet | null = null;
  private halls: Sheet | null = null;
  private troops: Sheet | null = null;
  private beams: Sheet | null = null;
  private sparks: Sheet | null = null;
  private decals: Sheet | null = null;
  private orders: Sheet | null = null;
  private readonly lamps: Lamp[] = [];
  private readonly flashLamps: Lamp[] = [];
  private cursorLamp: Lamp | null = null;
  private readonly treasuries: Treasury[] = [];
  private textures: THREE.Texture[] = [];
  private readonly flashes: Flash[] = [];
  private readonly plan: Plan[] = [];
  private readonly refused = new Map<number, Set<number>>();

  private clock = 0;
  private lastTurn = -1;
  private fogTurn = -1;
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly cursorAt = new THREE.Vector3();
  private readonly ledger = {
    game: 0, time: 0, tau: 0, side: 0, foe: 0, member: 0, gold: [0, 0], reward: 0, sensed: 0, sensedAgo: 0, sensedAt: -1,
    mine: new Int32Array(7), theirs: new Int32Array(7), selected: -1, mode: 0, key: -1,
  };

  override legend: LegendSpec = {
    title: 'MICRORTS',
    rows: [
      { key: 'you', label: 'you', tint: ACCENT, value: () => `${formatCount(this.ledger.gold[this.ledger.side])} resources · ${this.forces(this.ledger.mine)}` },
      { key: 'foe', label: 'foe', tint: FOE, value: () => `${this.foeName()} · ${formatCount(this.ledger.gold[1 - this.ledger.side])} resources · ${this.forces(this.ledger.theirs)}` },
      { key: 'command', label: 'command', tint: TOKENS.uiCyan, value: () => this.command() },
      { key: 'sensed', label: 'sensed', tint: 0xc9d3e6, value: () => this.sensed() },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(this.ledger.reward) },
    ],
    note: () => `game ${formatCount(this.ledger.game + 1)} · cycle ${formatCount(this.ledger.time)} of ${formatCount(this.shape?.maxCycles ?? 0)}`,
    bar: () => this.ledger.tau / Math.max(1, this.shape?.maxCycles ?? 1),
  };

  private forces(counts: Int32Array): string {
    const parts: string[] = [];
    for (const kind of [BASE, BARRACKS, WORKER, LIGHT, HEAVY, RANGED]) if (counts[kind]! > 0) parts.push(`${counts[kind]} ${counts[kind] === 1 ? KIND_NAMES[kind] : PLURALS[kind]}`);
    return parts.length ? parts.join(' · ') : 'routed';
  }

  private sensed(): string {
    const { sensed, sensedAgo } = this.ledger;
    if (sensed === 0) return '—';
    return `${SENSED[sensed] ?? '—'} · ${sensedAgo === 0 ? 'now' : `${formatCount(sensedAgo)} steps ago`}`;
  }

  private foeName(): string {
    const { foe, member } = this.ledger;
    return foe === DOCTRINE ? `${SCRIPTS[DOCTRINE]} ${member + 1}` : SCRIPTS[foe] ?? '—';
  }

  private command(): string {
    const { selected, mode, key } = this.ledger;
    const pressed = key < 0 ? '' : ` · ${KEY_NAMES[key]}`;
    if (selected < 0) return `all orders given${pressed}`;
    return `${KIND_NAMES[selected]} · ${this.modeName(selected, mode)}${pressed}`;
  }

  private modeName(kind: number, mode: number): string {
    if (kind === BASE) return 'train worker';
    if (kind === BARRACKS) return `train ${KIND_NAMES[LIGHT + mode]}`;
    if (kind === WORKER) return mode === 1 ? 'build base' : mode === 2 ? 'build barracks' : 'act';
    if (kind === RANGED && mode > 0) return `fire at target ${mode}`;
    return 'act';
  }

  private iconOf(kind: number, mode: number): number {
    if (kind === BASE) return ICON_WORKER;
    if (kind === BARRACKS) return [ICON_LIGHT, ICON_HEAVY, ICON_RANGED][mode]!;
    if (kind === WORKER) return mode === 1 ? ICON_BASE : mode === 2 ? ICON_BARRACKS : ICON_NONE;
    if (kind === RANGED && mode > 0) return ICON_TARGET;
    return ICON_NONE;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const shape = microShape(ctx.environment);
    this.shape = shape;
    this.grid = { w: shape.width, h: shape.width };
    this.chronicle = new Chronicle(shape);
    const half = shape.width / 2 + RIM + 0.25, wide = half + TRAY_WIDTH + 0.2;
    return new THREE.Box2(new THREE.Vector2(-wide, -half), new THREE.Vector2(wide, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xdde4fb).multiplyScalar(0.86);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { sighRadius: 0.06 * this.grid.w, sighHeight: 2 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        const cell = state.pose[MRTS_CURSOR]!;
        agent.x = cell % this.grid.w;
        agent.y = Math.floor(cell / this.grid.w);
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const shape = this.shape!, side = shape.width, lit = light.lit;
    this.scene.background = new THREE.Color(0x06080c);
    const board = boardMaterial(side, ctx.seed, lit);
    this.board = board;
    this.textures.push(...board.textures);
    const ground = new THREE.Mesh(flat(side + 60, side + 60), board.material);
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground);

    const capacity = shape.capacity;
    this.decals = this.sheet(flat(DECAL, DECAL), decalMaterial(), capacity, ['aDecal'], 1);
    this.minerals = this.sheet(flat(TOKEN, TOKEN), mineralMaterial(lit), capacity, ['aMineral'], 2);
    this.halls = this.sheet(flat(TOKEN, TOKEN), structureMaterial(lit), 2 * capacity, ['aHall', 'aHallLife'], 3);
    this.troops = this.sheet(flat(TROOP_QUAD, TROOP_QUAD), troopMaterial(lit), 3 * capacity, ['aTroop', 'aTroopLife', 'aTroopFx'], 4);
    this.beams = this.sheet(flat(1, 1), beamMaterial(), capacity, ['aBeam'], 5);
    this.sparks = this.sheet(flat(SPARK, SPARK), sparkMaterial(), 4 * capacity, ['aSpark'], 6);

    if (shape.partial) {
      const fog = fogMaterial(side, ctx.seed);
      this.fog = fog;
      this.textures.push(fog.now, fog.was, ...fog.textures);
      const mist = new THREE.Mesh(flat(side, side), fog.material);
      mist.renderOrder = 7;
      mist.frustumCulled = false;
      this.scene.add(mist);
    }

    for (const player of [0, 1]) {
      const reach = side / 2 + RIM, height = 0.4 * side + 0.6, treasury = treasuryMaterial(height, player === 0, lit), tray = new THREE.Mesh(flat(TRAY_WIDTH, height), treasury.material);
      tray.position.set((player === 0 ? -1 : 1) * (reach + 0.14 + TRAY_WIDTH / 2), 0, (player === 0 ? -1 : 1) * (reach - height / 2));
      tray.renderOrder = 1;
      tray.frustumCulled = false;
      this.scene.add(tray);
      this.treasuries.push(treasury);
    }

    this.orders = this.sheet(flat(ORDER, ORDER), orderMaterial(), capacity + 8, ['aOrder'], 8);
    const cursor = cursorMaterial();
    this.cursor = cursor;
    this.cursorMesh = new THREE.Mesh(flat(CURSOR, CURSOR), cursor.material);
    this.cursorMesh.renderOrder = 9;
    this.cursorMesh.frustumCulled = false;
    this.scene.add(this.cursorMesh);

    const lamp = (radius: number): Lamp => {
      const tint = new THREE.Color(), level = uniform(0);
      const mesh = light.emitter(new THREE.CircleGeometry(radius, 24).rotateX(-Math.PI / 2), (uniform(tint) as unknown as N<'vec3'>).mul(level));
      mesh.visible = false;
      this.scene.add(mesh);
      return { mesh, tint, level };
    };
    for (let i = 0; i < LAMPS; i++) this.lamps.push(lamp(0.13));
    for (let i = 0; i < FLASHES; i++) this.flashLamps.push(lamp(0.12));
    this.cursorLamp = lamp(0.1);
    this.cursorLamp.tint.set(ACCENT);
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 1 : 10; });
  }

  private sheet(geometry: THREE.BufferGeometry, material: THREE.Material, count: number, names: readonly string[], order: number): Sheet {
    const attributes = names.map((name) => {
      const attribute = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
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

  private put(sheet: Sheet, x: number, z: number, ...values: readonly (readonly number[])[]): void {
    if (sheet.used >= sheet.mesh.instanceMatrix.count) return;
    const i = sheet.used++;
    sheet.mesh.setMatrixAt(i, this.matrix.makeTranslation(x, 0, z));
    values.forEach((row, k) => sheet.attributes[k]!.setXYZW(i, row[0]!, row[1]!, row[2]!, row[3]!));
  }

  private commit(sheet: Sheet): void {
    sheet.mesh.count = sheet.used;
    sheet.mesh.instanceMatrix.needsUpdate = true;
    for (const attribute of sheet.attributes) attribute.needsUpdate = true;
    sheet.used = 0;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.chronicle?.clear();
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private fetcher(frame: FrameUpdate): Fetch {
    return (s) => {
      if (s < 0) return null;
      const f = frame.frameAt(s);
      return f.step === s ? f : null;
    };
  }

  private teamOf(turn: Turn, owner: number): number {
    return owner === NEUTRAL ? 2 : owner === turn.side ? 0 : 1;
  }

  private progress(piece: Piece, tau: number): number {
    const eta = etaOf(piece, this.shape!.types);
    if (!(eta > 1) || Number.isNaN(piece.start)) return -1;
    return clamp01((tau - piece.start - 1) / (eta - 1));
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, time);
    const chronicle = this.chronicle!, fetch = this.fetcher(frame), step = frame.state.step;
    chronicle.clock = this.clock;
    const A = fetch(step);
    if (!A) return;
    const turn = chronicle.at(step, fetch);
    if (!turn) return;
    const B = fetch(step + 1), u = B ? clamp01(frame.position - step) : 0;
    const now = turn.time + (step - turn.step) / turn.length;
    const next = B && B.words[MW.PHASE] === 0 && timeOf(B) > turn.time ? timeOf(B) : now + 1 / turn.length;
    const tau = now + u * (next - now);
    if (frame.scrubbed || this.lastTurn < 0 || turn.step < this.lastTurn) pool.clear();
    else if (turn.step !== this.lastTurn) this.celebrate(turn, pool, time);
    if (frame.events.some((event) => event.type === 'done')) this.finish(frame, turn, fetch, pool, time);
    this.lastTurn = turn.step;

    this.flashes.length = 0;
    const young = this.dawn(turn, fetch, tau);
    const later = B && B.words[MW.PHASE] === 0 && timeOf(B) - turn.time > 1 ? chronicle.at(B.step, fetch) : null;
    this.compose(turn, tau, young, later);
    this.echoes(turn, tau, fetch);
    this.steer(frame, fetch, A, B, u, turn, young);
    this.shade(turn, tau);
    this.illuminate(turn, young);
    this.record(A, turn, tau, fetch);
    this.tally(turn, tau);
    for (const sheet of [this.decals, this.minerals, this.halls, this.troops, this.beams, this.sparks, this.orders]) if (sheet) this.commit(sheet);
    if (this.board) {
      this.board.clock.value = clamp01(tau / this.shape!.maxCycles);
      this.board.time.value = this.clock;
    }
  }

  private dawn(turn: Turn, fetch: Fetch, tau: number): number {
    if (tau >= DAWN || turn.game === 0) return 1;
    let opening: Turn | null = turn;
    while (opening && !opening.opening) opening = this.chronicle!.cached(opening.prior);
    if (!opening || opening.step === 0) return 1;
    const last = this.chronicle!.finale(opening, fetch), ended = fetch(opening.step);
    if (last && ended) this.farewell(last, ended.reward, tau);
    return smoothstep01((tau - DUSK) / (DAWN - DUSK));
  }

  private farewell(last: Turn, reward: number, tau: number): void {
    const fade = 1 - smoothstep01(tau / DUSK);
    for (const piece of last.pieces) {
      if (piece.kind === RESOURCE) continue;
      const mine = piece.owner === last.side, fell = reward > 0 ? !mine : reward < 0 && mine;
      this.centre(piece.x, piece.y, this.at);
      if (fell && tau < DAWN) {
        this.put(this.sparks!, this.at.x, this.at.z, [IMPACT, tau / DAWN, this.teamOf(last, piece.owner), 0]);
        this.flashes.push({ x: this.at.x, z: this.at.z, strength: 1 - tau / DAWN, tint: mine ? TOKENS.failureBlue : GOLD });
      }
      if (fade <= 0) continue;
      this.draw(last, piece, this.at.x, this.at.z, fade, -1, 0, 0);
    }
  }

  private draw(turn: Turn, piece: Piece, x: number, z: number, fade: number, busy: number, hit: number, swing: number): void {
    const type = this.shape!.types[piece.kind]!, team = this.teamOf(turn, piece.owner);
    if (piece.kind === RESOURCE) this.put(this.minerals!, x, z, [piece.carry / this.shape!.richest, fade, hit, piece.cell * 0.37]);
    else if (type.mobile) this.put(this.troops!, x, z, [piece.kind, team, piece.heading, fade], [piece.hp, type.hp, piece.carry, -1], [hit, swing, 0, 0]);
    else this.put(this.halls!, x, z, [piece.kind, team, fade, busy], [piece.hp, type.hp, hit, -1]);
  }

  private compose(turn: Turn, tau: number, young: number, later: Turn | null): void {
    const shape = this.shape!, chronicle = this.chronicle!, prior = chronicle.cached(turn.prior);
    const since = tau - turn.time;
    for (const held of turn.pieces) {
      const piece = later?.orders.find((order) => order.cell === held.cell) ?? held, type = shape.types[piece.kind]!, team = this.teamOf(turn, piece.owner), p = this.progress(piece, tau);
      const before = prior && piece.from >= 0 ? prior.pieces[piece.from]! : null;
      const struck = before && piece.kind !== RESOURCE && before.hp > piece.hp ? 1 - clamp01(since / 0.5) : 0;
      const mined = before && piece.kind === RESOURCE && before.carry > piece.carry ? 1 - clamp01(since / 0.4) : 0;
      this.centre(piece.x, piece.y, this.at);
      let swing = 0;
      if (piece.act === MOVE) {
        this.centre(piece.x + STEP_X[piece.arg & 3]!, piece.y + STEP_Y[piece.arg & 3]!, this.other);
        this.put(this.decals!, this.other.x, this.other.z, [RESERVED, Math.max(p, 0), 0, team]);
        if (p >= 0) this.at.lerp(this.other, ease((p - HOLD) / (1 - HOLD)));
      } else if (piece.act === ATTACK && p >= 0) swing = type.range <= 1 ? Math.sin(Math.PI * clamp01((p - 0.7) / 0.3)) : -0.4 * Math.sin(Math.PI * clamp01((p - 0.45) / 0.2));
      else if (piece.act === HARVEST && p >= 0) swing = 0.55 * Math.abs(Math.sin(p * Math.PI * 5));
      const born = Number.isFinite(piece.born) ? smoothstep01((tau - piece.born) / 0.6) : 1;
      const fade = piece.kind === RESOURCE ? born : born * young;
      this.draw(turn, piece, this.at.x, this.at.z, fade, piece.act === PRODUCE ? p : -1, struck, swing);
      if (mined > 0) this.put(this.sparks!, this.at.x, this.at.z, [CHIP, 1 - mined, 2, 0]);
      if (piece.act === PRODUCE) {
        const make = makeOf(piece), aim = aimOf(piece), made = shape.types[make]!, built = Math.max(p, 0);
        this.centre(aim.x, aim.y, this.other);
        const centre = (shape.width - 1) / 2, facing = this.chronicle!.headingToward(aim.x, aim.y, centre, centre);
        if (made.mobile) this.put(this.troops!, this.other.x, this.other.z, [make, team, facing, young], [made.hp, made.hp, 0, built], [0, 0, 0, 0]);
        else this.put(this.halls!, this.other.x, this.other.z, [make, team, young, -1], [made.hp, made.hp, 0, built]);
      }
      if (piece.act === ATTACK) this.beam(piece, p, team);
      if (piece.act === NONE && !Number.isNaN(piece.start)) this.put(this.sparks!, this.at.x, this.at.z, [WAIT, 0, team, Math.PI * smoothstep01(clamp01((tau - piece.start - 1) % 4 - 3))]);
    }
  }

  private beam(piece: Piece, p: number, team: number): void {
    const aim = aimOf(piece), beams = this.beams!;
    this.centre(piece.x, piece.y, this.at);
    this.centre(aim.x, aim.y, this.other);
    const dx = this.other.x - this.at.x, dz = this.other.z - this.at.z, span = Math.hypot(dx, dz);
    if (beams.used >= beams.mesh.instanceMatrix.count || span === 0) return;
    const i = beams.used++;
    this.quaternion.setFromAxisAngle(this.axis, Math.atan2(-dz, dx));
    beams.mesh.setMatrixAt(i, this.matrix.compose(this.at.lerp(this.other, 0.5), this.quaternion, this.scale.set(span + 2 * BEAM_PAD, 1, BEAM_WIDTH)));
    beams.attributes[0]!.setXYZW(i, span, p, team, this.shape!.types[piece.kind]!.range <= 1 ? 1 : 0);
  }

  private refusals(later: Turn, earlier: Turn, fetch: Fetch): Set<number> {
    const known = this.refused.get(later.step);
    if (known) return known;
    const cells = new Set<number>(), ordered = new Set<number>();
    const single = later.time - earlier.time === 1;
    let complete = true;
    for (let s = earlier.step + 1; s <= later.step; s++) {
      const now = fetch(s), was = fetch(s - 1);
      if (!now || !was) { complete = false; continue; }
      const key = keyOf(now), picked = was.words[MW.SELECTED]!;
      if (picked === NOBODY || key === 0 || key === MODE_KEY || (s < later.step && now.words[MW.SELECTED] === picked)) continue;
      ordered.add(was.words[MW.CURSOR]!);
    }
    for (const piece of later.pieces) {
      if (piece.owner !== later.side) continue;
      if ((piece.act === NONE && piece.start === earlier.time) || (single && ordered.has(piece.cell) && piece.act === IDLE)) cells.add(piece.cell);
    }
    if (complete) {
      this.refused.set(later.step, cells);
      if (this.refused.size > 16) this.refused.delete(this.refused.keys().next().value!);
    }
    return cells;
  }

  private echoes(turn: Turn, tau: number, fetch: Fetch): void {
    const chronicle = this.chronicle!, shape = this.shape!;
    let later: Turn | null = turn;
    for (let n = 0; later && n < ECHO && !later.opening; n++) {
      const earlier = chronicle.cached(later.prior);
      if (!earlier) break;
      const age = tau - later.time, landed = later.time;
      for (const piece of later.fallen) {
        this.centre(piece.x, piece.y, this.at);
        if (piece.kind === RESOURCE) {
          if (age < SHATTER_LIFE) this.put(this.sparks!, this.at.x, this.at.z, [SHATTER, age / SHATTER_LIFE, 2, 0]);
          continue;
        }
        this.put(this.decals!, this.at.x, this.at.z, [shape.types[piece.kind]!.mobile ? WRECK_UNIT : WRECK_HALL, age / WRECK, piece.cell * 1.618, this.teamOf(later, piece.owner)]);
      }
      for (const piece of later.orders.length ? earlier.pieces.concat(later.orders) : earlier.pieces) {
        const end = piece.start + etaOf(piece, shape.types);
        if (piece.act === IDLE || piece.act === NONE || !(end > earlier.time && end <= landed)) continue;
        const aim = aimOf(piece), team = this.teamOf(earlier, piece.owner);
        this.centre(aim.x, aim.y, this.at);
        if (piece.act === ATTACK && age < IMPACT_LIFE) {
          this.put(this.sparks!, this.at.x, this.at.z, [IMPACT, age / IMPACT_LIFE, team, 0]);
          this.flashes.push({ x: this.at.x, z: this.at.z, strength: 1 - age / IMPACT_LIFE, tint: team === 0 ? ACCENT : FOE });
        }
        if (piece.act === RETURN && piece.carry > 0 && age < COIN_LIFE) this.put(this.sparks!, this.at.x, this.at.z, [COIN, age / COIN_LIFE, team, 0]);
        if (piece.act === HARVEST && age < CHIP_LIFE) {
          this.centre(piece.x, piece.y, this.other);
          this.put(this.sparks!, this.other.x, this.other.z, [CHIP, age / CHIP_LIFE, 2, 0]);
        }
      }
      const refused = age < REFUSE_LIFE ? this.refusals(later, earlier, fetch) : null;
      for (const piece of later.pieces) {
        this.centre(piece.x, piece.y, this.at);
        if (piece.born === landed && age < SPAWN_LIFE) this.put(this.sparks!, this.at.x, this.at.z, [SPAWN, age / SPAWN_LIFE, this.teamOf(later, piece.owner), 0]);
        if (refused?.has(piece.cell)) this.put(this.sparks!, this.at.x, this.at.z, [REJECT, age / REFUSE_LIFE, 0, 0]);
      }
      later = earlier;
    }
  }

  private steer(frame: FrameUpdate, fetch: Fetch, A: EnvFrame, B: EnvFrame | null, u: number, turn: Turn, young: number): void {
    const cursor = this.cursor!, mesh = this.cursorMesh!, orders = this.orders!, width = this.grid.w;
    const cell = A.words[MW.CURSOR]!, selected = A.words[MW.SELECTED]!, mode = A.words[MW.MODE]!;
    const unit = selected === NOBODY ? null : turn.pieces.find((piece) => piece.cell === cell && piece.owner === turn.side) ?? null;
    this.ledger.selected = unit ? unit.kind : -1;
    this.ledger.mode = mode;
    this.ledger.key = A.step > turn.step && !A.done ? keyOf(A) : -1;
    this.plan.length = 0;
    mesh.visible = young > 0.5;
    if (!mesh.visible) return;
    this.centre(cell % width, Math.floor(cell / width), this.cursorAt);
    if (B && B.words[MW.PHASE] !== 0 && B.words[MW.CURSOR] !== cell) {
      const next = B.words[MW.CURSOR]!;
      this.centre(next % width, Math.floor(next / width), this.other);
      this.cursorAt.lerp(this.other, ease(clamp01(u / 0.7)));
    }
    mesh.position.set(this.cursorAt.x, 0, this.cursorAt.z);
    cursor.live.value = unit ? 1 : 0;
    cursor.code.value = unit ? this.iconOf(unit.kind, mode) : 0;
    cursor.pips.value = unit && unit.kind === RANGED ? mode : 0;
    cursor.time.value = this.clock;
    cursor.pressed.value = clamp01((frame.position - A.step) * 4);

    const addressed = new Set<number>();
    let refused = 0;
    for (let s = turn.step + 1; s <= A.step; s++) {
      const now = fetch(s), was = fetch(s - 1);
      if (!now || !was) continue;
      const key = keyOf(now), picked = was.words[MW.SELECTED]!, at = was.words[MW.CURSOR]!;
      if (picked === NOBODY) continue;
      if (key === MODE_KEY ? now.words[MW.MODE] === was.words[MW.MODE] : now.words[MW.SELECTED] === picked) {
        refused = Math.max(refused, 1 - (frame.position - s) / (PRESS_LIFE * this.controlHz));
        continue;
      }
      if (key === MODE_KEY) continue;
      const piece = turn.pieces.find((candidate) => candidate.cell === at && candidate.owner === turn.side);
      if (!piece) continue;
      addressed.add(piece.cell);
      this.plan.push({ x: piece.x, y: piece.y, key });
      this.centre(piece.x, piece.y, this.at);
      if (key === 0) this.put(orders, this.at.x, this.at.z, [SKIPPED, 0, 0, 0]);
      else this.put(orders, this.at.x, this.at.z, [PENDING, key - 1, this.iconOf(piece.kind, was.words[MW.MODE]!), this.clock]);
    }
    cursor.refused.value = clamp01(refused);
    for (const piece of turn.pieces) {
      if (piece.owner !== turn.side || piece.act !== IDLE || addressed.has(piece.cell) || (unit && piece.cell === unit.cell)) continue;
      this.centre(piece.x, piece.y, this.at);
      this.put(orders, this.at.x, this.at.z, [AWAITING, 0, 0, this.clock + piece.cell]);
    }
  }

  private shade(turn: Turn, tau: number): void {
    const fog = this.fog;
    if (!fog) return;
    if (turn.step !== this.fogTurn) {
      const prior = this.chronicle!.cached(turn.prior);
      this.paint(fog.now, turn.sight);
      this.paint(fog.was, prior && !turn.opening ? prior.sight : turn.sight);
      this.fogTurn = turn.step;
    }
    fog.blend.value = smoothstep01((tau - turn.time) / 0.45);
    fog.time.value = this.clock;
  }

  private paint(map: THREE.DataTexture, sight: Uint8Array): void {
    const data = map.image.data as Uint8Array;
    for (let i = 0; i < sight.length; i++) data[4 * i] = sight[i]! ? 255 : 0;
    map.needsUpdate = true;
  }

  private illuminate(turn: Turn, young: number): void {
    const shape = this.shape!;
    let used = 0;
    for (const piece of turn.pieces) {
      if (used >= LAMPS) break;
      const lamp = this.lamps[used];
      if (!lamp) break;
      if (piece.kind === BASE) {
        lamp.tint.set(TEAMS[this.teamOf(turn, piece.owner)] ?? MINT);
        lamp.level.value = BASE_LIGHT * young * (piece.act === PRODUCE ? 1.25 + 0.2 * Math.sin(this.clock * 6) : 1);
      } else if (piece.kind === RESOURCE) {
        lamp.tint.set(MINT);
        lamp.level.value = ORE_LIGHT * (0.35 + 0.65 * piece.carry / shape.richest);
      } else continue;
      this.centre(piece.x, piece.y, lamp.mesh.position).setY(0.02);
      if (piece.kind === BASE) lamp.mesh.position.z -= 0.03;
      lamp.mesh.visible = true;
      used++;
    }
    for (let i = used; i < LAMPS; i++) this.lamps[i]!.mesh.visible = false;
    this.flashes.sort((a, b) => b.strength - a.strength);
    this.flashLamps.forEach((lamp, i) => {
      const flash = this.flashes[i];
      lamp.mesh.visible = flash !== undefined && flash.strength > 0;
      if (!flash) return;
      lamp.mesh.position.set(flash.x, 0.02, flash.z);
      lamp.tint.set(flash.tint);
      lamp.level.value = FLASH_LIGHT * flash.strength * flash.strength;
    });
    const cursorLamp = this.cursorLamp!;
    cursorLamp.mesh.position.set(this.cursorAt.x, 0.02, this.cursorAt.z);
    cursorLamp.level.value = this.cursor!.live.value > 0.5 ? 0.9 : 0.25;
    cursorLamp.mesh.visible = this.cursorMesh!.visible;
  }

  private tally(turn: Turn, tau: number): void {
    const prior = turn.opening ? null : this.chronicle!.cached(turn.prior), since = tau - turn.time;
    this.treasuries.forEach((treasury, player) => {
      treasury.count.value = turn.gold[player]!;
      treasury.team.value = player === turn.side ? 0 : 1;
      treasury.fresh.value = prior && turn.gold[player]! > prior.gold[player]! ? 1 - clamp01(since / 1.2) : 0;
    });
  }

  private celebrate(turn: Turn, pool: CelebrationPool, time: number): void {
    const chronicle = this.chronicle!, chain: Turn[] = [];
    for (let t: Turn | null = turn; t && t.step > this.lastTurn && !t.opening && chain.length < 8; t = chronicle.cached(t.prior)) chain.push(t);
    for (const t of chain.reverse()) {
      for (const piece of t.fallen) {
        if (piece.kind === RESOURCE) continue;
        this.centre(piece.x, piece.y, this.at);
        if (piece.owner === t.side) {
          pool.burst(this.other.copy(this.at).setY(38), TOKENS.failureBlue, 0.35, time);
          pool.pulse.fireSigh(0.35);
        } else pool.burst(this.other.copy(this.at).setY(38), TOKENS.celebrationGold, 0.45, time);
      }
      for (const piece of t.pieces) {
        if (piece.owner !== t.side || piece.born !== t.time) continue;
        this.centre(piece.x, piece.y, this.at);
        pool.ripple(this.other.copy(this.at).setY(0), ACCENT, 0.18, time);
      }
    }
  }

  private finish(frame: FrameUpdate, turn: Turn, fetch: Fetch, pool: CelebrationPool, time: number): void {
    const won = frame.events.some((event) => event.type === 'goal'), lost = frame.events.some((event) => event.type === 'harm');
    let opening: Turn | null = turn;
    while (opening && !opening.opening) opening = this.chronicle!.cached(opening.prior);
    const last = opening ? this.chronicle!.finale(opening, fetch) : null;
    const home = last?.pieces.find((piece) => piece.kind === BASE && (won ? piece.owner !== last.side : piece.owner === last.side)) ?? last?.pieces.find((piece) => piece.owner === last.side);
    if (home) this.centre(home.x, home.y, this.at);
    else this.at.set(0, 0, 0);
    if (won) {
      pool.burst(this.other.copy(this.at).setY(38), TOKENS.celebrationGold, 1.4, time);
      pool.ripple(this.other.copy(this.at).setY(0), TOKENS.celebrationGold, 0.8, time);
      pool.pulse.fire(1);
    } else if (lost) {
      pool.sigh(1, this.other.copy(this.at).setY(2), time);
      pool.ripple(this.other.copy(this.at).setY(0), TOKENS.failureBlue, 0.6, time);
    } else pool.ripple(this.other.set(0, 0, 0), 0xc9d3e6, 0.12 * this.grid.w, time);
  }

  private record(A: EnvFrame, turn: Turn, tau: number, fetch: Fetch): void {
    const ledger = this.ledger;
    ledger.game = turn.game;
    ledger.time = turn.time;
    ledger.tau = tau;
    ledger.side = turn.side;
    ledger.foe = turn.foe;
    ledger.member = turn.member;
    ledger.gold = [...turn.gold];
    ledger.reward = A.reward;
    if (A.step === ledger.sensedAt + 1 && ledger.sensedAt >= 0) {
      if (A.words[MW.EVENT] !== 0) {
        ledger.sensed = A.words[MW.EVENT]!;
        ledger.sensedAgo = 0;
      } else if (++ledger.sensedAgo > RECALL) ledger.sensed = 0;
    } else if (A.step !== ledger.sensedAt) {
      ledger.sensed = 0;
      for (let s = A.step; s >= Math.max(0, A.step - RECALL); s--) {
        const f = s === A.step ? A : fetch(s);
        if (!f) break;
        if (f.words[MW.EVENT] === 0) continue;
        ledger.sensed = f.words[MW.EVENT]!;
        ledger.sensedAgo = A.step - s;
        break;
      }
    }
    ledger.sensedAt = A.step;
    ledger.mine.fill(0);
    ledger.theirs.fill(0);
    for (const piece of turn.pieces) {
      if (piece.owner === turn.side) ledger.mine[piece.kind]!++;
      else if (piece.owner !== NEUTRAL) ledger.theirs[piece.kind]!++;
    }
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const fetch = this.fetcher(frame), turn = this.chronicle!.at(frame.state.step, fetch), A = fetch(frame.state.step);
    if (!turn || !A) return;
    const shape = this.shape!, width = shape.width, capacity = shape.capacity;
    const units = kit.marks('units', capacity), health = kit.bars('health', capacity), fog = kit.marks('fog', width * width), sweep = kit.marks('sweep', 1), aims = kit.marks('aims', capacity);
    for (const piece of turn.pieces) {
      const type = shape.types[piece.kind]!, team = this.teamOf(turn, piece.owner), color = team === 0 ? ACCENT : team === 1 ? FOE : MINT;
      const share = piece.kind === RESOURCE ? piece.carry / shape.richest : piece.hp / type.hp;
      this.centre(piece.x, piece.y, this.at);
      units.set(this.at.x, 0.05, this.at.z, type.mobile ? 0.12 : 0.2, color);
      health.set(this.at.x - 0.4 + 0.4 * share, 0.02, this.at.z + 0.42, 0.01, color, 0.8 * share, 0.06);
      if (piece.act === IDLE || piece.act === NONE) continue;
      const aim = aimOf(piece);
      this.centre(aim.x, aim.y, this.other);
      aims.set(this.other.x, 0.06, this.other.z, 0.07, piece.act === ATTACK ? 0xff5a4a : piece.act === PRODUCE ? GOLD : piece.act === HARVEST || piece.act === RETURN ? MINT : 0xe6e9f0);
    }
    for (let c = 0; c < width * width; c++) {
      if (turn.sight[c]) continue;
      this.centre(c % width, Math.floor(c / width), this.at);
      fog.set(this.at.x - 0.36, 0.04, this.at.z - 0.36, 0.05, 0x3a4a66);
    }
    const swept = turn.pieces[A.words[MW.SWEEP]!];
    if (swept && A.observations[SWEEP_AT] !== 0) {
      this.centre(swept.x, swept.y, this.at);
      sweep.set(this.at.x - 0.36, 0.06, this.at.z + 0.36, 0.07, TOKENS.uiCyan);
    }
    turn.pieces.forEach((piece, i) => {
      const line = kit.curve(`act${i}`, 2, 0xe6e9f0);
      if (piece.act === IDLE || piece.act === NONE) return;
      const aim = aimOf(piece);
      this.centre(piece.x, piece.y, this.at);
      this.centre(aim.x, aim.y, this.other);
      line.point(this.at.x, 0.07, this.at.z);
      line.point(this.other.x, 0.07, this.other.z);
    });
    for (let i = turn.pieces.length; i < capacity; i++) kit.curve(`act${i}`, 2, 0xe6e9f0);
    const orders = kit.marks('plan', capacity);
    for (const order of this.plan) {
      this.centre(order.x, order.y, this.at);
      if (order.key === 0) orders.set(this.at.x + 0.3, 0.06, this.at.z - 0.3, 0.05, ACCENT);
      else orders.set(this.at.x + 0.55 * STEP_X[order.key - 1]!, 0.06, this.at.z + 0.55 * STEP_Y[order.key - 1]!, 0.08, ACCENT);
    }
    const card = kit.marks('card', SPAN * SPAN), foe = kit.curve('nearestFoe', 2, FOE), cursor = A.words[MW.CURSOR]!;
    if (A.words[MW.SELECTED] === NOBODY) return;
    const cx = cursor % width, cy = Math.floor(cursor / width);
    for (let i = 0; i < SPAN * SPAN; i++) {
      const code = A.observations[PATCH_AT + i]!;
      if (code === OFF_MAP_GLYPH) continue;
      this.centre(cx + (i % SPAN) - REACH, cy + Math.floor(i / SPAN) - REACH, this.at);
      card.set(this.at.x, 0.05, this.at.z, 0.06, code === UNSEEN_GLYPH ? 0x3a4a66 : code === ORE_GLYPH ? MINT : code > 0 ? ACCENT : code < 0 ? FOE : 0xe6e9f0);
    }
    const dx = A.observations[NEAREST_AT]!, dy = A.observations[NEAREST_AT + 1]!;
    if (dx === 0 && dy === 0) return;
    this.centre(cx, cy, this.at);
    this.centre(cx + dx * 16, cy + dy * 16, this.other);
    foe.point(this.at.x, 0.08, this.at.z);
    foe.point(this.other.x, 0.08, this.other.z);
  }
}

export default (() => new MicroRtsScene()) satisfies SceneFactory;
