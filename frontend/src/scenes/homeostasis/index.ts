import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { hash2f } from '@kit/rng';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { HOMEOSTASIS_EMPTY, HOMEOSTASIS_GROUP, HOMEOSTASIS_KINDS, HOMEOSTASIS_SLOTS, HOMEOSTASIS_X, HOMEOSTASIS_Y, homeostasisAgent, homeostasisCode, homeostasisKind, homeostasisLevel, homeostasisMeal } from '@sim/envs/homeostasis';
import type { GridCell } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount } from '../../format';
import {
  AGENT, CRAVED, GAUGE_H, GAUGE_W, GOLD, ITEM, LIKED, MARGIN, THREAD, TRAIL, VIEW,
  eyeMaterial, flat, foragerArt, gaugeMaterial, iconMaterial, itemMaterial, plateMaterial, threadMaterial, trailMaterial, viewMaterial,
  type ForagerArt, type GaugeSpec, type Level,
} from './art';

const ACCENT = TOKENS.accents.Homeostasis!;
const BOTH = new THREE.Color(LIKED).lerp(new THREE.Color(CRAVED), 0.5).getHex();
const HUES = [LIKED, CRAVED, BOTH] as const;
const NAMES = ['liked', 'craved', 'liked + craved'] as const;
const COMPASS: Record<string, string> = { '0,-1': '↑ north', '0,1': '↓ south', '-1,0': '← west', '1,0': 'east →' };
const VIEW_CELLS = 9;
const IMAGES = 9;
const TRAIL_STEPS = 24;
const FAST = 5;
const TURN = 16;
const ARRIVE = 0.55;
const GAP = 1.1;
const DECADE = 0.8;
const LINEAR = 3.6;
const BELOW = 5;
const ABOVE = 3;
const BOTTOM = -5.3;
const ICON_Y = 5.5;
const EYE_Y = 6.35;
const LIGHT = 9;
const BODY_LIGHT = 3.5;

const ease = (u: number) => smoothstep01(u);
const pop = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : 1 + 2.4 * (u - 1) ** 3 + 1.4 * (u - 1) ** 2);
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}`;
const level = (value: number) => `${value < 0 ? '−' : ''}${formatCount(Math.abs(value))}`;

interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute; used: number; capacity: number }
interface Gauge { level: Level; bump: Level; x: number; zero: number }
interface Meal { slot: number; kind: number; reward: number; x: number; y: number; from: number }
interface Moment {
  x: number;
  y: number;
  dx: number;
  dy: number;
  meal: Meal | null;
  reward: number;
  liked: readonly [number, number];
  craved: readonly [number, number];
  codes: Int16Array;
  toward: Int8Array;
}

class HomeostasisScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['underwater', 'accentLight'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'homeostasis';

  private side = 12;
  private grid: GridSize = { w: 12, h: 12 };
  private scale = 20;
  private full = 200;
  private low = 100;
  private high = 500;
  private hidden = false;
  private perKind = 6;
  private towardAt = 11;
  private window = 30;
  private slope = 0.12;
  private bend = 3;

  private readonly time = uniform(0);
  private items: Batch | null = null;
  private threads: Batch | null = null;
  private trail: Batch | null = null;
  private view: Batch | null = null;
  private forager: Batch | null = null;
  private art: ForagerArt | null = null;
  private readonly gauges: Gauge[] = [];
  private readonly owned: THREE.Mesh[] = [];
  private readonly textures: THREE.Texture[] = [];
  private readonly itemLights: THREE.Mesh[][] = [];
  private readonly bodyLights: THREE.Mesh[] = [];
  private readonly bodyGlow = uniform(BODY_LIGHT);

  private current: Moment | null = null;
  private pending: Meal | null = null;
  private rest = true;
  private step = -1;
  private clock = 0;
  private pace = 0;
  private lastPosition = 0;
  private heading = 0;
  private flashAt = -Infinity;
  private readonly bumpAt = [-Infinity, -Infinity];
  private readonly path: number[] = [];
  private readonly offsets: number[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scaling = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly origin = new THREE.Vector3();
  private readonly tint = new THREE.Color();
  private readonly ledger = { x: 0, y: 0, move: '—', reward: 0, liked: 0, craved: 0, step: 0, meal: undefined as number | undefined, kind: 0, looked: -Infinity };

  override legend: LegendSpec = {
    title: 'HOMEOSTASIS',
    rows: [
      { key: 'at', label: 'forager', tint: ACCENT, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'move', label: 'move', tint: 0xdcf4ff, value: () => this.ledger.move },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(this.ledger.reward) },
      { key: 'liked', label: 'liked level', tint: LIKED, value: () => `${level(this.ledger.liked)} · ${this.ledger.liked <= this.full ? 'not full' : 'full'}` },
      { key: 'craved', label: 'craved level', tint: CRAVED, value: () => `${level(this.ledger.craved)} · ${this.ledger.craved <= this.low ? 'below band' : this.ledger.craved >= this.high ? 'above band' : 'in band'}` },
      { key: 'meal', label: 'last meal', tint: ACCENT, value: () => this.mealLabel() },
    ],
    note: () => (this.hidden ? 'levels hidden from the forager' : 'levels observed by the forager'),
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.side = constant(env, 'HOMEOSTASIS_SIDE');
    this.grid = { w: this.side, h: this.side };
    this.scale = constant(env, 'HOMEOSTASIS_SCALE');
    this.full = constant(env, 'HOMEOSTASIS_FULL') * this.scale;
    this.low = constant(env, 'HOMEOSTASIS_LOW') * this.scale;
    this.high = constant(env, 'HOMEOSTASIS_HIGH') * this.scale;
    this.perKind = constant(env, 'HOMEOSTASIS_PER_TYPE');
    this.hidden = parameter(env, 'hidden') !== 0;
    this.towardAt = VIEW_CELLS + (this.hidden ? 0 : 2);
    if (env.obs !== this.towardAt + 2 * HOMEOSTASIS_KINDS) throw new Error(`${env.envId} observes ${env.obs} values, not a 3×3 view${this.hidden ? '' : ', two levels'} and three nearest-item offsets`);
    this.window = (this.high + this.low) / this.scale;
    this.slope = LINEAR / this.window;
    this.bend = DECADE / (this.slope * Math.LN10);
    const half = this.side / 2, reach = half + MARGIN + GAP + GAUGE_W / 2 + 0.15;
    return new THREE.Box2(new THREE.Vector2(-reach, -half - MARGIN - 0.25), new THREE.Vector2(reach, half + MARGIN + 1.1));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8fb0c8).multiplyScalar(0.55);
  }

  protected override poolOptions() {
    return { burstSize: 40 };
  }

  private gaugeY(value: number): number {
    const u = value / this.scale;
    if (u < 0) return -DECADE * Math.log10(1 - u / this.bend);
    if (u <= this.window) return u * this.slope;
    return this.window * this.slope + DECADE * Math.log10(1 + (u - this.window) / this.bend);
  }

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private images(x: number, z: number, radius: number): number {
    const reach = this.side / 2 + MARGIN + radius, offsets = this.offsets;
    offsets.length = 0;
    const kx = Math.round(x / this.side), kz = Math.round(z / this.side);
    for (let i = -kx - 1; i <= -kx + 1; i++) {
      const ox = i * this.side;
      if (Math.abs(x + ox) > reach) continue;
      for (let j = -kz - 1; j <= -kz + 1; j++) {
        const oz = j * this.side;
        if (Math.abs(z + oz) <= reach) offsets.push(ox, oz);
      }
    }
    return offsets.length / 2;
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[HOMEOSTASIS_X]!;
        agent.y = state.pose[HOMEOSTASIS_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.truth(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const side = this.side, half = side / 2, accent = this.accentOf(ctx), lit = light.lit;
    this.scene.background = new THREE.Color(0x020406);
    const maps = { mottle: fbm(ctx.seed ^ 0x40e0, { size: 256, frequency: 4, octaves: 5 }), grain: fbm(ctx.seed ^ 0x9a17, { size: 256, frequency: 16, octaves: 3 }) };
    this.textures.push(maps.mottle, maps.grain);
    this.add(new THREE.Mesh(flat(80, 80), plateMaterial(side, maps, accent, lit)), 0);

    this.trail = this.batch(flat(1, 1), 'aTrail', trailMaterial(side, accent), TRAIL_STEPS * IMAGES, 1);
    this.view = this.batch(flat(VIEW, VIEW), 'aView', viewMaterial(side, accent), IMAGES, 2);
    this.items = this.batch(flat(ITEM, ITEM), 'aItem', itemMaterial(side, this.time), (HOMEOSTASIS_SLOTS + 1) * IMAGES, 3);
    this.threads = this.batch(flat(1, 1), 'aThread', threadMaterial(side, this.time), HOMEOSTASIS_KINDS * IMAGES, 4);
    const art = foragerArt(side, accent, this.time);
    this.art = art;
    this.forager = this.batch(flat(AGENT, AGENT), 'aForager', art.material, IMAGES, 5);

    const disc = new THREE.CircleGeometry(0.11, 24).rotateX(-Math.PI / 2);
    for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) {
      const copies: THREE.Mesh[] = [];
      if (slot % HOMEOSTASIS_GROUP < this.perKind) for (let k = 0; k < 4; k++) copies.push(this.add(light.emitter(disc, rgb(HUES[homeostasisKind(slot) - 1]!).mul(LIGHT)), 0));
      this.itemLights.push(copies);
    }
    for (let k = 0; k < 4; k++) this.bodyLights.push(this.add(light.emitter(new THREE.CircleGeometry(0.2, 32).rotateX(-Math.PI / 2), rgb(accent).mul(this.bodyGlow)), 0));

    const top = BOTTOM + (BELOW + ABOVE) * DECADE + LINEAR, zero = BOTTOM + BELOW * DECADE;
    const specs: GaugeSpec[] = [
      { hue: LIKED, outward: -1, zero, top, bottom: BOTTOM, window: LINEAR, step: this.low / this.scale * this.slope, decade: DECADE, good: [-GAUGE_H, zero + this.gaugeY(this.full)], marks: [zero + this.gaugeY(this.full)], hidden: this.hidden },
      { hue: CRAVED, outward: 1, zero, top, bottom: BOTTOM, window: LINEAR, step: this.low / this.scale * this.slope, decade: DECADE, good: [zero + this.gaugeY(this.low), zero + this.gaugeY(this.high)], marks: [zero + this.gaugeY(this.low), zero + this.gaugeY(this.high)], hidden: this.hidden },
    ];
    specs.forEach((spec, i) => {
      const x = spec.outward * (half + MARGIN + GAP), level = uniform(zero), bump = uniform(0);
      this.add(new THREE.Mesh(flat(GAUGE_W, GAUGE_H), gaugeMaterial(spec, level, bump, this.time)), 6).position.set(x, 0, 0);
      this.add(new THREE.Mesh(flat(ITEM, ITEM), iconMaterial(i + 1)), 6).position.set(x, 0, -ICON_Y);
      this.add(new THREE.Mesh(flat(0.9, 0.9), eyeMaterial(this.hidden)), 6).position.set(x, 0, -EYE_Y);
      this.gauges.push({ level, bump, x, zero });
    });
    pool.group.traverse((object) => { object.renderOrder = 10; });
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.owned.push(mesh);
    this.scene.add(mesh);
    return mesh;
  }

  private batch(geometry: THREE.BufferGeometry, name: string, material: THREE.Material, capacity: number, order: number): Batch {
    const state = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, state);
    const mesh = instancedMesh(geometry, material, capacity);
    mesh.count = 0;
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return { mesh, state, used: 0, capacity };
  }

  private put(batch: Batch, x: number, z: number, angle: number, sx: number, sz: number, a: number, b: number, c: number, d: number): void {
    if (batch.used >= batch.capacity) throw new Error(`${batch.mesh.name || 'homeostasis batch'} holds ${batch.capacity} instances`);
    this.quaternion.setFromAxisAngle(this.axis, angle);
    batch.mesh.setMatrixAt(batch.used, this.matrix.compose(this.at.set(x, 0, z), this.quaternion, this.scaling.set(sx, 1, sz)));
    batch.state.setXYZW(batch.used, a, b, c, d);
    batch.used++;
  }

  private commit(batch: Batch): void {
    batch.mesh.count = batch.used;
    batch.mesh.instanceMatrix.needsUpdate = true;
    batch.state.needsUpdate = true;
    batch.used = 0;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    for (const batch of [this.items, this.threads, this.trail, this.view, this.forager]) {
      if (!batch) continue;
      batch.mesh.geometry.dispose();
      (batch.mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
  }

  private read(frame: FrameUpdate, s: number): Moment | null {
    const now = frame.frameAt(s);
    if (now.step !== s) return null;
    const before = s > 0 ? frame.frameAt(s - 1) : now;
    if (before.step !== Math.max(0, s - 1)) return null;
    const at = homeostasisAgent(now), was = homeostasisAgent(before), side = this.side;
    const x = was & 15, y = was >>> 4, dx = (((at & 15) - x + side + side / 2) % side) - side / 2, dy = (((at >>> 4) - y + side + side / 2) % side) - side / 2;
    const codes = new Int16Array(HOMEOSTASIS_SLOTS), toward = new Int8Array(2 * HOMEOSTASIS_KINDS);
    for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) {
      const code = homeostasisCode(now, slot);
      codes[slot] = code === HOMEOSTASIS_EMPTY ? -1 : code;
    }
    for (let i = 0; i < toward.length; i++) toward[i] = Math.round(now.observations[this.towardAt + i]! * side / 2);
    const slot = s > 0 ? homeostasisMeal(before, now) : -1;
    const meal = slot < 0 ? null : { slot, kind: homeostasisKind(slot), reward: now.reward, x: at & 15, y: at >>> 4, from: homeostasisCode(before, slot) };
    return {
      x, y, dx, dy, meal, reward: now.reward,
      liked: [homeostasisLevel(before, 0), homeostasisLevel(now, 0)],
      craved: [homeostasisLevel(before, 1), homeostasisLevel(now, 1)],
      codes, toward,
    };
  }

  protected tick(frame: FrameUpdate, { pool }: Stage2D): void {
    const dt = frame.dtReal;
    this.clock += dt;
    this.time.value = this.clock;
    pool.update(dt, this.clock);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST, step = frame.state.step;
    if (frame.scrubbed || step < this.step || this.current === null) this.settle(frame, step);
    else if (step !== this.step) this.advance(frame, step, fast, pool);
    if (!this.current) return;
    const u = this.rest || fast || frame.frameAt(step + 1).step !== step + 1 ? 1 : clamp01(frame.position - step);
    if (this.pending && u >= ARRIVE) {
      this.celebrate(this.pending, pool);
      this.pending = null;
    }
    if (this.clock - this.ledger.looked > 0.25) this.lookBack(frame, step);
    this.draw(this.current, u, dt);
  }

  private settle(frame: FrameUpdate, step: number): void {
    this.pending = null;
    this.rest = true;
    this.flashAt = this.bumpAt[0] = this.bumpAt[1] = -Infinity;
    this.current = this.read(frame, step);
    this.step = this.current ? step : -1;
    if (!this.current) return;
    const { dx, dy } = this.current;
    if (dx || dy) this.heading = Math.atan2(-dy, dx);
    this.remember(frame, step);
    this.ledger.looked = -Infinity;
  }

  private advance(frame: FrameUpdate, step: number, fast: boolean, pool: CelebrationPool): void {
    if (this.pending) this.celebrate(this.pending, pool);
    if (!fast) for (let s = this.step + 1; s < step; s++) {
      const skipped = this.read(frame, s);
      if (skipped?.meal) this.celebrate(skipped.meal, pool);
    }
    const next = this.read(frame, step);
    if (!next) {
      this.settle(frame, step);
      return;
    }
    this.current = next;
    this.pending = next.meal && !fast ? next.meal : null;
    this.rest = false;
    this.step = step;
    this.remember(frame, step);
  }

  private remember(frame: FrameUpdate, step: number): void {
    const now = this.current!, ledger = this.ledger, side = this.side;
    ledger.x = (now.x + now.dx + side) % side;
    ledger.y = (now.y + now.dy + side) % side;
    const crossed = now.x + now.dx < 0 || now.x + now.dx >= side || now.y + now.dy < 0 || now.y + now.dy >= side;
    ledger.move = step < 1 ? '—' : `${COMPASS[`${now.dx},${now.dy}`] ?? '—'}${crossed ? ' · wraps' : ''}`;
    ledger.reward = now.reward;
    ledger.liked = now.liked[1];
    ledger.craved = now.craved[1];
    ledger.step = step;
    const path = this.path;
    path.length = 0;
    let px = now.x, py = now.y;
    path.push(px, py);
    for (let k = 1; k < TRAIL_STEPS && step - k >= 1; k++) {
      const later = frame.frameAt(step - k), earlier = frame.frameAt(step - k - 1);
      if (later.step !== step - k || earlier.step !== step - k - 1) break;
      const a = homeostasisAgent(later), b = homeostasisAgent(earlier);
      px -= (((a & 15) - (b & 15) + side + side / 2) % side) - side / 2;
      py -= (((a >>> 4) - (b >>> 4) + side + side / 2) % side) - side / 2;
      path.push(px, py);
    }
  }

  private lookBack(frame: FrameUpdate, step: number): void {
    const ledger = this.ledger;
    ledger.looked = this.clock;
    ledger.meal = lastEvent(frame.eventLog, 'cue', step);
    ledger.kind = 0;
    if (ledger.meal === undefined || ledger.meal < 1) return;
    const now = frame.frameAt(ledger.meal), before = frame.frameAt(ledger.meal - 1);
    if (now.step !== ledger.meal || before.step !== ledger.meal - 1) return;
    const slot = homeostasisMeal(before, now);
    ledger.kind = slot < 0 ? 0 : homeostasisKind(slot);
  }

  private mealLabel(): string {
    const { meal, kind, step } = this.ledger;
    if (meal === undefined) return 'none yet';
    const when = meal === step ? 'now' : `${formatCount(step - meal)} step${step - meal === 1 ? '' : 's'} ago`;
    return kind ? `${NAMES[kind - 1]} · ${when}` : when;
  }

  private celebrate(meal: Meal, pool: CelebrationPool): void {
    const at = this.place(meal.x, meal.y, this.other), hue = HUES[meal.kind - 1]!;
    pool.ripple(at.setY(0), meal.reward < 0 ? TOKENS.failureBlue : hue, 0.3, this.clock);
    if (meal.reward > 0) pool.burst(at.setY(34), GOLD, meal.reward > 1 ? 0.7 : 0.32, this.clock);
    else if (meal.reward < 0) pool.burst(at.setY(34), TOKENS.failureBlue, meal.reward < -1 ? 0.4 : 0.24, this.clock);
    this.flashAt = this.clock;
    const tint = this.tint.set(hue);
    this.art?.tint.value.set(tint.r, tint.g, tint.b);
    if (meal.kind & 1) this.bumpAt[0] = this.clock;
    if (meal.kind & 2) this.bumpAt[1] = this.clock;
  }

  private draw(now: Moment, u: number, dt: number): void {
    const items = this.items, threads = this.threads, trail = this.trail, view = this.view, forager = this.forager, art = this.art;
    if (!items || !threads || !trail || !view || !forager || !art) return;
    const e = ease(u), moved = now.dx !== 0 || now.dy !== 0;
    const base = this.place(now.x, now.y, this.origin), ax = base.x + now.dx * e, az = base.z + now.dy * e;
    if (moved) this.heading = turn(this.heading, Math.atan2(-now.dy, now.dx), 1 - Math.exp(-TURN * dt));
    art.stretch.value = moved && u < 1 ? Math.sin(Math.PI * u) ** 0.8 : 0;
    const settle = now.meal ? ease(clamp01((u - 0.5) / 0.35)) : e;
    const likedLevel = now.liked[0] + (now.liked[1] - now.liked[0]) * settle, cravedLevel = now.craved[0] + (now.craved[1] - now.craved[0]) * settle;
    art.liked.value = likedLevel <= this.full ? 1 : 0;
    art.craved.value = cravedLevel > this.low && cravedLevel < this.high ? 1 : 0;
    art.flash.value = Math.exp(-(this.clock - this.flashAt) * 3.5);
    const [liked, craved] = this.gauges;
    if (liked && craved) {
      liked.level.value = liked.zero + this.gaugeY(likedLevel);
      craved.level.value = craved.zero + this.gaugeY(cravedLevel);
      liked.bump.value = Math.exp(-(this.clock - this.bumpAt[0]!) * 3);
      craved.bump.value = Math.exp(-(this.clock - this.bumpAt[1]!) * 3);
    }

    for (let i = 0, n = this.images(ax, az, AGENT * 0.4); i < n; i++) {
      const ox = this.offsets[2 * i]!, oz = this.offsets[2 * i + 1]!;
      this.put(forager, ax + ox, az + oz, this.heading, 1, 1, 0, 0, 0, 0);
      const lamp = this.bodyLights[i];
      if (lamp) lamp.position.set(ax + ox, 0.02, az + oz);
    }
    this.bodyLights.forEach((lamp, i) => { lamp.visible = i < forager.used; });
    this.bodyGlow.value = BODY_LIGHT * (1 + 1.5 * art.flash.value);
    for (let i = 0, n = this.images(ax, az, VIEW / 2); i < n; i++) this.put(view, ax + this.offsets[2 * i]!, az + this.offsets[2 * i + 1]!, 0, 1, 1, 0, 0, 0, 0);

    const meal = now.meal;
    for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) {
      const code = now.codes[slot]!;
      if (code < 0) continue;
      const kind = homeostasisKind(slot), lights = this.itemLights[slot]!;
      let lightCode = code, lightGrow = 1, used = 0;
      if (meal && meal.slot === slot) {
        const fade = 1 - ease(clamp01((u - 0.42) / 0.36)), grow = pop(clamp01((u - ARRIVE) / (1 - ARRIVE)));
        if (fade > 0) this.item(meal.from, slot, kind, fade);
        if (grow > 0) this.item(code, slot, kind, grow);
        if (u < ARRIVE) { lightCode = meal.from; lightGrow = fade; } else lightGrow = grow;
      } else this.item(code, slot, kind, 1);
      this.place(lightCode & 15, lightCode >>> 4, this.at);
      const lx = this.at.x, lz = this.at.z;
      for (let i = 0, n = this.images(lx, lz, 0.6); i < n && i < lights.length; i++, used++) {
        const lamp = lights[i]!;
        lamp.position.set(lx + this.offsets[2 * i]!, 0.02, lz + this.offsets[2 * i + 1]!);
        lamp.scale.setScalar(Math.max(lightGrow, 0.001));
      }
      lights.forEach((lamp, i) => { lamp.visible = i < used && lightGrow > 0.01; });
    }

    const destX = base.x + now.dx, destZ = base.z + now.dy;
    for (let kind = 0; kind < HOMEOSTASIS_KINDS; kind++) {
      const tx = destX + now.toward[2 * kind]!, tz = destZ + now.toward[2 * kind + 1]!, vx = tx - ax, vz = tz - az, span = Math.hypot(vx, vz);
      if (span < 1e-3) continue;
      const angle = Math.atan2(-vz, vx), mx = (ax + tx) / 2, mz = (az + tz) / 2;
      for (let i = 0, n = this.images(mx, mz, span / 2 + 0.6); i < n; i++) this.put(threads, mx + this.offsets[2 * i]!, mz + this.offsets[2 * i + 1]!, angle, span + 0.9, THREAD, span, kind + 1, 0.85, 0);
    }

    const path = this.path;
    for (let k = 0; k === 0 || 2 * k + 1 < path.length; k++) {
      let x0: number, z0: number, x1: number, z1: number;
      if (k === 0) {
        if (!moved) continue;
        x0 = base.x; z0 = base.z; x1 = ax; z1 = az;
      } else {
        x1 = base.x + path[2 * k - 2]! - now.x;
        z1 = base.z + path[2 * k - 1]! - now.y;
        x0 = base.x + path[2 * k]! - now.x;
        z0 = base.z + path[2 * k + 1]! - now.y;
      }
      const vx = x1 - x0, vz = z1 - z0, span = Math.hypot(vx, vz);
      if (span < 1e-3) continue;
      const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2, alpha = (1 - k / TRAIL_STEPS) ** 1.4;
      for (let i = 0, n = this.images(mx, mz, span / 2 + TRAIL); i < n; i++) this.put(trail, mx + this.offsets[2 * i]!, mz + this.offsets[2 * i + 1]!, Math.atan2(-vz, vx), span + TRAIL, TRAIL, span, alpha, 0, 0);
    }

    for (const batch of [items, threads, trail, view, forager]) this.commit(batch);
  }

  private item(code: number, slot: number, kind: number, grow: number): void {
    const items = this.items!;
    this.place(code & 15, code >>> 4, this.at);
    const x = this.at.x, z = this.at.z, spin = hash2f(slot, code, 0x51ed) * Math.PI * 2;
    for (let i = 0, n = this.images(x, z, ITEM / 2); i < n; i++) this.put(items, x + this.offsets[2 * i]!, z + this.offsets[2 * i + 1]!, 0, 1, 1, kind, spin, grow, 0);
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const step = frame.state.step, now = frame.frameAt(step);
    if (now.step !== step) return;
    const side = this.side, at = homeostasisAgent(now), x = at & 15, y = at >>> 4;
    const items = kit.marks('items', HOMEOSTASIS_SLOTS);
    for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) {
      const code = homeostasisCode(now, slot);
      if (code === HOMEOSTASIS_EMPTY) continue;
      this.place(code & 15, code >>> 4, this.at);
      items.set(this.at.x, 0.05, this.at.z, 0.12, HUES[homeostasisKind(slot) - 1]!);
    }
    const seen = kit.marks('seen', VIEW_CELLS);
    for (let cell = 0; cell < VIEW_CELLS; cell++) {
      const value = now.observations[cell]!;
      if (value <= 0) continue;
      const cx = (x + (cell % 3) - 1 + side) % side, cy = (y + Math.floor(cell / 3) - 1 + side) % side;
      this.place(cx, cy, this.at);
      seen.set(this.at.x, 0.04, this.at.z, 0.24, value < 0.5 ? LIKED : value < 0.9 ? CRAVED : 0xffffff);
    }
    this.place(x, y, this.other);
    const box = kit.curve('view', 5, TOKENS.uiCyan);
    for (const [dx, dz] of [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]] as const) box.point(this.other.x + dx, 0.06, this.other.z + dz);
    const targets = kit.marks('targets', HOMEOSTASIS_KINDS);
    for (let kind = 0; kind < HOMEOSTASIS_KINDS; kind++) {
      const ox = Math.round(now.observations[this.towardAt + 2 * kind]! * side / 2), oy = Math.round(now.observations[this.towardAt + 2 * kind + 1]! * side / 2);
      const line = kit.curve(`toward${kind}`, 2, HUES[kind]!);
      line.point(this.other.x, 0.07, this.other.z);
      line.point(this.other.x + ox, 0.07, this.other.z + oy);
      this.place((x + ox + side) % side, (y + oy + side) % side, this.at);
      targets.set(this.at.x, 0.08, this.at.z, 0.2, HUES[kind]!);
    }
    [homeostasisLevel(now, 0), homeostasisLevel(now, 1)].forEach((value, i) => {
      const gauge = this.gauges[i];
      if (!gauge) return;
      const line = kit.curve(`level${i}`, 2, i ? CRAVED : LIKED), z = -(gauge.zero + this.gaugeY(value));
      line.point(gauge.x - 0.5, 0.08, z);
      line.point(gauge.x + 0.5, 0.08, z);
    });
  }
}

export default (() => new HomeostasisScene()) satisfies SceneFactory;
