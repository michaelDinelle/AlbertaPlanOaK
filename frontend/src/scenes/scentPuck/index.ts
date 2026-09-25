import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendRow, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { SCENTPUCK_LURE_X, SCENTPUCK_LURE_Y, SCENTPUCK_RIVAL_SCENT, SCENTPUCK_RIVAL_X, SCENTPUCK_RIVAL_Y, SCENTPUCK_SCENT, SCENTPUCK_X, SCENTPUCK_Y, scentPuckDecoder } from '@sim/envs/scentpuck';
import type { EnvFrame, EventStream } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { DAB, dabMaterial, flat, floorMaterial, LURE, lureMaterial, MARK, markMaterial, PALETTE, PUCK, puckMaterial, RIVAL, rivalMaterial, TRAY, PLATE, WEAR, wearMaterial, type Lamp, type Pools, type Puck } from './art';

const SCALE = 10;
const VIEW = 11;
const DOCK = 2.2;
const TRAIL = 720;
const WORN = 4096;
const WORN_TAU = 1400;
const WORN_SPOTS = 1024;
const WORN_EVERY = 0.5;
const REBUILD = 0.05;
const FADE = 0.9;
const GULP = 0.35;
const SPROUT = 0.55;
const FLASH = 0.9;
const PING = 0.8;
const RECENT = 1.6;
const TURN = 14;
const SNIFF = 0.16;
const LURE_LIGHT = 2;
const RIVAL_LIGHT = 2;
const LURE_POOL = 1.6;
const RIVAL_POOL = 1.4;
const LED_LIGHT = 5;

const at = (frame: EnvFrame, slot: number): number => wordFloat(frame.words[slot]!);
const wrapAngle = (angle: number): number => Math.atan2(Math.sin(angle), Math.cos(angle));
const signed = (value: number): string => formatSigned(value).replace('-', '−');
const coord = (value: number): string => `${value <= -0.0005 ? '−' : ''}${Math.abs(value).toFixed(3)}`;
const reading = (value: number): string => value.toFixed(3);
const sprout = (k: number): number => k >= 1 ? 1 : k <= 0 ? 0 : 1 + 2.2 * (k - 1) ** 3 + 1.2 * (k - 1) ** 2;

function heading(from: EnvFrame, to: EnvFrame): number | null {
  const dx = at(to, SCENTPUCK_X) - at(from, SCENTPUCK_X), dy = at(to, SCENTPUCK_Y) - at(from, SCENTPUCK_Y);
  return dx === 0 && dy === 0 ? null : Math.atan2(dy, dx);
}

function moveOf(from: EnvFrame, to: EnvFrame): string {
  const dx = at(to, SCENTPUCK_X) - at(from, SCENTPUCK_X), dy = at(to, SCENTPUCK_Y) - at(from, SCENTPUCK_Y);
  if (dx === 0 && dy === 0) return 'held';
  if (Math.abs(dx) >= Math.abs(dy)) return dx > 0 ? '→ east' : '← west';
  return dy > 0 ? '↑ north' : '↓ south';
}

interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute }

class ScentPuckScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'scentPuck';

  private competing = false;
  private moves = 1;
  private goalReward = 0;
  private puckMesh: THREE.Mesh | null = null;
  private leds: THREE.Mesh | null = null;
  private lureMesh: THREE.Mesh | null = null;
  private gulpMesh: THREE.Mesh | null = null;
  private lureLamp: THREE.Mesh | null = null;
  private rivalMesh: THREE.Mesh | null = null;
  private fleeMesh: THREE.Mesh | null = null;
  private rivalLamp: THREE.Mesh | null = null;
  private mark: THREE.Mesh | null = null;
  private dabs: Batch | null = null;
  private wear: Batch | null = null;

  private readonly clock = uniform(0);
  private readonly puck: Puck = { heading: uniform(new THREE.Vector2(0, 1)), scent: uniform(0), rival: uniform(0), sniff: uniform(0), flash: uniform(0) };
  private readonly lure: Lamp = { glow: uniform(0), fade: uniform(1), clock: this.clock, ping: uniform(0) };
  private readonly gulp: Lamp = { glow: uniform(1), fade: uniform(0), clock: this.clock, ping: uniform(0) };
  private readonly rival: Lamp = { glow: uniform(0), fade: uniform(1), clock: this.clock, ping: uniform(0) };
  private readonly flee: Lamp = { glow: uniform(0), fade: uniform(0), clock: this.clock, ping: uniform(0) };
  private readonly lureShine = uniform(LURE_LIGHT);
  private readonly rivalShine = uniform(RIVAL_LIGHT);
  private readonly ledScent = uniform(0);
  private readonly ledRival = uniform(0);
  private readonly pools: Pools = { lure: uniform(new THREE.Vector4()), rival: uniform(new THREE.Vector4()), on: uniform(1) };

  private step = -1;
  private wall = 0;
  private angle = Math.PI / 2;
  private sniffAt = -Infinity;
  private caught = -Infinity;
  private hunt = 0;
  private readonly lureFrom = new THREE.Vector2();
  private readonly rivalFrom = new THREE.Vector2();
  private trailAt = -Infinity;
  private trailStale = true;
  private wornAt = -Infinity;
  private wornStale = true;
  private readonly spots = new Map<number, number>();
  private readonly wornX = new Float32Array(WORN_SPOTS);
  private readonly wornY = new Float32Array(WORN_SPOTS);
  private readonly wornW = new Float32Array(WORN_SPOTS);
  private readonly keys = new Set<number>();
  private readonly shown = { x: 0, y: 0 };
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, catches: 0 };
  private readonly ledger = { x: 0, y: 0, action: '—', scent: 0, rival: 0, distance: 0, reward: 0, step: 0, looked: -Infinity };

  private readonly matrix = new THREE.Matrix4();
  private readonly spot = new THREE.Vector3();

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(x * SCALE, 0, -y * SCALE);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.competing = parameter(env, 'competing') === 1;
    this.goalReward = constant(env, 'SCENTPUCK_GOAL_REWARD');
    this.moves = env.actions;
    const accent = this.accentOf(ctx), ledger = this.ledger;
    const rows: LegendRow[] = [
      { key: 'puck', label: 'puck', tint: accent, value: () => `x ${coord(ledger.x)} · y ${coord(ledger.y)}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ledger.action },
      { key: 'scent', label: 'scent', tint: PALETTE.scent, value: () => reading(ledger.scent) },
      ...(this.competing ? [{ key: 'rival', label: 'rival scent', tint: PALETTE.rival, value: () => reading(ledger.rival) }] : []),
      { key: 'lure', label: 'lure', tint: PALETTE.gold, value: () => `${ledger.distance.toFixed(3)} away` },
      { key: 'reward', label: 'reward', tint: 0xffe2a8, value: () => signed(ledger.reward) },
      { key: 'catches', label: 'catches', tint: TOKENS.celebrationGold, value: () => `${formatCount(this.tally.catches)} · +${formatCount(this.goalReward)} each` },
    ];
    this.legend = {
      title: this.competing ? 'SCENT PUCK · RIVAL' : 'SCENT PUCK',
      rows,
      note: () => `hunting ${formatCount(ledger.step - this.hunt)} step${ledger.step - this.hunt === 1 ? '' : 's'}`,
      bar: () => clamp01(ledger.scent),
    } satisfies LegendSpec;
    return new THREE.Box2(new THREE.Vector2(-VIEW, -VIEW), new THREE.Vector2(VIEW, VIEW + DOCK));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8ea4d6).multiplyScalar(0.62);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const spot = new THREE.Vector3(), other = new THREE.Vector3();
    return {
      debug: (frame, kit) => {
        const step = frame.state.step, cur = frame.frameAt(step);
        if (cur.step !== step) return;
        const first = frame.frameAt(0);
        if (first.step === 0) {
          this.place(at(first, SCENTPUCK_X), at(first, SCENTPUCK_Y), spot);
          kit.marks('start', 1).set(spot.x, 0.05, spot.z, 0.22, 0x86f0b4);
        }
        const samples = kit.marks('samples', 2 * TRAIL), route = kit.curve('route', TRAIL + 1, TOKENS.uiCyan);
        this.keys.clear();
        for (let s = step; s >= 0 && s > step - TRAIL; s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          this.place(at(f, SCENTPUCK_X), at(f, SCENTPUCK_Y), spot);
          route.point(spot.x, 0.06, spot.z);
          const key = f.words[SCENTPUCK_X]! * 4294967296 + f.words[SCENTPUCK_Y]!;
          if (!this.keys.has(key)) {
            this.keys.add(key);
            samples.set(spot.x, 0.05, spot.z, 0.04 + 0.2 * clamp01(f.observations[SCENTPUCK_SCENT]!), PALETTE.scent);
            if (this.competing) samples.set(spot.x + 0.12, 0.05, spot.z, 0.04 + 0.2 * clamp01(f.observations[SCENTPUCK_RIVAL_SCENT]!), PALETTE.rival);
          }
          if (f.done) break;
        }
        this.place(at(cur, SCENTPUCK_X), at(cur, SCENTPUCK_Y), spot);
        kit.marks('puck', 1).set(spot.x, 0.07, spot.z, 0.3, TOKENS.uiCyan);
        this.place(at(cur, SCENTPUCK_LURE_X), at(cur, SCENTPUCK_LURE_Y), other);
        kit.marks('lure', 1).set(other.x, 0.07, other.z, 0.3, TOKENS.celebrationGold);
        const tether = kit.curve('tether', 2, TOKENS.celebrationGold);
        tether.point(spot.x, 0.07, spot.z);
        tether.point(other.x, 0.07, other.z);
        if (this.competing) {
          this.place(at(cur, SCENTPUCK_RIVAL_X), at(cur, SCENTPUCK_RIVAL_Y), other);
          kit.marks('rival', 1).set(other.x, 0.07, other.z, 0.3, PALETTE.rival);
          const leash = kit.curve('leash', 2, PALETTE.rival);
          leash.point(spot.x, 0.07, spot.z);
          leash.point(other.x, 0.07, other.z);
        }
        const next = frame.frameAt(step + 1), move = kit.curve('move', 2, 0xffa05a);
        if (next.step !== step + 1) return;
        move.point(spot.x, 0.08, spot.z);
        this.place(at(next, SCENTPUCK_X), at(next, SCENTPUCK_Y), other);
        move.point(other.x, 0.08, other.z);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit;
    this.scene.background = new THREE.Color(PALETTE.ink);
    const floor = new THREE.Mesh(flat(VIEW * 8, VIEW * 8), floorMaterial(lit, this.pools, SCALE));
    floor.renderOrder = 0;
    floor.frustumCulled = false;
    const mark = new THREE.Mesh(flat(MARK, MARK), markMaterial(lit));
    mark.renderOrder = 2;
    mark.visible = false;
    this.mark = mark;
    this.scene.add(floor, mark);

    this.wear = this.batch(flat(WEAR, WEAR), 'aWear', wearMaterial(lit, this.pools), WORN_SPOTS, 1);
    this.dabs = this.batch(flat(DAB, DAB), 'aDab', dabMaterial(), 2 * TRAIL, 3);

    const sprite = (size: number, material: THREE.Material, order: number): THREE.Mesh => {
      const mesh = new THREE.Mesh(flat(size, size), material);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return mesh;
    };
    const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);
    if (this.competing) {
      this.fleeMesh = sprite(RIVAL, rivalMaterial(this.flee), 4);
      this.rivalMesh = sprite(RIVAL, rivalMaterial(this.rival), 5);
      this.rivalLamp = light.emitter(disc(PLATE - 0.04), rgb(PALETTE.rival).mul(this.rivalShine));
      this.rivalLamp.renderOrder = 1;
      this.scene.add(this.rivalLamp);
    }
    this.gulpMesh = sprite(LURE, lureMaterial(lit, this.gulp), 4);
    this.lureMesh = sprite(LURE, lureMaterial(lit, this.lure), 5);
    this.lureLamp = light.emitter(disc(TRAY - 0.04), rgb(PALETTE.liquid).mul(this.lureShine));
    this.lureLamp.renderOrder = 1;
    this.puckMesh = sprite(PUCK, puckMaterial(this.accentOf(ctx), this.competing, lit, this.puck), 7);
    this.leds = light.emitter(disc(0.32), rgb(PALETTE.scent).mul(this.ledScent).add(rgb(PALETTE.rival).mul(this.ledRival)));
    this.leds.renderOrder = 2;
    this.scene.add(this.lureLamp, this.leds);
    pool.group.traverse((object) => { object.renderOrder = 20; });
  }

  private batch(geometry: THREE.BufferGeometry, name: string, material: THREE.Material, count: number, order: number): Batch {
    const state = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, state);
    const mesh = instancedMesh(geometry, material, count);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    return { mesh, state };
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    this.pools.on.value = on ? 0 : 1;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.dabs = null;
    this.wear = null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.wall += frame.dtReal;
    const step = frame.state.step;
    if (frame.scrubbed || step < this.step) this.settle(frame, step, pool);
    else for (let s = this.step + 1; s <= step; s++) if (!this.advance(frame, s, pool)) break;
    pool.update(frame.dtReal, time);
    if (step !== this.step) {
      this.sniffAt = this.wall;
      this.trailStale = this.wornStale = true;
    }
    this.step = step;
    if (this.trailStale && this.wall - this.trailAt >= REBUILD) this.trace(frame, step);
    if (this.wornStale && this.wall - this.wornAt >= WORN_EVERY) this.scuff(frame, step);
    this.record(frame, step);
    this.count(frame.eventLog, step);
    this.pose(frame, step);
    this.clock.value = this.wall;
  }

  private settle(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    pool.clear();
    this.caught = -Infinity;
    this.hunt = lastEvent(frame.eventLog, 'goal', step) ?? 0;
    this.ledger.looked = -Infinity;
    this.trailAt = -Infinity;
    for (let s = step; s > 0 && s >= step - Math.ceil(RECENT * this.controlHz); s--) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1) break;
      if (this.caughtBy(s, now, before, pool)) break;
    }
    for (let s = step + 1; s > 0 && s > step - 64; s--) {
      const later = frame.frameAt(s), earlier = frame.frameAt(s - 1);
      if (later.step !== s || earlier.step !== s - 1) continue;
      const toward = heading(earlier, later);
      if (toward === null) continue;
      this.angle = toward;
      break;
    }
    const first = frame.frameAt(0);
    if (this.mark && first.step === 0) {
      this.place(at(first, SCENTPUCK_X), at(first, SCENTPUCK_Y), this.mark.position);
      this.mark.visible = true;
    }
    this.step = -1;
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool): boolean {
    const now = frame.frameAt(s), before = frame.frameAt(s - 1);
    if (now.step !== s || before.step !== s - 1) {
      this.settle(frame, frame.state.step, pool);
      return false;
    }
    this.caughtBy(s, now, before, pool);
    return true;
  }

  private caughtBy(s: number, now: EnvFrame, before: EnvFrame, pool: CelebrationPool): boolean {
    if (!scentPuckDecoder.deriveEvents(before, now).some((event) => event.type === 'goal')) return false;
    this.hunt = Math.max(this.hunt, s);
    this.caught = s;
    this.lureFrom.set(at(before, SCENTPUCK_LURE_X), at(before, SCENTPUCK_LURE_Y));
    this.rivalFrom.set(at(before, SCENTPUCK_RIVAL_X), at(before, SCENTPUCK_RIVAL_Y));
    this.place(this.lureFrom.x, this.lureFrom.y, this.spot);
    pool.burst(this.spot.setY(32), TOKENS.celebrationGold, 0.6, s / this.controlHz);
    return true;
  }

  private trace(frame: FrameUpdate, step: number): void {
    const dabs = this.dabs;
    if (!dabs) return;
    this.trailAt = this.wall;
    this.trailStale = false;
    this.keys.clear();
    let n = 0, fade = 1, older = false;
    for (let s = step; s >= 0 && s > step - TRAIL; s--) {
      const f = frame.frameAt(s);
      if (f.step !== s) {
        this.trailStale = true;
        break;
      }
      const key = f.words[SCENTPUCK_X]! * 4294967296 + f.words[SCENTPUCK_Y]!;
      if (!this.keys.has(key)) {
        this.keys.add(key);
        dabs.mesh.setMatrixAt(n, this.matrix.makeTranslation(this.place(at(f, SCENTPUCK_X), at(f, SCENTPUCK_Y), this.spot)));
        dabs.state.setXYZW(n, f.observations[SCENTPUCK_SCENT]!, this.competing ? f.observations[SCENTPUCK_RIVAL_SCENT]! : 0, fade * (1 - (step - s) / TRAIL) ** 1.3, 0);
        n++;
      }
      if (!f.done) continue;
      const age = (frame.position - s) / this.controlHz;
      if (older || age >= FADE) break;
      older = true;
      fade = 1 - age / FADE;
      this.trailStale = true;
      this.keys.clear();
    }
    dabs.mesh.count = n;
    dabs.mesh.instanceMatrix.needsUpdate = true;
    dabs.state.needsUpdate = true;
  }

  private scuff(frame: FrameUpdate, step: number): void {
    const wear = this.wear;
    if (!wear) return;
    this.wornAt = this.wall;
    this.wornStale = false;
    this.spots.clear();
    let n = 0;
    for (let age = 0; age < WORN && step - age >= 0; age++) {
      const f = frame.frameAt(step - age);
      if (f.step !== step - age) {
        this.wornStale = true;
        break;
      }
      const key = f.words[SCENTPUCK_X]! * 4294967296 + f.words[SCENTPUCK_Y]!;
      let spot = this.spots.get(key);
      if (spot === undefined) {
        if (n === WORN_SPOTS) continue;
        spot = n++;
        this.spots.set(key, spot);
        this.wornX[spot] = at(f, SCENTPUCK_X);
        this.wornY[spot] = at(f, SCENTPUCK_Y);
        this.wornW[spot] = 0;
      }
      this.wornW[spot]! += Math.exp(-age / WORN_TAU);
    }
    for (let k = 0; k < n; k++) {
      wear.mesh.setMatrixAt(k, this.matrix.makeTranslation(this.place(this.wornX[k]!, this.wornY[k]!, this.spot)));
      wear.state.setXYZW(k, 0.5 * (1 - Math.exp(-this.wornW[k]! / 8)), 0, 0, 0);
    }
    wear.mesh.count = n;
    wear.mesh.instanceMatrix.needsUpdate = true;
    wear.state.needsUpdate = true;
  }

  private record(frame: FrameUpdate, step: number): void {
    const cur = frame.frameAt(step), ledger = this.ledger;
    if (cur.step !== step) return;
    ledger.step = step;
    ledger.x = at(cur, SCENTPUCK_X);
    ledger.y = at(cur, SCENTPUCK_Y);
    ledger.scent = cur.observations[SCENTPUCK_SCENT]!;
    ledger.rival = cur.observations[SCENTPUCK_RIVAL_SCENT]!;
    ledger.distance = Math.hypot(ledger.x - at(cur, SCENTPUCK_LURE_X), ledger.y - at(cur, SCENTPUCK_LURE_Y));
    ledger.reward = cur.reward;
    const before = frame.frameAt(step - 1);
    ledger.action = step < 1 || before.step !== step - 1 ? '—' : `${(Math.trunc(cur.actions[0]!) >>> 0) % this.moves} · ${moveOf(before, cur)}`;
    if (this.wall - ledger.looked > 1) {
      this.hunt = Math.max(this.hunt, lastEvent(frame.eventLog, 'goal', step) ?? 0);
      ledger.looked = this.wall;
    }
  }

  private count(log: EventStream, upTo: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || upTo < tally.through) Object.assign(tally, { log, count: log.count, through: -1, catches: 0 });
    if (upTo === tally.through) return;
    tally.catches += countEvents(log, 'goal', tally.through, upTo);
    tally.through = upTo;
  }

  private pose(frame: FrameUpdate, step: number): void {
    const cur = frame.frameAt(step), next = frame.frameAt(step + 1);
    if (cur.step !== step) return;
    const live = next.step === step + 1, t = live ? smoothstep01(clamp01(frame.position - step)) : 0;
    const x0 = at(cur, SCENTPUCK_X), y0 = at(cur, SCENTPUCK_Y);
    this.shown.x = live ? x0 + (at(next, SCENTPUCK_X) - x0) * t : x0;
    this.shown.y = live ? y0 + (at(next, SCENTPUCK_Y) - y0) * t : y0;
    const toward = live ? heading(cur, next) : null;
    if (toward !== null) this.angle = frame.scrubbed ? toward : this.angle + wrapAngle(toward - this.angle) * (1 - Math.exp(-TURN * frame.dtReal));

    const age = (frame.position - this.caught) / this.controlHz, fresh = age >= 0 && age < RECENT;
    const gulp = fresh ? clamp01(age / GULP) : 1, grow = fresh ? sprout((age - GULP * 0.6) / SPROUT) : 1, ring = fresh ? clamp01((age - GULP * 0.6) / PING) : 0;
    const flash = fresh ? Math.max(0, 1 - age / FLASH) ** 2 : 0;
    const breath = Math.sin(this.wall * 2.1), scent = cur.observations[SCENTPUCK_SCENT]!, rival = this.competing ? cur.observations[SCENTPUCK_RIVAL_SCENT]! : 0;

    this.place(this.shown.x, this.shown.y, this.spot);
    for (const mesh of [this.puckMesh, this.leds]) mesh?.position.copy(this.spot);
    this.puck.heading.value.set(Math.cos(this.angle), Math.sin(this.angle));
    this.puck.scent.value = scent;
    this.puck.rival.value = rival;
    this.puck.sniff.value = Math.exp(-(this.wall - this.sniffAt) / SNIFF) * (0.4 + clamp01(scent));
    this.puck.flash.value = flash;
    this.ledScent.value = LED_LIGHT * (clamp01(scent) + 0.4 * flash);
    this.ledRival.value = LED_LIGHT * clamp01(rival);

    const lure = this.lureMesh, lamp = this.lureLamp, ghost = this.gulpMesh;
    if (lure && lamp && ghost) {
      this.place(at(cur, SCENTPUCK_LURE_X), at(cur, SCENTPUCK_LURE_Y), lure.position);
      lamp.position.copy(lure.position);
      lure.visible = lamp.visible = grow > 0.02;
      lure.scale.setScalar(Math.max(0.02, grow));
      lamp.scale.setScalar(Math.max(0.02, grow));
      this.lure.glow.value = 0.5 + 0.25 * breath;
      this.lure.ping.value = ring < 1 ? ring : 0;
      this.lureShine.value = LURE_LIGHT * (1 + 0.12 * breath) * grow;
      this.pools.lure.value.set(lure.position.x, -lure.position.z, LURE_POOL * (1 + 0.08 * breath) * grow, 0);
      ghost.visible = gulp < 1;
      if (ghost.visible) {
        const k = smoothstep01(gulp);
        this.place(this.lureFrom.x + (this.shown.x - this.lureFrom.x) * k, this.lureFrom.y + (this.shown.y - this.lureFrom.y) * k, ghost.position);
        ghost.scale.setScalar(1 - 0.8 * k);
        this.gulp.glow.value = 1 + 2 * k;
        this.gulp.fade.value = 1 - k * k;
      }
    }

    const foe = this.rivalMesh, glow = this.rivalLamp, flee = this.fleeMesh;
    if (foe && glow && flee) {
      this.place(at(cur, SCENTPUCK_RIVAL_X), at(cur, SCENTPUCK_RIVAL_Y), foe.position);
      glow.position.copy(foe.position);
      foe.visible = glow.visible = grow > 0.02;
      foe.scale.setScalar(Math.max(0.02, grow));
      glow.scale.setScalar(Math.max(0.02, grow));
      this.rival.glow.value = 0.5 + 0.25 * Math.sin(this.wall * 2.7 + 1.3);
      this.rival.ping.value = ring < 1 ? ring : 0;
      this.rivalShine.value = RIVAL_LIGHT * (1 + 0.1 * Math.sin(this.wall * 2.7 + 1.3)) * grow;
      this.pools.rival.value.set(foe.position.x, -foe.position.z, RIVAL_POOL * (1 + 0.06 * Math.sin(this.wall * 2.7 + 1.3)) * grow, 0);
      flee.visible = gulp < 1;
      if (flee.visible) {
        this.place(this.rivalFrom.x, this.rivalFrom.y, flee.position);
        flee.scale.setScalar(1 + 0.3 * gulp);
        this.flee.fade.value = 1 - smoothstep01(gulp);
      }
    }
  }
}

export default (() => new ScentPuckScene()) satisfies SceneFactory;
