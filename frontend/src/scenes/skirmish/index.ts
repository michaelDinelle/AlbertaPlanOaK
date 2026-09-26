import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { hash2f } from '@kit/rng';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import {
  ATTACK, CYCLE, MOVES, NONE, RULE, SKW, STEP_X, STEP_Y, hpOf, indexOf, keyOf, kindOf, nowOf, outcomeOf, shieldOf, skirmishShape, slotOf, targetOf, waitOf, xOf, yOf,
  type Outcome, type SkirmishShape,
} from '@sim/envs/skirmish';
import type { EnvFrame } from '@sim/types';
import { formatCount } from '../../format';
import {
  ARRIVE, FX, HUE, RING_PAD, STREAK, agentRingMaterial, barGeometry, barMaterial, flat, focusMaterial, fxGeometry, fxMaterial, groundMaterial, lampBank, panelMaterial, radiusOf, reticleMaterial,
  rowGeometry, rowMaterial, shadowMaterial, streakGeometry, streakMaterial, unitGeometry, unitMaterial, wreckGeometry, wreckMaterial,
  type AgentRing, type Focus, type Ground, type Reticle,
} from './art';

const NAMES = ['strider', 'blade', 'titan', 'trooper', 'breacher', 'medic', 'skitter', 'spitter', 'burster'] as const;
const LOOKS = ['bolt', 'blade', 'lance', 'rifle', 'shell', 'mend', 'claw', 'spine', 'burst'] as const;
const STREAK_OF: Record<(typeof LOOKS)[number], number> = { bolt: STREAK.BOLT, blade: -1, lance: STREAK.BEAM, rifle: STREAK.TRACER, shell: STREAK.SHELL, mend: -1, claw: -1, spine: STREAK.SPINE, burst: -1 };
const KEY_NAMES = ['↑ move', '↓ move', '← move', '→ move', 'attack', 'cycle'] as const;
const RESULT: Record<Outcome, string> = { won: 'won', lost: 'lost', drawn: 'drawn', timeout: 'timed out' };
const AIR = 5;

const GAP = 1.1;
const PANEL = 7.4;
const ROW_MAX = 2.3;
const HEADER = 0.9;
const WINDOW = 1.0;
const HIT = 0.22;
const WARP = 0.55;
const FADE = 0.8;
const TURN = 12;
const LIFE = { muzzle: 0.09, impact: 0.26, slash: 0.24, frost: 0.5, splash: 0.35, boom: 0.95, acid: 0.8, heal: 0.55, streak: 0.22, mend: 0.3 };
const CACHE = 6000;

interface Shot { from: number; to: number; kind: number; team: number }
interface Death { index: number; kind: number; team: number; x: number; y: number; blast: boolean }
interface Heal { from: number; to: number; team: number }
interface Happening { shots: Shot[]; deaths: Death[]; heals: Heal[]; hits: number[]; shielded: number[]; outcome: Outcome | null }
interface Sheet { mesh: THREE.InstancedMesh; channels: THREE.InstancedBufferAttribute[]; used: number }

const EMPTY: Happening = { shots: [], deaths: [], heals: [], hits: [], shielded: [], outcome: null };
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const wrap = (angle: number) => Math.atan2(Math.sin(angle), Math.cos(angle));
const headingOf = (dx: number, dy: number) => Math.atan2(-dy, dx);

class SkirmishScene extends Stage2DScene {
  post: ScenePost = { bloom: 0.42, fog: { from: ['night', 'fog'] } };
  controlHz = 32;
  protected readonly stageName = 'skirmish';

  private shape!: SkirmishShape;
  private accent = 0xe8866e;
  private half = 16;
  private count = 0;
  private radii: number[] = [];
  private maxRange = 1;

  private ground: Ground | null = null;
  private wrecks: Sheet | null = null;
  private shadows: Sheet | null = null;
  private bodies: Sheet | null = null;
  private icons: Sheet | null = null;
  private streaks: Sheet | null = null;
  private effects: Sheet | null = null;
  private bars: Sheet | null = null;
  private rosterBars: Sheet | null = null;
  private rows: Sheet | null = null;
  private lamps: THREE.InstancedMesh | null = null;
  private lampCount = 0;
  private ring: AgentRing | null = null;
  private ringMesh: THREE.Mesh | null = null;
  private reticle: Reticle | null = null;
  private reticleMesh: THREE.Mesh | null = null;
  private focus: Focus | null = null;
  private focusMesh: THREE.Mesh | null = null;
  private readonly clock = uniform(0);
  private readonly owned: { dispose(): void }[] = [];

  private readonly cache = new Map<number, Happening>();
  private px = new Float32Array(0);
  private py = new Float32Array(0);
  private heading = new Float32Array(0);
  private hit = new Float32Array(0);
  private shieldHit = new Float32Array(0);
  private deathAge = new Float32Array(0);
  private phaseFrom: EnvFrame[] = [];
  private phaseTo: EnvFrame[] = [];
  private phaseT = new Float32Array(0);
  private battle = -1;
  private step = -1;
  private lastTime = 0;
  private lastDone = -1;
  private lastText = 'none yet';

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly color = new THREE.Color();
  private readonly ledger = { you: '—', key: '—', target: '—', focus: 'none', forces: '—', note: '', clock: 0 };

  private legendFor(): LegendSpec {
    const ledger = this.ledger;
    return {
      title: 'SKIRMISH',
      rows: [
        { key: 'you', label: 'you', tint: this.accent, value: () => ledger.you },
        { key: 'key', label: 'key', tint: 0xdcf4ff, value: () => ledger.key },
        { key: 'target', label: 'target', tint: this.accent, value: () => ledger.target },
        { key: 'focus', label: 'focus', tint: HUE.focus, value: () => ledger.focus },
        { key: 'forces', label: 'forces', tint: 0xf5f8ff, value: () => ledger.forces },
        { key: 'last', label: 'last battle', tint: TOKENS.celebrationGold, value: () => this.lastText },
      ],
      note: () => ledger.note,
      bar: () => ledger.clock,
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, shape = skirmishShape(env);
    if (shape.kinds.length !== NAMES.length || shape.perRace !== 3) throw new Error(`${env.envId} lists ${shape.kinds.length} unit kinds in races of ${shape.perRace}; the scene draws nine in three races of three`);
    shape.kinds.forEach((kind, t) => {
      const look = LOOKS[t]!, melee = kind.range <= 2 && !kind.suicide;
      if ((look === 'blade' || look === 'claw') !== melee || (look === 'lance') !== (kind.splash > 0 && !kind.suicide) || (look === 'shell') !== (kind.slow > 0) || (look === 'mend') !== (kind.heal > 0) || (look === 'burst') !== kind.suicide)
        throw new Error(`${env.envId} unit kind ${t} no longer fights like a ${NAMES[t]}`);
    });
    this.shape = shape;
    this.accent = this.accentOf(ctx);
    this.half = shape.arena / 2;
    this.count = shape.allies + shape.enemies;
    this.controlHz = shape.slices / shape.tick;
    this.radii = shape.kinds.map((kind) => radiusOf(kind.hp, kind.shield));
    this.maxRange = Math.max(...shape.kinds.map((kind) => kind.range));
    this.px = new Float32Array(this.count);
    this.py = new Float32Array(this.count);
    this.heading = new Float32Array(this.count);
    this.hit = new Float32Array(this.count);
    this.shieldHit = new Float32Array(this.count);
    this.deathAge = new Float32Array(this.count);
    this.phaseT = new Float32Array(shape.slices);
    this.legend = this.legendFor();
    const x = this.half + GAP + PANEL + 0.4, z = this.half + 2.4;
    return new THREE.Box2(new THREE.Vector2(-x, -z), new THREE.Vector2(x, z));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xc4cce0).multiplyScalar(0.8);
  }

  protected override poolOptions() {
    return { burstSize: 40 };
  }

  private sheet(mesh: THREE.InstancedMesh, channels: THREE.InstancedBufferAttribute[], order: number): Sheet {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    this.owned.push(mesh);
    return { mesh, channels, used: 0 };
  }

  private single(geometry: THREE.BufferGeometry, material: THREE.Material, order: number): THREE.Mesh {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.visible = false;
    this.scene.add(mesh);
    this.owned.push(geometry, material);
    return mesh;
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const shape = this.shape, half = this.half, accent = this.accent, lit = light.lit, n = this.count;
    this.scene.background = new THREE.Color(HUE.void);
    pool.group.traverse((object) => { object.renderOrder = 30; });

    const ground = groundMaterial(shape, lit, accent);
    this.ground = ground;
    this.single(flat(4 * half + 2 * (GAP + PANEL) + 20, 2 * half + 20), ground.material, 0).visible = true;

    for (const side of [0, 1]) {
      const height = HEADER + this.rowHeight(side) * (side ? shape.enemies : shape.allies) + 0.35;
      const panel = this.single(flat(PANEL, height), panelMaterial(side ? HUE.theirs : accent, PANEL, height), 10);
      panel.position.set((side ? 1 : -1) * (half + GAP + PANEL / 2), 0, -half + height / 2);
      panel.visible = true;
    }

    const wreck = wreckGeometry(2 * n);
    this.owned.push(wreck.geometry);
    this.wrecks = this.sheet(instancedMesh(wreck.geometry, this.own(wreckMaterial(accent, lit)), 2 * n), [wreck.state, wreck.fade], 1);

    const units = unitGeometry(2 * n), body = this.own(unitMaterial(accent, this.clock));
    this.owned.push(units.geometry);
    this.shadows = this.sheet(instancedMesh(units.geometry, this.own(shadowMaterial()), 2 * n), [units.body, units.life, units.mark], 2);
    this.bodies = this.sheet(instancedMesh(units.geometry, body, 2 * n), [units.body, units.life, units.mark], 4);

    const size = 2 * (this.maxRange + RING_PAD);
    this.ring = agentRingMaterial(accent, this.clock, size);
    this.ringMesh = this.single(flat(size, size), this.ring.material, 3);
    this.reticle = reticleMaterial(accent, this.clock, 4);
    this.reticleMesh = this.single(flat(4, 4), this.reticle.material, 9);
    this.focus = focusMaterial(this.clock, 5);
    this.focusMesh = this.single(flat(5, 5), this.focus.material, 9);

    const streak = streakGeometry(4 * n);
    this.owned.push(streak.geometry);
    this.streaks = this.sheet(instancedMesh(streak.geometry, this.own(streakMaterial(this.clock)), 4 * n), [streak.state, streak.tint], 6);
    const fx = fxGeometry(8 * n);
    this.owned.push(fx.geometry);
    this.effects = this.sheet(instancedMesh(fx.geometry, this.own(fxMaterial()), 8 * n), [fx.state, fx.tint], 7);

    const bar = barGeometry(n), barLook = this.own(barMaterial());
    this.owned.push(bar.geometry);
    this.bars = this.sheet(instancedMesh(bar.geometry, barLook, n), [bar.state, bar.size], 8);

    const row = rowGeometry(n);
    this.owned.push(row.geometry);
    this.rows = this.sheet(instancedMesh(row.geometry, this.own(rowMaterial(accent, accent)), n), [row.state, row.size], 11);
    const roster = unitGeometry(n);
    this.owned.push(roster.geometry);
    this.icons = this.sheet(instancedMesh(roster.geometry, body, n), [roster.body, roster.life, roster.mark], 12);
    const rosterBar = barGeometry(n);
    this.owned.push(rosterBar.geometry);
    this.rosterBars = this.sheet(instancedMesh(rosterBar.geometry, barLook, n), [rosterBar.state, rosterBar.size], 13);

    const lamps = lampBank(light, 16 * n);
    this.owned.push(lamps.geometry, lamps.material as THREE.Material, lamps);
    this.scene.add(lamps);
    this.lamps = lamps;
    void ctx;
  }

  private rowHeight(side: number): number {
    return Math.min(ROW_MAX, (2 * this.half - HEADER - 0.35) / (side ? this.shape.enemies : this.shape.allies));
  }

  private own<T extends { dispose(): void }>(item: T): T {
    this.owned.push(item);
    return item;
  }

  protected override drop(): void {
    for (const item of this.owned) item.dispose();
    this.owned.length = 0;
    this.cache.clear();
  }

  private frameOr(frame: FrameUpdate, step: number, fallback: EnvFrame): EnvFrame {
    const found = frame.frameAt(step);
    return found.step === step ? found : fallback;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.clock.value = time;
    const i = frame.state.step, now = frame.frameAt(i);
    if (now.step !== i || !this.ground) return;
    const shape = this.shape, p = Math.min(i + 0.999, Math.max(i, frame.position)), start = i - nowOf(now, shape.slices), battle = now.words[SKW.BATTLE]!;
    const fresh = frame.scrubbed || battle !== this.battle || i < this.step;
    const dt = fresh ? 0 : Math.min(0.1, Math.max(0, time - this.lastTime));
    this.battle = battle;
    this.step = i;
    this.lastTime = time;
    this.ground.surround.value = now.words[SKW.START]!;
    const key = keyOf(this.frameOr(frame, i + 1, now));
    this.place(frame, now, p, start);
    this.aim(now, key, dt, fresh);
    this.beginSheets();
    this.gather(frame, now, i, p, start);
    this.drawUnits(frame, now, p, start);
    this.drawMarkers(now, key, p, start);
    this.drawRoster(now);
    this.commitSheets();
    if (!frame.scrubbed) this.celebrate(frame, pool, time);
    this.record(frame, now, key);
  }

  private place(frame: FrameUpdate, now: EnvFrame, p: number, start: number): void {
    const shape = this.shape, slices = shape.slices, i = now.step, battle = now.words[SKW.BATTLE];
    for (let phase = 0; phase < slices; phase++) {
      const first = start + phase + 1, a = i < first ? start : first + Math.floor((i - first) / slices) * slices, b = i < first ? first : a + slices;
      const from = this.frameOr(frame, a, now), to = frame.frameAt(b);
      const ok = from.step === a && to.step === b && to.done === 0 && to.words[SKW.BATTLE] === battle;
      this.phaseFrom[phase] = from;
      this.phaseTo[phase] = ok ? to : from;
      this.phaseT[phase] = ok ? clamp01((p - a) / (b - a)) : 0;
    }
    const pose = frame.state.pose, next = frame.frameAt(i + 1), cut = next.step === i + 1 && next.done !== 0;
    this.px[0] = cut ? xOf(now, 0) : pose[0]!;
    this.py[0] = cut ? yOf(now, 0) : pose[1]!;
    for (let k = 1; k < this.count; k++) {
      const phase = slotOf(k, shape.allies) % slices, from = this.phaseFrom[phase]!, to = this.phaseTo[phase]!, t = this.phaseT[phase]!;
      this.px[k] = lerp(xOf(from, k), xOf(to, k), t);
      this.py[k] = lerp(yOf(from, k), yOf(to, k), t);
    }
  }

  private aim(now: EnvFrame, key: number, dt: number, fresh: boolean): void {
    const shape = this.shape, allies = shape.allies, select = indexOf(now.words[SKW.SELECT]!, allies);
    let ox = 0, oy = 0, tx = 0, ty = 0, on = 0, tn = 0;
    for (let k = 0; k < this.count; k++) {
      if (hpOf(now, k) <= 0) continue;
      if (k < allies) { ox += this.px[k]!; oy += this.py[k]!; on++; } else { tx += this.px[k]!; ty += this.py[k]!; tn++; }
    }
    for (let k = 0; k < this.count; k++) {
      if (hpOf(now, k) <= 0) continue;
      const team = k < allies ? 0 : 1, target = targetOf(now, k), aimed = target === NONE ? -1 : indexOf(target, allies);
      let desired = Number.NaN;
      if (k === 0 && key < MOVES) desired = headingOf(STEP_X[key]!, STEP_Y[key]!);
      else if (k === 0 && select < this.count && hpOf(now, select) > 0) desired = headingOf(this.px[select]! - this.px[0]!, this.py[select]! - this.py[0]!);
      else if (aimed >= 0 && aimed < this.count && hpOf(now, aimed) > 0 && aimed !== k) desired = headingOf(this.px[aimed]! - this.px[k]!, this.py[aimed]! - this.py[k]!);
      else {
        const phase = slotOf(k, allies) % shape.slices, from = this.phaseFrom[phase]!, to = this.phaseTo[phase]!, dx = xOf(to, k) - xOf(from, k), dy = yOf(to, k) - yOf(from, k);
        if (dx * dx + dy * dy > 1e-6) desired = headingOf(dx, dy);
      }
      if (fresh && Number.isNaN(desired)) {
        const cx = team ? ox / Math.max(1, on) : tx / Math.max(1, tn), cy = team ? oy / Math.max(1, on) : ty / Math.max(1, tn);
        desired = (team ? on : tn) > 0 ? headingOf(cx - this.px[k]!, cy - this.py[k]!) : team ? Math.PI : 0;
      }
      if (Number.isNaN(desired)) continue;
      this.heading[k] = fresh ? desired : this.heading[k]! + wrap(desired - this.heading[k]!) * Math.min(1, dt * TURN);
    }
  }

  private happening(frame: FrameUpdate, s: number): Happening | null {
    const cached = this.cache.get(s);
    if (cached) return cached;
    if (s < 1) return EMPTY;
    const after = frame.frameAt(s), before = frame.frameAt(s - 1);
    if (after.step !== s || before.step !== s - 1) return null;
    const shape = this.shape, allies = shape.allies, n = this.count, found: Happening = { shots: [], deaths: [], heals: [], hits: [], shielded: [], outcome: null };
    if (after.done) {
      const outcome = outcomeOf(before, after);
      found.outcome = outcome;
      for (let k = 0; k < n; k++) {
        const ours = k < allies, gone = outcome === 'drawn' || (outcome === 'won' && !ours) || (outcome === 'lost' && ours);
        if (gone && hpOf(before, k) > 0) found.deaths.push({ index: k, kind: kindOf(before, k), team: ours ? 0 : 1, x: xOf(before, k), y: yOf(before, k), blast: false });
      }
    } else {
      const pre = before.words[SKW.SLICE]!;
      for (let k = 0; k < n; k++) {
        if (hpOf(before, k) <= 0) continue;
        const kind = kindOf(after, k), team = k < allies ? 0 : 1, hp = hpOf(after, k), was = hpOf(before, k), shield = shieldOf(after, k), had = shieldOf(before, k);
        if (hp + shield < was + had) {
          found.hits.push(k);
          if (shield < had) found.shielded.push(k);
        }
        if (waitOf(before, k) === 0 && waitOf(after, k) > 0) {
          const target = targetOf(after, k);
          if (target !== NONE) found.shots.push({ from: k, to: indexOf(target, allies), kind, team });
        }
        if (hp === 0) {
          const kindInfo = shape.kinds[kind]!, x = xOf(after, k), y = yOf(after, k), slot = slotOf(k, allies);
          let blast = false;
          if (kindInfo.suicide && (slot === 0 || slot % shape.slices === pre)) {
            for (let v = team ? 0 : allies; v < (team ? allies : n) && !blast; v++) {
              const dx = xOf(after, v) - x, dy = yOf(after, v) - y;
              blast = dx * dx + dy * dy <= kindInfo.splash * kindInfo.splash + 1e-3 && hpOf(after, v) + shieldOf(after, v) < hpOf(before, v) + shieldOf(before, v);
            }
          }
          found.deaths.push({ index: k, kind, team, x, y, blast });
        }
        if (hp > was) {
          const slot = slotOf(k, allies);
          for (let h = team ? allies : 0; h < (team ? n : allies); h++) {
            if (h !== k && shape.kinds[kindOf(after, h)]!.heal > 0 && targetOf(after, h) === slot && hpOf(before, h) > 0) {
              found.heals.push({ from: h, to: k, team });
              break;
            }
          }
        }
      }
    }
    if (this.cache.size >= CACHE) for (const old of this.cache.keys()) { this.cache.delete(old); if (this.cache.size < CACHE / 2) break; }
    this.cache.set(s, found);
    return found;
  }

  private beginSheets(): void {
    for (const sheet of [this.wrecks, this.shadows, this.bodies, this.icons, this.streaks, this.effects, this.bars, this.rosterBars, this.rows]) if (sheet) sheet.used = 0;
    this.lampCount = 0;
  }

  private commitSheets(): void {
    for (const sheet of [this.wrecks, this.shadows, this.bodies, this.icons, this.streaks, this.effects, this.bars, this.rosterBars, this.rows]) {
      if (!sheet) continue;
      sheet.mesh.count = sheet.used;
      sheet.mesh.instanceMatrix.needsUpdate = true;
      for (const channel of sheet.channels) channel.needsUpdate = true;
    }
    if (this.lamps) {
      this.lamps.count = this.lampCount;
      this.lamps.instanceMatrix.needsUpdate = true;
      if (this.lamps.instanceColor) this.lamps.instanceColor.needsUpdate = true;
    }
  }

  private put(sheet: Sheet, x: number, y: number, angle: number, sx: number, sz: number): number {
    if (sheet.used >= sheet.mesh.instanceMatrix.count) throw new Error(`Skirmish ${sheet.mesh.name || 'sheet'} holds ${sheet.mesh.instanceMatrix.count} sprites`);
    const i = sheet.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    sheet.mesh.setMatrixAt(i, this.matrix.compose(this.at.set(x - this.half, 0, y - this.half), this.quaternion, this.scale.set(sx, 1, sz)));
    return i;
  }

  private lamp(x: number, y: number, radius: number, tint: number, power: number): void {
    const lamps = this.lamps;
    if (!lamps || power <= 0.01 || this.lampCount >= lamps.instanceMatrix.count) return;
    const i = this.lampCount++;
    this.quaternion.identity();
    lamps.setMatrixAt(i, this.matrix.compose(this.at.set(x - this.half, 0.01, y - this.half), this.quaternion, this.scale.set(radius, 1, radius)));
    lamps.setColorAt(i, this.color.set(tint).multiplyScalar(power));
  }

  private tintOf(team: number): number {
    return team ? HUE.theirs : this.accent;
  }

  private fx(kind: number, x: number, y: number, size: number, t: number, team: number, angle = 0, seed = 0): void {
    const effects = this.effects;
    if (!effects || t < 0 || t >= 1) return;
    const i = this.put(effects, x, y, angle, size, size);
    effects.channels[0]!.setXYZW(i, kind, t, size, seed);
    this.color.set(this.tintOf(team));
    effects.channels[1]!.setXYZW(i, this.color.r, this.color.g, this.color.b, 1);
  }

  private streak(kind: number, ax: number, ay: number, bx: number, by: number, t: number, team: number, width: number): void {
    const streaks = this.streaks, dx = bx - ax, dy = by - ay, span = Math.hypot(dx, dy);
    if (!streaks || t < 0 || t >= 1 || span < 0.05) return;
    const i = this.put(streaks, (ax + bx) / 2, (ay + by) / 2, headingOf(dx, dy), span, width);
    streaks.channels[0]!.setXYZW(i, kind, t, span, width);
    this.color.set(this.tintOf(team));
    streaks.channels[1]!.setXYZW(i, this.color.r, this.color.g, this.color.b, 1);
  }

  private gather(frame: FrameUpdate, now: EnvFrame, i: number, p: number, start: number): void {
    const shape = this.shape, hz = this.controlHz, kinds = shape.kinds;
    this.hit.fill(0);
    this.shieldHit.fill(0);
    this.deathAge.fill(Number.POSITIVE_INFINITY);
    const reach = Math.ceil(WINDOW * hz);
    for (let s = i; s > i - reach && s > start; s--) {
      const found = this.happening(frame, s);
      if (!found) continue;
      const age = (p - s) / hz;
      for (const k of found.hits) this.hit[k] = Math.max(this.hit[k]!, 1 - age / HIT);
      for (const k of found.shielded) this.shieldHit[k] = Math.max(this.shieldHit[k]!, 1 - age / HIT);
      for (const death of found.deaths) {
        this.deathAge[death.index] = Math.min(this.deathAge[death.index]!, age);
        this.blast(death, age, hz, s);
      }
      for (const shot of found.shots) {
        const kind = kinds[shot.kind]!, look = LOOKS[shot.kind]!, ax = this.px[shot.from]!, ay = this.py[shot.from]!, bx = this.px[shot.to]!, by = this.py[shot.to]!;
        const angle = headingOf(ax - bx, ay - by), r = this.radii[kindOf(now, shot.to)]!;
        if (look === 'blade' || look === 'claw') {
          this.fx(FX.SLASH, bx, by, r + 0.45, age / LIFE.slash, shot.team, angle, s % 7);
          this.lamp(bx, by, 0.5, this.tintOf(shot.team), 5 * (1 - age / LIFE.slash));
          continue;
        }
        const dx = bx - ax, dy = by - ay, span = Math.hypot(dx, dy) || 1, mr = this.radii[shot.kind]! * 1.05, mx = ax + (dx / span) * mr, my = ay + (dy / span) * mr;
        const streak = STREAK_OF[look], flight = streak === STREAK.TRACER || streak === STREAK.BEAM ? 0 : ARRIVE * LIFE.streak, landed = age - flight;
        this.streak(streak, mx, my, bx, by, age / (look === 'lance' ? LIFE.streak * 1.2 : LIFE.streak), shot.team, look === 'lance' ? 0.5 : 0.36);
        this.fx(FX.MUZZLE, mx, my, 0.34, age / LIFE.muzzle, shot.team, headingOf(dx, dy));
        this.lamp(mx, my, 0.35, this.tintOf(shot.team), 7 * (1 - age / LIFE.muzzle));
        if (landed < 0) continue;
        if (look === 'shell') this.fx(FX.FROST, bx, by, r + 0.5, landed / LIFE.frost, shot.team, 0, s % 5);
        else if (look === 'lance') this.fx(FX.SPLASH, bx, by, kind.splash, landed / LIFE.splash, shot.team);
        this.fx(FX.IMPACT, bx, by, 0.5, landed / LIFE.impact, shot.team, 0, s % 3);
        this.lamp(bx, by, 0.45, this.tintOf(shot.team), 4 * (1 - landed / LIFE.impact));
      }
      for (const heal of found.heals) {
        const ax = this.px[heal.from]!, ay = this.py[heal.from]!, bx = this.px[heal.to]!, by = this.py[heal.to]!;
        this.streak(STREAK.HEAL, ax, ay, bx, by, age / (shape.tick + LIFE.mend), heal.team, 0.4);
        this.fx(FX.HEAL, bx, by, 0.55, age / LIFE.heal, heal.team, 0, s % 4);
        this.lamp(bx, by, 0.5, HUE.heal, 1.5 * (1 - age / LIFE.heal));
      }
    }
  }

  private blast(death: Death, age: number, hz: number, s: number): void {
    const kind = this.shape.kinds[death.kind]!, r = this.radii[death.kind]!, tint = this.tintOf(death.team);
    if (death.blast) {
      this.fx(FX.ACID, death.x, death.y, kind.splash, age / LIFE.acid, death.team, 0, (s % 11) / 11);
      this.lamp(death.x, death.y, 1.2, HUE.acid, 8 * (1 - age / LIFE.acid) ** 2);
    }
    this.fx(FX.BOOM, death.x, death.y, 1.5 * r + 0.6, age / LIFE.boom, death.team, 0, (s % 13) / 13);
    this.lamp(death.x, death.y, 0.9, HUE.fire, 14 * Math.max(0, 1 - age / LIFE.boom) ** 2);
    this.lamp(death.x, death.y, 0.5, tint, 4 * Math.max(0, 1 - age / LIFE.boom));
    void hz;
  }

  private unit(sheet: Sheet, shadow: Sheet | null, k: number, x: number, y: number, kind: number, team: number, heading: number, size: number, hp: number, shield: number, hit: number, alpha: number, slowed: number, agent: number): void {
    const r = this.radii[kind]!, info = this.shape.kinds[kind]!;
    const i = this.put(sheet, x, y, 0, size, size);
    sheet.channels[0]!.setXYZW(i, kind, team, heading, r);
    sheet.channels[1]!.setXYZW(i, hp / info.hp, info.shield > 0 ? shield / info.shield : 0, hit, alpha);
    sheet.channels[2]!.setXYZW(i, slowed, agent, 0, k);
    if (!shadow) return;
    const j = this.put(shadow, x, y, 0, size, size);
    if (j !== i) throw new Error('Skirmish shadows fell out of step with their units');
  }

  private drawUnits(frame: FrameUpdate, now: EnvFrame, p: number, start: number): void {
    const shape = this.shape, allies = shape.allies, hz = this.controlHz, bodies = this.bodies!, shadows = this.shadows!, wrecks = this.wrecks!;
    const since = (p - start) / hz, battleSeed = now.words[SKW.BATTLE]!;
    for (let k = 0; k < this.count; k++) {
      if (hpOf(now, k) > 0) continue;
      const kind = kindOf(now, k), w = this.put(wrecks, xOf(now, k), yOf(now, k), 0, 0.6 + this.radii[kind]!, 0.6 + this.radii[kind]!);
      wrecks.channels[0]!.setXYZW(w, Math.floor(kind / 3), k < allies ? 0 : 1, this.deathAge[k]!, hash2f(battleSeed, k));
      wrecks.channels[1]!.setXYZW(w, 1, 0, 0, 0);
    }
    if (start > 0 && since < FADE) this.ghosts(frame, start, since);
    const order: number[] = [];
    for (let k = 1; k < this.count; k++) if (hpOf(now, k) > 0 && kindOf(now, k) !== AIR) order.push(k);
    if (hpOf(now, 0) > 0) order.push(0);
    for (let k = 1; k < this.count; k++) if (hpOf(now, k) > 0 && kindOf(now, k) === AIR) order.push(k);
    const warp = clamp01(since / WARP), grow = 0.55 + 0.45 * smoothstep01(warp);
    for (const k of order) {
      const kind = kindOf(now, k), team = k < allies ? 0 : 1, word = now.view[3 * k + 2]!, slowed = (word >>> 10) & 1;
      this.unit(bodies, shadows, k, this.px[k]!, this.py[k]!, kind, team, this.heading[k]!, grow, hpOf(now, k), shieldOf(now, k), this.hit[k]!, smoothstep01(warp * 1.6), slowed, k === 0 ? 1 : 0);
      const r = this.radii[kind]!, info = shape.kinds[kind]!;
      const shine = kind < 3 && info.shield > 0 ? shieldOf(now, k) / info.shield : 0;
      this.lamp(this.px[k]!, this.py[k]!, r * 1.1, this.tintOf(team), (2 + 1.2 * shine + 4 * this.shieldHit[k]!) * smoothstep01(warp));
      if (since < WARP) {
        this.fx(FX.WARP, this.px[k]!, this.py[k]!, r + 0.7, warp, team);
        this.lamp(this.px[k]!, this.py[k]!, r + 0.3, this.tintOf(team), 3 * (1 - warp));
      }
      if (kind === AIR) this.lamp(this.px[k]!, this.py[k]!, 0.3, HUE.heal, 0.6);
    }
    this.fieldBars(now);
  }

  private ghosts(frame: FrameUpdate, start: number, since: number): void {
    const last = frame.frameAt(start - 1), shape = this.shape, allies = shape.allies, bodies = this.bodies!, shadows = this.shadows!, wrecks = this.wrecks!;
    if (last.step !== start - 1) return;
    const ending = this.happening(frame, start), fade = 1 - since / FADE, gone = new Set(ending?.deaths.map((death) => death.index));
    for (const death of ending?.deaths ?? []) this.blast(death, since, this.controlHz, start);
    for (let k = 0; k < this.count; k++) {
      const kind = kindOf(last, k), team = k < allies ? 0 : 1, x = xOf(last, k), y = yOf(last, k);
      if (hpOf(last, k) <= 0 || gone.has(k)) {
        const w = this.put(wrecks, x, y, 0, 0.6 + this.radii[kind]!, 0.6 + this.radii[kind]!);
        wrecks.channels[0]!.setXYZW(w, Math.floor(kind / 3), team, gone.has(k) ? since : 99, hash2f(last.words[SKW.BATTLE]!, k));
        wrecks.channels[1]!.setXYZW(w, fade, 0, 0, 0);
        continue;
      }
      const target = targetOf(last, k), aimed = target === NONE ? -1 : indexOf(target, allies);
      const heading = aimed >= 0 && aimed < this.count ? headingOf(xOf(last, aimed) - x, yOf(last, aimed) - y) : team ? Math.PI : 0;
      this.unit(bodies, shadows, k, x, y, kind, team, heading, 1, hpOf(last, k), shieldOf(last, k), 0, fade * fade, 0, 0);
    }
  }

  private fieldBars(now: EnvFrame): void {
    const shape = this.shape, bars = this.bars!, select = indexOf(now.words[SKW.SELECT]!, shape.allies), call = now.words[SKW.CALL]! === NONE ? -1 : indexOf(now.words[SKW.CALL]!, shape.allies);
    for (let k = 0; k < this.count; k++) {
      const hp = hpOf(now, k);
      if (hp <= 0) continue;
      const kind = kindOf(now, k), info = shape.kinds[kind]!, shield = shieldOf(now, k), r = this.radii[kind]!;
      const damaged = hp < Math.ceil(info.hp) || shield < Math.ceil(info.shield);
      if (!damaged && k !== 0 && k !== select && k !== call) continue;
      const has = info.shield > 0, W = Math.max(1.05, 2.5 * r), H = has ? 0.34 : 0.22;
      const i = this.put(bars, this.px[k]!, this.py[k]! - r - 0.42, 0, W, H);
      bars.channels[0]!.setXYZW(i, hp / info.hp, has ? shield / info.shield : -1, 0.95, 0);
      bars.channels[1]!.setXYZW(i, W, H, 0, 0);
    }
  }

  private drawMarkers(now: EnvFrame, key: number, p: number, start: number): void {
    const shape = this.shape, allies = shape.allies, ring = this.ring, ringMesh = this.ringMesh, reticle = this.reticle, reticleMesh = this.reticleMesh, focus = this.focus, focusMesh = this.focusMesh;
    if (!ring || !ringMesh || !reticle || !reticleMesh || !focus || !focusMesh) return;
    const warp = clamp01((p - start) / this.controlHz / WARP), alive = hpOf(now, 0) > 0, kind = kindOf(now, 0), obs = now.observations;
    ringMesh.visible = alive;
    if (alive) {
      ringMesh.position.set(this.px[0]! - this.half, 0, this.py[0]! - this.half);
      ring.radius.value = this.radii[kind]!;
      ring.range.value = shape.kinds[kind]!.range;
      ring.cool.value = clamp01(obs[1]!);
      ring.reach.value = obs[9]! > 0.5 ? 1 : 0;
      ring.alpha.value = smoothstep01(warp);
      ring.aimX.value = key < MOVES ? STEP_X[key]! : 0;
      ring.aimY.value = key < MOVES ? -STEP_Y[key]! : 0;
      ring.push.value = key < MOVES ? 1 : 0;
    }
    const select = indexOf(now.words[SKW.SELECT]!, allies), called = now.words[SKW.CALL]!, call = called === NONE ? -1 : indexOf(called, allies);
    const selectLive = select >= allies && select < this.count && hpOf(now, select) > 0;
    reticleMesh.visible = selectLive && alive;
    if (reticleMesh.visible) {
      reticleMesh.position.set(this.px[select]! - this.half, 0, this.py[select]! - this.half);
      reticle.radius.value = this.radii[kindOf(now, select)]!;
      reticle.jump.value = key === CYCLE ? 0.25 : 0;
      reticle.alpha.value = smoothstep01(warp);
    }
    focusMesh.visible = call >= 0 && call < this.count && hpOf(now, call) > 0;
    if (focusMesh.visible) {
      focusMesh.position.set(this.px[call]! - this.half, 0, this.py[call]! - this.half);
      focus.radius.value = this.radii[kindOf(now, call)]!;
      focus.pulse.value = key === ATTACK ? 1 : 0;
      focus.alpha.value = 1;
    }
  }

  private drawRoster(now: EnvFrame): void {
    const shape = this.shape, allies = shape.allies, rows = this.rows!, icons = this.icons!, bars = this.rosterBars!, half = this.half;
    const select = now.words[SKW.SELECT]!, call = now.words[SKW.CALL]!;
    for (const side of [0, 1]) {
      const n = side ? shape.enemies : allies, h = this.rowHeight(side), left = side ? half + GAP : -(half + GAP + PANEL);
      for (let r = 0; r < n; r++) {
        const k = side ? allies + r : r, slot = slotOf(k, allies), kind = kindOf(now, k), info = shape.kinds[kind]!, hp = hpOf(now, k), shield = shieldOf(now, k), dead = hp <= 0 ? 1 : 0;
        const cx = left + PANEL / 2, cz = -half + HEADER + (r + 0.5) * h, W = PANEL - 0.5, rowH = h - 0.12;
        const role = slot === 0 ? 1 : slot === call ? 3 : slot === select ? 2 : 0;
        const ri = this.putAt(rows, cx, cz, 0, W, rowH);
        rows.channels[0]!.setXYZW(ri, side, role, dead, this.hit[k]!);
        rows.channels[1]!.setXYZW(ri, W, rowH, 0, 0);
        const iconX = left + 0.35 + rowH * 0.55, size = Math.min(1.5, (0.42 * rowH) / (1.15 * this.radii[kind]!));
        const ii = this.putAt(icons, iconX, cz, 0, size, size);
        icons.channels[0]!.setXYZW(ii, kind, side, side ? Math.PI : 0, this.radii[kind]!);
        icons.channels[1]!.setXYZW(ii, hp / info.hp, info.shield > 0 ? shield / info.shield : 0, this.hit[k]!, dead ? 0.28 : 1);
        icons.channels[2]!.setXYZW(ii, 0, 0, 0, k);
        if (dead) continue;
        const has = info.shield > 0, barLeft = iconX + rowH * 0.62, barW = left + PANEL - 0.45 - barLeft, barH = Math.min(has ? 0.62 : 0.4, rowH * (has ? 0.5 : 0.34));
        const bi = this.putAt(bars, barLeft + barW / 2, cz, 0, barW, barH);
        bars.channels[0]!.setXYZW(bi, hp / info.hp, has ? shield / info.shield : -1, 1, 0);
        bars.channels[1]!.setXYZW(bi, barW, barH, 0, 0);
      }
    }
  }

  private putAt(sheet: Sheet, wx: number, wz: number, angle: number, sx: number, sz: number): number {
    return this.put(sheet, wx + this.half, wz + this.half, angle, sx, sz);
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    if (frame.events.length === 0) return;
    const i = frame.state.step;
    let done = i;
    while (done > i - 16 && frame.frameAt(done).step === done && frame.frameAt(done).done === 0) done--;
    const ended = frame.frameAt(done), last = frame.frameAt(done - 1);
    if (ended.step !== done || ended.done === 0 || last.step !== done - 1) return;
    const allies = this.shape.allies;
    let x = 0, y = 0, n = 0;
    for (let k = 0; k < allies; k++) if (hpOf(last, k) > 0) { x += xOf(last, k); y += yOf(last, k); n++; }
    const cx = (n ? x / n : this.half) - this.half, cz = (n ? y / n : this.half) - this.half;
    for (const event of frame.events) {
      if (event.type === 'goal') {
        pool.burst(this.at.set(cx, 38, cz), TOKENS.celebrationGold, 1.6, time);
        pool.ripple(this.at.set(cx, 0, cz), TOKENS.celebrationGold, 1.4, time);
      } else if (event.type === 'harm') pool.sigh(0.9, this.at.set(0, 0, 0), time);
      else if (event.type === 'done' && !frame.events.some((other) => other.type === 'goal' || other.type === 'harm')) pool.ripple(this.at.set(0, 0, 0), 0xf5f8ff, 1.2, time);
    }
  }

  private record(frame: FrameUpdate, now: EnvFrame, key: number): void {
    const shape = this.shape, allies = shape.allies, ledger = this.ledger, w = now.words, kind = kindOf(now, 0), hp = hpOf(now, 0), shield = shieldOf(now, 0), obs = now.observations;
    const cool = clamp01(obs[1]!);
    const info = shape.kinds[kind]!, armour = (h: number, sh: number, has: boolean) => `${h} hp${has ? ` · ${sh} shield` : ''}`;
    const reload = info.heal > 0 ? 'mends' : cool > 0 ? `reloading ${Math.round((1 - cool) * 100)}%` : 'ready';
    ledger.you = hp > 0 ? `${NAMES[kind]} · ${armour(hp, shield, info.shield > 0)} · ${reload}` : `${NAMES[kind]} · fallen`;
    ledger.key = KEY_NAMES[key]!;
    const select = indexOf(w[SKW.SELECT]!, allies), called = w[SKW.CALL]!;
    if (select >= allies && select < this.count && hpOf(now, select) > 0) {
      const d = Math.hypot(xOf(now, select) - xOf(now, 0), yOf(now, select) - yOf(now, 0));
      const sk = kindOf(now, select);
      ledger.target = `${NAMES[sk]} · ${armour(hpOf(now, select), shieldOf(now, select), shape.kinds[sk]!.shield > 0)} · ${d.toFixed(1)} away${obs[9]! > 0.5 ? ' · in range' : ''}`;
    } else ledger.target = 'none';
    const call = called === NONE ? -1 : indexOf(called, allies);
    ledger.focus = call < 0 || hpOf(now, call) <= 0 ? 'none' : `${NAMES[kindOf(now, call)]} · ${hpOf(now, call)} hp${call === select ? ' · the target' : ''}`;
    let mine = 0, theirs = 0;
    for (let k = 0; k < this.count; k++) if (hpOf(now, k) > 0) { if (k < allies) mine++; else theirs++; }
    ledger.forces = `${mine} of ${allies} vs ${theirs} of ${shape.enemies}`;
    const results = [[w[SKW.WINS]!, 'won'], [w[SKW.LOSSES]!, 'lost'], [w[SKW.DRAWN]!, 'drawn'], [w[SKW.TIMEOUTS]!, 'timed out']] as const;
    const tally = results.filter(([count], at) => at < 2 || count > 0).map(([count, name]) => `${name} ${formatCount(count)}`).join(' · ');
    ledger.note = `battle ${formatCount(w[SKW.BATTLE]! + 1)} · ${tally}${shape.league ? ` · rival ${w[SKW.MEMBER]! + 1}` : ''}`;
    ledger.clock = clamp01((w[SKW.TICK]! + w[SKW.SLICE]! / shape.slices) / shape.battleTicks);
    this.lastBattle(frame, now.step);
  }

  private lastBattle(frame: FrameUpdate, step: number): void {
    const done = lastEvent(frame.eventLog, 'done', step);
    if (done === undefined) {
      this.lastDone = -1;
      this.lastText = 'none yet';
      return;
    }
    if (done === this.lastDone) return;
    const after = frame.frameAt(done), before = frame.frameAt(done - 1);
    if (after.step !== done || before.step !== done - 1) return;
    const outcome = outcomeOf(before, after), reward = after.reward;
    this.lastDone = done;
    this.lastText = `${RESULT[outcome]} ${reward > 0 ? '+' : reward < 0 ? '−' : ''}${Math.abs(reward)} · ${formatCount(step - done)} steps ago`;
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const i = frame.state.step, now = frame.frameAt(i);
    if (now.step !== i) return;
    const shape = this.shape, allies = shape.allies, half = this.half, n = this.count;
    const live = kit.marks('units', n), fallen = kit.marks('fallen', n), flags = kit.marks('flags', 2 * n);
    const X = (k: number) => xOf(now, k) - half, Z = (k: number) => yOf(now, k) - half;
    for (let k = 0; k < n; k++) {
      const alive = hpOf(now, k) > 0, team = k < allies, color = team ? TOKENS.uiCyan : TOKENS.cueMagenta, kind = kindOf(now, k), info = shape.kinds[kind]!, r = this.radii[kind]!;
      const aim = kit.curve(`aim ${k}`, 2, color), drawn = kit.curve(`drawn ${k}`, 2, 0xffffff), hp = kit.curve(`hp ${k}`, 2, HUE.hpGood), shield = kit.curve(`shield ${k}`, 2, HUE.shield);
      if (!alive) {
        fallen.set(X(k), 0.4, Z(k), 0.14, 0x7d8594);
        continue;
      }
      live.set(X(k), 0.4, Z(k), 0.24, color);
      const target = targetOf(now, k);
      if (target !== NONE) {
        const v = indexOf(target, allies);
        aim.point(X(k), 0.3, Z(k));
        aim.point(X(v), 0.3, Z(v));
      }
      if (Math.hypot(this.px[k]! - half - X(k), this.py[k]! - half - Z(k)) > 0.01) {
        drawn.point(X(k), 0.3, Z(k));
        drawn.point(this.px[k]! - half, 0.3, this.py[k]! - half);
      }
      const top = Z(k) - r - 0.62;
      hp.point(X(k) - 0.6, 0.3, top);
      hp.point(X(k) - 0.6 + 1.2 * hpOf(now, k) / info.hp, 0.3, top);
      if (info.shield > 0 && shieldOf(now, k) > 0) {
        shield.point(X(k) - 0.6, 0.3, top - 0.14);
        shield.point(X(k) - 0.6 + 1.2 * shieldOf(now, k) / info.shield, 0.3, top - 0.14);
      }
      const word = now.view[3 * k + 2]!;
      if ((word >>> 10) & 1) flags.set(X(k) + 0.45, 0.4, Z(k) + 0.45, 0.12, HUE.frost);
      if ((word >>> 11) & 1) flags.set(X(k) - 0.45, 0.4, Z(k) + 0.45, 0.12, HUE.focus);
    }
    const range = kit.curve('range', 65, this.accent), nearest = kit.curve('observed nearest foe', 2, 0xf5f8ff), obs = now.observations;
    if (obs[10] !== 0 || obs[11] !== 0) {
      nearest.point(X(0), 0.3, Z(0));
      nearest.point(X(0) + obs[10]! * shape.arena, 0.3, Z(0) + obs[11]! * shape.arena);
    }
    if (hpOf(now, 0) > 0) {
      const reach = shape.kinds[kindOf(now, 0)]!.range;
      for (let s = 0; s <= 64; s++) range.point(X(0) + reach * Math.cos((s / 64) * 2 * Math.PI), 0.3, Z(0) + reach * Math.sin((s / 64) * 2 * Math.PI));
    }
    const select = indexOf(now.words[SKW.SELECT]!, allies), called = now.words[SKW.CALL]!;
    if (select < n) kit.marks('select', 1).set(X(select), 0.5, Z(select) - 0.9, 0.2, this.accent);
    if (called !== NONE) kit.marks('call', 1).set(X(indexOf(called, allies)), 0.5, Z(indexOf(called, allies)) - 1.2, 0.2, HUE.focus);
    const first = i - nowOf(now, shape.slices), opening = frame.frameAt(first);
    const arrive = shape.rules[RULE.ARRIVE]!;
    for (const side of [0, 1]) {
      const goal = kit.curve(side ? 'their goal' : 'our goal', 33, side ? TOKENS.cueMagenta : TOKENS.uiCyan);
      if (opening.step !== first || opening.words[SKW.BATTLE] !== now.words[SKW.BATTLE]) continue;
      let gx = 0, gy = 0;
      const from = side ? 0 : allies, to = side ? allies : n;
      for (let k = from; k < to; k++) { gx += xOf(opening, k) / (to - from); gy += yOf(opening, k) / (to - from); }
      for (let s = 0; s <= 32; s++) goal.point(gx - half + arrive * Math.cos((s / 32) * 2 * Math.PI), 0.3, gy - half + arrive * Math.sin((s / 32) * 2 * Math.PI));
    }
    const bounds = kit.curve('arena', 5, 0x8fa6c4);
    for (const [x, z] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) bounds.point(x * half, 0.3, z * half);
  }
}

export default (() => new SkirmishScene()) satisfies SceneFactory;
