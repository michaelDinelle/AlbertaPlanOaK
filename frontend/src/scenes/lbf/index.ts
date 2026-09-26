import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugCurve, DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { LBF_COL, LBF_ROW, LOAD, LW, NONE, STEP_COL, STEP_ROW, besideOf, colOf, foodAt, intentOf, isMove, kindOf, lastOf, lbfDecoder, lbfShape, levelOf, playerAt, rowOf, seenBy, type LbfShape, type Seen } from '@sim/envs/lbf';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { ANT, BERRY, BUNCH, DROP, FRAME, GOLD, SPOT, TINTS, antMaterial, bunchMaterial, dropMaterial, flat, groundMaterial, spotMaterial, type Ground, type Level } from './art';

type N<T extends string> = THREE.Node<T>;

const NAMES = ['wait', '↑ north', '↓ south', '← west', '→ east', 'load'] as const;
const KINDS = ['—', 'H1 nearest', 'H2 centre', 'H3 solo', 'H4 team'] as const;
const PIPS = 6;
const SOUTH = Math.PI;
const FLASHES = 4;
const TRAIL = 12;

const facing = (dx: number, dz: number) => Math.atan2(-dx, -dz);
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => formatSigned(value).replace('-', '−');

interface Sheet { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[]; used: number }
interface Flash { mesh: THREE.Mesh; tint: THREE.Color; level: Level; born: number; peak: number }

class LbfScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'lbf';

  private shape: LbfShape = { field: 8, players: 2, foods: 2, sight: 8, levels: 2, steps: 50, full: true };
  private grid: GridSize = { w: 8, h: 8 };
  private ground: Ground | null = null;
  private spots: Sheet | null = null;
  private drops: Sheet | null = null;
  private bunches: Sheet | null = null;
  private ants: Sheet | null = null;
  private blend: Level | null = null;
  private aura: THREE.Mesh | null = null;
  private readonly auraLevel = uniform(0);
  private readonly flashes: Flash[] = [];
  private flashCursor = 0;
  private textures: THREE.Texture[] = [];

  private step = -1;
  private clock = 0;
  private faceStep = -1;
  private readonly face = new Float32Array(3);
  private readonly seen = new Map<number, Seen>();
  private readonly loadsA = new Float32Array(9);
  private readonly loadsB = new Float32Array(9);
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly shown = new THREE.Vector3();
  private readonly ledger = { step: -1, level: [0, 0, 0], kind: [0, 0, 0], act: [-1, -1, -1], bumped: [false, false, false], intent: -1, gathered: 0, spawned: 0, reward: 0, clock: 0, episode: 0, total: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const shape = lbfShape(ctx.environment), { field, players, levels } = shape;
    if (players * levels > PIPS || levels > 2) throw new Error(`${ctx.environment.envId} can spawn level ${players * levels} food and level ${levels} players past the ${PIPS} berries and 2 pips the scene draws`);
    this.shape = shape;
    this.grid = { w: field, h: field };
    const half = field / 2 + FRAME + 0.3;
    return new THREE.Box2(new THREE.Vector2(-half, -half), new THREE.Vector2(half, half + 0.16 * field));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff0da).multiplyScalar(0.93);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[LBF_COL]!;
        agent.y = state.pose[LBF_ROW]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const { field, steps, players, foods, full } = this.shape;
    const ground = groundMaterial(field, steps, ctx.seed, light.lit);
    ground.fog.value = full ? 0 : 1;
    this.ground = ground;
    this.textures.push(...ground.textures);
    const floor = new THREE.Mesh(flat(80, 80), ground.material);
    floor.renderOrder = 0;
    floor.frustumCulled = false;
    this.scene.add(floor);

    this.spots = this.sheet(flat(SPOT, SPOT), spotMaterial(light.lit), 2 * foods, ['aSpot'], 1);
    this.drops = this.sheet(flat(DROP, DROP), dropMaterial(light.lit), 3 * TRAIL, ['aDrop'], 1);
    const bunch = bunchMaterial();
    this.blend = bunch.blend;
    this.bunches = this.sheet(flat(BUNCH, BUNCH), bunch.material, 2 * foods, ['aBunch', 'aRingA', 'aRingB'], 3);
    this.ants = this.sheet(flat(ANT, ANT), antMaterial(), 2 * players, ['aAnt', 'aPose'], 5);

    this.aura = light.emitter(new THREE.CircleGeometry(0.11, 24).rotateX(-Math.PI / 2), rgb(TINTS[0]).mul(this.auraLevel));
    this.scene.add(this.aura);
    for (let k = 0; k < FLASHES; k++) {
      const tint = new THREE.Color(), level = uniform(0);
      const mesh = light.emitter(new THREE.CircleGeometry(0.2, 24).rotateX(-Math.PI / 2), (uniform(tint) as unknown as N<'vec3'>).mul(level));
      mesh.visible = false;
      this.scene.add(mesh);
      this.flashes.push({ mesh, tint, level, born: -Infinity, peak: 0 });
    }
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 2 : 8; });

    const ledger = this.ledger;
    const partner = (i: number) => ({
      key: `partner${i}`, label: players > 2 ? `partner ${i}` : 'partner', tint: TINTS[i]!,
      value: () => `level ${ledger.level[i]} · ${KINDS[ledger.kind[i]!] ?? '—'} · ${this.action(i)}`,
    });
    this.legend = {
      title: 'LEVEL-BASED FORAGING',
      rows: [
        { key: 'you', label: 'you', tint: TINTS[0], value: () => `level ${ledger.level[0]} · ${this.action(0)}` },
        ...Array.from({ length: players - 1 }, (_, n) => partner(n + 1)),
        { key: 'gathered', label: 'gathered', tint: BERRY, value: () => `${ledger.gathered} of ${ledger.spawned} berries` },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(ledger.reward) },
        { key: 'step', label: 'step', tint: GOLD, value: () => `${ledger.clock} of ${steps}` },
      ],
      note: () => `episode ${formatCount(ledger.episode + 1)} · ${signed(ledger.total)} so far`,
      bar: () => ledger.clock / steps,
    } satisfies LegendSpec;
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

  private put(sheet: Sheet, position: THREE.Vector3, angle: number, size: number, ...values: readonly number[][]): void {
    const i = sheet.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    sheet.mesh.setMatrixAt(i, this.matrix.compose(position, this.quaternion, this.scale.setScalar(size)));
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
  }

  private centre(entity: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(colOf(entity), rowOf(entity), this.grid, out);
  }

  private seenOf(frame: EnvFrame): Seen {
    let seen = this.seen.get(frame.step);
    if (seen) return seen;
    seen = seenBy(frame, this.shape);
    if (this.seen.size > 8) this.seen.clear();
    this.seen.set(frame.step, seen);
    return seen;
  }

  private headingOf(frame: EnvFrame, layout: EnvFrame, i: number, fallback: number): number {
    const act = lastOf(frame.words, i);
    if (isMove(act)) return facing(STEP_COL[act]!, STEP_ROW[act]!);
    if (act !== LOAD) return fallback;
    const p = playerAt(layout.words, i), j = besideOf(layout.words, this.shape.foods, p);
    return j < 0 ? fallback : facing(colOf(foodAt(layout.words, j)) - colOf(p), rowOf(foodAt(layout.words, j)) - rowOf(p));
  }

  private facingOf(frame: FrameUpdate, A: EnvFrame): Float32Array {
    if (this.faceStep === A.step) return this.face;
    this.faceStep = A.step;
    for (let i = 0; i < this.shape.players; i++) {
      let angle = NaN;
      for (let s = A.step; s > 0 && Number.isNaN(angle) && A.step - s <= this.shape.steps; s--) {
        const f = s === A.step ? A : frame.frameAt(s);
        if (f.step !== s) this.faceStep = -1;
        if (f.step !== s || f.done) break;
        angle = this.headingOf(f, f, i, NaN);
      }
      this.face[i] = Number.isNaN(angle) ? SOUTH : angle;
    }
    return this.face;
  }

  private loads(out: Float32Array, frame: EnvFrame, layout: EnvFrame): Float32Array {
    out.fill(0);
    for (let i = 0; i < this.shape.players; i++) {
      if (lastOf(frame.words, i) !== LOAD) continue;
      const p = playerAt(layout.words, i), j = besideOf(layout.words, this.shape.foods, p);
      if (j >= 0) out[3 * j + i] = levelOf(p);
    }
    return out;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, time);
    const at = frame.state.step, A = frame.frameAt(at);
    if (A.step !== at) return;
    if (frame.scrubbed || at < this.step || this.step < 0) {
      pool.clear();
      for (const flash of this.flashes) flash.born = -Infinity;
    } else for (let s = this.step + 1; s <= at; s++) this.effects(frame, s, pool, time);
    this.step = at;
    const after = frame.frameAt(at + 1), B = after.step === at + 1 ? after : A;
    this.compose(frame, A, B, B === A ? 0 : clamp01(frame.position - at));
    this.glow(time);
    this.record(frame, A);
    if (this.ground) this.ground.time.value = this.clock;
  }

  private compose(frame: FrameUpdate, A: EnvFrame, B: EnvFrame, t: number): void {
    const ants = this.ants, bunches = this.bunches, spots = this.spots, ground = this.ground;
    if (!ants || !bunches || !spots || !ground || !this.blend || !this.aura) return;
    const { players, foods, steps, sight, field, full } = this.shape;
    const next = B !== A, boundary = next && B.done !== 0, fresh = A.step === 0 || A.done !== 0;
    const e = smoothstep01(t / 0.8), k = smoothstep01(t / 0.4);
    const fade = boundary ? 1 - smoothstep01((t - 0.35) / 0.3) : 1, rise = boundary ? smoothstep01((t - 0.5) / 0.35) : 0;
    const seenA = this.seenOf(A), seenB = next ? this.seenOf(B) : seenA, face = this.facingOf(frame, A);
    const cleared = boundary && A.words[LW.CLOCK]! + 1 < steps;
    const mine = boundary && B.reward > 0 ? besideOf(A.words, foods, playerAt(A.words, 0)) : -1;
    const loadsA = fresh ? this.loadsA.fill(0) : this.loads(this.loadsA, A, A), loadsB = next ? this.loads(this.loadsB, B, A) : this.loadsB.fill(0);
    const loading = (loads: Float32Array, j: number) => loads[3 * j]! + loads[3 * j + 1]! + loads[3 * j + 2]! > 0;
    this.blend.value = k;

    for (let j = 0; j < foods; j++) {
      const f = foodAt(A.words, j);
      if (f === 0) continue;
      this.centre(f, this.at);
      const level = levelOf(f);
      if (level === 0) {
        this.put(spots, this.at, 0, 1, [fade, 0, 0, 0]);
        continue;
      }
      const gone = boundary ? cleared || j === mine : next && levelOf(foodAt(B.words, j)) === 0;
      const lift = gone ? smoothstep01(t / (boundary ? 0.45 : 0.75)) : 0;
      const vanish = gone ? smoothstep01((t - (boundary ? 0.3 : 0.75)) / (boundary ? 0.3 : 0.25)) : 0;
      const wantA = loading(loadsA, j), wantB = loading(loadsB, j);
      const shake = wantB && !gone ? 0.09 * Math.sin(t * 6 * Math.PI) * (1 - t) : 0;
      const seen = gone || boundary ? Number(seenA.foods[j]) : THREE.MathUtils.lerp(Number(seenA.foods[j]), Number(seenB.foods[j]), e);
      this.put(bunches, this.at, shake, 1, [level, (1 - vanish) * (gone ? 1 : fade), lift, seen],
        [loadsA[3 * j]!, loadsA[3 * j + 1]!, loadsA[3 * j + 2]!, wantA ? 1 : 0], [loadsB[3 * j]!, loadsB[3 * j + 1]!, loadsB[3 * j + 2]!, wantB ? 1 : 0]);
      if (gone && !boundary) this.put(spots, this.at, 0, 1, [vanish, 0, 0, 0]);
    }
    if (boundary) for (let j = 0; j < foods; j++) {
      const f = foodAt(B.words, j);
      if (levelOf(f) === 0) continue;
      this.put(bunches, this.centre(f, this.at), 0, 0.55 + 0.45 * rise, [levelOf(f), rise, 0, Number(seenB.foods[j])], [0, 0, 0, 0], [0, 0, 0, 0]);
    }

    let auraX = 0, auraZ = 0;
    for (let n = 0; n < players; n++) {
      const i = (n + 1) % players, a = playerAt(A.words, i), act = next ? lastOf(B.words, i) : NONE, level = levelOf(a);
      const from = this.centre(a, this.at), pos = this.shown.copy(from);
      let walk = 0;
      if (next && !boundary) {
        const b = playerAt(B.words, i), moved = (a & 255) !== (b & 255);
        pos.lerp(this.centre(b, this.other), e);
        if (moved) walk = t;
        else if (isMove(act)) this.nudge(pos, act, 0.2, t);
        else if (i === 0 && act === NONE && isMove(intentOf(B))) this.nudge(pos, intentOf(B), 0.12, t);
      } else if (boundary && isMove(act)) {
        pos.add(this.other.set(STEP_COL[act]!, 0, STEP_ROW[act]!).multiplyScalar(0.3 * smoothstep01(t / 0.5)));
        walk = t;
      }
      const angle = turn(face[i]!, next ? this.headingOf(B, A, i, face[i]!) : face[i]!, smoothstep01(t / 0.35));
      const lean = THREE.MathUtils.lerp(!fresh && lastOf(A.words, i) === LOAD ? 1 : 0, act === LOAD ? 1 : 0, k);
      const j = besideOf(A.words, foods, a);
      if (lean > 0 && j >= 0) pos.add(this.other.set(colOf(foodAt(A.words, j)) - colOf(a), 0, rowOf(foodAt(A.words, j)) - rowOf(a)).multiplyScalar(0.1 * lean));
      const seen = boundary ? Number(seenA.players[i]) : THREE.MathUtils.lerp(Number(seenA.players[i]), Number(seenB.players[i]), e);
      const wiggle = 0.008 * Math.sin(this.clock * 2.3 + 1.7 * i) + 0.012 * lean * Math.sin(this.clock * 11 + i);
      this.put(ants, pos, angle, this.sizeOf(level) * (1 + 0.04 * Math.sin(Math.PI * walk)), [i, level, fade, seen], [walk, lean, angle, wiggle]);
      if (i === 0) { auraX = pos.x; auraZ = pos.z; }
    }
    if (boundary) for (let n = 0; n < players; n++) {
      const i = (n + 1) % players, b = playerAt(B.words, i), pos = this.centre(b, this.shown);
      this.put(ants, pos, SOUTH, this.sizeOf(levelOf(b)) * (1.3 - 0.3 * rise), [i, levelOf(b), rise, Number(seenB.players[i])], [0, 0, SOUTH, 0.008 * Math.sin(this.clock * 2.3 + 1.7 * i)]);
      if (i === 0 && rise > fade) { auraX = pos.x; auraZ = pos.z; }
    }
    this.commit(spots);
    this.trail(frame, A, fade);
    this.commit(bunches);
    this.commit(ants);
    this.aura.position.set(auraX, 0.02, auraZ);
    this.auraLevel.value = 5 * (boundary ? Math.max(fade, rise) : 1);

    const self = playerAt(A.words, 0), target = playerAt(B.words, 0), w = boundary ? smoothstep01((t - 0.35) / 0.4) : e;
    if (!full) {
      const r = THREE.MathUtils.lerp(rowOf(self), rowOf(target), w), c = THREE.MathUtils.lerp(colOf(self), colOf(target), w);
      ground.window.value.set(Math.max(0, c - sight), Math.max(0, r - sight), Math.min(field, c + sight + 1), Math.min(field, r + sight + 1));
    }
    const clock = A.words[LW.CLOCK]!;
    ground.clock.value = THREE.MathUtils.lerp(clock, boundary ? clock + 1 : next ? B.words[LW.CLOCK]! : clock, e) / steps * (boundary ? 1 - smoothstep01((t - 0.35) / 0.4) : 1);
  }

  private trail(frame: FrameUpdate, A: EnvFrame, fade: number): void {
    const drops = this.drops;
    if (!drops) return;
    let later = A;
    for (let n = 0; n < TRAIL && later.step > 0 && later.done === 0; n++) {
      const earlier = frame.frameAt(later.step - 1);
      if (earlier.step !== later.step - 1) break;
      const a = playerAt(earlier.words, 0), b = playerAt(later.words, 0);
      if ((a & 255) !== (b & 255)) {
        this.centre(a, this.at);
        this.centre(b, this.other);
        for (let k = 0; k < 3; k++) this.put(drops, this.shown.lerpVectors(this.at, this.other, (k + 0.5) / 3), 0, 1, [0.55 * fade * (1 - (n + 1 - (k + 0.5) / 3) / TRAIL), 0, 0, 0]);
      }
      later = earlier;
    }
    this.commit(drops);
  }

  private sizeOf(level: number): number {
    return 1.32 * (0.8 + 0.2 * (level - 1) / Math.max(1, this.shape.levels - 1));
  }

  private nudge(pos: THREE.Vector3, action: number, reach: number, t: number): void {
    pos.add(this.other.set(STEP_COL[action]!, 0, STEP_ROW[action]!).multiplyScalar(reach * Math.sin(Math.PI * clamp01(t / 0.6))));
  }

  private effects(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const A = frame.frameAt(s - 1), B = frame.frameAt(s);
    if (A.step !== s - 1 || B.step !== s) return;
    const { foods, steps, field } = this.shape, boundary = B.done !== 0;
    const cleared = boundary && A.words[LW.CLOCK]! + 1 < steps;
    const mine = besideOf(A.words, foods, playerAt(A.words, 0)), spot = mine >= 0 ? this.centre(foodAt(A.words, mine), this.at) : null;
    for (const event of lbfDecoder.deriveEvents(A, B)) {
      if (event.type === 'goal' && spot) {
        pool.burst(this.other.copy(spot).setY(38), TOKENS.celebrationGold, 0.45 + 0.9 * Math.min(1, B.reward * 2), time);
        pool.ripple(this.other.copy(spot).setY(0), TOKENS.celebrationGold, 0.26, time);
        this.flash(spot, TOKENS.celebrationGold, 30, time);
      }
      if (event.type === 'harm' && spot) {
        pool.burst(this.other.copy(spot).setY(38), TOKENS.failureBlue, 0.3, time);
        pool.ripple(this.other.copy(spot).setY(0), TOKENS.failureBlue, 0.2, time);
        pool.pulse.fireSigh(0.7);
        this.flash(spot, TOKENS.failureBlue, 14, time);
      }
      if (event.type === 'done' && cleared) pool.ripple(this.other.set(0, 0, 0), TINTS[0], field * 0.4, time);
    }
    for (let j = 0; j < foods; j++) {
      const f = foodAt(A.words, j);
      if (levelOf(f) === 0 || (j === mine && B.reward > 0)) continue;
      if (!(boundary ? cleared : levelOf(foodAt(B.words, j)) === 0)) continue;
      this.centre(f, this.at);
      pool.burst(this.other.copy(this.at).setY(38), BERRY, 0.3, time);
      this.flash(this.at, 0xffd9a8, 9, time);
    }
  }

  private flash(at: THREE.Vector3, color: number, peak: number, time: number): void {
    const flash = this.flashes[this.flashCursor]!;
    this.flashCursor = (this.flashCursor + 1) % FLASHES;
    flash.mesh.position.set(at.x, 0.02, at.z);
    flash.tint.set(color);
    flash.born = time;
    flash.peak = peak;
  }

  private glow(time: number): void {
    for (const flash of this.flashes) {
      const age = time - flash.born, life = age >= 0 && age < 0.9 ? (1 - age / 0.9) ** 2 : 0;
      flash.mesh.visible = life > 0;
      flash.level.value = flash.peak * life;
    }
  }

  private action(i: number): string {
    const { act, intent, bumped } = this.ledger, done = act[i]!;
    if (done < 0) return '—';
    if (i === 0 && done !== intent) return isMove(intent) ? `${NAMES[intent]} · blocked` : intent === LOAD ? 'load · nothing in reach' : NAMES[done]!;
    return bumped[i] ? `${NAMES[done]} · bumped` : NAMES[done]!;
  }

  private record(frame: FrameUpdate, A: EnvFrame): void {
    const ledger = this.ledger, { players, foods } = this.shape;
    for (let i = 0; i < players; i++) {
      ledger.level[i] = levelOf(playerAt(A.words, i));
      ledger.kind[i] = kindOf(A.words, i);
      ledger.act[i] = A.step > 0 ? lastOf(A.words, i) : -1;
    }
    ledger.intent = A.step > 0 ? intentOf(A) : -1;
    let alive = 0;
    for (let j = 0; j < foods; j++) alive += levelOf(foodAt(A.words, j));
    ledger.spawned = A.words[LW.SPAWNED]!;
    ledger.gathered = ledger.spawned - alive;
    ledger.reward = A.reward;
    ledger.clock = A.words[LW.CLOCK]!;
    ledger.episode = A.words[LW.EPISODE]!;
    if (ledger.step === A.step) return;
    ledger.step = A.step;
    const before = frame.frameAt(A.step - 1), known = A.step > 0 && before.step === A.step - 1 && A.done === 0;
    for (let i = 0; i < players; i++) ledger.bumped[i] = known && isMove(lastOf(A.words, i)) && playerAt(before.words, i) === playerAt(A.words, i);
    ledger.total = 0;
    for (let s = A.step, n = ledger.clock; n > 0; s--, n--) {
      const f = s === A.step ? A : frame.frameAt(s);
      if (f.step !== s) break;
      ledger.total += f.reward;
    }
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const A = frame.frameAt(frame.state.step);
    if (A.step !== frame.state.step) return;
    const { players, foods, full, sight, field } = this.shape, seen = this.seenOf(A), at = this.at;
    const food = kit.marks('food', foods), pips = kit.marks('pips', PIPS * (players + foods)), eyes = kit.marks('seen', players + foods), mates = kit.marks('partners', players);
    const count = (level: number, color: number) => { for (let n = 0; n < level; n++) pips.set(at.x - 0.36 + 0.12 * n, 0.05, at.z + 0.38, 0.035, color); };
    for (let j = 0; j < foods; j++) {
      const f = foodAt(A.words, j);
      if (f === 0) continue;
      this.centre(f, at);
      food.set(at.x, 0.05, at.z, levelOf(f) ? 0.09 + 0.02 * levelOf(f) : 0.05, levelOf(f) ? TOKENS.celebrationGold : 0x8a8a8a);
      count(levelOf(f), TOKENS.celebrationGold);
      if (seen.foods[j]) eyes.set(at.x + 0.36, 0.05, at.z - 0.36, 0.05, TOKENS.uiCyan);
    }
    for (let i = 0; i < players; i++) {
      const p = playerAt(A.words, i);
      this.centre(p, at);
      if (i > 0) mates.set(at.x, 0.05, at.z, 0.12, TINTS[i]!);
      count(levelOf(p), TINTS[i]!);
      if (i > 0 && seen.players[i]) eyes.set(at.x + 0.36, 0.05, at.z - 0.36, 0.05, TOKENS.uiCyan);
      const tether = kit.curve(`load${i}`, 2, TINTS[i]!), j = besideOf(A.words, foods, p);
      if (A.step > 0 && A.done === 0 && lastOf(A.words, i) === LOAD && j >= 0) {
        tether.point(at.x, 0.06, at.z);
        this.centre(foodAt(A.words, j), this.other);
        tether.point(this.other.x, 0.06, this.other.z);
      }
    }
    if (!full) {
      const self = playerAt(A.words, 0);
      this.rect(kit.curve('sight', 5, TINTS[0]), seen.top, seen.left, Math.min(field - 1, rowOf(self) + sight), Math.min(field - 1, colOf(self) + sight), 0);
      this.rect(kit.curve('reach', 5, TOKENS.uiCyan), seen.top, seen.left, Math.min(field - 1, seen.top + 2 * sight), Math.min(field - 1, seen.left + 2 * sight), 0.06);
    }
    const arrow = kit.curve('action', 5, TOKENS.uiCyan), act = lastOf(A.words, 0), before = frame.frameAt(A.step - 1);
    if (A.step > 0 && A.done === 0 && before.step === A.step - 1 && isMove(act)) {
      const from = this.centre(playerAt(before.words, 0), this.other), dx = STEP_COL[act]!, dz = STEP_ROW[act]!, tipX = from.x + 0.62 * dx, tipZ = from.z + 0.62 * dz;
      arrow.point(from.x, 0.07, from.z);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.14 * dx - 0.1 * dz, 0.07, tipZ - 0.14 * dz + 0.1 * dx);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.14 * dx + 0.1 * dz, 0.07, tipZ - 0.14 * dz - 0.1 * dx);
    }
  }

  private rect(curve: DebugCurve, top: number, left: number, bottom: number, right: number, inset: number): void {
    const corners = [[left, top], [right + 1, top], [right + 1, bottom + 1], [left, bottom + 1], [left, top]] as const;
    for (const [x, y] of corners) {
      cellToWorldVec(x - 0.5, y - 0.5, this.grid, this.other);
      curve.point(this.other.x + (x === left ? inset : -inset), 0.06, this.other.z + (y === top ? inset : -inset));
    }
  }
}

export default (() => new LbfScene()) satisfies SceneFactory;
