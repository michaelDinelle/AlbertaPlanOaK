import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { LIGHT_LAYER } from '@kit/light2d';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { CARRIED, FORWARD, NONE, RWARE_X, RWARE_Y, STEP_X, STEP_Y, TOGGLE, actionOf, agentAt, aimOf, cellAt, cellOf, clockOf, colOf, entryAt, facingOf, homesOf, loadedOf, placeOf, rowOf, rwareDecoder, rwareShape, scoreOf, standingAt, stuckOf, wanted, type RwareShape } from '@sim/envs/rware';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatSigned } from '../../format';
import { BAY, BOT, DUST, LAMP, POD, RING, SIGHT, TEAL, TINTS, flat, floorMaterial, podMaterial, ringMaterial, robotMaterial, sightMaterial, stationMaterial, trackMaterial, type Floor, type Level, type Station } from './art';

type N<T extends string> = THREE.Node<T>;

const ANGLE = [0, Math.PI, Math.PI / 2, -Math.PI / 2] as const;
const LOOK = 5;
const TRAIL = 28;
const SCAN = 256;
const REACH = 6;
const FOOTPRINT = 0.84;

const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => formatSigned(value).replace('-', '−');

interface Sheet { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[]; used: number }
interface Glow { mesh: THREE.Mesh; level: Level }

class RwareScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dawn', 'hemiSky'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'rware';

  private shape: RwareShape = { w: 10, h: 11, agents: 2, requests: 2, steps: 500, queue: 3, board: 4 };
  private grid: GridSize = { w: 10, h: 11 };
  private floor: Floor | null = null;
  private homes: Uint8Array | null = null;
  private homeCells: number[] = [];
  private readonly goals: number[] = [];
  private readonly scanned = new Set<number>();
  private pods: Sheet | null = null;
  private cargo: Sheet | null = null;
  private bots: Sheet | null = null;
  private rings: Sheet | null = null;
  private tracks: Sheet | null = null;
  private shade: THREE.InstancedMesh | null = null;
  private sight: THREE.Mesh | null = null;
  private sightLevel: Level | null = null;
  private station: THREE.Mesh | null = null;
  private stationLook: Station | null = null;
  private readonly span = uniform(2);
  private readonly time = uniform(0);
  private lamp: Glow | null = null;
  private flare: Glow | null = null;
  private readonly flareTint = new THREE.Color();
  private readonly beacons: Glow[] = [];
  private readonly headlamps: Glow[] = [];
  private aura: Glow | null = null;
  private textures: THREE.Texture[] = [];

  private step = -1;
  private clock = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly shown = new THREE.Vector3();
  private readonly tint = new THREE.Color();
  private readonly ledger = { step: -1, action: -1, blocked: false, toggled: false, cargo: [0, 0, 0, 0], under: [false, false, false, false], score: [0, 0, 0, 0], waiting: 0, riding: 0, reward: 0, clock: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.shape = rwareShape(ctx.environment);
    const { w, h } = this.shape;
    this.grid = { w, h };
    const top = -h / 2 - 0.55, bottom = h / 2 + BAY + 0.1;
    return new THREE.Box2(new THREE.Vector2(-w / 2 - 0.55, top), new THREE.Vector2(w / 2 + 0.55, bottom + 0.14 * (bottom - top)));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb3bfd2).multiplyScalar(1.02);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[RWARE_X]!;
        agent.y = state.pose[RWARE_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const { w, h, agents, requests, steps } = this.shape, lit = light.lit, time = this.time as unknown as N<'float'>;
    const floor = floorMaterial(w, h, ctx.seed, lit);
    this.floor = floor;
    this.textures.push(...floor.textures);
    const ground = new THREE.Mesh(flat(120, 120), floor.material);
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground);

    this.tracks = this.sheet(flat(1, 1), trackMaterial(lit), TRAIL, ['aTrack'], 1);
    const pod = podMaterial(lit, time);
    this.pods = this.sheet(flat(POD, POD), pod, 2 * w * h, ['aPod'], 3);
    this.rings = this.sheet(flat(RING, RING), ringMaterial(), (LOOK + 1) * (3 * agents + requests), ['aRing', 'aTint'], 5);
    this.cargo = this.sheet(flat(POD, POD), pod, 2 * agents, ['aPod'], 6);
    this.bots = this.sheet(flat(BOT, BOT), robotMaterial(lit, time), 2 * agents, ['aBot', 'aPose'], 7);

    const look = stationMaterial(this.span as unknown as N<'float'>, lit, time);
    this.stationLook = look;
    this.station = new THREE.Mesh(flat(1, BAY), look.material);
    this.station.renderOrder = 4;
    this.station.visible = false;
    this.scene.add(this.station);

    const sight = sightMaterial(w, h);
    this.sightLevel = sight.level;
    this.sight = new THREE.Mesh(flat(SIGHT, SIGHT), sight.material);
    this.sight.renderOrder = 8;
    this.scene.add(this.sight);

    const shadeGeometry = flat(FOOTPRINT, FOOTPRINT), proto = light.occluder(shadeGeometry);
    this.shade = instancedMesh(shadeGeometry, proto.material, w * h);
    this.shade.layers.set(LIGHT_LAYER);
    this.shade.renderOrder = 1;
    this.shade.frustumCulled = false;
    this.shade.count = 0;
    this.scene.add(this.shade);

    const glow = (geometry: THREE.BufferGeometry, tint: N<'vec3'>, order: number): Glow => {
      const level = uniform(0), mesh = light.emitter(geometry, tint.mul(level));
      mesh.renderOrder = order;
      mesh.visible = false;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return { mesh, level };
    };
    for (let k = 0; k < 2 * requests; k++) this.beacons.push(glow(flat(FOOTPRINT, FOOTPRINT), rgb(TEAL), 2));
    for (let k = 0; k < 2 * agents; k++) this.headlamps.push(glow(new THREE.CircleGeometry(0.075, 20).rotateX(-Math.PI / 2), rgb(LAMP), 3));
    this.aura = glow(new THREE.CircleGeometry(0.1, 24).rotateX(-Math.PI / 2), rgb(TINTS[0]), 3);
    this.lamp = glow(flat(1, 0.1), rgb(LAMP), 2);
    this.flare = glow(new THREE.CircleGeometry(0.2, 24).rotateX(-Math.PI / 2), uniform(this.flareTint) as unknown as N<'vec3'>, 4);
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 2 : 9; });

    const ledger = this.ledger;
    const partner = (i: number) => ({
      key: `partner${i}`, label: agents > 2 ? `robot ${i + 1}` : 'partner', tint: TINTS[i]!,
      value: () => `${this.cargoText(i)} · ${ledger.score[i]} delivered`,
    });
    this.legend = {
      title: 'ROBOTIC WAREHOUSE',
      rows: [
        { key: 'you', label: 'you', tint: TINTS[0], value: () => `${this.actionText()} · ${this.cargoText(0)}` },
        ...Array.from({ length: agents - 1 }, (_, n) => partner(n + 1)),
        { key: 'requests', label: 'requests', tint: TEAL, value: () => `${ledger.waiting} on the floor · ${ledger.riding} on robots` },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(ledger.reward) },
        { key: 'shift', label: 'shift', tint: LAMP, value: () => `step ${ledger.clock} of ${steps}` },
      ],
      note: () => `you delivered ${ledger.score[0]} · the team ${ledger.score.slice(1, agents).reduce((sum, n) => sum + n, 0)} this shift`,
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

  private put(sheet: Sheet, position: THREE.Vector3, angle: number, size: number, ...values: readonly (readonly number[])[]): void {
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

  private centre(cell: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(colOf(cell), rowOf(cell), this.grid, out);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    this.time.value = this.clock;
    pool.update(frame.dtReal, time);
    const at = frame.state.step, A = frame.frameAt(at);
    if (A.step !== at) return;
    if (!this.homes) this.settle(frame.frameAt(0));
    this.discover(frame, at);
    if (frame.scrubbed || at < this.step || this.step < 0) pool.clear();
    else for (let s = this.step + 1; s <= at; s++) this.celebrate(frame, s, pool, time);
    this.step = at;
    const after = frame.frameAt(at + 1), B = after.step === at + 1 ? after : A;
    const t = B === A ? 0 : clamp01(frame.position - at);
    this.compose(A, B, t);
    this.effects(frame, at, t);
    this.trail(frame, A, B.done !== 0 && B !== A ? 1 - smoothstep01((t - 0.3) / 0.35) : 1);
    this.record(frame, A);
  }

  private settle(zero: EnvFrame): void {
    const { w, h } = this.shape, homes = homesOf(zero, this.shape), floor = this.floor;
    this.homes = homes;
    this.homeCells = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (homes[y * w + x]) this.homeCells.push(cellAt(x, y));
    if (!floor) return;
    const data = floor.cells.image.data as Uint8Array;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[((y + 1) * (w + 2) + x + 1) * 4] = homes[y * w + x] ? 255 : 0;
    floor.cells.needsUpdate = true;
  }

  private discover(frame: FrameUpdate, at: number): void {
    const { agents } = this.shape, block = Math.floor(at / SCAN);
    for (let b = Math.max(0, block - REACH); b <= block + REACH; b++) {
      const from = b * SCAN, to = from + SCAN - 1;
      if (this.scanned.has(b) || frame.frameAt(from).step !== from || frame.frameAt(to).step !== to) continue;
      let complete = true;
      for (let s = from + 1; s <= to; s++) {
        const after = frame.frameAt(s), before = frame.frameAt(s - 1);
        if (after.step !== s || before.step !== s - 1) { complete = false; continue; }
        if (after.done) continue;
        for (let i = 0; i < agents; i++) if (scoreOf(agentAt(after.words, i)) !== scoreOf(agentAt(before.words, i))) this.found(cellOf(agentAt(after.words, i)));
      }
      if (complete) this.scanned.add(b);
    }
  }

  private found(cell: number): void {
    const { w, h } = this.shape, floor = this.floor, station = this.station, lamp = this.lamp;
    if (this.goals.includes(cell)) return;
    if (rowOf(cell) !== h - 1) throw new Error(`RWARE delivers at (${colOf(cell)}, ${rowOf(cell)}), away from the bottom edge where the pick station stands`);
    this.goals.push(cell);
    this.goals.sort((a, b) => a - b);
    if (!floor || !station || !lamp) return;
    const data = floor.cells.image.data as Uint8Array;
    data[(h * (w + 2) + colOf(cell) + 1) * 4 + 1] = 255;
    floor.cells.needsUpdate = true;
    const first = colOf(this.goals[0]!), last = colOf(this.goals[this.goals.length - 1]!), span = last - first + 1.9;
    this.span.value = span;
    cellToWorldVec((first + last) / 2, 0, this.grid, this.at);
    station.position.set(this.at.x, 0, h / 2 + BAY / 2);
    station.scale.set(span, 1, 1);
    station.visible = true;
    lamp.mesh.position.set(this.at.x, 0.02, h / 2 + 0.1);
    lamp.mesh.scale.set(span - 0.7, 1, 1);
    lamp.mesh.visible = true;
    lamp.level.value = 3.6;
  }

  private check(frame: EnvFrame): void {
    const shape = this.shape, homes = this.homes!, { w, h, agents, requests } = shape;
    let pods = 0;
    for (let word = 0; word < (h + 1) >> 1; word++) {
      for (let bits = frame.words[shape.board + word]!; bits !== 0; bits &= bits - 1) {
        const cell = word * 32 + 31 - Math.clz32(bits & -bits);
        if (colOf(cell) >= w || rowOf(cell) >= h || !homes[rowOf(cell) * w + colOf(cell)]) throw new Error(`RWARE step ${frame.step} stands a shelf at (${colOf(cell)}, ${rowOf(cell)}), which is no shelf slot`);
        pods++;
      }
    }
    for (let i = 0; i < agents; i++) {
      const word = agentAt(frame.words, i), cell = cellOf(word);
      if (colOf(cell) >= w || rowOf(cell) >= h) throw new Error(`RWARE step ${frame.step} puts robot ${i} at (${colOf(cell)}, ${rowOf(cell)}) off the floor`);
      pods += loadedOf(word);
    }
    if (pods !== this.homeCells.length) throw new Error(`RWARE step ${frame.step} holds ${pods} shelves where it opened with ${this.homeCells.length}`);
    for (let s = 0; s < requests; s++) {
      const entry = entryAt(frame.words, shape, s);
      if (entry & CARRIED ? (entry & 3) >= agents || !loadedOf(agentAt(frame.words, entry & 3)) : !standingAt(frame.words, shape, entry)) throw new Error(`RWARE step ${frame.step} requests ${entry}, which is neither a standing nor a carried shelf`);
    }
  }

  private compose(A: EnvFrame, B: EnvFrame, t: number): void {
    const pods = this.pods, cargo = this.cargo, bots = this.bots, shade = this.shade, sight = this.sight, aura = this.aura;
    if (!pods || !cargo || !bots || !shade || !sight || !aura || !this.sightLevel) return;
    const shape = this.shape, { agents } = shape, next = B !== A, boundary = next && B.done !== 0, flow = next && !boundary;
    this.check(A);
    if (next) this.check(B);
    const e = smoothstep01(t / 0.85), k = smoothstep01(t / 0.6), blend = smoothstep01((t - 0.3) / 0.5);
    const fade = boundary ? 1 - smoothstep01((t - 0.3) / 0.35) : 1, rise = boundary ? smoothstep01((t - 0.45) / 0.4) : 0;
    const lit = t < 0.5 || !next ? A : B;
    let shades = 0, beacons = 0;
    const beacon = (at: THREE.Vector3, level: number) => {
      const glow = level > 0.01 ? this.beacons[beacons++] : undefined;
      if (!glow) return;
      glow.mesh.position.set(at.x, 0.02, at.z);
      glow.mesh.visible = true;
      glow.level.value = 0.55 * level;
    };

    for (const cell of this.homeCells) {
      const inA = standingAt(A.words, shape, cell), inB = next ? standingAt(B.words, shape, cell) : inA;
      const wantA = inA && wanted(A.words, shape, cell) ? 1 : 0, wantB = !next ? wantA : inB && wanted(B.words, shape, cell) ? 1 : 0;
      this.centre(cell, this.at);
      if (inA && inB) {
        const level = THREE.MathUtils.lerp(wantA, wantB, boundary ? blend : k);
        this.put(pods, this.at, 0, 1, [1, 0, level, flow && wantA !== wantB ? Math.sin(Math.PI * clamp01(t / 0.8)) : 0]);
        beacon(this.at, level);
      } else if (boundary && inA) {
        this.put(pods, this.at, 0, 1, [fade, 0, wantA, 0]);
        beacon(this.at, wantA * fade);
      } else if (boundary && inB) {
        this.put(pods, this.at, 0, 1, [rise, 0, wantB, 0]);
        beacon(this.at, wantB * rise);
      }
      if (standingAt(lit.words, shape, cell) && !(t < 0.5 || !next ? wantA : wantB)) {
        shade.setMatrixAt(shades++, this.matrix.makeTranslation(this.at.x, 0.01, this.at.z));
      }
    }

    let sightX = 0, sightZ = 0;
    for (let i = agents - 1; i >= 0; i--) {
      const a = agentAt(A.words, i), ca = cellOf(a), la = loadedOf(a);
      const pos = this.centre(ca, this.shown);
      if (flow) {
        const b = agentAt(B.words, i), cb = cellOf(b), lb = loadedOf(b), moved = ca !== cb;
        if (la !== lb && moved) throw new Error(`RWARE step ${B.step} moves robot ${i} while it ${lb ? 'lifts' : 'sets down'} a shelf`);
        if (la !== lb && standingAt(A.words, shape, ca) !== Boolean(lb)) throw new Error(`RWARE step ${B.step} has robot ${i} ${lb ? 'lift' : 'set down'} a shelf where the floor disagrees`);
        pos.lerp(this.centre(cb, this.other), e);
        const pushed = !moved && (i === 0 ? actionOf(B) === FORWARD : stuckOf(b) > stuckOf(a));
        if (pushed) pos.add(this.other.set(STEP_X[facingOf(a)]!, 0, STEP_Y[facingOf(a)]!).multiplyScalar(0.12 * Math.sin(Math.PI * clamp01(t / 0.7))));
        const angle = turn(ANGLE[facingOf(a)]!, ANGLE[facingOf(b)]!, smoothstep01(t / 0.55));
        const lift = THREE.MathUtils.lerp(la, lb, k);
        const underA = !la && standingAt(A.words, shape, ca) ? 1 : 0, underB = !lb && standingAt(B.words, shape, cb) ? 1 : 0;
        const wantA = la ? Number(wanted(A.words, shape, CARRIED | i)) : Number(lb === 1 && wanted(A.words, shape, ca));
        const wantB = lb ? Number(wanted(B.words, shape, CARRIED | i)) : Number(la === 1 && wanted(B.words, shape, cb));
        if (la || lb) this.put(cargo, pos, 0, 1, [1, lift, THREE.MathUtils.lerp(wantA, wantB, k), wantA !== wantB ? Math.sin(Math.PI * clamp01(t / 0.8)) : 0]);
        const cargoLevel = (la || lb) && lift > 0.5 ? 1 + (lb ? wantB : wantA) : 0;
        this.put(bots, pos, angle, 1, [i, 1, lift, cargoLevel], [moved ? e : 0, THREE.MathUtils.lerp(underA, underB, e), i === 0 ? 1 : 0, 0]);
        this.headlamp(i, pos, angle, 1, THREE.MathUtils.lerp(underA, underB, e));
      } else {
        const angle = ANGLE[facingOf(a)]!, under = !la && standingAt(A.words, shape, ca) ? 1 : 0, want = la ? Number(wanted(A.words, shape, CARRIED | i)) : 0;
        if (la) this.put(cargo, pos, 0, 1, [fade, 1, want, 0]);
        this.put(bots, pos, angle, 1, [i, fade, la, la ? 1 + want : 0], [0, under, i === 0 ? 1 : 0, 0]);
        this.headlamp(i, pos, angle, fade, under);
      }
      if (i === 0) { sightX = pos.x; sightZ = pos.z; }
    }
    if (boundary) for (let i = agents - 1; i >= 0; i--) {
      const b = agentAt(B.words, i), cb = cellOf(b), pos = this.centre(cb, this.shown), under = standingAt(B.words, shape, cb) ? 1 : 0;
      this.put(bots, pos, ANGLE[facingOf(b)]!, 1.25 - 0.25 * rise, [i, rise, 0, 0], [0, under, i === 0 ? 1 : 0, 0]);
      this.headlamp(agents + i, pos, ANGLE[facingOf(b)]!, rise, under);
      if (i === 0 && rise > fade) { sightX = pos.x; sightZ = pos.z; }
    }
    for (let k2 = this.headlamps.length - 1; k2 >= (boundary ? 2 * agents : agents); k2--) this.headlamps[k2]!.mesh.visible = false;
    for (let n = beacons; n < this.beacons.length; n++) this.beacons[n]!.mesh.visible = false;

    this.commit(pods);
    this.commit(cargo);
    this.commit(bots);
    shade.count = shades;
    shade.instanceMatrix.needsUpdate = true;
    sight.position.set(sightX, 0, sightZ);
    this.sightLevel.value = 0.85 * (boundary ? Math.max(fade, rise) : 1);
    aura.mesh.position.set(sightX, 0.02, sightZ);
    aura.mesh.visible = true;
    aura.level.value = 4.5 * (boundary ? Math.max(fade, rise) : 1);
  }

  private headlamp(slot: number, pos: THREE.Vector3, angle: number, level: number, under: number): void {
    const glow = this.headlamps[slot];
    if (!glow) return;
    glow.mesh.position.set(pos.x - 0.33 * Math.sin(angle), 0.02, pos.z - 0.33 * Math.cos(angle));
    glow.mesh.visible = level > 0.01;
    glow.level.value = 5 * level * (1 - 0.8 * under);
  }

  private effects(frame: FrameUpdate, at: number, t: number): void {
    const rings = this.rings, look = this.stationLook, flare = this.flare;
    if (!rings || !look || !flare) return;
    const shape = this.shape, { agents, requests } = shape, now = at + t, current = frame.frameAt(at);
    let flash = 0, gold = 0, flareAge = Infinity, flareCell = -1, flareMine = false;
    for (let s = at + 1; s > at + 1 - LOOK && s >= 1; s--) {
      const after = frame.frameAt(s), before = frame.frameAt(s - 1), age = now - (s - 1);
      if (after.step !== s || before.step !== s - 1 || age < 0 || age >= LOOK) continue;
      if (after.done && s > at) continue;
      if (after.done) break;
      const p = age / LOOK;
      for (let i = 0; i < agents; i++) {
        const a = agentAt(before.words, i), b = agentAt(after.words, i);
        if (scoreOf(a) !== scoreOf(b)) {
          this.centre(cellOf(b), this.at);
          this.put(rings, this.at, 0, 1, [p, 1, i === 0 ? 1.15 : 0.8, 0.05], [...this.rgbOf(i === 0 ? TOKENS.celebrationGold : TEAL), 0]);
          const level = (1 - p) ** 2;
          if (level > flash) { flash = level; gold = i === 0 ? 1 : 0; }
          if (age < flareAge) { flareAge = age; flareCell = cellOf(b); flareMine = i === 0; }
        }
        if (loadedOf(a) !== loadedOf(b)) {
          this.centre(cellOf(b), this.at);
          this.put(rings, this.at, 0, 1, [Math.min(1, p * 2), 0.45, 0.64, 0.022], [...this.rgbOf(DUST), 0.42]);
        }
      }
      for (let slot = 0; slot < requests; slot++) {
        const ea = entryAt(before.words, shape, slot), eb = entryAt(after.words, shape, slot);
        if (ea === eb || placeOf(before.words, ea) === placeOf(after.words, eb)) continue;
        this.centre(s > at ? placeOf(after.words, eb) : placeOf(current.words, entryAt(current.words, shape, slot)), this.at);
        this.put(rings, this.at, 0, 1, [p, 1, 0.95, 0.045], [...this.rgbOf(TEAL), 0]);
      }
    }
    this.commit(rings);
    look.flash.value = flash;
    look.gold.value = gold;
    if (flareCell < 0) {
      flare.mesh.visible = false;
      return;
    }
    this.centre(flareCell, this.at);
    flare.mesh.position.set(this.at.x, 0.02, this.at.z);
    flare.mesh.visible = true;
    this.flareTint.set(flareMine ? TOKENS.celebrationGold : TEAL);
    flare.level.value = (flareMine ? 9 : 4) * (1 - flareAge / LOOK) ** 2;
  }

  private rgbOf(color: number): [number, number, number] {
    this.tint.set(color);
    return [this.tint.r, this.tint.g, this.tint.b];
  }

  private trail(frame: FrameUpdate, A: EnvFrame, fade: number): void {
    const tracks = this.tracks;
    if (!tracks) return;
    let later = A;
    for (let n = 0; n < TRAIL && later.step > 0 && later.done === 0; n++) {
      const earlier = frame.frameAt(later.step - 1);
      if (earlier.step !== later.step - 1) break;
      const a = cellOf(agentAt(earlier.words, 0)), b = cellOf(agentAt(later.words, 0));
      if (a !== b) {
        this.centre(a, this.at);
        this.centre(b, this.other);
        this.shown.lerpVectors(this.at, this.other, 0.5);
        this.put(tracks, this.shown, Math.atan2(-(this.other.x - this.at.x), -(this.other.z - this.at.z)), 1, [0.42 * fade * (1 - n / TRAIL), 0, 0, 0]);
      }
      later = earlier;
    }
    this.commit(tracks);
  }

  private celebrate(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const before = frame.frameAt(s - 1), after = frame.frameAt(s);
    if (before.step !== s - 1 || after.step !== s) return;
    for (const event of rwareDecoder.deriveEvents(before, after)) {
      if (event.type !== 'goal') continue;
      if (after.done) {
        const first = this.goals[0], last = this.goals[this.goals.length - 1];
        if (first === undefined || last === undefined) continue;
        cellToWorldVec((colOf(first) + colOf(last)) / 2, rowOf(first), this.grid, this.at);
      } else this.centre(cellOf(agentAt(after.words, 0)), this.at);
      pool.burst(this.other.copy(this.at).setY(38), TOKENS.celebrationGold, 0.9, time);
      pool.ripple(this.other.copy(this.at).setY(0), TOKENS.celebrationGold, 0.34, time);
    }
  }

  private cargoText(i: number): string {
    const cargo = this.ledger.cargo[i]!;
    return cargo === 2 ? 'carrying a requested shelf' : cargo === 1 ? 'returning a shelf' : this.ledger.under[i] ? 'empty · under a shelf' : 'empty';
  }

  private actionText(): string {
    const { action, blocked, toggled, cargo } = this.ledger;
    if (action < 0) return '—';
    if (action === FORWARD) return blocked ? 'forward · blocked' : 'forward';
    if (action === TOGGLE) return toggled ? (cargo[0] ? 'lifts a shelf' : 'sets a shelf down') : 'toggle · nothing to do';
    return ['waits', 'forward', 'turns left', 'turns right'][action]!;
  }

  private record(frame: FrameUpdate, A: EnvFrame): void {
    const ledger = this.ledger, shape = this.shape, { agents, requests } = shape;
    for (let i = 0; i < agents; i++) {
      const word = agentAt(A.words, i), cell = cellOf(word);
      ledger.cargo[i] = loadedOf(word) ? (wanted(A.words, shape, CARRIED | i) ? 2 : 1) : 0;
      ledger.under[i] = !loadedOf(word) && standingAt(A.words, shape, cell);
      ledger.score[i] = scoreOf(word);
    }
    ledger.waiting = 0;
    ledger.riding = 0;
    for (let s = 0; s < requests; s++) {
      if (entryAt(A.words, shape, s) & CARRIED) ledger.riding++;
      else ledger.waiting++;
    }
    ledger.reward = A.reward;
    ledger.clock = clockOf(A.words);
    if (ledger.step === A.step) return;
    ledger.step = A.step;
    const before = frame.frameAt(A.step - 1), known = A.step > 0 && before.step === A.step - 1 && A.done === 0;
    ledger.action = known ? actionOf(A) : -1;
    const a = known ? agentAt(before.words, 0) : 0, b = agentAt(A.words, 0);
    ledger.blocked = known && cellOf(a) === cellOf(b);
    ledger.toggled = known && loadedOf(a) !== loadedOf(b);
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const A = frame.frameAt(frame.state.step);
    if (A.step !== frame.state.step || !this.homes) return;
    const shape = this.shape, { w, agents, requests } = shape, at = this.at;
    const pods = kit.marks('pods', w * shape.h), vacant = kit.marks('vacant', w * shape.h);
    for (const cell of this.homeCells) {
      this.centre(cell, at);
      if (standingAt(A.words, shape, cell)) pods.set(at.x, 0.05, at.z, 0.1, 0x6a70b8);
      else vacant.set(at.x, 0.05, at.z, 0.04, 0x8a9098);
    }
    const wants = kit.marks('requests', requests);
    for (let s = 0; s < requests; s++) {
      this.centre(placeOf(A.words, entryAt(A.words, shape, s)), at);
      wants.set(at.x + 0.24, 0.06, at.z - 0.24, 0.1, TEAL);
    }
    const goals = kit.marks('goals', w);
    for (const cell of this.goals) {
      this.centre(cell, at);
      goals.set(at.x, 0.05, at.z, 0.14, TOKENS.celebrationGold);
    }
    const bots = kit.marks('robots', agents), loads = kit.marks('loads', agents);
    for (let i = 0; i < agents; i++) {
      const word = agentAt(A.words, i), cell = cellOf(word), f = facingOf(word), dx = STEP_X[f]!, dz = STEP_Y[f]!;
      this.centre(cell, at);
      bots.set(at.x, 0.05, at.z, 0.12, TINTS[i]!);
      if (loadedOf(word)) loads.set(at.x - 0.24, 0.06, at.z + 0.24, 0.08, 0xffffff);
      const arrow = kit.curve(`facing${i}`, 5, TINTS[i]!), tipX = at.x + 0.42 * dx, tipZ = at.z + 0.42 * dz;
      arrow.point(at.x, 0.07, at.z);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.13 * dx - 0.1 * dz, 0.07, tipZ - 0.13 * dz + 0.1 * dx);
      arrow.point(tipX, 0.07, tipZ);
      arrow.point(tipX - 0.13 * dx + 0.1 * dz, 0.07, tipZ - 0.13 * dz - 0.1 * dx);
      const aim = kit.curve(`aim${i}`, 2, TINTS[i]!);
      if (i > 0 && aimOf(word) !== NONE) {
        aim.point(at.x, 0.06, at.z);
        this.centre(aimOf(word), this.other);
        aim.point(this.other.x, 0.06, this.other.z);
      }
    }
    const o = A.observations, self = cellOf(agentAt(A.words, 0));
    if (o[0] !== colOf(self) || o[1] !== rowOf(self)) throw new Error(`RWARE step ${A.step} observes its robot at (${o[0]}, ${o[1]}) though it stands at (${colOf(self)}, ${rowOf(self)})`);
    const sensed = kit.marks('sensed', 9), frameCurve = kit.curve('sensor', 5, TINTS[0]);
    for (let n = 0; n < 9; n++) {
      const code = o[5 + n]!, x = colOf(self) + (n % 3) - 1, y = rowOf(self) + Math.floor(n / 3) - 1;
      if (code === 0) continue;
      cellToWorldVec(x, y, this.grid, at);
      sensed.set(at.x - 0.24, 0.06, at.z - 0.24, 0.07, code >= 4 ? 0xffffff : code & 2 ? TEAL : 0x6a70b8);
    }
    for (const [x, y] of [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]] as const) {
      cellToWorldVec(colOf(self) + x, rowOf(self) + y, this.grid, at);
      frameCurve.point(at.x, 0.06, at.z);
    }
  }
}

export default (() => new RwareScene()) satisfies SceneFactory;
