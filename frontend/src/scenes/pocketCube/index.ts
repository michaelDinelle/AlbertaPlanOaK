import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, envelope, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { CORNERS, CUBE_SLOTS, cubeKey, cubieAt, faceName, faceletsOf, movesOf, pocketCubeModel, twistAt, type Axis, type CubeModel, type CubeTurn } from '@sim/envs/pocketcube';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';
import {
  GOLD, KEY, KEY_QUAD, LAPSE, PAINT, SLASH, TILE_QUAD, arrowMaterial, consoleMaterial, cubieMaterial, flat, glyphAtlas, hexagonGeometry, keyMaterial, labelMaterial, matMaterial, pinMaterial, tileMaterial,
  type Atlas, type Console, type Level,
} from './art';

type N<T extends string> = THREE.Node<T>;

const S2 = Math.SQRT1_2, S3 = 1 / Math.sqrt(3), S6 = 1 / Math.sqrt(6);
const FRONT = new THREE.Matrix4().set(S2, 0, -S2, 0, S3, S3, S3, 0, S6, -2 * S6, S6, 0, 0, 0, 0, 1);
const BACK = new THREE.Matrix4().set(-S2, 0, S2, 0, -S3, -S3, -S3, 0, S6, -2 * S6, S6, 0, 0, 0, 0, 1);
const CUBIE = 0.98;
const FACE = 0.5 + CUBIE / 2;
const HEX = FACE * Math.sqrt(8 / 3);
const LIFT = 8;
const SETS = 2;
const TOWARD = new THREE.Vector3(-0.15, 1, -0.5).normalize();
const MAT = new THREE.Vector2(7.55, 4.05);
const TAPE_SPAN = 14;
const TAPE_PITCH = 0.47;
const LED_DIGITS = 8;
const FAST = 3;
const TURN_FROM = 0.45;
const HOLD = 0.42;
const REVEALED = 0.6;
const CHEER_SECONDS = 1.6;
const SIGH_SECONDS = 1.1;
const LOOK_SECONDS = 0.5;

interface View { x: number; z: number; radius: number; basis: THREE.Matrix4; world: THREE.Matrix4 }
interface Shown { words: ArrayLike<number>; turn: CubeTurn | null; swing: number }
interface Item { glyph: number; fill: number; fresh: number; mood: number }

const ease = (t: number) => t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const faceId = (axis: Axis, side: number) => 2 * axis + (side > 0 ? 1 : 0);
const axisVector = (axis: Axis, side = 1) => new THREE.Vector3().setComponent(axis, side);

function makeView(x: number, z: number, radius: number, basis: THREE.Matrix4): View {
  const scale = radius / HEX;
  return { x, z, radius, basis, world: new THREE.Matrix4().makeTranslation(x, LIFT, z).multiply(basis).multiply(new THREE.Matrix4().makeScale(scale, scale, scale)) };
}

class PocketCubeScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.32, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 2;
  protected readonly stageName = 'pocketCube';

  private model: CubeModel = { turns: [], fixed: [] };
  private budget = 0;
  private shell = 0;
  private random = false;
  private capacity = TAPE_SPAN;
  private accent = TOKENS.accents.PocketCube!;
  private views: View[] = [];
  private console: Console | null = null;
  private keySpots: THREE.Vector2[] = [];
  private atlas: Atlas | null = null;

  private cubies: THREE.InstancedMesh | null = null;
  private stick: THREE.InstancedBufferAttribute | null = null;
  private fx: THREE.InstancedBufferAttribute | null = null;
  private arrow: { mesh: THREE.Mesh; alpha: Level; turn: Level; frames: THREE.Matrix4[]; view: number[] } | null = null;
  private keys: { mesh: THREE.InstancedMesh; look: THREE.InstancedBufferAttribute; lamps: THREE.Color[] } | null = null;
  private tiles: { mesh: THREE.InstancedMesh; look: THREE.InstancedBufferAttribute } | null = null;
  private flashes: THREE.Color[] = [];
  private readonly disposables: { dispose(): void }[] = [];

  private pace = 0;
  private lastPosition = 0;
  private cheerAt = -Infinity;
  private sighAt = -Infinity;
  private clock = 0;
  private shown: Shown = { words: [0, 0, 0], turn: null, swing: 0 };
  private log: FrameUpdate['eventLog'] | null = null;
  private readonly matrix = new THREE.Matrix4();
  private readonly local = new THREE.Matrix4();
  private readonly spin = new THREE.Matrix4();
  private readonly point = new THREE.Vector3();
  private readonly blue = new THREE.Color();
  private readonly ledger = { step: 0, turn: '—', reward: 0, moves: 0, home: 0, solved: 0, scrambles: 0, looked: -Infinity };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.model = pocketCubeModel(env);
    this.budget = parameter(env, 'budget');
    this.shell = parameter(env, 'shell');
    this.random = parameter(env, 'randomState') > 0;
    this.capacity = this.budget > 0 ? this.budget : TAPE_SPAN;
    this.accent = this.accentOf(ctx);
    this.views = [makeView(-2.5, -0.4, 2.5, FRONT), makeView(3.6, -1.95, 1.42, BACK)];
    const cells = this.budget > 0 ? 2 * String(this.budget).length + 1 : LED_DIGITS;
    this.console = { x: 3.6, z: 1.55, half: new THREE.Vector2(1.55, 1.25), led: new THREE.Vector2(1.28, 0.27), ledZ: 0.72, cells: Array.from({ length: cells }, () => uniform(-1)) };
    const keys = this.model.turns.length, columns = Math.ceil(keys / 2);
    this.keySpots = Array.from({ length: keys }, (_, key) => new THREE.Vector2(3.6 + (Math.floor(key / 2) - (columns - 1) / 2) * 0.95, 1.52 + (key % 2) * 0.8));
    this.legend = this.legendFor();
    return new THREE.Box2(new THREE.Vector2(-8.2, -4.55), new THREE.Vector2(8.2, 4.55));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff1e2).multiplyScalar(0.92);
  }

  protected override poolOptions() {
    return { burstSize: 34 };
  }

  private legendFor(): LegendSpec {
    const ledger = this.ledger;
    return {
      title: 'POCKET CUBE',
      rows: [
        { key: 'turn', label: 'turn', tint: this.accent, value: () => ledger.turn },
        { key: 'reward', label: 'reward', tint: 0xffd166, value: () => ledger.reward > 0 ? `+${formatCount(ledger.reward)}` : formatCount(ledger.reward) },
        { key: 'solved', label: 'solved', tint: GOLD, value: () => this.solvedText() },
        { key: 'attempt', label: 'this attempt', tint: 0xdcf4ff, value: () => this.budget > 0 ? `${formatCount(ledger.moves)} of ${formatCount(this.budget)} turns` : `${formatCount(ledger.moves)} turns` },
        { key: 'scramble', label: 'scramble', tint: 0xb9c6d6, value: () => this.random ? 'random state' : `${formatCount(this.shell)} quarter turns deep` },
      ],
      note: () => `${ledger.home} of ${CUBE_SLOTS} corners home`,
      bar: () => ledger.home / CUBE_SLOTS,
    };
  }

  private solvedText(): string {
    const ledger = this.ledger;
    if (this.log && this.clock - ledger.looked > LOOK_SECONDS) {
      ledger.solved = countEvents(this.log, 'goal', -1, ledger.step);
      ledger.scrambles = countEvents(this.log, 'done', -1, ledger.step);
      ledger.looked = this.clock;
    }
    return ledger.solved ? `${formatCount(ledger.solved)} of ${formatCount(ledger.scrambles)} attempts` : 'none yet';
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (_frame, kit) => this.groundTruth(kit) };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    const turns = this.model.turns, console = this.console!;
    this.scene.background = new THREE.Color(0x0c0907);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const add = <T extends THREE.Mesh>(mesh: T, order: number): T => {
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return mesh;
    };
    const faces = new Set<string>();
    for (const view of this.views) for (const axis of [0, 1, 2] as const) for (const side of [1, -1]) if (axisVector(axis, side).applyMatrix4(view.basis).y > 0.3) faces.add(faceName(axis, side));
    const atlas = glyphAtlas([...new Set([...turns.map((turn) => turn.label), ...faces])]);
    this.atlas = atlas;
    this.disposables.push(atlas.texture);

    add(new THREE.Mesh(flat(60, 40), matMaterial(light, MAT, this.views, console)), 0);

    const count = SETS * this.views.length * CUBE_SLOTS;
    const stick = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const side = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const seat = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const fx = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < count; i++) {
      const view = this.views[Math.floor(i / CUBE_SLOTS) % this.views.length]!;
      side.setXYZ(i, ...CORNERS[i % CUBE_SLOTS]!);
      seat.setXYZ(i, view.x, view.z, view.radius);
    }
    const cubies = add(instancedMesh(new THREE.BoxGeometry(1, 1, 1), cubieMaterial(light, { stick, side, seat, fx }, TOWARD), count), 10);
    this.disposables.push(cubies);
    this.cubies = cubies;
    this.stick = stick;
    this.fx = fx;

    const alpha = uniform(0), turn = uniform(1);
    const arrow = add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), arrowMaterial(alpha, turn, this.accent)), 11);
    arrow.matrixAutoUpdate = false;
    arrow.visible = false;
    const frames = turns.map((t) => {
      const n = axisVector(t.axis, t.side), u = axisVector(((t.axis + 1) % 3) as Axis), v = new THREE.Vector3().crossVectors(n, u);
      return new THREE.Matrix4().makeBasis(u, v, n).setPosition(n.multiplyScalar(FACE + 0.008));
    });
    const facing = turns.map((t) => this.views.findIndex((view) => axisVector(t.axis, t.side).applyMatrix4(view.basis).y > 0.3));
    this.arrow = { mesh: arrow, alpha, turn, frames, view: facing };

    add(new THREE.Mesh(flat(2 * console.half.x + 0.2, 2 * console.half.y + 0.2), consoleMaterial(light, console, this.keySpots)), 1).position.set(console.x, 0, console.z);
    const keyLook = new THREE.InstancedBufferAttribute(new Float32Array(turns.length * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const keys = add(instancedMesh(flat(KEY_QUAD.x, KEY_QUAD.y), keyMaterial(light, atlas, keyLook, this.accent), turns.length), 2);
    this.disposables.push(keys);
    const lamps = turns.map(() => new THREE.Color(0, 0, 0));
    this.keySpots.forEach((spot, key) => {
      keys.setMatrixAt(key, this.matrix.makeTranslation(spot.x, 0, spot.y));
      keyLook.setXYZW(key, atlas.index.get(turns[key]!.label)!, 0, 0, 0);
      const lamp = light.emitter(flat(2 * KEY.x, 2 * KEY.y), uniform(lamps[key]!) as unknown as N<'vec3'>);
      lamp.position.set(spot.x, 0, spot.y);
      this.scene.add(lamp);
      this.disposables.push(lamp.geometry, lamp.material as THREE.Material);
    });
    keys.instanceMatrix.needsUpdate = true;
    this.keys = { mesh: keys, look: keyLook, lamps };

    const led = light.emitter(flat(2 * console.led.x, 2 * console.led.y), uniform(new THREE.Color(0xffa12e).multiplyScalar(0.55)) as unknown as N<'vec3'>);
    led.position.set(console.x, 0, console.ledZ);
    this.scene.add(led);
    this.disposables.push(led.geometry, led.material as THREE.Material);

    const tileLook = new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const tiles = add(instancedMesh(flat(TILE_QUAD, TILE_QUAD), tileMaterial(light, atlas, tileLook, this.accent), this.capacity), 2);
    this.disposables.push(tiles);
    const hero = this.views[0]!;
    for (let i = 0; i < this.capacity; i++) tiles.setMatrixAt(i, this.matrix.makeTranslation(hero.x + (i - (this.capacity - 1) / 2) * TAPE_PITCH, 0, hero.z + hero.radius + 0.95));
    tiles.instanceMatrix.needsUpdate = true;
    this.tiles = { mesh: tiles, look: tileLook };

    const letters: { glyph: number; x: number; z: number }[] = [];
    for (const view of this.views) for (const axis of [0, 1, 2] as const) for (const side of [1, -1]) {
      const normal = axisVector(axis, side).applyMatrix4(view.basis);
      if (normal.y <= 0.3) continue;
      const out = new THREE.Vector2(normal.x, normal.z).normalize().multiplyScalar(view.radius + 0.36);
      letters.push({ glyph: atlas.index.get(faceName(axis, side))!, x: view.x + out.x, z: view.z + out.y });
    }
    const labelLook = new THREE.InstancedBufferAttribute(new Float32Array(letters.length * 4), 4);
    const labels = add(instancedMesh(flat(0.42, 0.42), labelMaterial(atlas, labelLook), letters.length), 1);
    this.disposables.push(labels);
    letters.forEach((letter, i) => {
      labels.setMatrixAt(i, this.matrix.makeTranslation(letter.x, 0, letter.z));
      labelLook.setXYZW(i, letter.glyph, 0.5, 0, 0);
    });
    labels.instanceMatrix.needsUpdate = true;

    for (const slot of this.model.fixed) for (const view of this.views) {
      const corner = new THREE.Vector3(...CORNERS[slot]!);
      if (corner.clone().applyMatrix4(view.basis).y < Math.sqrt(3) - 1e-6) continue;
      const at = corner.multiplyScalar(FACE).applyMatrix4(view.world);
      add(new THREE.Mesh(flat(0.3, 0.3), pinMaterial(light)), 12).position.set(at.x, 0, at.z);
    }

    this.flashes = this.views.map((view) => {
      const tint = new THREE.Color(0, 0, 0), lamp = light.emitter(hexagonGeometry(view.radius * 0.96), uniform(tint) as unknown as N<'vec3'>);
      lamp.position.set(view.x, 0, view.z);
      this.scene.add(lamp);
      this.disposables.push(lamp.geometry, lamp.material as THREE.Material);
      return tint;
    });
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.cubies = null;
    this.arrow = null;
    this.keys = null;
    this.tiles = null;
    this.flashes = [];
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal, turns = this.model.turns, keys = turns.length;
    pool.update(dt, time);
    this.clock += dt;
    this.log = frame.eventLog;
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const t = frame.state.step, now = frame.frameAt(t), peek = frame.frameAt(t + 1), ahead = peek.step === t + 1 ? peek : null;
    const prior = t > 0 ? frame.frameAt(t - 1) : null, before = prior !== null && prior.step === t - 1 ? prior : null;
    const calm = this.pace < 2 * FAST * this.controlHz, u = clamp01(frame.position - t);
    if (frame.scrubbed) this.cheerAt = this.sighAt = -Infinity;
    let solved = false, ended = false;
    for (const event of frame.events) {
      if (event.type === 'goal') solved = true;
      if (event.type === 'done') ended = true;
    }
    if (solved) this.cheer(pool, time);
    else if (ended) this.lapse(pool, time);

    const closing = calm && before !== null && now.done !== 0 && u < REVEALED ? before : null;
    const reveal = closing ? smoothstep01((u - HOLD) / (REVEALED - HOLD)) : 1;
    const start = now.done !== 0 && before !== null ? REVEALED + 0.02 : TURN_FROM;
    const next = calm && ahead ? turns[cubeKey(ahead.actions, keys)]! : null;
    const swing = next ? ease(clamp01((u - start) / (1 - start))) : 0;
    const final = closing ? turns[cubeKey(now.actions, keys)]! : null;
    const won = closing !== null && now.reward > 0;
    const glow = won ? Math.max(0.28, envelope(time - this.cheerAt, CHEER_SECONDS, 1.2)) : 0;
    const dim = closing && !won ? Math.max(0.4, envelope(time - this.sighAt, SIGH_SECONDS, 0.8)) : 0;

    this.pose(0, now.words, next, swing, 0, 0, 0, closing ? reveal : 2);
    this.pose(1, closing ? closing.words : null, final, 1, glow, dim, reveal, 2);
    this.cubies!.instanceMatrix.needsUpdate = true;
    this.stick!.needsUpdate = true;
    this.fx!.needsUpdate = true;
    this.shown = closing && reveal < 0.5 ? { words: closing.words, turn: final, swing: 1 } : { words: now.words, turn: next, swing };

    const gate = now.done !== 0 && before !== null ? smoothstep01((u - REVEALED) / 0.08) : 1, approach = u < start ? 0.55 + 0.45 * smoothstep01(u / start) : 1;
    this.pointArrow(next, swing, next ? gate * approach * (1 - smoothstep01((swing - 0.6) / 0.4)) : 0);

    const pressed = t > 0 && calm ? cubeKey(now.actions, keys) : -1, coming = next ? cubeKey(ahead!.actions, keys) : -1;
    const fading = 1 - smoothstep01(u / 0.35), rising = smoothstep01((u - (start - 0.12)) / 0.12);
    const keyState = this.keys!;
    for (let key = 0; key < keys; key++) {
      const press = Math.max(key === pressed ? fading : 0, key === coming ? rising : 0, !calm && t > 0 && key === cubeKey(now.actions, keys) ? 1 : 0);
      keyState.look.setY(key, press);
      keyState.lamps[key]!.set(this.accent).multiplyScalar(1.6 * press);
    }
    keyState.look.needsUpdate = true;

    const cheer = envelope(time - this.cheerAt, CHEER_SECONDS, 1.6), sigh = envelope(time - this.sighAt, SIGH_SECONDS, 1.4);
    this.flashes.forEach((tint, i) => tint.set(GOLD).multiplyScalar((i === 0 ? 0.5 : 0.32) * Math.max(cheer, glow * 0.5)).add(this.blue.set(LAPSE).multiplyScalar(0.6 * Math.max(sigh, dim * 0.4))));

    const made = closing && reveal < 0.5 ? movesOf(closing.words) + 1 : movesOf(now.words);
    this.fillTape(frame, t, closing ? movesOf(closing.words) + 1 : made, closing !== null, won, reveal, next, swing);
    this.showCount(made);

    const ledger = this.ledger;
    ledger.step = t;
    ledger.reward = now.reward;
    ledger.moves = made;
    ledger.home = won ? CUBE_SLOTS : this.home(now.words);
    ledger.turn = t > 0 ? `${turns[cubeKey(now.actions, keys)]!.label} · key ${cubeKey(now.actions, keys) + 1}` : '—';
  }

  private home(words: ArrayLike<number>): number {
    let home = 0;
    for (let slot = 0; slot < CUBE_SLOTS; slot++) if (cubieAt(words, slot) === slot && twistAt(words, slot) === 0) home++;
    return home;
  }

  private cheer(pool: CelebrationPool, time: number): void {
    this.cheerAt = time;
    const hero = this.views[0]!;
    for (const [i, paint] of PAINT.entries()) {
      const angle = (i / PAINT.length) * Math.PI * 2;
      pool.burst({ x: hero.x + 0.6 * Math.cos(angle), y: 38, z: hero.z + 0.6 * Math.sin(angle) }, paint, 0.55, time);
    }
    pool.ripple({ x: hero.x, y: 0, z: hero.z }, GOLD, 0.8, time);
  }

  private lapse(pool: CelebrationPool, time: number): void {
    this.sighAt = time;
    const hero = this.views[0]!;
    pool.ripple({ x: hero.x, y: 0, z: hero.z }, LAPSE, 0.7, time);
  }

  private layerOf(turn: CubeTurn | null, swing: number, slot: number): THREE.Matrix4 | null {
    if (!turn || swing === 0 || CORNERS[slot]![turn.axis] !== turn.side) return null;
    return this.spin.makeRotationAxis(axisVector(turn.axis), turn.quarter * swing * Math.PI / 2);
  }

  private pose(set: number, words: ArrayLike<number> | null, turn: CubeTurn | null, swing: number, glow: number, dim: number, lo: number, hi: number): void {
    const cubies = this.cubies!, stick = this.stick!, fx = this.fx!;
    this.views.forEach((view, v) => {
      for (let slot = 0; slot < CUBE_SLOTS; slot++) {
        const i = (set * this.views.length + v) * CUBE_SLOTS + slot;
        if (!words) {
          cubies.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
          fx.setXYZW(i, 0, 0, 2, 2);
          continue;
        }
        const [x, y, z] = CORNERS[slot]!;
        this.local.makeScale(CUBIE, CUBIE, CUBIE).setPosition(x / 2, y / 2, z / 2);
        const layer = this.layerOf(turn, swing, slot);
        if (layer) this.local.premultiply(layer);
        cubies.setMatrixAt(i, this.matrix.multiplyMatrices(view.world, this.local));
        const cubie = cubieAt(words, slot), twist = twistAt(words, slot), home = faceletsOf(cubie), here = faceletsOf(slot), paint = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          const at = here[k]!, from = home[(k - twist + 3) % 3]!;
          paint[at.axis] = faceId(from.axis, from.side);
        }
        stick.setXYZ(i, paint[0]!, paint[1]!, paint[2]!);
        fx.setXYZW(i, glow, dim, lo, hi);
      }
    });
  }

  private pointArrow(next: CubeTurn | null, swing: number, alpha: number): void {
    const arrow = this.arrow!;
    const key = next ? this.model.turns.indexOf(next) : -1, view = key >= 0 ? this.views[arrow.view[key]!] : undefined;
    arrow.mesh.visible = alpha > 0.002 && view !== undefined;
    if (!arrow.mesh.visible || !next || !view) return;
    arrow.alpha.value = alpha;
    arrow.turn.value = next.quarter === next.side ? -1 : 1;
    this.local.makeRotationAxis(axisVector(next.axis), next.quarter * swing * Math.PI / 2).multiply(arrow.frames[key]!);
    arrow.mesh.matrix.multiplyMatrices(view.world, this.local);
    arrow.mesh.matrixWorldNeedsUpdate = true;
  }

  private fillTape(frame: FrameUpdate, t: number, made: number, closing: boolean, won: boolean, reveal: number, next: CubeTurn | null, swing: number): void {
    const tiles = this.tiles!, atlas = this.atlas!, turns = this.model.turns, keys = turns.length;
    const items: Item[] = [];
    const upcoming = !closing && next !== null && swing > 0;
    const room = this.capacity - (upcoming ? 1 : 0), first = t - Math.min(made, room) + 1;
    for (let s = first; s <= t; s++) {
      const f: EnvFrame = frame.frameAt(s);
      if (f.step !== s) continue;
      items.push({ glyph: atlas.index.get(turns[cubeKey(f.actions, keys)]!.label)!, fill: closing ? 1 - reveal : 1, fresh: 0, mood: closing ? (won ? 1 : -1) * (1 - reveal) : 0 });
    }
    if (upcoming) items.push({ glyph: atlas.index.get(next.label)!, fill: Math.min(1, 1.15 * Math.sin(Math.min(1, swing * 1.4) * Math.PI * 0.62) / Math.sin(Math.PI * 0.62)), fresh: 1, mood: 0 });
    else if (items.length && !closing) items[items.length - 1]!.fresh = 0.6;
    for (let i = 0; i < this.capacity; i++) {
      const item = items[i];
      tiles.look.setXYZW(i, item ? item.glyph : -1, item ? Math.max(item.fill, 0) : 0, item ? item.fresh : 0, item ? item.mood : 0);
    }
    tiles.look.needsUpdate = true;
  }

  private showCount(made: number): void {
    const cells = this.console!.cells, digits = (value: number, width: number) => String(value).padStart(width, ' ').slice(-width).split('').map((c) => (c === ' ' ? -1 : Number(c)));
    const values = this.budget > 0 ? [...digits(made, String(this.budget).length), SLASH, ...digits(this.budget, String(this.budget).length)] : digits(made, cells.length);
    values.forEach((value, i) => { cells[i]!.value = value; });
  }

  private groundTruth(kit: DebugKit): void {
    const shown = this.shown, hero = this.views[0]!;
    const marks = kit.marks('reference', this.views.length * CUBE_SLOTS);
    const place = (view: THREE.Matrix4, slot: number, offset: THREE.Vector3, moving = true): THREE.Vector3 => {
      const [x, y, z] = CORNERS[slot]!;
      this.point.set(x / 2, y / 2, z / 2).add(offset);
      const layer = moving ? this.layerOf(shown.turn, shown.swing, slot) : null;
      if (layer) this.point.applyMatrix4(layer);
      return this.point.applyMatrix4(view);
    };
    for (const view of this.views) for (let slot = 0; slot < CUBE_SLOTS; slot++) {
      const f = faceletsOf(slot)[twistAt(shown.words, slot)]!, out = axisVector(f.axis, f.side).multiplyScalar(CUBIE / 2 + 0.02);
      const normal = axisVector(f.axis, f.side), layer = this.layerOf(shown.turn, shown.swing, slot);
      if (layer) normal.transformDirection(layer);
      if (normal.transformDirection(view.basis).y < 0.2) continue;
      const at = place(view.world, slot, out);
      marks.set(at.x, 30, at.z, 0.07 * view.radius, TOKENS.uiCyan);
    }
    for (let slot = 0; slot < CUBE_SLOTS; slot++) {
      const cubie = cubieAt(shown.words, slot), line = kit.curve(`home${slot}`, 2, TOKENS.cueMagenta);
      if (cubie === slot) continue;
      const a = place(hero.world, slot, new THREE.Vector3()).clone(), b = place(hero.world, cubie, new THREE.Vector3(), false);
      line.point(a.x, 30, a.z);
      line.point(b.x, 30, b.z);
    }
    const axis = kit.curve('axis', 2, this.accent);
    if (shown.turn) {
      const reach = axisVector(shown.turn.axis, shown.turn.side).multiplyScalar(1.7), a = reach.clone().applyMatrix4(hero.world), b = reach.negate().applyMatrix4(hero.world);
      axis.point(a.x, 30, a.z);
      axis.point(b.x, 30, b.z);
    }
    const pins = kit.marks('fixed', this.model.fixed.length);
    for (const slot of this.model.fixed) {
      const back = this.views[this.views.length - 1]!, at = new THREE.Vector3(...CORNERS[slot]!).multiplyScalar(FACE).applyMatrix4(back.world);
      pins.set(at.x, 31, at.z, 0.1, GOLD);
    }
  }
}

export default (() => new PocketCubeScene()) satisfies SceneFactory;
