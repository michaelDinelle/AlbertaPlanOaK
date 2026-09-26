import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { keyDoorTake } from '@sim/envs/keydoor';
import { SIM_EVENT_TYPES, type EnvFrame, type EventStream } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  AMBER, ANNEX, DOOR, FIGURE, FLAME, FROST, GOLD, HAND, HINGE, KEY, LANTERN, LEDGER, LOCK, MIDDLE, MISS, PLINTH, SPAN, WALL,
  annex, backdrop, candle, disc, doorLeaf, flagstones, flat, keySprite, ledger, plinth, tint, traveler, treasure, vectorOf, walls, type Comet,
} from './art';

interface Loop { w: number; h: number; x: Int32Array; y: Int32Array }
interface Run { x0: number; y0: number; x1: number; y1: number }
type Kind = 'rest' | 'walk' | 'take' | 'door';

const WEST = Math.PI;
const LAYOUT = 2.2;
const HOLD = 0.45;
const FADE = 1.5;
const WAVE = 3.2;
const FAST = 5;
const SWING = 1.62;
const CARRY = 0.78;
const BURST_Y = 38;
const LANTERN_POWER = 0.3;
const KEY_POWER = 0.09;
const NOOK_POWER = 0.12;
const VAULT_POWER = 3.6;
const DOOR_KIND = SIM_EVENT_TYPES.indexOf('door');
const GOLDEN = vectorOf(GOLD);
const COLD = vectorOf(MISS);

const ease = (v: number, from: number, span: number) => smoothstep01((v - from) / span);
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => formatSigned(value).replace('-', '−');
const steps = (n: number) => `${formatCount(n)} step${n === 1 ? '' : 's'}`;

function doorCurve(v: number): number {
  if (v < -0.55) return 0;
  if (v < 0.8) return ease(v, -0.55, 1.35);
  if (v < 2.6) return 1;
  return 1 - ease(v, 2.6, 1.6);
}

function loopOf(delay: number): Loop {
  let w = 0, h = 0, best = 0;
  for (let rows = 2; rows * 2 <= delay; rows += 2) {
    if (delay % rows) continue;
    const cols = delay / rows, fit = Math.min(LAYOUT / (cols + ANNEX + 0.8), 1 / (rows + 0.8));
    if (fit > best) [best, w, h] = [fit, cols, rows];
  }
  if (!best) throw new Error(`KeyDoor delay ${delay} does not fold into a closed serpentine`);
  const x = new Int32Array(delay), y = new Int32Array(delay);
  let n = 0;
  const put = (cx: number, cy: number) => { x[n] = cx; y[n++] = cy; };
  for (let cx = 0; cx < w; cx++) put(cx, 0);
  for (let cy = 1; cy < h; cy++) for (let k = 1; k < w; k++) put(cy % 2 ? w - k : k, cy);
  for (let cy = h - 1; cy >= 1; cy--) put(0, cy);
  return { w, h, x, y };
}

function wallRuns({ w, h, x, y }: Loop, scale: number): Run[] {
  const delay = x.length, at = new Int32Array(w * h), runs: Run[] = [];
  for (let p = 0; p < delay; p++) at[y[p]! * w + x[p]!] = p;
  const linked = (a: number, b: number) => { const d = Math.abs(at[a]! - at[b]!); return d === 1 || d === delay - 1; };
  for (let r = 0; r <= h; r++) {
    let start = -1;
    for (let c = 0; c <= w; c++) {
      const wall = c < w && (r === 0 || r === h || !linked((r - 1) * w + c, r * w + c));
      if (wall && start < 0) start = c;
      if (!wall && start >= 0) {
        runs.push({ x0: start, y0: r, x1: c, y1: r });
        start = -1;
      }
    }
  }
  for (let c = 0; c <= w; c++) {
    let start = -1;
    for (let r = 0; r <= h; r++) {
      const wall = r < h && (c === w || (c === 0 ? r >= 2 : !linked(r * w + c - 1, r * w + c)));
      if (wall && start < 0) start = r;
      if (!wall && start >= 0) {
        runs.push({ x0: c, y0: start, x1: c, y1: r });
        start = -1;
      }
    }
  }
  const west = -ANNEX * scale, top = 1 - scale, bottom = 1 + scale;
  runs.push({ x0: west, y0: top, x1: 0, y1: top }, { x0: west, y0: 1, x1: 0, y1: 1 }, { x0: west, y0: bottom, x1: 0, y1: bottom }, { x0: west, y0: top, x1: west, y1: bottom });
  if (top < 0) runs.push({ x0: 0, y0: top, x1: 0, y1: 0 });
  return runs;
}

function slabs(runs: readonly Run[], w: number, h: number, thick: number): THREE.BufferGeometry {
  const positions: number[] = [], lines: number[] = [], index: number[] = [];
  for (const run of runs) {
    const x0 = run.x0 - thick / 2 - w / 2, x1 = run.x1 + thick / 2 - w / 2, z0 = run.y0 - thick / 2 - h / 2, z1 = run.y1 + thick / 2 - h / 2, k = positions.length / 3;
    const vertical = run.x0 === run.x1, line = vertical ? run.x0 - w / 2 : run.y0 - h / 2;
    positions.push(x0, 0, z0, x1, 0, z0, x1, 0, z1, x0, 0, z1);
    for (let n = 0; n < 4; n++) lines.push(vertical ? 1 : 0, line);
    index.push(k, k + 3, k + 2, k, k + 2, k + 1);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('aRun', new THREE.Float32BufferAttribute(lines, 2));
  geometry.setIndex(index);
  return geometry;
}

interface Parts {
  figure: THREE.Mesh;
  held: THREE.Mesh;
  resting: THREE.Mesh;
  door: THREE.Group;
  lantern: THREE.Mesh;
  keyLight: THREE.Mesh;
  vaultLight: THREE.Mesh;
}

class KeyDoorScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'keyDoor';

  private delay = 1;
  private loop: Loop = { w: 1, h: 1, x: new Int32Array(1), y: new Int32Array(1) };
  private unit = 1;
  private texel = 0.05;
  private scale = 1;
  private accent = TOKENS.accents.KeyDoor!;
  private cells: THREE.Vector3[] = [];
  private headings = new Float32Array(1);
  private rewards = new Float32Array(1);
  private previous = new Float32Array(1);
  private data: THREE.DataTexture | null = null;
  private parts: Parts | null = null;
  private keyScale = 1;
  private ledgerValues: number[] = [];
  private readonly owned: { dispose(): void }[] = [];

  private readonly comet: Comet = { head: uniform(0), tail: uniform(2), strength: uniform(0), tint: tint(GOLD) };
  private readonly open = uniform(0);
  private readonly sway = uniform(0);
  private readonly lamp = uniform(2.6);
  private readonly heldFade = uniform(0);
  private readonly heldShine = uniform(0.35);
  private readonly restFade = uniform(1);
  private readonly lanternGlow = uniform(0);
  private readonly keyGlow = uniform(0);
  private readonly nookGlow = uniform(0);
  private readonly flame = uniform(2.4);
  private readonly vaultGlow = uniform(0);

  private readonly cushionAt = new THREE.Vector3();
  private readonly hinge = new THREE.Vector3();
  private readonly doorway = new THREE.Vector3();
  private readonly shown = new THREE.Vector3();
  private readonly hand = new THREE.Vector3();
  private readonly lock = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly axis = new THREE.Vector3(0, 1, 0);

  private surveyed = -1;
  private lapStart = 0;
  private hasPrevious = false;
  private pace = 0;
  private lastPosition = 0;
  private readonly lap = { noise: 0, door: 0 };
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, opened: 0, locked: 0 };
  private readonly drawn = { step: -1, count: -1 };
  private readonly row = { step: 0, phase: 0, held: 0, action: '—', reward: 0 };

  override legend: LegendSpec = {
    title: 'KEY DOOR',
    rows: [
      { key: 'key', label: 'key', tint: this.accent, value: () => (this.row.held ? 'in hand' : 'on the cushion') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.row.action },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(this.row.reward) },
      { key: 'noise', label: 'lap noise', tint: AMBER, value: () => signed(this.lap.noise) },
      { key: 'doors', label: 'doors', tint: GOLD, value: () => `${formatCount(this.tally.opened)} opened · ${formatCount(this.tally.locked)} locked` },
    ],
    note: () => `lap ${formatCount(Math.floor(this.row.step / this.delay) + 1)} · ${this.row.phase === 0 ? 'at the key' : `door in ${steps(this.delay - this.row.phase)}`}`,
    bar: () => this.row.phase / this.delay,
  };

  private world(gx: number, gy: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(gx - this.loop.w / 2, 0, gy - this.loop.h / 2);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.delay = parameter(ctx.environment, 'delay');
    if (!Number.isInteger(this.delay) || this.delay < 4) throw new Error(`${ctx.environment.envId} has delay ${this.delay}`);
    const aspect = ctx.size.width / Math.max(1, ctx.size.height);
    this.loop = loopOf(this.delay);
    const { w, h } = this.loop;
    this.scale = Math.max(1, h / 8);
    const west = -w / 2 - ANNEX * this.scale - WALL / 2, east = w / 2 + WALL / 2, top = -h / 2 + Math.min(0, 1 - this.scale) - WALL / 2, south = h / 2 + WALL / 2;
    const view = Math.max(south - top, (east - west) / aspect) * 1.12;
    this.unit = view / 9;
    this.texel = view / Math.max(1, ctx.size.height * ctx.size.dpr * 0.5);
    const pad = 0.22 * this.unit, north = Math.min(top, -h / 2 - WALL / 2 - 0.62 * this.unit);
    return new THREE.Box2(new THREE.Vector2(west - pad, north - pad), new THREE.Vector2(east + pad, south + pad));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x7d8bb0).multiplyScalar(0.42);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, head = frame.frameAt(i), phase = head.words[0]!, held = head.words[1]!, start = i - phase;
        const route = kit.curve('loop', this.delay + 1, TOKENS.uiCyan);
        for (let p = 0; p <= this.delay; p++) {
          const cell = this.cells[p % this.delay]!;
          route.point(cell.x, 0.05, cell.z);
        }
        const cell = this.cells[phase]!;
        kit.marks('traveler', 1).set(cell.x, 0.1, cell.z, 0.17, this.accent);
        at.copy(held ? cell : this.cushionAt);
        kit.marks('key', 1).set(at.x + (held ? 0.2 : 0), 0.1, at.z, 0.1, GOLD);
        const noise = kit.marks('noise', this.delay);
        for (let p = 1; p <= phase; p++) {
          const f = frame.frameAt(start + p);
          if (f.step !== start + p) continue;
          const c = this.cells[p]!;
          noise.set(c.x + 0.28, 0.1, c.z + 0.28, 0.03 + 0.5 * Math.sqrt(Math.abs(f.reward)), f.reward >= 0 ? AMBER : FROST);
        }
        const door = start > 0 ? frame.frameAt(start) : null, known = door !== null && door.step === start;
        kit.marks('door', 1).set(this.doorway.x, 0.1, this.doorway.z, 0.16, !known ? 0x59606e : door.reward > 0 ? GOLD : MISS);
        if (i > start) {
          const choice = frame.frameAt(start + 1);
          if (choice.step === start + 1) kit.marks('choice', 1).set(this.cushionAt.x, 0.1, this.cushionAt.z - 0.3, 0.08, choice.words[1] ? GOLD : MISS);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const { w, h, x, y } = this.loop, delay = this.delay, lit = light.lit;
    this.accent = this.accentOf(ctx);
    this.cells = Array.from({ length: delay }, (_, p) => new THREE.Vector3(x[p]! + 0.5 - w / 2, 0, y[p]! + 0.5 - h / 2));
    this.headings = Float32Array.from({ length: delay }, (_, p) => {
      const q = (p + 1) % delay;
      return Math.atan2(-(y[q]! - y[p]!), x[q]! - x[p]!);
    });
    this.rewards = new Float32Array(delay);
    this.previous = new Float32Array(delay);
    const S = this.scale, AW = ANNEX * S;
    this.world(-AW / 2, 1 - S / 2, this.cushionAt);
    this.world(0, HINGE, this.hinge);
    this.world(0, 1.5, this.doorway);

    const texels = new Float32Array(w * h * 4);
    for (let p = 0; p < delay; p++) {
      const q = (p + 1) % delay, at = (y[p]! * w + x[p]!) * 4;
      texels[at + 2] = p;
      texels[at + 3] = x[q]! > x[p]! ? 0 : y[q]! > y[p]! ? 1 : x[q]! < x[p]! ? 2 : 3;
    }
    const data = new THREE.DataTexture(texels, w, h, THREE.RGBAFormat, THREE.FloatType);
    data.minFilter = data.magFilter = THREE.NearestFilter;
    data.needsUpdate = true;
    this.data = data;

    const add = (geometry: THREE.BufferGeometry, material: THREE.Material, order: number, at?: THREE.Vector3) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      if (at) mesh.position.copy(at);
      this.owned.push(geometry, material);
      this.scene.add(mesh);
      return mesh;
    };
    const emit = (radius: number, color: number, level: THREE.UniformNode<'float', number>) => {
      const geometry = disc(radius), mesh = light.emitter(geometry, rgb(color).mul(level));
      this.owned.push(geometry, mesh.material as THREE.Material);
      this.scene.add(mesh);
      return mesh;
    };
    const glowRadius = (base: number) => Math.max(base, 2.2 * this.texel);

    add(flat(400, 400), backdrop(), 0);
    add(flat(w, h), flagstones(w, h, data, this.comet, this.accent, lit), 1);
    add(flat(AW, 2 * S), annex(w, h, lit, this.open), 1, this.world(-AW / 2, 1, new THREE.Vector3()));
    const runs = wallRuns(this.loop, S);
    add(slabs(runs, w, h, WALL), walls(lit), 3);
    const shade = slabs(runs, w, h, Math.max(WALL, 3.2 * this.texel)), occluder = light.occluder(shade);
    this.owned.push(shade, occluder.material as THREE.Material);
    this.scene.add(occluder);

    const vault = this.world(-AW / 2, 1 + S / 2, new THREE.Vector3());
    add(flat(0.8, 0.8), treasure(lit, this.open), 4, vault).scale.setScalar((Math.min(AW, S) - WALL) / 0.8);
    const vaultLight = light.emitter(flat(AW - WALL - 0.12, S - WALL - 0.12), rgb(GOLD).mul(this.vaultGlow));
    vaultLight.position.copy(vault);
    this.owned.push(vaultLight.geometry, vaultLight.material as THREE.Material);
    this.scene.add(vaultLight);
    add(flat(PLINTH, PLINTH), plinth(lit), 4, this.cushionAt).scale.setScalar(S * Math.min(1, (ANNEX - WALL - 0.04) / PLINTH));
    const reach = 1.4 + 1.1 * this.unit, nookRadius = glowRadius(0.03 * S), nook = this.world(-AW + WALL / 2 + 0.13 * S, 1 - S + WALL / 2 + 0.12 * S, new THREE.Vector3());
    add(flat(0.24, 0.24), candle(this.accent, lit, this.flame), 5, nook).scale.setScalar(S);
    emit(nookRadius, FLAME, this.nookGlow).position.copy(nook);
    this.nookGlow.value = (NOOK_POWER * reach) / nookRadius;

    const door = new THREE.Group();
    door.position.copy(this.hinge);
    const leafGeometry = flat(DOOR + 0.1, SPAN).translate(0, 0, MIDDLE), leaf = new THREE.Mesh(leafGeometry, doorLeaf(this.accent, lit));
    leaf.renderOrder = 5;
    leaf.frustumCulled = false;
    const leafShade = flat(Math.max(DOOR, 3.2 * this.texel), SPAN + 0.04).translate(0, 0, MIDDLE), leafOccluder = light.occluder(leafShade);
    door.add(leaf, leafOccluder);
    this.owned.push(leafGeometry, leaf.material as THREE.Material, leafShade, leafOccluder.material as THREE.Material);
    this.scene.add(door);

    const resting = add(flat(KEY, KEY), keySprite(this.accent, this.restFade, uniform(0.2)), 6, this.cushionAt);
    resting.rotation.y = 0.55;
    resting.scale.setScalar(S);
    const figure = add(flat(FIGURE, FIGURE), traveler(lit, this.sway, this.lamp), 7);
    const held = add(flat(KEY, KEY), keySprite(this.accent, this.heldFade, this.heldShine), 8);
    const lanternRadius = glowRadius(0.045), keyRadius = glowRadius(0.04);
    const lantern = emit(lanternRadius, FLAME, this.lanternGlow), keyLight = emit(keyRadius, this.accent, this.keyGlow);
    vaultLight.visible = keyLight.visible = false;
    this.lanternGlow.value = (LANTERN_POWER * reach) / lanternRadius;
    this.keyScale = (KEY_POWER * reach) / keyRadius;

    const spacing = 0.26 * this.unit, radius = 0.075 * this.unit, board = ledger(spacing, radius, this.accent, lit);
    this.ledgerValues = board.values;
    add(flat(spacing * LEDGER, radius * 4), board.material, 3, this.world(0.15 + (spacing * LEDGER) / 2, -WALL / 2 - 0.3 * this.unit, new THREE.Vector3()));

    this.parts = { figure, held, resting, door, lantern, keyLight, vaultLight };
    pool.group.traverse((object) => { object.renderOrder = 10; });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const parts = this.parts, data = this.data;
    if (!parts || !data) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const delay = this.delay, i = frame.state.step, head = frame.frameAt(i);
    if (head.step !== i) return;
    const ahead = frame.frameAt(i + 1), next = ahead.step === i + 1 ? ahead : null, u = next ? clamp01(frame.position - i) : 0;
    const phase = head.words[0]!, held = head.words[1] === 1;
    if (phase >= delay) throw new Error(`KeyDoor phase ${phase} is outside its delay ${delay}`);
    if (i !== this.surveyed) this.survey(frame, i, phase);
    const kind: Kind = !next ? 'rest' : next.done ? 'door' : phase === 0 && next.words[1] === 1 ? 'take' : 'walk';

    const from = this.cells[phase]!, to = next ? this.cells[next.words[0]!]! : from;
    const move = fast ? u : kind === 'walk' ? ease(u, 0.08, 0.84) : kind === 'rest' ? 0 : ease(u, HOLD, 0.98 - HOLD);
    this.shown.lerpVectors(from, to, move);
    const inbound = this.headings[(phase + delay - 1) % delay]!, outbound = this.headings[phase]!;
    const heading = fast || kind === 'walk' ? turn(inbound, outbound, ease(u, 0, 0.45))
      : kind === 'rest' ? inbound
      : u < HOLD ? turn(inbound, WEST, ease(u, 0, 0.25)) : turn(WEST, outbound, ease(u, HOLD, 0.25));
    this.sway.value = Math.sin(Math.PI * move) * (phase % 2 ? 1 : -1);
    const figure = parts.figure;
    figure.position.copy(this.shown);
    figure.rotation.y = heading;
    const bob = 1 + 0.04 * Math.sin(Math.PI * move);
    figure.scale.setScalar(bob);
    this.hand.set(HAND.x, 0, -HAND.y).applyAxisAngle(this.axis, heading).multiplyScalar(bob).add(this.shown);
    parts.lantern.position.set(LANTERN.x, 0, -LANTERN.y).applyAxisAngle(this.axis, heading).multiplyScalar(bob).add(this.shown);
    this.lamp.value = 2.6 + 0.25 * Math.sin(time * 11.3) * Math.sin(time * 4.1);
    this.flame.value = 2.4 + 0.3 * Math.sin(time * 9.7 + 1.3) * Math.sin(time * 3.3);

    let D = -1, opened = false;
    if (next && next.done) {
      D = i + 1;
      opened = next.reward > 0;
    } else if (this.lapStart > 0) {
      D = this.lapStart;
      opened = this.lap.door > 0;
    }
    const v = D >= 0 ? (frame.position - D) / Math.min(1, delay / 8) : Infinity;
    const open = opened ? doorCurve(v) : 0;
    const rattle = !opened && D === i + 1 && !fast ? 0.07 * Math.sin(u * 44) * Math.sin(Math.PI * clamp01((u - 0.06) / 0.4)) : 0;
    this.open.value = open;
    this.vaultGlow.value = open * VAULT_POWER;
    parts.vaultLight.visible = open > 0.001;
    parts.door.rotation.y = -SWING * open + rattle;
    this.lock.set(LOCK.x, 0, LOCK.y).applyAxisAngle(this.axis, parts.door.rotation.y).add(this.hinge);

    const heldMesh = parts.held, resting = parts.resting;
    let carried = held ? 1 : 0, rest = held ? 0 : 1, shine = 0.35, squash = 1;
    heldMesh.position.copy(this.hand);
    heldMesh.rotation.y = heading + 0.5;
    if (kind === 'take') {
      const k = fast ? 1 : ease(u, 0.06, 0.34), lift = 1 + 0.35 * Math.sin(Math.PI * k);
      heldMesh.position.lerpVectors(this.cushionAt, this.hand, k);
      heldMesh.rotation.y = turn(resting.rotation.y, heading + 0.5, k);
      heldMesh.scale.setScalar(lift * (this.scale + (CARRY - this.scale) * k));
      carried = 1;
      rest = 0;
      shine = 0.35 + 0.65 * Math.sin(Math.PI * k);
    } else if (kind === 'door' && held && opened && !fast) {
      const reach = ease(u, 0.05, 0.25), twist = ease(u, 0.3, 0.15);
      this.scratch.set(-0.13, 0, 0).applyAxisAngle(this.axis, WEST + parts.door.rotation.y).add(this.lock);
      heldMesh.position.lerpVectors(this.hand, this.scratch, reach);
      heldMesh.rotation.y = turn(heading + 0.5, WEST + parts.door.rotation.y, reach);
      heldMesh.scale.setScalar(CARRY);
      squash = 1 - 0.75 * twist;
      carried = 1 - ease(u, 0.5, 0.4);
      rest = ease(u, 0.55, 0.4);
      shine = 0.35 + 0.9 * twist;
    } else if (kind === 'door' && held) {
      carried = 1 - ease(u, 0.3, 0.5);
      rest = ease(u, 0.55, 0.4);
      heldMesh.scale.setScalar(CARRY);
    } else heldMesh.scale.setScalar(CARRY);
    heldMesh.scale.z *= squash;
    heldMesh.visible = carried > 0.001;
    resting.visible = rest > 0.001;
    this.heldFade.value = carried;
    this.heldShine.value = shine;
    this.restFade.value = rest;
    parts.keyLight.position.copy(heldMesh.position);
    parts.keyLight.visible = carried > 0.01;
    this.keyGlow.value = this.keyScale * carried * (0.8 + 0.4 * shine);

    this.paint(frame, phase, next, u, move, fast, data);
    this.credit(D, v, opened);
    if (!frame.scrubbed && !fast) this.celebrate(frame, pool, time);

    this.ledgerRead(frame.eventLog, i);
    this.score(frame.eventLog, i);
    this.row.step = i;
    this.row.phase = phase;
    this.row.held = held ? 1 : 0;
    this.row.reward = head.reward;
    this.row.action = this.actionOf(frame, i, head);
  }

  private survey(frame: FrameUpdate, i: number, phase: number): void {
    const start = i - phase, delay = this.delay, rewards = this.rewards;
    let complete = true, noise = 0;
    const reward = (step: number): number => {
      const f = frame.frameAt(step);
      if (f.step === step) return f.reward;
      complete = false;
      return 0;
    };
    this.lapStart = start;
    rewards.fill(0);
    for (let p = 1; p <= phase; p++) noise += rewards[p] = reward(start + p);
    this.lap.noise = noise;
    const door = start > 0 ? frame.frameAt(start) : null;
    if (door && door.step !== start) complete = false;
    this.lap.door = door && door.step === start ? (door.reward > 0 ? 1 : -1) : 0;
    this.hasPrevious = start > 0 && phase <= FADE + 1;
    if (this.hasPrevious) for (let p = 1; p < delay; p++) this.previous[p] = reward(start - delay + p);
    this.surveyed = complete ? i : -1;
  }

  private paint(frame: FrameUpdate, phase: number, next: EnvFrame | null, u: number, move: number, fast: boolean, data: THREE.DataTexture): void {
    const { w, x, y } = this.loop, texels = data.image.data as Float32Array, delay = this.delay;
    const entering = next && !next.done ? next.words[0]! : -1, arrive = fast ? u : ease(move, 0.55, 0.4);
    const fade = this.hasPrevious ? 1 - ease(frame.position - this.lapStart, 0, FADE) : 0;
    for (let p = 0; p < delay; p++) {
      const at = (y[p]! * w + x[p]!) * 4;
      let glint = 0, seen = 1;
      if (p === 0 || p <= phase) glint = this.rewards[p]!;
      else {
        const old = this.hasPrevious ? this.previous[p]! * fade : 0;
        seen = fade;
        if (p === entering) {
          glint = next!.reward * arrive + old * (1 - arrive);
          seen = arrive + fade * (1 - arrive);
        } else glint = old;
      }
      texels[at] = glint;
      texels[at + 1] = seen;
    }
    data.needsUpdate = true;
  }

  private credit(D: number, v: number, opened: boolean): void {
    const comet = this.comet;
    if (D < 0 || !(v >= -0.45 && v < WAVE + 0.8)) {
      comet.strength.value = 0;
      return;
    }
    comet.head.value = (this.delay - 1) * (1 - ease(v, -0.45, WAVE + 0.45));
    comet.tail.value = Math.max(1.5, this.delay / 12);
    comet.strength.value = (opened ? 1 : 0.5) * (1 - ease(v, WAVE, 0.8));
    comet.tint.value.copy(opened ? GOLDEN : COLD);
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    for (const event of frame.events) {
      if (event.type === 'door' && event.open) {
        pool.burst(this.scratch.copy(this.doorway).setY(BURST_Y), GOLD, Math.min(1, 0.45 + 0.2 * this.unit), time);
        pool.ripple(this.scratch.copy(this.doorway).setY(0), GOLD, 0.3 + 0.27 * this.unit, time);
      } else if (event.type === 'door') pool.burst(this.scratch.copy(this.doorway).setY(BURST_Y), MISS, 0.4, time);
      else if (event.type === 'key') pool.burst(this.scratch.copy(this.cushionAt).setY(BURST_Y), this.accent, 0.35, time);
    }
  }

  private actionOf(frame: FrameUpdate, i: number, head: EnvFrame): string {
    if (i < 1) return '—';
    const before = frame.frameAt(i - 1), take = keyDoorTake(head);
    if (before.step !== i - 1) return take ? 'grab' : 'pass';
    if (before.words[0] === 0) return take ? 'grab · key taken' : 'pass · key left';
    return take ? 'grab · nothing here' : 'pass';
  }

  private ledgerRead(log: EventStream, at: number): void {
    if (this.drawn.step === at && this.drawn.count === log.count) return;
    this.drawn.step = at;
    this.drawn.count = log.count;
    const values = this.ledgerValues;
    values.fill(0);
    let k = 0;
    for (let b = Math.floor(at / log.blockFrames); b >= 0 && k < LEDGER; b--) {
      const block = log.blocks[b];
      if (!block) break;
      for (let n = block.steps.length - 1; n >= 0 && k < LEDGER; n--) if (block.steps[n]! <= at && block.kinds[n] === DOOR_KIND) values[k++] = block.values[n]! >>> 16 ? 1 : -1;
    }
  }

  private score(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, opened: 0, locked: 0 });
    if (at === tally.through) return;
    tally.opened += countEvents(log, 'goal', tally.through, at);
    tally.locked += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }

  protected override drop(): void {
    for (const item of this.owned) item.dispose();
    this.owned.length = 0;
    this.data?.dispose();
    this.data = null;
    this.parts = null;
  }
}

export default (() => new KeyDoorScene()) satisfies SceneFactory;
