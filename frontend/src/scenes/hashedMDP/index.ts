import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BIAS, DEPTH, DIGIT_H, DIGIT_W, DISPLAY_X, DISPLAY_Z, EXTENT, HALF, HEX, KEY, KEY_X, LAMP_GAP, REACH, ROW, ROWS, SHARE, SIDE, SPINE, TAPE_Z0, TAPE_Z1,
  arrayMaterial, barMaterial, digitMaterial, flat, groundMaterial, keyMaterial, keyZ, lampMaterial, layoutOf, lineMaterial, reticleMaterial, shareZ, treeMaterial,
  type Layout, type Level, type LineStyle,
} from './art';
import { History, type Edge, type Loop } from './history';

type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

const HISTORY = 4096;
const TRAIL = 256;
const TRAIL_TAU = 24;
const LEAP_TAU = 10;
const RATE = 1000;
const LOOP_LONGEST = 64;
const HOP = 0.55;
const FLARE_LIFE = 0.8;
const PRESS_LIFE = 0.55;
const FLIP_LIFE = 0.6;
const RETRY = 0.35;
const BURST_GAP = 0.9;
const DEBUG_PATH = 64;
const DIGIT_GAP = 0.26;
const GOLD = TOKENS.celebrationGold;
const LOSS = TOKENS.failureBlue;
const PATH: LineStyle = { pad: 0.16, halo: 0.22, gain: 1.7, gradient: false };
const TRACE: LineStyle = { pad: 0.14, halo: 0.28, gain: 1.7, gradient: true };
const Y = new THREE.Vector3(0, 1, 0);

interface Batch { mesh: THREE.InstancedMesh; shape: THREE.InstancedBufferAttribute; style: LineStyle | null; used: number }
interface Hop { from: number; to: number; t: number }

const signed = (value: number) => formatSigned(value).replace('-', '−');
const hex = (value: number, bits: number) => `0x${value.toString(16).toUpperCase().padStart(bits / 4, '0')}`;
const popcount = (value: number) => {
  let count = 0;
  for (let v = value >>> 0; v; v &= v - 1) count++;
  return count;
};

class HashedScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 3;
  protected readonly stageName = 'hashedMDP';

  private layout: Layout = layoutOf(8);
  private local = true;
  private actionCount = 4;
  private accent = new THREE.Color();
  private history: History | null = null;
  private heat = new Float32Array(0);
  private readonly touched: number[] = [];
  private charge: THREE.DataTexture | null = null;
  private tau = 400;
  private readonly edges = new Map<number, Edge>();
  private readonly shares = new Float64Array(4);
  private paths: Batch | null = null;
  private trail: Batch | null = null;
  private bars: Batch | null = null;
  private lamps: Batch | null = null;
  private keys: Batch | null = null;
  private digits: Batch | null = null;
  private reticle: THREE.Mesh | null = null;
  private readonly flash: Level = uniform(0);
  private agentLight: THREE.Mesh | null = null;
  private columnLight: THREE.Mesh | null = null;
  private rowLight: THREE.Mesh | null = null;
  private keyLight: THREE.Mesh | null = null;
  private readonly keyTint: Tint = uniform(new THREE.Vector3());
  private flareLight: THREE.Mesh | null = null;
  private readonly flareTint: Tint = uniform(new THREE.Vector3());
  private readonly lampTints: Tint[] = [];

  private step = -1;
  private simTime = 0;
  private clock = 0;
  private retried = -Infinity;
  private hop: Hop = { from: 0, to: 0, t: 1 };
  private pressed = -1;
  private pressedAt = -Infinity;
  private flippedAt = -Infinity;
  private flipped = 0;
  private readonly flare = { at: new THREE.Vector3(), born: -Infinity, color: new THREE.Color(), strength: 0 };
  private burstAt = -Infinity;
  private scale = 1;
  private loop: Loop | null = null;
  private readonly ledger = { state: 0, action: -1, diff: 0, reward: 0, mean: NaN, paid: NaN, window: 0, distinct: 0 };

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly scaleVector = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly shown = new THREE.Vector3();
  private readonly color = new THREE.Color();
  private readonly gold = new THREE.Color(GOLD);
  private readonly loss = new THREE.Color(LOSS);
  private readonly white = new THREE.Color(0xffffff);
  private readonly silver = new THREE.Color(0xc8cfe6);
  private readonly route: number[] = [];
  private readonly distinct = new Set<number>();

  override legend: LegendSpec = {
    title: 'HASHED MDP',
    rows: [
      { key: 'state', label: 'state', tint: TOKENS.accents.HashedMDP!, value: () => `${hex(this.ledger.state, this.layout.bits)} · #${this.ledger.state}` },
      { key: 'key', label: 'key', tint: 0xe6deff, value: () => (this.ledger.action < 0 ? '—' : `${this.ledger.action + 1}`) },
      { key: 'move', label: 'move', tint: 0x9fe6ff, value: () => this.moveText() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => (this.ledger.action < 0 ? '—' : signed(this.ledger.reward)) },
      { key: 'mean', label: 'mean reward', tint: GOLD, value: () => (Number.isFinite(this.ledger.mean) ? `${signed(this.ledger.mean)} · last ${formatCount(this.ledger.window)}` : '—') },
    ],
    note: () => {
      if (this.loop) return `${this.loop.length}-step loop · ${signed(this.loop.reward)} per lap`;
      return this.ledger.window > 0 ? `${formatCount(this.ledger.distinct)} states in ${formatCount(this.ledger.window)} steps` : `${formatCount(this.layout.span ** 2)} states`;
    },
    bar: () => (Number.isFinite(this.ledger.paid) ? this.ledger.paid : 0),
  };

  private moveText(): string {
    const { action, diff } = this.ledger;
    if (action < 0) return '—';
    if (diff === 0) return 'stayed';
    const flips = popcount(diff);
    if (flips > 1) return `jumped · ${flips} bits changed`;
    const bit = 31 - Math.clz32(diff), levels = this.layout.levels;
    return `bit ${bit} flipped · ${bit < levels ? 'column' : 'row'}`;
  }

  private centre(state: number, out: THREE.Vector3): THREE.Vector3 {
    const { span, levels, cell } = this.layout;
    return cellToWorldVec(state & (span - 1), state >>> levels, { w: span, h: span }, out).multiplyScalar(cell);
  }

  private lampAt(index: number, out: THREE.Vector3): THREE.Vector3 {
    const { levels, band, bits } = this.layout;
    if (index === bits) return out.set(BIAS, 0, BIAS);
    const row = index >= levels, level = levels - 1 - (index % levels), along = -HALF - DEPTH + (level + 0.5) * band;
    return row ? out.set(along, 0, HALF + LAMP_GAP) : out.set(HALF + LAMP_GAP, 0, along);
  }

  private keyAt(action: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(KEY_X, 0, keyZ(action));
  }

  private digitAt(index: number, out: THREE.Vector3): { at: THREE.Vector3; size: number } {
    const count = 2 * this.layout.digits, pitch = Math.min(0.9, (2.9 - DIGIT_GAP) / count), size = pitch / 0.9;
    const width = count * pitch + DIGIT_GAP;
    out.set(DISPLAY_X - width / 2 + (index + 0.5) * pitch + (index >= this.layout.digits ? DIGIT_GAP : 0), 0, DISPLAY_Z);
    return { at: out, size };
  }

  private branch(value: number, level: number): [number, number] {
    const bit = this.layout.levels - 1 - level, size = 2 ** (bit + 1), node = (Math.floor(value / size) + 0.5) * size;
    return [node, node + (((value >>> bit) & 1) ? 1 : -1) * 2 ** (bit - 1)];
  }

  private decoderPoint(rows: boolean, u: number, v: number, out: THREE.Vector3): THREE.Vector3 {
    const { cell, band } = this.layout, across = -HALF + u * cell, deep = -HALF - DEPTH + v * band;
    return rows ? out.set(deep, 0, across) : out.set(across, 0, deep);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.layout = layoutOf(parameter(env, 'bits'));
    this.local = parameter(env, 'local') !== 0;
    this.actionCount = constant(env, 'HASHED_ACTIONS');
    if (env.actions !== this.actionCount || this.actionCount > 4) throw new Error(`${env.envId} has ${env.actions} actions but the keypad holds ${this.actionCount}`);
    this.tau = this.layout.bits > 8 ? 500 : 150;
    return EXTENT.clone();
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xc4bce0).multiplyScalar(0.42);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const lines = 16;
    return {
      lattice: { w: lines, h: lines, origin: [-HALF, 0.04, -HALF], u: [SIDE / lines, 0, 0], v: [0, 0, SIDE / lines] },
      debug: (frame, kit) => {
        const step = frame.state.step, now = frame.frameAt(step), state = now.words[0]!, { bits, levels, span } = this.layout;
        this.centre(state, this.at);
        kit.marks('agent', 1).set(this.at.x, 0.1, this.at.z, Math.max(this.layout.cell * 0.45, 0.12), TOKENS.uiCyan);
        const lamps = kit.marks('bits', bits + 1);
        for (let k = 0; k <= bits; k++) {
          const seen = now.observations[k]!, truth = k === bits ? 1 : (state >>> k) & 1;
          this.lampAt(k, this.at);
          lamps.set(this.at.x, 0.1, this.at.z, this.layout.lamp * 0.55, seen !== truth ? 0xff4d4d : truth ? GOLD : 0x4a5570);
        }
        const trace = (name: string, rows: boolean, value: number) => {
          const curve = kit.curve(name, 3 * levels + 2, TOKENS.uiCyan);
          for (let level = 0; level < levels; level++) {
            const [node, child] = this.branch(value, level);
            for (const [u, v] of [[node, level], [node, level + 0.5], [child, level + 0.5]] as const) {
              this.decoderPoint(rows, u, v, this.at);
              curve.point(this.at.x, 0.12, this.at.z);
            }
          }
          this.decoderPoint(rows, value + 0.5, levels, this.at);
          curve.point(this.at.x, 0.12, this.at.z);
          this.centre(state, this.at);
          curve.point(this.at.x, 0.12, this.at.z);
        };
        trace('column address', false, state & (span - 1));
        trace('row address', true, state >>> levels);
        const path = kit.curve('trajectory', DEBUG_PATH + 1, 0xffa05a), cells: number[] = [];
        for (let s = step; s >= Math.max(0, step - DEBUG_PATH); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          cells.push(f.words[0]!);
        }
        for (let i = cells.length - 1; i >= 0; i--) {
          this.centre(cells[i]!, this.at);
          path.point(this.at.x, 0.11, this.at.z);
        }
        if (step < 1) return;
        const reward = now.reward, reach = this.scale > 0 ? (REACH * Math.abs(reward)) / this.scale : 0;
        if (reward !== 0) kit.bars('reward', 1).set(SPINE + (Math.sign(reward) * reach) / 2, 0.1, TAPE_Z1 - ROW / 2, 0.05, reward > 0 ? GOLD : LOSS, Math.max(reach, 0.02), 0.08);
        this.keyAt((now.actions[0]! >>> 0) % this.actionCount, this.at);
        kit.marks('key', 1).set(this.at.x, 0.1, this.at.z, 0.3, TOKENS.uiCyan);
      },
    };
  }

  private batch(name: string, material: THREE.Material, capacity: number, order: number, style: LineStyle | null): Batch {
    const geometry = flat(1, 1);
    const shape = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, shape);
    const mesh = instancedMesh(geometry, material, capacity);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3).fill(1), 3).setUsage(THREE.DynamicDrawUsage);
    mesh.count = 0;
    mesh.frustumCulled = false;
    mesh.renderOrder = order;
    this.scene.add(mesh);
    return { mesh, shape, style, used: 0 };
  }

  private place(batch: Batch, index: number, at: THREE.Vector3, width: number, depth = width): void {
    batch.mesh.setMatrixAt(index, this.matrix.compose(at, this.quaternion.identity(), this.scaleVector.set(width, 1, depth)));
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const layout = this.layout, lit = light.lit, accent = this.accentOf(ctx);
    this.accent.set(accent);
    this.scene.background = new THREE.Color(0x040308);
    this.history = new History(HISTORY, this.actionCount);
    this.heat = new Float32Array(layout.span * layout.span);
    const charge = new THREE.DataTexture(new Uint8Array(layout.span * layout.span), layout.span, layout.span, THREE.RedFormat, THREE.UnsignedByteType);
    charge.minFilter = charge.magFilter = THREE.NearestFilter;
    charge.needsUpdate = true;
    this.charge = charge;

    const ground = new THREE.Mesh(flat(160, 160), groundMaterial(layout, lit, this.actionCount));
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    const array = new THREE.Mesh(flat(SIDE, SIDE), arrayMaterial(layout, lit, charge, accent));
    array.renderOrder = 1;
    const columns = new THREE.Mesh(flat(SIDE, DEPTH), treeMaterial(layout, lit, false));
    columns.position.set(0, 0, -HALF - DEPTH / 2);
    columns.renderOrder = 1;
    const rows = new THREE.Mesh(flat(DEPTH, SIDE), treeMaterial(layout, lit, true));
    rows.position.set(-HALF - DEPTH / 2, 0, 0);
    rows.renderOrder = 1;
    this.scene.add(ground, array, columns, rows);

    this.trail = this.batch('aLine', lineMaterial(TRACE), TRAIL + 1, 2, TRACE);
    this.paths = this.batch('aLine', lineMaterial(PATH), 6 * layout.levels * 2 + 10, 3, PATH);
    this.bars = this.batch('aBar', barMaterial(), ROWS + 2 + this.actionCount, 3, null);
    this.lamps = this.batch('aLamp', lampMaterial(accent), layout.bits + 1, 4, null);
    this.keys = this.batch('aKey', keyMaterial(lit, accent), this.actionCount, 4, null);
    this.digits = this.batch('aDigit', digitMaterial(accent), 2 * layout.digits, 4, null);

    const lampSize = layout.lamp * 3.6;
    for (let k = 0; k <= layout.bits; k++) {
      this.place(this.lamps, k, this.lampAt(k, this.at), lampSize);
      const tint: Tint = uniform(new THREE.Vector3());
      const lamp = light.emitter(new THREE.CircleGeometry(layout.lamp * 0.62, 20).rotateX(-Math.PI / 2), tint);
      lamp.position.copy(this.at);
      this.lampTints.push(tint);
      this.scene.add(lamp);
    }
    this.lamps.mesh.count = layout.bits + 1;
    this.lamps.mesh.instanceMatrix.needsUpdate = true;
    for (let a = 0; a < this.actionCount; a++) {
      this.place(this.keys, a, this.keyAt(a, this.at), KEY + 0.6);
      this.keys.shape.setXYZW(a, a + 1, 0, 0, 0);
    }
    this.keys.mesh.count = this.actionCount;
    this.keys.mesh.instanceMatrix.needsUpdate = true;
    this.keys.shape.needsUpdate = true;
    for (let i = 0; i < 2 * layout.digits; i++) {
      const { at, size } = this.digitAt(i, this.at);
      this.place(this.digits, i, at, DIGIT_W * size, DIGIT_H * size);
    }
    this.digits.mesh.count = 2 * layout.digits;
    this.digits.mesh.instanceMatrix.needsUpdate = true;

    const reticle = reticleMaterial(layout, accent, this.flash);
    this.reticle = new THREE.Mesh(flat(reticle.size, reticle.size), reticle.material);
    this.reticle.renderOrder = 6;
    this.reticle.frustumCulled = false;
    this.scene.add(this.reticle);

    const lavender = rgb(accent);
    this.agentLight = light.emitter(new THREE.CircleGeometry(0.16, 24).rotateX(-Math.PI / 2), lavender.mul(30));
    this.columnLight = light.emitter(flat(0.07, SIDE), lavender.mul(1.2));
    this.rowLight = light.emitter(flat(SIDE, 0.07), lavender.mul(1.2));
    this.keyLight = light.emitter(new THREE.RingGeometry(KEY / 2 + 0.16, KEY / 2 + 0.26, 48).rotateX(-Math.PI / 2), this.keyTint);
    this.flareLight = light.emitter(new THREE.CircleGeometry(0.22, 24).rotateX(-Math.PI / 2), this.flareTint);
    this.scene.add(this.agentLight, this.columnLight, this.rowLight, this.keyLight, this.flareLight);
  }

  protected override drop(): void {
    this.charge?.dispose();
    this.charge = null;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const history = this.history;
    if (!history) return;
    const dt = Math.max(time - this.simTime, frame.dtReal, 0);
    this.simTime = time;
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, time);
    const step = frame.state.step, at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    const moved = step !== this.step;
    if (moved || (this.clock - this.retried > RETRY && !history.complete(step))) {
      this.retried = this.clock;
      if (history.fill(step, at) || moved) this.derive(step);
    }
    if (frame.scrubbed || step < this.step || this.step < 0) this.settle(step, pool);
    else if (moved) this.advance(frame, step, pool, time);
    this.step = step;
    this.animate(dt, time);
  }

  private derive(step: number): void {
    const history = this.history!;
    const data = this.charge!.image.data as Uint8Array, touched = this.touched;
    for (const cell of touched) this.heat[cell] = data[cell] = 0;
    touched.length = 0;
    let scale = 0;
    for (let s = Math.max(0, step - HISTORY + 1); s <= step; s++) {
      if (!history.has(s)) continue;
      const cell = history.state(s);
      if (this.heat[cell] === 0) touched.push(cell);
      this.heat[cell]! += Math.exp(-(step - s) / this.tau);
      if (s >= 1) scale = Math.max(scale, Math.abs(history.reward(s)));
    }
    this.scale = scale;
    for (const cell of touched) data[cell] = Math.round(255 * (1 - Math.exp(-0.45 * this.heat[cell]!)));
    this.charge!.needsUpdate = true;
    history.edges(step, TRAIL, this.local ? TRAIL_TAU : LEAP_TAU, this.edges);
    this.loop = history.loop(step, LOOP_LONGEST);
    let sum = 0, paid = 0, count = 0;
    this.distinct.clear();
    this.shares.fill(0);
    for (let s = step; s > step - RATE && s >= 1; s--) {
      if (!history.moved(s)) continue;
      const reward = history.reward(s);
      sum += reward;
      if (reward > 0) paid++;
      count++;
      this.shares[history.action(s)]!++;
      this.distinct.add(history.state(s));
    }
    for (let a = 0; a < this.actionCount; a++) this.shares[a] = count > 0 ? this.shares[a]! / count : 0;
    const ledger = this.ledger, moved = history.moved(step);
    ledger.mean = count > 0 ? sum / count : NaN;
    ledger.paid = count > 0 ? paid / count : NaN;
    ledger.window = count;
    ledger.distinct = this.distinct.size;
    if (history.has(step)) ledger.state = history.state(step);
    ledger.action = moved ? history.action(step) : -1;
    ledger.diff = moved ? (history.state(step) ^ history.state(step - 1)) >>> 0 : 0;
    ledger.reward = moved ? history.reward(step) : 0;
  }

  private settle(step: number, pool: CelebrationPool): void {
    const history = this.history!, state = history.has(step) ? history.state(step) : this.ledger.state;
    this.hop = { from: state, to: state, t: 1 };
    this.pressed = history.moved(step) ? history.action(step) : -1;
    this.pressedAt = this.flippedAt = this.flare.born = this.burstAt = -Infinity;
    this.flipped = 0;
    pool.clear();
  }

  private advance(frame: FrameUpdate, step: number, pool: CelebrationPool, time: number): void {
    const history = this.history!;
    if (!history.moved(step)) {
      this.settle(step, pool);
      return;
    }
    const from = history.state(step - 1), to = history.state(step), reward = history.reward(step);
    if (this.local && popcount(from ^ to) !== 1) throw new Error(`HashedMDP step ${step} moved from ${from} to ${to}, which is not one bit flip`);
    this.hop = { from, to, t: 0 };
    this.pressed = history.action(step);
    this.pressedAt = time;
    this.flipped = (from ^ to) >>> 0;
    this.flippedAt = time;
    if (reward !== 0) {
      this.centre(to, this.flare.at);
      this.flare.born = time;
      this.flare.color.copy(reward > 0 ? this.gold : this.loss);
      this.flare.strength = this.scale > 0 ? Math.abs(reward) / this.scale : 1;
    }
    let paid = false, charged = false;
    for (const event of frame.events) {
      if (event.type === 'goal') paid = true;
      if (event.type === 'harm') charged = true;
    }
    const size = this.scale > 0 ? Math.abs(reward) / this.scale : 0;
    this.centre(to, this.at);
    if (reward > 0 && paid) {
      pool.ripple(this.at, GOLD, 0.16 + 0.22 * size, time);
      if (this.clock - this.burstAt > BURST_GAP) {
        this.burstAt = this.clock;
        pool.burst(this.at.setY(38), GOLD, 0.22 + 0.4 * size, time);
      }
    } else if (reward < 0 && charged) pool.ripple(this.at, LOSS, 0.12 + 0.16 * size, time);
  }

  private animate(dt: number, time: number): void {
    const hop = this.hop;
    hop.t = Math.min(1, hop.t + (dt * this.controlHz) / HOP);
    const e = smoothstep01(hop.t), { levels, span } = this.layout;
    this.agentPosition(e, this.shown);
    const leap = popcount(hop.from ^ hop.to) > 1 ? Math.sin(Math.PI * e) : 0;
    const flare = Math.max(0, 1 - (time - this.flare.born) / FLARE_LIFE) ** 2 * this.flare.strength;
    this.flash.value = this.flare.color.equals(this.gold) ? flare : 0;
    if (this.reticle) {
      this.reticle.position.copy(this.shown);
      this.reticle.scale.setScalar(1 + 0.45 * leap);
    }
    this.agentLight?.position.copy(this.shown);
    this.columnLight?.position.set(this.shown.x, 0, 0);
    this.rowLight?.position.set(0, 0, this.shown.z);
    this.flareLight?.position.copy(this.flare.at);
    this.color.copy(this.flare.color).multiplyScalar(26 * flare);
    this.flareTint.value.set(this.color.r, this.color.g, this.color.b);

    const paths = this.paths!;
    paths.used = 0;
    this.decoder(paths, false, hop.from & (span - 1), hop.to & (span - 1), e);
    this.decoder(paths, true, hop.from >>> levels, hop.to >>> levels, e);
    if (hop.t < 1) {
      this.pulse(paths, false, hop.from & (span - 1), hop.to & (span - 1), e);
      this.pulse(paths, true, hop.from >>> levels, hop.to >>> levels, e);
    }
    this.commit(paths);
    this.drawTrail(e);
    this.drawTape(e);
    this.drawLamps(time);
    this.drawKeys(time);
    this.drawDigits(time);
  }

  private agentPosition(e: number, out: THREE.Vector3): THREE.Vector3 {
    const hop = this.hop;
    this.centre(hop.from, this.from);
    this.centre(hop.to, this.to);
    if (popcount(hop.from ^ hop.to) <= 1) return out.lerpVectors(this.from, this.to, e);
    const dx = this.to.x - this.from.x, dz = this.to.z - this.from.z, mx = (this.from.x + this.to.x) / 2, mz = (this.from.z + this.to.z) / 2;
    const bend = mx * -dz + mz * dx > 0 ? -0.22 : 0.22, cx = mx - dz * bend, cz = mz + dx * bend, k = 1 - e;
    return out.set(k * k * this.from.x + 2 * k * e * cx + e * e * this.to.x, 0, k * k * this.from.z + 2 * k * e * cz + e * e * this.to.z);
  }

  private decoder(batch: Batch, rows: boolean, from: number, to: number, e: number): void {
    const { levels } = this.layout, diff = from ^ to;
    if (diff === 0) {
      this.addressPath(batch, rows, to, 0, levels, 0, 1);
      return;
    }
    const pivot = levels - 1 - (31 - Math.clz32(diff));
    this.addressPath(batch, rows, to, 0, pivot, 0, 1);
    const [node, childFrom] = this.branch(from, pivot), childTo = this.branch(to, pivot)[1];
    const child = childFrom + (childTo - childFrom) * e, lavender = this.color.copy(this.accent);
    this.segment(batch, rows, node, pivot, node, pivot + 0.5, 1, lavender);
    this.segment(batch, rows, node, pivot + 0.5, child, pivot + 0.5, 1, lavender);
    this.segment(batch, rows, child, pivot + 0.5, child, pivot + 1, 1, lavender);
    if ((diff & (diff - 1)) === 0) this.addressPath(batch, rows, from, pivot + 1, levels, child - childFrom, 1);
    else {
      this.addressPath(batch, rows, from, pivot + 1, levels, 0, 1 - e);
      this.addressPath(batch, rows, to, pivot + 1, levels, 0, e);
    }
  }

  private pulse(batch: Batch, rows: boolean, from: number, to: number, e: number): void {
    const diff = from ^ to;
    if (diff === 0) return;
    const { levels, band } = this.layout, pivot = levels - 1 - (31 - Math.clz32(diff)), route = this.route;
    const [node, childFrom] = this.branch(from, pivot), childTo = this.branch(to, pivot)[1];
    const shift = (diff & (diff - 1)) === 0 ? childFrom + (childTo - childFrom) * e - childTo : 0;
    route.length = 0;
    const push = (u: number, v: number) => {
      this.decoderPoint(rows, u, v, this.at);
      route.push(this.at.x, this.at.z);
    };
    push(node, pivot + 0.5);
    push(childTo + shift, pivot + 0.5);
    for (let level = pivot + 1; level < levels; level++) {
      const [at, child] = this.branch(to, level);
      push(at + shift, level + 0.5);
      push(child + shift, level + 0.5);
    }
    push(to + 0.5 + shift, levels);
    this.centre(this.hop.to, this.to);
    const target = (rows ? this.to.x - (-HALF - DEPTH) : this.to.z - (-HALF - DEPTH)) / band;
    push(to + 0.5 + shift, target);
    let total = 0;
    for (let i = 2; i < route.length; i += 2) total += Math.hypot(route[i]! - route[i - 2]!, route[i + 1]! - route[i - 1]!);
    let left = total * e, x = route[0]!, z = route[1]!;
    for (let i = 2; i < route.length; i += 2) {
      const dx = route[i]! - route[i - 2]!, dz = route[i + 1]! - route[i - 1]!, piece = Math.hypot(dx, dz);
      if (left <= piece && piece > 0) {
        x = route[i - 2]! + (dx * left) / piece;
        z = route[i - 1]! + (dz * left) / piece;
        break;
      }
      left -= piece;
      x = route[i]!;
      z = route[i + 1]!;
    }
    this.line(batch, x, z, x, z, 0.3, Math.sqrt(Math.sin(Math.PI * e)), this.color.copy(this.accent).lerp(this.white, 0.7).multiplyScalar(2.2));
  }

  private addressPath(batch: Batch, rows: boolean, value: number, first: number, last: number, shift: number, alpha: number): void {
    if (alpha <= 0.002) return;
    const { levels } = this.layout, lavender = this.color.copy(this.accent);
    for (let level = first; level < last; level++) {
      const [at, reach] = this.branch(value, level), node = at + shift, child = reach + shift;
      this.segment(batch, rows, node, level, node, level + 0.5, alpha, lavender);
      this.segment(batch, rows, node, level + 0.5, child, level + 0.5, alpha, lavender);
      this.segment(batch, rows, child, level + 0.5, child, level + 1, alpha, lavender);
    }
    if (last < levels) return;
    const u = value + 0.5 + shift;
    this.decoderPoint(rows, u, levels, this.from);
    this.decoderPoint(rows, u, levels + SIDE / this.layout.band, this.to);
    this.line(batch, this.from.x, this.from.z, this.to.x, this.to.z, 0.035, alpha * 0.4, lavender);
  }

  private segment(batch: Batch, rows: boolean, u0: number, v0: number, u1: number, v1: number, alpha: number, color: THREE.Color): void {
    this.decoderPoint(rows, u0, v0, this.from);
    const x0 = this.from.x, z0 = this.from.z;
    this.decoderPoint(rows, u1, v1, this.from);
    this.line(batch, x0, z0, this.from.x, this.from.z, this.layout.levels > 4 ? 0.05 : 0.065, alpha, color);
  }

  private line(batch: Batch, x0: number, z0: number, x1: number, z1: number, width: number, alpha: number, color: THREE.Color): void {
    const i = batch.used++, pad = batch.style!.pad;
    if (i >= batch.mesh.instanceMatrix.count) throw new Error(`${this.stageName} ran out of line instances`);
    const dx = x1 - x0, dz = z1 - z0, length = Math.max(Math.hypot(dx, dz), 1e-3);
    this.quaternion.setFromAxisAngle(Y, Math.atan2(-dz, dx));
    this.matrix.compose(this.at.set((x0 + x1) / 2, 0, (z0 + z1) / 2), this.quaternion, this.scaleVector.set(length + 2 * pad, 1, width + 2 * pad));
    batch.mesh.setMatrixAt(i, this.matrix);
    batch.shape.setXYZW(i, length, width, alpha, 0);
    batch.mesh.setColorAt(i, color);
  }

  private commit(batch: Batch): void {
    batch.mesh.count = batch.used;
    batch.mesh.instanceMatrix.needsUpdate = true;
    batch.mesh.instanceColor!.needsUpdate = true;
    batch.shape.needsUpdate = true;
  }

  private edgeColor(reward: number, out: THREE.Color): THREE.Color {
    if (reward === 0) return out.copy(this.silver).multiplyScalar(0.42);
    const size = this.scale > 0 ? Math.min(1, Math.abs(reward) / this.scale) : 1;
    return out.copy(reward > 0 ? this.gold : this.loss).multiplyScalar(0.6 + 0.7 * size);
  }

  private drawTrail(e: number): void {
    const trail = this.trail!, cell = this.layout.cell, width = cell >= 1 ? 0.09 : 0.045, lane = cell >= 1 ? 0.15 : 0;
    trail.used = 0;
    const newest = this.hop.t < 1 && this.hop.from !== this.hop.to ? this.step : -1;
    for (const edge of this.edges.values()) {
      const alpha = 1 - Math.exp(-1.25 * edge.weight);
      if (alpha < 0.01) continue;
      this.centre(edge.from, this.from);
      this.centre(edge.to, this.to);
      if (edge.last === newest && edge.weight < 1.01) this.to.lerp(this.from, 1 - e);
      const dx = this.to.x - this.from.x, dz = this.to.z - this.from.z, length = Math.hypot(dx, dz);
      const ox = length > 0 ? (-dz / length) * lane : 0, oz = length > 0 ? (dx / length) * lane : 0;
      this.line(trail, this.from.x + ox, this.from.z + oz, this.to.x + ox, this.to.z + oz, edge.reward === 0 ? width : width * 1.5, alpha, this.edgeColor(edge.reward, this.color));
    }
    this.commit(trail);
  }

  private drawTape(e: number): void {
    const bars = this.bars!, history = this.history!;
    bars.used = 0;
    const slide = (1 - e) * ROW;
    for (let r = 0; r <= ROWS; r++) {
      const s = this.step - r;
      if (s < 1 || !history.moved(s)) continue;
      const z = TAPE_Z1 - (r + 0.5) * ROW + slide;
      const fade = clamp01((TAPE_Z1 - z) / (ROW * 0.6)) * clamp01((z - TAPE_Z0) / (ROW * 0.6));
      if (fade <= 0) continue;
      const reward = history.reward(s), age = r === 0 ? 1 : 0.85 - 0.45 * (r / ROWS);
      const reach = reward === 0 ? 0 : Math.max(0.06, (REACH * Math.abs(reward)) / Math.max(this.scale, 1e-9));
      const i = bars.used++, height = reward === 0 ? 0.045 : 0.22, width = reward === 0 ? 0.18 : reach;
      this.place(bars, i, this.at.set(reward === 0 ? SPINE : SPINE + (Math.sign(reward) * reach) / 2, 0, z), width + 0.2, height + 0.2);
      bars.shape.setXYZW(i, width, height, fade * age * (reward === 0 ? 0.5 : 0.78), 0);
      bars.mesh.setColorAt(i, this.edgeColor(reward, this.color));
    }
    for (let a = 0; a < this.actionCount; a++) {
      const share = this.shares[a]!;
      if (share <= 0) continue;
      const i = bars.used++, width = SHARE * share;
      this.place(bars, i, this.at.set(KEY_X - SHARE / 2 + width / 2, 0, shareZ(a)), width + 0.2, 0.36);
      bars.shape.setXYZW(i, width, 0.16, 0.9, 0);
      bars.mesh.setColorAt(i, this.color.copy(this.accent).multiplyScalar(0.8));
    }
    this.commit(bars);
  }

  private drawLamps(time: number): void {
    const lamps = this.lamps!, { bits } = this.layout, state = this.hop.to, flip = Math.max(0, 1 - (time - this.flippedAt) / FLIP_LIFE);
    for (let k = 0; k <= bits; k++) {
      const on = k === bits ? 1 : (state >>> k) & 1, flashed = k < bits && ((this.flipped >>> k) & 1) === 1 ? flip : 0;
      lamps.shape.setXYZW(k, on, flashed, 0, 0);
      this.color.copy(this.accent).multiplyScalar((k === bits ? 3 : 6) * on + 9 * flashed);
      this.lampTints[k]!.value.set(this.color.r, this.color.g, this.color.b);
    }
    lamps.shape.needsUpdate = true;
  }

  private drawKeys(time: number): void {
    const keys = this.keys!, press = Math.max(0, 1 - (time - this.pressedAt) / PRESS_LIFE);
    for (let a = 0; a < this.actionCount; a++) {
      const chosen = a === this.pressed;
      keys.shape.setXYZW(a, a + 1, chosen ? press : 0, chosen ? 0.4 + 0.6 * press : 0, 0);
    }
    keys.shape.needsUpdate = true;
    if (this.keyLight && this.pressed >= 0) this.keyAt(this.pressed, this.keyLight.position);
    this.color.copy(this.accent).multiplyScalar(this.pressed >= 0 ? 1.5 + 9 * press : 0);
    this.keyTint.value.set(this.color.r, this.color.g, this.color.b);
  }

  private drawDigits(time: number): void {
    const digits = this.digits!, { levels, digits: count } = this.layout, state = this.hop.to, flip = Math.max(0, 1 - (time - this.flippedAt) / FLIP_LIFE);
    const values = [state >>> levels, state & ((1 << levels) - 1)], changed = [this.flipped >>> levels, this.flipped & ((1 << levels) - 1)];
    for (let i = 0; i < 2 * count; i++) {
      const group = i < count ? 0 : 1, shift = 4 * (count - 1 - (i % count));
      const nibble = (values[group]! >>> shift) & 15, touched = ((changed[group]! >>> shift) & 15) !== 0;
      digits.shape.setXYZW(i, HEX[nibble]!, 1 + (touched ? 0.8 * flip : 0), 0, 0);
    }
    digits.shape.needsUpdate = true;
  }
}

export default (() => new HashedScene()) satisfies SceneFactory;
