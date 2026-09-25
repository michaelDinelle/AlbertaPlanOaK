import * as THREE from 'three/webgpu';
import { abs, dot, float, floor, fract, max, min, mix, positionWorld, screenCoordinate, sin, smoothstep, step, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendRow, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { constantList, parameter } from '../../data/envs';
import { countEvents } from '@sim/events';
import {
  bit, FREEWAY_CHICKEN, MINATAR_AGENT_X, MINATAR_AGENT_Y, MINATAR_ASTERIX, MINATAR_BREAKOUT, MINATAR_FREEWAY, MINATAR_INVADERS, MINATAR_LAST, MINATAR_SEAQUEST,
  MINATAR_TARGET_X, MINATAR_TARGET_Y, s8, SEA_COUNT, SEA_SHOTS, SEA_VIEW, u8,
} from '@sim/envs/minatar';
import { instancedMesh } from '@kit/instances';
import type { EnvFrame, EventStream } from '@sim/types';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Shape = (p: N<'vec2'>) => N<'float'>;

const TITLES = ['BREAKOUT', 'SPACE INVADERS', 'FREEWAY', 'ASTERIX', 'SEAQUEST'] as const;
const NAMES = ['no-op', '← left', '↑ up', 'right →', '↓ down', 'fire'] as const;
const BEZEL = 0.45;
const STRIP = 0.8;
const MISS = TOKENS.failureBlue;
const GOLD = TOKENS.celebrationGold;
const BRICKS = [0xff6b9a, 0xffa94d, 0xffe46b] as const;
const CARS = [0x5ec8ff, 0x7cff9e, 0xffd24a, 0xff8f4a, 0xff4a6e] as const;

const dither = fract(fract(dot(floor(screenCoordinate.xy), vec2(0.06711056, 0.00583715))).mul(52.9829189)).sub(0.5).mul(0.0016);
const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const hash = (p: N<'vec2'>) => fract(sin(dot(p, vec2(12.9898, 78.233))).mul(43758.5453));
const turn = (p: N<'vec2'>) => vec2(p.x.add(p.y), p.y.sub(p.x)).mul(Math.SQRT1_2);
const ellipse = (p: N<'vec2'>, rx: number, ry: number) => p.div(vec2(rx, ry)).length().sub(1).mul(Math.min(rx, ry));
const union = (...d: N<'float'>[]) => d.reduce((a, b) => min(a, b));
const cut = (d: N<'float'>, hole: N<'float'>) => max(d, hole.negate());
const V = (x: number, y: number) => vec2(x, y);

const SHAPES: Record<string, Shape> = {
  brick: (p) => sdBox(p, V(0.45, 0.32), 0.07),
  paddle: (p) => sdBox(p.sub(V(0, -0.2)), V(0.47, 0.1), 0.1),
  ball: (p) => sdCircle(p, 0.2),
  spark: (p) => union(sdCircle(p, 0.1), ...[0, 1, 2, 3].map((k) => {
    const a = k * Math.PI / 4, c = Math.cos(a) * 0.4, s = Math.sin(a) * 0.4;
    return sdSegment(p, V(-c, -s), V(c, s), 0.035);
  })),
  alien: (p) => union(
    cut(sdBox(p.sub(V(0, 0.02)), V(0.3, 0.19), 0.12), union(sdCircle(p.sub(V(-0.12, 0.06)), 0.06), sdCircle(p.sub(V(0.12, 0.06)), 0.06))),
    sdSegment(p, V(-0.18, -0.1), V(-0.3, -0.3), 0.04), sdSegment(p, V(0.18, -0.1), V(0.3, -0.3), 0.04),
    sdSegment(p, V(-0.1, 0.18), V(-0.22, 0.34), 0.03), sdSegment(p, V(0.1, 0.18), V(0.22, 0.34), 0.03),
  ),
  cannon: (p) => union(sdBox(p.sub(V(0, -0.24)), V(0.42, 0.1), 0.08), sdBox(p.sub(V(0, -0.08)), V(0.22, 0.1), 0.06), sdBox(p.sub(V(0, 0.1)), V(0.06, 0.18), 0.04)),
  shot: (p) => sdSegment(p, V(0, -0.2), V(0, 0.2), 0.05),
  bomb: (p) => sdBox(turn(p), V(0.1, 0.1), 0.03),
  chicken: (p) => union(sdCircle(p.sub(V(-0.04, -0.08)), 0.23), sdCircle(p.sub(V(0.12, 0.18)), 0.13), sdCircle(p.sub(V(0.1, 0.33)), 0.06), sdSegment(p, V(0.22, 0.18), V(0.33, 0.14), 0.035)),
  car: (p) => union(
    cut(sdBox(p, V(0.44, 0.22), 0.12), union(sdBox(p.sub(V(0.15, 0)), V(0.06, 0.15), 0.04), sdBox(p.sub(V(-0.2, 0)), V(0.05, 0.14), 0.03))),
    sdBox(p.sub(V(0.26, 0.25)), V(0.08, 0.04), 0.02), sdBox(p.sub(V(-0.26, 0.25)), V(0.08, 0.04), 0.02),
    sdBox(p.sub(V(0.26, -0.25)), V(0.08, 0.04), 0.02), sdBox(p.sub(V(-0.26, -0.25)), V(0.08, 0.04), 0.02),
  ),
  hero: (p) => union(sdCircle(p.sub(V(0, 0.08)), 0.18), sdBox(p.sub(V(0, -0.22)), V(0.15, 0.12), 0.06), sdSegment(p, V(-0.12, 0.2), V(-0.3, 0.36), 0.035), sdSegment(p, V(0.12, 0.2), V(0.3, 0.36), 0.035)),
  foe: (p) => union(sdCircle(p, 0.22), ...[0, 1, 2, 3].map((k) => {
    const a = k * Math.PI / 4 + Math.PI / 8, c = Math.cos(a) * 0.36, s = Math.sin(a) * 0.36;
    return sdSegment(p, V(-c, -s), V(c, s), 0.03);
  })),
  gem: (p) => union(cut(sdBox(turn(p), V(0.2, 0.2), 0.04), sdBox(turn(p), V(0.08, 0.08), 0.01)), sdCircle(p, 0.05)),
  sub: (p) => cut(union(sdSegment(p, V(-0.26, -0.06), V(0.26, -0.06), 0.15), sdBox(p.sub(V(0.02, 0.13)), V(0.1, 0.08), 0.03), sdSegment(p, V(0.06, 0.2), V(0.06, 0.32), 0.025), sdSegment(p, V(0.06, 0.32), V(0.15, 0.32), 0.025), sdBox(p.sub(V(-0.43, -0.06)), V(0.035, 0.11), 0.02)), sdCircle(p.sub(V(0.17, -0.06)), 0.05)),
  fish: (p) => cut(union(ellipse(p.sub(V(0.06, 0)), 0.28, 0.16), sdSegment(p, V(-0.2, 0), V(-0.38, 0.15), 0.04), sdSegment(p, V(-0.2, 0), V(-0.38, -0.15), 0.04), sdSegment(p, V(-0.38, 0.15), V(-0.38, -0.15), 0.04)), sdCircle(p.sub(V(0.2, 0.04)), 0.035)),
  diver: (p) => union(sdCircle(p.sub(V(0.22, 0.08)), 0.1), sdSegment(p, V(0.1, 0.04), V(-0.18, -0.02), 0.08), sdSegment(p, V(-0.18, -0.02), V(-0.38, 0.08), 0.035), sdSegment(p, V(-0.18, -0.02), V(-0.38, -0.12), 0.035), sdSegment(p, V(0.08, 0.02), V(0.3, -0.12), 0.03)),
  torpedo: (p) => sdSegment(p, V(-0.2, 0), V(0.2, 0), 0.05),
  mine: (p) => union(sdCircle(p, 0.1), sdSegment(p, V(-0.16, 0), V(0.16, 0), 0.025), sdSegment(p, V(0, -0.16), V(0, 0.16), 0.025)),
};

const KINDS = [
  ['brick', 'paddle', 'ball', 'spark'],
  ['alien', 'cannon', 'shot', 'bomb', 'spark'],
  ['car', 'chicken', 'spark'],
  ['foe', 'gem', 'hero', 'spark'],
  ['fish', 'sub', 'diver', 'torpedo', 'mine', 'spark'],
] as const;

const SIZES: Record<string, number> = { chicken: 1.3, sub: 1.2, hero: 1.15, diver: 1.1, fish: 1.1, gem: 1.1, foe: 1.05 };

class Batch {
  readonly mesh: THREE.InstancedMesh;
  private n = 0;
  size = 1;
  private readonly m = new THREE.Matrix4();
  private readonly c = new THREE.Color();
  private readonly v = new THREE.Vector3();
  private readonly s = new THREE.Vector3();
  private readonly q = new THREE.Quaternion();

  constructor(shape: Shape, capacity: number, order: number, private readonly grid: GridSize, clip: N<'float'>) {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });
    const p = uv().sub(0.5), d = shape(p);
    const shade = float(0.8).add(p.y.mul(0.45)).mul(smoothstep(-0.06, -0.01, d).mul(-0.35).add(1));
    material.colorNode = vec3(shade, shade, shade);
    material.opacityNode = coverage(d).mul(clip);
    this.mesh = instancedMesh(flat(1, 1), material, capacity);
    for (let i = 0; i < capacity; i++) this.mesh.setColorAt(i, this.c.set(0xffffff));
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = order;
  }

  begin(): void {
    this.n = 0;
  }

  add(x: number, y: number, color: number, scale = 1, flip = false): void {
    if (this.n >= this.mesh.instanceMatrix.count || scale <= 0.001) return;
    cellToWorldVec(x, y, this.grid, this.v);
    scale *= this.size;
    this.s.set(flip ? -scale : scale, 1, scale);
    this.mesh.setMatrixAt(this.n, this.m.compose(this.v, this.q, this.s));
    this.mesh.setColorAt(this.n++, this.c.set(color));
  }

  commit(): void {
    this.mesh.count = this.n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

const ease = (t: number) => 1 - (1 - t) ** 2;

class MinAtarScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'minAtar';

  private game = 0;
  private side = 10;
  private grid: GridSize = { w: 10, h: 10 };
  private map: number[] = [];
  private batches: Record<string, Batch> = {};
  private readonly meter = uniform(0);
  private readonly pips = uniform(0);
  private readonly flash = uniform(0);
  private readonly scratch = new THREE.Vector3();
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, goals: 0, harms: 0 };
  private readonly ledger = { a: '—', b: '—', c: '—', action: '—', reward: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.game = parameter(ctx.environment, 'game');
    const grid = ctx.geometry.grid;
    if (!grid) throw new Error(`${ctx.environment.envId} lacks its board`);
    this.side = grid.w;
    this.grid = { w: grid.w, h: grid.h };
    const x = grid.w / 2 + BEZEL + 0.3, top = -grid.h / 2 - BEZEL - 0.3, bottom = grid.h / 2 + BEZEL + 0.3 + (this.game === MINATAR_SEAQUEST ? STRIP : 0);
    return new THREE.Box2(new THREE.Vector2(-x, top), new THREE.Vector2(x, bottom));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x404058);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent = { x: 0, y: 0 }, target = { x: 0, y: 0 }, v = new THREE.Vector3();
    return {
      agentCell: (state) => {
        agent.x = state.pose[MINATAR_AGENT_X]!;
        agent.y = state.pose[MINATAR_AGENT_Y]!;
        return agent.x < 0 ? null : agent;
      },
      goalCell: (state) => {
        target.x = state.pose[MINATAR_TARGET_X]!;
        target.y = state.pose[MINATAR_TARGET_Y]!;
        return target.x < 0 ? null : target;
      },
      debug: (frame, kit) => {
        const f = frame.frameAt(frame.state.step);
        if (f.step !== frame.state.step) return;
        const marks = kit.marks('things', 200);
        this.things(f, (x, y, color) => {
          cellToWorldVec(x, y, this.grid, v);
          marks.set(v.x, 0.1, v.z, 0.14, color);
        });
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    light.setEnabled(false);
    this.map = constantList(ctx.environment, 'MINATAR_ACTION_MAP');
    this.scene.background = new THREE.Color(0x040308);
    this.scene.add(this.board(this.accentOf(ctx)));
    const n = this.side, half = n / 2;
    const clip = coverage(sdBox(vec2(positionWorld.x, positionWorld.z), vec2(half, half)));
    KINDS[this.game]!.forEach((kind, i) => {
      const batch = new Batch(SHAPES[kind]!, n * n + 8, 2 + i, this.grid, clip);
      batch.size = SIZES[kind] ?? 1;
      this.batches[kind] = batch;
      this.scene.add(batch.mesh);
    });
    this.legend = this.legendFor();
  }

  private legendFor(): LegendSpec {
    const labels = [
      ['paddle', 'ball', 'bricks'],
      ['cannon', 'invaders', 'reload'],
      ['chicken', 'traffic', 'clock'],
      ['player', 'lanes', 'pace'],
      ['sub', 'oxygen', 'divers'],
    ][this.game]!;
    const tints = [TOKENS.accents.MinAtar ?? 0xff7ad9, 0x9ef6ff, 0xffd166];
    const keys = ['a', 'b', 'c'] as const;
    const rows: LegendRow[] = labels.map((label, i) => ({ key: label, label, tint: tints[i]!, value: () => this.ledger[keys[i]!] }));
    rows.push({ key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action });
    rows.push({ key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) });
    const noun = [['bricks broken', 'balls lost'], ['invaders shot', 'cannons lost'], ['crossings', 'hits'], ['treasures', 'caught'], ['scoring steps', 'subs lost']][this.game]!;
    return {
      title: `MINATAR · ${TITLES[this.game]}`,
      rows,
      note: () => `${formatCount(this.tally.goals)} ${noun[0]} · ${formatCount(this.tally.harms)} ${noun[1]}`,
    };
  }

  private field(q: N<'vec2'>): N<'vec3'> {
    const n = this.side, row = floor(q.y), local = fract(q).sub(0.5);
    const g = this.game;
    if (g === MINATAR_BREAKOUT) {
      const dots = coverage(sdCircle(fract(q.add(0.5)).sub(0.5), 0.035));
      const band = step(1, q.y).mul(step(q.y, 4)).mul(0.5);
      return mix(rgb(0x121640), rgb(0x06071a), q.y.div(n)).add(rgb(0x1a1c44).mul(band.mul(0.4))).add(rgb(0x2d3170).mul(dots.mul(0.5)));
    }
    if (g === MINATAR_INVADERS) {
      const s = q.mul(3), cell = floor(s), h = hash(cell), off = vec2(hash(cell.add(7.1)), hash(cell.add(3.7))).sub(0.5).mul(0.6);
      const star = coverage(sdCircle(fract(s).sub(0.5).sub(off), 0.05)).mul(step(0.78, h)).mul(h);
      const ground = coverage(abs(q.y.sub(n - 0.04)).sub(0.03));
      return mix(rgb(0x0b0820), rgb(0x03040a), q.y.div(n)).add(rgb(0x8f9cff).mul(star.mul(0.35))).add(rgb(0x6a4aa0).mul(ground.mul(0.6)));
    }
    if (g === MINATAR_FREEWAY) {
      const walk = step(q.y, 1).add(step(n - 1, q.y));
      const checker = fract(floor(q.x.mul(2)).add(floor(q.y.mul(2))).mul(0.5)).mul(2).mul(step(q.y, 1));
      const asphalt = rgb(0x1c1e26).add(noise(q.mul(2.5)).mul(0.012));
      const kerb = mix(rgb(0x3a404e), rgb(0x4c5364), checker.mul(0.7));
      const seam = abs(q.y.sub(floor(q.y.add(0.5)))).sub(0.025);
      const inner = step(1.5, q.y).mul(step(q.y, n - 1.5));
      const dash = coverage(seam).mul(inner).mul(step(0.5, fract(q.x.mul(1.5))));
      const edge = coverage(abs(q.y.sub(1)).sub(0.03)).add(coverage(abs(q.y.sub(n - 1)).sub(0.03)));
      return mix(asphalt, kerb, clampNode(walk)).add(rgb(0x9a9ca8).mul(dash.mul(0.5))).add(rgb(0xc9a94a).mul(edge.mul(0.7)));
    }
    if (g === MINATAR_ASTERIX) {
      const wall = step(q.y, 1).add(step(n - 1, q.y));
      const shift = fract(row.mul(0.5)).mul(2).mul(0.5), mortar = min(abs(fract(q.y.mul(2)).sub(0.5)), abs(fract(q.x.add(shift.mul(0.5)).mul(1.5)).sub(0.5)).mul(0.66));
      const stone = mix(rgb(0x2a2520), rgb(0x4a4238), smoothstep(0.02, 0.06, mortar)).add(noise(q.mul(7)).mul(0.03));
      const tuft = coverage(sdSegment(local.sub(vec2(hash(floor(q)).sub(0.5).mul(0.6), 0)), vec2(0, -0.08), vec2(0.03, 0.06), 0.012)).mul(step(0.55, hash(floor(q).add(2.3))));
      const grass = mix(rgb(0x173421), rgb(0x1d3f28), fract(row.mul(0.5)).mul(2)).sub(rgb(0x060d08).mul(coverage(float(0.47).sub(abs(local.y))))).add(rgb(0x3f7a4a).mul(tuft.mul(0.5)));
      return mix(grass, stone, clampNode(wall));
    }
    const wave = sin(q.x.mul(Math.PI * 2).add(0.4)).mul(0.05).add(1);
    const sky = mix(rgb(0x6aa9d6), rgb(0x2b5f8e), q.y);
    const water = mix(rgb(0x125676), rgb(0x03172a), clampNode(q.y.sub(1).div(n - 1)));
    const caustic = smoothstep(0.62, 0.8, noise(q.mul(vec2(1.6, 0.8)))).mul(smoothstep(n, 1, q.y)).mul(0.07);
    const bed = smoothstep(n - 1.02 + 0.08, n - 0.98 + 0.08, q.y.add(sin(q.x.mul(2.3)).mul(0.07)));
    const sand = rgb(0x3b3222).add(noise(q.mul(8)).mul(0.05));
    let c = mix(water.add(caustic), sand, bed);
    c = mix(c, sky, step(q.y, wave));
    return c.add(rgb(0xcfefff).mul(coverage(abs(q.y.sub(wave)).sub(0.025)).mul(0.6)));
  }

  private board(accent: number): THREE.Mesh {
    const n = this.side, strip = this.game === MINATAR_SEAQUEST ? STRIP : 0, width = n + 2 * BEZEL, depth = n + 2 * BEZEL + strip;
    const q = vec2(uv().x.mul(width).sub(BEZEL), uv().y.oneMinus().mul(depth).sub(BEZEL));
    const centred = q.sub(n / 2), window = sdBox(centred, vec2(n / 2, n / 2), 0.06);
    const shell = sdBox(q.sub(vec2(n / 2, (n + strip) / 2)), vec2(n / 2 + BEZEL, (n + strip) / 2 + BEZEL), 0.4);
    const bezel = mix(rgb(0x2a2233), rgb(0x16111c), smoothstep(-BEZEL, n + BEZEL + strip, q.y));
    let color = mix(bezel, this.field(q), coverage(window));
    color = color.add(rgb(accent).mul(coverage(abs(window.sub(0.05)).sub(0.018)).mul(0.55)));
    color = color.add(rgb(MISS).mul(this.flash.mul(0.35)).mul(coverage(window)));
    if (strip) {
      const y = n + 0.42, left = 0.1, span = 6.3;
      const trough = sdBox(q.sub(vec2(left + span / 2, y)), vec2(span / 2, 0.14), 0.14);
      const fill = coverage(trough.add(0.04)).mul(step(q.x, this.meter.mul(span).add(left)));
      const tint = mix(rgb(0xff4a5e), rgb(0x6fe6ff), smoothstep(0.1, 0.5, this.meter));
      color = mix(color, rgb(0x07060c), coverage(trough));
      color = mix(color, tint.mul(0.85), fill);
      const at = q.x.sub(7).div(0.5), slot = floor(at), lit = step(0, slot).mul(step(slot, this.pips.sub(0.5)));
      const pip = coverage(sdCircle(vec2(fract(at).sub(0.5).mul(0.5), q.y.sub(y)), 0.14)).mul(lit);
      color = mix(color, rgb(0xa6e3ff).mul(0.9), pip);
    }
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = color.add(dither);
    material.opacityNode = coverage(shell);
    const mesh = new THREE.Mesh(flat(width, depth), material);
    mesh.position.z = strip / 2;
    mesh.renderOrder = 0;
    return mesh;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.batches = {};
  }

  private things(f: EnvFrame, put: (x: number, y: number, color: number) => void): void {
    const w = f.words, n = this.side, g = this.game;
    if (g === MINATAR_BREAKOUT) {
      for (let k = 0; k < 3 * n; k++) if (bit(w, 0, k)) put(k % n, 1 + Math.floor(k / n), GOLD);
      put(u8(w, 5), u8(w, 6), 0x9ef6ff);
      put(u8(w, 4), n - 1, 0xff7ad9);
    } else if (g === MINATAR_INVADERS) {
      for (let k = 0; k < n * n; k++) {
        if (bit(w, 0, k)) put(k % n, Math.floor(k / n), 0xff4a5e);
        if (bit(w, 4, k)) put(k % n, Math.floor(k / n), 0x9ef6ff);
        if (bit(w, 8, k)) put(k % n, Math.floor(k / n), 0xff8a3d);
      }
      put(u8(w, 48), n - 1, 0xff7ad9);
    } else if (g === MINATAR_FREEWAY) {
      for (let i = 0; i < 8; i++) put(u8(w, i), i + 1, 0xff4a5e);
      put(FREEWAY_CHICKEN, u8(w, 24), 0xff7ad9);
    } else if (g === MINATAR_ASTERIX) {
      for (let r = 0; r < 8; r++) {
        const e = u8(w, r);
        if (e) put(e & 15, r + 1, e & 0x20 ? GOLD : 0xff4a5e);
      }
      put(u8(w, 8), u8(w, 9), 0xff7ad9);
    } else {
      const tint = [0xff4a5e, 0xff4a5e, GOLD, 0x9ef6ff, 0xff8a3d];
      for (let l = 0; l < 5; l++) for (let i = 0; i < u8(w, SEA_COUNT[l]!); i++) {
        const e = f.view[SEA_VIEW[l]! + i]!;
        put(e & 15, (e >> 4) & 15, tint[l]!);
      }
      put(u8(w, 8), u8(w, 9), 0xff7ad9);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const b = this.batches;
    if (!b.spark) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    const at = frame.state.step, now = frame.frameAt(at), next = frame.frameAt(at + 1);
    const live = next.step === at + 1 && now.step === at, go = live && !next.done;
    const t = live ? clamp01(frame.position - at) : 0, e = ease(t);
    const before = now.done && at > 0 ? frame.frameAt(at - 1) : null, lost = before && before.step === at - 1 ? before : null;
    for (const batch of Object.values(b)) batch.begin();
    const g = this.game;
    if (g === MINATAR_BREAKOUT) this.breakout(now, next, go, t, e, lost);
    else if (g === MINATAR_INVADERS) this.invaders(now, next, go, t, e, lost);
    else if (g === MINATAR_FREEWAY) this.freeway(now, next, go, t, e, lost);
    else if (g === MINATAR_ASTERIX) this.asterix(now, next, go, t, e, lost);
    else this.seaquest(now, next, go, t, e, lost);
    for (const batch of Object.values(b)) batch.commit();
    this.flash.value = lost && g !== MINATAR_FREEWAY ? 0.35 * (1 - t) ** 2 : 0;

    const pose = frame.state.pose, prev = frame.prev?.pose ?? pose;
    for (const event of frame.events) {
      if (event.type === 'goal' || event.type === 'key') {
        const tx = pose[MINATAR_TARGET_X]!, x = tx >= 0 ? tx : pose[MINATAR_AGENT_X]!, y = tx >= 0 ? pose[MINATAR_TARGET_Y]! : pose[MINATAR_AGENT_Y]!;
        cellToWorldVec(x, y, this.grid, this.scratch);
        pool.ripple(this.scratch, event.type === 'goal' ? GOLD : 0xa6e3ff, 0.5, time);
        this.scratch.y = 40;
        pool.burst(this.scratch, GOLD, 0.6, time);
        pool.pulse.fire(0.25);
      }
      if (event.type === 'harm') {
        cellToWorldVec(prev[MINATAR_AGENT_X]!, prev[MINATAR_AGENT_Y]!, this.grid, this.scratch);
        pool.ripple(this.scratch, MISS, 0.7, time);
      }
    }

    this.count(frame.eventLog, at);
    const src = live ? next : now, idx = Math.trunc(src.actions[0] ?? -1);
    const chosen = idx >= 0 ? this.map[g * 6 + idx] ?? 0 : -1, applied = u8(src.words, MINATAR_LAST[g]!);
    this.ledger.action = chosen < 0 ? '—' : applied === chosen ? NAMES[applied] ?? '—' : `${NAMES[applied] ?? '—'} · sticky`;
    this.ledger.reward = frame.state.reward;
  }

  private breakout(now: EnvFrame, next: EnvFrame, go: boolean, t: number, e: number, lost: EnvFrame | null): void {
    const b = this.batches, w = now.words, nw = next.words, n = this.side;
    const gone = go ? w[0]! & ~nw[0]! : 0;
    for (let k = 0; k < 3 * n; k++) {
      if (!bit(w, 0, k)) continue;
      const x = k % n, y = 1 + Math.floor(k / n), hit = (gone >>> k) & 1;
      b.brick!.add(x, y, BRICKS[y - 1] ?? BRICKS[0], hit ? 1 - smoothstep01(t) : 1);
      if (hit) b.spark!.add(x, y, BRICKS[y - 1] ?? GOLD, 0.5 + 0.7 * t);
    }
    const lerp = (i: number) => go ? u8(w, i) + (u8(nw, i) - u8(w, i)) * t : u8(w, i);
    const px = go ? u8(w, 4) + (u8(nw, 4) - u8(w, 4)) * e : u8(w, 4);
    b.paddle!.add(px, n - 1, 0x8fe3ff);
    b.ball!.add(lerp(7), go ? u8(w, 8) + (u8(nw, 8) - u8(w, 8)) * t : u8(w, 8), 0x3f7f9a, 0.7);
    b.ball!.add(lerp(5), lerp(6), 0xf4fbff);
    if (lost) b.spark!.add(u8(lost.words, 5), u8(lost.words, 6), MISS, 1.2 * (1 - t));
    const count = popcount(w[0]! & ((1 << (3 * n)) - 1));
    this.ledger.a = `column ${u8(w, 4) + 1}`;
    this.ledger.b = `(${u8(w, 5) + 1}, ${u8(w, 6) + 1}) ${s8(w, 12) < 0 ? '↑' : '↓'}${s8(w, 11) < 0 ? '←' : '→'}`;
    this.ledger.c = `${count} standing`;
  }

  private invaders(now: EnvFrame, next: EnvFrame, go: boolean, t: number, e: number, lost: EnvFrame | null): void {
    const b = this.batches, w = now.words, nw = next.words, n = this.side, cells = n * n;
    let aliens = 0;
    for (let k = 0; k < cells; k++) {
      const x = k % n, y = Math.floor(k / n);
      if (bit(w, 0, k)) {
        aliens++;
        b.alien!.add(x, y, y % 2 ? 0x7affd6 : 0xb6ff7a);
      }
      if (go) {
        if (bit(nw, 4, k)) b.shot!.add(x, y + 1 - t, 0x9ef6ff);
        if (bit(nw, 8, k)) b.bomb!.add(x, y > 0 && bit(w, 8, k - n) ? y - 1 + t : y, 0xff8a3d, y > 0 && bit(w, 8, k - n) ? 1 : smoothstep01(t));
        if (bit(w, 4, k) && !(y > 0 && bit(nw, 4, k - n))) {
          if (y > 0 && next.reward > 0) {
            b.shot!.add(x, y - t, 0x9ef6ff, 1 - t);
            b.spark!.add(x, y - 1, 0xb6ff7a, 0.4 + 0.8 * t);
          } else b.shot!.add(x, y - t, 0x9ef6ff);
        }
        if (bit(w, 8, k) && y === n - 1) b.bomb!.add(x, y + t, 0xff8a3d);
      } else {
        if (bit(w, 4, k)) b.shot!.add(x, y, 0x9ef6ff);
        if (bit(w, 8, k)) b.bomb!.add(x, y, 0xff8a3d);
      }
    }
    const cx = go ? u8(w, 48) + (u8(nw, 48) - u8(w, 48)) * e : u8(w, 48);
    b.cannon!.add(cx, n - 1, 0xff7ad9);
    if (lost) b.spark!.add(u8(lost.words, 48), n - 1, MISS, 1.3 * (1 - t));
    this.ledger.a = `column ${u8(w, 48) + 1}`;
    this.ledger.b = `${aliens} marching ${u8(w, 49) ? '→' : '←'}`;
    this.ledger.c = u8(w, 50) ? `${u8(w, 50)} steps` : 'ready';
  }

  private freeway(now: EnvFrame, next: EnvFrame, go: boolean, t: number, e: number, lost: EnvFrame | null): void {
    const b = this.batches, w = now.words, nw = next.words, n = this.side;
    for (let i = 0; i < 8; i++) {
      const x = u8(w, i), speed = s8(go ? nw : w, 16 + i), dir = speed < 0 ? -1 : 1;
      const pos = go && u8(nw, i) !== x ? x + dir * t : x, tint = CARS[(Math.abs(speed) - 1 + CARS.length) % CARS.length]!;
      b.car!.add(pos, i + 1, tint, 1, dir < 0);
      if (pos > n - 1) b.car!.add(pos - n, i + 1, tint, 1, dir < 0);
      if (pos < 0) b.car!.add(pos + n, i + 1, tint, 1, dir < 0);
    }
    const y = u8(w, 24), ny = u8(nw, 24);
    if (go && next.reward > 0) {
      b.chicken!.add(FREEWAY_CHICKEN, y - e, 0xfff1c4, 1 - 0.6 * t);
      b.spark!.add(FREEWAY_CHICKEN, y - 1, GOLD, 0.3 + 0.9 * t);
    } else if (go && ny === n - 1 && y !== n - 1) {
      b.chicken!.add(FREEWAY_CHICKEN, y, 0xfff1c4, 1 - 0.3 * t);
      b.spark!.add(FREEWAY_CHICKEN, y, MISS, 0.4 + 0.8 * t);
    } else b.chicken!.add(FREEWAY_CHICKEN, go ? y + (ny - y) * e : y, 0xfff1c4);
    if (lost) b.spark!.add(FREEWAY_CHICKEN, u8(lost.words, 24), 0xdcf4ff, 1.2 * (1 - t));
    this.ledger.a = y === n - 1 ? 'on the kerb' : `lane ${n - 1 - y} of 8`;
    this.ledger.b = u8(w, 25) ? `cooling ${u8(w, 25)}` : 'free to step';
    this.ledger.c = `${formatCount(w[7]!)} steps left`;
  }

  private asterix(now: EnvFrame, next: EnvFrame, go: boolean, t: number, e: number, lost: EnvFrame | null): void {
    const b = this.batches, w = now.words, nw = next.words;
    let golds = 0, foes = 0;
    for (let r = 0; r < 8; r++) {
      const v = u8(w, r);
      if (!v) continue;
      const nv = u8(nw, r), x = v & 15, same = go && nv && (nv & 0x30) === (v & 0x30), east = (v & 0x10) !== 0;
      const moving = go && !same && !nv && (east ? x === 9 : x === 0);
      const pos = same ? x + ((nv & 15) - x) * t : moving ? x + (east ? t : -t) : x;
      const gold = (v & 0x20) !== 0;
      if (gold) golds++;
      else foes++;
      const taken = go && !same && !moving && gold && next.reward > 0;
      if (gold) b.gem!.add(pos, r + 1, GOLD, taken ? 1 - t : 1);
      else b.foe!.add(pos, r + 1, 0xff4d6d, 1, !east);
      if (taken) b.spark!.add(x, r + 1, GOLD, 0.4 + 0.8 * t);
    }
    const hx = go ? u8(w, 8) + (u8(nw, 8) - u8(w, 8)) * e : u8(w, 8), hy = go ? u8(w, 9) + (u8(nw, 9) - u8(w, 9)) * e : u8(w, 9);
    b.hero!.add(hx, hy, 0x7ad0ff);
    if (lost) b.spark!.add(u8(lost.words, 8), u8(lost.words, 9), MISS, 1.3 * (1 - t));
    this.ledger.a = `(${u8(w, 8) + 1}, ${u8(w, 9)})`;
    this.ledger.b = `${golds} gold · ${foes} foes`;
    this.ledger.c = `spawn ${u8(w, 12)} · move ${u8(w, 13)}`;
  }

  private seaquest(now: EnvFrame, next: EnvFrame, go: boolean, t: number, e: number, lost: EnvFrame | null): void {
    const b = this.batches, w = now.words, nw = next.words, n = this.side;
    const kinds = [b.fish!, b.sub!, b.diver!, b.torpedo!, b.mine!], tints = [0x7dffb2, 0xff5a6e, 0xa6e3ff, 0xfff4b0, 0xff7a4a];
    const src = go ? next : now, sw = src.words;
    for (let l = 0; l < 5; l++) {
      const count = u8(sw, SEA_COUNT[l]!), base = SEA_VIEW[l]!, always = l >= SEA_SHOTS;
      for (let i = 0; i < count; i++) {
        const v = src.view[base + i]!, x = v & 15, y = (v >> 4) & 15, east = (v & 0x100) !== 0, dir = east ? 1 : -1;
        let from = x;
        if (go) {
          if (always) from = x - dir;
          else for (let j = 0; j < u8(w, SEA_COUNT[l]!); j++) {
            const o = now.view[base + j]!;
            if (((o >> 4) & 15) === y && (o & 0x100) === (v & 0x100) && (o & 15) === x - dir && ((o >> 12) & 15) === 0) { from = x - dir; break; }
          }
        }
        kinds[l]!.add(from + (x - from) * t, y, tints[l]!, 1, !east);
      }
      if (!go) continue;
      for (let j = 0; j < u8(w, SEA_COUNT[l]!); j++) {
        const o = now.view[base + j]!, x = o & 15, y = (o >> 4) & 15, east = (o & 0x100) !== 0, to = x + (east ? 1 : -1);
        if ((to < 0 || to > n - 1) && (always || ((o >> 12) & 15) === 0)) kinds[l]!.add(x + (to - x) * t, y, tints[l]!, 1, !east);
      }
    }
    if (go && next.reward > 0 && u8(nw, 9) !== 0) {
      for (let j = 0; j < u8(w, SEA_COUNT[SEA_SHOTS]!); j++) {
        const o = now.view[SEA_VIEW[SEA_SHOTS]! + j]!, x = (o & 15) + (o & 0x100 ? 1 : -1), y = (o >> 4) & 15;
        let alive = false;
        for (let i = 0; i < u8(nw, SEA_COUNT[SEA_SHOTS]!); i++) {
          const v = next.view[SEA_VIEW[SEA_SHOTS]! + i]!;
          if ((v & 15) === x && ((v >> 4) & 15) === y) alive = true;
        }
        if (!alive && x >= 0 && x < n) b.spark!.add(x, y, 0xfff4b0, 0.4 + 0.8 * t);
      }
    }
    const sx = go ? u8(w, 8) + (u8(nw, 8) - u8(w, 8)) * e : u8(w, 8), sy = go ? u8(w, 9) + (u8(nw, 9) - u8(w, 9)) * e : u8(w, 9);
    b.sub!.add(sx, sy, 0xffd24a, 1, !u8(go ? nw : w, 10));
    if (lost) b.spark!.add(u8(lost.words, 8), u8(lost.words, 9), MISS, 1.3 * (1 - t));
    const oxygen = Math.max(0, w[0]! | 0), rescued = Math.max(0, w[1]! | 0);
    this.meter.value = now.observations[3] ?? 0;
    this.pips.value = rescued;
    this.ledger.a = `(${u8(w, 8) + 1}, ${u8(w, 9)}) facing ${u8(w, 10) ? '→' : '←'}`;
    this.ledger.b = `${oxygen}${u8(w, 9) === 0 ? ' · surfaced' : ''}`;
    this.ledger.c = `${rescued} aboard`;  }

  private count(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, goals: 0, harms: 0 });
    if (at === tally.through) return;
    tally.goals += countEvents(log, 'goal', tally.through, at);
    tally.harms += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }
}

const clampNode = (v: N<'float'>) => min(max(v, 0), 1);

function popcount(v: number): number {
  let c = 0;
  for (let x = v >>> 0; x; x &= x - 1) c++;
  return c;
}

export default (() => new MinAtarScene()) satisfies SceneFactory;
