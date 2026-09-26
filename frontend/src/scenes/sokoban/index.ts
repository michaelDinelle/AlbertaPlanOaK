import * as THREE from 'three/webgpu';
import { abs, exp, float, floor, fract, length, max, min, mix, select, sin, smoothstep, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, type GridSize } from '@kit/grid';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { SOKOBAN_BOX, SOKOBAN_BOXES, SOKOBAN_PX, SOKOBAN_PY, SOKOBAN_SLOTS, SOKOBAN_TARGET, sokobanDecoder, sokobanLevel, sokobanOpen } from '@sim/envs/sokoban';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Kind = 'rest' | 'walk' | 'push' | 'bump';

interface Layout {
  pose: Float32Array;
  walls: boolean[] | null;
  key: string;
  clock: number;
  placed: number;
}

const PAD = 2;
const RIM = 0.32;
const INSET = 0.015;
const CRATE = 1.2;
const KEEPER = 1.1;
const TURN = 18;
const BENCH = 0x15110e;
const CONCRETE = 0x8e897d;
const GROOVE = 0x57534b;
const BRICK = 0x7c4433;
const MORTAR = 0x3b2a24;
const PAINT = 0xf0a534;
const WOOD = 0xc68b4c;
const FRAME = 0x87552a;
const STAINED = 0x3f8f63;
const EMERALD = 0x8af2bd;
const VEST = 0xee7424;
const HAT = 0xf4c632;
const SLEEVE = 0x2d4675;
const SKIN = 0xd9a47c;
const GOLD = TOKENS.celebrationGold;
const GLOOM = TOKENS.failureBlue;
const WIRE = new THREE.Color(0xc98a6a);
const COMPASS: Record<string, string> = { '0,-1': 'north', '0,1': 'south', '-1,0': 'west', '1,0': 'east' };

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width * CELL, depth * CELL).rotateX(-Math.PI / 2);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const ease = (u: number) => u * u * (3 - 2 * u);
const angleOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
const ellipse = (p: N<'vec2'>, rx: number, ry: number) => length(p.div(vec2(rx, ry))).sub(1).mul(Math.min(rx, ry));
const hash = (c: N<'vec2'>) => fract(sin(c.x.mul(12.9898).add(c.y.mul(78.233))).mul(43758.5453));

class Layers {
  pre: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): void {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
  }

  material(additive = false): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.pre.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha;
    if (additive) material.blending = THREE.AdditiveBlending;
    return material;
  }
}

interface Crate {
  mesh: THREE.Mesh;
  placed: { value: number };
  glow: { value: number };
  from: THREE.Vector3;
  to: THREE.Vector3;
  on: boolean;
  landedAt: number;
}

class SokobanScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.35, fog: { from: ['cave', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'sokoban';

  private side = 5;
  private open = true;
  private grid: GridSize = { w: 5, h: 5 };
  private limit = 1;
  private levels = 0;
  private mask: THREE.DataTexture | null = null;
  private readonly flare = uniform(0);
  private readonly gloom = uniform(0);
  private readonly reach = uniform(0);
  private readonly crates: Crate[] = [];
  private keeper: THREE.Mesh | null = null;
  private readonly owned: THREE.Mesh[] = [];

  private clock = 0;
  private step = -1;
  private paintedKey = '';
  private kind: Kind = 'rest';
  private heading = 0;
  private aim = 0;
  private flareAt = -Infinity;
  private gloomAt = -Infinity;
  private log: FrameUpdate['eventLog'] | null = null;
  private now: Layout | null = null;
  private readonly keeperFrom = new THREE.Vector3();
  private readonly keeperTo = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly lift = new THREE.Vector3();
  private readonly truth = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly ledger = { x: 0, y: 0, reward: 0, outcome: '—', level: 0 };

  override legend: LegendSpec = {
    title: 'SOKOBAN',
    rows: [
      { key: 'keeper', label: 'keeper', tint: VEST, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'move', label: 'last move', tint: 0xdcecff, value: () => this.ledger.outcome },
      { key: 'placed', label: 'on target', tint: EMERALD, value: () => `${this.now?.placed ?? 0} / ${this.crateCount()}` },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'solves', label: 'solved', tint: PAINT, value: () => this.solves() },
    ],
    note: () => (this.open ? `step ${formatCount(this.now?.clock ?? 0)} of ${formatCount(this.limit)} before a new deal` : `level ${formatCount(this.ledger.level + 1)} of ${formatCount(this.levels)} · step ${formatCount(this.now?.clock ?? 0)} of ${formatCount(this.limit)}`),
    bar: () => clamp01((this.now?.clock ?? 0) / this.limit),
  };

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private texel(x: number, y: number): number {
    return ((y + PAD) * (this.side + 2 * PAD) + x + PAD) * 4;
  }

  private crateCount(): number {
    const pose = this.now?.pose;
    let n = 0;
    for (let i = 0; pose && i < SOKOBAN_BOXES; i++) if (pose[SOKOBAN_BOX + 2 * i]! >= 0) n++;
    return n;
  }

  private layout(frame: EnvFrame): Layout {
    const pose = new Float32Array(SOKOBAN_SLOTS);
    sokobanDecoder.decode(frame, pose);
    const open = sokobanOpen(frame), level = open ? null : sokobanLevel(frame);
    let placed = 0;
    for (let i = 0; i < SOKOBAN_BOXES; i++) {
      const bx = pose[SOKOBAN_BOX + 2 * i]!, by = pose[SOKOBAN_BOX + 2 * i + 1]!;
      if (bx >= 0 && this.onTarget(pose, bx, by)) placed++;
    }
    return {
      pose,
      walls: level ? level.floor.map((f) => !f) : null,
      key: open ? `${pose[SOKOBAN_TARGET]},${pose[SOKOBAN_TARGET + 1]}` : `${frame.words[4]}`,
      clock: open ? frame.words[6]! : frame.words[3]!,
      placed,
    };
  }

  private onTarget(pose: Float32Array, x: number, y: number): boolean {
    for (let t = 0; t < SOKOBAN_BOXES; t++) if (pose[SOKOBAN_TARGET + 2 * t] === x && pose[SOKOBAN_TARGET + 2 * t + 1] === y) return true;
    return false;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, side = parameter(env, 'side');
    if (!Number.isInteger(side) || side < 3) throw new Error(`${env.envId} lacks its side`);
    this.side = side;
    this.open = env.words >= 7;
    this.grid = { w: side, h: side };
    this.limit = this.open ? constant(env, 'SOKOBAN_PATIENCE') : parameter(env, 'maxSteps');
    this.levels = this.open ? 0 : parameter(env, 'pool');
    const half = side / 2 + (this.open ? RIM : 0) + 0.35;
    return new THREE.Box2(new THREE.Vector2(-half, -half).multiplyScalar(CELL), new THREE.Vector2(half, half).multiplyScalar(CELL));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb8a58c).multiplyScalar(0.3);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const side = this.side, accent = this.accentOf(ctx);
    return {
      cellCapacity: side * side,
      lattice: { w: side, h: side, origin: [-side / 2 * CELL, 0.02, -side / 2 * CELL], u: [CELL, 0, 0], v: [0, 0, CELL] },
      frame: (state) => ({ originX: 0, originY: 0, originZ: 0, agent: this.place(state.pose[SOKOBAN_PX]!, state.pose[SOKOBAN_PY]!, this.truth), cone: this.noCone, coneCount: 0 }),
      debug: (frame, kit) => this.groundTruth(frame, kit, accent),
    };
  }

  protected build(_ctx: SceneContext, { pool }: Stage2D): void {
    const side = this.side, margin = this.open ? RIM : 0;
    this.scene.background = new THREE.Color(BENCH);

    const TW = side + 2 * PAD, data = new Uint8Array(TW * TW * 4).fill(0);
    for (let y = -PAD; y < side + PAD; y++) for (let x = -PAD; x < side + PAD; x++) if (x < 0 || y < 0 || x >= side || y >= side) data[this.texel(x, y)] = 255;
    const mask = new THREE.DataTexture(data, TW, TW, THREE.RGBAFormat);
    mask.magFilter = mask.minFilter = THREE.NearestFilter;
    mask.needsUpdate = true;
    this.mask = mask;

    const sample = (c: N<'vec2'>) => texture(mask, c.add(PAD + 0.5).div(vec2(TW, TW)));
    const walls = (q: N<'vec2'>): N<'float'> => {
      const c = floor(q);
      let solid: N<'float'> = float(4), gap: N<'float'> = float(4);
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        const n = c.add(vec2(i, j)), on = sample(n).r.greaterThan(0.5), d = sdBox(q.sub(n).sub(0.5), vec2(0.5));
        solid = min(solid, select(on, d, float(4)));
        gap = min(gap, select(on, float(4), d));
      }
      return select(sample(c).r.greaterThan(0.5), gap.negate(), solid).add(INSET);
    };

    const span = side + 2 * margin + 0.6;
    const p = uv().sub(0.5).mul(span), q = vec2(p.x.add(side / 2), float(side / 2).sub(p.y));
    const cell = floor(q), local = fract(q).sub(0.5), centre = vec2(side / 2), half = vec2(side / 2);
    const dBoard = sdBox(q.sub(centre), half.add(margin), margin > 0 ? 0.12 : 0.06);

    const border = min(float(0.5).sub(abs(local.x)), float(0.5).sub(abs(local.y)));
    const tile = hash(cell).sub(0.5).mul(0.07).add(1);
    let concrete: N<'vec3'> = rgb(CONCRETE).mul(tile).mul(fractal(q.mul(3.1), 4).mul(0.07).add(1)).mul(fractal(q.mul(0.6).add(4.1), 3).mul(0.05).add(1));
    concrete = mix(concrete, rgb(GROOVE), coverage(border.sub(0.018)).mul(0.75));
    const target = sample(cell).g.greaterThan(0.5);
    const ring = abs(length(local).sub(0.3)).sub(0.03), diamond = abs(local.x).add(abs(local.y)).sub(0.1);
    const scuff = fractal(q.mul(9), 3).mul(0.25).add(0.8);
    const paint = select(target, max(coverage(ring), coverage(diamond)).mul(scuff).min(1), float(0));
    const halo = select(target, exp(length(local).mul(-5)).mul(0.22), float(0));
    concrete = mix(concrete, rgb(PAINT), paint).add(rgb(PAINT).mul(halo.mul(this.flare.mul(1.5).add(0.35))));
    const shadow = smoothstep(-0.05, 0.2, walls(q.sub(vec2(0.07, 0.1)))).oneMinus().mul(0.5);
    const floorColor = concrete.mul(shadow.oneMinus()).mul(float(1).sub(this.gloom.mul(0.35))).add(rgb(GLOOM).mul(this.gloom.mul(0.08)));

    const dWall = max(walls(q), dBoard);
    const bq = q.mul(vec2(2.5, 5)), row = floor(bq.y), shifted = vec2(bq.x.add(fract(row.mul(0.5))), bq.y);
    const bl = fract(shifted).sub(0.5), bid = floor(shifted);
    const mortar = min(float(0.5).sub(abs(bl.x)).div(2.5), float(0.5).sub(abs(bl.y)).div(5));
    let brick: N<'vec3'> = rgb(BRICK).mul(hash(bid.add(3.7)).sub(0.5).mul(0.22).add(1)).mul(fractal(q.mul(7), 3).mul(0.1).add(1));
    brick = mix(brick, rgb(MORTAR), coverage(mortar.sub(0.012)));
    const bevel = smoothstep(-0.07, -0.015, dWall).mul(0.3).add(1).sub(smoothstep(-0.2, -0.07, dWall).oneMinus().mul(0.12));
    const wallTop = brick.mul(bevel).mul(0.9);
    const drop = smoothstep(-0.05, 0.45, sdBox(q.sub(centre).sub(vec2(0.08, 0.14)), half.add(margin), 0.12)).oneMinus().mul(0.65);

    const board = new Layers();
    board.over(drop, vec3(0));
    board.over(coverage(sdBox(q.sub(centre), half)), floorColor);
    board.over(coverage(dWall), wallTop);
    this.add(new THREE.Mesh(flat(span, span), board.material()), 0);

    for (let i = 0; i < SOKOBAN_BOXES; i++) this.crates.push(this.crate(i));

    const m = uv().sub(0.5).mul(KEEPER / 1.45), reach = this.reach;
    const keeper = new Layers();
    keeper.over(smoothstep(-0.02, 0.1, ellipse(m.sub(vec2(0.04, -0.07)), 0.24, 0.17)).oneMinus().mul(0.45), vec3(0));
    const arm = (s: number) => sdSegment(m, vec2(s * 0.17, 0), vec2(s * 0.12, reach.mul(0.08).add(0.17)), 0.045);
    const hand = (s: number) => sdCircle(m.sub(vec2(s * 0.12, reach.mul(0.08).add(0.19))), 0.045);
    keeper.over(coverage(min(arm(-1), arm(1))), rgb(SLEEVE));
    keeper.over(coverage(min(hand(-1), hand(1))), rgb(SKIN));
    const body = ellipse(m.sub(vec2(0, -0.03)), 0.21, 0.13);
    const stripe = min(abs(m.x.sub(0.09)), abs(m.x.add(0.09))).sub(0.018);
    keeper.over(coverage(body), mix(rgb(VEST), rgb(0xf6f0da), coverage(stripe).mul(0.85)).mul(smoothstep(-0.12, 0, body).mul(-0.25).add(1)));
    const brim = ellipse(m.sub(vec2(0, 0.03)), 0.125, 0.14), hat = sdCircle(m.sub(vec2(0, 0.0)), 0.1);
    keeper.over(coverage(brim), rgb(HAT).mul(0.78));
    keeper.over(coverage(hat), rgb(HAT).mul(smoothstep(-0.1, 0, hat).mul(-0.2).add(1.05)));
    keeper.over(coverage(max(abs(m.x).sub(0.016), abs(m.y).sub(0.085))), rgb(0xfde58a));
    this.keeper = this.add(new THREE.Mesh(flat(KEEPER, KEEPER), keeper.material()), 4);

    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 1.5 : 5; });
  }

  private crate(i: number): Crate {
    const placed = uniform(0), glow = uniform(0);
    const s = uv().sub(0.5).mul(CRATE);
    const box = sdBox(s, vec2(0.39), 0.045), inner = sdBox(s, vec2(0.29), 0.02);
    const plank = abs(fract(s.y.mul(1 / 0.145).add(0.5)).sub(0.5)).mul(0.145).sub(0.006);
    const brace = min(sdSegment(s, vec2(-0.27, -0.27), vec2(0.27, 0.27), 0.05), sdSegment(s, vec2(-0.27, 0.27), vec2(0.27, -0.27), 0.05));
    const nail = sdCircle(vec2(abs(s.x), abs(s.y)).sub(0.34), 0.022);
    const grainTone = fractal(s.mul(vec2(3, 22)).add(i * 5.3), 3).mul(0.14).add(0.93);
    const wood = mix(rgb(WOOD), rgb(STAINED), placed).mul(grainTone);
    const frame = mix(rgb(FRAME), rgb(STAINED).mul(0.62), placed).mul(grainTone);
    const layers = new Layers();
    layers.over(smoothstep(-0.03, 0.12, sdBox(s.sub(vec2(0.06, -0.09)), vec2(0.39), 0.05)).oneMinus().mul(0.55), vec3(0));
    layers.over(exp(max(box, 0).mul(-26)).mul(0.55).mul(placed).mul(coverage(box).oneMinus()), rgb(EMERALD).mul(glow.mul(0.5).add(0.55)));
    let face: N<'vec3'> = mix(frame, wood.mul(coverage(plank).mul(-0.3).add(1)), coverage(inner));
    face = mix(face, frame.mul(1.08), coverage(brace).mul(coverage(inner.sub(0.04))));
    face = mix(face, rgb(0x2a2420), coverage(nail));
    face = face.mul(smoothstep(-0.06, -0.01, box).mul(-0.3).add(1)).add(rgb(EMERALD).mul(glow.mul(0.35)));
    layers.over(coverage(box), face);
    const mesh = this.add(new THREE.Mesh(flat(CRATE, CRATE), layers.material()), 3);
    mesh.visible = false;
    return { mesh, placed, glow, from: new THREE.Vector3(), to: new THREE.Vector3(), on: false, landedAt: -Infinity };
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.owned.push(mesh);
    this.scene.add(mesh);
    return mesh;
  }

  protected override drop(): void {
    this.mask?.dispose();
    this.mask = null;
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
    this.crates.length = 0;
    this.keeper = null;
  }

  private paint(layout: Layout): void {
    const mask = this.mask;
    if (!mask || layout.key === this.paintedKey) return;
    const data = mask.image.data as Uint8Array, side = this.side, pose = layout.pose;
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
      const at = this.texel(x, y);
      data[at] = layout.walls?.[y * side + x] ? 255 : 0;
      data[at + 1] = this.onTarget(pose, x, y) ? 255 : 0;
    }
    mask.needsUpdate = true;
    this.paintedKey = layout.key;
  }

  protected tick(frame: FrameUpdate, { ctx, pool }: Stage2D): void {
    const dt = frame.dtReal, step = frame.state.step;
    this.clock += dt;
    pool.update(dt, this.clock);
    this.log = frame.eventLog;
    if (step !== this.step || frame.scrubbed || !this.now) this.enter(frame, step, pool, this.accentOf(ctx));
    this.animate(frame, step);
  }

  private enter(frame: FrameUpdate, step: number, pool: Stage2D['pool'], accent: number): void {
    const f = frame.frameAt(step), now = this.layout(f), prior = step > 0 ? frame.frameAt(step - 1) : null;
    const before = prior && prior.step === step - 1 ? this.layout(prior) : null;
    const joined = !!before && !f.done && before.key === now.key && now.clock === before.clock + 1 && !frame.scrubbed;
    this.step = step;
    this.now = now;
    this.paint(now);
    const P = now.pose, B = joined ? before!.pose : P;
    const px = P[SOKOBAN_PX]!, py = P[SOKOBAN_PY]!, dx = px - B[SOKOBAN_PX]!, dy = py - B[SOKOBAN_PY]!;
    let pushed = false;
    for (let i = 0; i < this.crates.length; i++) {
      const c = this.crates[i]!, x = P[SOKOBAN_BOX + 2 * i]!, y = P[SOKOBAN_BOX + 2 * i + 1]!;
      c.mesh.visible = x >= 0;
      if (x < 0) continue;
      const ox = B[SOKOBAN_BOX + 2 * i]!, oy = B[SOKOBAN_BOX + 2 * i + 1]!;
      if (ox !== x || oy !== y) pushed = true;
      this.place(ox, oy, c.from);
      this.place(x, y, c.to);
      const on = this.onTarget(P, x, y);
      if (joined && on && !c.on) c.landedAt = this.clock;
      if (!joined) c.landedAt = -Infinity;
      c.on = on;
      if (frame.scrubbed || !joined) c.placed.value = on ? 1 : 0;
    }
    this.place(B[SOKOBAN_PX]!, B[SOKOBAN_PY]!, this.keeperFrom);
    this.place(px, py, this.keeperTo);
    this.kind = !joined ? 'rest' : pushed ? 'push' : dx !== 0 || dy !== 0 ? 'walk' : 'bump';
    if (this.kind === 'walk' || this.kind === 'push') this.aim = angleOf(dx, dy);
    const name = COMPASS[`${dx},${dy}`];
    this.ledger.outcome = !joined ? (f.done || now.clock === 0 ? 'new deal' : '—') : this.kind === 'push' ? `pushed ${name}` : this.kind === 'walk' ? `walked ${name}` : 'no move';
    this.ledger.x = px;
    this.ledger.y = py;
    this.ledger.reward = frame.state.reward;
    this.ledger.level = this.open ? 0 : f.words[4]!;
    if (frame.scrubbed) return;
    for (const event of frame.events) {
      if (event.type === 'goal') {
        this.flareAt = this.clock;
        const cell = before ? this.completed(before.pose) : null;
        if (cell) {
          this.place(cell[0], cell[1], this.scratch);
          pool.ripple(this.scratch, GOLD, 0.7, this.clock);
          pool.ripple(this.scratch, EMERALD, 0.4, this.clock);
          this.lift.set(this.scratch.x, 40, this.scratch.z);
          pool.burst(this.lift, GOLD, 1.1, this.clock);
        }
        this.lift.set(0, 40, 0);
        pool.burst(this.lift, accent, 0.8, this.clock);
        pool.pulse.fire(0.8);
      } else if (event.type === 'harm' || event.type === 'teleport') {
        this.gloomAt = this.clock;
        this.place(px, py, this.scratch);
        pool.ripple(this.scratch, GLOOM, 0.5, this.clock);
      } else if (event.type === 'cue') {
        for (const c of this.crates) if (c.mesh.visible && c.from.distanceToSquared(c.to) > 0) pool.ripple(c.to, event.value > 0 ? EMERALD : GLOOM, 0.4, this.clock);
      }
    }
  }

  private completed(pose: Float32Array): [number, number] | null {
    for (let t = 0; t < SOKOBAN_BOXES; t++) {
      const x = pose[SOKOBAN_TARGET + 2 * t]!, y = pose[SOKOBAN_TARGET + 2 * t + 1]!;
      if (x < 0) continue;
      let covered = false;
      for (let i = 0; i < SOKOBAN_BOXES; i++) if (pose[SOKOBAN_BOX + 2 * i] === x && pose[SOKOBAN_BOX + 2 * i + 1] === y) covered = true;
      if (!covered) return [x, y];
    }
    return null;
  }

  private animate(frame: FrameUpdate, step: number): void {
    const keeper = this.keeper;
    if (!keeper) return;
    const u = this.kind === 'rest' ? 1 : clamp01(frame.position - step), k = ease(u);
    this.scratch.lerpVectors(this.keeperFrom, this.keeperTo, k);
    const nudge = this.kind === 'bump' ? 0.09 * Math.sin(Math.PI * u) : 0;
    const turn = Math.atan2(Math.sin(this.aim - this.heading), Math.cos(this.aim - this.heading));
    this.heading = frame.scrubbed ? this.aim : this.heading + turn * (1 - Math.exp(-TURN * frame.dtReal));
    keeper.position.set(this.scratch.x - Math.sin(this.aim) * nudge * CELL, 0, this.scratch.z - Math.cos(this.aim) * nudge * CELL);
    keeper.rotation.y = this.heading;
    const pushing = this.kind === 'push' ? 1 : this.kind === 'bump' ? Math.sin(Math.PI * u) : 0;
    this.reach.value += (pushing - this.reach.value) * (1 - Math.exp(-14 * frame.dtReal));
    const fade = 1 - Math.exp(-10 * frame.dtReal);
    for (const c of this.crates) {
      if (!c.mesh.visible) continue;
      c.mesh.position.lerpVectors(c.from, c.to, k);
      c.placed.value += ((c.on ? 1 : 0) - c.placed.value) * (frame.scrubbed ? 1 : fade);
      const age = this.clock - c.landedAt;
      c.glow.value = age >= 0 && age < 3 ? Math.exp(-age / 0.5) : 0;
    }
    this.flare.value = Math.exp(-Math.max(0, this.clock - this.flareAt) / 0.6);
    this.gloom.value = Math.exp(-Math.max(0, this.clock - this.gloomAt) / 0.5);
  }

  private solves(): string {
    const log = this.log, step = this.step;
    if (!log) return '0';
    const count = countEvents(log, 'goal', -1, step), last = lastEvent(log, 'goal', step);
    if (last === undefined) return `${formatCount(count)} · none yet`;
    const since = step - last;
    return `${formatCount(count)} · ${since === 0 ? 'now' : `${formatCount(since)} ${since === 1 ? 'step' : 'steps'} ago`}`;
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit, accent: number): void {
    const side = this.side, boxes = this.overlay?.cellBoxes(), now = this.now;
    if (!now) return;
    if (boxes) {
      boxes.begin();
      for (let i = 0; now.walls && i < side * side; i++) {
        if (!now.walls[i]) continue;
        this.place(i % side, Math.floor(i / side), this.scratch);
        boxes.box(this.scratch.x, this.scratch.z, 0, 0, 0, 0, WIRE.r, WIRE.g, WIRE.b);
      }
      boxes.commit();
    }
    const P = frame.state.pose;
    for (let i = 0; i < SOKOBAN_BOXES; i++) {
      const tx = P[SOKOBAN_TARGET + 2 * i]!, ty = P[SOKOBAN_TARGET + 2 * i + 1]!, bx = P[SOKOBAN_BOX + 2 * i]!, by = P[SOKOBAN_BOX + 2 * i + 1]!;
      const target = kit.curve(`target${i}`, 10, PAINT), placed = kit.curve(`placed${i}`, 10, EMERALD), loose = kit.curve(`crate${i}`, 10, accent);
      const crate = bx >= 0 && this.onTarget(P, bx, by) ? placed : loose;
      if (tx >= 0) for (const r of [0.3, 0.27]) this.outline(target.point.bind(target), tx, ty, r, true);
      if (bx >= 0) for (const r of [0.4, 0.37]) this.outline(crate.point.bind(crate), bx, by, r, false);
    }
  }

  private outline(point: (x: number, y: number, z: number) => void, x: number, y: number, r: number, diamond: boolean): void {
    this.place(x, y, this.scratch);
    const corners = diamond ? [[0, -r], [r, 0], [0, r], [-r, 0], [0, -r]] : [[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]];
    for (const [cx, cz] of corners) point(this.scratch.x + cx! * CELL, 0.1, this.scratch.z + cz! * CELL);
  }
}

export default (() => new SokobanScene()) satisfies SceneFactory;
