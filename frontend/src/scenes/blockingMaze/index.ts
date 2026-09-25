import * as THREE from 'three/webgpu';
import { abs, exp, float, floor, fract, length, max, min, mix, select, sin, smoothstep, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { DebugCurve, DebugKit, GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, type GridSize } from '@kit/grid';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { BLOCKING_X, BLOCKING_Y } from '@sim/envs/blocking';
import type { EnvFrame, GridCell } from '@sim/types';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Mask = { w: number; h: number; walls: number[] };
type Shift = readonly [number, number];
type Kind = 'rest' | 'move' | 'bump' | 'goal';

const PAD = 2;
const RIM = 0.3;
const SKIRT = 0.7;
const INSET = 0.06;
const TRAIL = 32;
const SCAN = 8192;
const SLIDE_STEPS = 2;
const CHEER_SECONDS = 0.6;
const FLASH_SECONDS = 0.18;
const FLARE_SECONDS = 0.5;
const NOSE = 0.32;
const TURN = 16;
const MOUSE = 1.1;
const PORT = 1;
const BENCH = 0x0d0f0d;
const FLOOR = 0x7f8991;
const LINE = 0xa6b0b8;
const WOOD = 0xa99570;
const FUR = 0xeee7da;
const PINK = 0xe7a3a6;
const GOLD = TOKENS.celebrationGold;
const SPARK = 0xcfe4ff;
const DUST = 0xd6c298;
const LAMP = 0xffe2b4;
const LAMP_LIGHT = 1.5;
const PORT_LIGHT = 4.5;
const SHADOW = [0.06, 0.1] as const;
const WIRE = new THREE.Color(0x8fa6c4);
const COMPASS: Record<string, string> = { '0,-1': '↑ north', '0,1': '↓ south', '-1,0': '← west', '1,0': 'east →' };

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width * CELL, depth * CELL).rotateX(-Math.PI / 2);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const ease = (u: number) => u * u * (3 - 2 * u);
const pop = (u: number) => 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2;
const angleOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
const ellipse = (p: N<'vec2'>, rx: number, ry: number) => length(p.div(vec2(rx, ry))).sub(1).mul(Math.min(rx, ry));

function smin(a: N<'float'>, b: N<'float'>, k: number): N<'float'> {
  const h = max(float(k).sub(abs(a.sub(b))), 0).div(k);
  return min(a, b).sub(h.mul(h).mul(k / 4));
}

function shiftOf(a: Mask, b: Mask): Shift | null {
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
    let same = true;
    for (let y = 0; y < a.h && same; y++) for (let x = 0; x < a.w && same; x++) {
      const sx = x - dx, sy = y - dy, was = sx >= 0 && sy >= 0 && sx < a.w && sy < a.h ? a.walls[sy * a.w + sx]! : 0;
      same = (was !== 0) === (b.walls[y * b.w + x] !== 0);
    }
    if (same) return [dx, dy];
  }
  return null;
}

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

class BlockingMazeScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'blockingMaze';

  private grid: GridSize = { w: 1, h: 1 };
  private phases: Mask[] = [];
  private period = 1;
  private goal: GridCell = { x: 0, y: 0 };
  private start: GridCell | null = null;
  private union: number[] = [];
  private shifts: (Shift | null)[][] = [];
  private openings: number[][] = [];
  private reach = 0;
  private capacity = 1;
  private readonly delta: (Shift | null)[] = [null, null, null, null];

  private mask: THREE.DataTexture | null = null;
  private readonly slide = uniform(new THREE.Vector2());
  private readonly startCell = uniform(new THREE.Vector2(-9, -9));
  private readonly flare = uniform(0);
  private readonly flash = uniform(0);
  private readonly sway = uniform(0);
  private blocks: THREE.Mesh[] = [];
  private mouse: THREE.Mesh | null = null;
  private shade: THREE.Mesh | null = null;
  private spark: THREE.Mesh | null = null;
  private readonly owned: THREE.Mesh[] = [];

  private clock = 0;
  private step = -1;
  private phase = -1;
  private shown = -1;
  private slideFrom = -1;
  private slideAt = 0;
  private cheerAt = -Infinity;
  private slideShift: Shift = [0, 0];
  private kind: Kind = 'rest';
  private outcome: Kind = 'rest';
  private action = 0;
  private heading = 0;
  private aim = 0;
  private flashAt = -Infinity;
  private flareAt = -Infinity;
  private log: FrameUpdate['eventLog'] | null = null;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly goalAt = new THREE.Vector3();
  private readonly startAt = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly truth = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly ledger = { x: 0, y: 0, step: 0, reward: 0 };

  override legend: LegendSpec = {
    title: 'BLOCKING MAZE',
    rows: [
      { key: 'at', label: 'mouse', tint: FUR, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionLabel() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'opening', label: 'opening', tint: WOOD, value: () => this.opening() },
      { key: 'goal', label: 'last goal', tint: 0xf0a860, value: () => this.sinceGoal() },
    ],
    note: () => `slides in ${formatCount(this.period - this.ledger.step % this.period)}`,
    bar: () => (this.ledger.step % this.period) / this.period,
  };

  private phaseAt(step: number): number {
    return Math.floor(step / this.period) % this.phases.length;
  }

  private texel(x: number, y: number): number {
    return ((y + PAD) * (this.grid.w + 2 * PAD) + x + PAD) * 4;
  }

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, phases = env.parameters['phase'] as Mask[] | undefined, first = phases?.[0], goal = env.goal;
    if (!phases || !first || !goal || phases.some((p) => p.w !== first.w || p.h !== first.h || p.walls.length !== p.w * p.h)) throw new Error(`${env.envId} lacks its wall phases or goal`);
    const { w, h } = first;
    this.phases = phases;
    this.grid = { w, h };
    this.goal = { x: goal[0], y: goal[1] };
    this.period = constant(env, 'BLOCKING_PERIOD');
    this.union = first.walls.map((_, i) => (phases.some((p) => p.walls[i]) ? 1 : 0));
    this.capacity = Math.max(1, ...phases.map((p) => p.walls.filter(Boolean).length));
    this.shifts = phases.map((a) => phases.map((b) => (a === b ? null : shiftOf(a, b))));
    this.openings = phases.map((p) => p.walls.flatMap((wall, i) => (this.union[i] && !wall ? [i % w + 1] : [])));
    const halfW = w / 2 + RIM, halfH = h / 2 + RIM;
    return new THREE.Box2(new THREE.Vector2(-halfW - 0.3, -halfH - 0.3).multiplyScalar(CELL), new THREE.Vector2(halfW + 0.3, halfH + 1.2).multiplyScalar(CELL));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa9b9d6).multiplyScalar(0.32);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const { w, h } = this.grid, accent = this.accentOf(ctx);
    return {
      cellCapacity: this.capacity,
      lattice: { w, h, origin: [-w / 2 * CELL, 0.02, -h / 2 * CELL], u: [CELL, 0, 0], v: [0, 0, CELL] },
      frame: (state) => ({ originX: 0, originY: 0, originZ: 0, agent: this.place(state.pose[BLOCKING_X]!, state.pose[BLOCKING_Y]!, this.truth), cone: this.noCone, coneCount: 0 }),
      debug: (frame, kit) => this.groundTruth(frame, kit, accent),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const { w, h } = this.grid, accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(BENCH);
    this.place(this.goal.x, this.goal.y, this.goalAt);

    const TW = w + 2 * PAD, TH = h + 2 * PAD, data = new Uint8Array(TW * TH * 4);
    for (let i = 0; i < w * h; i++) data[this.texel(i % w, Math.floor(i / w)) + 1] = this.union[i] ? 255 : 0;
    const mask = new THREE.DataTexture(data, TW, TH, THREE.RGBAFormat);
    mask.magFilter = mask.minFilter = THREE.NearestFilter;
    mask.needsUpdate = true;
    this.mask = mask;

    const sample = (c: N<'vec2'>) => texture(mask, c.add(PAD + 0.5).div(vec2(TW, TH)));
    const walls = (q: N<'vec2'>): N<'float'> => {
      const c = floor(q);
      let solid: N<'float'> = float(4), open: N<'float'> = float(4);
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        const n = c.add(vec2(i, j)), on = sample(n).r.greaterThan(0.5), d = sdBox(q.sub(n).sub(0.5), vec2(0.5));
        solid = min(solid, select(on, d, float(4)));
        open = min(open, select(on, float(4), d));
      }
      return select(sample(c).r.greaterThan(0.5), open.negate(), solid).add(INSET);
    };

    const spanW = w + 2 * (RIM + SKIRT), spanH = h + 2 * (RIM + SKIRT);
    const p = uv().sub(0.5).mul(vec2(spanW, spanH)), q = vec2(p.x.add(w / 2), float(h / 2).sub(p.y));
    const cell = floor(q), local = fract(q).sub(0.5), centre = vec2(w / 2, h / 2), half = vec2(w / 2, h / 2);
    const shift = vec2(SHADOW[0], SHADOW[1]);

    const border = min(float(0.5).sub(abs(local.x)), float(0.5).sub(abs(local.y)));
    const here = sample(cell).g, across = max(sample(cell.sub(vec2(1, 0))).g, sample(cell.add(vec2(1, 0))).g), along = max(sample(cell.sub(vec2(0, 1))).g, sample(cell.add(vec2(0, 1))).g);
    const rail = here.greaterThan(0.5).and(max(across, along).greaterThan(0.5));
    const offset = select(across.greaterThan(0.5), abs(local.y), abs(local.x));
    const groove = select(rail, coverage(offset.sub(0.05)), float(0)), lip = select(rail, coverage(abs(offset.sub(0.064)).sub(0.01)), float(0));
    const box = sdBox(q.sub(this.startCell).sub(0.5), vec2(0.37), 0.07);

    const panel = fract(sin(cell.x.mul(12.9898).add(cell.y.mul(78.233))).mul(43758.5453)).sub(0.5).mul(0.08).add(1);
    let albedo: N<'vec3'> = rgb(FLOOR).mul(panel).mul(fractal(q.mul(2.6), 4).mul(0.04).add(1)).mul(fractal(q.mul(0.5).add(7.3), 3).mul(0.03).add(1));
    albedo = mix(albedo, rgb(LINE), coverage(border.sub(0.011)).mul(0.6));
    albedo = mix(albedo, rgb(0x161a17), groove);
    albedo = mix(albedo, rgb(0xb4bdb6), lip.mul(0.85));
    const wallShadow = smoothstep(-0.06, 0.16, walls(q.sub(this.slide).sub(shift))).oneMinus();
    const rimShadow = smoothstep(-0.12, 0.05, sdBox(q.sub(shift).sub(centre), half));
    const floorColor = light.lit(albedo.mul(max(wallShadow, rimShadow).mul(-0.55).add(1)));
    const trail = sample(cell).b.mul(smoothstep(0.04, 0.26, length(local)).oneMinus());

    const woodAt = (at: N<'vec2'>, streak: N<'vec2'>) => rgb(WOOD).mul(fractal(at.mul(streak), 3).mul(0.08).add(1));
    const bevel = (d: N<'float'>) => smoothstep(-0.05, -0.012, d).mul(-0.42).add(1).add(smoothstep(-0.075, -0.05, d).mul(smoothstep(-0.05, -0.03, d).oneMinus()).mul(0.12));
    const dWall = walls(q.sub(this.slide)), slid = q.sub(this.slide);
    const wallTop = woodAt(slid, vec2(1.1, 13)).mul(bevel(dWall));
    const dInner = sdBox(q.sub(centre), half), dOuter = sdBox(q.sub(centre), half.add(RIM), 0.1), dRim = max(dOuter, dInner.negate());
    const vertical = abs(q.x.sub(w / 2)).sub(w / 2).greaterThan(abs(q.y.sub(h / 2)).sub(h / 2));
    const rimTop = woodAt(q, select(vertical, vec2(13, 1.1), vec2(1.1, 13))).mul(0.9).mul(bevel(dRim));
    const drop = smoothstep(-0.05, 0.5, sdBox(q.sub(centre).sub(vec2(0.1, 0.16)), half.add(RIM), 0.1)).oneMinus().mul(0.7);

    const board = new Layers();
    board.over(drop, vec3(0));
    board.over(coverage(dInner.sub(0.02)), mix(floorColor.add(rgb(accent).mul(trail.mul(0.22))), rgb(accent).mul(0.7), coverage(abs(box).sub(0.02)).mul(0.85)));
    board.over(coverage(dWall), wallTop);
    board.over(coverage(dRim), rimTop);
    const strip = abs(q.y.add(RIM * 0.3)).sub(0.022), bar = max(strip, abs(q.x.sub(w / 2)).sub(w / 2 - 0.08));
    board.over(coverage(bar).add(exp(max(bar, 0).mul(-38)).mul(0.35)).min(1), rgb(LAMP).mul(1.05));
    this.add(new THREE.Mesh(flat(spanW, spanH), board.material()), 0);

    const r = length(uv().sub(0.5).mul(2)), flare = this.flare;
    const port = new Layers();
    port.over(smoothstep(0.8, 0.98, r).oneMinus().mul(0.5), vec3(0));
    port.over(coverage(r.sub(0.86)), mix(rgb(0x5d4219), rgb(0xe2c273), smoothstep(-0.8, 0.8, uv().y.sub(uv().x))));
    port.over(coverage(r.sub(0.66)), mix(rgb(0x191108), rgb(0x3a2a12), smoothstep(-0.8, 0.8, uv().x.sub(uv().y))));
    port.over(coverage(r.sub(0.24)).add(exp(r.mul(-6)).mul(0.7)).min(1), rgb(GOLD).mul(flare.mul(2.5).add(2.2)));
    const portMesh = this.add(new THREE.Mesh(flat(PORT, PORT), port.material()), 1);
    portMesh.position.copy(this.goalAt);
    const beam = this.add(light.emitter(new THREE.CircleGeometry(0.36 * CELL, 48).rotateX(-Math.PI / 2), rgb(GOLD).mul(flare.mul(0.8).add(1)).mul(PORT_LIGHT)), 0);
    beam.position.copy(this.goalAt);
    const lamp = this.add(light.emitter(flat(w + 2 * RIM, RIM), rgb(LAMP).mul(LAMP_LIGHT)), 0);
    lamp.position.set(0, 0, -(h + RIM) / 2 * CELL);

    const cube = new THREE.PlaneGeometry(CELL, CELL).rotateX(-Math.PI / 2);
    for (let i = 0; i < this.capacity; i++) {
      const block = this.add(light.occluder(cube), 0);
      block.visible = false;
      this.blocks.push(block);
    }

    const s = uv().sub(0.5).mul(vec2(0.7, 0.3)), line = sdSegment(s, vec2(-0.25, 0), vec2(0.25, 0), 0);
    const spark = new Layers();
    spark.over(coverage(line.sub(0.012)).add(exp(line.mul(-24)).mul(0.6)).min(1).mul(this.flash), rgb(SPARK).mul(1.6));
    this.spark = this.add(new THREE.Mesh(flat(0.7, 0.3), spark.material(true)), 3);

    const m = uv().sub(0.5).mul(MOUSE);
    const shadow = new Layers();
    shadow.over(smoothstep(-0.02, 0.09, ellipse(m.sub(vec2(0, -0.05)), 0.16, 0.3)).oneMinus().mul(0.5), vec3(0));
    this.shade = this.add(new THREE.Mesh(flat(MOUSE, MOUSE), shadow.material()), 3.5);

    const fur = smin(ellipse(m.sub(vec2(0, -0.07)), 0.15, 0.225), ellipse(m.sub(vec2(0, 0.15)), 0.1, 0.15), 0.09);
    const back = m.y.add(0.27).negate(), wave = sin(back.mul(15).sub(this.sway)).mul(back.mul(0.16));
    const tail = max(abs(m.x.sub(wave)).sub(float(0.021).sub(back.mul(0.042))), max(back.negate().sub(0.02), back.sub(0.26)));
    const ear = (side: number, radius: number, out: number) => sdCircle(m.sub(vec2(side * out, 0.13)), radius);
    const mouse = new Layers();
    mouse.over(coverage(tail), rgb(PINK).mul(0.9));
    mouse.over(coverage(min(ear(-1, 0.062, 0.1), ear(1, 0.062, 0.1))), rgb(0xe3cbc2));
    mouse.over(coverage(min(ear(-1, 0.037, 0.108), ear(1, 0.037, 0.108))), rgb(PINK));
    mouse.over(coverage(fur), rgb(FUR).mul(smoothstep(-0.1, 0, fur).mul(-0.3).add(1)).add(smoothstep(0, 0.06, abs(m.x)).oneMinus().mul(0.05)));
    mouse.over(coverage(min(sdCircle(m.sub(vec2(-0.05, 0.215)), 0.017), sdCircle(m.sub(vec2(0.05, 0.215)), 0.017))), rgb(0x1b1418));
    mouse.over(coverage(sdCircle(m.sub(vec2(0, 0.297)), 0.02)), rgb(0xef8898));
    this.mouse = this.add(new THREE.Mesh(flat(MOUSE, MOUSE), mouse.material()), 4);

    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 2.5 : 5; });
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
    this.blocks = [];
  }

  protected tick(frame: FrameUpdate, { ctx, pool }: Stage2D): void {
    const dt = frame.dtReal, step = frame.state.step;
    this.clock += dt;
    pool.update(dt, this.clock);
    this.log = frame.eventLog;
    if (!this.start) {
      const initial = frame.frameAt(0);
      this.start = { x: initial.words[0]!, y: initial.words[1]! };
      this.startCell.value.set(this.start.x, this.start.y);
      this.place(this.start.x, this.start.y, this.startAt);
    }
    this.syncPhase(frame, step, pool);
    if (step !== this.step || frame.scrubbed) this.enter(frame, step, pool, this.accentOf(ctx));
    this.animate(frame, step);
  }

  private syncPhase(frame: FrameUpdate, step: number, pool: Stage2D['pool']): void {
    const phase = this.phaseAt(step);
    if (phase !== this.phase) {
      const shift = this.phase >= 0 && !frame.scrubbed ? this.shifts[this.phase]![phase] : null;
      this.slideFrom = shift ? this.phase : -1;
      this.slideShift = shift ?? [0, 0];
      this.slideAt = step - step % this.period;
      this.phase = phase;
    }
    const t = this.slideFrom >= 0 ? clamp01((frame.position - this.slideAt) / SLIDE_STEPS) : 1;
    if (this.slideFrom >= 0 && t >= 1) {
      this.land(this.phases[this.slideFrom]!, this.phases[this.phase]!, pool);
      this.slideFrom = -1;
    }
    const showing = this.slideFrom >= 0 ? this.slideFrom : this.phase, k = this.slideFrom >= 0 ? ease(t) : 0;
    this.slide.value.set(this.slideShift[0] * k, this.slideShift[1] * k);
    if (showing !== this.shown) this.paintWalls(showing);
    const walls = this.phases[showing]!.walls;
    let used = 0;
    for (let i = 0; i < walls.length; i++) {
      if (!walls[i]) continue;
      const block = this.blocks[used++]!;
      this.place(i % this.grid.w + this.slide.value.x, Math.floor(i / this.grid.w) + this.slide.value.y, block.position);
      block.visible = true;
    }
    for (let i = used; i < this.blocks.length; i++) this.blocks[i]!.visible = false;
  }

  private land(before: Mask, after: Mask, pool: Stage2D['pool']): void {
    for (let i = 0; i < after.walls.length; i++) {
      if (!after.walls[i] === !before.walls[i]) continue;
      this.place(i % this.grid.w, Math.floor(i / this.grid.w), this.scratch);
      if (after.walls[i]) {
        pool.burst(this.scratch, DUST, 0.8, this.clock);
        pool.ripple(this.scratch, DUST, 0.35, this.clock);
      } else pool.ripple(this.scratch, LAMP, 0.5, this.clock);
    }
    pool.pulse.fire(0.5);
  }

  private paintWalls(phase: number): void {
    const mask = this.mask;
    if (!mask) return;
    const data = mask.image.data as Uint8Array, walls = this.phases[phase]!.walls;
    for (let i = 0; i < walls.length; i++) data[this.texel(i % this.grid.w, Math.floor(i / this.grid.w))] = walls[i] ? 255 : 0;
    mask.needsUpdate = true;
    this.shown = phase;
  }

  private paintTrail(frame: FrameUpdate, step: number): void {
    const mask = this.mask;
    if (!mask) return;
    const data = mask.image.data as Uint8Array, { w, h } = this.grid;
    for (let i = 0; i < w * h; i++) data[this.texel(i % w, Math.floor(i / w)) + 2] = 0;
    for (let n = 0; n < TRAIL && step - n >= 0; n++) {
      const f = frame.frameAt(step - n);
      if (f.step !== step - n) break;
      const at = this.texel(f.words[0]!, f.words[1]!) + 2, value = Math.round(255 * (1 - n / TRAIL));
      if (value > data[at]!) data[at] = value;
    }
    mask.needsUpdate = true;
  }

  private learn(before: EnvFrame, now: EnvFrame): void {
    const action = Math.trunc(now.actions[0]!) & 3;
    if (this.delta[action]) return;
    const x = now.done ? this.goal.x : now.words[0]!, y = now.done ? this.goal.y : now.words[1]!;
    const dx = x - before.words[0]!, dy = y - before.words[1]!;
    if (Math.abs(dx) + Math.abs(dy) === 1) this.delta[action] = [dx, dy];
  }

  private recall(frame: FrameUpdate, step: number): void {
    for (let s = step; s > Math.max(0, step - SCAN) && this.delta.some((d) => !d); s--) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1) return;
      this.learn(before, now);
    }
  }

  private enter(frame: FrameUpdate, step: number, pool: Stage2D['pool'], accent: number): void {
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null, known = before !== null && before.step === step - 1;
    const x = now.words[0]!, y = now.words[1]!, bx = known ? before.words[0]! : x, by = known ? before.words[1]! : y;
    if (frame.scrubbed) this.recall(frame, step);
    else if (known) this.learn(before, now);
    this.step = step;
    this.action = Math.trunc(now.actions[0]!) & 3;
    this.outcome = !known ? 'rest' : now.done ? 'goal' : bx === x && by === y ? 'bump' : 'move';
    this.kind = frame.scrubbed ? 'rest' : this.outcome;
    this.ledger.x = x;
    this.ledger.y = y;
    this.ledger.step = step;
    this.ledger.reward = frame.state.reward;
    this.place(x, y, this.to);
    this.place(bx, by, this.from);
    this.paintTrail(frame, step);
    const d = this.delta[this.action];
    this.aim = this.outcome === 'goal' ? angleOf(this.goal.x - bx, this.goal.y - by) : this.outcome === 'move' ? angleOf(x - bx, y - by) : step > 0 && d ? angleOf(d[0], d[1]) : this.aim;
    this.reach = 0;
    if (this.kind === 'bump' && d) {
      const tx = x + d[0], ty = y + d[1], inside = tx >= 0 && ty >= 0 && tx < this.grid.w && ty < this.grid.h;
      this.reach = inside ? 0.5 + INSET : 0.5;
      this.spark!.position.set(this.to.x + d[0] * this.reach * CELL, 0, this.to.z + d[1] * this.reach * CELL);
      this.spark!.rotation.y = this.aim;
      this.flashAt = this.clock;
    }
    for (const event of frame.events) {
      if (event.type === 'goal') {
        this.flareAt = this.clock;
        if (this.clock - this.cheerAt < CHEER_SECONDS) continue;
        this.cheerAt = this.clock;
        pool.burst(this.goalAt, GOLD, 0.6, this.clock);
        pool.ripple(this.goalAt, GOLD, 0.5, this.clock);
        pool.ripple(this.startAt, accent, 0.35, this.clock);
      }
    }
  }

  private animate(frame: FrameUpdate, step: number): void {
    const mouse = this.mouse, shade = this.shade;
    if (!mouse || !shade) return;
    const u = this.kind === 'rest' ? 1 : clamp01(frame.position - step);
    let scale = 1, nudge = 0;
    if (this.kind === 'move') this.at.lerpVectors(this.from, this.to, ease(u));
    else if (this.kind === 'bump') {
      this.at.copy(this.to);
      nudge = Math.max(0, this.reach - NOSE) * Math.sin(Math.PI * u);
    } else if (this.kind === 'goal' && u < 0.5) {
      this.at.lerpVectors(this.from, this.goalAt, ease(u * 2));
      scale = 1 - 0.6 * ease(u * 2) ** 2;
    } else if (this.kind === 'goal') {
      this.at.copy(this.to);
      scale = pop(clamp01((u - 0.5) * 2.5));
    } else this.at.copy(this.to);
    const turn = Math.atan2(Math.sin(this.aim - this.heading), Math.cos(this.aim - this.heading));
    this.heading += turn * (1 - Math.exp(-TURN * frame.dtReal));
    if (frame.scrubbed) this.heading = this.aim;
    mouse.position.set(this.at.x - Math.sin(this.aim) * nudge * CELL, 0, this.at.z - Math.cos(this.aim) * nudge * CELL);
    mouse.rotation.y = this.heading;
    mouse.scale.setScalar(Math.max(0.001, scale));
    shade.position.set(mouse.position.x + 0.05 * CELL, 0, mouse.position.z + 0.08 * CELL);
    shade.rotation.y = this.heading;
    shade.scale.copy(mouse.scale);
    this.sway.value += frame.dtReal * (this.kind === 'move' || this.kind === 'goal' ? 11 : 3.5);
    this.flash.value = Math.exp(-Math.max(0, this.clock - this.flashAt) / FLASH_SECONDS) * (this.clock - this.flashAt < 4 * FLASH_SECONDS ? 1 : 0);
    this.flare.value = Math.exp(-Math.max(0, this.clock - this.flareAt) / FLARE_SECONDS);
  }

  private actionLabel(): string {
    if (this.ledger.step === 0) return '—';
    const d = this.delta[this.action], name = d ? COMPASS[`${d[0]},${d[1]}`]! : `action ${this.action}`;
    return this.outcome === 'bump' ? `${name} · blocked` : this.outcome === 'goal' ? `${name} · goal` : name;
  }

  private opening(): string {
    const now = this.openings[this.phaseAt(this.ledger.step)], next = this.openings[(this.phaseAt(this.ledger.step) + 1) % this.phases.length];
    if (!now?.length) return 'none';
    return next && next.join() !== now.join() ? `column ${now.join(', ')} → ${next.join(', ')}` : `column ${now.join(', ')}`;
  }

  private sinceGoal(): string {
    const last = this.log ? lastEvent(this.log, 'goal', this.ledger.step) : undefined;
    const since = last === undefined ? 0 : this.ledger.step - last;
    return last === undefined ? 'none yet' : since === 0 ? 'now' : `${formatCount(since)} ${since === 1 ? 'step' : 'steps'} ago`;
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit, accent: number): void {
    const { w } = this.grid, step = frame.state.step, walls = this.phases[this.phaseAt(step)]!.walls, boxes = this.overlay?.cellBoxes();
    if (boxes) {
      boxes.begin();
      for (let i = 0; i < walls.length; i++) {
        if (!walls[i]) continue;
        this.place(i % w, Math.floor(i / w), this.scratch);
        boxes.box(this.scratch.x, this.scratch.z, 0, 0, 0, 0, WIRE.r, WIRE.g, WIRE.b);
      }
      boxes.commit();
    }
    this.outline(kit.curve('goal', 5, GOLD), this.goal, 0.42, true);
    if (this.start) this.outline(kit.curve('start', 5, accent), this.start, 0.36, false);
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null;
    const moved = kit.curve('moved', 5, TOKENS.uiCyan), blocked = kit.curve('blocked', 5, TOKENS.failureBlue), scored = kit.curve('scored', 5, GOLD);
    if (!before || before.step !== step - 1) return;
    const d = this.delta[Math.trunc(now.actions[0]!) & 3], bx = before.words[0]!, by = before.words[1]!;
    if (!d) return;
    const curve = now.done ? scored : bx === now.words[0] && by === now.words[1] ? blocked : moved;
    this.place(bx, by, this.scratch);
    const cx = this.scratch.x, cz = this.scratch.z, tip = 0.62 * CELL, back = 0.44 * CELL, side = 0.14 * CELL;
    curve.point(cx + d[0] * 0.08 * CELL, 0.1, cz + d[1] * 0.08 * CELL);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back - d[1] * side, 0.1, cz + d[1] * back + d[0] * side);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back + d[1] * side, 0.1, cz + d[1] * back - d[0] * side);
  }

  private outline(curve: DebugCurve, cell: GridCell, r: number, diamond: boolean): void {
    this.place(cell.x, cell.y, this.scratch);
    const corners = diamond ? [[0, -r], [r, 0], [0, r], [-r, 0], [0, -r]] : [[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]];
    for (const [x, z] of corners) curve.point(this.scratch.x + x! * CELL, 0.1, this.scratch.z + z! * CELL);
  }
}

export default (() => new BlockingMazeScene()) satisfies SceneFactory;
