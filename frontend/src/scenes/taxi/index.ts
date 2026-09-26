import * as THREE from 'three/webgpu';
import { abs, float, floor, fract, max, min, mix, smoothstep, step as stepOf, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { CELL, type GridSize } from '@kit/grid';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { TAXI_X, TAXI_Y } from '@sim/envs/taxi';
import type { EnvFrame } from '@sim/types';
import { constant, constantList } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Seg = readonly [number, number, number, number];
type Kind = 'rest' | 'move' | 'bump' | 'pickup' | 'refused' | 'deliver';

const RIM = 0.34;
const SKIRT = 0.7;
const WALL_R = 0.065;
const CAB = 1.6;
const CAB_ZOOM = 1.25;
const FIGURE = 0.5;
const FIGURE_ZOOM = 1.6;
const CORNER = 0.24;
const SCAN = 8192;
const TURN = 14;
const HAZARD_SECONDS = 0.9;
const CHEER_SECONDS = 0.6;
const DEPOT_LIGHT = 2.6;
const BENCH = 0x0a0c12;
const ASPHALT = 0x3a3e48;
const SEAM = 0x4c515d;
const LANE = 0xd8cfae;
const KERB = 0x6a6966;
const KERB_EDGE = 0x9a968d;
const CONCRETE = 0xbdb6a8;
const STRIPE = 0xe3a23b;
const STRIPE_DARK = 0x2a2724;
const CAB_PAINT = 0xf0b43a;
const GLASS = 0x1b2430;
const LAMP = 0xfff0cf;
const TAIL = 0xd8323a;
const HAZARD = 0xff9a2a;
const SIGN_OFF = 0x5a5650;
const SKIN = 0x3a2a24;
const GOLD = TOKENS.celebrationGold;
const WIRE = 0x8fa6c4;
const DEPOTS = [
  { name: 'coral', color: 0xe8606a },
  { name: 'mint', color: 0x5fcf9a },
  { name: 'violet', color: 0xb58cf0 },
  { name: 'azure', color: 0x4fa8e8 },
] as const;
const COMPASS: Record<string, string> = { '0,-1': '↑ north', '0,1': '↓ south', '-1,0': '← west', '1,0': 'east →' };

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width * CELL, depth * CELL).rotateX(-Math.PI / 2);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const ease = (u: number) => u * u * (3 - 2 * u);
const pop = (u: number) => 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2;
const angleOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
const ellipse = (p: N<'vec2'>, rx: number, ry: number) => p.div(vec2(rx, ry)).length().sub(1).mul(Math.min(rx, ry));
const depotOf = (i: number) => DEPOTS[i % DEPOTS.length]!;

function glyph(k: number, r: N<'vec2'>): N<'float'> {
  const t = vec2(r.x, r.y.negate().add(0.02));
  if (k % 4 === 0) return sdCircle(r, 0.085);
  if (k % 4 === 1) return max(abs(t.x).mul(0.866).add(t.y.mul(0.5)), t.y.negate()).sub(0.05);
  if (k % 4 === 2) return sdBox(r, vec2(0.075), 0.012);
  return abs(r.x).add(abs(r.y)).mul(0.7071).sub(0.08);
}

class Layers {
  pre: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): void {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
  }

  material(fade: N<'float'> | null = null, additive = false): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.pre.div(max(this.alpha, 1e-4));
    material.opacityNode = fade ? this.alpha.mul(fade) : this.alpha;
    if (additive) material.blending = THREE.AdditiveBlending;
    return material;
  }
}

class Figure {
  readonly coat = uniform(new THREE.Color());
  readonly fade = uniform(1);
  mesh: THREE.Mesh | null = null;

  build(): THREE.MeshBasicNodeMaterial {
    const n = uv().sub(0.5).mul(FIGURE / FIGURE_ZOOM), coat = rgb(0xffffff).mul(this.coat), halo = abs(n.length().sub(0.12)).sub(0.007);
    const body = ellipse(n.sub(vec2(0, -0.01)), 0.095, 0.058);
    const arm = sdSegment(n, vec2(0.07, 0.0), vec2(0.12, 0.11), 0.022);
    const layers = new Layers();
    layers.over(smoothstep(0.0, 0.022, n.length().sub(0.125)).oneMinus().mul(0.35), vec3(0));
    layers.over(coverage(halo).mul(0.85), coat);
    layers.over(smoothstep(-0.02, 0.07, ellipse(n.sub(vec2(0.025, -0.035)), 0.11, 0.075)).oneMinus().mul(0.45), vec3(0));
    layers.over(coverage(arm), coat.mul(0.82));
    layers.over(coverage(sdCircle(n.sub(vec2(0.12, 0.11)), 0.024)), rgb(0xe6c3a4));
    layers.over(coverage(body), coat.mul(smoothstep(-0.05, 0, body).mul(-0.28).add(1)));
    layers.over(coverage(sdCircle(n.sub(vec2(0, -0.005)), 0.043)), rgb(SKIN));
    layers.over(coverage(sdCircle(n.sub(vec2(-0.012, 0.008)), 0.014)).mul(0.35), rgb(0xffffff));
    return layers.material(this.fade);
  }
}

class TaxiScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'taxi';

  private grid: GridSize = { w: 1, h: 1 };
  private depots: { x: number; y: number }[] = [];
  private segs: Seg[] = [];
  private moves = 4;
  private readonly delta: ([number, number] | null)[] = [];

  private readonly glow: (N<'float'> & { value: number })[] = [];
  private readonly sign = uniform(new THREE.Color(SIGN_OFF));
  private readonly hazard = uniform(0);
  private readonly rider = new Figure();
  private readonly alighted = new Figure();
  private cab: THREE.Mesh | null = null;
  private beams: THREE.Mesh | null = null;
  private readonly owned: THREE.Mesh[] = [];

  private clock = 0;
  private step = -1;
  private kind: Kind = 'rest';
  private outcome: Kind = 'rest';
  private action = 0;
  private heading = 0;
  private aim = 0;
  private hazardAt = -Infinity;
  private cheerAt = -Infinity;
  private before: number[] = [0, 0, 0, 0];
  private now: number[] = [0, 0, 0, 0];
  private log: FrameUpdate['eventLog'] | null = null;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly truth = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly ledger = { reward: 0, step: 0 };

  override legend: LegendSpec = {
    title: 'TAXI',
    rows: [
      { key: 'cab', label: 'cab', tint: CAB_PAINT, value: () => `column ${this.now[0]! + 1} · row ${this.now[1]! + 1}` },
      { key: 'passenger', label: 'passenger', tint: 0xe6c3a4, value: () => (this.riding() ? 'riding' : `waiting at ${depotOf(this.now[2]!).name}`) },
      { key: 'destination', label: 'destination', tint: depotOf(0).color, value: () => depotOf(this.now[3]!).name },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionLabel() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'fares', label: 'deliveries', tint: GOLD, value: () => formatCount(this.log ? countEvents(this.log, 'goal', -1, this.ledger.step) : 0) },
    ],
    note: () => this.sinceDelivery(),
  };

  private riding(): boolean {
    return this.now[2]! >= this.depots.length;
  }

  private toWorld(qx: number, qy: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set((qx - this.grid.w / 2) * CELL, 0, (qy - this.grid.h / 2) * CELL);
  }

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return this.toWorld(x + 0.5, y + 0.5, out);
  }

  private corner(i: number, out: THREE.Vector3): THREE.Vector3 {
    const d = this.depots[i]!, ox = d.x < this.grid.w / 2 ? -1 : 1, oy = d.y < this.grid.h / 2 ? -1 : 1;
    return this.toWorld(d.x + 0.5 + ox * CORNER, d.y + 0.5 + oy * CORNER, out);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, side = constant(env, 'TAXI_SIDE'), count = constant(env, 'TAXI_DEPOTS');
    const xs = constantList(env, 'TAXI_DEPOT_X'), ys = constantList(env, 'TAXI_DEPOT_Y');
    if (xs.length < count || ys.length < count || !env.fences) throw new Error(`${env.envId} lacks its depots or fences`);
    this.grid = { w: side, h: side };
    this.depots = Array.from({ length: count }, (_, i) => ({ x: xs[i]!, y: ys[i]! }));
    this.moves = env.actions - 2;
    for (let a = 0; a < this.moves; a++) this.delta.push(null);
    const seen = new Set<string>();
    for (const [x, y, nx, ny] of env.fences) {
      const seg: Seg = nx !== x ? [Math.max(x, nx), y, Math.max(x, nx), y + 1] : [x, Math.max(y, ny), x + 1, Math.max(y, ny)];
      if (seen.has(seg.join())) continue;
      seen.add(seg.join());
      this.segs.push(seg);
    }
    const halfW = side / 2 + RIM, halfH = side / 2 + RIM;
    return new THREE.Box2(new THREE.Vector2(-halfW - 0.3, -halfH - 0.3).multiplyScalar(CELL), new THREE.Vector2(halfW + 0.3, halfH + 1.2).multiplyScalar(CELL));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa6b4d2).multiplyScalar(0.52);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const { w, h } = this.grid, accent = this.accentOf(ctx);
    return {
      cellCapacity: 1,
      lattice: { w, h, origin: [-w / 2 * CELL, 0.02, -h / 2 * CELL], u: [CELL, 0, 0], v: [0, 0, CELL] },
      frame: (state) => ({ originX: 0, originY: 0, originZ: 0, agent: this.place(state.pose[TAXI_X]!, state.pose[TAXI_Y]!, this.truth), cone: this.noCone, coneCount: 0 }),
      debug: (frame, kit) => this.groundTruth(frame, kit, accent),
    };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    const { w, h } = this.grid;
    this.scene.background = new THREE.Color(BENCH);
    for (let i = 0; i < this.depots.length; i++) this.glow.push(uniform(0.15));

    const walls = (q: N<'vec2'>): N<'float'> => {
      let d: N<'float'> = float(9);
      for (const [ax, ay, bx, by] of this.segs) d = min(d, sdSegment(q, vec2(ax, ay), vec2(bx, by), 0));
      return d.sub(WALL_R);
    };

    const spanW = w + 2 * (RIM + SKIRT), spanH = h + 2 * (RIM + SKIRT);
    const p = uv().sub(0.5).mul(vec2(spanW, spanH)), q = vec2(p.x.add(w / 2), float(h / 2).sub(p.y));
    const local = fract(q).sub(0.5), centre = vec2(w / 2, h / 2), half = vec2(w / 2, h / 2);
    const border = min(float(0.5).sub(abs(local.x)), float(0.5).sub(abs(local.y)));

    let albedo: N<'vec3'> = rgb(ASPHALT).mul(fractal(q.mul(3.1), 4).mul(0.12).add(1)).mul(fractal(q.mul(0.6).add(3.7), 3).mul(0.06).add(1));
    albedo = mix(albedo, rgb(SEAM), coverage(border.sub(0.008)).mul(0.5));
    const dashX = max(abs(local.y).sub(0.013), abs(fract(q.x.mul(3)).sub(0.5)).sub(0.22));
    const dashY = max(abs(local.x).sub(0.013), abs(fract(q.y.mul(3)).sub(0.5)).sub(0.22));
    albedo = mix(albedo, rgb(LANE), coverage(min(dashX, dashY)).mul(0.42));
    let glowAdd: N<'vec3'> = vec3(0);
    this.depots.forEach((d, i) => {
      const color = rgb(depotOf(i).color), r = q.sub(vec2(d.x + 0.5, d.y + 0.5)), bay = sdBox(r, vec2(0.41), 0.09);
      albedo = mix(albedo, color, coverage(bay).mul(0.16));
      albedo = mix(albedo, color.mul(0.95), coverage(abs(bay).sub(0.02)));
      albedo = mix(albedo, color, coverage(glyph(i, r)));
      glowAdd = glowAdd.add(color.mul(coverage(bay).mul(0.22).add(coverage(abs(bay).sub(0.03)).mul(0.55)).mul(this.glow[i]!)));
    });
    const wallShadow = smoothstep(-0.04, 0.14, walls(q.sub(vec2(0.05, 0.08)))).oneMinus();
    const rimShadow = smoothstep(-0.12, 0.05, sdBox(q.sub(vec2(0.05, 0.08)).sub(centre), half));
    const floorColor = light.lit(albedo.mul(max(wallShadow, rimShadow).mul(-0.5).add(1))).add(glowAdd);

    const dWall = walls(q);
    const stripes = mix(rgb(STRIPE_DARK), rgb(STRIPE), stepOf(0.5, fract(q.x.add(q.y).mul(4))));
    const wallTop = mix(stripes, rgb(CONCRETE), smoothstep(-0.034, -0.022, dWall)).mul(smoothstep(-0.02, 0, dWall).mul(-0.3).add(1));
    const dInner = sdBox(q.sub(centre), half), dOuter = sdBox(q.sub(centre), half.add(RIM), 0.12), dRim = max(dOuter, dInner.negate());
    const slab = min(abs(fract(q.x.mul(2)).sub(0.5)), abs(fract(q.y.mul(2)).sub(0.5)));
    const rimTop = mix(rgb(KERB).mul(fractal(q.mul(4), 3).mul(0.08).add(1)), rgb(KERB).mul(0.7), coverage(slab.sub(0.012)).mul(0.5));
    const rimColor = mix(rimTop, rgb(KERB_EDGE), coverage(abs(dInner).sub(0.028))).mul(smoothstep(-0.04, 0, dOuter).mul(-0.35).add(1));
    const drop = smoothstep(-0.05, 0.5, sdBox(q.sub(centre).sub(vec2(0.1, 0.16)), half.add(RIM), 0.12)).oneMinus().mul(0.7);

    const board = new Layers();
    board.over(drop, vec3(0));
    board.over(coverage(dInner.sub(0.02)), floorColor);
    board.over(coverage(dWall), wallTop);
    board.over(coverage(dRim), rimColor);
    this.add(new THREE.Mesh(flat(spanW, spanH), board.material()), 0);

    this.depots.forEach((d, i) => {
      const emitter = this.add(light.emitter(new THREE.CircleGeometry(0.45 * CELL, 40).rotateX(-Math.PI / 2), rgb(depotOf(i).color).mul(this.glow[i]!.mul(DEPOT_LIGHT))), 0);
      this.place(d.x, d.y, emitter.position);
    });
    for (const [ax, ay, bx, by] of this.segs) {
      const vertical = ax === bx, len = vertical ? by - ay : bx - ax;
      const block = this.add(light.occluder(vertical ? flat(2 * WALL_R, len) : flat(len, 2 * WALL_R)), 0);
      this.toWorld((ax + bx) / 2, (ay + by) / 2, block.position);
    }

    const m = uv().sub(0.5).mul(CAB / CAB_ZOOM);
    const beam = new Layers();
    let cone: N<'float'> = float(0);
    for (const s of [-1, 1]) {
      const b = m.sub(vec2(s * 0.1, 0.29)), along = b.y, width = abs(b.x).sub(along.mul(0.38)).sub(0.025);
      cone = max(cone, smoothstep(0.03, -0.05, width).mul(smoothstep(0, 0.04, along)).mul(smoothstep(0.34, 0.02, along)));
    }
    beam.over(cone.mul(0.22), rgb(LAMP));
    this.beams = this.add(new THREE.Mesh(flat(CAB, CAB), beam.material(null, true)), 3);

    const body = sdBox(m, vec2(0.17, 0.3), 0.09);
    const cab = new Layers();
    cab.over(smoothstep(-0.03, 0.1, sdBox(m, vec2(0.19, 0.32), 0.1)).oneMinus().mul(0.5), vec3(0));
    cab.over(coverage(min(ellipse(m.sub(vec2(-0.19, 0.07)), 0.03, 0.018), ellipse(m.sub(vec2(0.19, 0.07)), 0.03, 0.018))), rgb(0x2a2622));
    cab.over(coverage(body), rgb(CAB_PAINT).mul(smoothstep(-0.07, 0, body).mul(-0.3).add(1)).mul(abs(m.x).mul(-0.6).add(1.04)));
    const checks = stepOf(0.5, fract(m.y.mul(16))), band = max(abs(abs(m.x).sub(0.148)).sub(0.012), abs(m.y).sub(0.2));
    cab.over(coverage(band).mul(checks), rgb(0x1c1a18));
    cab.over(coverage(sdBox(m.sub(vec2(0, 0.115)), vec2(0.135, 0.052), 0.03)), rgb(GLASS).add(rgb(0x6f8aa8).mul(smoothstep(0.03, -0.03, abs(m.x.add(m.y).sub(0.07)).sub(0.02)).mul(0.35))));
    cab.over(coverage(sdBox(m.sub(vec2(0, -0.195)), vec2(0.125, 0.034), 0.02)), rgb(GLASS));
    cab.over(coverage(sdBox(m.sub(vec2(0, -0.04)), vec2(0.13, 0.1), 0.04)), rgb(CAB_PAINT).mul(1.06));
    cab.over(coverage(sdBox(m.sub(vec2(0, -0.04)), vec2(0.085, 0.034), 0.014)), rgb(0xffffff).mul(this.sign));
    cab.over(coverage(min(sdCircle(m.sub(vec2(-0.105, 0.29)), 0.028), sdCircle(m.sub(vec2(0.105, 0.29)), 0.028))), rgb(LAMP));
    cab.over(coverage(min(sdBox(m.sub(vec2(-0.115, -0.295)), vec2(0.03, 0.01)), sdBox(m.sub(vec2(0.115, -0.295)), vec2(0.03, 0.01)))), rgb(TAIL));
    let flasher: N<'float'> = float(9);
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) flasher = min(flasher, sdCircle(m.sub(vec2(sx * 0.165, sy * 0.265)), 0.02));
    cab.over(coverage(flasher).mul(this.hazard), rgb(HAZARD));
    cab.over(smoothstep(0.0, 0.07, flasher).oneMinus().mul(this.hazard).mul(0.55), rgb(HAZARD));
    this.cab = this.add(new THREE.Mesh(flat(CAB, CAB), cab.material()), 4);

    this.rider.mesh = this.add(new THREE.Mesh(flat(FIGURE, FIGURE), this.rider.build()), 3.5);
    this.alighted.mesh = this.add(new THREE.Mesh(flat(FIGURE, FIGURE), this.alighted.build()), 3.6);
    this.alighted.mesh.visible = false;

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
    for (const mesh of this.owned) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.owned.length = 0;
  }

  protected tick(frame: FrameUpdate, { ctx, pool }: Stage2D): void {
    const dt = frame.dtReal, step = frame.state.step;
    this.clock += dt;
    pool.update(dt, this.clock);
    this.log = frame.eventLog;
    if (step !== this.step || frame.scrubbed) this.enter(frame, step, pool, this.accentOf(ctx));
    this.animate(frame, step);
  }

  private learn(before: EnvFrame, now: EnvFrame): void {
    const action = Math.trunc(now.actions[0]!);
    if (action < 0 || action >= this.moves || this.delta[action] || now.done) return;
    const dx = now.words[0]! - before.words[0]!, dy = now.words[1]! - before.words[1]!;
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
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null, known = before !== null && before.step === step - 1 && now.step === step;
    for (let i = 0; i < 4; i++) {
      this.now[i] = now.words[i]!;
      this.before[i] = known ? before.words[i]! : now.words[i]!;
    }
    if (frame.scrubbed) this.recall(frame, step);
    else if (known) this.learn(before, now);
    const [x, y, pas, dest] = this.now as [number, number, number, number];
    const [bx, by, bp, bd] = this.before as [number, number, number, number];
    const count = this.depots.length, gap = Math.abs(x - bx) + Math.abs(y - by);
    this.step = step;
    this.action = Math.trunc(now.actions[0]!);
    this.outcome = !known || gap > 1 ? 'rest' : now.done ? 'deliver' : bp < count && pas >= count ? 'pickup' : now.reward < -1 ? 'refused' : gap === 1 ? 'move' : this.action < this.moves ? 'bump' : 'rest';
    this.kind = frame.scrubbed ? 'rest' : this.outcome;
    this.ledger.reward = now.reward;
    this.ledger.step = step;
    this.place(x, y, this.to);
    this.place(this.kind === 'rest' ? x : bx, this.kind === 'rest' ? y : by, this.from);
    const d = this.delta[this.action];
    this.aim = this.outcome === 'move' ? angleOf(x - bx, y - by) : this.outcome === 'bump' && d ? angleOf(d[0], d[1]) : this.aim;
    this.rider.coat.value.set(depotOf(dest).color);
    this.alighted.coat.value.set(depotOf(bd).color);
    for (const event of frame.events) {
      if (event.type === 'goal' && this.clock - this.cheerAt >= CHEER_SECONDS) {
        this.cheerAt = this.clock;
        this.place(x, y, this.scratch);
        pool.ripple(this.scratch, depotOf(bd).color, 0.55, this.clock);
        this.scratch.y = 40;
        pool.burst(this.scratch, GOLD, 0.8, this.clock);
        if (pas < count) pool.ripple(this.corner(pas, this.other), accent, 0.3, this.clock);
        pool.pulse.fire(0.6);
      } else if (event.type === 'key' && event.index < count) {
        pool.ripple(this.corner(event.index, this.scratch), depotOf(dest).color, 0.35, this.clock);
      } else if (event.type === 'harm') {
        this.hazardAt = this.clock;
        pool.ripple(this.to, TOKENS.failureBlue, 0.3, this.clock);
      }
    }
    if (!known || frame.scrubbed) this.hazardAt = -Infinity;
  }

  private animate(frame: FrameUpdate, step: number): void {
    const cab = this.cab, beams = this.beams, rider = this.rider.mesh, alighted = this.alighted.mesh;
    if (!cab || !beams || !rider || !alighted) return;
    const u = this.kind === 'rest' ? 1 : clamp01(frame.position - step), count = this.depots.length;
    const [, , pas, dest] = this.now as [number, number, number, number];
    const [, , bp, bd] = this.before as [number, number, number, number];
    let nudge = 0;
    if (this.kind === 'move') this.at.lerpVectors(this.from, this.to, ease(u));
    else this.at.copy(this.to);
    if (this.kind === 'bump') nudge = 0.12 * Math.sin(Math.PI * u);
    const turn = Math.atan2(Math.sin(this.aim - this.heading), Math.cos(this.aim - this.heading));
    this.heading += turn * (1 - Math.exp(-TURN * frame.dtReal));
    if (frame.scrubbed || this.kind === 'rest') this.heading = this.aim;
    cab.position.set(this.at.x - Math.sin(this.aim) * nudge * CELL, 0, this.at.z - Math.cos(this.aim) * nudge * CELL);
    cab.rotation.y = this.heading;
    beams.position.copy(cab.position);
    beams.rotation.y = this.heading;

    const riding = pas >= count;
    rider.visible = !riding || this.kind === 'pickup';
    this.rider.fade.value = 1;
    rider.scale.setScalar(1);
    if (this.kind === 'pickup') {
      this.corner(bp, this.scratch);
      rider.position.lerpVectors(this.scratch, cab.position, ease(u));
      rider.scale.setScalar(Math.max(0.001, 1 - 0.7 * ease(u)));
      this.rider.fade.value = 1 - clamp01((u - 0.7) / 0.3);
      rider.rotation.y = angleOf(cab.position.x - this.scratch.x, cab.position.z - this.scratch.z);
    } else if (!riding) {
      this.corner(pas, rider.position);
      this.place(this.depots[pas]!.x, this.depots[pas]!.y, this.scratch);
      rider.rotation.y = angleOf(this.scratch.x - rider.position.x, this.scratch.z - rider.position.z);
      if (this.kind === 'deliver') rider.scale.setScalar(Math.max(0.001, pop(clamp01((u - 0.4) / 0.6))));
    }

    alighted.visible = this.kind === 'deliver' && bd < count;
    if (alighted.visible) {
      this.corner(bd, this.scratch);
      alighted.position.lerpVectors(cab.position, this.scratch, ease(clamp01(u * 1.6)));
      alighted.rotation.y = angleOf(this.scratch.x - cab.position.x, this.scratch.z - cab.position.z);
      alighted.scale.setScalar(0.4 + 0.6 * ease(clamp01(u * 1.6)));
      this.alighted.fade.value = 1 - clamp01((u - 0.65) / 0.35);
    }

    const signOn = riding && (this.kind !== 'pickup' || u > 0.6);
    this.sign.value.set(signOn ? depotOf(dest).color : SIGN_OFF);
    const age = this.clock - this.hazardAt;
    this.hazard.value = age >= 0 && age < HAZARD_SECONDS ? 0.5 + 0.5 * Math.cos(age * Math.PI * 2 * 3.3) : 0;
    const beat = 0.88 + 0.12 * Math.sin(this.clock * 3);
    this.glow.forEach((g, i) => { g.value = i === dest ? (riding ? 1 : 0.55) * beat : 0.16; });
  }

  private actionLabel(): string {
    if (this.ledger.step === 0 || this.outcome === 'rest' && this.action >= this.moves) return '—';
    const d = this.delta[this.action];
    const name = this.action < this.moves ? (d ? COMPASS[`${d[0]},${d[1]}`]! : `move ${this.action}`) : this.action === this.moves ? 'pick up' : 'drop off';
    const tail: Partial<Record<Kind, string>> = { bump: 'blocked', refused: 'refused', pickup: 'aboard', deliver: 'delivered' };
    return tail[this.outcome] ? `${name} · ${tail[this.outcome]}` : name;
  }

  private sinceDelivery(): string {
    const last = this.log ? lastEvent(this.log, 'goal', this.ledger.step) : undefined;
    if (last === undefined) return 'no delivery yet';
    const since = this.ledger.step - last;
    return since === 0 ? 'delivered now' : `last delivery ${formatCount(since)} ${since === 1 ? 'step' : 'steps'} ago`;
  }

  private square(kit: DebugKit, name: string, color: number, cx: number, cz: number, r: number, diamond = false): void {
    const curve = kit.curve(name, 5, color), R = r * CELL;
    const corners = diamond ? [[0, -R], [R, 0], [0, R], [-R, 0], [0, -R]] : [[-R, -R], [R, -R], [R, R], [-R, R], [-R, -R]];
    for (const [x, z] of corners) curve.point(cx + x!, 0.1, cz + z!);
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit, accent: number): void {
    const step = frame.state.step, now = frame.frameAt(step), count = this.depots.length;
    this.segs.forEach(([ax, ay, bx, by], i) => {
      const curve = kit.curve(`fence${i}`, 2, WIRE);
      this.toWorld(ax, ay, this.scratch);
      curve.point(this.scratch.x, 0.1, this.scratch.z);
      this.toWorld(bx, by, this.scratch);
      curve.point(this.scratch.x, 0.1, this.scratch.z);
    });
    this.depots.forEach((d, i) => {
      this.place(d.x, d.y, this.scratch);
      this.square(kit, `depot${i}`, depotOf(i).color, this.scratch.x, this.scratch.z, 0.41);
    });
    if (now.step !== step) return;
    const pas = now.words[2]!, dest = now.words[3]!;
    if (dest < count) {
      this.place(this.depots[dest]!.x, this.depots[dest]!.y, this.scratch);
      this.square(kit, 'destination', GOLD, this.scratch.x, this.scratch.z, 0.3, true);
    }
    if (pas < count) this.corner(pas, this.scratch);
    else this.place(now.words[0]!, now.words[1]!, this.scratch);
    this.square(kit, 'passenger', accent, this.scratch.x, this.scratch.z, pas < count ? 0.1 : 0.34);
    const before = step > 0 ? frame.frameAt(step - 1) : null, d = this.delta[Math.trunc(now.actions[0]!)];
    const moved = kit.curve('moved', 5, TOKENS.uiCyan), blocked = kit.curve('blocked', 5, TOKENS.failureBlue);
    if (!before || before.step !== step - 1 || !d) return;
    const bx = before.words[0]!, by = before.words[1]!, curve = bx === now.words[0] && by === now.words[1] ? blocked : moved;
    this.place(bx, by, this.scratch);
    const cx = this.scratch.x, cz = this.scratch.z, tip = 0.62 * CELL, back = 0.44 * CELL, side = 0.14 * CELL;
    curve.point(cx + d[0] * 0.08 * CELL, 0.1, cz + d[1] * 0.08 * CELL);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back - d[1] * side, 0.1, cz + d[1] * back + d[0] * side);
    curve.point(cx + d[0] * tip, 0.1, cz + d[1] * tip);
    curve.point(cx + d[0] * back + d[1] * side, 0.1, cz + d[1] * back - d[0] * side);
  }
}

export default (() => new TaxiScene()) satisfies SceneFactory;
