import * as THREE from 'three/webgpu';
import { abs, attribute, float, floor, fract, length, max, min, mix, positionWorld, select, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import type { EnvFrame } from '@sim/types';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
type Scalar = THREE.UniformNode<'float', number>;
type Pip = readonly [number, number, number];

const REST = 0.35;
const SINK = 0.8;
const BUMP = 0.55;
const DROP = 0.95;
const BELT = 1.4;
const BELT_STEPS = 12;
const LANDED = 0.62;
const ARRIVE = 0.45;
const RELEASE_FROM = 0.45;
const RELEASE_TO = 0.85;
const WINDOW = 4096;
const LANE_SAMPLES = 16;
const DOOR = 0.24;
const MINT = 0x57eea0;
const EMBER = 0xff8a3d;
const FULL = 0xff4d5e;
const REJECT = 0x6f88c4;
const GOLD = 0xffd166;
const RAMP = [0x8f9bb8, 0x4ea6ff, 0xb98cff, 0xff5c8a].map((hex) => new THREE.Color(hex));

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const smooth = (u: number) => u * u * (3 - 2 * u);
const easeOut = (u: number) => 1 - (1 - u) ** 3;
const easeIn = (u: number) => u * u * u;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const tint = (): Tint => uniform(new THREE.Vector3());
const setTint = (target: Tint, color: THREE.Color, gain: number) => target.value.set(color.r * gain, color.g * gain, color.b * gain);

function hueOf(rank: number, count: number, out: THREE.Color): THREE.Color {
  const x = count > 1 ? rank / (count - 1) * (RAMP.length - 1) : RAMP.length - 1, k = Math.min(RAMP.length - 2, Math.floor(x));
  return out.copy(RAMP[k]!).lerp(RAMP[k + 1]!, x - k);
}

function pipLayout(count: number): Pip[] {
  if (count <= 1) return [[0, 0, 0.27]];
  const ring = count <= 2 ? 0.38 : count <= 4 ? 0.42 : 0.55, start = count === 2 ? 0 : count <= 4 ? Math.PI / 4 : Math.PI / 2;
  const radius = Math.min(0.21, ring * Math.sin(Math.PI / count) * 0.74);
  return Array.from({ length: count }, (_, j): Pip => {
    const a = start + j * 2 * Math.PI / count;
    return [ring * Math.cos(a), ring * Math.sin(a), radius];
  });
}

function sprite(color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  if (additive) material.blending = THREE.AdditiveBlending;
  return material;
}

function tokenMaterial(hue: N<'vec3'>, kind: N<'float'>, fade: N<'float'>, layouts: readonly Pip[][]): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(2), r = length(p);
  const distances = layouts.map((layout) => layout.reduce<N<'float'>>((d, [x, y, radius]) => min(d, sdCircle(p.sub(vec2(x, y)), radius)), float(4)));
  let pips = distances[distances.length - 1]!;
  for (let k = distances.length - 2; k >= 0; k--) pips = select(kind.lessThan(k + 0.5), distances[k]!, pips);
  const face = hue.mul(float(1.3).sub(r.mul(0.45)));
  const rimmed = mix(face, hue.mul(0.3).add(0.62), coverage(abs(r.sub(0.83)).sub(0.045)).mul(0.75));
  return sprite(mix(rimmed, rgb(0x0a0d15), coverage(pips)), coverage(sdCircle(p, 0.95)).mul(fade));
}

interface Token {
  mesh: THREE.Mesh;
  hue: Tint;
  kind: Scalar;
  fade: Scalar;
  glow: Tint;
  emitter: THREE.Mesh;
}

interface Lane {
  points: THREE.Vector2[];
  lengths: number[];
}

class DecisionWindow {
  readonly visits: Float32Array;
  readonly accepts: Float32Array;
  decisions = 0;
  freed = 0;
  exposed = 0;
  private readonly codes = new Int32Array(WINDOW).fill(-2);
  private readonly releases = new Int16Array(WINDOW);
  private readonly busy = new Int16Array(WINDOW);
  private through = -WINDOW;
  private holes = 0;

  constructor(private readonly cols: number, rows: number, private readonly servers: number) {
    this.visits = new Float32Array(cols * rows);
    this.accepts = new Float32Array(cols * rows);
  }

  sync(step: number, at: (s: number) => EnvFrame | null): boolean {
    if (step === this.through) return false;
    if (step < this.through || step - this.through >= WINDOW) this.clear(step - WINDOW);
    for (let s = this.through + 1; s <= step; s++) this.push(s, at);
    this.through = step;
    return true;
  }

  repair(at: (s: number) => EnvFrame | null): boolean {
    if (this.holes === 0) return false;
    let filled = false;
    for (let slot = 0; slot < WINDOW; slot++) {
      if (this.codes[slot] !== -1) continue;
      const s = this.through - ((this.through - slot) % WINDOW + WINDOW) % WINDOW;
      if (this.fill(slot, s, at)) { this.holes--; filled = true; }
    }
    return filled;
  }

  private clear(before: number): void {
    this.codes.fill(-2);
    this.visits.fill(0);
    this.accepts.fill(0);
    this.decisions = this.freed = this.exposed = this.holes = 0;
    this.through = before;
  }

  private push(s: number, at: (s: number) => EnvFrame | null): void {
    const slot = (s % WINDOW + WINDOW) % WINDOW, old = this.codes[slot]!;
    if (old >= 0) {
      this.visits[old >> 1]!--;
      this.accepts[old >> 1]! -= old & 1;
      this.decisions--;
      this.freed -= this.releases[slot]!;
      this.exposed -= this.busy[slot]!;
    } else if (old === -1) this.holes--;
    this.codes[slot] = -2;
    if (s < 1) return;
    if (!this.fill(slot, s, at)) { this.codes[slot] = -1; this.holes++; }
  }

  private fill(slot: number, s: number, at: (s: number) => EnvFrame | null): boolean {
    const before = at(s - 1), after = at(s);
    if (!before || !after) return false;
    const free = before.words[0]!, rank = before.words[1]!, act = Math.trunc(after.actions[0]!) & 1, taken = act === 1 && free > 0 ? 1 : 0;
    const cell = rank * this.cols + free, busy = this.servers - free + taken;
    this.codes[slot] = cell << 1 | act;
    this.releases[slot] = after.words[0]! - free + taken;
    this.busy[slot] = busy;
    this.visits[cell]!++;
    this.accepts[cell]! += act;
    this.decisions++;
    this.freed += this.releases[slot]!;
    this.exposed += busy;
    return true;
  }
}

class AccessControlScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 2;
  protected readonly stageName = 'accessControl';

  private servers = 0;
  private cols = 0;
  private rows = 0;
  private payments: number[] = [];
  private period = 0;
  private x0 = 0;
  private z0 = 0;
  private wallZ = 0;
  private lampZ = 0;
  private backZ = 0;
  private gutterZ = 0;
  private doorX = 0;
  private fanX = 0;
  private badgeX = 0;
  private readonly bounds = new THREE.Box2();
  private readonly hues: THREE.Color[] = [];
  private lanes: Lane[] = [];
  private window: DecisionWindow | null = null;
  private policy: THREE.DataTexture | null = null;
  private lamps: THREE.InstancedMesh | null = null;
  private lampGlow: Tint[] = [];
  private tokens: Token[] = [];
  private belt: THREE.InstancedMesh | null = null;
  private readonly beltShift = uniform(0);
  private readonly recent: (EnvFrame | null)[] = [];
  private doorway: { mesh: THREE.Mesh; emitter: THREE.Mesh; glow: Tint; light: Tint } | null = null;
  private portal: Tint | null = null;
  private flashes = new Float32Array(0);
  private readonly color = new THREE.Color();
  private readonly scratch = new THREE.Color();
  private readonly point = new THREE.Vector3();
  private readonly matrix = new THREE.Matrix4();
  private lastPosition = 0;
  private pace = 0;
  private sinceRepair = 0;
  private readonly ledger = { pays: 0, free: 0, decision: '—', reward: 0, rate: NaN, decisions: 0, step: 0 };

  override legend: LegendSpec | null = null;

  private cellX(column: number): number {
    return this.x0 + column + 0.5;
  }

  private rowZ(rank: number): number {
    return this.z0 + (this.rows - 1 - rank) + 0.5;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.servers = constant(env, 'ACCESS_SERVERS');
    this.rows = constant(env, 'ACCESS_PRIORITIES');
    this.payments = constantList(env, 'ACCESS_REWARD');
    if (this.payments.length !== this.rows) throw new Error(`${env.envId} pays ${this.payments.length} priorities but declares ${this.rows}`);
    this.period = parameter(env, 'drift') ? constant(env, 'ACCESS_DRIFT_PERIOD') : 0;
    this.cols = this.servers + 1;
    this.x0 = -this.cols / 2;
    this.z0 = -this.rows / 2;
    this.wallZ = this.z0 - 0.3;
    this.lampZ = this.z0 - 0.95;
    this.backZ = this.z0 - 1.55;
    this.gutterZ = this.z0 + this.rows + 0.55;
    this.doorX = this.x0 - 2.75;
    this.fanX = this.x0 - 1.95;
    this.badgeX = this.x0 - 0.55;
    return this.bounds.set(new THREE.Vector2(this.doorX - 0.45, this.backZ - 0.35), new THREE.Vector2(this.x0 + this.cols + 0.45, this.gutterZ + 1.25)).clone();
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x6a7898).multiplyScalar(0.75);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      lattice: { w: this.cols, h: this.rows, origin: [this.x0, 0.02, this.z0], u: [1, 0, 0], v: [0, 0, 1] },
      debug: (frame, kit) => {
        const step = frame.state.step, head = frame.frameAt(step), free = head.words[0]!, rank = head.words[1]!;
        kit.marks('state', 1).set(this.cellX(free), 0.05, this.rowZ(rank), 0.2, TOKENS.uiCyan);
        const lamps = kit.marks('servers', this.cols);
        lamps.set(this.cellX(0), 0.05, this.lampZ, 0.12, free === 0 ? FULL : 0x40262b);
        for (let c = 1; c <= this.servers; c++) lamps.set(this.cellX(c), 0.05, this.lampZ, 0.12, c <= free ? MINT : EMBER);
        const next = frame.frameAt(step + 1), decision = kit.marks('decision', 1);
        if (next.step !== step + 1) return;
        const act = Math.trunc(next.actions[0]!) & 1;
        if (act === 1 && free > 0) decision.set(this.cellX(free), 0.05, this.wallZ, 0.16, GOLD);
        else if (act === 1) decision.set(this.cellX(0), 0.05, this.wallZ + 0.35, 0.16, FULL);
        else decision.set(this.cellX(free), 0.05, this.gutterZ, 0.16, TOKENS.failureBlue);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    pool.group.traverse((part) => { part.renderOrder = 6; });
    const { cols, rows, x0, z0, ledger, period } = this, cx = x0 + cols / 2, accent = new THREE.Color(this.accentOf(ctx));
    for (let k = 0; k < rows; k++) this.hues.push(hueOf(k, rows, new THREE.Color()));
    this.window = new DecisionWindow(cols, rows, this.servers);
    this.flashes = new Float32Array(cols);
    this.scene.background = new THREE.Color(0x04060b);
    const sample = () => `policy from the last ${formatCount(ledger.decisions)} decisions`;
    this.legend = {
      title: 'ACCESS CONTROL',
      rows: [
        { key: 'customer', label: 'customer pays', tint: this.hues[rows - 1]!.getHex(), value: () => String(ledger.pays) },
        { key: 'free', label: 'free servers', tint: MINT, value: () => `${ledger.free} of ${this.servers}` },
        { key: 'decision', label: 'decision', tint: accent.getHex(), value: () => ledger.decision },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(ledger.reward) },
        { key: 'service', label: 'busy servers finish', tint: EMBER, value: () => (Number.isFinite(ledger.rate) ? `${formatPercent(ledger.rate)} per step` : '—') },
      ],
      ...(period > 0
        ? { note: () => `drift cycle ${Math.floor(ledger.step / period) + 1} · ${sample()}`, bar: () => (ledger.step % period) / period }
        : { note: sample }),
    };
    const q = positionWorld.xz;

    const hall = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const tiles = abs(fract(q.mul(0.5)).sub(0.5)), seam = coverage(min(tiles.x, tiles.y).mul(2).sub(0.02));
    hall.colorNode = light.lit(rgb(0x3a4252).mul(noise(q.mul(0.9)).mul(0.12).add(0.94)).mul(seam.mul(-0.35).add(1)));
    const size = this.bounds.getSize(new THREE.Vector2()), centre = this.bounds.getCenter(new THREE.Vector2());
    const floorMesh = new THREE.Mesh(flat(size.x * 2, size.y * 2), hall);
    floorMesh.position.set(centre.x, 0, centre.y);
    floorMesh.renderOrder = 0;
    this.scene.add(floorMesh);

    const policy = new THREE.DataTexture(new Uint8Array(cols * rows * 4), cols, rows, THREE.RGBAFormat);
    policy.magFilter = policy.minFilter = THREE.NearestFilter;
    policy.needsUpdate = true;
    this.policy = policy;
    const p = uv().mul(vec2(cols, rows)), cell = floor(p), local = fract(p).sub(0.5);
    const data = texture(policy, cell.add(0.5).div(vec2(cols, rows)));
    const lean = data.r.sub(0.5).mul(2), sure = data.g, closed = float(1).sub(step(0.5, cell.x));
    const up = min(sdSegment(local, vec2(-0.16, -0.06), vec2(0, 0.1), 0.045), sdSegment(local, vec2(0.16, -0.06), vec2(0, 0.1), 0.045));
    const down = min(sdSegment(local, vec2(-0.16, 0.06), vec2(0, -0.1), 0.045), sdSegment(local, vec2(0.16, 0.06), vec2(0, -0.1), 0.045));
    const chevron = coverage(select(lean.greaterThanEqual(0), up, down)).mul(abs(lean)).mul(sure).mul(closed.mul(-0.55).add(1));
    const shade = mix(rgb(REJECT), rgb(accent), data.r);
    const stripes = local.x.add(local.y).mul(4), hatch = coverage(abs(fract(stripes).sub(0.5)).sub(0.2).div(4)).mul(closed).mul(0.04);
    const glass = rgb(0x080b12).add(shade.mul(sure.mul(0.2))).add(rgb(FULL).mul(hatch));
    const stone = mix(light.lit(rgb(0x1e2430)), light.lit(rgb(0x6b7486)), coverage(sdBox(local, vec2(0.46), 0.1)));
    const board = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    board.colorNode = mix(mix(stone, glass, coverage(sdBox(local, vec2(0.37), 0.08))), shade.mul(1.6), chevron);
    const boardMesh = new THREE.Mesh(flat(cols, rows), board);
    boardMesh.position.set(cx, 0, z0 + rows / 2);
    boardMesh.renderOrder = 1;
    this.scene.add(boardMesh);

    const railMid = (this.backZ + this.wallZ) / 2, railHalf = (this.wallZ - this.backZ) / 2 + 0.07, column = q.x.sub(x0), lx = fract(column).sub(0.5);
    const housing = sdBox(q.sub(vec2(cx, railMid)), vec2(cols / 2 + 0.12, railHalf), 0.1);
    const bay = coverage(sdBox(vec2(lx, q.y.sub(this.lampZ)), vec2(0.4, 0.47), 0.08));
    const wall = max(max(abs(q.y.sub(this.wallZ)).sub(0.07), float(DOOR).sub(abs(lx))), abs(q.x.sub(cx)).sub(cols / 2 + 0.12));
    const divider = max(abs(fract(column.add(0.5)).sub(0.5)).sub(0.05), abs(q.y.sub(railMid)).sub(railHalf));
    const ridge = coverage(min(wall, divider)), lip = smoothstep(0.07, -0.07, q.y.sub(this.wallZ)).mul(ridge);
    const metal = mix(light.lit(rgb(0x4d5566)), light.lit(rgb(0x171b25)), bay);
    const rail = sprite(mix(metal, light.lit(rgb(0x8a93a6).mul(lip.mul(0.45).add(0.75))), ridge), coverage(housing));
    const railMesh = new THREE.Mesh(flat(cols + 0.4, railHalf * 2 + 0.2), rail);
    railMesh.position.set(cx, 0, railMid);
    railMesh.renderOrder = 1;
    this.scene.add(railMesh);

    const g = q.sub(vec2(cx, this.gutterZ)), trench = sdBox(g, vec2(cols / 2 + 0.1, 0.3), 0.26);
    const pit = mix(rgb(0x05070b), rgb(0x2a303c), smoothstep(-0.3, 0.3, g.y)), travel = fract(q.x.sub(this.beltShift)).sub(0.5);
    const arrows = coverage(abs(travel.add(abs(g.y).mul(0.8))).sub(0.025)).mul(step(abs(g.y), 0.16)).mul(step(trench, -0.08));
    const gutter = sprite(mix(light.lit(pit).add(light.lit(rgb(0x4a5468)).mul(arrows.mul(0.5))), light.lit(rgb(0x7a8497)), coverage(abs(trench).sub(0.03))), coverage(trench.sub(0.035)));
    const gutterMesh = new THREE.Mesh(flat(cols + 0.4, 0.8), gutter);
    gutterMesh.position.set(cx, 0, this.gutterZ);
    gutterMesh.renderOrder = 1;
    this.scene.add(gutterMesh);

    this.lanes = [];
    for (let k = 0; k < rows; k++) {
      const z = this.rowZ(k), a = new THREE.Vector2(this.doorX, 0), b = new THREE.Vector2(this.fanX, 0), c = new THREE.Vector2(this.fanX + 0.75, z), d = new THREE.Vector2(this.badgeX, z);
      const curve = new THREE.CubicBezierCurve(a, b, c, d), points = curve.getSpacedPoints(LANE_SAMPLES), lengths = [0];
      for (let j = 1; j < points.length; j++) lengths.push(lengths[j - 1]! + points[j]!.distanceTo(points[j - 1]!));
      this.lanes.push({ points, lengths });
    }
    let channel: N<'float'> = float(9);
    for (const lane of this.lanes) for (let j = 1; j < lane.points.length; j++) channel = min(channel, sdSegment(q, vec2(lane.points[j - 1]!.x, lane.points[j - 1]!.y), vec2(lane.points[j]!.x, lane.points[j]!.y), 0.23));
    for (let k = 0; k < rows; k++) channel = min(channel, sdSegment(q, vec2(this.badgeX, this.rowZ(k)), vec2(x0 - 0.25, this.rowZ(k)), 0.23));
    const arrow = fract(q.x.mul(1.6)).sub(0.5), chevrons = coverage(abs(abs(q.y).mul(1.4).add(arrow)).sub(0.045)).mul(step(q.x, this.fanX - 0.2)).mul(step(this.doorX + 0.2, q.x)).mul(step(abs(q.y), 0.16));
    const opening = sdBox(q.sub(vec2(this.doorX - 0.14, 0)), vec2(0.12, 0.52), 0.05);
    const lane = mix(light.lit(rgb(0x4a5364)), light.lit(rgb(0x8a93a6)), coverage(abs(channel).sub(0.025))).add(light.lit(rgb(0x6d7890)).mul(chevrons.mul(0.5)));
    const door = mix(light.lit(rgb(0x8a93a6)), rgb(0x010203), coverage(opening.add(0.035)));
    const queue = sprite(mix(lane, door, coverage(opening)), max(coverage(channel), coverage(opening.sub(0.05))));
    const queueMesh = new THREE.Mesh(flat(this.x0 - this.doorX + 0.8, rows + 0.5), queue);
    queueMesh.position.set((this.doorX + x0) / 2 - 0.2, 0, 0);
    queueMesh.renderOrder = 1;
    this.scene.add(queueMesh);

    const layouts = this.payments.map((paid) => pipLayout(paid));
    for (let k = 0; k < rows; k++) {
      const badge = new THREE.Mesh(flat(0.5, 0.5), tokenMaterial(rgb(this.hues[k]!).mul(0.55), float(k), float(0.9), layouts));
      badge.position.set(this.badgeX, 0, this.rowZ(k));
      badge.renderOrder = 2;
      this.scene.add(badge);
    }

    const face = uv().sub(0.5).mul(vec2(0.62, 0.3)), lens = sdBox(face, vec2(0.27, 0.11), 0.06);
    const lampMaterial = sprite(vec3(smoothstep(-0.11, 0.11, face.y).mul(0.45).add(0.75)), coverage(lens));
    const lamps = instancedMesh(flat(0.62, 0.3), lampMaterial, cols);
    lamps.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cols * 3), 3);
    lamps.frustumCulled = false;
    lamps.renderOrder = 3;
    const matrix = new THREE.Matrix4();
    for (let c = 0; c < cols; c++) lamps.setMatrixAt(c, matrix.makeTranslation(this.cellX(c), 0, this.lampZ));
    lamps.instanceMatrix.needsUpdate = true;
    this.lamps = lamps;
    this.scene.add(lamps);
    this.lampGlow = [];
    for (let c = 0; c < cols; c++) {
      const glow = tint(), emitter = light.emitter(flat(0.56, 0.2), glow);
      emitter.position.set(this.cellX(c), 0, this.lampZ);
      this.lampGlow.push(glow);
      this.scene.add(emitter);
    }

    const block = (x: number, z: number, w: number, d: number) => {
      const occluder = light.occluder(flat(w, d));
      occluder.position.set(x, 0, z);
      this.scene.add(occluder);
    };
    const wallLeft = x0 - 0.12, wallRight = x0 + cols + 0.12;
    block((wallLeft + this.cellX(0) - DOOR) / 2, this.wallZ, this.cellX(0) - DOOR - wallLeft, 0.14);
    for (let c = 0; c < cols - 1; c++) block((this.cellX(c) + this.cellX(c + 1)) / 2, this.wallZ, 1 - 2 * DOOR, 0.14);
    block((this.cellX(cols - 1) + DOOR + wallRight) / 2, this.wallZ, wallRight - this.cellX(cols - 1) - DOOR, 0.14);
    for (let c = 0; c <= cols; c++) block(x0 + c, railMid, 0.1, railHalf * 2);
    block(cx, this.backZ, cols + 0.24, 0.14);

    const doorGlow = tint(), doorLight = tint();
    const slot = sdBox(uv().sub(0.5).mul(vec2(0.9, 0.6)), vec2(DOOR - 0.02, 0.05), 0.04);
    const doorMesh = new THREE.Mesh(flat(0.9, 0.6), sprite(doorGlow, max(coverage(slot), smoothstep(0.22, 0, slot).mul(0.3)), true));
    doorMesh.renderOrder = 2;
    const doorEmitter = light.emitter(flat(0.4, 0.14), doorLight);
    this.scene.add(doorMesh, doorEmitter);
    this.doorway = { mesh: doorMesh, emitter: doorEmitter, glow: doorGlow, light: doorLight };
    const portal = tint(), portalEmitter = light.emitter(flat(0.14, 0.9), portal);
    portalEmitter.position.set(this.doorX - 0.14, 0, 0);
    this.portal = portal;
    this.scene.add(portalEmitter);

    this.tokens = [0, 1].map(() => {
      const hue = tint(), kind = uniform(0), fade = uniform(0), glow = tint();
      const mesh = new THREE.Mesh(flat(1, 1), tokenMaterial(hue, kind, fade, layouts));
      mesh.renderOrder = 5;
      mesh.visible = false;
      const emitter = light.emitter(new THREE.CircleGeometry(0.26, 32).rotateX(-Math.PI / 2), glow);
      this.scene.add(mesh, emitter);
      return { mesh, hue, kind, fade, glow, emitter };
    });

    const beltGeometry = flat(1, 1);
    beltGeometry.setAttribute('hue', new THREE.InstancedBufferAttribute(new Float32Array(BELT_STEPS * 3), 3));
    beltGeometry.setAttribute('kind', new THREE.InstancedBufferAttribute(new Float32Array(BELT_STEPS), 1));
    beltGeometry.setAttribute('fade', new THREE.InstancedBufferAttribute(new Float32Array(BELT_STEPS), 1));
    const belt = instancedMesh(beltGeometry, tokenMaterial(attribute<'vec3'>('hue', 'vec3'), attribute<'float'>('kind', 'float'), attribute<'float'>('fade', 'float'), layouts), BELT_STEPS);
    belt.count = 0;
    belt.frustumCulled = false;
    belt.renderOrder = 4;
    this.belt = belt;
    this.scene.add(belt);
  }

  private convey(position: number, first: number): void {
    const belt = this.belt!, geometry = belt.geometry, hue = geometry.getAttribute('hue') as THREE.InstancedBufferAttribute, kind = geometry.getAttribute('kind') as THREE.InstancedBufferAttribute, fade = geometry.getAttribute('fade') as THREE.InstancedBufferAttribute;
    const end = this.x0 + this.cols + 0.05, matrix = this.matrix;
    let n = 0;
    for (let j = 1; j < this.recent.length; j++) {
      const before = this.recent[j - 1], after = this.recent[j];
      if (!before || !after) continue;
      const was = before.words[0]!, rank = before.words[1]!, act = Math.trunc(after.actions[0]!) & 1;
      if (act === 1 && was > 0) continue;
      const age = position - (first + j - 1 + (act === 1 ? DROP : SINK)), x = this.cellX(was) + BELT * age;
      if (age < 0 || x > end) continue;
      const color = this.hues[rank]!, size = this.radius(rank) * 2 * LANDED;
      belt.setMatrixAt(n, matrix.makeScale(size, 1, size).setPosition(x, 0, this.gutterZ));
      hue.setXYZ(n, color.r * 0.8, color.g * 0.8, color.b * 0.8);
      kind.setX(n, rank);
      fade.setX(n, 1 - span(x, end - 0.9, end));
      n++;
    }
    belt.count = n;
    belt.instanceMatrix.needsUpdate = hue.needsUpdate = kind.needsUpdate = fade.needsUpdate = true;
  }

  protected override drop(): void {
    const pool = this.pool?.group;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || object.parent === pool) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
      if (object instanceof THREE.InstancedMesh) object.dispose();
    });
    this.policy?.dispose();
  }

  private radius(rank: number): number {
    return 0.2 + 0.18 * (this.rows > 1 ? rank / (this.rows - 1) : 1);
  }

  private place(token: Token, rank: number, x: number, z: number, scale: number, fade: number, dim = 1): void {
    const visible = fade > 0.002 && scale > 0.002;
    token.mesh.visible = visible;
    const hue = this.hues[rank]!;
    setTint(token.hue, hue, 1 - 0.2 * (1 - dim));
    token.kind.value = rank;
    token.fade.value = fade;
    token.mesh.position.set(x, 0, z);
    token.mesh.scale.setScalar(this.radius(rank) * 2 * scale);
    token.emitter.position.set(x, 0, z);
    token.emitter.scale.setScalar(this.radius(rank) / 0.26 * scale);
    setTint(token.glow, hue, visible ? 5 * fade * dim : 0);
  }

  private hide(token: Token): void {
    token.mesh.visible = false;
    token.glow.value.set(0, 0, 0);
  }

  private arrive(token: Token, lane: Lane, rank: number, column: number, u: number, fade: number): void {
    const bend = lane.lengths[lane.lengths.length - 1]!, total = bend + this.cellX(column) - this.badgeX, s = u * total;
    if (s >= bend) {
      this.place(token, rank, this.badgeX + s - bend, this.rowZ(rank), 1, fade);
      return;
    }
    let j = 1;
    while (j < lane.lengths.length - 1 && lane.lengths[j]! < s) j++;
    const a = lane.points[j - 1]!, b = lane.points[j]!, w = (s - lane.lengths[j - 1]!) / Math.max(1e-6, lane.lengths[j]! - lane.lengths[j - 1]!);
    this.place(token, rank, lerp(a.x, b.x, w), lerp(a.y, b.y, w), 1, fade);
  }

  private celebrate(pool: Stage2D['pool'], column: number, rank: number, paid: number, now: number): void {
    const top = Math.max(...this.payments), calm = pool.pulse.bloom;
    pool.burst(this.point.set(this.cellX(column), 0, this.wallZ), this.hues[rank]!, 1 + 1.5 * paid / top, now);
    pool.pulse.bloom = Math.max(calm, 0.3 * paid / top);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const window = this.window, policy = this.policy, lamps = this.lamps, doorway = this.doorway, portal = this.portal;
    if (!window || !policy || !lamps || !doorway || !portal) return;
    pool.update(frame.dtReal, time);
    const at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * 5;
    const i = frame.state.step, t = clamp01(frame.position - i), head = frame.frameAt(i);
    const upcoming = at(i + 1), next = fast ? null : upcoming, prev = fast ? null : at(i - 1);
    const free = head.words[0]!, rank = head.words[1]!;

    let changed = window.sync(i, at);
    this.sinceRepair += frame.dtReal;
    if (this.sinceRepair > 0.5) {
      this.sinceRepair = 0;
      changed = window.repair(at) || changed;
    }
    if (changed) {
      const texels = policy.image.data as Uint8Array;
      for (let cell = 0; cell < this.cols * this.rows; cell++) {
        const visits = window.visits[cell]!;
        texels[cell * 4] = visits > 0 ? Math.round(255 * window.accepts[cell]! / visits) : 128;
        texels[cell * 4 + 1] = Math.round(255 * (1 - Math.exp(-visits / 6)));
      }
      policy.needsUpdate = true;
    }

    const act = next ? Math.trunc(next.actions[0]!) & 1 : 0, taken = next && act === 1 && free > 0 ? 1 : 0, blocked = next !== null && act === 1 && free === 0;
    const released = next ? next.words[0]! - free + taken : 0;
    const flashes = this.flashes.fill(0);
    let shown = free;
    if (next) {
      if (taken && t >= REST) {
        shown--;
        flashes[free] = 0.7 * Math.exp(-(t - REST) / 0.25);
      }
      for (let j = 0; j < released; j++) {
        const when = lerp(RELEASE_FROM, RELEASE_TO, (j + 0.5) / released);
        if (t < when) continue;
        shown++;
        flashes[free - taken + j + 1] = Math.max(flashes[free - taken + j + 1]!, Math.exp(-(t - when) / 0.3));
      }
      if (blocked && t >= BUMP) flashes[0] = Math.exp(-(t - BUMP) / 0.3);
    }
    let admitted = -1;
    if (prev) {
      const before = prev.words[0]!, took = (Math.trunc(head.actions[0]!) & 1) === 1 && before > 0 ? 1 : 0, freed = free - before + took;
      for (let j = 0; j < freed; j++) {
        const lamp = before - took + j + 1, age = t + 1 - lerp(RELEASE_FROM, RELEASE_TO, (j + 0.5) / freed);
        if (lamp < this.cols) flashes[lamp] = Math.max(flashes[lamp]!, Math.exp(-age / 0.3));
      }
      if (took) admitted = before;
    }

    for (let c = 0; c < this.cols; c++) {
      const lit = c === 0 ? shown === 0 : c <= shown;
      this.color.set(c === 0 ? (lit ? FULL : 0x3a1c22) : lit ? MINT : EMBER).multiplyScalar(c === 0 ? (lit ? 1.5 : 0.35) : lit ? 1.45 : 0.85);
      this.color.multiplyScalar(1 + 1.6 * flashes[c]!);
      lamps.setColorAt(c, this.color);
      setTint(this.lampGlow[c]!, this.color, c === 0 && !lit ? 0 : 8);
    }
    lamps.instanceColor!.needsUpdate = true;

    if (admitted >= 0 && prev) {
      const glow = Math.exp(-t / 0.45) * (0.45 + 0.55 * head.reward / Math.max(...this.payments));
      doorway.mesh.visible = true;
      doorway.mesh.position.set(this.cellX(admitted), 0, this.wallZ + 0.05);
      doorway.emitter.position.copy(doorway.mesh.position);
      setTint(doorway.glow, this.scratch.set(GOLD), 1.6 * glow);
      setTint(doorway.light, this.scratch.set(GOLD), 7 * glow);
    } else {
      doorway.mesh.visible = false;
      doorway.light.value.set(0, 0, 0);
    }

    this.recent.length = 0;
    if (!fast) for (let s = i + 1 - BELT_STEPS; s <= i + 1; s++) this.recent.push(s === i ? head : at(s));
    this.convey(frame.position, i + 1 - BELT_STEPS);
    if (!fast) this.beltShift.value = frame.position * BELT % 1;

    const [first, second] = this.tokens as [Token, Token];
    const x = this.cellX(free), z = this.rowZ(rank);
    if (!next || t < REST) this.place(first, rank, x, z, 1, 1);
    else if (taken) {
      const u = smooth(span(t, REST, 1));
      this.place(first, rank, x, lerp(z, this.wallZ, u), 1 - 0.75 * span(t, 0.72, 1), 1 - span(t, 0.86, 1));
    } else if (blocked) {
      const top = this.wallZ + 0.42, rise = easeOut(span(t, REST, BUMP)), fall = easeIn(span(t, BUMP, DROP));
      if (t >= DROP) this.hide(first);
      else this.place(first, rank, x, t < BUMP ? lerp(z, top, rise) : lerp(top, this.gutterZ, fall), lerp(1, LANDED, fall), 1, 1 - fall);
    } else {
      const u = easeIn(span(t, REST, SINK));
      if (t >= SINK) this.hide(first);
      else this.place(first, rank, x, lerp(z, this.gutterZ, u), lerp(1, LANDED, u), 1, 1 - u);
    }

    if (next && t >= ARRIVE) {
      const arriving = next.words[1]!;
      this.arrive(second, this.lanes[arriving]!, arriving, next.words[0]!, easeOut(span(t, ARRIVE, 1)), span(t, ARRIVE, ARRIVE + 0.08));
      setTint(portal, this.hues[arriving]!, 6 * Math.exp(-Math.max(0, t - ARRIVE) / 0.18));
    } else {
      this.hide(second);
      portal.value.set(0, 0, 0);
    }

    if (frame.scrubbed) pool.clear();
    else if (!fast && frame.prev && i === frame.prev.step + 1 && prev) {
      for (const event of frame.events) if (event.type === 'goal') this.celebrate(pool, prev.words[0]!, prev.words[1]!, head.reward, time);
    }

    this.ledger.pays = this.payments[rank]!;
    this.ledger.free = free;
    const choice = upcoming ? Math.trunc(upcoming.actions[0]!) & 1 : -1;
    this.ledger.decision = choice < 0 ? '—' : choice === 0 ? 'turn away' : free > 0 ? 'admit' : 'admit · no server free';
    this.ledger.reward = frame.state.reward;
    this.ledger.rate = window.exposed > 0 ? window.freed / window.exposed : NaN;
    this.ledger.decisions = window.decisions;
    this.ledger.step = i;
  }
}

export default (() => new AccessControlScene()) satisfies SceneFactory;
