import * as THREE from 'three/webgpu';
import { abs, atan, clamp, dot, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, uvec2, vec2 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { coinSwapLoaded, coinSwapToss } from '@sim/envs/coinswap';
import type { EnvFrame, EventStream } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Scalar = THREE.UniformNode<'float', number>;
type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
type Span = readonly [number, number];

const SLOTS = 40;
const PITCH = 0.52;
const JETON = 0.19;
const BRIDGE = 0.08;
const LINE = 0.036;
const COIN = 1;
const LANE = 1.35;
const GROOVE = 0.33;
const X0 = 1.55;
const XL = X0 - PITCH / 2;
const XR = X0 + (SLOTS - 0.5) * PITCH;
const BINS = 128;
const BIN_W = (XR - XL) / BINS;
const AXIS = -4.75;
const REACH = 1.45;
const SPAN = REACH * 0.9;
const PIVOT = -8;
const ARM = 5.65;
const HAND = 1.3;
const LEFT = -3.9;
const RIGHT = XR + 0.75;
const TOP = LANE + COIN + 0.8;
const BOTTOM = AXIS - REACH - 0.8;
const FAST = 5;
const GLIDE = 14;
const GLOW: Span = [0.46, 0.95];
const SPINS = 2;
const RING_SECONDS = 0.7;
const BOW = 0.62;

const AIM: Span = [0, 0.16];
const FLIP: Span = [0.1, 0.46];
const DROP: Span = [0.46, 0.7];
const TRADE: Span = [0.6, 1];

const GOLD = 0xe9ad45;
const GOLD_DARK = 0x6e440f;
const GOLD_LIGHT = 0xffe9a8;
const PEWTER = 0x8d96a4;
const PEWTER_DARK = 0x363c47;
const PEWTER_LIGHT = 0xe3e8ef;
const PAID = TOKENS.celebrationGold;
const BLANK = 0x56627c;
const BLANK_RIM = 0xaab6cf;
const PAID_RIM = 0x6b4712;
const EMBER = 0x7a4f14;
const BAIZE = 0x2b6450;
const WALNUT = 0x5c3a22;
const BRASS = 0xb48a45;
const INK = 0x1b140d;
const IVORY = 0xf2e8d2;
const TABLE = 0x1a130d;
const COAT = 0x2b3140;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const disc = (radius: number) => new THREE.CircleGeometry(radius, 64).rotateX(-Math.PI / 2);
const chartPoint = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (width: number, depth: number): N<'vec2'> => uv().sub(0.5).mul(vec2(width, depth));
const pixel = (): N<'float'> => max(fwidth(positionWorld.x), fwidth(positionWorld.z));
const cover = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const laneY = (lane: number) => (lane === 0 ? LANE : -LANE);
const laneOf = (flag: N<'float'>): N<'float'> => mix(float(LANE), float(-LANE), flag);
const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const light2 = vec2(-0.55, 0.83);

function sprite(geometry: THREE.BufferGeometry, order: number, color: N<'vec3'>, opacity: N<'float'>): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function jeton(p: N<'vec2'>, paid: N<'float'>, lit: Lit): N<'vec3'> {
  const r = length(p).div(JETON), facing = clamp(dot(p.div(JETON), light2), -1, 1);
  const face = mix(rgb(BLANK), rgb(PAID), paid).mul(facing.mul(0.16).add(0.92));
  const ring = smoothstep(0.52, 0.58, r).mul(smoothstep(0.7, 0.64, r)).mul(0.35);
  const rim = smoothstep(0.76, 0.84, r);
  const shaded = mix(mix(face, face.mul(0.7), ring), mix(rgb(BLANK_RIM), rgb(PAID_RIM), paid), rim);
  return lit(shaded).add(rgb(PAID).mul(paid.mul(rim.oneMinus()).mul(0.35)));
}

function sun(p: N<'vec2'>): N<'float'> {
  const turned = vec2(p.x.add(p.y), p.y.sub(p.x)).mul(Math.SQRT1_2);
  return min(min(sdBox(p, vec2(0.3), 0.04), sdBox(turned, vec2(0.3), 0.04)), sdCircle(p, 0.33));
}

function moon(p: N<'vec2'>): N<'float'> {
  return max(sdCircle(p, 0.42), sdCircle(p.sub(vec2(0.2, 0.12)), 0.36).negate());
}

function hand(p: N<'vec2'>) {
  const palm = sdBox(p.sub(vec2(-0.28, -0.02)), vec2(0.36, 0.29), 0.2);
  const index = sdSegment(p, vec2(0.02, 0.13), vec2(0.86, 0.13), 0.1);
  const thumb = sdSegment(p, vec2(-0.24, 0.25), vec2(0.16, 0.31), 0.085);
  const fingers = [
    sdSegment(p, vec2(0.0, -0.03), vec2(0.2, -0.03), 0.09),
    sdSegment(p, vec2(-0.02, -0.18), vec2(0.16, -0.18), 0.085),
    sdSegment(p, vec2(-0.06, -0.31), vec2(0.1, -0.31), 0.075),
  ];
  const parts = [index, thumb, ...fingers];
  return { body: parts.reduce((d, part) => min(d, part), palm), parts };
}

interface Bin { tossedZero: number; loadedZero: number; matched: number; count: number }

function binOf(b: number, size: number, i: number, at: (s: number) => EnvFrame | null): Bin | null {
  const first = Math.max(1, b * size), last = Math.min(b * size + size - 1, i);
  const bin: Bin = { tossedZero: 0, loadedZero: 0, matched: 0, count: 0 };
  if (last < first) return bin;
  let prev = at(first - 1);
  if (!prev) return null;
  for (let s = first; s <= last; s++) {
    const cur = at(s);
    if (!cur) return null;
    const toss = coinSwapToss(cur), gold = coinSwapLoaded(prev);
    if (toss === 0) bin.tossedZero++;
    if (gold === 0) bin.loadedZero++;
    if (toss === gold) bin.matched++;
    bin.count++;
    prev = cur;
  }
  return bin;
}

function eventsKnown(stream: EventStream, from: number, to: number): boolean {
  for (let b = Math.floor(from / stream.blockFrames); b <= Math.floor(to / stream.blockFrames); b++) if (!stream.blocks[b]) return false;
  return true;
}

interface Coin {
  mesh: THREE.Mesh;
  shadow: THREE.Mesh;
  shade: Scalar;
  side: Scalar;
  flash: Scalar;
  x: number;
  y: number;
}

class CoinSwapScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 2;
  protected readonly stageName = 'coinSwap';

  private swapRate = 0;
  private bin = 16;
  private readonly slots = new Float32Array((SLOTS + 1) * 4);
  private readonly lanes = new THREE.DataTexture(this.slots, SLOTS + 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly columns = new Float32Array(BINS * 4);
  private readonly ledgerTexture = new THREE.DataTexture(this.columns, BINS, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly bins = new Map<number, Bin>();
  private readonly scroll = uniform(0);
  private readonly arrived = uniform(0);
  private readonly flyerPaid = uniform(0);
  private readonly goldLight: Tint = uniform(new THREE.Vector3());
  private readonly flashLight: Tint = uniform(new THREE.Vector3());
  private readonly ringRadius = uniform(0);
  private readonly ringFade = uniform(0);
  private ring: THREE.Mesh | null = null;
  private ringAt = -Infinity;
  private readonly point = new THREE.Vector3();
  private readonly tint = new THREE.Color();
  private gold: Coin | null = null;
  private pewter: Coin | null = null;
  private arm: THREE.Group | null = null;
  private palm: THREE.Mesh | null = null;
  private flyer: THREE.Mesh | null = null;
  private goldGlow: THREE.Mesh | null = null;
  private flashGlow: THREE.Mesh | null = null;
  private pewterBlock: THREE.Mesh | null = null;
  private aim = 0;
  private pace = 0;
  private lastPosition = 0;
  private settled = -1;
  private readonly ledger = { step: 0, toss: 0, reward: 0, loaded: 0, share: NaN, since: null as number | null, known: false };

  override legend: LegendSpec | null = null;

  private legendFor(accent: number): LegendSpec {
    const ago = (steps: number) => `${formatCount(steps)} ${steps === 1 ? 'step' : 'steps'} ago`;
    return {
      title: 'COIN SWAP',
      rows: [
        { key: 'loaded', label: 'loaded coin', tint: GOLD, value: () => `action ${this.ledger.loaded}` },
        { key: 'toss', label: 'tossed', tint: accent, value: () => (this.ledger.step > 0 ? `action ${this.ledger.toss}` : '—') },
        { key: 'payout', label: 'payout', tint: PAID, value: () => (this.ledger.step > 0 ? formatSigned(this.ledger.reward) : '—') },
        { key: 'share', label: `on loaded, last ${formatCount(BINS * this.bin)}`, tint: accent, value: () => formatPercent(this.ledger.share) },
        { key: 'since', label: 'last swap', tint: GOLD_LIGHT, value: () => (!this.ledger.known ? '…' : this.ledger.since === null ? 'none yet' : ago(this.ledger.since)) },
      ],
      note: () => `odds 1 in ${formatCount(1 / this.swapRate)}`,
      bar: () => (Number.isFinite(this.ledger.share) ? this.ledger.share : 0),
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.swapRate = parameter(env, 'swapRate');
    this.bin = 2 ** Math.min(6, Math.max(3, Math.round(Math.log2(1 / (6 * this.swapRate)))));
    if (env.words !== 1 || env.actions !== 2 || env.box !== 0) throw new Error(`${env.envId} is not a two-coin swap`);
    return new THREE.Box2(new THREE.Vector2(LEFT - 0.4, -TOP - 0.4), new THREE.Vector2(RIGHT + 0.4, -BOTTOM + 1.6));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x9aa5b8).multiplyScalar(0.85);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, at = (s: number) => { if (s < 0) return null; const f = frame.frameAt(s); return f.step === s ? f : null; };
        const head = at(i)!, truth = kit.marks('loaded route', SLOTS), tosses = kit.marks('tosses', SLOTS);
        for (let k = 0; k < SLOTS; k++) {
          const s = i - k, cur = s >= 1 ? at(s) : null, prev = cur ? at(s - 1) : null;
          if (!cur || !prev) continue;
          const x = X0 + k * PITCH;
          truth.set(x, 0.1, -laneY(coinSwapLoaded(prev)), 0.08, TOKENS.celebrationGold);
          tosses.set(x, 0.1, -laneY(coinSwapToss(cur)), 0.15, cur.reward > 0 ? TOKENS.uiCyan : TOKENS.failureBlue);
        }
        kit.marks('loaded coin', 1).set(0, 0.1, -laneY(coinSwapLoaded(head)), 0.45, TOKENS.celebrationGold);
        if (i >= 1) {
          const toss = kit.curve('toss', 2, accent);
          toss.point(PIVOT, 0.1, 0);
          toss.point(0, 0.1, -laneY(coinSwapToss(head)));
        }
        const trace = kit.marks('trace', BINS), newest = Math.floor(i / this.bin);
        for (let c = 0; c < BINS; c++) {
          const bin = binOf(newest - c, this.bin, i, at);
          if (!bin || bin.count === 0) continue;
          trace.set(XL + (c + 0.5) * BIN_W, 0.1, -(AXIS + (2 * bin.tossedZero / bin.count - 1) * SPAN), 0.05, bin.loadedZero * 2 > bin.count ? TOKENS.celebrationGold : TOKENS.uiCyan);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit;
    this.legend = this.legendFor(this.accentOf(ctx));
    pool.group.renderOrder = 1;
    this.scene.background = new THREE.Color(0x050403);
    for (const table of [this.lanes, this.ledgerTexture]) {
      table.magFilter = table.minFilter = THREE.NearestFilter;
      table.needsUpdate = true;
    }
    this.scene.add(this.table(lit), this.board(lit), this.lanesMesh(lit), this.ledgerMesh(this.accentOf(ctx)));

    const goldGlow = light.emitter(disc(0.9), this.goldLight), flashGlow = light.emitter(disc(0.7), this.flashLight);
    this.goldGlow = goldGlow;
    this.flashGlow = flashGlow;
    this.pewterBlock = light.occluder(disc(0.95));
    this.scene.add(goldGlow, flashGlow, this.pewterBlock);

    this.gold = this.coin('gold');
    this.pewter = this.coin('pewter');
    this.flyer = sprite(flat(2 * JETON + 0.1, 2 * JETON + 0.1), 4, jeton(local(2 * JETON + 0.1, 2 * JETON + 0.1), this.flyerPaid, lit), coverage(sdCircle(local(2 * JETON + 0.1, 2 * JETON + 0.1), JETON)));
    this.flyer.visible = false;
    this.arm = this.armGroup(this.accentOf(ctx), light.occluder(flat(2.4, 0.7).translate(ARM - 0.4, 0, 0)));
    const ringP = local(8, 8), ringD = abs(length(ringP).sub(this.ringRadius)).sub(this.ringFade.mul(0.05).add(0.015));
    this.ring = sprite(flat(8, 8), 3, rgb(GOLD_LIGHT).mul(1.6), coverage(ringD).mul(this.ringFade));
    (this.ring.material as THREE.Material).blending = THREE.AdditiveBlending;
    this.ring.visible = false;
    this.scene.add(this.flyer, this.arm, this.ring);
  }

  private table(lit: Lit): THREE.Mesh {
    const q = chartPoint(), grain = noise(vec2(q.x.mul(0.2), q.y.mul(3))).mul(0.25).add(noise(q.mul(1.7)).mul(0.08)).add(0.85);
    const mesh = sprite(flat(120, 80), 0, lit(rgb(TABLE).mul(grain)), float(1));
    mesh.position.set((LEFT + RIGHT) / 2, 0, -(TOP + BOTTOM) / 2);
    return mesh;
  }

  private board(lit: Lit): THREE.Mesh {
    const q = chartPoint(), px = pixel(), cut = (d: N<'float'>) => cover(d, px);
    const cx = (LEFT + RIGHT) / 2, cy = (TOP + BOTTOM) / 2, hx = (RIGHT - LEFT) / 2, hy = (TOP - BOTTOM) / 2;
    const b = q.sub(vec2(cx, cy));
    const outer = sdBox(b, vec2(hx, hy), 0.42), inner = sdBox(b, vec2(hx - 0.34, hy - 0.34), 0.2);
    const grain = noise(vec2(q.x.mul(0.3), q.y.mul(8))).mul(0.2).add(noise(q.mul(4)).mul(0.05)).add(0.9);
    const bevel = cut(abs(inner.sub(0.06)).sub(0.05)).mul(smoothstep(-1, 1, b.y.div(hy).sub(b.x.div(hx))).mul(0.6).add(0.4));
    const wood = mix(rgb(WALNUT).mul(grain), rgb(0x8a5a33), bevel);
    const check = fract(floor(q.x.div(1.3)).add(floor(q.y.div(1.3))).mul(0.5));
    const cloth = rgb(BAIZE).mul(check.mul(0.1).add(0.95)).mul(noise(q.mul(11)).mul(0.07).add(1));
    const rimShadow = smoothstep(0.9, 0, inner.negate()).mul(0.35);
    let color: N<'vec3'> = mix(wood, cloth.mul(rimShadow.oneMinus()), cut(inner));

    const groove = (y: number) => sdBox(q.sub(vec2((XL + XR) / 2, y)), vec2((XR - XL) / 2 + 0.16, GROOVE), GROOVE);
    for (const y of [LANE, -LANE]) {
      const g = groove(y), inside = cut(g), depth = smoothstep(GROOVE, GROOVE - 0.16, q.y.sub(y)).mul(0.45).add(0.55);
      color = mix(color, cloth.mul(0.48).mul(depth), inside);
      color = mix(color, rgb(0x2f5a48), cut(abs(g.sub(0.02)).sub(0.018)).mul(step(q.y, y)).mul(0.7));
    }
    const sx = fract(q.x.sub(X0).div(PITCH).add(0.5)).sub(0.5).mul(PITCH);
    const within = step(XL, q.x).mul(step(q.x, XR));
    for (const y of [LANE, -LANE]) {
      const ring = abs(length(vec2(sx, q.y.sub(y))).sub(JETON + 0.02)).sub(0.012);
      color = mix(color, cloth.mul(0.34), cut(ring).mul(within).mul(0.8));
    }

    const channel = sdBox(q, vec2(0.34, LANE), 0.2);
    color = mix(color, cloth.mul(0.55), cut(channel));
    for (const y of [LANE, -LANE]) {
      const c = q.sub(vec2(0, y)), r = length(c);
      color = mix(color, cloth.mul(0.4).mul(smoothstep(-1.1, 1.1, dot(c, light2)).mul(-0.4).add(1)), cut(r.sub(COIN + 0.1)));
      const facing = dot(c.div(max(r, 1e-4)), light2);
      color = mix(color, mix(rgb(0x5a4420), rgb(0xe0bd78), facing.mul(0.5).add(0.5)), cut(abs(r.sub(COIN + 0.13)).sub(0.045)));
    }

    const panel = sdBox(q.sub(vec2((XL + XR) / 2, AXIS)), vec2((XR - XL) / 2 + 0.16, REACH + 0.16), 0.14);
    color = mix(color, rgb(0x0a1611).mul(smoothstep(REACH + 0.16, REACH - 0.3, abs(q.y.sub(AXIS))).mul(0.3).add(0.7)), cut(panel));
    color = mix(color, rgb(BRASS).mul(0.8), cut(abs(panel).sub(0.022)));
    const gauge = step(XL, q.x).mul(step(q.x, XR));
    color = mix(color, rgb(BRASS).mul(0.75), cut(abs(q.y.sub(AXIS)).sub(0.014)).mul(gauge));
    for (const half of [REACH / 2, -REACH / 2]) {
      const dash = step(0.5, fract(q.x.div(0.24)));
      color = mix(color, rgb(0x31473e), cut(abs(q.y.sub(AXIS + half)).sub(0.008)).mul(gauge).mul(dash));
    }
    return this.place(sprite(flat(2 * hx + 0.2, 2 * hy + 0.2), 1, lit(color), cut(outer)), cx, cy);
  }

  private lanesMesh(lit: Lit): THREE.Mesh {
    const q = chartPoint(), px = pixel(), cut = (d: N<'float'>) => cover(d, px), table = texture(this.lanes);
    const u = q.x.sub(X0).div(PITCH).add(1).sub(this.scroll), j = floor(u.add(0.5)), dx = u.sub(j).mul(PITCH);
    const near = select(dx.greaterThan(0), float(1), float(-1)), other = j.add(near);
    const load = (index: N<'float'>) => table.load(uvec2(vec2(clamp(index, 0, SLOTS), 0)));
    const inRange = (index: N<'float'>) => step(-0.5, index).mul(step(index, SLOTS + 0.5));
    const here = load(j), there = load(other);
    const hereOk = here.w.mul(inRange(j)), thereOk = there.w.mul(inRange(other));
    const boundary = float(PITCH / 2).sub(abs(dx));
    const differs = step(0.5, abs(there.x.sub(here.x)));
    const joined = thereOk.mul(differs.oneMinus());
    const floorY = laneOf(here.x), across = abs(q.y.sub(floorY));
    const bed = cut(across.sub(GROOVE - 0.05)).mul(max(joined, cut(boundary.negate()))).mul(hereOk);
    const bridge = cut(max(boundary.sub(BRIDGE), abs(q.y).sub(LANE))).mul(hereOk).mul(thereOk).mul(differs);
    const leaving = (index: N<'float'>) => select(index.greaterThan(SLOTS - 0.5), this.scroll.oneMinus(), float(1));
    const route = max(bed, bridge.mul(leaving(other))).mul(leaving(j));
    const fade = smoothstep(2, SLOTS, j).mul(-0.3).add(1);
    const shown = hereOk.mul(select(j.lessThan(0.5), this.arrived, leaving(j)));
    const p = vec2(dx, q.y.sub(laneOf(here.y)));
    const body = cut(length(p).sub(JETON)).mul(shown);
    const warmth = smoothstep(GROOVE, 0, across);
    const routeColor = mix(rgb(EMBER).mul(0.8), rgb(GOLD).mul(0.62), warmth).mul(fade);
    const face = jeton(p, here.z, lit).mul(fade);
    const alpha = body.add(route.mul(0.9).mul(body.oneMinus()));
    const color = face.mul(body).add(routeColor.mul(route.mul(0.9)).mul(body.oneMinus())).div(max(alpha, 1e-4));
    return this.place(sprite(flat(XR - XL + PITCH, 2 * (LANE + GROOVE)), 2, color, alpha), (XL + XR + PITCH) / 2, 0);
  }

  private ledgerMesh(accent: number): THREE.Mesh {
    const q = chartPoint(), px = pixel(), cut = (d: N<'float'>) => cover(d, px), table = texture(this.ledgerTexture);
    const x = q.x.sub(XL).div(BIN_W), c = floor(x);
    const load = (k: N<'float'>) => table.load(uvec2(vec2(clamp(k, 0, BINS - 1), 0)));
    const inRange = (k: N<'float'>) => step(-0.5, k).mul(step(k, BINS - 0.5));
    const point = (k: N<'float'>, cell: N<'vec4'>) => vec2(float(XL).add(k.add(0.5).mul(BIN_W)), float(AXIS).add(cell.x.mul(2).sub(1).mul(SPAN)));
    const before = c.sub(1), after = c.add(1), a = load(before), h = load(c), b = load(after);
    const live = (k: N<'float'>, cell: N<'vec4'>) => cell.w.mul(inRange(k)).mul(cell.z);
    const segment = (k0: N<'float'>, c0: N<'vec4'>, k1: N<'float'>, c1: N<'vec4'>) => cut(sdSegment(q, point(k0, c0), point(k1, c1), LINE)).mul(min(live(k0, c0), live(k1, c1)));
    const vertex = cut(sdCircle(q.sub(point(c, h)), LINE * 1.6)).mul(live(c, h));
    const trace = max(max(segment(before, a, c, h), segment(c, h, after, b)), vertex);
    const upper = step(AXIS, q.y), gold = mix(float(1).sub(h.y), h.y, upper).mul(h.w).mul(inRange(c));
    const rise = abs(q.y.sub(AXIS));
    const axis = cut(rise.sub(0.012)).mul(0.8);
    const band = max(gold.mul(cut(rise.sub(REACH))).mul(0.45), axis);
    const bandColor = mix(mix(rgb(EMBER).mul(0.75), rgb(GOLD).mul(0.5), smoothstep(REACH, 0, rise)), rgb(BRASS), axis);
    const lineColor = rgb(accent).mul(1.4);
    const alpha = trace.add(band.mul(trace.oneMinus()));
    const color = lineColor.mul(trace).add(bandColor.mul(band).mul(trace.oneMinus())).div(max(alpha, 1e-4));
    return this.place(sprite(flat(XR - XL, 2 * REACH + 0.2), 2, color, alpha), (XL + XR) / 2, AXIS);
  }

  private coin(kind: 'gold' | 'pewter'): Coin {
    const size = 2 * COIN + 0.16, p = local(size, size).div(COIN), r = length(p), unit = p.div(max(r, 1e-4));
    const side = uniform(0), flash = uniform(0), shade = uniform(1);
    const [dark, base, bright] = kind === 'gold' ? [GOLD_DARK, GOLD, GOLD_LIGHT] : [PEWTER_DARK, PEWTER, PEWTER_LIGHT];
    const facing = dot(unit, light2), reeds = sin(atan(p.y, p.x).mul(96)).mul(0.5).add(0.5);
    const rimTone = mix(rgb(dark), rgb(bright), facing.mul(0.5).add(0.5)).mul(reeds.mul(0.22).add(0.86));
    const sheen = exp(length(p.sub(vec2(-0.38, 0.42))).negate().div(0.45)).mul(0.4);
    let color: N<'vec3'> = mix(rgb(base), rgb(bright), sheen);
    const bead = cover(abs(r.sub(0.8)).sub(0.022), fwidth(r));
    color = mix(color, mix(rgb(dark), rgb(bright), facing.oneMinus().mul(0.5)), bead);
    const emblem = kind === 'gold' ? sun : moon;
    const obverse = (q: N<'vec2'>) => select(side.greaterThan(0.5), abs(length(q).sub(0.42)).sub(0.05), emblem(q));
    const raised = coverage(obverse(p)), lip = coverage(obverse(p.add(light2.mul(0.05)))), heel = coverage(obverse(p.sub(light2.mul(0.05))));
    color = mix(color, rgb(base).mul(1.06), raised);
    color = mix(color, rgb(bright), raised.mul(lip.oneMinus()).mul(0.85));
    color = mix(color, rgb(dark), raised.mul(heel.oneMinus()).mul(0.75));
    color = mix(color, rimTone, smoothstep(0.86, 0.9, r));
    color = mix(rgb(INK), color, coverage(r.sub(0.975)));
    const glow = kind === 'gold' ? 1.18 : 1;
    const out = color.mul(shade).mul(glow).add(rgb(PAID).mul(flash.mul(0.85)));
    const mesh = sprite(flat(size, size), kind === 'gold' ? 6 : 5, out, coverage(sdCircle(p, 1)));
    const shadow = sprite(flat(size + 0.6, size + 0.6), 3, rgb(0x000000), smoothstep(1.12, 0.72, length(local(size + 0.6, size + 0.6).div(COIN))).mul(0.5));
    this.scene.add(shadow, mesh);
    return { mesh, shadow, shade, side, flash, x: 0, y: 0 };
  }

  private armGroup(accent: number, block: THREE.Mesh): THREE.Group {
    const group = new THREE.Group();
    const q = local(2.2, 1.1), { body, parts } = hand(q), px = fwidth(q.x);
    const tone = smoothstep(-0.35, 0.3, q.y).mul(0.18).add(0.8);
    let color: N<'vec3'> = rgb(IVORY).mul(tone);
    for (const part of parts) color = mix(color, rgb(INK), cover(abs(part).sub(0.012), px).mul(cover(body.add(0.02), px)).mul(0.8));
    color = mix(rgb(INK), color, cover(body.add(0.035), px));
    const handMesh = sprite(flat(2.2, 1.1), 9, color, cover(body, px));
    handMesh.position.x = ARM;
    handMesh.scale.setScalar(HAND);
    this.palm = handMesh;
    const s = local(6.2, 1.1), sleeveD = sdBox(s, vec2(3.0, 0.42), 0.14), cuffD = sdBox(s.sub(vec2(2.72, 0)), vec2(0.28, 0.47), 0.1);
    const folds = sin(s.x.mul(3.1).add(sin(s.y.mul(5)).mul(0.9))).mul(0.5).add(0.5);
    const roll = smoothstep(-0.42, 0.3, s.y).mul(0.55).add(0.55);
    let sleeve: N<'vec3'> = rgb(COAT).mul(folds.mul(0.35).add(0.75)).mul(roll);
    const sp = fwidth(s.x);
    const trim = cover(abs(s.x.sub(2.5)).sub(0.035), sp).mul(0.8);
    sleeve = mix(sleeve, rgb(accent).mul(roll.mul(0.7).add(0.35)), cover(cuffD, sp));
    sleeve = mix(sleeve, rgb(INK), trim.mul(cover(cuffD, sp)));
    sleeve = mix(rgb(INK), sleeve, cover(min(sleeveD, cuffD).add(0.03), sp));
    const sleeveMesh = sprite(flat(6.2, 1.1), 8, sleeve, cover(min(sleeveD, cuffD), sp));
    sleeveMesh.position.x = ARM - 3.52;
    const shadow = sprite(flat(8.6, 1.4), 3, rgb(0x000000), smoothstep(0.6, 0.1, abs(local(8.6, 1.4).y)).mul(smoothstep(-4.3, -3.2, local(8.6, 1.4).x)).mul(smoothstep(4.3, 3.6, local(8.6, 1.4).x)).mul(0.4));
    shadow.position.set(ARM - 3.3, 0, 0.3);
    group.add(shadow, sleeveMesh, handMesh, block);
    group.position.set(PIVOT, 0, 0);
    return group;
  }

  private place(mesh: THREE.Mesh, x: number, y: number): THREE.Mesh {
    mesh.position.set(x, 0, -y);
    return mesh;
  }

  private setCoin(coin: Coin, x: number, y: number, lift: number, squash: number, side: number, flash: number, over: boolean): void {
    coin.x = x;
    coin.y = y;
    const size = 1 + 0.28 * lift;
    coin.mesh.position.set(x, 0, -y);
    coin.mesh.scale.set(size * Math.max(0.06, squash), 1, size);
    coin.mesh.renderOrder = over || lift > 0 ? 7 : coin === this.gold ? 6 : 5;
    coin.shadow.position.set(x + 0.16 + 0.22 * lift, 0, -y + 0.2 + 0.26 * lift);
    coin.shadow.scale.set(size * Math.max(0.2, squash), 1, size);
    coin.side.value = side;
    coin.flash.value = flash;
    coin.shade.value = 1 + 0.12 * lift;
  }

  private fillLanes(i: number, pending: boolean, at: (s: number) => EnvFrame | null): void {
    const t = this.slots;
    let prev = at(i - SLOTS);
    for (let j = SLOTS; j >= 0; j--) {
      const s = i + 1 - j, cur = j > 0 || pending ? at(s) : null, o = j * 4;
      if (s >= 1 && cur && prev) {
        t[o] = coinSwapLoaded(prev);
        t[o + 1] = coinSwapToss(cur);
        t[o + 2] = cur.reward > 0 ? 1 : 0;
        t[o + 3] = 1;
      } else t.fill(0, o, o + 4);
      prev = cur;
    }
    this.lanes.needsUpdate = true;
  }

  private fillLedger(i: number, at: (s: number) => EnvFrame | null): void {
    const size = this.bin, newest = Math.floor(i / size), t = this.columns;
    let matched = 0, count = 0;
    for (let c = 0; c < BINS; c++) {
      const b = newest - c, complete = b * size + size - 1 <= i, o = c * 4;
      let bin = complete ? this.bins.get(b) ?? null : null;
      if (!bin) {
        bin = b < 0 ? null : binOf(b, size, i, at);
        if (bin && complete) this.bins.set(b, bin);
      }
      if (bin && bin.count > 0) {
        t[o] = bin.tossedZero / bin.count;
        t[o + 1] = bin.loadedZero / bin.count;
        t[o + 2] = bin.count / size;
        t[o + 3] = 1;
        matched += bin.matched;
        count += bin.count;
      } else t.fill(0, o, o + 4);
    }
    for (const b of this.bins.keys()) if (b > newest || b < newest - BINS - 8) this.bins.delete(b);
    this.ledgerTexture.needsUpdate = true;
    this.ledger.share = count > 0 ? matched / count : NaN;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const gold = this.gold, pewter = this.pewter, arm = this.arm, palm = this.palm, flyer = this.flyer;
    const ring = this.ring;
    if (!gold || !pewter || !arm || !palm || !flyer || !ring || !this.goldGlow || !this.flashGlow || !this.pewterBlock) return;
    const dt = frame.dtReal;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = (s: number): EnvFrame | null => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    const i = frame.state.step, head = frame.frameAt(i), fast = this.pace > this.controlHz * FAST;
    const next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    this.fillLanes(i, next !== null, at);
    this.fillLedger(i, at);

    const loaded = coinSwapLoaded(head), toss = i >= 1 ? coinSwapToss(head) : -1;
    let goldY = laneY(loaded), pewterY = laneY(1 - loaded), aimY = toss < 0 ? 0 : laneY(toss);
    let lift = 0, squash = 1, side = 0, flash = 0, flipper: Coin | null = null, jab = 0;
    this.scroll.value = 0;
    this.arrived.value = 0;
    flyer.visible = false;
    if (next) {
      const nextToss = coinSwapToss(next), paid = next.reward > 0, after = coinSwapLoaded(next);
      aimY = (toss < 0 ? 0 : laneY(toss)) + (laneY(nextToss) - (toss < 0 ? 0 : laneY(toss))) * smoothstep01(span(t, AIM));
      const u = span(t, FLIP);
      flipper = nextToss === loaded ? gold : pewter;
      if (u > 0 && u < 1) {
        lift = Math.sin(Math.PI * u);
        const angle = 2 * Math.PI * SPINS * u;
        squash = Math.abs(Math.cos(angle));
        side = Math.cos(angle) < 0 ? 1 : 0;
        jab = Math.sin(Math.PI * Math.min(1, u * 3));
      }
      if (paid) flash = (1 - span(t, GLOW)) ** 3 * (t >= GLOW[0] ? 1 : 0) * clamp01((1.5 * this.controlHz) / Math.max(this.pace, this.controlHz));
      const v = span(t, DROP);
      this.scroll.value = smoothstep01(v);
      this.arrived.value = t >= DROP[1] ? 1 : 0;
      if (t >= DROP[0] && t < DROP[1]) {
        flyer.visible = true;
        this.flyerPaid.value = paid ? 1 : 0;
        const e = 1 - (1 - v) ** 3, hop = 1 + 0.35 * Math.sin(Math.PI * v);
        flyer.position.set(0.7 + (X0 - 0.7) * e, 0, -laneY(nextToss));
        flyer.scale.setScalar(hop);
      }
      if (after !== loaded && t >= TRADE[0]) {
        goldY = laneY(loaded) + (laneY(after) - laneY(loaded)) * smoothstep01(span(t, TRADE));
        pewterY = -goldY;
      }
    }
    if (fast || !next) {
      const k = frame.scrubbed ? 1 : 1 - Math.exp(-GLIDE * dt);
      goldY = gold.y + (goldY - gold.y) * k;
      pewterY = pewter.y + (pewterY - pewter.y) * k;
      aimY = this.aim + (aimY - this.aim) * k;
    }
    this.aim = aimY;
    const travel = 1 - (goldY / LANE) ** 2;
    this.setCoin(gold, BOW * travel, goldY, flipper === gold ? lift : 0, flipper === gold ? squash : 1, flipper === gold ? side : 0, flipper === gold ? flash : 0, travel > 1e-3);
    this.setCoin(pewter, -BOW * (1 - (pewterY / LANE) ** 2), pewterY, flipper === pewter ? lift : 0, flipper === pewter ? squash : 1, flipper === pewter ? side : 0, flipper === pewter ? flash : 0, false);
    arm.rotation.y = Math.atan2(aimY, -PIVOT);
    palm.position.x = ARM + 0.14 * jab;

    const gain = 3.4 * (1 + 0.25 * gold.flash.value), color = this.tint.set(GOLD);
    this.goldLight.value.set(color.r * gain, color.g * gain, color.b * gain);
    this.goldGlow.position.set(gold.x, 0, -gold.y);
    this.pewterBlock.position.set(pewter.x, 0, -pewter.y);
    const burst = flipper ? flipper.flash.value * 2.4 : 0;
    color.set(PAID);
    this.flashLight.value.set(color.r * burst, color.g * burst, color.b * burst);
    if (flipper) this.flashGlow.position.set(flipper.x, 0, -flipper.y);

    if (frame.scrubbed) {
      pool.clear();
      this.ringAt = -Infinity;
    } else if (frame.events.some((event) => event.type === 'cue')) {
      pool.burst(this.point.set(gold.x, 0.2, -gold.y), PAID, 0.5, time);
      this.ringAt = time;
    }
    const age = (time - this.ringAt) / RING_SECONDS;
    ring.visible = age >= 0 && age < 1;
    if (ring.visible) {
      this.ringRadius.value = 1.25 + 2.1 * (1 - (1 - age) ** 3);
      this.ringFade.value = (1 - age) ** 2;
    }

    this.ledger.step = i;
    this.ledger.toss = toss;
    this.ledger.reward = head.reward;
    this.ledger.loaded = loaded;
    if (this.settled !== i) {
      const log = frame.eventLog, found = lastEvent(log, 'cue', i);
      this.ledger.known = eventsKnown(log, found ?? 0, i);
      this.ledger.since = found === undefined ? null : i - found;
      if (this.ledger.known) this.settled = i;
    }
  }

  protected override drop(): void {
    this.lanes.dispose();
    this.ledgerTexture.dispose();
    this.bins.clear();
  }
}

export default (() => new CoinSwapScene()) satisfies SceneFactory;
