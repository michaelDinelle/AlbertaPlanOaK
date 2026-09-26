import * as THREE from 'three/webgpu';
import { abs, clamp, float, fract, length, max, mix, smoothstep, step, time, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { garnetAction, garnetBest, garnetEpoch, garnetNext, garnetProb, garnetReward, garnetState, type GarnetShape } from '@sim/envs/garnet';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';

type Span = readonly [number, number];
type At = (s: number) => EnvFrame | null;
type V2 = THREE.Node<'vec2'>;

const RING = 5;
const HISTORY = 48;
const WINDOW = 512;
const FAST = 5;
const GLIDE = 14;
const CHOOSE: Span = [0, 0.3];
const TRAVEL: Span = [0.3, 0.88];
const FLARE = 1.2;
const BACK = 0x0a0507;
const PLATE = 0x160b10;
const GEM_COLD = 0x4a0a1a;
const GEM_HOT = 0xff3a5e;
const ORBIT = 0x5c2a38;
const TINTS = [0x5cc8ff, 0xffb14f, 0xb58bff, 0x62f0a0];
const PANEL_X0 = 6.8;
const PANEL_X1 = 12.2;
const ZERO_X = 9.9;
const SPAN_X = 1.9;
const ROW_Y = 4.9;
const ROW_H = 1.05;
const HIST_Y = -3.4;
const BRANCH_Y = 0.2;
const HIST_H = 1.5;

const ease = (t: number, [a, b]: Span) => smoothstep01(clamp01((t - a) / (b - a)));
const PLANE = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
const quad = (): V2 => uv().sub(0.5).mul(2) as unknown as V2;
const hex = (p: V2, r: number) => {
  const q = abs(p);
  return max(q.x.mul(0.866).add(q.y.mul(0.5)), q.y).sub(r);
};

class Layer {
  readonly mesh: THREE.InstancedMesh;
  private n = 0;
  private readonly m = new THREE.Matrix4();
  private readonly q = new THREE.Quaternion();
  private readonly p = new THREE.Vector3();
  private readonly s = new THREE.Vector3();
  private readonly c = new THREE.Color();
  private static readonly UP = new THREE.Vector3(0, 1, 0);

  constructor(order: number, capacity: number, alpha: THREE.Node<'float'>, shade?: THREE.Node<'vec3'>) {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, depthTest: false });
    material.opacityNode = alpha;
    if (shade) material.colorNode = shade;
    this.mesh = instancedMesh(PLANE, material, capacity);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = order;
    this.mesh.count = 0;
  }

  put(x: number, y: number, angle: number, sx: number, sy: number, color: THREE.ColorRepresentation, k = 1): void {
    if (this.n >= this.mesh.instanceMatrix.count) return;
    this.mesh.setMatrixAt(this.n, this.m.compose(this.p.set(x, 0, -y), this.q.setFromAxisAngle(Layer.UP, angle), this.s.set(sx, 1, sy)));
    this.mesh.setColorAt(this.n, this.c.set(color).multiplyScalar(k));
    this.n++;
  }

  rect(x0: number, y0: number, x1: number, y1: number, color: THREE.ColorRepresentation, k = 1): void {
    this.put((x0 + x1) / 2, (y0 + y1) / 2, 0, Math.max(Math.abs(x1 - x0), 1e-4), Math.max(Math.abs(y1 - y0), 1e-4), color, k);
  }

  flush(): void {
    this.mesh.count = this.n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.n = 0;
  }
}

function gemLayer(capacity: number): Layer {
  const p = quad(), body = hex(p, 0.86), table = coverage(hex(p, 0.46));
  const crown = float(0.5).add(step(0, p.y).mul(0.22)).add(step(0, p.x.negate()).mul(0.1));
  const rim = coverage(body.add(0.12)).oneMinus().mul(0.3);
  const glint = coverage(length(p.sub(vec2(-0.2, 0.24))).sub(0.1)).mul(0.6);
  return new Layer(3, capacity, coverage(body), vec3(mix(crown, p.y.mul(0.15).add(0.95), table).sub(rim).add(glint)));
}

function edgeLayer(capacity: number): Layer {
  const p = quad(), flow = smoothstep(0, 0.3, abs(fract(p.x.mul(0.9).sub(time.mul(0.6))).sub(0.5))).oneMinus();
  return new Layer(1, capacity, coverage(abs(p.y).sub(0.8)), vec3(flow.mul(0.35).add(0.65).mul(p.x.mul(0.15).add(0.85))));
}

function headLayer(capacity: number): Layer {
  const p = quad(), side = p.x.sub(1).mul(0.8).add(abs(p.y).mul(1.8)).div(1.97);
  return new Layer(2, capacity, coverage(max(side, p.x.negate().sub(0.8))));
}

function ringLayer(order: number, capacity: number, thickness: number): Layer {
  const p = quad();
  return new Layer(order, capacity, coverage(abs(length(p).sub(0.85)).sub(thickness)));
}

function glowLayer(order: number, capacity: number): Layer {
  const r = length(quad()), core = coverage(r.sub(0.36)), halo = clamp(r.oneMinus(), 0, 1);
  return new Layer(order, capacity, max(core, halo.mul(halo).mul(0.6)), vec3(mix(float(0.75), float(1.25), core)));
}

function rectLayer(order: number, capacity: number): Layer {
  return new Layer(order, capacity, float(1));
}

class GarnetScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.35, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'garnet';

  private shape: GarnetShape = { states: 1, actions: 1, branching: 1 };
  private resample = false;
  private period = 0;
  private nodeR = 0.5;
  private accent = 0xffffff;
  private layers: Layer[] = [];
  private plate!: Layer;
  private orbit!: Layer;
  private halos!: Layer;
  private edges!: Layer;
  private heads!: Layer;
  private loops!: Layer;
  private gems!: Layer;
  private token!: Layer;
  private bars!: Layer;
  private counts = new Float32Array(1);
  private seen = 0;
  private heatStep = -1;
  private pace = 0;
  private lastPosition = 0;
  private lastStep = -1;
  private flareAt = -Infinity;
  private readonly glide = new THREE.Vector2(NaN, NaN);
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, state: 0, action: -1, reward: 0, land: -1, prob: 0, share: NaN, epoch: 0, rewards: [] as number[], scale: 1 };

  override legend: LegendSpec | null = null;

  private legendFor(): LegendSpec {
    const l = this.ledger, sh = this.shape;
    const actions = Array.from({ length: sh.actions }, (_, a) => ({
      key: `a${a}`,
      label: `action ${a}${l.action === a ? ' (taken)' : ''}`,
      tint: TINTS[a % TINTS.length]!,
      value: () => (l.rewards[a] === undefined ? '—' : `r ${formatSigned(l.rewards[a])}`),
    }));
    return {
      title: 'GARNET',
      rows: [
        { key: 'state', label: 'state', tint: GEM_HOT, value: () => `${l.state} of ${sh.states}` },
        ...actions,
        { key: 'taken', label: 'action taken', tint: this.accent, value: () => (l.action < 0 ? '—' : `action ${l.action}`) },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
        { key: 'branch', label: 'landed', tint: this.accent, value: () => (l.land < 0 ? '—' : `state ${l.land}, p ${l.prob.toFixed(2)}`) },
        { key: 'share', label: `time here, last ${formatCount(WINDOW)}`, tint: GEM_HOT, value: () => formatPercent(l.share) },
        { key: 'model', label: 'model', tint: ORBIT, value: () => (this.resample ? `draw ${l.epoch}` : 'fixed') },
      ],
      note: () => `${sh.states} states · ${sh.actions} actions · ${sh.branching} branches${this.resample ? ` · redrawn every ${formatCount(this.period)} steps` : ''}`,
      bar: () => clamp01((l.reward / l.scale + 1) / 2),
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.shape = { states: parameter(env, 'states'), actions: parameter(env, 'actions'), branching: parameter(env, 'branching') };
    this.resample = parameter(env, 'resample') > 0;
    this.period = constant(env, 'GARNET_PERIOD');
    this.nodeR = Math.min(0.68,(0.38 * 2 * Math.PI * RING) / this.shape.states);
    this.counts = new Float32Array(this.shape.states);
    return new THREE.Box2(new THREE.Vector2(-RING - 1.3, -RING - 1.3), new THREE.Vector2(PANEL_X1 + 0.3, RING + 1.3));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  private pos(s: number, out: THREE.Vector2): THREE.Vector2 {
    const a = Math.PI / 2 - (2 * Math.PI * s) / this.shape.states;
    return out.set(RING * Math.cos(a), RING * Math.sin(a));
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx), v = new THREE.Vector2();
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, head = frame.frameAt(i), sh = this.shape;
        if (head.step !== i) return;
        const s = garnetState(head);
        this.pos(s, v);
        kit.marks('state', 1).set(v.x, 0.1, -v.y, this.nodeR * 0.6, TOKENS.uiCyan);
        const next = frame.frameAt(i + 1);
        if (next.step !== i + 1) return;
        const a = garnetAction(next, sh), pair = s * sh.actions + a, fan = kit.marks('branches', sh.branching);
        for (let b = 0; b < sh.branching; b++) {
          this.pos(garnetNext(next, sh, pair, b), v);
          fan.set(v.x, 0.1, -v.y, this.nodeR * (0.3 + 0.9 * garnetProb(next, sh, pair, b)), accent);
        }
        this.pos(garnetState(next), v);
        kit.marks('landed', 1).set(v.x, 0.1, -v.y, this.nodeR * 0.25, TOKENS.cueMagenta);
        this.pos(Math.floor(garnetBest(next, sh) / sh.actions), v);
        kit.marks('best', 1).set(v.x, 0.1, -v.y, this.nodeR * 0.2, TOKENS.celebrationGold);
      },
    };
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    const sh = this.shape, fan = sh.actions * sh.branching;
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor();
    this.scene.background = new THREE.Color(BACK);
    pool.group.renderOrder = 10;
    this.plate = rectLayer(0, 4);
    this.orbit = ringLayer(0, 1, 0.0035);
    this.halos = glowLayer(2, 3);
    this.edges = edgeLayer(fan);
    this.heads = headLayer(fan);
    this.loops = ringLayer(2, fan, 0.09);
    this.gems = gemLayer(sh.states);
    this.token = glowLayer(5, 1);
    this.bars = rectLayer(1, 2 * sh.actions + HISTORY + 4);
    this.layers = [this.plate, this.orbit, this.halos, this.edges, this.heads, this.loops, this.gems, this.token, this.bars];
    for (const layer of this.layers) this.scene.add(layer.mesh);
  }

  protected override drop(): void {
    for (const layer of this.layers) (layer.mesh.material as THREE.Material).dispose();
    this.layers = [];
  }

  private reader(frame: FrameUpdate): At {
    return (s) => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
  }

  private heat(i: number, at: At): void {
    if (i === this.heatStep) return;
    this.counts.fill(0);
    let n = 0;
    for (let s = Math.max(0, i - WINDOW + 1); s <= i; s++) {
      const f = at(s);
      if (!f) continue;
      this.counts[garnetState(f)]!++;
      n++;
    }
    this.seen = n;
    this.heatStep = i;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal, sh = this.shape, nr = this.nodeR, K = sh.actions * sh.branching;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, head = frame.frameAt(i), fast = this.pace > this.controlHz * FAST;
    const next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    const model = next ?? head, s = garnetState(head);
    const choose = next ? ease(t, CHOOSE) : 0, travel = next ? ease(t, TRAVEL) : 0;
    const act = next ? garnetAction(next, sh) : -1, land = next ? garnetState(next) : s;
    this.heat(i, at);

    let scale = 1e-6;
    for (let p = 0; p < sh.states * sh.actions; p++) scale = Math.max(scale, Math.abs(garnetReward(model, sh, p)));
    const flare = Math.max(0, 1 - (time - this.flareAt) / FLARE);

    const plate = this.plate, bars = this.bars;
    plate.rect(PANEL_X0, ROW_Y + 0.55, PANEL_X1, ROW_Y - (sh.actions - 1) * ROW_H - 0.55, PLATE);
    plate.rect(PANEL_X0, HIST_Y + HIST_H + 0.3, PANEL_X1, HIST_Y - HIST_H - 0.3, PLATE);
    plate.rect(PANEL_X0, BRANCH_Y + 0.5, PANEL_X1, BRANCH_Y - 0.5, PLATE);
    plate.flush();
    this.orbit.put(0, 0, 0, (2 * RING) / 0.85, (2 * RING) / 0.85, ORBIT, 1 + 2.5 * flare);
    this.orbit.flush();

    const a0 = new THREE.Vector2(), a1 = new THREE.Vector2(), c = new THREE.Color(), tok = new THREE.Vector2();
    let peak = 1;
    for (let q = 0; q < sh.states; q++) peak = Math.max(peak, this.counts[q]!);
    for (let q = 0; q < sh.states; q++) {
      const h = Math.sqrt(this.counts[q]! / peak), r = nr * (0.82 + 0.18 * h);
      this.pos(q, a0);
      this.gems.put(a0.x, a0.y, 0, 2 * r, 2 * r, c.set(GEM_COLD).lerp(new THREE.Color(GEM_HOT), h), 0.55 + 0.45 * h);
    }
    this.gems.flush();

    this.pos(s, a0);
    this.halos.put(a0.x, a0.y, 0, 4.2 * nr, 4.2 * nr, this.accent, 0.55 * (1 - travel));
    if (next) {
      this.pos(land, a1);
      this.halos.put(a1.x, a1.y, 0, 4.2 * nr, 4.2 * nr, this.accent, 0.55 * travel);
    }
    this.halos.flush();

    const actPair = s * sh.actions + Math.max(act, 0);
    let realized = -1;
    if (next) for (let b = 0; b < sh.branching && realized < 0; b++) if (garnetNext(model, sh, actPair, b) === land) realized = act * sh.branching + b;
    const ox = a0.x / RING, oy = a0.y / RING;
    let loopIndex = 0;
    tok.set(a0.x, a0.y);
    for (let a = 0; a < sh.actions; a++) {
      const emphasis = act < 0 ? 0.5 : a === act ? 0.5 + 0.5 * choose : 0.5 * (1 - 0.65 * choose), tint = TINTS[a % TINTS.length]!;
      for (let b = 0; b < sh.branching; b++) {
        const k = a * sh.branching + b, pair = s * sh.actions + a, target = garnetNext(model, sh, pair, b), prob = garnetProb(model, sh, pair, b);
        const glow = emphasis * (0.45 + 0.55 * prob);
        if (target === s) {
          const rr = nr * (0.5 + 0.22 * loopIndex++), cx = a0.x + ox * (nr * 0.75 + rr), cy = a0.y + oy * (nr * 0.75 + rr);
          this.loops.put(cx, cy, 0, (2 * rr) / 0.85, (2 * rr) / 0.85, tint, glow);
          if (k === realized) {
            const start = Math.atan2(-oy, -ox) + 2 * Math.PI * travel;
            tok.set(cx + rr * Math.cos(start), cy + rr * Math.sin(start));
          }
          continue;
        }
        this.pos(target, a1);
        const dx = a1.x - a0.x, dy = a1.y - a0.y, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L, off = (k - (K - 1) / 2) * nr * 0.24;
        const bx = a0.x - uy * off, by = a0.y + ux * off, ex = a1.x - uy * off, ey = a1.y + ux * off;
        const w = nr * (0.07 + 0.3 * prob), hl = nr * 0.55, hw = Math.max(w * 2.4, nr * 0.34);
        const s0 = nr * 0.95, s1 = L - nr * 1.05 - hl * 0.6, angle = Math.atan2(dy, dx);
        if (s1 > s0) this.edges.put(bx + ux * (s0 + s1) / 2, by + uy * (s0 + s1) / 2, angle, s1 - s0, w / 0.8, tint, glow);
        const hc = L - nr * 1.05 - hl / 2;
        this.heads.put(bx + ux * hc, by + uy * hc, angle, hl, hw, tint, glow);
        if (k === realized) tok.set(bx + (ex - bx) * travel, by + (ey - by) * travel);
      }
    }
    this.edges.flush();
    this.heads.flush();
    this.loops.flush();

    if (!next) {
      const g = this.glide, k = frame.scrubbed || Number.isNaN(g.x) ? 1 : 1 - Math.exp(-GLIDE * dt);
      g.set(g.x + (tok.x - g.x) * k, g.y + (tok.y - g.y) * k);
      if (Number.isNaN(g.x)) g.copy(tok);
      tok.copy(g);
    } else this.glide.copy(tok);
    const tint = act >= 0 && realized >= 0 ? TINTS[act % TINTS.length]! : 0xffffff;
    this.token.put(tok.x, tok.y, 0, nr * 1.5, nr * 1.5, c.set(0xffffff).lerp(new THREE.Color(tint), 0.35), 1);
    this.token.flush();

    const rewards: number[] = [];
    for (let a = 0; a < sh.actions; a++) {
      const y = ROW_Y - a * ROW_H, r = garnetReward(model, sh, s * sh.actions + a), len = (r / scale) * SPAN_X, tint = TINTS[a % TINTS.length]!;
      const on = act < 0 ? 0.6 : a === act ? 0.6 + 0.4 * choose : 0.6 * (1 - 0.5 * choose);
      rewards.push(r);
      bars.rect(PANEL_X0 + 0.3, y - 0.24, PANEL_X0 + 0.78, y + 0.24, tint, on);
      bars.rect(ZERO_X, y - 0.23, ZERO_X + len, y + 0.23, tint, on);
    }
    bars.rect(ZERO_X - 0.015, ROW_Y + 0.4, ZERO_X + 0.015, ROW_Y - (sh.actions - 1) * ROW_H - 0.4, 0xffffff, 0.35);
    if (act >= 0) {
      const u0 = PANEL_X0 + 0.3, span = PANEL_X1 - 0.3 - u0, tint = TINTS[act % TINTS.length]!;
      let cum = 0;
      for (let b = 0; b < sh.branching; b++) {
        const pr = garnetProb(model, sh, actPair, b), l0 = u0 + cum * span, l1 = l0 + pr * span, hit = act * sh.branching + b === realized;
        bars.rect(l0 + 0.03, BRANCH_Y - 0.22, l1 - 0.03, BRANCH_Y + 0.22, tint, (hit ? 0.35 + 0.65 * travel : 0.3 + 0.15 * (b % 2)) * (0.4 + 0.6 * choose));
        if (hit) bars.rect(l0 + 0.03, BRANCH_Y - 0.36, l1 - 0.03, BRANCH_Y - 0.3, TOKENS.celebrationGold, travel);
        cum += pr;
      }
    }
    const x0 = PANEL_X0 + 0.3, x1 = PANEL_X1 - 0.3, bw = (x1 - x0) / HISTORY;
    bars.rect(x0, HIST_Y - 0.01, x1, HIST_Y + 0.01, 0xffffff, 0.3);
    for (let m = 0; m < HISTORY; m++) {
      const f = at(i - m);
      if (!f || f.step < 1) continue;
      const r = f.reward, x = x1 - (m + 0.5) * bw, hgt = (r / scale) * HIST_H;
      bars.rect(x - bw * 0.34, HIST_Y, x + bw * 0.34, HIST_Y + hgt, r >= 0 ? TOKENS.celebrationGold : TOKENS.failureBlue, m === 0 ? 1 : 0.4 + 0.4 * (1 - m / HISTORY));
    }
    bars.flush();

    this.celebrate(frame, i, at, time);
    const l = this.ledger;
    l.step = i;
    l.state = s;
    l.action = i >= 1 ? garnetAction(head, sh) : -1;
    l.reward = head.reward;
    l.rewards = rewards;
    l.scale = scale;
    l.epoch = garnetEpoch(head);
    l.share = this.seen > 0 ? this.counts[s]! / this.seen : NaN;
    const before = at(i - 1);
    if (before && i >= 1) {
      const pair = garnetState(before) * sh.actions + garnetAction(head, sh);
      l.land = s;
      l.prob = 0;
      for (let b = 0; b < sh.branching; b++) if (garnetNext(head, sh, pair, b) === s) l.prob += garnetProb(head, sh, pair, b);
    } else l.land = -1;
  }

  private celebrate(frame: FrameUpdate, i: number, at: At, time: number): void {
    const pool = this.pool;
    if (!pool) return;
    if (frame.scrubbed || i < this.lastStep) {
      pool.clear();
      this.flareAt = -Infinity;
      this.lastStep = i;
      return;
    }
    if (i === this.lastStep) return;
    this.lastStep = i;
    const v = new THREE.Vector2();
    for (const event of frame.events) {
      if (event.type === 'cue') {
        this.flareAt = time;
        pool.ripple(this.point.set(0, 0, 0), this.accent, RING * 0.8, time);
      } else if (event.type === 'goal') {
        const before = at(i - 1);
        if (!before) continue;
        this.pos(garnetState(before), v);
        pool.burst(this.point.set(v.x, 40, -v.y), TOKENS.celebrationGold, 0.6, time);
      }
    }
  }
}

export default (() => new GarnetScene()) satisfies SceneFactory;
