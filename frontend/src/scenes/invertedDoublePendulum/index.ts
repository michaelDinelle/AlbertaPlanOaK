import * as THREE from 'three/webgpu';
import { max, uniform } from 'three/tsl';
import type { DebugCurve, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { PENDULUM_CART, PENDULUM_HINGE, PENDULUM_HINGE2, invertedDoublePendulumDecoder } from '@sim/envs/planar';
import { constantRecord } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { BEAM, BLOCK, CARRIAGE, HUB, LED, LINK, PAD, PALETTE, ROD, board, bumper, carriage, flat, ghost, hub, led, link, push, rail, ribbon, sparks, sprite, type Level, type RailSpan, type Ribbon, type Sparks } from './sprites';

const PACE = 10;
const TRAIL = 240;
const SUB = 4;
const POINTS = TRAIL * SUB + 1;
const GHOST_STEPS = 2;
const GHOST_HOLD = 5;
const LONG = 24;
const RESETS = 35;
const CONSTRAINT = 8;
const WIDTH = 0.011;

const turn = (from: number, to: number): number => {
  const d = to - from;
  return d - 2 * Math.PI * Math.round(d / (2 * Math.PI));
};
const side = (value: number): string => Math.abs(value) < 0.005 ? 'centred' : `${Math.abs(value).toFixed(2)} m ${value < 0 ? 'left' : 'right'}`;
const signed = (value: number): string => formatSigned(value).replace('-', '−');

function circle(curve: DebugCurve, x: number, y: number, radius: number): void {
  const count = curve.capacity - 1;
  for (let i = 0; i <= count; i++) {
    const a = 2 * Math.PI * i / count;
    curve.point(x + radius * Math.cos(a), 0.05, -(y + radius * Math.sin(a)));
  }
}

class LongExposureScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.6, fog: { from: ['night', 'fog'] } };
  readonly controlHz = PACE;
  protected readonly stageName = 'longExposure';

  private pole = 1;
  private reach = 2;
  private range = 1;
  private margin = 0;
  private gear = 1;
  private span: RailSpan = { range: 1, face: 1, outer: 1, end: 1 };
  private readonly rig = new THREE.Group();
  private readonly cart = new THREE.Group();
  private readonly lower = new THREE.Group();
  private readonly upper = new THREE.Group();
  private readonly elbow = new THREE.Group();
  private readonly tip = new THREE.Group();
  private readonly shade = new THREE.Group();
  private trail: Ribbon | null = null;
  private embers: Sparks | null = null;
  private readonly force = uniform(0);
  private readonly flow = uniform(0);
  private readonly squeeze: readonly [Level, Level] = [uniform(0), uniform(0)];
  private readonly flare: readonly [Level, Level] = [uniform(0), uniform(0)];
  private readonly fade = uniform(0);
  private readonly ghostElbow = uniform(new THREE.Vector2());
  private readonly ghostTip = uniform(new THREE.Vector2());

  private readonly pose = new Float32Array(3);
  private readonly xs = new Float64Array(TRAIL + 2);
  private readonly lows = new Float64Array(TRAIL + 2);
  private readonly highs = new Float64Array(TRAIL + 2);
  private readonly episodes = new Uint32Array(TRAIL + 2);
  private readonly resets = new Uint8Array(TRAIL + 2);
  private readonly valid = new Uint8Array(TRAIL + 2);
  private readonly tips = new Float64Array(POINTS * 2);
  private readonly owners = new Uint32Array(POINTS);
  private readonly mark = new Float64Array(2);
  private readonly matrix = new THREE.Matrix4();
  private readonly spot = new THREE.Vector3();
  private weight = 0;
  private readonly ledger = { height: 0, x: 0, push: 0, reward: 0, attempt: 1, held: 0 };

  private forward(x: number, lower: number, upper: number, out: Float64Array, at: number): void {
    out[at] = x + this.pole * (Math.sin(lower) + Math.sin(upper));
    out[at + 1] = this.pole * (Math.cos(lower) + Math.cos(upper));
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const cart = constantRecord(ctx.environment, 'PLANAR_CART');
    this.pole = cart.pole as number;
    this.reach = 2 * this.pole;
    this.range = cart.range as number;
    this.margin = cart.margin as number;
    this.gear = cart.gear as number;
    const face = this.range - this.margin + CARRIAGE.half, outer = face + PAD.width;
    this.span = { range: this.range, face, outer, end: outer + BLOCK.width };
    const wide = this.span.end + 0.42;
    return new THREE.Box2(new THREE.Vector2(-wide, -this.reach - 0.13), new THREE.Vector2(wide, 0.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x9a94c4).multiplyScalar(0.62);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, red = 0xff6b57, slate = 0x8fa6c4, amber = PALETTE.warn;
    const end = new Float64Array(2);
    return {
      debug: (frame, kit) => {
        const { range, margin, reach } = this, pose = frame.state.pose, step = frame.state.step;
        const x = pose[PENDULUM_CART]!, lower = pose[PENDULUM_HINGE]!, upper = lower + pose[PENDULUM_HINGE2]!;
        this.forward(x, lower, upper, end, 0);
        const jx = x + this.pole * Math.sin(lower), jy = this.pole * Math.cos(lower);
        const track = kit.curve('rail', 2, slate);
        track.point(-range, 0.05, 0);
        track.point(range, 0.05, 0);
        for (const [name, at, color] of [['limitLeft', -range, red], ['limitRight', range, red], ['softLeft', margin - range, amber], ['softRight', range - margin, amber]] as const) {
          const mark = kit.curve(name, 2, color);
          mark.point(at, 0.05, -0.12);
          mark.point(at, 0.05, 0.12);
        }
        const top = kit.curve('upright', 2, gold);
        top.point(x - 0.25, 0.05, -reach);
        top.point(x + 0.25, 0.05, -reach);
        circle(kit.curve('reach', 129, slate), x, 0, reach);
        const arm = kit.curve('links', 3, cyan);
        arm.point(x, 0.05, 0);
        arm.point(jx, 0.05, -jy);
        arm.point(end[0]!, 0.05, -end[1]!);
        const joints = kit.marks('joints', 3);
        joints.set(x, 0.05, 0, 0.02, cyan);
        joints.set(jx, 0.05, -jy, 0.018, cyan);
        joints.set(end[0]!, 0.05, -end[1]!, 0.018, gold);
        const action = Math.max(-1, Math.min(1, frame.frameAt(step + 1).actions[0]!)), arrow = kit.curve('push', 5, cyan), tipX = x + action * 0.4;
        if (Math.abs(action) > 0.01) for (const [px, pz] of [[x, 0.2], [tipX, 0.2], [tipX - Math.sign(action) * 0.04, 0.17], [tipX, 0.2], [tipX - Math.sign(action) * 0.04, 0.23]] as const) arrow.point(px, 0.05, pz);
        const constraint = frame.state.observations[CONSTRAINT]!, contact = kit.marks('constraint', 1);
        if (constraint !== 0) contact.set(Math.sign(x) * range, 0.05, 0, 0.02 + 0.004 * Math.abs(constraint), red);
        const path = kit.curve('attempt', TRAIL + 1, gold), falls = kit.marks('falls', TRAIL);
        let attempt = true;
        for (let s = step; s >= Math.max(0, step - TRAIL); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          invertedDoublePendulumDecoder.decode(f, this.pose);
          const a = this.pose[PENDULUM_HINGE]!;
          this.forward(this.pose[PENDULUM_CART]!, a, a + this.pose[PENDULUM_HINGE2]!, end, 0);
          if (attempt) path.point(end[0]!, 0.05, -end[1]!);
          if (f.reset) attempt = false;
          if (s < step && frame.frameAt(s + 1).reset) falls.set(end[0]!, 0.05, -end[1]!, 0.014, red);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit, span = this.span, { pole, reach } = this, far = 12;
    this.scene.background = new THREE.Color(PALETTE.void);
    this.scene.add(this.rig);
    this.rig.add(sprite(-far, -far, far, far, 0, board(lit, reach, accent)));
    const railTop = BLOCK.top + 0.01;
    this.rig.add(sprite(-span.end - 0.01, BEAM.bottom - 3.2, span.end + 0.01, railTop, 2, rail(lit, span, accent)));
    for (const [i, s] of [[0, -1], [1, 1]] as const) this.rig.add(sprite(s < 0 ? -span.outer - 0.01 : span.face - 0.01, -PAD.half - 0.01, s < 0 ? 0.01 - span.face : span.outer + 0.01, PAD.half + 0.01, 3, bumper(lit, span, s, this.squeeze[i], this.flare[i])));

    const occlude = (geometry: THREE.BufferGeometry, x: number, y: number, parent: THREE.Object3D): void => {
      const mesh = light.occluder(geometry);
      mesh.position.set(x, 0, -y);
      mesh.renderOrder = 1;
      parent.add(mesh);
    };
    occlude(flat(2 * span.outer, ROD * 1.1), 0, 0, this.rig);
    occlude(flat(2 * span.end, (BEAM.top - BEAM.bottom) * 0.55), 0, (BEAM.top + BEAM.bottom) / 2, this.rig);
    for (const s of [-1, 1]) occlude(flat(BLOCK.width * 0.6, (BLOCK.top - BEAM.bottom) * 0.8), s * (span.outer + BLOCK.width / 2), (BLOCK.top + BEAM.bottom) / 2, this.rig);

    const glow = (radius: number, tint: THREE.Node<'vec3'>, parent: THREE.Object3D): void => {
      const mesh = light.emitter(new THREE.CircleGeometry(radius, 24).rotateX(-Math.PI / 2), tint);
      mesh.renderOrder = 2;
      parent.add(mesh);
    };

    const trail = ribbon(POINTS, accent, 1.9, 4), embers = sparks(TRAIL, 0.085, 5);
    this.trail = trail;
    this.embers = embers;
    this.rig.add(trail.mesh, embers.mesh);

    this.shade.add(sprite(-reach - 0.1, -reach - 0.1, reach + 0.1, reach + 0.1, 6, ghost(this.ghostElbow, this.ghostTip, this.fade)));
    this.shade.visible = false;

    this.cart.add(sprite(-CARRIAGE.half - 0.01, BEAM.top - 0.01, CARRIAGE.half + 0.01, CARRIAGE.y + CARRIAGE.height + 0.01, 7, carriage(lit, accent)));
    this.cart.add(sprite(-CARRIAGE.half - 0.45, CARRIAGE.y - 0.05, CARRIAGE.half + 0.45, CARRIAGE.y + 0.05, 7.5, push(this.force, this.flow)));
    occlude(flat(2 * CARRIAGE.half * 0.72, CARRIAGE.height * 1.1), 0, CARRIAGE.y, this.cart);
    for (const s of [-1, 1]) {
      const thrust = light.emitter(flat(0.03, 0.05), rgb(PALETTE.push).mul(max(this.force.mul(s), 0).mul(9)));
      thrust.position.set(s * (CARRIAGE.half + 0.025), 0, -CARRIAGE.y);
      thrust.renderOrder = 2;
      this.cart.add(thrust);
    }
    for (const [i, s] of [[0, -1], [1, 1]] as const) {
      const flash = light.emitter(flat(0.018, PAD.half * 1.6), rgb(PALETTE.ember).mul(this.flare[i].mul(14)));
      flash.position.set(s * (span.face + 0.012), 0, 0);
      flash.renderOrder = 2;
      this.rig.add(flash);
    }

    this.lower.add(sprite(-LINK - 0.01, -LINK - 0.01, pole + LINK + 0.01, LINK + 0.01, 8, link(lit, pole)));
    this.upper.add(sprite(-LINK - 0.01, -LINK - 0.01, pole + LINK + 0.01, LINK + 0.01, 9, link(lit, pole)));
    this.cart.add(sprite(-HUB.pivot - 0.01, -HUB.pivot - 0.01, HUB.pivot + 0.01, HUB.pivot + 0.01, 10, hub(lit, HUB.pivot, accent)));
    this.elbow.add(sprite(-HUB.elbow - 0.01, -HUB.elbow - 0.01, HUB.elbow + 0.01, HUB.elbow + 0.01, 10, hub(lit, HUB.elbow, PALETTE.deep)));
    this.elbow.add(sprite(-LED.elbow - 0.005, -LED.elbow - 0.005, LED.elbow + 0.005, LED.elbow + 0.005, 11, led(LED.elbow, PALETTE.amber)));
    glow(LED.glow, rgb(PALETTE.amber).mul(12), this.elbow);
    this.tip.add(sprite(-HUB.tip - 0.01, -HUB.tip - 0.01, HUB.tip + 0.01, HUB.tip + 0.01, 10, hub(lit, HUB.tip, PALETTE.deep)));
    this.tip.add(sprite(-LED.tip - 0.005, -LED.tip - 0.005, LED.tip + 0.005, LED.tip + 0.005, 11, led(LED.tip, accent)));
    glow(LED.glow, rgb(accent).mul(24), this.tip);
    this.rig.add(this.shade, this.cart, this.lower, this.upper, this.elbow, this.tip);
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const ledger = this.ledger;
    this.legend = {
      title: 'LONG EXPOSURE',
      rows: [
        { key: 'tip', label: 'tip height', tint: accent, value: () => `${ledger.height.toFixed(2)} of ${reach.toFixed(2)} m` },
        { key: 'cart', label: 'cart', tint: PALETTE.steel, value: () => side(ledger.x) },
        { key: 'push', label: 'push', tint: PALETTE.push, value: () => Math.abs(ledger.push) < 0.5 ? 'coast' : ledger.push < 0 ? `← ${formatCount(-ledger.push)} N` : `${formatCount(ledger.push)} N →` },
        { key: 'reward', label: 'reward', tint: ctx.tokens.celebrationGold, value: () => signed(ledger.reward) },
        { key: 'attempt', label: 'attempt', tint: PALETTE.ember, value: () => `#${formatCount(ledger.attempt)} · ${formatCount(ledger.held)} steps` },
      ],
      note: () => `${formatCount(ledger.attempt - 1)} falls`,
      bar: () => ledger.height / reach,
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    if (!this.trail) return;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    const state = frame.state, pose = state.pose, step = state.step, { pole, range, margin } = this;
    const x = pose[PENDULUM_CART]!, lower = pose[PENDULUM_HINGE]!, upper = lower + pose[PENDULUM_HINGE2]!;
    const ex = x + pole * Math.sin(lower), ey = pole * Math.cos(lower), tx = ex + pole * Math.sin(upper), ty = ey + pole * Math.cos(upper);
    this.cart.position.set(x, 0, 0);
    this.lower.position.set(x, 0, 0);
    this.lower.rotation.y = Math.PI / 2 - lower;
    this.upper.position.set(ex, 0, -ey);
    this.upper.rotation.y = Math.PI / 2 - upper;
    this.elbow.position.set(ex, 0, -ey);
    this.tip.position.set(tx, 0, -ty);

    const action = Math.max(-1, Math.min(1, frame.frameAt(step + 1).actions[0]!));
    this.force.value = action;
    this.flow.value = time;
    const squeeze = Math.min(Math.max(0, Math.abs(x) - (range - margin)), PAD.width * 0.9), constraint = Math.abs(state.observations[CONSTRAINT]!), flare = constraint / (1 + constraint), left = x < 0;
    this.squeeze[0].value = left ? squeeze : 0;
    this.squeeze[1].value = left ? 0 : squeeze;
    this.flare[0].value = left ? flare : 0;
    this.flare[1].value = left ? 0 : flare;

    this.expose(frame);

    const fell = lastEvent(frame.eventLog, 'done', step);
    const calm = frame.prev !== null && step - frame.prev.step <= 2;
    for (const event of frame.events) if (event.type === 'done' && calm && this.shade.visible) pool.ripple(this.spot.set(this.shade.position.x + this.ghostTip.value.x, 0, this.shade.position.z - this.ghostTip.value.y), PALETTE.ember, 0.03 + 0.07 * this.weight, time);

    const ledger = this.ledger;
    ledger.height = ty;
    ledger.x = x;
    ledger.push = action * this.gear;
    ledger.reward = state.reward;
    ledger.attempt = frame.frameAt(step).words[RESETS]!;
    ledger.held = step - (fell ?? 0);
  }

  private expose(frame: FrameUpdate): void {
    const trail = this.trail, embers = this.embers;
    if (!trail || !embers) return;
    const head = frame.position, step = frame.state.step, first = Math.max(0, step - TRAIL);
    const { xs, lows, highs, episodes, resets, valid, pose } = this;
    let low = first;
    for (let k = step + 1; k >= first; k--) {
      const f = frame.frameAt(k), i = k - first;
      valid[i] = f.step === k ? 1 : 0;
      if (!valid[i]) {
        if (k > step) continue;
        low = k + 1;
        break;
      }
      invertedDoublePendulumDecoder.decode(f, pose);
      xs[i] = pose[PENDULUM_CART]!;
      lows[i] = pose[PENDULUM_HINGE]!;
      highs[i] = pose[PENDULUM_HINGE]! + pose[PENDULUM_HINGE2]!;
      episodes[i] = f.words[RESETS]!;
      resets[i] = f.reset ? 1 : 0;
    }

    let count = 0;
    for (let j = 0; j < POINTS; j++) {
      const tau = head - j / SUB;
      if (tau < low) break;
      const k = Math.min(step, Math.floor(tau)), i = k - first, next = k + 1 <= step + 1 && valid[i + 1] === 1 && resets[i + 1] === 0, u = next ? tau - k : 0;
      const x = xs[i]! + (next ? (xs[i + 1]! - xs[i]!) * u : 0);
      const a = lows[i]! + (next ? turn(lows[i]!, lows[i + 1]!) * u : 0);
      const b = highs[i]! + (next ? turn(highs[i]!, highs[i + 1]!) * u : 0);
      this.forward(x, a, b, this.tips, 2 * count);
      this.owners[count] = episodes[i]!;
      count++;
    }
    this.lay(trail, this.tips, count);

    const mark = this.mark;
    let lit = 0, latest = -1, prior = low;
    for (let k = step; k > low; k--) {
      const i = k - first;
      if (!resets[i] || !valid[i - 1]) continue;
      if (latest < 0) latest = k;
      else if (prior === low) prior = k;
      this.forward(xs[i - 1]!, lows[i - 1]!, highs[i - 1]!, mark, 0);
      const age = (head - k) / TRAIL, level = Math.exp(-4.2 * age) - Math.exp(-4.2) * age;
      embers.mesh.setMatrixAt(lit, this.matrix.makeTranslation(mark[0]!, 0, -mark[1]!));
      embers.level.setX(lit, level);
      lit++;
    }
    embers.mesh.count = lit;
    embers.mesh.instanceMatrix.needsUpdate = true;
    embers.level.needsUpdate = true;

    this.weight = latest < 0 ? 0 : 1 - Math.exp(-(latest - prior) / LONG);
    const age = latest < 0 ? Infinity : head - latest, life = GHOST_STEPS + GHOST_HOLD * this.weight;
    this.shade.visible = age < life;
    if (this.shade.visible) {
      const i = latest - first, x = xs[i - 1]!, a = lows[i - 1]!, b = highs[i - 1]!;
      this.shade.position.set(x, 0, 0);
      this.ghostElbow.value.set(this.pole * Math.sin(a), this.pole * Math.cos(a));
      this.ghostTip.value.set(this.ghostElbow.value.x + this.pole * Math.sin(b), this.ghostElbow.value.y + this.pole * Math.cos(b));
      this.fade.value = this.weight * (1 - age / life) ** 2;
    }
  }

  private lay(trail: Ribbon, points: Float64Array, count: number): void {
    const out = trail.positions, ages = trail.ages, quads = trail.quads, owners = this.owners;
    let nx = 0, ny = 1, indices = 0;
    for (let j = 0; j < count; j++) {
      const a = j > 0 && owners[j - 1] === owners[j] ? j - 1 : j, b = j + 1 < count && owners[j + 1] === owners[j] ? j + 1 : j;
      const dx = points[2 * a]! - points[2 * b]!, dy = points[2 * a + 1]! - points[2 * b + 1]!, chord = Math.hypot(dx, dy);
      if (chord > 1e-6) {
        nx = -dy / chord;
        ny = dx / chord;
      }
      const age = j / (POINTS - 1), w = WIDTH * (1 - 0.55 * age), x = points[2 * j]!, y = points[2 * j + 1]!, o = 6 * j;
      out[o] = x + nx * w;
      out[o + 1] = 0;
      out[o + 2] = -(y + ny * w);
      out[o + 3] = x - nx * w;
      out[o + 4] = 0;
      out[o + 5] = -(y - ny * w);
      ages[2 * j] = ages[2 * j + 1] = age;
      if (b === j + 1) {
        const v = 2 * j;
        quads[indices++] = v;
        quads[indices++] = v + 1;
        quads[indices++] = v + 2;
        quads[indices++] = v + 1;
        quads[indices++] = v + 3;
        quads[indices++] = v + 2;
      }
    }
    trail.commit(indices);
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.trail = null;
    this.embers = null;
  }
}

export default (() => new LongExposureScene()) satisfies SceneFactory;
