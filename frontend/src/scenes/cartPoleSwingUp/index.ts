import * as THREE from 'three/webgpu';
import { float, mix, uniform } from 'three/tsl';
import type { DebugCurve, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { CARTPOLE_THETA, CARTPOLE_X, CARTPOLE_X_DOT } from '@sim/envs/cartpole';
import type { EnvFrame } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { CART, LAMP, PALETTE, RAIL, backdrop, cart, chevrons, dial, flat, ghost, hub, lamp, pylon, rail, ribbon, rod, sprite, type Ribbon } from './sprites';

const TRAIL = 50;
const SUB = 4;
const GHOST_STEPS = 30;
const FLASH_STEPS = 30;
const ROD = 0.017;
const HUB = 0.045;
const PUSH_GAP = 0.36;
const PYLON = 3;

const pushOf = (frame: EnvFrame): number => ((Math.trunc(frame.actions[0]!) >>> 0) % 3) - 1;

function circle(curve: DebugCurve, x: number, y: number, radius: number): void {
  const count = curve.capacity - 1;
  for (let i = 0; i <= count; i++) {
    const a = 2 * Math.PI * i / count;
    curve.point(x + radius * Math.cos(a), 0.05, -(y + radius * Math.sin(a)));
  }
}

class TrestleScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  controlHz = 50;
  protected readonly stageName = 'trestle';

  private track = 1;
  private reach = 1;
  private force = 1;
  private slip = 0;
  private readonly rig = new THREE.Group();
  private readonly body = new THREE.Group();
  private readonly pole = new THREE.Group();
  private readonly ring = new THREE.Group();
  private readonly push = new THREE.Group();
  private readonly wreck = new THREE.Group();
  private trail: Ribbon | null = null;
  private readonly roll = uniform(0);
  private readonly lift = uniform(0);
  private readonly flow = uniform(0);
  private readonly lamps = [uniform(0), uniform(0)] as const;
  private readonly fade = uniform(0);
  private readonly slant = uniform(new THREE.Vector2(0, -1));
  private readonly path = new Float32Array(TRAIL * SUB * 2);
  private readonly xs = new Float64Array(TRAIL + 2);
  private readonly thetas = new Float64Array(TRAIL + 2);
  private readonly spot = new THREE.Vector3();
  private readonly ledger = { angle: 0, x: 0, push: 0, reward: 0, lift: 0, since: null as number | null };

  private tipX(x: number, theta: number): number {
    return x + this.reach * Math.sin(theta);
  }

  private tipY(theta: number): number {
    return this.reach * Math.cos(theta);
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.controlHz = 1 / constant(env, 'CARTPOLE_DT');
    this.track = constant(env, 'CARTPOLE_TRACK');
    this.reach = 2 * constant(env, 'CARTPOLE_HALF');
    this.force = constant(env, 'CARTPOLE_FORCE');
    this.slip = parameter(env, 'slip');
    const wide = this.track + 0.6, tall = this.reach + 0.2;
    return new THREE.Box2(new THREE.Vector2(-wide, -tall), new THREE.Vector2(wide, tall));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa9b6dc).multiplyScalar(0.75);
  }

  protected override poolOptions() {
    return { burstSize: 70 };
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, red = 0xff6b57, slate = 0x8fa6c4;
    return {
      debug: (frame, kit) => {
        const { track, reach } = this, pose = frame.state.pose, step = frame.state.step;
        const x = pose[CARTPOLE_X]!, theta = pose[CARTPOLE_THETA]!, tipX = this.tipX(x, theta), tipY = this.tipY(theta);
        const bounds = kit.curve('track', 2, slate);
        bounds.point(-track, 0.05, 0);
        bounds.point(track, 0.05, 0);
        for (const [name, side] of [['edgeLeft', -1], ['edgeRight', 1]] as const) {
          const edge = kit.curve(name, 2, red);
          edge.point(side * track, 0.05, -0.5);
          edge.point(side * track, 0.05, 0.5);
        }
        circle(kit.curve('reach', 97, slate), x, 0, reach);
        const pole = kit.curve('pole', 2, cyan);
        pole.point(x, 0.05, 0);
        pole.point(tipX, 0.05, -tipY);
        const marks = kit.marks('points', 3);
        marks.set(x, 0.05, 0, 0.03, cyan);
        marks.set(tipX, 0.05, -tipY, 0.025, gold);
        marks.set(x, 0.05, -reach, 0.02, gold);
        const velocity = kit.curve('velocity', 2, slate), ahead = x + pose[CARTPOLE_X_DOT]! * 5 / this.controlHz;
        velocity.point(x, 0.05, 0.3);
        velocity.point(ahead, 0.05, 0.3);
        const push = pushOf(frame.frameAt(step + 1)), arrow = kit.curve('push', 5, cyan), end = x + push * 0.45;
        if (push !== 0) for (const [px, pz] of [[x, 0], [end, 0], [end - push * 0.08, -0.05], [end, 0], [end - push * 0.08, 0.05]] as const) arrow.point(px, 0.05, pz);
        const trace = kit.curve('trace', TRAIL, gold);
        for (let i = 0; i < TRAIL && step - i >= 0; i++) {
          const past = frame.frameAt(step - i);
          if (past.step !== step - i) break;
          const px = wordFloat(past.words[CARTPOLE_X]!), pt = wordFloat(past.words[CARTPOLE_THETA]!);
          trace.point(this.tipX(px, pt), 0.05, -this.tipY(pt));
          if (past.reset) break;
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit, { track, reach } = this;
    this.scene.background = new THREE.Color(PALETTE.night);
    this.scene.add(this.rig);
    const wide = track + 8, tall = reach + 8;
    this.rig.add(sprite(-wide, -tall, wide, tall, 0, backdrop(lit)));
    for (const side of [-1, 1]) {
      const tower = sprite(-0.16, -PYLON, 0.16, 0.001, 1, pylon(lit, PYLON));
      tower.position.set(side * track / 2, 0, -RAIL.bottom);
      this.rig.add(tower);
    }
    this.rig.add(sprite(-track - 0.05, RAIL.bottom - 0.01, track + 0.05, RAIL.top + 0.01, 2, rail(lit, track, accent)));
    for (const [thick, y] of [[RAIL.flange, RAIL.top - RAIL.flange / 2], [RAIL.chord, RAIL.bottom + RAIL.chord / 2]] as const) {
      const block = light.occluder(flat(2 * track, thick * 0.5));
      block.position.set(0, 0, -y);
      this.rig.add(block);
    }
    this.ring.add(sprite(-reach - 0.1, -reach - 0.1, reach + 0.1, reach + 0.1, 3, dial(reach, accent)));
    const trail = ribbon(TRAIL * SUB, reach);
    trail.mesh.renderOrder = 4;
    this.trail = trail;
    this.body.add(sprite(-CART.halfW - 0.01, CART.wheelY - CART.wheel - 0.01, CART.halfW + 0.01, CART.bodyY + CART.halfH + 0.01, 5, cart(lit, accent, this.roll)));
    this.pole.add(
      sprite(-0.04, -0.04, reach + 0.03, 0.04, 6, rod(lit, reach, ROD, this.lift)),
      light.emitter(flat(reach, 0.026).translate(reach / 2, 0, 0), mix(rgb(PALETTE.ember), rgb(PALETTE.filament), this.lift).mul(mix(float(0.3), float(7.5), this.lift.mul(this.lift)))),
      sprite(-HUB - 0.005, -HUB - 0.005, HUB + 0.005, HUB + 0.005, 7, hub(lit, HUB, accent)),
    );
    this.push.add(sprite(-0.09, -0.05, 0.09, 0.05, 5.5, chevrons(this.flow)), light.emitter(flat(0.13, 0.04), rgb(PALETTE.push).mul(3)));
    for (const [i, side] of [[0, -1], [1, 1]] as const) {
      const glass = sprite(-0.02, -0.02, 0.02, 0.02, 2.5, lamp(this.lamps[i]));
      const glow = light.emitter(new THREE.CircleGeometry(LAMP.radius, 20).rotateX(-Math.PI / 2), rgb(PALETTE.warn).mul(this.lamps[i]));
      for (const part of [glass, glow]) part.position.set(side * (track + LAMP.x), 0, -LAMP.y);
      this.rig.add(glass, glow);
    }
    this.wreck.add(sprite(-reach - 0.3, -reach - 0.3, reach + 0.3, reach + 0.3, 9, ghost(reach, ROD, this.slant, this.fade)));
    this.wreck.visible = false;
    this.rig.add(this.ring, trail.mesh, this.body, this.pole, this.push, this.wreck);
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const ledger = this.ledger, force = formatCount(this.force);
    this.legend = {
      title: 'TRESTLE',
      rows: [
        { key: 'angle', label: 'from upright', tint: PALETTE.filament, value: () => `${formatSigned(ledger.angle)}°` },
        { key: 'cart', label: 'cart at', tint: PALETTE.steel, value: () => `${formatSigned(ledger.x)} m` },
        { key: 'push', label: this.slip > 0 ? 'chosen push' : 'push', tint: PALETTE.push, value: () => ledger.push < 0 ? `← ${force} N` : ledger.push > 0 ? `${force} N →` : 'coast' },
        { key: 'reward', label: 'reward', tint: ctx.tokens.celebrationGold, value: () => formatSigned(ledger.reward) },
        { key: 'fall', label: 'last fall', tint: PALETTE.ember, value: () => ledger.since === null ? 'none yet' : `${formatCount(ledger.since)} steps ago` },
      ],
      note: () => this.slip > 0 ? `${Math.round(this.slip * 100)}% of pushes slip` : `rail ends ±${this.track.toFixed(1)} m`,
      bar: () => ledger.lift,
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const trail = this.trail;
    if (!trail) return;
    const state = frame.state, pose = state.pose, step = state.step, fraction = frame.position - step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();

    const x = pose[CARTPOLE_X]!, theta = pose[CARTPOLE_THETA]!, lift = (1 + Math.cos(theta)) / 2;
    for (const group of [this.body, this.pole, this.ring]) group.position.set(x, 0, 0);
    this.pole.rotation.y = Math.PI / 2 - theta;
    this.roll.value = -x / CART.wheel;
    this.lift.value = lift;

    const push = pushOf(frame.frameAt(step + 1));
    this.push.visible = push !== 0;
    this.push.position.set(x + push * PUSH_GAP, 0, -CART.bodyY);
    this.push.rotation.y = push < 0 ? Math.PI : 0;
    this.flow.value = time * 2.5;

    const fell = lastEvent(frame.eventLog, 'harm', step), age = fell === undefined ? Infinity : step - fell + fraction;
    let side = 0;
    this.wreck.visible = false;
    const before = fell !== undefined && age < Math.max(GHOST_STEPS, FLASH_STEPS) ? frame.frameAt(fell - 1) : null;
    if (before && before.step === fell! - 1) {
      const bx = wordFloat(before.words[CARTPOLE_X]!), bt = wordFloat(before.words[CARTPOLE_THETA]!), t = age / GHOST_STEPS;
      side = bx < 0 ? -1 : 1;
      if (t < 1) {
        this.wreck.visible = true;
        this.wreck.position.set(bx + side * 0.35 * t, 0, 1.9 * t * t);
        this.wreck.rotation.y = -side * 2.1 * t;
        this.slant.value.set(Math.sin(bt), Math.cos(bt));
        this.fade.value = (1 - t) ** 2;
      }
    }
    const flash = side === 0 ? 0 : Math.max(0, 1 - age / FLASH_STEPS) ** 2;
    for (const [i, s] of [[0, -1], [1, 1]] as const) {
      const near = Math.min(1, Math.max(0, (s * x / this.track - 0.5) / 0.5));
      this.lamps[i].value = 0.6 + 6 * near * near + (s === side ? 14 * flash : 0);
    }
    for (const event of frame.events) {
      if (event.type !== 'harm' || side === 0) continue;
      pool.burst(this.spot.set(side * this.track, 0, -RAIL.top), PALETTE.ember, 0.5, time);
    }

    this.stroke(frame, trail);
    const ledger = this.ledger;
    ledger.angle = Math.atan2(Math.sin(theta), Math.cos(theta)) * 180 / Math.PI;
    ledger.x = x;
    ledger.push = push;
    ledger.reward = state.reward;
    ledger.lift = lift;
    ledger.since = fell === undefined ? null : step - fell;
  }

  private stroke(frame: FrameUpdate, trail: Ribbon): void {
    const { path, xs, thetas } = this, step = frame.state.step, base = step - TRAIL, points = TRAIL * SUB, out = trail.positions;
    let start = Math.max(0, base), held = false;
    for (let k = step + 1; k >= start; k--) {
      const f = frame.frameAt(k);
      if (f.step !== k) {
        if (k > step) {
          held = true;
          continue;
        }
        start = k + 1;
        break;
      }
      xs[k - base] = wordFloat(f.words[CARTPOLE_X]!);
      thetas[k - base] = wordFloat(f.words[CARTPOLE_THETA]!);
      if (k > step) held = f.reset;
      else if (f.reset) start = k;
    }
    for (let j = 0; j < points; j++) {
      const tau = Math.max(start, frame.position - j / SUB), k = Math.min(step, Math.floor(tau)), i = k - base;
      const u = k === step && held ? 0 : tau - k, turn = thetas[i + 1]! - thetas[i]!;
      const x = xs[i]! + (xs[i + 1]! - xs[i]!) * u, theta = thetas[i]! + (turn - 2 * Math.PI * Math.round(turn / (2 * Math.PI))) * u;
      path[2 * j] = this.tipX(x, theta);
      path[2 * j + 1] = this.tipY(theta);
    }
    let nx = 0, ny = 1;
    for (let i = points - 1; i >= 0; i--) {
      const a = Math.max(0, i - 1), b = Math.min(points - 1, i + 1);
      const dx = path[2 * a]! - path[2 * b]!, dy = path[2 * a + 1]! - path[2 * b + 1]!, chord = Math.hypot(dx, dy);
      if (chord > 1e-5) {
        nx = -dy / chord;
        ny = dx / chord;
      }
      const width = 0.011 * (1 - 0.65 * i / (points - 1)), px = path[2 * i]!, py = path[2 * i + 1]!, o = 6 * i;
      out[o] = px + nx * width;
      out[o + 2] = -(py + ny * width);
      out[o + 3] = px - nx * width;
      out[o + 5] = -(py - ny * width);
    }
    trail.attribute.needsUpdate = true;
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.trail = null;
  }
}

export default (() => new TrestleScene()) satisfies SceneFactory;
