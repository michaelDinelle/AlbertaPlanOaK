import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { DebugCurve, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendRow, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { QUADROTOR_ANGLE, QUADROTOR_GOAL_X, QUADROTOR_GOAL_Y, QUADROTOR_GUST, QUADROTOR_MOVE, QUADROTOR_SPIN, QUADROTOR_VX, QUADROTOR_VY, QUADROTOR_WIND_X, QUADROTOR_WIND_Y, QUADROTOR_X, QUADROTOR_Y } from '@sim/envs/quadrotor';
import { SIM_EVENT_TYPES, type EnvFrame, type EventStream } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { GAUGE, HALO, LIFT, PALETTE, PROP, RING, drone, envelope, flat, gauge, gaugeAt, ghostRing, guides, halo, hall, lampXs, lay, lensAt, plume, ribbon, sprite, waypoint, wind, wreck, type Arena, type Ribbon } from './art';

const TRAIL = 60;
const SUB = 3;
const TIP = 14;
const TIP_SUB = 5;
const WRECK = 20;
const HOP = 50;
const GUST = 40;
const SPAWN = 10;
const PLUME = 0.26;
const CLEARANCE = 0.9;
const SPIN = 9;
const TRACE = 150;
const ARROWS = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'] as const;
const CUE = SIM_EVENT_TYPES.indexOf('cue');

const effort = (u: number): number => (u >= 1 ? 1 : u > -1 ? u : -1);
const word = (frame: EnvFrame, slot: number): number => wordFloat(frame.words[slot]!);
const toWorld = (x: number, y: number, out: THREE.Vector3): THREE.Vector3 => out.set(x, 0, -y);
const signed = (value: number): string => formatSigned(value).replace('-', '−');
const danger = (gap: number): number => Math.min(1, Math.max(0, 1 - gap / CLEARANCE)) ** 2;
const disc = (radius: number): THREE.BufferGeometry => new THREE.CircleGeometry(radius, 24).rotateX(-Math.PI / 2);

function tiltOf(degrees: number): string {
  const size = Math.abs(degrees);
  return size < 0.5 ? 'level' : `${size.toFixed(size < 10 ? 1 : 0)}° ${degrees > 0 ? 'left' : 'right'}`;
}

function spinOf(rate: number): string {
  const turns = Math.abs(rate) / (2 * Math.PI);
  return turns < 0.05 ? 'steady' : `${rate > 0 ? '↺' : '↻'} ${turns.toFixed(1)} rev/s`;
}

function windOf(x: number, y: number): string {
  const size = Math.hypot(x, y);
  if (size === 0) return 'calm';
  return `${ARROWS[(Math.round(Math.atan2(y, x) / (Math.PI / 4)) + 8) % 8]} ${size.toFixed(2)} m/s²`;
}

function lastCue(stream: EventStream, value: number, step: number, span: number): number | null {
  for (let b = Math.floor(step / stream.blockFrames); b >= 0 && b >= Math.floor((step - span) / stream.blockFrames); b--) {
    const block = stream.blocks[b];
    if (!block) continue;
    for (let i = block.steps.length - 1; i >= 0; i--) {
      const at = block.steps[i]!;
      if (at <= step && at > step - span && block.kinds[i] === CUE && block.values[i] === value) return at;
    }
  }
  return null;
}

function rectangle(curve: DebugCurve, x0: number, y0: number, x1: number, y1: number): void {
  for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] as const) curve.point(x, 0.05, -y);
}

class FlightCageScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  controlHz = 50;
  protected readonly stageName = 'flightCage';

  private arena: Arena = { wall: 1, ceiling: 1, reach: 1, low: 0, high: 1, arm: 0.1 };
  private gusty = false;
  private gustMax = 1;
  private dial: readonly [number, number] = [0, 0];
  private readonly rig = new THREE.Group();
  private readonly craft = new THREE.Group();
  private readonly wreckage = new THREE.Group();
  private target: THREE.Mesh | null = null;
  private beacon: THREE.Mesh | null = null;
  private ghost: THREE.Mesh | null = null;
  private burn: THREE.Mesh | null = null;
  private glowing: THREE.Mesh | null = null;
  private readonly rotors: THREE.Mesh[] = [];
  private trail: Ribbon | null = null;
  private tips: [Ribbon, Ribbon] | null = null;
  private readonly phase = [uniform(0), uniform(0)] as const;
  private readonly power = [uniform(1), uniform(1)] as const;
  private readonly flow = [uniform(0), uniform(0)] as const;
  private readonly spawn = uniform(0);
  private readonly aura = uniform(0.5);
  private readonly launch = uniform(new THREE.Vector2());
  private readonly known = uniform(0);
  private readonly droneAt = uniform(new THREE.Vector2());
  private readonly goalAt = uniform(new THREE.Vector2());
  private readonly hopFrom = uniform(new THREE.Vector2());
  private readonly hop = uniform(0);
  private readonly closeness = uniform(0);
  private readonly pulse = uniform(0);
  private readonly spin = uniform(0);
  private readonly shine = uniform(6);
  private readonly ghostFade = uniform(0);
  private readonly wreckFade = uniform(0);
  private readonly near = uniform(new THREE.Vector4());
  private readonly crash = uniform(new THREE.Vector4());
  private readonly wreckAt = uniform(new THREE.Vector2());
  private readonly windDirection = uniform(new THREE.Vector2(1, 0));
  private readonly windStrength = uniform(0);
  private readonly windFlow = uniform(0);
  private readonly windFlash = uniform(0);
  private readonly path = new Float32Array(TRAIL * SUB * 2);
  private readonly tipPath = new Float32Array(TIP * TIP_SUB * 2);
  private readonly xs = new Float64Array(TRAIL + 2);
  private readonly ys = new Float64Array(TRAIL + 2);
  private readonly as = new Float64Array(TRAIL + 2);
  private readonly pose = { x: 0, y: 0, a: 0 };
  private readonly spot = new THREE.Vector3();
  private base = 0;
  private start = 0;
  private held = false;
  private sim = 0;
  private clock = 0;
  private readonly tally = { at: -Infinity, crashes: 0 };
  private readonly ledger = { gap: 0, left: 100, right: 100, tilt: 0, spin: 0, windX: 0, windY: 0, reward: 0, airborne: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.arena = {
      wall: constant(env, 'QUADROTOR_WALL'),
      ceiling: constant(env, 'QUADROTOR_CEILING'),
      reach: constant(env, 'QUADROTOR_REACH'),
      low: constant(env, 'QUADROTOR_LOW'),
      high: constant(env, 'QUADROTOR_HIGH'),
      arm: constant(env, 'QUADROTOR_ARM'),
    };
    this.controlHz = 1 / (constant(env, 'QUADROTOR_SUBSTEPS') * constant(env, 'QUADROTOR_SUBSTEP'));
    this.gusty = parameter(env, 'gustPeriod') > 0;
    this.gustMax = Math.hypot(parameter(env, 'gustX'), parameter(env, 'gustY'));
    const { wall, ceiling } = this.arena;
    return new THREE.Box2(new THREE.Vector2(-wall - 0.45, -ceiling - 0.45), new THREE.Vector2(wall + 0.45, 0.3));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x9fb2d6).multiplyScalar(0.6);
  }

  protected override poolOptions() {
    return { burstSize: 56 };
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, red = 0xff6b57, slate = 0x8fa6c4, mint = 0x86f0b4;
    return {
      debug: (frame, kit) => {
        const { wall, ceiling, reach, low, high, arm } = this.arena, pose = frame.state.pose, step = frame.state.step;
        rectangle(kit.curve('envelope', 5, red), -wall, 0, wall, ceiling);
        rectangle(kit.curve('waypoints', 5, gold), -reach, low, reach, high);
        const x = pose[QUADROTOR_X]!, y = pose[QUADROTOR_Y]!, angle = pose[QUADROTOR_ANGLE]!, c = Math.cos(angle), s = Math.sin(angle);
        const now = frame.frameAt(step), next = frame.frameAt(step + 1), first = frame.frameAt(0);
        const gx = word(now, QUADROTOR_GOAL_X), gy = word(now, QUADROTOR_GOAL_Y);
        const body = kit.curve('body', 2, cyan);
        body.point(x - arm * c, 0.05, -(y - arm * s));
        body.point(x + arm * c, 0.05, -(y + arm * s));
        const marks = kit.marks('points', 5);
        marks.set(x, 0.05, -y, 0.03, cyan);
        marks.set(x + arm * c, 0.05, -(y + arm * s), 0.022, PALETTE.starboard);
        marks.set(x - arm * c, 0.05, -(y - arm * s), 0.022, PALETTE.port);
        marks.set(gx, 0.05, -gy, 0.035, gold);
        marks.set(word(first, QUADROTOR_X), 0.05, -word(first, QUADROTOR_Y), 0.03, mint);
        const gap = kit.curve('gap', 2, gold);
        gap.point(x, 0.05, -y);
        gap.point(gx, 0.05, -gy);
        for (const [i, side, name] of [[0, 1, 'thrustRight'], [1, -1, 'thrustLeft']] as const) {
          const rx = x + side * arm * c, ry = y + side * arm * s, span = PLUME * (1 + effort(next.actions[i]!));
          const thrust = kit.curve(name, 2, i === 0 ? PALETTE.starboard : PALETTE.port);
          thrust.point(rx, 0.05, -ry);
          thrust.point(rx - s * span, 0.05, -(ry + c * span));
        }
        const velocity = kit.curve('velocity', 2, slate);
        velocity.point(x, 0.05, -y);
        velocity.point(x + pose[QUADROTOR_VX]! * 0.25, 0.05, -(y + pose[QUADROTOR_VY]! * 0.25));
        if (this.gusty) {
          const push = kit.curve('wind', 2, PALETTE.wind);
          push.point(x, 0.05, -y);
          push.point(x + word(now, QUADROTOR_WIND_X) * 0.08, 0.05, -(y + word(now, QUADROTOR_WIND_Y) * 0.08));
        }
        const crashed = lastEvent(frame.eventLog, 'harm', step), last = crashed !== undefined && step - crashed < WRECK ? frame.frameAt(crashed - 1) : null;
        const wreckage = kit.curve('wreck', 2, red);
        if (last && last.step === crashed! - 1) {
          const wx = word(last, QUADROTOR_X), wy = word(last, QUADROTOR_Y), wa = word(last, QUADROTOR_ANGLE);
          wreckage.point(wx - arm * Math.cos(wa), 0.05, -(wy - arm * Math.sin(wa)));
          wreckage.point(wx + arm * Math.cos(wa), 0.05, -(wy + arm * Math.sin(wa)));
        }
        const trace = kit.curve('trace', TRACE, cyan);
        for (let i = 0; i < TRACE; i++) {
          const past = frame.frameAt(step - i);
          if (past.step !== step - i) break;
          trace.point(word(past, QUADROTOR_X), 0.05, -word(past, QUADROTOR_Y));
          if (past.reset) break;
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit, arena = this.arena, { wall, ceiling, arm } = arena;
    this.scene.background = new THREE.Color(PALETTE.void);
    this.scene.add(this.rig);
    this.rig.add(sprite(-40, -40, 40, 40, 0, hall(lit, arena, accent, this.launch, this.known)));
    if (this.gusty) {
      const [gx, gy, reach] = gaugeAt(arena), dial = sprite(-reach - 0.01, -GAUGE - 0.02, GAUGE + 0.02, GAUGE + 0.02, 1, gauge(lit, this.windDirection, this.windStrength, this.windFlash, reach));
      dial.position.set(gx, 0, -gy);
      this.dial = [gx, gy];
      this.rig.add(dial, sprite(-wall, 0, wall, ceiling, 1, wind(arena, this.windDirection, this.windStrength, this.windFlow, this.windFlash), true));
    }
    this.rig.add(sprite(-wall - 0.3, -0.3, wall + 0.3, ceiling + 0.3, 2, envelope(arena, this.near, this.droneAt, this.crash, this.wreckAt), true));
    const trail = ribbon(TRAIL * SUB, accent, 1.1);
    const tips: [Ribbon, Ribbon] = [ribbon(TIP * TIP_SUB, PALETTE.starboard, 1.6), ribbon(TIP * TIP_SUB, PALETTE.port, 1.6)];
    for (const line of [trail, ...tips]) {
      line.mesh.renderOrder = 3;
      this.rig.add(line.mesh);
    }
    this.trail = trail;
    this.tips = tips;
    this.rig.add(sprite(-wall, 0, wall, ceiling, 4, guides(this.droneAt, this.goalAt, this.hopFrom, this.hop, this.closeness), true));
    const ring = RING + 0.14;
    this.ghost = sprite(-ring, -ring, ring, ring, 5, ghostRing(this.ghostFade), true);
    this.target = sprite(-0.5, -0.5, 0.5, 0.5, 5, waypoint(this.closeness, this.pulse, this.spin), true);
    this.rig.add(this.ghost, this.target);

    this.craft.add(sprite(-HALO, -HALO, HALO, HALO, 5, halo(accent, this.aura), true));
    const reach = 2 * PLUME + 0.12;
    for (const [i, side] of [[0, 1], [1, -1]] as const) {
      const jet = sprite(-0.13, -reach, 0.13, 0.03, 6, plume(this.power[i], this.flow[i], PLUME), true);
      jet.position.set(side * arm, 0, -LIFT);
      const rotor = light.emitter(disc(0.05), rgb(PALETTE.plume).mul(2));
      rotor.position.set(side * arm, 0, -0.02);
      this.rotors.push(rotor);
      this.craft.add(jet, rotor);
    }
    this.craft.add(sprite(-arm - PROP - 0.03, -0.06, arm + PROP + 0.03, 0.08, 7, drone(lit, accent, arm, this.phase[0], this.phase[1], this.spawn)));
    this.wreckage.add(sprite(-arm - PROP - 0.03, -0.06, arm + PROP + 0.03, 0.08, 8, wreck(arm, this.wreckFade), true));
    this.wreckage.visible = false;
    this.rig.add(this.craft, this.wreckage);

    for (const x of lampXs(wall)) {
      const lamp = light.emitter(flat(0.26, 0.022), rgb(PALETTE.lamp).mul(9));
      lamp.position.set(x, 0, -(ceiling + 0.148));
      this.rig.add(lamp);
    }
    const [lensX, lensY] = lensAt(arena);
    for (const side of [-1, 1]) {
      const lens = light.emitter(disc(0.022), rgb(PALETTE.lens).mul(3));
      lens.position.set(side * lensX, 0, -lensY);
      this.rig.add(lens);
    }
    this.beacon = light.emitter(disc(0.1), rgb(PALETTE.gold).mul(this.shine));
    this.burn = light.emitter(disc(0.12), rgb(PALETTE.ember).mul(6));
    this.glowing = light.emitter(disc(0.1), rgb(accent).mul(5));
    this.burn.visible = this.glowing.visible = false;
    this.rig.add(this.beacon, this.burn, this.glowing);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    this.legend = this.describe(accent);
  }

  private describe(accent: number): LegendSpec {
    const ledger = this.ledger;
    const rows: LegendRow[] = [
      { key: 'gap', label: 'to waypoint', tint: PALETTE.gold, value: () => `${ledger.gap.toFixed(2)} m` },
      { key: 'left', label: 'left rotor', tint: PALETTE.port, value: () => `${ledger.left}% hover` },
      { key: 'right', label: 'right rotor', tint: PALETTE.starboard, value: () => `${ledger.right}% hover` },
      { key: 'tilt', label: 'tilt', tint: accent, value: () => tiltOf(ledger.tilt) },
      { key: 'spin', label: 'spin', tint: accent, value: () => spinOf(ledger.spin) },
    ];
    if (this.gusty) rows.push({ key: 'wind', label: 'unseen wind', tint: PALETTE.wind, value: () => windOf(ledger.windX, ledger.windY) });
    rows.push(
      { key: 'airborne', label: 'airborne', tint: PALETTE.ember, value: () => `${formatCount(ledger.airborne)} steps` },
      { key: 'reward', label: 'reward', tint: PALETTE.gold, value: () => signed(ledger.reward) },
    );
    return { title: 'FLIGHT CAGE', rows, note: () => `${formatCount(this.tally.crashes)} crash${this.tally.crashes === 1 ? '' : 'es'}` };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const { trail, tips, target, beacon, ghost, burn, glowing } = this;
    if (!trail || !tips || !target || !beacon || !ghost || !burn || !glowing) return;
    const state = frame.state, pose = state.pose, step = state.step, fraction = frame.position - step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    const sim = frame.position / frame.controlHz, dt = Math.min(0.1, Math.max(0, sim - this.sim));
    this.sim = sim;
    this.clock += frame.dtReal;
    const { wall, ceiling, arm } = this.arena;

    const first = frame.frameAt(0);
    if (first.step === 0) {
      this.launch.value.set(word(first, QUADROTOR_X), word(first, QUADROTOR_Y));
      this.known.value = 1;
    }

    const x = pose[QUADROTOR_X]!, y = pose[QUADROTOR_Y]!, angle = pose[QUADROTOR_ANGLE]!;
    this.craft.position.set(x, 0, -y);
    this.craft.rotation.y = angle;
    this.droneAt.value.set(x, y);

    const now = frame.frameAt(step), next = frame.frameAt(step + 1);
    const powers = [1 + effort(next.actions[0]!), 1 + effort(next.actions[1]!)] as const;
    for (let i = 0; i < 2; i++) {
      const power = powers[i]!;
      this.power[i]!.value = power;
      this.phase[i]!.value = (this.phase[i]!.value + dt * SPIN * 2 * Math.PI * Math.sqrt(power)) % (2 * Math.PI);
      this.flow[i]!.value = (this.flow[i]!.value + dt * 5 * power) % 1;
      this.rotors[i]!.scale.setScalar(Math.max(power / 2, 1e-3));
      this.rotors[i]!.visible = power > 0.05;
    }

    const gx = word(now, QUADROTOR_GOAL_X), gy = word(now, QUADROTOR_GOAL_Y);
    target.position.set(gx, 0, -gy);
    beacon.position.set(gx, 0, -gy);
    this.goalAt.value.set(gx, gy);
    this.closeness.value = Math.exp(-Math.hypot(x - gx, y - gy) / 0.35);
    this.spin.value = (sim * 0.2) % 1;

    const moved = lastCue(frame.eventLog, QUADROTOR_MOVE, step, HOP), hopAge = moved === null ? Infinity : step - moved + fraction;
    ghost.visible = moved !== null && hopAge < HOP;
    this.hop.value = 0;
    this.pulse.value = 0;
    if (moved !== null && hopAge < HOP) {
      const before = frame.frameAt(moved - 1), t = hopAge / HOP, ox = word(before, QUADROTOR_GOAL_X), oy = word(before, QUADROTOR_GOAL_Y);
      ghost.position.set(ox, 0, -oy);
      ghost.scale.setScalar(1 - 0.6 * t);
      this.ghostFade.value = (1 - t) ** 2;
      this.hopFrom.value.set(ox, oy);
      this.hop.value = 0.9 * (1 - t) ** 2;
      this.pulse.value = (1 - t) ** 3;
    }
    this.shine.value = 4 + 8 * this.closeness.value + 10 * this.pulse.value;

    const crashed = lastEvent(frame.eventLog, 'harm', step), crashAge = crashed === undefined ? Infinity : step - crashed + fraction;
    const before = crashed !== undefined && crashAge < WRECK ? frame.frameAt(crashed - 1) : null;
    this.wreckage.visible = burn.visible = false;
    this.wreckFade.value = 0;
    this.crash.value.set(0, 0, 0, 0);
    if (before && before.step === crashed! - 1) {
      const wx = word(before, QUADROTOR_X), wy = word(before, QUADROTOR_Y), wa = word(before, QUADROTOR_ANGLE), t = crashAge / WRECK, fade = (1 - t) ** 2;
      this.wreckage.visible = burn.visible = true;
      this.wreckage.position.set(wx, 0, -wy);
      this.wreckage.rotation.y = wa;
      this.wreckage.scale.setScalar(1 + 0.2 * t);
      this.wreckFade.value = fade;
      burn.position.set(wx, 0, -wy);
      burn.scale.setScalar(Math.max(fade, 1e-3));
      this.wreckAt.value.set(wx, wy);
      const ws = Math.abs(Math.sin(wa)), wc = Math.abs(Math.cos(wa));
      this.crash.value.set(danger(wy - arm * ws), danger(ceiling - wy - arm * ws), danger(wall + wx - arm * wc), danger(wall - wx - arm * wc)).multiplyScalar(fade);
    }
    this.spawn.value = crashAge < SPAWN ? (1 - crashAge / SPAWN) ** 2 : 0;
    glowing.visible = this.spawn.value > 0.02;
    glowing.scale.setScalar(Math.max(this.spawn.value, 1e-3));
    glowing.position.set(x, 0, -y);
    this.aura.value = 0.28 + 0.5 * this.spawn.value;

    const s = Math.abs(Math.sin(angle)), c = Math.abs(Math.cos(angle));
    this.near.value.set(danger(y - arm * s), danger(ceiling - y - arm * s), danger(wall + x - arm * c), danger(wall - x - arm * c));

    const windX = this.gusty ? word(now, QUADROTOR_WIND_X) : 0, windY = this.gusty ? word(now, QUADROTOR_WIND_Y) : 0;
    if (this.gusty) {
      const size = Math.hypot(windX, windY);
      if (size > 0) this.windDirection.value.set(windX / size, windY / size);
      this.windStrength.value = size / this.gustMax;
      this.windFlow.value += dt * size * 0.3;
      const gusted = lastCue(frame.eventLog, QUADROTOR_GUST, step, GUST), gustAge = gusted === null ? Infinity : step - gusted + fraction;
      this.windFlash.value = gustAge < GUST ? (1 - gustAge / GUST) ** 2 : 0;
    }

    for (const event of frame.events) {
      if (event.type === 'harm' && this.wreckage.visible) pool.burst(this.spot.copy(this.wreckage.position).setY(32), PALETTE.ember, 0.5, time);
      if (event.type === 'cue' && event.value === QUADROTOR_MOVE) pool.ripple(toWorld(gx, gy, this.spot), PALETTE.gold, 0.3, time);
      if (event.type === 'cue' && event.value === QUADROTOR_GUST) pool.ripple(toWorld(this.dial[0], this.dial[1], this.spot), PALETTE.wind, 0.2, time);
    }

    this.gather(frame);
    this.stroke(frame, trail, tips);
    if (frame.scrubbed || this.clock - this.tally.at > 0.5) {
      this.tally.crashes = countEvents(frame.eventLog, 'harm', 0, step);
      this.tally.at = this.clock;
    }
    const ledger = this.ledger;
    ledger.gap = Math.hypot(word(now, QUADROTOR_X) - gx, word(now, QUADROTOR_Y) - gy);
    ledger.right = Math.round(100 * powers[0]);
    ledger.left = Math.round(100 * powers[1]);
    ledger.tilt = Math.atan2(Math.sin(angle), Math.cos(angle)) * 180 / Math.PI;
    ledger.spin = pose[QUADROTOR_SPIN]!;
    ledger.windX = windX;
    ledger.windY = windY;
    ledger.reward = state.reward;
    ledger.airborne = step - (crashed ?? 0);
  }

  private gather(frame: FrameUpdate): void {
    const { xs, ys, as } = this, step = frame.state.step, base = step - TRAIL;
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
      xs[k - base] = word(f, QUADROTOR_X);
      ys[k - base] = word(f, QUADROTOR_Y);
      as[k - base] = word(f, QUADROTOR_ANGLE);
      if (k > step) held = f.reset;
      else if (f.reset) start = k;
    }
    this.base = base;
    this.start = start;
    this.held = held;
  }

  private sample(tau: number, step: number): { x: number; y: number; a: number } {
    const { xs, ys, as, pose } = this, k = Math.min(step, Math.floor(tau)), i = k - this.base, u = k === step && this.held ? 0 : tau - k;
    const turn = as[i + 1]! - as[i]!;
    pose.x = xs[i]! + (xs[i + 1]! - xs[i]!) * u;
    pose.y = ys[i]! + (ys[i + 1]! - ys[i]!) * u;
    pose.a = as[i]! + (turn - 2 * Math.PI * Math.round(turn / (2 * Math.PI))) * u;
    return pose;
  }

  private stroke(frame: FrameUpdate, trail: Ribbon, tips: [Ribbon, Ribbon]): void {
    const { path, tipPath, start } = this, step = frame.state.step, arm = this.arena.arm;
    for (let j = 0; j < trail.points; j++) {
      const at = this.sample(Math.max(start, frame.position - j / SUB), step);
      path[2 * j] = at.x;
      path[2 * j + 1] = at.y;
    }
    lay(trail, path, 0.016, 0.004);
    tips.forEach((tip, side) => {
      const sign = side === 0 ? 1 : -1;
      for (let j = 0; j < tip.points; j++) {
        const at = this.sample(Math.max(start, frame.position - j / TIP_SUB), step);
        tipPath[2 * j] = at.x + sign * arm * Math.cos(at.a);
        tipPath[2 * j + 1] = at.y + sign * arm * Math.sin(at.a);
      }
      lay(tip, tipPath, 0.011, 0.003);
    });
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.rotors.length = 0;
    this.trail = this.target = this.beacon = this.ghost = this.burn = this.glowing = null;
    this.tips = null;
  }
}

export default (() => new FlightCageScene()) satisfies SceneFactory;
