import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { patienceWaited } from '@sim/envs/patience';
import { SIM_EVENT_TYPES, type EnvFrame, type EventStream } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatRate, formatSigned } from '../../format';
import {
  BEAM_END, BOWL, COIN_PITCH, COPPER, EMBER, GOLD, HANG, LEDGER, LEDGER_TOP, PIN, REST, SMOKE_ROOT, STAND,
  disc, emberSprite, flameSprite, ledgerSprite, pinAt, pointAt, puffSprite, sealLayout, sealPath, slots, smokeSprite, standSprite, table, threadPoint, threadSprite, tray,
  type Scalar, type Seal, type Tint,
} from './art';

const GOAL = SIM_EVENT_TYPES.indexOf('goal');
const DONE = SIM_EVENT_TYPES.indexOf('done');
const FAST = 5;
const REACH = 0.6;
const BURN_OUT = 0.74;
const LET_GO = 0.8;
const SNUFF = 0.3;
const RELIGHT = 0.64;
const SETTLE = 0.45;
const REARM = 0.4;
const RING = 0.45;
const COOL = 0.12;
const TAIL = 1;

const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const ease = (t: number, a: number, b: number) => smoothstep01(span(t, a, b));

class PatienceScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 3;
  protected readonly stageName = 'patience';

  private steps = 1;
  private cashOut = 0;
  private seal: Seal = sealLayout();
  private readonly path = sealPath(this.seal);
  private standAt = new THREE.Vector2();
  private ledgerX = 0;
  private accent = 0;
  private readonly burn = { front: uniform(0) as Scalar, renew: uniform(0) as Scalar, glow: uniform(1) as Scalar, tail: uniform(0.1) as Scalar };
  private readonly heat: Scalar = uniform(1);
  private readonly flicker: Scalar = uniform(1);
  private readonly smokeHeat: Scalar = uniform(1);
  private readonly clockNode: Scalar = uniform(0);
  private readonly puffAge: Scalar = uniform(-1);
  private readonly outerBurn: Scalar = uniform(0);
  private readonly outerAlpha: Scalar = uniform(1);
  private readonly flameHeat: Scalar = uniform(0);
  private readonly bell = { thread: uniform(1) as Scalar, burn: uniform(0) as Scalar, armed: uniform(1) as Scalar, ball: uniform(HANG) as Scalar, loose: uniform(0) as Scalar, ring: uniform(0) as Scalar };
  private readonly slide: Scalar = uniform(0);
  private readonly coins = slots(LEDGER + 1);
  private readonly emberTint: Tint = uniform(new THREE.Vector3());
  private readonly flameTint: Tint = uniform(new THREE.Vector3());
  private readonly bowlTint: Tint = uniform(new THREE.Vector3());
  private parts: { ember: THREE.Mesh; emberLight: THREE.Mesh; smoke: THREE.Mesh; puff: THREE.Mesh; flame: THREE.Mesh; flameLight: THREE.Mesh } | null = null;
  private threadFrom = new THREE.Vector2();
  private threadTo = new THREE.Vector2();
  private readonly point = new THREE.Vector2();
  private readonly spot = new THREE.Vector3();
  private readonly glow = new THREE.Color();
  private clock = 0;
  private pace = 0;
  private lastPosition = 0;
  private burstAt = -Infinity;
  private readonly ledger = { step: -1, count: -1, values: new Float32Array(LEDGER) };
  private readonly tally = { log: null as EventStream | null, through: -1, goals: 0, dones: 0, contiguous: 0, complete: false };
  private readonly row = { step: 0, position: 0, wait: false, reward: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.steps = parameter(env, 'length');
    this.cashOut = parameter(env, 'cashOut');
    if (!Number.isInteger(this.steps) || this.steps < 1) throw new Error(`${env.envId} waits ${this.steps} steps`);
    if (env.words !== 1 || env.actions !== 2 || env.obs !== 2) throw new Error(`${env.envId} carries ${env.words} words, ${env.actions} actions and ${env.obs} observations`);
    const seal = this.seal;
    this.standAt.set(seal.tray.x + 0.55 + STAND.x / 2, -0.35);
    this.ledgerX = -seal.tray.x - 0.95;
    this.threadFrom.copy(seal.end);
    this.threadTo.copy(this.standAt).add(BEAM_END);
    return new THREE.Box2(new THREE.Vector2(this.ledgerX - 0.6, -seal.tray.y - 0.35), new THREE.Vector2(this.standAt.x + STAND.x / 2 + 0.1, seal.tray.y + 0.35));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xd8c3a5).multiplyScalar(0.62);
  }

  private place(mesh: THREE.Object3D, at: THREE.Vector2): void {
    mesh.position.set(at.x, 0, -at.y);
  }

  private coinY(slot: number): number {
    return LEDGER_TOP - (slot - 1) * COIN_PITCH;
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const seal = this.seal, steps = this.steps, unit = seal.total / steps, head = frame.frameAt(frame.state.step);
        const line = kit.curve('seal', this.path.length, TOKENS.uiCyan);
        for (const at of this.path) line.point(at.x, 0.1, -at.y);
        const ticks = kit.marks('steps', steps + 1);
        for (let k = 0; k <= steps; k++) {
          pointAt(seal, k * unit, this.point);
          ticks.set(this.point.x, 0.1, -this.point.y, (5 * k) % steps === 0 ? 0.07 : 0.035, (5 * k) % steps === 0 ? TOKENS.celebrationGold : 0x8fa6c4);
        }
        pointAt(seal, head.words[0]! * unit, this.point);
        kit.marks('position', 1).set(this.point.x, 0.12, -this.point.y, 0.17, this.accent);
        const bars = kit.bars('observation', 2), width = 2 * seal.tray.x;
        head.observations.forEach((value, k) => {
          const reach = Math.max(1e-3, value * width);
          bars.set(-seal.tray.x + reach / 2, 0.1, -(seal.tray.y + 0.12 + 0.1 * k), 0.02, k === 0 ? this.accent : 0x8fa6c4, reach, 0.05);
        });
        const truth = kit.marks('ledger', LEDGER);
        this.ledger.values.forEach((value, k) => {
          if (value !== 0) truth.set(this.ledgerX + 0.34, 0.1, -this.coinY(k + 1), 0.08, value > 0 ? TOKENS.celebrationGold : COPPER);
        });
        if (frame.state.step >= 1) {
          const acted = kit.marks('action', 1);
          if (patienceWaited(head)) acted.set(this.point.x, 0.14, -this.point.y, 0.09, EMBER);
          else acted.set(this.ledgerX - 0.34, 0.1, -this.coinY(1), 0.09, EMBER);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit, seal = this.seal, steps = this.steps;
    this.accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(0x0b0706);
    for (const child of pool.group.children) child.renderOrder = 10;

    const pins = [1, 2, 3, 4].map((j) => pinAt(seal, (seal.total * j) / 5, new THREE.Vector2()));
    this.scene.add(table(lit, seal, this.standAt, this.ledgerX), tray(lit, seal, steps, this.accent, this.burn, pins));
    for (const pin of pins) {
      const shade = light.occluder(disc(PIN));
      this.place(shade, pin);
      this.scene.add(shade);
    }

    const from = this.threadFrom, to = this.threadTo, reach = from.distanceTo(to);
    const thread = threadSprite(lit, reach, this.outerBurn, this.outerAlpha);
    thread.position.set((from.x + to.x) / 2, 0, -(from.y + to.y) / 2);
    thread.rotation.y = Math.atan2(to.y - from.y, to.x - from.x);
    const stand = standSprite(lit, this.bell);
    this.place(stand, this.standAt);
    const bowlLight = light.emitter(disc(0.5), this.bowlTint);
    bowlLight.scale.set(1.1, 1, 0.22);
    this.place(bowlLight, this.point.copy(this.standAt).add(BOWL));
    const ledger = ledgerSprite(lit, this.coins, this.slide);
    this.place(ledger, this.point.set(this.ledgerX, 0));

    const ember = emberSprite(this.heat, this.flicker), emberLight = light.emitter(disc(0.06), this.emberTint);
    const smoke = smokeSprite(this.smokeHeat, this.clockNode), puff = puffSprite(this.puffAge, this.clockNode);
    const flame = flameSprite(this.flameHeat), flameLight = light.emitter(disc(0.04), this.flameTint);
    this.scene.add(thread, stand, bowlLight, ledger, ember, emberLight, smoke, puff, flame, flameLight);
    this.parts = { ember, emberLight, smoke, puff, flame, flameLight };

    const row = this.row, tally = this.tally;
    this.legend = {
      title: 'PATIENCE',
      rows: [
        { key: 'waited', label: 'waited', tint: this.accent, value: () => `${formatCount(row.position)} of ${formatCount(steps)}` },
        { key: 'choice', label: 'last choice', tint: EMBER, value: () => (row.step < 1 ? '—' : row.wait ? 'wait' : 'cash out') },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => (row.step < 1 ? '—' : formatSigned(row.reward)) },
        { key: 'payouts', label: 'full payouts', tint: GOLD, value: () => (tally.complete ? formatCount(tally.goals) : 'counting…') },
        { key: 'cashouts', label: 'cash-outs', tint: COPPER, value: () => (tally.complete ? formatCount(tally.dones - tally.goals) : 'counting…') },
      ],
      note: () => (tally.complete && row.step > 0 ? `${formatRate((tally.goals * steps + (tally.dones - tally.goals) * this.cashOut) / row.step)} reward per step` : `${formatCount(steps)} after ${formatCount(steps)} waits · ${formatRate(this.cashOut)} to cash out`),
      bar: () => row.position / steps,
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const parts = this.parts;
    if (!parts) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    this.clock += frame.dtReal;
    this.clockNode.value = this.clock;
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const at = (s: number): EnvFrame | null => {
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    const seal = this.seal, steps = this.steps, unit = seal.total / steps;
    const i = frame.state.step, head = frame.frameAt(i), next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    const a = head.words[0]!, waited = next !== null && patienceWaited(next), ends = next !== null && next.done !== 0;
    const payout = waited && ends, cash = next !== null && !waited;

    let front = a * unit, renew = 0, glow = 1, heat = 1, emberAt = front, puff = -1, outer = 0, flame = -1;
    let armed = 1, loose = 0, ball = HANG, burnt = 0, thread = 1;
    if (next && waited && !ends) front = emberAt = (a + t) * unit;
    else if (payout) {
      front = (a + (steps - a) * span(t, 0, REACH)) * unit;
      emberAt = t < RELIGHT ? front : 0;
      heat = t < RELIGHT ? 1 - ease(t, REACH, REACH + 0.04) : ease(t, RELIGHT + 0.04, 1);
      glow = 1 - ease(t, REACH, REACH + 0.2);
      renew = ease(t, REACH + 0.06, 0.96);
      outer = span(t, REACH, BURN_OUT);
      burnt = span(t, BURN_OUT, LET_GO);
      if (t >= REACH && t < LET_GO) flame = t;
      if (t >= LET_GO) {
        armed = 0;
        loose = 1;
        ball = HANG - (HANG - REST) * span(t, LET_GO, 1) ** 2;
      }
    } else if (cash) {
      emberAt = t < RELIGHT ? front : 0;
      heat = t < RELIGHT ? 1 - ease(t, SNUFF, SNUFF + 0.16) : ease(t, RELIGHT + 0.04, 1);
      glow = 1 - ease(t, SNUFF, SNUFF + 0.16);
      renew = ease(t, 0.5, 0.9);
      if (t >= SNUFF) puff = span(t, SNUFF, 0.96);
    }

    let paid = -1;
    for (let k = 0; k <= Math.ceil(this.controlHz * (SETTLE + REARM)) + 1; k++) {
      const f = at(i - k);
      if (!f) break;
      if (f.done && patienceWaited(f) && f.step > 0) {
        paid = f.step;
        break;
      }
    }
    const since = paid >= 0 ? (frame.position - paid) / this.controlHz : Infinity;
    const ring = since >= 0 && since < 3 ? Math.exp(-since / RING) : 0;
    if (!payout && !fast && since < SETTLE + REARM) {
      loose = 1 - ease(since, SETTLE, SETTLE + 0.2);
      ball = REST + 0.05 * Math.abs(Math.sin(since * 14)) * Math.exp(-since * 9);
      armed = thread = ease(since, SETTLE, SETTLE + REARM);
    }

    this.burn.front.value = front;
    this.burn.renew.value = renew;
    this.burn.glow.value = glow;
    this.burn.tail.value = Math.min(TAIL, Math.max(0.02, COOL * this.controlHz * unit));
    const flick = 0.9 + 0.06 * Math.sin(this.clock * 23.7) * Math.sin(this.clock * 7.1) + 0.04 * Math.sin(this.clock * 51.3);
    this.heat.value = heat;
    this.flicker.value = flick;
    this.smokeHeat.value = heat;
    pointAt(seal, emberAt, this.point);
    this.place(parts.ember, this.point);
    this.place(parts.emberLight, this.point);
    parts.smoke.position.set(this.point.x + SMOKE_ROOT.x, 0, -(this.point.y + SMOKE_ROOT.y));
    this.setTint(this.emberTint, EMBER, 11 * heat * flick);

    this.puffAge.value = puff;
    parts.puff.visible = puff >= 0 && puff < 1;
    if (parts.puff.visible) this.place(parts.puff, pointAt(seal, front, this.point));

    this.outerBurn.value = outer;
    this.outerAlpha.value = payout ? (t < BURN_OUT ? 1 : 0) : thread;
    const bell = this.bell;
    bell.thread.value = payout ? (t < LET_GO ? 1 : 0) : thread;
    bell.burn.value = burnt;
    bell.armed.value = armed;
    bell.loose.value = loose;
    bell.ball.value = ball;
    bell.ring.value = ring;
    this.setTint(this.bowlTint, GOLD, 14 * ring);

    parts.flame.visible = flame >= 0;
    this.flameHeat.value = flame >= 0 ? 1 : 0;
    if (flame >= 0) {
      if (flame < BURN_OUT) this.point.lerpVectors(this.threadFrom, this.threadTo, outer);
      else threadPoint(burnt, this.point).add(this.standAt);
      this.place(parts.flame, this.point);
      this.place(parts.flameLight, this.point);
    }
    this.setTint(this.flameTint, EMBER, flame >= 0 ? 10 : 0);

    this.fillLedger(frame.eventLog, i);
    const values = this.coins.values;
    values[0] = next && ends ? (waited ? 1 : -1) : 0;
    for (let k = 0; k < LEDGER; k++) values[k + 1] = this.ledger.values[k]!;
    this.slide.value = values[0] === 0 ? 0 : payout ? ease(t, LET_GO - 0.04, 1) : ease(t, 0.42, 0.86);

    if (!frame.scrubbed) for (const event of frame.events) {
      if (event.type !== 'goal' || (fast && this.clock - this.burstAt < 0.5)) continue;
      this.burstAt = this.clock;
      this.spot.set(this.standAt.x + BOWL.x, 38, -(this.standAt.y + BOWL.y));
      pool.burst(this.spot, TOKENS.celebrationGold, fast ? 0.4 : 0.55 + 0.15 * Math.log10(steps), time);
      if (!fast) pool.ripple(this.spot.setY(0), TOKENS.celebrationGold, 0.32, time);
    }

    this.count(frame.eventLog, i);
    Object.assign(this.row, { step: i, position: a, wait: patienceWaited(head), reward: head.reward });
  }

  private setTint(tint: Tint, hex: number, gain: number): void {
    this.glow.set(hex);
    tint.value.set(this.glow.r * gain, this.glow.g * gain, this.glow.b * gain);
  }

  private fillLedger(log: EventStream, step: number): void {
    const ledger = this.ledger;
    if (ledger.step === step && ledger.count === log.count) return;
    ledger.step = step;
    ledger.count = log.count;
    ledger.values.fill(0);
    let k = 0, ended = -1;
    for (let b = Math.floor(step / log.blockFrames); b >= 0 && k < LEDGER; b--) {
      const block = log.blocks[b];
      if (!block) break;
      for (let n = block.steps.length - 1; n >= 0; n--) {
        const at = block.steps[n]!;
        if (at > step) continue;
        if (block.kinds[n] === DONE) {
          if (k >= LEDGER) break;
          ledger.values[k++] = -1;
          ended = at;
        } else if (block.kinds[n] === GOAL && at === ended) ledger.values[k - 1] = 1;
      }
    }
  }

  private count(log: EventStream, step: number): void {
    const tally = this.tally;
    if (tally.log !== log || step < tally.through) Object.assign(tally, { log, through: -1, goals: 0, dones: 0, contiguous: 0 });
    while (log.blocks[tally.contiguous]) tally.contiguous++;
    tally.complete = Math.floor(step / log.blockFrames) < tally.contiguous;
    if (!tally.complete || step === tally.through) return;
    tally.goals += countEvents(log, 'goal', tally.through, step);
    tally.dones += countEvents(log, 'done', tally.through, step);
    tally.through = step;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.parts = null;
  }
}


export default (() => new PatienceScene()) satisfies SceneFactory;
