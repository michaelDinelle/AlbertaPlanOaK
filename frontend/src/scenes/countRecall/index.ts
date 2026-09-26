import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { COUNT_EVENT, COUNT_PHASE, COUNT_QUERY, COUNT_TALLY, countKey } from '@sim/envs/count';
import { SIM_EVENT_TYPES, type EnvFrame, type EventStream } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BEAD_TINT, BELL, BELL_TINT, BELL_Y, GOLD, KEY, LAMP_TINT, LEDGER, MISS, R_CASE, R_JEWEL, R_KEY, R_LANE, R_SOCKET, TAU,
  along, beads, bell, desk, dial, disc, glowBank, keys, ledger, loupe, pilot, slots, type Beads, type Keys, type Point, type Scalar,
} from './art';

type Span = readonly [number, number];
type Light = THREE.UniformNode<'vec3', THREE.Vector3>;

const FAST = 5;
const PRESS: Span = [0, 0.2];
const DROP: Span = [0.3, 0.62];
const HOP: Span = [0.5, 0.78];
const MOVE: Span = [0.42, 0.86];
const POP: Span = [0.8, 1];
const RING_IN: Span = [0.82, 1];
const RING_OUT: Span = [0, 0.6];
const JUDGE: Span = [0.2, 0.4];
const FADE_OUT: Span = [0.05, 0.3];
const FADE_IN: Span = [0.32, 0.5];
const GOAL = SIM_EVENT_TYPES.indexOf('goal');
const HARM = SIM_EVENT_TYPES.indexOf('harm');
const TALLY = 0x9fe6ff;
const ARC = 16;
const WAVE = 1;

const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const ease = (t: number, s: Span) => smoothstep01(span(t, s));
const snap = (u: number) => 1 + 2.1 * (u - 1) ** 3 + 1.1 * (u - 1) ** 2;
const drainOf = (slot: number, period: number): Span => [0.05 + (0.18 * slot) / period, 0.23 + (0.18 * slot) / period];

class CountRecallScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'countRecall';

  private period = 1;
  private modulus = 1;
  private accent = 0;
  private readonly sweep: Scalar = uniform(0);
  private readonly ring: Scalar = uniform(0);
  private readonly wave: Scalar = uniform(0);
  private readonly pilotAlpha: Scalar = uniform(1);
  private readonly falling = { at: uniform(new THREE.Vector2()) as Point, presence: uniform(0) as Scalar, gain: uniform(1.3) as Scalar };
  private readonly arrive = { at: uniform(new THREE.Vector2()) as Point, presence: uniform(0) as Scalar };
  private readonly history = slots(LEDGER);
  private readonly loupeTint: Light = uniform(new THREE.Vector3());
  private readonly dropTint: Light = uniform(new THREE.Vector3());
  private readonly arriveTint: Light = uniform(new THREE.Vector3());
  private readonly bellTint: Light = uniform(new THREE.Vector3());
  private readonly pilotTint: Light = uniform(new THREE.Vector3());
  private keyTints: Light[] = [];
  private beads: Beads | null = null;
  private keys: Keys | null = null;
  private parts: { loupe: THREE.Group; bell: THREE.Group; pilot: THREE.Mesh; loupeLight: THREE.Mesh; dropLight: THREE.Mesh; arriveLight: THREE.Mesh; pilotLight: THREE.Mesh; sockets: THREE.InstancedMesh } | null = null;
  private readonly point = new THREE.Vector3();
  private readonly bellPoint = new THREE.Vector3(0, 0, -BELL_Y);
  private readonly matrix = new THREE.Matrix4();
  private readonly shade = new THREE.Color();
  private readonly glow = new THREE.Color();
  private pace = 0;
  private lastPosition = 0;
  private readonly drawn = { step: -1, count: -1 };
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, right: 0, wrong: 0 };
  private readonly row = { step: 0, count: 0, event: 0, query: false, wait: 0, key: -1, verdict: 0, reward: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.period = parameter(ctx.environment, 'period');
    this.modulus = constant(ctx.environment, 'COUNT_MODULUS');
    if (!Number.isInteger(this.period) || this.period < 1 || !Number.isInteger(this.modulus) || this.modulus < 1) throw new Error(`${ctx.environment.envId} has period ${this.period} and modulus ${this.modulus}`);
    const x = R_CASE + 0.35, z = BELL_Y + BELL + 0.62;
    return new THREE.Box2(new THREE.Vector2(-x, -z), new THREE.Vector2(x, z));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8a93b0);
  }

  private keyOf(frame: EnvFrame): number {
    return countKey(frame) % this.modulus;
  }

  protected override overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    return {
      debug: (frame, kit) => {
        const P = this.period, M = this.modulus, i = frame.state.step, head = frame.frameAt(i);
        const phase = head.words[COUNT_PHASE]!, count = head.words[COUNT_TALLY]!;
        const counted = kit.marks('counted', P);
        for (let j = i - 1; j >= Math.max(0, i - P); j--) {
          const f = frame.frameAt(j);
          if (f.step !== j) break;
          if (f.words[COUNT_EVENT]) {
            along((TAU * f.words[COUNT_PHASE]!) / P, R_SOCKET, at);
            counted.set(at.x, 0.1, at.z, 0.12, BEAD_TINT);
          }
          if (f.words[COUNT_QUERY] === 1) break;
        }
        const round = kit.curve('round', ARC * P + 1, LAMP_TINT);
        for (let k = 0; k <= ARC * phase; k++) {
          along((TAU * k) / (ARC * P), R_LANE, at);
          round.point(at.x, 0.1, at.z);
        }
        along((TAU * phase) / P, R_LANE, at);
        kit.marks('observed', 1).set(at.x, 0.1, at.z, head.words[COUNT_EVENT] ? 0.15 : 0.07, head.words[COUNT_EVENT] ? BEAD_TINT : LAMP_TINT);
        if (head.words[COUNT_QUERY] === 1) kit.marks('bell', 1).set(0, 0.1, -BELL_Y, 0.2, BELL_TINT);
        along((TAU * (count % M)) / M, R_JEWEL, at);
        kit.marks('tally', 1).set(at.x, 0.1, at.z, 0.12, TALLY);
        if (i >= 1) {
          along((TAU * this.keyOf(head)) / M, R_KEY, at);
          kit.marks('key', 1).set(at.x, 0.1, at.z, 0.16, this.accent);
        }
        const before = i >= 1 ? frame.frameAt(i - 1) : null;
        const asked = head.words[COUNT_QUERY] === 1 ? head : before && before.step === i - 1 && before.words[COUNT_QUERY] === 1 ? before : null;
        if (asked) {
          along((TAU * (asked.words[COUNT_TALLY]! % M)) / M, R_KEY - KEY - 0.15, at);
          kit.marks('target', 1).set(at.x, 0.1, at.z, 0.12, BELL_TINT);
        }
        if (asked && asked !== head) {
          along((TAU * this.keyOf(head)) / M, R_KEY, at);
          kit.marks('verdict', 1).set(at.x, 0.1, at.z, 0.24, head.reward > 0 ? GOLD : MISS);
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const P = this.period, M = this.modulus, lit = light.lit;
    this.accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(0x05060a);
    const bead = Math.min(0.2, (0.6 * Math.PI * R_SOCKET) / P);
    this.beads = beads(P, bead, this.falling, this.arrive);
    this.keys = keys(lit, M, this.accent);
    const face = loupe(lit), chime = bell(lit, this.ring, this.wave), jewel = pilot(this.pilotAlpha);
    chime.position.copy(this.bellPoint);
    this.scene.add(desk(lit), dial(lit, P, M, bead, this.sweep), ledger(lit, this.history), this.beads.mesh, this.keys.body, this.keys.glow, face, chime, jewel);

    const emit = (radius: number, tint: Light, geometry: THREE.BufferGeometry = disc(radius)) => {
      const mesh = light.emitter(geometry, tint);
      this.scene.add(mesh);
      return mesh;
    };
    const sockets = glowBank(P, bead * 0.55, BEAD_TINT, 3.2);
    sockets.count = P;
    for (let k = 0; k < P; k++) sockets.setMatrixAt(k, this.matrix.makeScale(0, 0, 0));
    this.scene.add(sockets);
    emit(0.2, this.bellTint).position.copy(this.bellPoint);
    this.keyTints = Array.from({ length: M }, (_, k) => {
      const tint: Light = uniform(new THREE.Vector3());
      along((TAU * k) / M, R_KEY, emit(KEY, tint, new THREE.RingGeometry(KEY * 0.97, KEY + 0.07, 48).rotateX(-Math.PI / 2)).position);
      return tint;
    });
    this.parts = {
      loupe: face, bell: chime, pilot: jewel, sockets,
      loupeLight: emit(0.12, this.loupeTint), dropLight: emit(bead * 0.5, this.dropTint), arriveLight: emit(bead * 0.5, this.arriveTint), pilotLight: emit(0.08, this.pilotTint),
    };
    pool.group.traverse((object) => { object.renderOrder = 10; });

    const tally = this.tally, row = this.row;
    this.legend = {
      title: 'COUNT RECALL',
      rows: [
        { key: 'bell', label: 'bell', tint: BELL_TINT, value: () => (row.query ? 'ringing' : `in ${row.wait} step${row.wait === 1 ? '' : 's'}`) },
        { key: 'event', label: 'event', tint: BEAD_TINT, value: () => (row.event ? 'bead' : 'none') },
        { key: 'count', label: 'counted', tint: TALLY, value: () => `${row.count} · mod ${M} = ${row.count % M}` },
        { key: 'key', label: 'key', tint: this.accent, value: () => (row.key < 0 ? '—' : `${row.key}${row.verdict > 0 ? ' · right' : row.verdict < 0 ? ' · wrong' : ''}`) },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(row.reward) },
      ],
      note: () => `${formatCount(tally.right)} right · ${formatCount(tally.wrong)} wrong`,
      bar: () => tally.right / Math.max(1, tally.right + tally.wrong),
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const parts = this.parts, beadsPart = this.beads, keysPart = this.keys;
    if (!parts || !beadsPart || !keysPart) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const P = this.period, M = this.modulus, i = frame.state.step, head = frame.frameAt(i), ahead = frame.frameAt(i + 1);
    const next = !fast && ahead.step === i + 1 ? ahead : null, t = next ? clamp01(frame.position - i) : 0;
    const phase = head.words[COUNT_PHASE]!, event = head.words[COUNT_EVENT]!, query = head.words[COUNT_QUERY] === 1, count = head.words[COUNT_TALLY]!;

    const presence = beadsPart.presence.values;
    presence.fill(0);
    for (let j = i - 1; j >= Math.max(0, i - P); j--) {
      const f = frame.frameAt(j);
      if (f.step !== j) break;
      const slot = f.words[COUNT_PHASE]!;
      if (f.words[COUNT_EVENT]) presence[slot] = query && next ? 1 - ease(t, drainOf(slot, P)) : 1;
      if (f.words[COUNT_QUERY] === 1) break;
    }
    for (let k = 0; k < P; k++) {
      const v = presence[k]!;
      along((TAU * k) / P, R_SOCKET, this.point);
      parts.sockets.setMatrixAt(k, v > 0.02 ? this.matrix.makeTranslation(this.point.x, 0, this.point.z) : this.matrix.makeScale(0, 0, 0));
      parts.sockets.setColorAt(k, this.shade.setScalar(v));
    }
    parts.sockets.instanceMatrix.needsUpdate = true;
    parts.sockets.instanceColor!.needsUpdate = true;

    const from = (TAU * phase) / P, fall = next ? ease(t, DROP) : 0, dropRadius = R_LANE + (R_SOCKET - R_LANE) * fall;
    this.falling.at.value.set(dropRadius * Math.sin(from), dropRadius * Math.cos(from));
    this.falling.presence.value = event;
    this.falling.gain.value = 1.3 - 0.58 * fall;
    along(from, dropRadius, parts.dropLight.position);
    this.dropTint.value.set(0, 0, 0).addScaledVector(this.vector(BEAD_TINT), 9 * event * (1 - 0.5 * fall));
    const arriving = next ? next.words[COUNT_EVENT]! * ease(t, POP) : 0, to = next ? (TAU * next.words[COUNT_PHASE]!) / P : from;
    this.arrive.at.value.set(R_LANE * Math.sin(to), R_LANE * Math.cos(to));
    this.arrive.presence.value = arriving;
    along(to, R_LANE, parts.arriveLight.position);
    this.arriveTint.value.set(0, 0, 0).addScaledVector(this.vector(BEAD_TINT), 9 * arriving);

    const move = fast ? clamp01(frame.position - i) : next ? snap(span(t, MOVE)) : 0, theta = (TAU * (phase + move)) / P;
    this.sweep.value = Math.min(TAU, Math.max(0, theta));
    along(theta, R_LANE, parts.loupe.position);
    parts.loupe.rotation.y = -theta;
    parts.loupeLight.position.copy(parts.loupe.position);
    this.loupeTint.value.set(0, 0, 0).addScaledVector(this.vector(LAMP_TINT), 11);

    let ring = query ? (next ? 1 - ease(t, RING_OUT) : 1) : 0;
    if (next && next.words[COUNT_QUERY] === 1) ring = Math.max(ring, ease(t, RING_IN));
    this.ring.value = ring;
    parts.bell.scale.setScalar(1 + 0.03 * ring * Math.sin(time * 40));
    this.bellTint.value.set(0, 0, 0).addScaledVector(this.vector(BELL_TINT), 3.2 * ring);

    let pilotAt = count % M, pilotAlpha = 1;
    if (next && query) {
      if (t < FADE_IN[0]) pilotAlpha = 1 - ease(t, FADE_OUT);
      else {
        pilotAt = event * ease(t, HOP);
        pilotAlpha = ease(t, FADE_IN);
      }
    } else if (next) pilotAt += event * ease(t, HOP);
    this.pilotAlpha.value = pilotAlpha;
    along((TAU * pilotAt) / M, R_JEWEL, parts.pilot.position);
    parts.pilotLight.position.copy(parts.pilot.position);
    this.pilotTint.value.set(0, 0, 0).addScaledVector(this.vector(BEAD_TINT), 12 * pilotAlpha);

    const press = keysPart.press.values, halo = keysPart.halo.values, gold = keysPart.gold.values, blue = keysPart.blue.values;
    press.fill(0);
    halo.fill(0);
    gold.fill(0);
    blue.fill(0);
    const now = i >= 1 ? this.keyOf(head) : -1, then = next ? this.keyOf(next) : now;
    if (now >= 0) press[now] = 1;
    if (next && then !== now) {
      const u = ease(t, PRESS);
      if (now >= 0) press[now] = 1 - u;
      press[then] = u;
    }
    const asked = i - phase, question = asked > 0 ? frame.frameAt(asked) : null;
    let answered = -1;
    this.wave.value = 0;
    if (question && question.step === asked && question.words[COUNT_QUERY] === 1) {
      const age = i + t - asked, target = question.words[COUNT_TALLY]! % M;
      if (!fast && age < WAVE) this.wave.value = age / WAVE;
      halo[target] = age <= 1 ? 1 : 1 - smoothstep01((age - 1) / 2);
      const reply = frame.frameAt(asked + 1);
      if (asked + 1 <= i && reply.step === asked + 1) {
        const v = i + t - asked - 1, key = this.keyOf(reply);
        (reply.reward > 0 ? gold : blue)[key] = v <= 0.6 ? 1 : Math.max(0, 1 - (v - 0.6) / 2.4);
        answered = key;
      } else if (next && asked === i) (next.reward > 0 ? gold : blue)[then] = ease(t, JUDGE);
    }
    if (next && next.words[COUNT_QUERY] === 1) {
      const target = next.words[COUNT_TALLY]! % M;
      halo[target] = Math.max(halo[target]!, ease(t, RING_IN));
    }
    for (let k = 0; k < M; k++) {
      const tint = this.keyTints[k]!.value.set(0, 0, 0);
      tint.addScaledVector(this.vector(this.accent), 1.4 * press[k]!).addScaledVector(this.vector(GOLD), 2.4 * gold[k]!).addScaledVector(this.vector(MISS), 1.2 * blue[k]!);
    }

    if (!frame.scrubbed && !fast) {
      let judged: 'goal' | 'harm' | null = null, rang = false;
      for (const item of frame.events) {
        if (item.type === 'goal' || item.type === 'harm') judged = item.type;
        if (item.type === 'cue') rang = true;
      }
      if (judged && answered >= 0) {
        along((TAU * answered) / M, R_KEY, this.point);
        if (judged === 'goal') {
          pool.burst(this.point, GOLD, 0.7, time);
          pool.pulse.fire(0.3);
        } else pool.burst(this.point, MISS, 0.3, time);
      }
      if (rang) pool.burst(this.bellPoint, BELL_TINT, 0.3, time);
    }

    this.recent(frame.eventLog, i);
    this.score(frame.eventLog, i);
    const before = i >= 1 ? frame.frameAt(i - 1) : null, scored = before !== null && before.step === i - 1 && before.words[COUNT_QUERY] === 1;
    Object.assign(this.row, {
      step: i, count, event, query, wait: phase === 0 && !query ? P : P - phase, key: now,
      verdict: scored ? (head.reward > 0 ? 1 : -1) : 0, reward: head.reward,
    });
  }

  private readonly vectors = new Map<number, THREE.Vector3>();

  private vector(hex: number): THREE.Vector3 {
    let found = this.vectors.get(hex);
    if (!found) {
      this.glow.set(hex);
      found = new THREE.Vector3(this.glow.r, this.glow.g, this.glow.b);
      this.vectors.set(hex, found);
    }
    return found;
  }

  private recent(log: EventStream, at: number): void {
    if (this.drawn.step === at && this.drawn.count === log.count) return;
    this.drawn.step = at;
    this.drawn.count = log.count;
    const values = this.history.values;
    values.fill(0);
    let k = LEDGER - 1;
    for (let b = Math.floor(at / log.blockFrames); b >= 0 && k >= 0; b--) {
      const block = log.blocks[b];
      if (!block) break;
      for (let n = block.steps.length - 1; n >= 0 && k >= 0; n--) {
        if (block.steps[n]! > at) continue;
        if (block.kinds[n] === GOAL) values[k--] = 1;
        else if (block.kinds[n] === HARM) values[k--] = -1;
      }
    }
  }

  private score(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, right: 0, wrong: 0 });
    if (at === tally.through) return;
    tally.right += countEvents(log, 'goal', tally.through, at);
    tally.wrong += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.parts = null;
    this.beads = null;
    this.keys = null;
  }
}

export default (() => new CountRecallScene()) satisfies SceneFactory;
