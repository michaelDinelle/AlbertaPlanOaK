import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { hash2f } from '@kit/rng';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { RIVER_STATION, riverSwimDecoder, riverUpstream } from '@sim/envs/riverswim';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';
import { BUOY, CAMP, FOAM, GOLD, LAMP, SPAN, SPRING, SWIMMER, SWIRL, buoyMaterial, campMaterial, disc, flat, ground, reedMaterial, springMaterial, stoneMaterial, swimmerMaterial, swirlMaterial, treeMaterial, type Level, type Maps } from './art';
import { CAP, Course, HALF } from './course';

const ACCENT = TOKENS.accents.RiverSwim!;
const HOP = 0.24;
const TURN = 0.2;
const WINDOW = 2048;
const SURVEY_EVERY = 0.25;
const LOOK_EVERY = 1;
const FLARE_LIFE = 1.2;
const EMBER_LIFE = 0.7;
const SWIRL_LIFE = 1;
const SWIRLS = 6;
const FIRE = { s: -CAP - 0.6, lateral: 0.3 } as const;
const HEAD = 0.21;
const LUNGE = CAP + 0.5 - HEAD;
const NUDGE = 0.1;
const TAP = 0.05;
const SIDE = HALF - 0.17;
const STROKE = 1.6;
const DIP = 0.12;
const ARM = 0.19;
const SHOULDER = new THREE.Vector2(0.065, 0.066);
const STROKE_TINT = 0xdcf4ff;
const AIM = 0.2;

type Kind = 'glide' | 'swept' | 'hold' | 'touch';
interface Move { from: number; to: number; upstream: boolean; kind: Kind }
interface Swirl { born: number; at: THREE.Vector3 }
interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute }

const ease = (t: number) => t * t * (3 - 2 * t);
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const plain = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '').replace('-', '−');

const STROKES: Record<Kind, readonly [string, string]> = {
  glide: ['upstream · advanced', 'downstream'],
  swept: ['upstream · swept back', 'downstream · carried upstream'],
  hold: ['upstream · held', 'downstream · held'],
  touch: ['upstream · reached the source', 'downstream · touched the bank'],
};

class RiverSwimScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['dusk', 'hemiSky'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'riverSwim';

  private stations = 2;
  private course: Course | null = null;
  private textures: THREE.Texture[] = [];
  private swimmer: THREE.Mesh | null = null;
  private buoys: Batch | null = null;
  private swirls: Batch | null = null;
  private lamps: Level[] = [];
  private readonly clock = uniform(0);
  private readonly flare = uniform(0);
  private readonly ember = uniform(0);
  private readonly fire = uniform(1);
  private readonly arms = uniform(new THREE.Vector4(0.25, 0.06, -0.1, -0.03));
  private readonly elbows = uniform(new THREE.Vector4(0.16, 0.07, -0.02, -0.05));
  private readonly dips = uniform(new THREE.Vector2(0, 1));
  private readonly bow = uniform(0);
  private readonly wake = uniform(0);
  private readonly glow = uniform(0);

  private readonly walk = { from: 0, to: 0, t: 1, kind: 'hold' as Kind, facing: 1 };
  private shown = 0;
  private face = 0;
  private faceFrom = 0;
  private faceT = 1;
  private phase = 0.9;
  private step = -1;
  private simTime = 0;
  private seconds = 0;
  private lastPosition = 0;
  private flared = -Infinity;
  private warmed = -Infinity;
  private surveyed = -Infinity;
  private surveyStep = -1;
  private readonly swirlSlots: Swirl[] = Array.from({ length: SWIRLS }, () => ({ born: -Infinity, at: new THREE.Vector3() }));
  private swirlCursor = 0;
  private occupancy = new Float32Array(0);
  private tally = { advanced: 0, held: 0, swept: 0 };
  private window = 0;

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { station: 0, step: 0, reward: 0, stroke: '—', goal: undefined as number | undefined, known: false, looked: -Infinity };

  override legend: LegendSpec = {
    title: 'RIVER SWIM',
    rows: [
      { key: 'swimmer', label: 'swimmer', tint: ACCENT, value: () => `station ${this.ledger.station + 1} of ${this.stations}` },
      { key: 'stroke', label: 'stroke', tint: STROKE_TINT, value: () => this.ledger.stroke },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => (this.ledger.step > 0 ? plain(this.ledger.reward) : '—') },
      { key: 'source', label: 'last source reward', tint: GOLD, value: () => this.since(this.ledger.goal) },
      { key: 'lamps', label: 'time spent', tint: LAMP, value: () => `bank ${formatPercent(this.occupancy[0], 0)} · source ${formatPercent(this.occupancy[this.stations - 1], 0)}` },
      { key: 'current', label: 'upstream', tint: FOAM, value: () => this.odds() },
    ],
    note: () => (this.window === 1 ? 'last step' : `last ${formatCount(this.window)} steps`),
    bar: () => this.ledger.station / Math.max(1, this.stations - 1),
  };

  private odds(): string {
    const { advanced, held, swept } = this.tally, total = advanced + held + swept;
    return total === 0 ? 'none' : `${formatPercent(advanced / total, 0)} advanced · ${formatPercent(held / total, 0)} held · ${formatPercent(swept / total, 0)} swept back`;
  }

  private since(found: number | undefined): string {
    const step = this.ledger.step;
    if (!this.ledger.known) return '…';
    return found === undefined ? 'none yet' : found === step ? 'now' : `${formatCount(step - found)} ${step - found === 1 ? 'step' : 'steps'} ago`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.stations = parameter(ctx.environment, 'states');
    this.course = new Course(this.stations);
    this.occupancy = new Float32Array(this.stations);
    const box = this.course.bounds();
    return new THREE.Box2(new THREE.Vector2(box.min.x, -box.max.y), new THREE.Vector2(box.max.x, -box.min.y));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa9b3dc).multiplyScalar(0.95);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const samples = 160;
    return {
      debug: (frame, kit) => {
        const course = this.course!, n = this.stations, at = this.at;
        const line = kit.curve('course', samples + 1, TOKENS.uiCyan);
        for (let i = 0; i <= samples; i++) {
          course.world((course.length * i) / samples, 0, at);
          line.point(at.x, 0.05, at.z);
        }
        const marks = kit.marks('stations', n), lamps = kit.marks('occupancy', n);
        for (let k = 0; k < n; k++) {
          course.world(course.station(k), -SIDE, at);
          marks.set(at.x, 0.05, at.z, 0.07, TOKENS.uiCyan);
          if (this.occupancy[k]! > 0) {
            course.world(course.station(k), SIDE, at);
            lamps.set(at.x, 0.05, at.z, 0.03 + 0.22 * Math.sqrt(this.occupancy[k]!), LAMP);
          }
        }
        const ends = kit.marks('ends', 2);
        course.world(course.station(0) - LUNGE, 0, at);
        ends.set(at.x, 0.05, at.z, 0.1, 0xe0c99a);
        course.world(course.station(n - 1) + LUNGE, 0, at);
        ends.set(at.x, 0.05, at.z, 0.1, TOKENS.celebrationGold);
        const truth = frame.state.pose[RIVER_STATION]!;
        course.world(course.station(truth), 0, at);
        kit.marks('swimmer', 1).set(at.x, 0.06, at.z, 0.16, ACCENT);
        const step = frame.state.step, now = frame.frameAt(step), before = frame.frameAt(step - 1);
        const intent = kit.curve('intent', 5, STROKE_TINT), moved = kit.curve('moved', 9, 0xffa05a);
        if (step < 1 || now.step !== step || before.step !== step - 1) return;
        const move = this.classify(before, now), from = course.station(move.from), to = this.rest(move);
        this.arrow(intent, from, move.upstream ? 0.55 : -0.55);
        if (to !== from) for (let i = 0; i <= 8; i++) {
          course.world(from + ((to - from) * i) / 8, 0.08, at);
          moved.point(at.x, 0.07, at.z);
        }
      },
    };
  }

  private arrow(curve: { point(x: number, y: number, z: number): void }, from: number, length: number): void {
    const course = this.course!, tip = from + length, back = tip - Math.sign(length) * 0.14;
    course.world(from, AIM, this.at);
    curve.point(this.at.x, 0.07, this.at.z);
    course.world(tip, AIM, this.at);
    curve.point(this.at.x, 0.07, this.at.z);
    course.world(back, AIM + 0.1, this.other);
    curve.point(this.other.x, 0.07, this.other.z);
    curve.point(this.at.x, 0.07, this.at.z);
    course.world(back, AIM - 0.1, this.other);
    curve.point(this.other.x, 0.07, this.other.z);
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const course = this.course!, lit = light.lit, n = this.stations;
    this.scene.background = new THREE.Color(0x0b1410);
    pool.group.renderOrder = 10;
    const maps: Maps = { mottle: fbm(ctx.seed ^ 0x51f7, { size: 256, frequency: 4, octaves: 5 }), grain: fbm(ctx.seed ^ 0x9e21, { size: 256, frequency: 16, octaves: 3 }) };
    this.textures.push(maps.mottle, maps.grain);
    const floor = ground(course, maps, lit, this.clock);
    floor.renderOrder = 0;
    this.scene.add(floor);

    const place = (mesh: THREE.Mesh, s: number, lateral: number, turn: number, order: number) => {
      course.world(s, lateral, mesh.position);
      mesh.rotation.y = course.heading(Math.min(Math.max(s, 0), course.length)) + turn;
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return mesh;
    };
    const batch = (geometry: THREE.BufferGeometry, name: string, material: THREE.Material, states: number[][], matrices: THREE.Matrix4[], order: number): Batch => {
      const state = new THREE.InstancedBufferAttribute(new Float32Array(states.flat()), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, state);
      const mesh = instancedMesh(geometry, material, matrices.length);
      matrices.forEach((matrix, i) => mesh.setMatrixAt(i, matrix));
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return { mesh, state };
    };

    const trees = this.plant(ctx.seed), stones = this.stones(ctx.seed), reeds = this.reeds(ctx.seed);
    batch(flat(1, 1), 'aReed', reedMaterial(lit, this.clock), reeds.states, reeds.matrices, 1);
    batch(flat(1, 1), 'aStone', stoneMaterial(maps, lit), stones.states, stones.matrices, 2);
    batch(flat(1, 1), 'aTree', treeMaterial(maps, lit), trees.states, trees.matrices, 3);

    const spots: THREE.Matrix4[] = [], seeds: number[][] = [];
    for (let k = 0; k < n; k++) {
      const s = course.station(k);
      course.world(s, -SIDE, this.at);
      spots.push(new THREE.Matrix4().compose(this.at, this.quaternion.setFromAxisAngle(this.axis, course.heading(s)), this.scale.setScalar(1)));
      seeds.push([0, hash2f(k, 7, ctx.seed) * 6.28, 0, 0]);
      const level = uniform(0);
      this.lamps.push(level);
      const lamp = light.emitter(disc(0.05), rgb(LAMP).mul(level.mul(30)));
      lamp.position.copy(this.at);
      this.scene.add(lamp);
    }
    this.buoys = batch(flat(BUOY, BUOY), 'aBuoy', buoyMaterial(lit, this.clock), seeds, spots, 4);

    place(new THREE.Mesh(flat(SPRING, SPRING), springMaterial(maps, lit, this.clock, this.flare)), course.length + CAP - 0.05, 0, 0, 5);
    place(light.emitter(disc(0.13), rgb(GOLD).mul(this.flare.mul(2).add(2.5))), course.length + CAP + 0.05, 0, 0, 1);
    place(new THREE.Mesh(flat(CAMP, CAMP), campMaterial(maps, lit, this.clock, this.ember)), FIRE.s, FIRE.lateral, 0, 5);
    place(light.emitter(disc(0.05), rgb(0xff9a4a).mul(this.fire.mul(16))), FIRE.s, FIRE.lateral, 0, 1);

    this.swirls = batch(flat(SWIRL, SWIRL), 'aSwirl', swirlMaterial(lit), Array.from({ length: SWIRLS }, () => [0, 0, 0, 0]), Array.from({ length: SWIRLS }, () => new THREE.Matrix4().makeScale(0, 0, 0)), 6);
    this.swimmer = new THREE.Mesh(flat(SWIMMER, SWIMMER), swimmerMaterial(ACCENT, maps, lit, { arms: this.arms, elbows: this.elbows, dips: this.dips, bow: this.bow, wake: this.wake, glow: this.glow, clock: this.clock }));
    this.swimmer.renderOrder = 7;
    this.swimmer.frustumCulled = false;
    this.scene.add(this.swimmer);
  }

  private plant(seed: number): { matrices: THREE.Matrix4[]; states: number[][] } {
    const course = this.course!, box = course.bounds(), matrices: THREE.Matrix4[] = [], states: number[][] = [];
    const beach = course.point(-CAP - 0.5, new THREE.Vector2()), source = course.point(course.length + CAP, new THREE.Vector2());
    const pitch = 0.82;
    for (let j = Math.floor((box.min.y - 6) / pitch); j <= Math.ceil((box.max.y + 6) / pitch); j++) {
      for (let i = Math.floor((box.min.x - 9) / pitch); i <= Math.ceil((box.max.x + 9) / pitch); i++) {
        if (hash2f(i, j, seed ^ 0x7ee) > 0.72) continue;
        const x = (i + 0.15 + 0.7 * hash2f(i, j, seed ^ 0x11)) * pitch, y = (j + 0.15 + 0.7 * hash2f(i, j, seed ^ 0x22)) * pitch;
        const r = 0.24 + 0.24 * hash2f(i, j, seed ^ 0x33), clear = course.near(x, y).d - HALF - 0.3 - r;
        if (clear < 0 || Math.hypot(x - beach.x, y - beach.y) < 1.25 + r || Math.hypot(x - source.x, y - source.y) < 1.05 + r) continue;
        const size = (2 * SPAN * r) / 0.84;
        matrices.push(new THREE.Matrix4().compose(this.at.set(x, 0, -y), this.quaternion.setFromAxisAngle(this.axis, hash2f(i, j, seed ^ 0x44) * 6.28), this.scale.set(size, 1, size)));
        states.push([hash2f(i, j, seed ^ 0x55), hash2f(i, j, seed ^ 0x66), 0, 0]);
      }
    }
    return { matrices, states };
  }

  private reeds(seed: number): { matrices: THREE.Matrix4[]; states: number[][] } {
    const course = this.course!, matrices: THREE.Matrix4[] = [], states: number[][] = [];
    let k = 0;
    for (const bank of [1, -1]) {
      for (let s = 0.2; s < course.length - 0.7; s += 0.3, k++) {
        if (hash2f(k, 11, seed) > 0.42) continue;
        const lateral = bank * (HALF + 0.14 + 0.1 * hash2f(k, 12, seed)), size = SPAN * (0.34 + 0.2 * hash2f(k, 13, seed));
        course.world(s + 0.2 * hash2f(k, 14, seed), lateral, this.at);
        matrices.push(new THREE.Matrix4().compose(this.at, this.quaternion.setFromAxisAngle(this.axis, hash2f(k, 15, seed) * 6.28), this.scale.set(size, 1, size)));
        states.push([hash2f(k, 16, seed), hash2f(k, 17, seed), 0, 0]);
      }
    }
    return { matrices, states };
  }

  private stones(seed: number): { matrices: THREE.Matrix4[]; states: number[][] } {
    const course = this.course!, matrices: THREE.Matrix4[] = [], states: number[][] = [];
    const end = course.length + CAP * 0.3, heading = course.heading(course.length), centre = course.point(end, new THREE.Vector2());
    const add = (x: number, y: number, r: number, k: number) => {
      const size = (2 * SPAN * r) / 0.8;
      matrices.push(new THREE.Matrix4().compose(this.at.set(x, 0, -y), this.quaternion.setFromAxisAngle(this.axis, hash2f(k, 1, seed) * 6.28), this.scale.set(size, 1, size * (0.8 + 0.3 * hash2f(k, 2, seed)))));
      states.push([hash2f(k, 3, seed), hash2f(k, 4, seed), 0, 0]);
    };
    const ring = [-2.05, -1.62, -1.2, -0.62, 0.62, 1.2, 1.62, 2.05];
    ring.forEach((angle, k) => {
      const a = heading + angle + (hash2f(k, 5, seed) - 0.5) * 0.2, reach = HALF + 0.12 + 0.08 * hash2f(k, 6, seed), r = 0.15 + 0.1 * hash2f(k, 7, seed) + (Math.abs(angle) < 1 ? 0.08 : 0);
      add(centre.x + Math.cos(a) * (reach + r * 0.6), centre.y + Math.sin(a) * (reach + r * 0.6), r, k);
    });
    [-0.28, 0.28].forEach((lateral, k) => add(centre.x + Math.cos(heading) * 0.95 - Math.sin(heading) * lateral, centre.y + Math.sin(heading) * 0.95 + Math.cos(heading) * lateral, 0.3, 20 + k));
    return { matrices, states };
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = Math.max(time - this.simTime, frame.dtReal, 0);
    this.simTime = time;
    this.seconds += frame.dtReal;
    const step = frame.state.step;
    if (frame.scrubbed || step < this.step || this.step < 0) this.settle(frame, step, pool);
    else for (let s = this.step + 1; s <= step && this.advance(frame, s, pool); s++);
    this.step = step;
    this.ledger.step = step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) this.phase = step * Math.PI + 0.9;
    else this.phase += Math.PI * Math.min(Math.max(frame.position - this.lastPosition, 0), frame.dtReal * this.controlHz * STROKE);
    this.lastPosition = frame.position;
    if (this.seconds - this.ledger.looked > LOOK_EVERY) {
      const log = frame.eventLog, found = lastEvent(log, 'goal', step);
      let known = true;
      for (let b = Math.floor((found ?? 0) / log.blockFrames); b <= Math.floor(step / log.blockFrames) && known; b++) known = log.blocks[b] !== undefined;
      this.ledger.goal = found;
      this.ledger.known = known;
      this.ledger.looked = this.seconds;
    }
    if (step !== this.surveyStep && this.seconds - this.surveyed > SURVEY_EVERY) this.survey(frame, step);
    this.clock.value = this.seconds;
    this.animateSwimmer(dt);
    this.animateSwirls(time);
    const flare = clamp01(1 - (time - this.flared) / FLARE_LIFE);
    this.flare.value = flare * flare;
    this.glow.value = clamp01(1 - (time - this.flared) / (FLARE_LIFE * 0.5));
    this.ember.value = clamp01(1 - (time - this.warmed) / EMBER_LIFE);
    this.fire.value = 1 + 0.6 * this.ember.value + 0.06 * Math.sin(this.seconds * 8.3) * Math.sin(this.seconds * 3.1);
  }

  private settle(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    const now = frame.frameAt(step), before = frame.frameAt(step - 1);
    pool.clear();
    this.flared = this.warmed = -Infinity;
    for (const swirl of this.swirlSlots) swirl.born = -Infinity;
    const reach = Math.ceil(FLARE_LIFE * frame.controlHz);
    for (let s = Math.max(1, step - reach); s <= step; s++) {
      const later = frame.frameAt(s), earlier = frame.frameAt(s - 1);
      if (later.step === s && earlier.step === s - 1) this.effects(earlier, later, this.classify(earlier, later), s, frame.controlHz, null);
    }
    const move = step >= 1 && before.step === step - 1 ? this.classify(before, now) : null;
    const upstream = step < 1 || riverUpstream(now);
    this.walk.from = this.walk.to = this.shown = move ? this.rest(move) : this.course!.station(now.words[0]!);
    this.walk.t = 1;
    this.walk.kind = 'hold';
    this.walk.facing = upstream ? 1 : -1;
    this.face = this.faceFrom = upstream ? 0 : Math.PI;
    this.faceT = 1;
    this.record(now, move);
    this.ledger.looked = -Infinity;
    this.surveyed = -Infinity;
    this.surveyStep = -1;
  }

  private classify(before: EnvFrame, now: EnvFrame): Move {
    const from = before.words[0]!, to = now.words[0]!, upstream = riverUpstream(now), delta = to - from;
    if (Math.abs(delta) > 1 || to >= this.stations) throw new Error(`RiverSwim step ${now.step} moved from station ${from} to ${to}`);
    const kind: Kind = delta !== 0 ? (delta > 0) === upstream ? 'glide' : 'swept' : now.reward > 0 ? 'touch' : 'hold';
    return { from, to, upstream, kind };
  }

  private rest(move: Move): number {
    return this.course!.station(move.to) + (move.kind === 'touch' ? (move.upstream ? LUNGE : -LUNGE) : 0);
  }

  private record(now: EnvFrame, move: Move | null): void {
    this.ledger.station = now.words[0]!;
    this.ledger.reward = now.reward;
    this.ledger.stroke = move ? STROKES[move.kind][move.upstream ? 0 : 1] : '—';
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool): boolean {
    const now = frame.frameAt(s), before = frame.frameAt(s - 1);
    if (now.step !== s || before.step !== s - 1) {
      this.settle(frame, frame.state.step, pool);
      return false;
    }
    const move = this.classify(before, now);
    this.record(now, move);
    this.effects(before, now, move, s, frame.controlHz, pool);
    const walk = this.walk;
    walk.from = this.shown;
    walk.to = this.rest(move);
    walk.kind = move.kind;
    walk.facing = move.upstream ? 1 : -1;
    walk.t = 0;
    const face = move.upstream ? 0 : Math.PI;
    if (face !== this.face || this.faceT < 1) {
      this.faceFrom = this.faceAngle();
      this.face = face;
      this.faceT = 0;
    }
    return true;
  }

  private effects(before: EnvFrame, now: EnvFrame, move: Move, s: number, hz: number, pool: CelebrationPool | null): void {
    const course = this.course!, born = s / hz;
    for (const event of riverSwimDecoder.deriveEvents(before, now)) {
      if (event.type !== 'goal') continue;
      this.flared = born;
      this.ledger.goal = s;
      this.ledger.known = true;
      if (!pool) continue;
      course.world(course.length + CAP + 0.25, 0, this.at);
      pool.burst(this.at.setY(38), TOKENS.celebrationGold, 0.5, born);
      pool.ripple(this.at.setY(0), TOKENS.celebrationGold, 0.24, born);
    }
    if (move.kind === 'touch' && !move.upstream) this.warmed = born;
    if (move.kind === 'swept') {
      const swirl = this.swirlSlots[this.swirlCursor]!;
      this.swirlCursor = (this.swirlCursor + 1) % SWIRLS;
      swirl.born = born;
      course.world(course.station(move.from), 0, swirl.at);
    }
  }

  private faceAngle(): number {
    const k = ease(clamp01(this.faceT));
    return this.faceFrom + (this.face - this.faceFrom) * k;
  }

  private animateSwimmer(dt: number): void {
    const swimmer = this.swimmer, course = this.course;
    if (!swimmer || !course) return;
    const walk = this.walk;
    walk.t = Math.min(1, walk.t + dt / HOP);
    this.faceT = Math.min(1, this.faceT + dt / TURN);
    const arc = Math.sin(Math.PI * walk.t), bump = walk.kind === 'hold' ? NUDGE : walk.kind === 'touch' ? TAP : 0;
    this.shown = walk.from + (walk.to - walk.from) * ease(walk.t) + walk.facing * bump * arc;
    const wobble = walk.kind === 'swept' ? 0.55 * arc * walk.facing : 0;
    const angle = course.heading(this.shown) + this.faceAngle() + wobble;
    course.world(this.shown, 0, swimmer.position);
    swimmer.rotation.y = angle;
    this.strokeArms();
    const moving = walk.t < 1 && Math.abs(walk.to - walk.from) > 0.05 ? 1 : 0;
    this.bow.value += ((walk.facing > 0 ? 1 : 0.25) - this.bow.value) * clamp01(dt * 6);
    this.wake.value += (Math.max(moving, 0.35) - this.wake.value) * clamp01(dt * 8);
  }

  private strokeArms(): void {
    const reach = (phase: number, side: number) => {
      const u = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      if (u < Math.PI) {
        const t = u / Math.PI, a = 0.22 + (Math.PI - 0.44) * t, hx = SHOULDER.x + ARM * Math.cos(a), hy = side * (SHOULDER.y - 0.035 * Math.sin(a));
        return [hx, hy, (SHOULDER.x + hx) / 2, (side * SHOULDER.y + hy) / 2, clamp01(Math.min(t, 1 - t) / DIP)] as const;
      }
      const t = (u - Math.PI) / Math.PI, a = Math.PI - 0.22 - (Math.PI - 0.44) * t, lift = Math.sin(Math.PI * t);
      const hx = SHOULDER.x + ARM * Math.cos(a), hy = side * (SHOULDER.y + 0.05 + 0.07 * lift);
      return [hx, hy, SHOULDER.x + 0.5 * ARM * Math.cos(a) - 0.03 * lift, side * (SHOULDER.y + 0.05 + 0.1 * lift), 0] as const;
    };
    const left = reach(this.phase, 1), right = reach(this.phase + Math.PI, -1);
    this.arms.value.set(left[0], left[1], right[0], right[1]);
    this.elbows.value.set(left[2], left[3], right[2], right[3]);
    this.dips.value.set(left[4], right[4]);
  }

  private animateSwirls(time: number): void {
    const swirls = this.swirls;
    if (!swirls) return;
    this.swirlSlots.forEach((swirl, i) => {
      const age = (time - swirl.born) / SWIRL_LIFE;
      if (!(age >= 0 && age < 1)) {
        swirls.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
        swirls.state.setXYZW(i, 0, 0, 0, 0);
        return;
      }
      swirls.mesh.setMatrixAt(i, this.matrix.compose(swirl.at, this.quaternion.setFromAxisAngle(this.axis, age * 2.4), this.scale.setScalar(1)));
      swirls.state.setXYZW(i, age, 1, 0, 0);
    });
    swirls.mesh.instanceMatrix.needsUpdate = true;
    swirls.state.needsUpdate = true;
  }

  private survey(frame: FrameUpdate, step: number): void {
    const counts = new Float32Array(this.stations), tally = { advanced: 0, held: 0, swept: 0 };
    let total = 0, complete = true, before = -1;
    for (let s = Math.max(0, step - WINDOW); s <= step; s++) {
      const now = frame.frameAt(s);
      if (now.step !== s) {
        complete = false;
        before = -1;
        continue;
      }
      const station = now.words[0]!;
      if (s > step - WINDOW) {
        counts[station]! += 1;
        total++;
      }
      if (before >= 0 && riverUpstream(now)) {
        if (station > before || now.reward > 0) tally.advanced++;
        else if (station < before) tally.swept++;
        else tally.held++;
      }
      before = station;
    }
    for (let k = 0; k < this.stations; k++) {
      const share = total > 0 ? counts[k]! / total : 0;
      this.occupancy[k] = share;
      const level = share > 0 ? 0.12 + 0.88 * Math.sqrt(share) : 0;
      this.lamps[k]!.value = level;
      this.buoys?.state.setX(k, level);
    }
    if (this.buoys) this.buoys.state.needsUpdate = true;
    this.tally = tally;
    this.window = total;
    this.surveyed = this.seconds;
    this.surveyStep = complete ? step : -1;
  }
}

export default (() => new RiverSwimScene()) satisfies SceneFactory;
