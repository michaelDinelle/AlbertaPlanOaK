import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, envelope, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { HANOI_CODES, HANOI_PEGS, hanoiCode, hanoiDistance, hanoiHeight, hanoiMoved, hanoiPegOf, hanoiRoute, hanoiTarget, type HanoiMove } from '@sim/envs/hanoi';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';
import {
  FRONT, GILT, GOLD, HALF, KEY_H, KEY_W, KEY_Z, PLATE, PLATE_Z, RIM, TILT, TOP, consoleMaterial, diskMaterial, diskQuad, flashMaterial, flat, ghostMaterial, keyMaterial, keyQuad,
  keyX, layoutOf, pegMaterial, pegX, radiusOf, roomMaterial, trailMaterial, type Layout, type Level,
} from './art';

type N<T extends string> = THREE.Node<T>;

const NAMES = ['left', 'middle', 'right'] as const;
const LACQUER = [0xc4432f, 0xdc7a2c, 0xe2b13c, 0x93ac3e, 0x3f9a78, 0x2f7fb0, 0x4c5cb8, 0x8b52a8] as const;
const REFUSED = TOKENS.failureBlue;
const CUE = TOKENS.cueMagenta;
const LIVE = 0xfff1d6;
const FAST = 3;
const LINGER = 0.45;
const TARGET_SECONDS = 0.22;
const CHEER_SECONDS = 0.6;

interface Flight { move: HanoiMove; x: number; top: number; threaded: number; grounded: number; reach: number }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }
interface Path { ends: THREE.UniformNode<'vec4', THREE.Vector4>; reach: Level; alpha: Level }
interface Flash { mesh: THREE.Mesh; lamp: Lamp; power: Level }

const RIM_Y = RIM * TILT;
const REFUSED_LIGHT = new THREE.Color(REFUSED);
const CUE_LIGHT = new THREE.Color(CUE);

const levelTop = (layout: Layout, level: number) => -(level + 1) * layout.height;

class HanoiScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'hanoi';

  private layout: Layout = layoutOf(4, 6);
  private readonly bounds = new THREE.Box2();
  private remaps = false;
  private accent = 0xd6a36e;
  private colors: THREE.Color[] = [];
  private readonly weights = uniform(new THREE.Vector3());
  private readonly disposables: { dispose(): void }[] = [];

  private disks: THREE.InstancedMesh | null = null;
  private shape: THREE.InstancedBufferAttribute | null = null;
  private body: THREE.InstancedBufferAttribute | null = null;
  private look: THREE.InstancedBufferAttribute | null = null;
  private glow: THREE.InstancedBufferAttribute | null = null;
  private live: Path | null = null;
  private last: Path | null = null;
  private flashes: Flash[] = [];
  private beacons: Lamp[] = [];
  private keyLamp: Lamp | null = null;

  private pace = 0;
  private lastPosition = 0;
  private cheerAt = -Infinity;
  private cheerPeg = 0;
  private shimmerAt = -Infinity;
  private log: FrameUpdate['eventLog'] | null = null;
  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly scratch = new THREE.Color();
  private readonly ledger = { step: 0, target: 2, press: '—', reward: 0, distance: 0, total: 1 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, disks = parameter(env, 'disks');
    this.layout = layoutOf(disks, env.actions);
    this.remaps = parameter(env, 'remapRate') > 0;
    this.accent = this.accentOf(ctx);
    this.colors = Array.from({ length: disks }, (_, size) => new THREE.Color(LACQUER[Math.round((disks - 1 - size) * (LACQUER.length - 1) / Math.max(1, disks - 1))]!));
    this.ledger.total = 2 ** disks - 1;
    this.legend = this.legendFor();
    const top = this.layout.fly - this.layout.height - RIM_Y - 0.4;
    return this.bounds.set(new THREE.Vector2(-HALF - 0.35, top), new THREE.Vector2(HALF + 0.35, TOP + FRONT + 0.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffe6cc).multiplyScalar(0.7);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighRadius: 2.4, sighHeight: 1 };
  }

  private legendFor(): LegendSpec {
    const rows: LegendSpec['rows'][number][] = [
      { key: 'target', label: 'target', tint: GOLD, value: () => `${NAMES[this.ledger.target]} peg` },
      { key: 'press', label: 'press', tint: this.accent, value: () => this.ledger.press },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => this.ledger.reward > 0 ? `+${formatCount(this.ledger.reward)}` : formatCount(this.ledger.reward) },
      { key: 'solved', label: 'solved', tint: GILT, value: () => this.solved() },
      { key: 'round', label: 'this round', tint: 0xdcf4ff, value: () => this.round() },
    ];
    if (this.remaps) rows.push({ key: 'remap', label: 'keys remapped', tint: CUE, value: () => this.remapped() });
    return {
      title: 'TOWER OF HANOI',
      rows,
      note: () => `${formatCount(this.ledger.distance)} of ${formatCount(this.ledger.total)} moves to go`,
      bar: () => 1 - this.ledger.distance / this.ledger.total,
    };
  }

  private solved(): string {
    const count = this.log ? countEvents(this.log, 'goal', -1, this.ledger.step) : 0;
    return count ? `${formatCount(count)}×` : 'none yet';
  }

  private round(): string {
    const since = this.log ? lastEvent(this.log, 'goal', this.ledger.step) ?? 0 : 0;
    return `${formatCount(this.ledger.step - since)} steps · fewest ${formatCount(this.ledger.total)}`;
  }

  private remapped(): string {
    if (!this.log) return '—';
    const count = countEvents(this.log, 'cue', -1, this.ledger.step), last = lastEvent(this.log, 'cue', this.ledger.step);
    return last === undefined ? 'not yet' : `${formatCount(count)}× · ${formatCount(this.ledger.step - last)} steps ago`;
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.groundTruth(frame, kit) };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    const layout = this.layout, n = layout.disks;
    this.scene.background = new THREE.Color(0x0d0a08);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const add = (mesh: THREE.Mesh, order: number) => {
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return mesh;
    };
    const centre = this.bounds.getCenter(new THREE.Vector2()), size = this.bounds.getSize(new THREE.Vector2()).multiplyScalar(3);
    add(new THREE.Mesh(flat(size.x, size.y), roomMaterial(light)), 0).position.set(centre.x, 0, centre.y);
    add(new THREE.Mesh(flat(2 * HALF + 0.4, TOP + FRONT + TOP + 0.4), consoleMaterial(light, layout)), 1).position.set(0, 0, FRONT / 2);
    const tower = n * layout.height + 2 * RIM_Y + 0.4, span = 2 * (HALF + 0.2);
    add(new THREE.Mesh(flat(span, tower), ghostMaterial(layout, this.weights)), 3).position.set(0, 0, -tower / 2 + RIM_Y + 0.2);
    const reach = -layout.tip + 0.3;
    add(new THREE.Mesh(flat(span, reach), pegMaterial(light, layout, this.weights)), 4).position.set(0, 0, -reach / 2 + 0.15);

    const path = (tint: number, order: number): Path => {
      const ends = uniform(new THREE.Vector4()), reachLevel = uniform(1), alpha = uniform(0);
      add(new THREE.Mesh(flat(size.x, size.y), trailMaterial(layout, tint, ends, reachLevel, alpha)), order).position.set(centre.x, 0, centre.y);
      return { ends, reach: reachLevel, alpha };
    };
    this.last = path(this.accent, 5);
    this.live = path(LIVE, 5);

    const quad = diskQuad(layout);
    this.shape = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.body = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const disks = instancedMesh(flat(quad.width, quad.depth).translate(0, 0, layout.height / 2), diskMaterial(light, layout, this.shape, this.body, this.weights), n);
    add(disks, 6);
    this.disposables.push(disks);
    this.disks = disks;

    this.look = new THREE.InstancedBufferAttribute(new Float32Array(layout.keys * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.glow = new THREE.InstancedBufferAttribute(new Float32Array(layout.keys * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const keys = instancedMesh(flat(keyQuad.width, keyQuad.depth), keyMaterial(light, this.look, this.glow, this.accent, REFUSED, CUE), layout.keys);
    add(keys, 2);
    this.disposables.push(keys);
    for (let key = 0; key < layout.keys; key++) {
      keys.setMatrixAt(key, this.matrix.makeTranslation(keyX(layout, key), 0, KEY_Z));
      this.look.setXYZW(key, key + 1, 0, 0, 1);
    }
    keys.instanceMatrix.needsUpdate = true;

    const lamp = (geometry: THREE.BufferGeometry, x: number, z: number): Lamp => {
      const tint = new THREE.Color(), mesh = light.emitter(geometry, uniform(tint) as unknown as N<'vec3'>);
      mesh.position.set(x, 0, z);
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return { mesh, tint };
    };
    this.beacons = Array.from({ length: HANOI_PEGS }, (_, peg) => lamp(flat(PLATE.x, PLATE.y), pegX(peg), PLATE_Z));
    this.keyLamp = lamp(flat(KEY_W - 0.2, 0.06), 0, KEY_Z + KEY_H / 2 + 0.06);
    this.flashes = [0, 1].map(() => {
      const power = uniform(0), mesh = add(new THREE.Mesh(flat(0.9, 0.9), flashMaterial(REFUSED, power)), 7);
      return { mesh, power, lamp: lamp(new THREE.CircleGeometry(0.16, 20).rotateX(-Math.PI / 2), 0, 0) };
    });
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.flashes = [];
    this.beacons = [];
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal, layout = this.layout;
    pool.update(dt, time);
    this.log = frame.eventLog;
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const t = frame.state.step, now = frame.frameAt(t), peek = frame.frameAt(t + 1), ahead = peek.step === t + 1 ? peek : null;
    const prior = t > 0 ? frame.frameAt(t - 1) : null, before = prior !== null && prior.step === t - 1 ? prior : null;
    const calm = clamp01(2 - this.pace / (FAST * this.controlHz)), u = ahead && calm > 0 ? clamp01(frame.position - t) : 0;
    if (frame.scrubbed) this.cheerAt = this.shimmerAt = -Infinity;
    for (const event of frame.events) {
      if (event.type === 'goal') this.cheer(now, pool, time);
      if (event.type === 'cue') this.shimmer(pool, time);
    }

    const target = hanoiTarget(now.words), ease = frame.scrubbed ? 1 : 1 - Math.exp(-dt / TARGET_SECONDS), w = this.weights.value;
    w.set(w.x + ((target === 0 ? 1 : 0) - w.x) * ease, w.y + ((target === 1 ? 1 : 0) - w.y) * ease, w.z + ((target === 2 ? 1 : 0) - w.z) * ease);
    const cheer = envelope(time - this.cheerAt, 1.4, 2.2);
    this.beacons.forEach((beacon, peg) => this.shine(beacon, beacon.tint.set(GOLD).multiplyScalar(3 * w.getComponent(peg) + (peg === this.cheerPeg ? 2.5 * cheer : 0))));

    const moved = ahead ? hanoiMoved(now.words, ahead.words) : null, press = ahead ? hanoiCode(ahead.actions, layout.keys) : -1;
    const codes = now.words[HANOI_CODES], attempt = ahead && !moved && codes !== undefined ? hanoiRoute(codes, press) : null;
    const flight = moved && u > 0 ? this.flight(now, moved, u) : null;
    const lastPress = t > 0 ? hanoiCode(now.actions, layout.keys) : -1, lastMove = before ? hanoiMoved(before.words, now.words) : null;
    const lastCodes = before?.words[HANOI_CODES], lastAttempt = before && !lastMove && lastCodes !== undefined ? hanoiRoute(lastCodes, lastPress) : null;
    const hop = attempt && hanoiHeight(now.words, attempt.from) > 0 ? { peg: attempt.from, rise: 0.45 * layout.height * Math.sin(Math.PI * Math.min(1, u / 0.7)) } : null;

    this.drawDisks(now, flight, hop, time);
    this.drawKeys(now, ahead, u, press, moved, lastPress, lastMove, before !== null, time);
    this.drawPaths(before, now, lastMove, flight, u, calm);
    this.drawFlash(this.flashes[0]!, now, attempt, Math.sin(Math.PI * Math.min(1, u / 0.6)) * calm);
    this.drawFlash(this.flashes[1]!, before ?? now, lastAttempt, LINGER * (1 - u) * calm);

    const ledger = this.ledger;
    ledger.step = t;
    ledger.target = target;
    ledger.reward = now.reward;
    ledger.distance = hanoiDistance(now.words, layout.disks);
    ledger.press = lastPress < 0 ? '—' : this.pressText(lastPress, lastMove, lastAttempt, before !== null);
  }

  private shine(lamp: Lamp, tint: THREE.Color): void {
    lamp.mesh.visible = tint.r + tint.g + tint.b > 0.01;
  }

  private pressText(key: number, move: HanoiMove | null, attempt: { from: number; to: number } | null, known: boolean): string {
    if (move) return `key ${key + 1} · ${NAMES[move.from]} → ${NAMES[move.to]}`;
    if (attempt) return `key ${key + 1} · ${NAMES[attempt.from]} → ${NAMES[attempt.to]} blocked`;
    return known ? `key ${key + 1} · no disk moved` : `key ${key + 1}`;
  }

  private flight(now: EnvFrame, move: HanoiMove, u: number): Flight {
    const layout = this.layout, H = layout.height;
    const fromBottom = -(hanoiHeight(now.words, move.from) - 1) * H, toBottom = -hanoiHeight(now.words, move.to) * H;
    const lift = smoothstep01(u / 0.38), carry = smoothstep01((u - 0.3) / 0.4), fall = smoothstep01((u - 0.66) / 0.34);
    const xa = pegX(move.from), xb = pegX(move.to), rise = fromBottom - layout.fly, sink = toBottom - layout.fly, span = Math.abs(xb - xa);
    const bottom = fromBottom - rise * lift + sink * fall, gap = carry < 0.5 ? fromBottom - bottom : toBottom - bottom;
    return {
      move, x: xa + (xb - xa) * carry, top: bottom - H,
      threaded: carry <= 0 || carry >= 1 ? 1 : 0, grounded: 1 - clamp01(gap / 0.3),
      reach: (rise * lift + span * carry + sink * fall) / (rise + span + sink),
    };
  }

  private drawDisks(now: EnvFrame, flight: Flight | null, hop: { peg: number; rise: number } | null, time: number): void {
    const disks = this.disks!, shape = this.shape!, body = this.body!, layout = this.layout;
    let i = 0;
    const place = (size: number, x: number, top: number, threaded: number, grounded: number, flare: number) => {
      disks.setMatrixAt(i, this.matrix.makeTranslation(x, 0, top));
      shape.setXYZW(i, radiusOf(layout, size), threaded, grounded, flare);
      const color = this.colors[size]!;
      body.setXYZ(i, color.r, color.g, color.b);
      i++;
    };
    for (let peg = 0; peg < HANOI_PEGS; peg++) {
      const height = hanoiHeight(now.words, peg);
      for (let size = layout.disks - 1, level = 0; size >= 0; size--) {
        if (hanoiPegOf(now.words, size) !== peg) continue;
        if (flight?.move.disk !== size) {
          const lifted = hop?.peg === peg && level === height - 1 ? hop.rise : 0;
          const flare = peg === this.cheerPeg ? envelope(time - this.cheerAt - 0.07 * level, 0.9, 2.5) : 0;
          place(size, pegX(peg), levelTop(layout, level) - lifted, 1, 1 - clamp01(lifted / (0.3 * layout.height)), flare);
        }
        level++;
      }
    }
    if (flight) place(flight.move.disk, flight.x, flight.top, flight.threaded, flight.grounded, 0);
    disks.count = i;
    disks.instanceMatrix.needsUpdate = true;
    shape.needsUpdate = true;
    body.needsUpdate = true;
  }

  private drawKeys(now: EnvFrame, ahead: EnvFrame | null, u: number, press: number, moved: HanoiMove | null, lastPress: number, lastMove: HanoiMove | null, known: boolean, time: number): void {
    const look = this.look!, glow = this.glow!, layout = this.layout;
    const codes = now.words[HANOI_CODES], next = ahead?.words[HANOI_CODES], flipping = codes !== undefined && next !== undefined && next !== codes;
    const shown = flipping && u > 0.5 ? next : codes, shimmer = envelope(time - this.shimmerAt, 1.6, 1.6);
    const rising = u < 0.12 ? u / 0.12 : LINGER + (1 - LINGER) * (1 - smoothstep01((u - 0.12) / 0.88));
    let best = 0, lampKey = 0, lampWarm = 0, lampCold = 0;
    for (let key = 0; key < layout.keys; key++) {
      let warm = 0, cold = 0, route = 0, flip = 1, from = 0, to = 0;
      if (key === lastPress) {
        const fade = LINGER * (ahead ? 1 - u : 1);
        if (lastMove || !known) warm = fade; else cold = fade;
        if (lastMove && shown === undefined) ({ from, to } = lastMove), route = ahead ? 1 - u : 1;
      }
      if (key === press && ahead) {
        if (moved) warm = Math.max(warm, rising); else cold = Math.max(cold, rising);
        if (moved && shown === undefined) ({ from, to } = moved), route = Math.max(route, Math.min(1, u / 0.12));
      }
      if (shown !== undefined) ({ from, to } = hanoiRoute(shown, key)), route = 1, flip = flipping ? Math.abs(Math.cos(Math.PI * u)) : 1;
      look.setXYZW(key, key + 1, from * HANOI_PEGS + to, route, flip);
      glow.setXYZW(key, key === press && ahead ? Math.sin(Math.PI * Math.min(1, u / 0.5)) : 0, warm, cold, shimmer);
      if (warm + cold > best) [best, lampKey, lampWarm, lampCold] = [warm + cold, key, warm, cold];
    }
    look.needsUpdate = true;
    glow.needsUpdate = true;
    const lamp = this.keyLamp!;
    lamp.mesh.position.x = keyX(layout, lampKey);
    this.shine(lamp, lamp.tint.set(this.accent).multiplyScalar(3 * lampWarm).add(this.scratch.copy(REFUSED_LIGHT).multiplyScalar(3 * lampCold)).add(this.scratch.copy(CUE_LIGHT).multiplyScalar(2 * shimmer)));
  }

  private drawPaths(before: EnvFrame | null, now: EnvFrame, lastMove: HanoiMove | null, flight: Flight | null, u: number, calm: number): void {
    const H = this.layout.height, live = this.live!, last = this.last!;
    live.alpha.value = flight ? 0.5 * calm : 0;
    if (flight) {
      const move = flight.move;
      live.ends.value.set(pegX(move.from), (hanoiHeight(now.words, move.from) - 0.5) * H, pegX(move.to), (hanoiHeight(now.words, move.to) + 0.5) * H);
      live.reach.value = flight.reach;
    }
    last.alpha.value = before && lastMove ? 0.5 * calm * (1 - u) : 0;
    if (before && lastMove) {
      last.ends.value.set(pegX(lastMove.from), (hanoiHeight(before.words, lastMove.from) - 0.5) * H, pegX(lastMove.to), (hanoiHeight(now.words, lastMove.to) - 0.5) * H);
      last.reach.value = 1;
    }
  }

  private drawFlash(flash: Flash, board: EnvFrame, attempt: { from: number; to: number } | null, power: number): void {
    const shown = attempt !== null && power > 0.01;
    flash.mesh.visible = flash.lamp.mesh.visible = shown;
    flash.power.value = shown ? power : 0;
    if (!shown) return;
    const empty = hanoiHeight(board.words, attempt.from) === 0, peg = empty ? attempt.from : attempt.to;
    const z = empty ? 0 : levelTop(this.layout, hanoiHeight(board.words, peg) - 1) - 0.18;
    flash.mesh.position.set(pegX(peg), 0, z);
    flash.lamp.mesh.position.set(pegX(peg), 0, z);
    flash.lamp.tint.set(REFUSED).multiplyScalar(2.5 * power);
  }

  private cheer(now: EnvFrame, pool: CelebrationPool, time: number): void {
    const peg = hanoiPegOf(now.words, this.layout.disks - 1), top = levelTop(this.layout, this.layout.disks - 1);
    this.cheerPeg = peg;
    if (time - this.cheerAt < CHEER_SECONDS) return;
    this.cheerAt = time;
    pool.burst(this.at.set(pegX(peg), 0, top - 0.4), GOLD, 1.3, time);
    pool.burst(this.at, this.accent, 0.8, time);
    pool.pulse.fire(0.7);
  }

  private shimmer(pool: CelebrationPool, time: number): void {
    this.shimmerAt = time;
    pool.burst(this.at.set(0, 0, KEY_Z), CUE, 0.7, time);
    pool.pulse.fire(0.35);
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit): void {
    const layout = this.layout, n = layout.disks, H = layout.height, step = frame.state.step, now = frame.frameAt(step), obs = now.observations;
    const words = kit.marks('words', n), seen = kit.marks('observed', n), aim = kit.marks('aim', 1);
    for (let peg = 0; peg < HANOI_PEGS; peg++) {
      for (let size = n - 1, level = 0; size >= 0; size--) {
        if (hanoiPegOf(now.words, size) !== peg) continue;
        words.set(pegX(peg) - radiusOf(layout, size) - 0.16, 0.05, levelTop(layout, level) + H / 2, 0.09, this.colors[size]!);
        level++;
      }
    }
    for (let size = 0; size < n; size++) {
      const peg = Math.round(obs[size]! * 2);
      let level = 0;
      for (let larger = size + 1; larger < n; larger++) if (Math.round(obs[larger]! * 2) === peg) level++;
      seen.set(pegX(peg) + radiusOf(layout, size) + 0.16, 0.05, levelTop(layout, level) + H / 2, 0.07, 0xffffff);
    }
    const observed = Math.round(obs[n]! * 2), target = hanoiTarget(now.words), box = kit.curve('target', 5, GOLD);
    for (const [dx, z] of [[-0.4, 0.3], [0.4, 0.3], [0.4, layout.tip - 0.2], [-0.4, layout.tip - 0.2], [-0.4, 0.3]] as const) box.point(pegX(target) + dx, 0.05, z);
    aim.set(pegX(observed), 0.05, layout.tip - 0.45, 0.12, 0xffffff);
    const before = step > 0 ? frame.frameAt(step - 1) : null, arrow = kit.curve('move', 7, TOKENS.uiCyan), refused = kit.curve('refused', 7, REFUSED);
    if (before && before.step === step - 1) {
      const move = hanoiMoved(before.words, now.words), codes = before.words[HANOI_CODES];
      const route = move ?? (codes !== undefined ? hanoiRoute(codes, hanoiCode(now.actions, layout.keys)) : null);
      if (route) this.arrow(move ? arrow : refused, before, now, route.from, route.to);
    }
    const codes = now.words[HANOI_CODES];
    if (codes === undefined) return;
    for (let key = 0; key < layout.keys; key++) {
      const { from, to } = hanoiRoute(codes, key), curve = kit.curve(`key${key}`, 5, CUE), x = keyX(layout, key), z = KEY_Z + KEY_H / 2 + 0.16;
      const head = x + (to - 1) * 0.3, back = head - Math.sign(to - from) * 0.08;
      curve.point(x + (from - 1) * 0.3, 0.05, z);
      curve.point(head, 0.05, z);
      curve.point(back, 0.05, z - 0.06);
      curve.point(head, 0.05, z);
      curve.point(back, 0.05, z + 0.06);
    }
  }

  private arrow(curve: ReturnType<DebugKit['curve']>, before: EnvFrame, now: EnvFrame, from: number, to: number): void {
    const layout = this.layout, H = layout.height, fly = layout.fly - H / 2;
    const a = -(Math.max(1, hanoiHeight(before.words, from)) - 0.5) * H, b = -(Math.max(1, hanoiHeight(now.words, to)) - 0.5) * H;
    curve.point(pegX(from), 0.05, a);
    curve.point(pegX(from), 0.05, fly);
    curve.point(pegX(to), 0.05, fly);
    curve.point(pegX(to), 0.05, b);
    curve.point(pegX(to) - 0.15, 0.05, b - 0.2);
    curve.point(pegX(to), 0.05, b);
    curve.point(pegX(to) + 0.15, 0.05, b - 0.2);
  }
}

export default (() => new HanoiScene()) satisfies SceneFactory;
