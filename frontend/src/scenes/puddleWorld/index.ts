import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import { puddleDecoder, puddleWet } from '@sim/envs/puddle';
import type { EnvFrame, EventStream } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { flat, groundMaterial, HEDGE, LANTERN, lanternMaterial, PALETTE, PATH, PRINT, printMaterial, rainMaterial, RIPPLE, rippleMaterial, SPLAT, splatMaterial, WALKER, walkerMaterial, type Walker } from './art';

const ARROWS = ['↑ north', '↓ south', '← west', 'east →'] as const;
const DX = [0, 0, -1, 1] as const;
const DY = [1, -1, 0, 0] as const;
const ACCENT = TOKENS.accents.PuddleWorld!;

const TRAIL = 36;
const SOUNDINGS = 3000;
const WET_TEXELS = 512;
const WET_SCALE = 8;
const ARRIVE = 0.35;
const RING = 0.7;
const FLARE_LIFE = 1.6;
const NUDGE = 0.22;
const TURN = 16;
const LAMP = 24;

const signed = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace('-', '−');
const wrapAngle = (angle: number) => Math.atan2(Math.sin(angle), Math.cos(angle));
const actionOf = (frame: EnvFrame) => Math.trunc(frame.actions[0]!) & 3;
const xOf = (frame: EnvFrame) => wordFloat(frame.words[0]!);
const yOf = (frame: EnvFrame) => wordFloat(frame.words[1]!);

interface Sounding { step: number; x: number; y: number; wet: number }
interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute }

class PuddleWorldScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'puddleWorld';

  private side = 20;
  private stride = 0.05;
  private goal = 0.95;
  private penalty = 400;
  private splats: { mesh: THREE.InstancedMesh; weight: THREE.InstancedBufferAttribute } | null = null;
  private wet: THREE.RenderTarget | null = null;
  private readonly wetScene = new THREE.Scene();
  private readonly wetCamera = new THREE.OrthographicCamera();
  private prints: Batch | null = null;
  private walkerMesh: THREE.Mesh | null = null;
  private rippleMesh: THREE.Mesh | null = null;
  private readonly walker: Walker = { facing: uniform(new THREE.Vector2(0, 1)), spin: uniform(0), stride: uniform(0), alpha: uniform(1), glow: uniform(0), ring: uniform(0) };
  private readonly wade = uniform(0);
  private readonly clock = uniform(0);
  private readonly flare = uniform(0);
  private readonly lamp = uniform(LAMP);

  private soundings: Sounding[] = [];
  private soundStale = true;
  private soundAt = -Infinity;
  private step = -1;
  private angle = Math.PI / 2;
  private flared = -Infinity;
  private arrived = -Infinity;
  private wall = 0;
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, arrivals: 0 };
  private readonly ledger = { x: 0, y: 0, action: -1, blocked: false, reward: 0, known: false, step: 0, goal: undefined as number | undefined, looked: -Infinity };

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly clear = new THREE.Color();

  override legend: LegendSpec = {
    title: 'PUDDLE WORLD',
    rows: [
      { key: 'at', label: 'walker', tint: ACCENT, value: () => `x ${this.ledger.x.toFixed(3)} · y ${this.ledger.y.toFixed(3)}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action < 0 ? '—' : `${ARROWS[this.ledger.action]}${this.ledger.blocked ? ' · into the hedge' : ''}` },
      { key: 'reward', label: 'reward', tint: PALETTE.gold, value: () => signed(this.ledger.reward) },
      { key: 'puddle', label: 'puddle', tint: PALETTE.sheen, value: () => !this.ledger.known ? '—' : this.ledger.reward < -1 ? `${this.depthOf(this.ledger.reward).toFixed(4)} deep` : 'dry' },
      { key: 'porch', label: 'porch', tint: PALETTE.gold, value: () => `${this.porchDistance(this.ledger.x, this.ledger.y).toFixed(3)} away` },
      { key: 'arrival', label: 'last arrival', tint: PALETTE.flame, value: () => this.since() },
      { key: 'soundings', label: 'wet steps', tint: PALETTE.water[0], value: () => `${formatCount(this.soundings.length)} of the last ${formatCount(SOUNDINGS)}` },
    ],
    note: () => `${formatCount(this.tally.arrivals)} arrival${this.tally.arrivals === 1 ? '' : 's'}`,
    bar: () => 1 - this.porchDistance(this.ledger.x, this.ledger.y) / (this.goal * Math.SQRT2),
  };

  private since(): string {
    const found = this.ledger.goal;
    return found === undefined ? 'none yet' : found === this.ledger.step ? 'now' : `${formatCount(this.ledger.step - found)} steps ago`;
  }

  private porchDistance(x: number, y: number): number {
    return Math.hypot(Math.max(0, this.goal - x), Math.max(0, this.goal - y));
  }

  private depthOf(reward: number): number {
    return (-reward - 1) / this.penalty;
  }

  private wetness(frame: EnvFrame): number {
    return puddleWet(frame) ? 1 - Math.exp((frame.reward + 1) / WET_SCALE) : 0;
  }

  private weight(sounding: Sounding, step: number): number {
    const age = step - sounding.step;
    return age < 0 || age >= SOUNDINGS ? 0 : (0.55 + 0.45 * sounding.wet) * Math.sqrt(1 - age / SOUNDINGS);
  }

  private place(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set((x - 0.5) * this.side, 0, (0.5 - y) * this.side);
  }

  private outside(x: number, y: number, action: number): boolean {
    const tx = x + DX[action]! * this.stride, ty = y + DY[action]! * this.stride;
    return tx < 0 || tx > 1 || ty < 0 || ty > 1;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.stride = constant(env, 'PUDDLE_STEP');
    this.goal = constant(env, 'PUDDLE_GOAL');
    this.penalty = constant(env, 'PUDDLE_PENALTY');
    this.side = 1 / this.stride;
    const reach = this.side / 2 + HEDGE + PATH;
    return new THREE.Box2(new THREE.Vector2(-reach, -reach), new THREE.Vector2(reach, reach));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa3b4dc).multiplyScalar(1.05);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const half = this.side / 2, tenth = this.side / 10, at = new THREE.Vector3();
    return {
      lattice: { w: 10, h: 10, origin: [-half, 0.02, half], u: [tenth, 0, 0], v: [0, 0, -tenth] },
      debug: (frame, kit) => {
        const step = frame.state.step, cur = frame.frameAt(step), next = frame.frameAt(step + 1);
        const porch = kit.curve('porch', 5, TOKENS.celebrationGold);
        for (const [x, y] of [[this.goal, this.goal], [1, this.goal], [1, 1], [this.goal, 1], [this.goal, this.goal]] as const) {
          this.place(x, y, at);
          porch.point(at.x, 0.05, at.z);
        }
        const wet = kit.marks('soundings', SOUNDINGS);
        for (const sounding of this.soundings) {
          this.place(sounding.x, sounding.y, at);
          wet.set(at.x, 0.05, at.z, 0.2 * this.weight(sounding, step), 0x6fc3ff);
        }
        const route = kit.curve('route', TRAIL + 1, TOKENS.uiCyan), path: EnvFrame[] = [];
        for (let s = step; s >= Math.max(0, step - TRAIL); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          path.push(f);
          if (f.done) break;
        }
        for (let i = path.length - 1; i >= 0; i--) {
          this.place(xOf(path[i]!), yOf(path[i]!), at);
          route.point(at.x, 0.06, at.z);
        }
        if (cur.step !== step) return;
        this.place(xOf(cur), yOf(cur), at);
        kit.marks('agent', 1).set(at.x, 0.07, at.z, 0.14, TOKENS.uiCyan);
        if (next.step !== step + 1) return;
        const action = actionOf(next), dx = DX[action]!, dz = -DY[action]!, tip = { x: at.x + dx, z: at.z + dz };
        const intent = kit.curve('intent', 5, 0xffa05a);
        intent.point(at.x, 0.07, at.z);
        intent.point(tip.x, 0.07, tip.z);
        intent.point(tip.x - dx * 0.22 - dz * 0.14, 0.07, tip.z - dz * 0.22 + dx * 0.14);
        intent.point(tip.x, 0.07, tip.z);
        intent.point(tip.x - dx * 0.22 + dz * 0.14, 0.07, tip.z - dz * 0.22 - dx * 0.14);
        if (next.done) return;
        const move = kit.curve('move', 2, TOKENS.uiCyan);
        move.point(at.x, 0.08, at.z);
        this.place(xOf(next), yOf(next), at);
        move.point(at.x, 0.08, at.z);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const half = this.side / 2, lit = light.lit, accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(PALETTE.night);
    const wet = new THREE.RenderTarget(WET_TEXELS, WET_TEXELS, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.wet = wet;
    Object.assign(this.wetCamera, { left: -half, right: half, top: half, bottom: -half, near: 0.1, far: 100 });
    this.wetCamera.position.set(0, 50, 0);
    this.wetCamera.up.set(0, 0, -1);
    this.wetCamera.lookAt(0, 0, 0);
    this.wetCamera.updateProjectionMatrix();
    const weight = new THREE.InstancedBufferAttribute(new Float32Array(SOUNDINGS), 1).setUsage(THREE.DynamicDrawUsage);
    const splat = flat(SPLAT, SPLAT);
    splat.setAttribute('aSplat', weight);
    const splats = instancedMesh(splat, splatMaterial(), SOUNDINGS);
    splats.frustumCulled = false;
    splats.count = 0;
    this.wetScene.add(splats);
    this.splats = { mesh: splats, weight };

    const ground = new THREE.Mesh(flat(this.side * 4, this.side * 4), groundMaterial(half, (this.goal - 0.5) * this.side, lit, wet.texture, this.clock));
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground);

    this.prints = this.batch(flat(PRINT, PRINT), 'aPrint', printMaterial(lit), TRAIL, 2);

    const ripple = new THREE.Mesh(flat(RIPPLE, RIPPLE), rippleMaterial(this.wade, this.clock));
    ripple.renderOrder = 3;
    ripple.frustumCulled = false;
    const walker = new THREE.Mesh(flat(WALKER, WALKER), walkerMaterial(accent, lit, this.walker));
    walker.renderOrder = 5;
    walker.frustumCulled = false;
    const corner = half + HEDGE / 2;
    const lantern = new THREE.Mesh(flat(LANTERN, LANTERN), lanternMaterial(this.flare));
    lantern.position.set(corner, 0, -corner);
    lantern.renderOrder = 6;
    const glow = light.emitter(new THREE.CircleGeometry(0.26, 32).rotateX(-Math.PI / 2), rgb(PALETTE.flame).mul(this.lamp));
    glow.position.set(corner, 0.01, -corner);
    glow.renderOrder = 2;
    const rain = new THREE.Mesh(flat(this.side * 4, this.side * 4), rainMaterial(half, this.clock));
    rain.renderOrder = 9;
    rain.frustumCulled = false;
    this.rippleMesh = ripple;
    this.walkerMesh = walker;
    this.scene.add(ripple, walker, lantern, glow, rain);
    pool.group.traverse((object) => { object.renderOrder = 20; });
  }

  private batch(geometry: THREE.BufferGeometry, name: string, material: THREE.Material, count: number, order: number): Batch {
    const state = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, state);
    const mesh = instancedMesh(geometry, material, count);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    return { mesh, state };
  }

  protected override drop(): void {
    for (const scene of [this.scene, this.wetScene]) scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.wetScene.clear();
    this.wet?.dispose();
    this.wet = null;
    this.splats = null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.wall += frame.dtReal;
    pool.update(frame.dtReal, time);
    const step = frame.state.step;
    if (frame.scrubbed || step < this.step) this.settle(frame, step, pool);
    else for (let s = this.step + 1; s <= step; s++) if (!this.advance(frame, s, pool, time)) break;
    if (this.soundStale && this.wall - this.soundAt > 0.25) this.sound(frame, step);
    if (step !== this.step) {
      this.footprints(frame, step);
      this.paintSoundings(step);
    }
    this.step = step;
    this.ledger.step = step;
    if (this.wall - this.ledger.looked > 1) {
      this.ledger.goal = lastEvent(frame.eventLog, 'goal', step);
      this.ledger.looked = this.wall;
    }
    this.count(frame.eventLog, step);
    this.pose(frame, step, time);
    const flare = Math.max(0, 1 - (time - this.flared) / FLARE_LIFE);
    this.flare.value = flare;
    this.lamp.value = LAMP * (1 + 0.6 * flare);
    this.clock.value = time;
  }

  private settle(frame: FrameUpdate, step: number, pool: CelebrationPool): void {
    pool.clear();
    this.flared = -Infinity;
    this.arrived = -Infinity;
    this.ledger.looked = -Infinity;
    this.soundStale = true;
    this.record(frame.frameAt(step), frame.frameAt(step - 1), step);
    const now = frame.frameAt(step + 1).step === step + 1 ? frame.frameAt(step + 1) : frame.frameAt(step);
    if (now.step > 0) this.angle = Math.atan2(DY[actionOf(now)]!, DX[actionOf(now)]!);
    this.step = -1;
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): boolean {
    const now = frame.frameAt(s), before = frame.frameAt(s - 1);
    if (now.step !== s || before.step !== s - 1) {
      this.settle(frame, frame.state.step, pool);
      return false;
    }
    this.record(now, before, s);
    const wet = this.wetness(now);
    if (wet > 0) this.soundings.push({ step: s, x: xOf(now), y: yOf(now), wet });
    while (this.soundings.length > 0 && this.soundings[0]!.step <= s - SOUNDINGS) this.soundings.shift();
    for (const event of puddleDecoder.deriveEvents(before, now)) {
      if (event.type === 'harm') {
        this.place(xOf(now), yOf(now), this.at);
        pool.burst(this.at.setY(38), PALETTE.sheen, 0.35, time);
      }
      if (event.type === 'goal') {
        const middle = (this.goal + 1) / 2;
        this.place(middle, middle, this.at);
        pool.burst(this.at.setY(30), TOKENS.celebrationGold, 0.8, time);
        this.flared = time;
        this.ledger.goal = s;
      }
      if (event.type === 'teleport') this.arrived = this.wall;
    }
    return true;
  }

  private record(now: EnvFrame, before: EnvFrame, s: number): void {
    const ledger = this.ledger;
    ledger.x = xOf(now);
    ledger.y = yOf(now);
    ledger.reward = now.reward;
    ledger.known = s > 0 && !now.done;
    ledger.action = s > 0 ? actionOf(now) : -1;
    ledger.blocked = s > 0 && before.step === s - 1 && !now.done && this.outside(xOf(before), yOf(before), ledger.action);
  }

  private sound(frame: FrameUpdate, step: number): void {
    this.soundings.length = 0;
    this.soundStale = false;
    this.soundAt = this.wall;
    for (let s = Math.max(1, step - SOUNDINGS + 1); s <= step; s++) {
      const f = frame.frameAt(s);
      if (f.step !== s) {
        this.soundStale = true;
        continue;
      }
      const wet = this.wetness(f);
      if (wet > 0) this.soundings.push({ step: s, x: xOf(f), y: yOf(f), wet });
    }
    this.paintSoundings(step);
  }

  private paintSoundings(step: number): void {
    const splats = this.splats, wet = this.wet, renderer = this.ctx?.renderer;
    if (!splats || !wet || !renderer) return;
    let n = 0;
    for (const sounding of this.soundings) {
      const age = step - sounding.step;
      if (age < 0 || age >= SOUNDINGS) continue;
      splats.mesh.setMatrixAt(n, this.matrix.makeTranslation(this.place(sounding.x, sounding.y, this.at)));
      splats.weight.setX(n, this.weight(sounding, step));
      n++;
    }
    splats.mesh.count = n;
    splats.mesh.instanceMatrix.needsUpdate = true;
    splats.weight.needsUpdate = true;
    const previous = renderer.getRenderTarget(), alpha = renderer.getClearAlpha();
    renderer.getClearColor(this.clear);
    renderer.setRenderTarget(wet);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(this.wetScene, this.wetCamera);
    renderer.setClearColor(this.clear, alpha);
    renderer.setRenderTarget(previous);
  }

  private footprints(frame: FrameUpdate, step: number): void {
    const prints = this.prints;
    if (!prints) return;
    let n = 0;
    for (let i = 0; i < TRAIL; i++) {
      const s = step - i;
      if (s < 1) break;
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1 || now.done) continue;
      const action = actionOf(now);
      this.place(xOf(now), yOf(now), this.at);
      this.quaternion.setFromAxisAngle(this.axis, Math.atan2(-DX[action]!, DY[action]!));
      prints.mesh.setMatrixAt(n, this.matrix.compose(this.at, this.quaternion, this.scale.setScalar(1)));
      prints.state.setXYZW(n, 0.75 * (1 - i / TRAIL) ** 1.2, s % 2 ? 1 : -1, this.wetness(now), 0);
      n++;
    }
    prints.mesh.count = n;
    prints.mesh.instanceMatrix.needsUpdate = true;
    prints.state.needsUpdate = true;
  }

  private pose(frame: FrameUpdate, step: number, time: number): void {
    const walker = this.walkerMesh, ripple = this.rippleMesh;
    if (!walker || !ripple) return;
    const cur = frame.frameAt(step), next = frame.frameAt(step + 1), live = cur.step === step && next.step === step + 1;
    const t = live ? clamp01(frame.position - step) : 0, x0 = xOf(cur), y0 = yOf(cur);
    let x = x0, y = y0, alpha = 1, glow = 0, wade = cur.done ? 0 : this.wetness(cur), nudge = 0;
    let action = step > 0 && !cur.done ? actionOf(cur) : -1;
    if (live) {
      action = actionOf(next);
      if (next.done) {
        const k = smoothstep01(t);
        x = clamp01(x0 + DX[action]! * this.stride * k);
        y = clamp01(y0 + DY[action]! * this.stride * k);
        alpha = 1 - smoothstep01((t - 0.45) / 0.55);
        glow = 0.85 * smoothstep01((t - 0.25) / 0.75);
      } else {
        x = x0 + (xOf(next) - x0) * t;
        y = y0 + (yOf(next) - y0) * t;
        wade += (this.wetness(next) - wade) * t;
        if (this.outside(x0, y0, action)) nudge = NUDGE * Math.sin(Math.PI * t);
      }
    }
    const k = clamp01((this.wall - this.arrived) / ARRIVE);
    alpha *= smoothstep01(k);
    glow = Math.max(glow, 0.6 * (1 - k));
    this.place(x, y, this.at);
    if (action >= 0) {
      this.at.x += DX[action]! * nudge;
      this.at.z -= DY[action]! * nudge;
      const target = Math.atan2(DY[action]!, DX[action]!);
      this.angle = frame.scrubbed ? target : this.angle + wrapAngle(target - this.angle) * (1 - Math.exp(-TURN * frame.dtReal));
    }
    walker.position.copy(this.at);
    ripple.position.copy(this.at);
    this.walker.facing.value.set(Math.cos(this.angle), Math.sin(this.angle));
    this.walker.spin.value = this.angle + 0.08 * Math.sin(time * 1.7);
    this.walker.stride.value = (step % 2) + t;
    this.walker.alpha.value = alpha;
    this.walker.glow.value = glow;
    this.walker.ring.value = 1 - clamp01((this.wall - this.arrived) / RING);
    this.wade.value = wade * alpha;
  }

  private count(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, arrivals: 0 });
    if (at === tally.through) return;
    tally.arrivals += countEvents(log, 'goal', tally.through, at);
    tally.through = at;
  }
}

export default (() => new PuddleWorldScene()) satisfies SceneFactory;
