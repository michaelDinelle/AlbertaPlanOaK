import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { ringCall, ringCell, ringColour, ringDecoder } from '@sim/envs/ring';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount } from '../../format';
import { chameleon, chameleonMaterial, SPRITE, type Chameleon } from './chameleon';
import { BASE, discGeometry, discMaterial, flat, florets, groundMaterial, HUES, NAMES, onRing, petalEmitters, petalLight, petalMaterial, REACH, reachOf, TAU, TIP, WALK } from './flower';

const SKINS = [0xe79b22, 0x7b5ae6] as const;
const SIZE = 1.6;
const LAPS = 16;
const FLORETS = 2200;
const PETAL_LIGHT = 1.6;
const CHANGE_AT = 0.18;
const CHANGE_SPAN = 0.34;
const WALK_AT = 0.42;
const STRIDE = 0.45;
const RETRY = 0.25;
const QUICK = 14;

class RingScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'ring';
  override legend: LegendSpec | null = null;

  private cells = 1;
  private arc = 1;
  private window = 1;
  private data = new Float32Array(4);
  private verdicts = new THREE.DataTexture(this.data, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private flips = new Uint8Array(1);
  private hits = new Uint8Array(1);
  private tags = new Float64Array(1);
  private readonly phase = uniform(0);
  private readonly lizard: Chameleon = chameleon();
  private readonly lights = uniform(1);
  private readonly glow = new THREE.Color();
  private figure: THREE.Mesh | null = null;
  private readonly neutral = new THREE.Color();
  private filled = -1;
  private stale = false;
  private tried = -Infinity;
  private shown = -1;
  private wall = 0;
  private rate = 0;
  private lastPosition = NaN;
  private readonly point = new THREE.Vector3();
  private readonly tint = new THREE.Color();
  private readonly ledger = { cell: 0, sees: 0, called: -1, reward: 0, lit: 0, known: 0, flips: 0, caught: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.cells = constant(ctx.environment, 'RING_CELLS');
    this.arc = parameter(ctx.environment, 'arcLength');
    this.window = LAPS * this.cells;
    this.data = new Float32Array(this.cells * 4);
    for (let c = 0; c < this.cells; c++) this.data[c * 4 + 2] = reachOf(c);
    this.verdicts = new THREE.DataTexture(this.data, this.cells, 1, THREE.RGBAFormat, THREE.FloatType);
    this.flips = new Uint8Array(this.window);
    this.hits = new Uint8Array(this.window);
    this.tags = new Float64Array(this.window).fill(NaN);
    return new THREE.Box2(new THREE.Vector2(-REACH, -REACH), new THREE.Vector2(REACH, REACH));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8796c8).multiplyScalar(0.42);
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    this.lights.value = on ? 0 : 1;
    this.lizard.lights.value = on ? 0 : 1;
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 30, sighSize: 18, sighRadius: 0.24, sighHeight: 0.9 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit, cells = this.cells, arc = this.arc, accent = this.accentOf(ctx);
    this.neutral.set(accent);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(0x05070a);
    pool.group.renderOrder = 10;
    this.verdicts.magFilter = this.verdicts.minFilter = THREE.NearestFilter;
    this.verdicts.needsUpdate = true;
    const place = (mesh: THREE.Mesh, order: number, y = 0): THREE.Mesh => {
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      mesh.position.y = y;
      this.scene.add(mesh);
      return mesh;
    };
    const ambient = this.ambient(), state = { verdicts: this.verdicts, phase: this.phase, lights: this.lights, ambient };
    place(new THREE.Mesh(flat(90, 90), groundMaterial(lit)), 0);
    place(new THREE.Mesh(new THREE.RingGeometry(BASE - 0.05, TIP + 0.05, 512, 1).rotateX(-Math.PI / 2), petalMaterial(cells, arc, state)), 1);
    place(light.emitter(petalEmitters(cells), petalLight(cells, arc, state, PETAL_LIGHT)), 1, 0.02);
    place(new THREE.Mesh(discGeometry(), discMaterial(lit)), 2);
    place(florets(lit, FLORETS), 3);
    this.figure = place(new THREE.Mesh(flat(SPRITE.x, SPRITE.y), chameleonMaterial(accent, ambient, this.lizard)), 6);
  }

  protected override drop(): void {
    this.verdicts.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private at(frame: FrameUpdate, s: number): EnvFrame | null {
    if (s < 0) return null;
    const found = frame.frameAt(s);
    return found.step === s ? found : null;
  }

  private check(f: EnvFrame, before: EnvFrame | null): void {
    const cell = ringCell(f), colour = ringColour(cell, this.arc);
    if (cell >= this.cells || f.observations[0] !== colour || f.observations[1] !== 1 || f.done !== 0
      || (before !== null && (cell !== (ringCell(before) + 1) % this.cells || f.reward !== (ringCall(f) === colour ? 1 : 0)))) {
      throw new Error(`Ring step ${f.step} stands on cell ${cell} seeing ${f.observations[0]} and was paid ${f.reward} for calling ${ringCall(f)} after cell ${before ? ringCell(before) : 'none'}`);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.wall += frame.dtReal;
    if (frame.dtReal > 0 && Number.isFinite(this.lastPosition)) {
      const speed = Math.abs(frame.position - this.lastPosition) / frame.dtReal;
      this.rate += (speed - this.rate) * Math.min(1, frame.dtReal * 4);
    }
    this.lastPosition = frame.position;
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0;
    if (k !== this.filled || (this.stale && this.wall - this.tried > RETRY)) this.fill(frame, k);
    this.phase.value = f;
    this.celebrate(frame, k, pool, time);
    this.shown = k;
    this.pose(frame, k, now, next, f);
  }

  private fill(frame: FrameUpdate, k: number): void {
    const cells = this.cells, data = this.data;
    let stale = false, lit = 0, known = 0;
    for (let c = 0; c < cells; c++) data[c * 4] = data[c * 4 + 1] = 0;
    for (let i = 0; i < cells && k - i >= 1; i++) {
      const s = k - i, f = this.at(frame, s);
      if (!f) { stale = true; continue; }
      this.check(f, this.at(frame, s - 1));
      const cell = ringCell(f);
      data[cell * 4] = f.reward > 0 ? 1 : 2;
      data[cell * 4 + 1] = i;
      known++;
      if (f.reward > 0) lit++;
    }
    this.verdicts.needsUpdate = true;
    let flips = 0, caught = 0;
    for (let s = Math.max(1, k - this.window + 1); s <= k; s++) {
      const slot = s % this.window;
      if (this.tags[slot] !== s) {
        const f = this.at(frame, s), before = this.at(frame, s - 1);
        if (!f || !before) { stale = true; continue; }
        this.check(f, before);
        this.flips[slot] = f.observations[0] !== before.observations[0] ? 1 : 0;
        this.hits[slot] = f.reward > 0 ? 1 : 0;
        this.tags[slot] = s;
      }
      if (this.flips[slot]) {
        flips++;
        caught += this.hits[slot]!;
      }
    }
    const now = this.at(frame, k)!, cell = ringCell(now);
    Object.assign(this.ledger, { cell, sees: ringColour(cell, this.arc), called: k > 0 ? ringCall(now) : -1, reward: now.reward, lit, known, flips, caught });
    this.filled = k;
    this.stale = stale;
    this.tried = this.wall;
  }

  private celebrate(frame: FrameUpdate, k: number, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed || this.shown < 0 || k < this.shown) {
      pool.clear();
      return;
    }
    for (let s = Math.max(this.shown + 1, k - 11); s <= k; s++) {
      const f = this.at(frame, s), before = this.at(frame, s - 1);
      if (!f || !before) continue;
      const cell = ringCell(f);
      for (const event of ringDecoder.deriveEvents(before, f)) {
        onRing(cell, this.cells, WALK, this.point);
        if (event.type === 'goal') {
          pool.burst(this.point.setY(38), HUES[ringColour(cell, this.arc)]!, 0.8, time);
          pool.burst(this.point.setY(39), TOKENS.celebrationGold, 0.35, time);
        } else if (event.type === 'harm') pool.sigh(0.5, this.point.setY(44), time);
      }
    }
  }

  private skin(call: number, out: THREE.Vector3): void {
    this.tint.set(call < 0 ? this.neutral : SKINS[call]!);
    out.set(this.tint.r, this.tint.g, this.tint.b);
  }

  private pose(frame: FrameUpdate, k: number, now: EnvFrame, next: EnvFrame | null, f: number): void {
    const figure = this.figure, lizard = this.lizard;
    if (!figure) return;
    const cells = this.cells, cell = ringCell(now), call = k > 0 ? ringCall(now) : -1, coming = next ? ringCall(next) : call;
    const quick = this.rate > QUICK, walk = next ? smoothstep01((f - WALK_AT) / (1 - WALK_AT)) : 0;
    const change = coming === call ? 1 : quick ? 1 : smoothstep01((f - CHANGE_AT) / CHANGE_SPAN);
    const angle = (TAU * (cell + 0.5 + walk)) / cells, amp = quick ? 0 : 1;
    figure.position.set(WALK * Math.sin(angle), 0, -WALK * Math.cos(angle));
    figure.rotation.y = -angle - 0.05 * amp * Math.sin(TAU * walk);
    this.skin(call, lizard.from.value);
    this.skin(coming, lizard.to.value);
    lizard.front.value = -0.15 + 1.5 * change;
    const pitch = (TAU * WALK) / cells, stride = (STRIDE * pitch * amp) / SIZE;
    lizard.swingA.value = stride * (smoothstep01(walk / 0.5) - walk);
    lizard.swingB.value = stride * (smoothstep01((walk - 0.5) / 0.5) - walk);
    const hit = k > 0 && now.reward > 0, before = this.at(frame, k - 1), was = before !== null && k > 1 && before.reward > 0;
    const settle = quick ? 1 : smoothstep01(f / 0.22);
    lizard.blend.value = (hit ? (was ? 1 : settle) : 0) * (1 - (coming === call ? 0 : change));
    lizard.startle.value = k > 0 && !hit && !quick ? Math.max(0, 1 - f / 0.6) : 0;
    lizard.breath.value = 1 + 0.025 * Math.sin(this.wall * 4.7);
    const look = coming !== call ? change : 0;
    lizard.outer.value.set(0.85, 0.35 - 0.35 * look).normalize();
    const wander = Math.sin(this.wall * 0.9 + k * 0.37);
    lizard.inner.value.set(0.3 * wander + 0.7 * look, -0.9 + 0.9 * look).normalize();
    this.shine(cell + walk, f, lizard.light.value);
    const caught = hit && before !== null && now.observations[0] !== before.observations[0];
    const hop = caught && !quick ? Math.sin(Math.PI * clamp01(f / 0.3)) : 0;
    figure.scale.setScalar(SIZE * (1 + 0.08 * hop + 0.05 * lizard.startle.value));
  }

  private shine(at: number, f: number, out: THREE.Vector3): void {
    const cells = this.cells, data = this.data, glow = this.glow;
    let weight = 0;
    out.set(0, 0, 0);
    for (let c = Math.floor(at) - 2; c <= Math.floor(at) + 3; c++) {
      const w = Math.max(0, 1 - Math.abs(c - at) / 2.5), cell = ((c % cells) + cells) % cells, age = data[cell * 4 + 1]! + f;
      weight += w;
      if (data[cell * 4] !== 1 || w === 0) continue;
      const level = 1 - (0.5 * age) / cells + 0.5 * Math.exp(-age / 0.6);
      glow.set(HUES[ringColour(cell, this.arc)]!);
      out.x += glow.r * level * w;
      out.y += glow.g * level * w;
      out.z += glow.b * level * w;
    }
    out.multiplyScalar((PETAL_LIGHT * 0.55) / weight);
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, named = (hue: number) => `${NAMES[hue]} · ${hue}`;
    return {
      title: 'RING',
      rows: [
        { key: 'petal', label: 'chameleon', tint: accent, value: () => `petal ${l.cell + 1} of ${this.cells}` },
        { key: 'sees', label: 'sees', tint: 0xe8e2d4, value: () => named(l.sees) },
        { key: 'called', label: 'called', tint: accent, value: () => (l.called < 0 ? '—' : named(l.called)) },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.called < 0 ? '—' : l.reward > 0 ? '+1' : '0') },
        { key: 'lit', label: 'petals lit, last lap', tint: HUES[0], value: () => `${l.lit} of ${l.known}` },
        { key: 'caught', label: `flips caught, ${LAPS} laps`, tint: TOKENS.celebrationGold, value: () => (l.flips ? `${formatCount(l.caught)} of ${formatCount(l.flips)}` : '—') },
      ],
      note: () => `${Math.ceil(this.cells / this.arc)} arcs of ${this.arc} · flip in ${this.arc - (l.cell % this.arc)}`,
      bar: () => (l.flips ? l.caught / l.flips : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const cells = this.cells, arc = this.arc, at = this.point;
    const ring = kit.curve('ring', cells + 1, 0x8fa6c4);
    for (let c = 0; c <= cells; c++) {
      const a = (TAU * c) / cells;
      ring.point(WALK * Math.sin(a), 0.05, -WALK * Math.cos(a));
    }
    const colours = kit.marks('colours', cells), flips = kit.marks('flips', cells), verdicts = kit.marks('verdicts', cells);
    for (let c = 0; c < cells; c++) {
      const hue = ringColour(c, arc), code = this.data[c * 4]!;
      onRing(c, cells, TIP - 0.35, at);
      colours.set(at.x, 0.06, at.z, 0.08, HUES[hue]!);
      if (hue !== ringColour((c + cells - 1) % cells, arc)) {
        const a = (TAU * c) / cells;
        flips.set((TIP - 0.1) * Math.sin(a), 0.07, -(TIP - 0.1) * Math.cos(a), 0.1, 0xffffff);
      }
      onRing(c, cells, BASE + 0.55, at);
      verdicts.set(at.x, 0.06, at.z, code === 0 ? 0.04 : 0.09, code === 1 ? TOKENS.celebrationGold : code === 2 ? TOKENS.failureBlue : 0x6b6f7a);
    }
    const cell = ringCell(now);
    onRing(cell, cells, WALK, at);
    kit.marks('agent', 1).set(at.x, 0.08, at.z, 0.16, TOKENS.uiCyan);
    const next = this.at(frame, k + 1), call = kit.marks('call', 1);
    if (!next) return;
    onRing(ringCell(next), cells, WALK, at);
    call.set(at.x, 0.1, at.z, 0.11, HUES[ringCall(next)]!);
  }
}

export default (() => new RingScene()) satisfies SceneFactory;
