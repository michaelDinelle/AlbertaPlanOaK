// kit/celebrate — THE shared reward grammar.
//
// STYLE.md: "Every +1, in every scene, is the same three beats, skinned per scene:
// 1. burst(pos, accent) — radial particle bloom  2. ripple(groundPos) — expanding ground
// ring  3. 300 ms bloom-strength pulse (host post hook — automatic). Failure /
// non-reward `done` = the shared cool-blue sigh."
//
// Everything is POOLED: a scene fires celebrations from inside update(), which may not
// allocate. Firing writes into preallocated attribute buffers and bumps a ring cursor —
// no geometry, no material, no Vector3 is created after construction. All particle
// motion happens in the vertex shader from `uTime`, so a paused playback clock freezes
// the celebration mid-air exactly like everything else.
//
// The 300 ms bloom pulse is not applied here (scenes must not touch the composer): it is
// published on a PulseChannel that the HOST reads once per frame and feeds to
// `ComposerHandle.setBloomPulse`.

import * as THREE from 'three';
import { mulberry32 } from './rng';
import { radialSprite } from './textures';
import { TOKENS } from './tokens';
import type { QualityTier } from './types';

// ---------------------------------------------------------------------------
// pulse channel — the host's post hook
// ---------------------------------------------------------------------------

/** STYLE.md beat 3: a 300 ms bloom-strength pulse. */
export const PULSE_SECONDS = 0.3;
/** the cool-blue failure sigh lasts a touch longer than a celebration */
export const SIGH_SECONDS = 0.9;

export interface PulseChannel {
  /** 0..1 celebration pulse — host: `post.setBloomPulse(pulse.bloom)` */
  bloom: number;
  /** 0..1 failure sigh — host may desaturate/cool the grade pass by this much */
  sigh: number;
  fire(strength?: number): void;
  fireSigh(strength?: number): void;
  /** decay in WALL seconds (the pulse is a UI beat, not a sim quantity) */
  update(dt: number): void;
  reset(): void;
}

export function makePulseChannel(): PulseChannel {
  return {
    bloom: 0,
    sigh: 0,
    fire(strength = 1) {
      this.bloom = Math.min(1, Math.max(this.bloom, strength));
    },
    fireSigh(strength = 1) {
      this.sigh = Math.min(1, Math.max(this.sigh, strength));
    },
    update(dt) {
      if (this.bloom > 0) this.bloom = Math.max(0, this.bloom - dt / PULSE_SECONDS);
      if (this.sigh > 0) this.sigh = Math.max(0, this.sigh - dt / SIGH_SECONDS);
    },
    reset() {
      this.bloom = 0;
      this.sigh = 0;
    },
  };
}

/**
 * A convenience channel for a SINGLE-scene host that wants one global read point:
 * pass it explicitly (`new CelebrationPool({ pulse: pulseChannel })`).
 *
 * It is NOT the default any more. Pools used to share it implicitly, so the moment
 * two scenes were alive at once (the theater wall, a thumb grid, a hero + preview
 * pair) one lane's +1 bloomed every other lane's frame. A pool owns its channel
 * unless a host deliberately hands it one; hosts read `celebrationPulse(scene)`.
 */
export const pulseChannel: PulseChannel = makePulseChannel();

// ---------------------------------------------------------------------------
// pool
// ---------------------------------------------------------------------------

/** particle capacity / ripple capacity per tier (thumb celebrates silently). */
export function celebrationCapacity(tier: QualityTier | undefined): { particles: number; ripples: number } {
  switch (tier) {
    case 'thumb':
      return { particles: 0, ripples: 0 };
    case 'medium':
      return { particles: 384, ripples: 4 };
    default:
      return { particles: 768, ripples: 8 };
  }
}

export interface CelebrationPoolOptions {
  tier?: QualityTier;
  /** override the tier capacity */
  particleCapacity?: number;
  rippleCapacity?: number;
  /** particles consumed by one full-scale burst (default 48) */
  burstSize?: number;
  /** particles consumed by one sigh (default 40) */
  sighSize?: number;
  /** radius of the sigh mote curtain (default 6) */
  sighRadius?: number;
  /** height the sigh motes fall from (default 3.2) */
  sighHeight?: number;
  /** deterministic direction seed (defaults to the lane seed) */
  seed?: number;
  /** share a channel with other pools / the host; omitted = this pool mints its own */
  pulse?: PulseChannel;
  name?: string;
}

const BURST_VERT = /* glsl */ `
#include <common>
uniform float uTime;
uniform float uPixelRatio;
attribute vec3 aDir;
attribute vec3 aColor;
attribute float aStart;
attribute float aLife;
attribute float aSize;
attribute float aScale;
attribute float aGravity;
varying vec3 vColor;
varying float vAlpha;
void main() {
	float age = uTime - aStart;
	float live = step( 0.0, aStart ) * step( 0.0, age ) * step( age, aLife );
	float t = clamp( age / max( aLife, 0.0001 ), 0.0, 1.0 );
	// ease-out travel: fast bloom, slow settle
	float travel = 1.0 - pow( 1.0 - t, 2.4 );
	vec3 p = position + aDir * ( aScale * travel );
	p.y -= aGravity * aScale * t * t;
	vec4 mv = modelViewMatrix * vec4( p, 1.0 );
	vColor = aColor;
	vAlpha = live * ( 1.0 - t ) * ( 1.0 - t );
	gl_PointSize = live * aSize * aScale * uPixelRatio * ( 220.0 / max( -mv.z, 0.001 ) );
	gl_Position = projectionMatrix * mv;
}
`;

const BURST_FRAG = /* glsl */ `
#include <common>
uniform sampler2D uSprite;
uniform float uIntensity;
varying vec3 vColor;
varying float vAlpha;
void main() {
	float a = texture2D( uSprite, gl_PointCoord ).a * vAlpha;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( vColor * uIntensity, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

const RIPPLE_VERT = /* glsl */ `
#include <common>
varying vec2 vUvR;
void main() {
	vUvR = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const RIPPLE_FRAG = /* glsl */ `
#include <common>
uniform vec3 uColor;
uniform float uProgress;
uniform float uIntensity;
varying vec2 vUvR;
void main() {
	// a soft annulus that thins as it expands
	float r = length( vUvR - vec2( 0.5 ) ) * 2.0;
	float width = mix( 0.42, 0.08, uProgress );
	float band = 1.0 - smoothstep( 0.0, width, abs( r - 1.0 ) );
	float fade = ( 1.0 - uProgress ) * ( 1.0 - uProgress );
	float a = band * fade;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( uColor * uIntensity, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

interface RippleSlot {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  start: number;
  life: number;
  scale: number;
}

/**
 * The per-scene celebration pool. Scenes own one, add `pool.group` to their scene, and
 * call `pool.update(dtReal, ctx.time())` every frame.
 */
export class CelebrationPool {
  readonly group = new THREE.Group();
  readonly pulse: PulseChannel;

  private readonly capacity: number;
  private readonly burstSize: number;
  private readonly sighSize: number;
  private readonly sighRadius: number;
  private readonly sighHeight: number;

  private readonly points: THREE.Points | null;
  private readonly pointsMaterial: THREE.ShaderMaterial | null;
  private readonly pointsGeometry: THREE.BufferGeometry | null;
  private readonly sprite: THREE.DataTexture | null;

  private readonly posArray: Float32Array;
  private readonly dirArray: Float32Array;
  private readonly colorArray: Float32Array;
  private readonly startArray: Float32Array;
  private readonly lifeArray: Float32Array;
  private readonly sizeArray: Float32Array;
  private readonly scaleArray: Float32Array;
  private readonly gravityArray: Float32Array;
  private readonly attrs: THREE.BufferAttribute[] = [];

  private readonly ripples: RippleSlot[] = [];
  private readonly rippleGeometry: THREE.BufferGeometry | null;

  private cursor = 0;
  private rippleCursor = 0;
  private time = 0;
  private burstsFired = 0;
  private ripplesFired = 0;
  private readonly color = new THREE.Color();

  constructor(opts: CelebrationPoolOptions = {}) {
    const cap = celebrationCapacity(opts.tier);
    this.capacity = Math.max(0, opts.particleCapacity ?? cap.particles);
    const rippleCapacity = Math.max(0, opts.rippleCapacity ?? cap.ripples);
    this.burstSize = Math.max(1, opts.burstSize ?? 48);
    this.sighSize = Math.max(1, opts.sighSize ?? 40);
    this.sighRadius = opts.sighRadius ?? 6;
    this.sighHeight = opts.sighHeight ?? 3.2;
    // own channel by default — sharing the module singleton was cross-scene crosstalk
    this.pulse = opts.pulse ?? makePulseChannel();
    this.group.name = opts.name ?? 'oakCelebrations';

    // ---- burst points ----
    const n = this.capacity;
    this.posArray = new Float32Array(Math.max(1, n) * 3);
    this.dirArray = new Float32Array(Math.max(1, n) * 3);
    this.colorArray = new Float32Array(Math.max(1, n) * 3);
    this.startArray = new Float32Array(Math.max(1, n)).fill(-1);
    this.lifeArray = new Float32Array(Math.max(1, n)).fill(1);
    this.sizeArray = new Float32Array(Math.max(1, n)).fill(1);
    this.scaleArray = new Float32Array(Math.max(1, n)).fill(1);
    this.gravityArray = new Float32Array(Math.max(1, n));

    // deterministic per-slot directions: a slot always throws the same way, so a
    // scrubbed replay produces the same bloom.
    const rng = mulberry32((opts.seed ?? 0) ^ 0x51a3f7);
    for (let i = 0; i < n; i++) {
      const u = rng() * 2 - 1;
      const a = rng() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - u * u));
      // bias upward: a celebration blooms, it does not sink
      const y = Math.abs(u) * 0.85 + 0.15;
      this.dirArray[i * 3] = Math.cos(a) * r;
      this.dirArray[i * 3 + 1] = y;
      this.dirArray[i * 3 + 2] = Math.sin(a) * r;
      this.sizeArray[i] = 0.5 + rng() * 0.9;
    }

    if (n > 0) {
      this.sprite = radialSprite(64, 2.2);
      this.pointsGeometry = new THREE.BufferGeometry();
      const add = (name: string, array: Float32Array, size: number): void => {
        const attr = new THREE.BufferAttribute(array, size);
        attr.setUsage(THREE.DynamicDrawUsage);
        this.pointsGeometry?.setAttribute(name, attr);
        this.attrs.push(attr);
      };
      add('position', this.posArray, 3);
      add('aDir', this.dirArray, 3);
      add('aColor', this.colorArray, 3);
      add('aStart', this.startArray, 1);
      add('aLife', this.lifeArray, 1);
      add('aSize', this.sizeArray, 1);
      add('aScale', this.scaleArray, 1);
      add('aGravity', this.gravityArray, 1);
      this.pointsGeometry.setDrawRange(0, n);
      this.pointsGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);

      this.pointsMaterial = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uPixelRatio: { value: 1 },
          uSprite: { value: this.sprite },
          uIntensity: { value: 2.4 },
        },
        vertexShader: BURST_VERT,
        fragmentShader: BURST_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        fog: false,
      });
      this.pointsMaterial.name = 'oakCelebrationBurst';
      this.points = new THREE.Points(this.pointsGeometry, this.pointsMaterial);
      this.points.name = 'oakCelebrationBurst';
      this.points.frustumCulled = false;
      this.group.add(this.points);
    } else {
      this.points = null;
      this.pointsMaterial = null;
      this.pointsGeometry = null;
      this.sprite = null;
    }

    // ---- ground ripples ----
    if (rippleCapacity > 0) {
      this.rippleGeometry = new THREE.PlaneGeometry(1, 1);
      this.rippleGeometry.rotateX(-Math.PI / 2);
      for (let i = 0; i < rippleCapacity; i++) {
        const material = new THREE.ShaderMaterial({
          uniforms: {
            uColor: { value: new THREE.Color(TOKENS.celebrationGold) },
            uProgress: { value: 1 },
            uIntensity: { value: 1.8 },
          },
          vertexShader: RIPPLE_VERT,
          fragmentShader: RIPPLE_FRAG,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
          fog: false,
        });
        material.name = 'oakCelebrationRipple';
        const mesh = new THREE.Mesh(this.rippleGeometry, material);
        mesh.name = `oakCelebrationRipple${i}`;
        mesh.visible = false;
        mesh.frustumCulled = false;
        mesh.renderOrder = 2;
        this.group.add(mesh);
        this.ripples.push({ mesh, material, start: -1, life: 1, scale: 1 });
      }
    } else {
      this.rippleGeometry = null;
    }
  }

  /** particles + ripples fired so far — the pool-reuse assertion in tests. */
  get stats(): { bursts: number; ripples: number; particles: number; cursor: number } {
    return { bursts: this.burstsFired, ripples: this.ripplesFired, particles: this.capacity, cursor: this.cursor };
  }

  /** live ripple slots (visible meshes) — never grows past the pool capacity. */
  get activeRipples(): number {
    let n = 0;
    for (const slot of this.ripples) if (slot.mesh.visible) n++;
    return n;
  }

  /**
   * Adopt the host playback clock without advancing anything else.
   *
   * Particle age is `uTime - aStart`, both on the host clock, so a pool whose
   * `time` is still 0 stamps every burst at 0 — and the first `update(dt, 1234)`
   * renders them as 1234 s old, i.e. already dead. Either call `update()` before
   * firing (the normal per-frame order), or pass `now` / call this first.
   */
  setTime(time: number): void {
    this.time = time;
    if (this.pointsMaterial) (this.pointsMaterial.uniforms.uTime as { value: number }).value = time;
  }

  /** BEAT 1: a radial particle bloom at a world position. `now` = host clock, if it has moved since the last update(). */
  burst(pos: { x: number; y: number; z: number }, color: THREE.ColorRepresentation, scale = 1, now?: number): void {
    if (now !== undefined) this.setTime(now);
    this.burstsFired++;
    this.pulse.fire(Math.min(1, scale));
    if (!this.points || this.capacity === 0) return;
    this.color.set(color);
    const count = Math.max(1, Math.min(this.capacity, Math.round(this.burstSize * Math.max(0.25, scale))));
    const life = 0.75 + 0.5 * scale;
    for (let k = 0; k < count; k++) {
      const i = this.cursor;
      this.cursor = (this.cursor + 1) % this.capacity;
      this.posArray[i * 3] = pos.x;
      this.posArray[i * 3 + 1] = pos.y;
      this.posArray[i * 3 + 2] = pos.z;
      this.colorArray[i * 3] = this.color.r;
      this.colorArray[i * 3 + 1] = this.color.g;
      this.colorArray[i * 3 + 2] = this.color.b;
      this.startArray[i] = this.time;
      this.lifeArray[i] = life;
      this.scaleArray[i] = 0.9 * scale;
      this.gravityArray[i] = 0.55;
    }
    this.markParticlesDirty();
  }

  /** BEAT 2: an expanding ground ring at a world position (y is the floor plane). */
  ripple(
    groundPos: { x: number; y: number; z: number },
    color: THREE.ColorRepresentation,
    scale = 1,
    now?: number,
  ): void {
    if (now !== undefined) this.setTime(now);
    this.ripplesFired++;
    if (this.ripples.length === 0) return;
    const slot = this.ripples[this.rippleCursor] as RippleSlot;
    this.rippleCursor = (this.rippleCursor + 1) % this.ripples.length;
    slot.start = this.time;
    slot.life = 0.85 + 0.35 * scale;
    slot.scale = 3.2 * scale;
    (slot.material.uniforms.uColor as { value: THREE.Color }).value.set(color);
    (slot.material.uniforms.uProgress as { value: number }).value = 0;
    slot.mesh.position.set(groundPos.x, groundPos.y + 0.02, groundPos.z);
    slot.mesh.scale.setScalar(0.001);
    slot.mesh.visible = true;
  }

  /**
   * The failure grammar: cool-blue motes falling through the frame plus a desaturation
   * beat on the pulse channel. Used for every non-reward `done`.
   */
  sigh(strength = 1, center?: { x: number; y: number; z: number }, now?: number): void {
    if (now !== undefined) this.setTime(now);
    this.pulse.fireSigh(Math.min(1, strength));
    if (!this.points || this.capacity === 0) return;
    this.color.set(TOKENS.failureBlue);
    const cx = center?.x ?? 0;
    const cy = center?.y ?? 0;
    const cz = center?.z ?? 0;
    const count = Math.max(1, Math.min(this.capacity, Math.round(this.sighSize * Math.max(0.25, strength))));
    for (let k = 0; k < count; k++) {
      const i = this.cursor;
      this.cursor = (this.cursor + 1) % this.capacity;
      // reuse the slot's fixed direction as a deterministic disk sample
      const dx = this.dirArray[i * 3] ?? 0;
      const dz = this.dirArray[i * 3 + 2] ?? 0;
      const dy = this.dirArray[i * 3 + 1] ?? 0;
      this.posArray[i * 3] = cx + dx * this.sighRadius;
      this.posArray[i * 3 + 1] = cy + this.sighHeight * (0.4 + 0.6 * dy);
      this.posArray[i * 3 + 2] = cz + dz * this.sighRadius;
      this.colorArray[i * 3] = this.color.r;
      this.colorArray[i * 3 + 1] = this.color.g;
      this.colorArray[i * 3 + 2] = this.color.b;
      this.startArray[i] = this.time;
      this.lifeArray[i] = SIGH_SECONDS * (0.7 + 0.5 * dy);
      this.scaleArray[i] = 0.05;
      this.gravityArray[i] = 3.2;
    }
    this.markParticlesDirty();
  }

  /** Host DPR changes (a hero->thumb tier switch) — keeps point size physical. */
  setPixelRatio(ratio: number): void {
    if (this.pointsMaterial) (this.pointsMaterial.uniforms.uPixelRatio as { value: number }).value = ratio;
  }

  /**
   * `dt` is wall seconds (pulse decay); `time` is the host playback clock.
   * Ordering contract: update() FIRST, then fire this frame's beats — or pass the
   * clock at the call site (`burst(..., now)`), which is what a scene that
   * celebrates from inside its own update() before the pool's should do.
   */
  update(dt: number, time: number): void {
    this.time = time;
    this.pulse.update(dt);
    if (this.pointsMaterial) (this.pointsMaterial.uniforms.uTime as { value: number }).value = time;
    for (const slot of this.ripples) {
      if (!slot.mesh.visible) continue;
      const age = time - slot.start;
      if (age < 0 || age > slot.life) {
        slot.mesh.visible = false;
        continue;
      }
      const t = age / slot.life;
      (slot.material.uniforms.uProgress as { value: number }).value = t;
      slot.mesh.scale.setScalar(Math.max(0.001, slot.scale * (0.15 + 0.85 * Math.sqrt(t))));
    }
  }

  /** Scrub / teleport: nothing may survive a discontinuity. */
  clear(): void {
    this.startArray.fill(-1);
    this.markParticlesDirty();
    for (const slot of this.ripples) {
      slot.mesh.visible = false;
      slot.start = -1;
    }
    this.pulse.reset();
  }

  dispose(): void {
    this.pointsGeometry?.dispose();
    this.pointsMaterial?.dispose();
    this.sprite?.dispose();
    this.rippleGeometry?.dispose();
    for (const slot of this.ripples) slot.material.dispose();
    this.ripples.length = 0;
    this.group.removeFromParent();
    this.group.clear();
  }

  private markParticlesDirty(): void {
    for (const attr of this.attrs) attr.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// scene-attached convenience API (STYLE.md's burst/ripple/sigh signatures)
// ---------------------------------------------------------------------------

const POOL_KEY = 'oakCelebrations';

interface PoolCarrier {
  userData: Record<string, unknown>;
}

/** Attach an explicitly-built pool to a scene (and add its group). */
export function attachCelebrations(scene: THREE.Scene, pool: CelebrationPool): CelebrationPool {
  (scene as unknown as PoolCarrier).userData[POOL_KEY] = pool;
  if (pool.group.parent !== scene) scene.add(pool.group);
  return pool;
}

/**
 * The scene's pool, created on first use with its OWN pulse channel. Scenes normally
 * attach one explicitly; hosts read the channel back with `celebrationPulse(scene)`.
 */
export function celebrationsFor(scene: THREE.Scene, opts?: CelebrationPoolOptions): CelebrationPool {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (existing instanceof CelebrationPool) return existing;
  return attachCelebrations(scene, new CelebrationPool(opts));
}

/** The attached pool, or null. A LOOKUP — never creates one (hosts poll this per frame). */
export function existingCelebrations(scene: THREE.Scene): CelebrationPool | null {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  return existing instanceof CelebrationPool ? existing : null;
}

/**
 * The channel a HOST should read for a scene: the attached pool's own channel, or
 * the module convenience channel when the scene never attached one. This is the
 * whole reason pools no longer share a global: the host resolves per scene.
 */
export function celebrationPulse(scene: THREE.Scene): PulseChannel {
  return existingCelebrations(scene)?.pulse ?? pulseChannel;
}

/** BEAT 1 (STYLE.md signature). `now` = host clock when the pool has not been updated yet this frame. */
export function burst(
  scene: THREE.Scene,
  pos: { x: number; y: number; z: number },
  color: THREE.ColorRepresentation,
  scale = 1,
  now?: number,
): void {
  celebrationsFor(scene).burst(pos, color, scale, now);
}

/** BEAT 2 (STYLE.md signature). */
export function ripple(
  scene: THREE.Scene,
  groundPos: { x: number; y: number; z: number },
  color: THREE.ColorRepresentation,
  scale = 1,
  now?: number,
): void {
  celebrationsFor(scene).ripple(groundPos, color, scale, now);
}

/** The failure grammar (STYLE.md signature). */
export function sigh(scene: THREE.Scene, strength = 1, now?: number): void {
  celebrationsFor(scene).sigh(strength, undefined, now);
}

/** Release a scene's implicit pool (scene dispose()). */
export function disposeCelebrations(scene: THREE.Scene): void {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (existing instanceof CelebrationPool) existing.dispose();
  delete (scene as unknown as PoolCarrier).userData[POOL_KEY];
}

/**
 * The full three-beat reward: burst at `pos`, ripple on the ground under it, and the
 * 300 ms bloom pulse. Scenes skin it by choosing the color and scale.
 */
export function celebrate(
  pool: CelebrationPool,
  pos: { x: number; y: number; z: number },
  groundY: number,
  color: THREE.ColorRepresentation,
  scale = 1,
  now?: number,
): void {
  pool.burst(pos, color, scale, now);
  pool.ripple({ x: pos.x, y: groundY, z: pos.z }, color, scale, now);
}
