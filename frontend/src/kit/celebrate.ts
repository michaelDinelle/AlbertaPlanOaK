import * as THREE from 'three';
import { mulberry32 } from './rng';
import { radialSprite } from './textures';
import { TOKENS } from './tokens';

export const PULSE_SECONDS = 0.3;
export const SIGH_SECONDS = 0.9;

export interface PulseChannel {
  bloom: number;
  sigh: number;
  fire(strength?: number): void;
  fireSigh(strength?: number): void;
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

export const pulseChannel: PulseChannel = makePulseChannel();

const PARTICLE_CAPACITY = 768;
const RIPPLE_CAPACITY = 8;

export interface CelebrationPoolOptions {
  burstSize?: number;
  sighSize?: number;
  sighRadius?: number;
  sighHeight?: number;
  seed?: number;
  pulse?: PulseChannel;
  name?: string;
}

const BURST_VERT =  `
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

const BURST_FRAG =  `
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

const RIPPLE_VERT =  `
#include <common>
varying vec2 vUvR;
void main() {
	vUvR = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const RIPPLE_FRAG =  `
#include <common>
uniform vec3 uColor;
uniform float uProgress;
uniform float uIntensity;
varying vec2 vUvR;
void main() {
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
    this.capacity = PARTICLE_CAPACITY;
    this.burstSize = Math.max(1, opts.burstSize ?? 48);
    this.sighSize = Math.max(1, opts.sighSize ?? 40);
    this.sighRadius = opts.sighRadius ?? 6;
    this.sighHeight = opts.sighHeight ?? 3.2;
    this.pulse = opts.pulse ?? makePulseChannel();
    this.group.name = opts.name ?? 'oakCelebrations';

    const n = this.capacity;
    this.posArray = new Float32Array(Math.max(1, n) * 3);
    this.dirArray = new Float32Array(Math.max(1, n) * 3);
    this.colorArray = new Float32Array(Math.max(1, n) * 3);
    this.startArray = new Float32Array(Math.max(1, n)).fill(-1);
    this.lifeArray = new Float32Array(Math.max(1, n)).fill(1);
    this.sizeArray = new Float32Array(Math.max(1, n)).fill(1);
    this.scaleArray = new Float32Array(Math.max(1, n)).fill(1);
    this.gravityArray = new Float32Array(Math.max(1, n));

    const rng = mulberry32((opts.seed ?? 0) ^ 0x51a3f7);
    for (let i = 0; i < n; i++) {
      const u = rng() * 2 - 1;
      const a = rng() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - u * u));
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

    this.rippleGeometry = new THREE.PlaneGeometry(1, 1);
    this.rippleGeometry.rotateX(-Math.PI / 2);
    for (let i = 0; i < RIPPLE_CAPACITY; i++) {
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
  
  }

  get stats(): { bursts: number; ripples: number; particles: number; cursor: number } {
    return { bursts: this.burstsFired, ripples: this.ripplesFired, particles: this.capacity, cursor: this.cursor };
  }

  get activeRipples(): number {
    let n = 0;
    for (const slot of this.ripples) if (slot.mesh.visible) n++;
    return n;
  }

  setTime(time: number): void {
    this.time = time;
    if (this.pointsMaterial) (this.pointsMaterial.uniforms.uTime as { value: number }).value = time;
  }

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

  setPixelRatio(ratio: number): void {
    if (this.pointsMaterial) (this.pointsMaterial.uniforms.uPixelRatio as { value: number }).value = ratio;
  }

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

const POOL_KEY = 'oakCelebrations';

interface PoolCarrier {
  userData: Record<string, unknown>;
}

export function attachCelebrations(scene: THREE.Scene, pool: CelebrationPool): CelebrationPool {
  (scene as unknown as PoolCarrier).userData[POOL_KEY] = pool;
  if (pool.group.parent !== scene) scene.add(pool.group);
  return pool;
}

export function celebrationsFor(scene: THREE.Scene, opts?: CelebrationPoolOptions): CelebrationPool {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (existing instanceof CelebrationPool) return existing;
  return attachCelebrations(scene, new CelebrationPool(opts));
}

export function existingCelebrations(scene: THREE.Scene): CelebrationPool | null {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  return existing instanceof CelebrationPool ? existing : null;
}

export function celebrationPulse(scene: THREE.Scene): PulseChannel {
  return existingCelebrations(scene)?.pulse ?? pulseChannel;
}

export function burst(
  scene: THREE.Scene,
  pos: { x: number; y: number; z: number },
  color: THREE.ColorRepresentation,
  scale = 1,
  now?: number,
): void {
  celebrationsFor(scene).burst(pos, color, scale, now);
}

export function ripple(
  scene: THREE.Scene,
  groundPos: { x: number; y: number; z: number },
  color: THREE.ColorRepresentation,
  scale = 1,
  now?: number,
): void {
  celebrationsFor(scene).ripple(groundPos, color, scale, now);
}

export function sigh(scene: THREE.Scene, strength = 1, now?: number): void {
  celebrationsFor(scene).sigh(strength, undefined, now);
}

export function disposeCelebrations(scene: THREE.Scene): void {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (existing instanceof CelebrationPool) existing.dispose();
  delete (scene as unknown as PoolCarrier).userData[POOL_KEY];
}

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
