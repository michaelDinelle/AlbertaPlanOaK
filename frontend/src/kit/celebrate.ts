import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, float, length, max, mix, positionView, pow, smoothstep, step, texture, uniform, uv, varying, vec3 } from 'three/tsl';
import type { NumberUniform } from './materials';
import { pointSize } from './particles';
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

const PARTICLE_CAPACITY = 768;
const RIPPLE_CAPACITY = 8;

export interface CelebrationPoolOptions {
  burstSize?: number;
  sighSize?: number;
  sighRadius?: number;
  sighHeight?: number;
  seed?: number;
  name?: string;
}

interface RippleSlot {
  mesh: THREE.Mesh;
  material: THREE.MeshBasicNodeMaterial;
  color: THREE.Color;
  progress: NumberUniform;
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

  private readonly pointsMaterial: THREE.PointsNodeMaterial;
  private readonly pointsGeometry: THREE.BufferGeometry;
  private readonly sprite: THREE.DataTexture;
  private readonly clock = uniform(0);

  private readonly posArray: Float32Array;
  private readonly dirArray: Float32Array;
  private readonly colorArray: Float32Array;
  private readonly startArray: Float32Array;
  private readonly lifeArray: Float32Array;
  private readonly scaleArray: Float32Array;
  private readonly gravityArray: Float32Array;
  private readonly attrs: THREE.BufferAttribute[] = [];

  private readonly ripples: RippleSlot[] = [];
  private readonly rippleGeometry: THREE.BufferGeometry;

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
    this.pulse = makePulseChannel();
    this.group.name = opts.name ?? 'oakCelebrations';

    const n = this.capacity;
    this.posArray = new Float32Array(n * 3);
    this.dirArray = new Float32Array(n * 4);
    this.colorArray = new Float32Array(n * 3);
    this.startArray = new Float32Array(n).fill(-1);
    this.lifeArray = new Float32Array(n).fill(1);
    this.scaleArray = new Float32Array(n).fill(1);
    this.gravityArray = new Float32Array(n);

    const rng = mulberry32((opts.seed ?? 0) ^ 0x51a3f7);
    for (let i = 0; i < n; i++) {
      const u = rng() * 2 - 1;
      const a = rng() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - u * u));
      const y = Math.abs(u) * 0.85 + 0.15;
      this.dirArray[i * 4] = Math.cos(a) * r;
      this.dirArray[i * 4 + 1] = y;
      this.dirArray[i * 4 + 2] = Math.sin(a) * r;
      this.dirArray[i * 4 + 3] = 0.5 + rng() * 0.9;
    }

    this.sprite = radialSprite(64, 2.2);
    const start = attribute<'float'>('aStart', 'float'), life = attribute<'float'>('aLife', 'float'), scale = attribute<'float'>('aScale', 'float'), dir = attribute<'vec4'>('aDir', 'vec4');
    const age = this.clock.sub(start);
    const live = step(0, start).mul(step(0, age)).mul(step(age, life));
    const t = clamp(age.div(max(life, 0.0001)), 0, 1);
    const p = attribute<'vec3'>('aPosition', 'vec3').add(dir.xyz.mul(scale.mul(pow(t.oneMinus(), 2.4).oneMinus())));
    this.pointsMaterial = new THREE.PointsNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: false,
      sizeAttenuation: false,
      alphaTest: 0.004,
      positionNode: vec3(p.x, p.y.sub(attribute<'float'>('aGravity', 'float').mul(scale).mul(t).mul(t)), p.z),
      sizeNode: pointSize(live.mul(dir.w).mul(scale).mul(float(220).div(max(positionView.z.negate(), 0.001)))),
      colorNode: attribute<'vec3'>('aColor', 'vec3').mul(2.4),
      opacityNode: texture(this.sprite, uv()).a.mul(varying(live.mul(t.oneMinus()).mul(t.oneMinus()))),
    });
    this.pointsMaterial.name = 'oakCelebrationBurst';
    const points = new THREE.Sprite(this.pointsMaterial as unknown as THREE.SpriteMaterial);
    this.pointsGeometry = points.geometry.clone();
    const add = (name: string, array: Float32Array, size: number): void => {
      const attr = new THREE.InstancedBufferAttribute(array, size);
      this.pointsGeometry.setAttribute(name, attr);
      this.attrs.push(attr);
    };
    add('aPosition', this.posArray, 3);
    add('aDir', this.dirArray, 4);
    add('aColor', this.colorArray, 3);
    add('aStart', this.startArray, 1);
    add('aLife', this.lifeArray, 1);
    add('aScale', this.scaleArray, 1);
    add('aGravity', this.gravityArray, 1);
    points.geometry = this.pointsGeometry;
    points.count = n;
    points.name = 'oakCelebrationBurst';
    points.frustumCulled = false;
    this.group.add(points);

    this.rippleGeometry = new THREE.PlaneGeometry(1.5, 1.5);
    this.rippleGeometry.rotateX(-Math.PI / 2);
    const radius = length(uv().sub(0.5)).mul(3);
    for (let i = 0; i < RIPPLE_CAPACITY; i++) {
      const color = new THREE.Color(TOKENS.celebrationGold), progress = uniform(1);
      const material = new THREE.MeshBasicNodeMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        fog: false,
        alphaTest: 0.004,
        colorNode: uniform(color).mul(1.8),
        opacityNode: smoothstep(0, mix(0.42, 0.08, progress), abs(radius.sub(1))).oneMinus().mul(progress.oneMinus().mul(progress.oneMinus())),
      });
      material.name = 'oakCelebrationRipple';
      const mesh = new THREE.Mesh(this.rippleGeometry, material);
      mesh.name = `oakCelebrationRipple${i}`;
      mesh.visible = false;
      mesh.frustumCulled = false;
      mesh.renderOrder = 2;
      this.group.add(mesh);
      this.ripples.push({ mesh, material, color, progress, start: -1, life: 1, scale: 1 });
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
    this.clock.value = time;
  }

  burst(pos: { x: number; y: number; z: number }, color: THREE.ColorRepresentation, scale = 1, now?: number): void {
    if (now !== undefined) this.setTime(now);
    this.burstsFired++;
    this.pulse.fire(Math.min(1, scale));
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
    slot.color.set(color);
    slot.progress.value = 0;
    slot.mesh.position.set(groundPos.x, groundPos.y + 0.02, groundPos.z);
    slot.mesh.scale.setScalar(0.001);
    slot.mesh.visible = true;
  }

  sigh(strength = 1, center?: { x: number; y: number; z: number }, now?: number): void {
    if (now !== undefined) this.setTime(now);
    this.pulse.fireSigh(Math.min(1, strength));
    this.color.set(TOKENS.failureBlue);
    const cx = center?.x ?? 0;
    const cy = center?.y ?? 0;
    const cz = center?.z ?? 0;
    const count = Math.max(1, Math.min(this.capacity, Math.round(this.sighSize * Math.max(0.25, strength))));
    for (let k = 0; k < count; k++) {
      const i = this.cursor;
      this.cursor = (this.cursor + 1) % this.capacity;
      const dx = this.dirArray[i * 4] ?? 0;
      const dz = this.dirArray[i * 4 + 2] ?? 0;
      const dy = this.dirArray[i * 4 + 1] ?? 0;
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

  update(dt: number, time: number): void {
    this.time = time;
    this.pulse.update(dt);
    this.clock.value = time;
    for (const slot of this.ripples) {
      if (!slot.mesh.visible) continue;
      const age = time - slot.start;
      if (age < 0 || age > slot.life) {
        slot.mesh.visible = false;
        continue;
      }
      const t = age / slot.life;
      slot.progress.value = t;
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
    this.pointsGeometry.dispose();
    this.pointsMaterial.dispose();
    this.sprite.dispose();
    this.rippleGeometry.dispose();
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


export function celebrationPulse(scene: THREE.Scene): PulseChannel {
  const pool = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (!(pool instanceof CelebrationPool)) throw new Error(`${scene.name} has no celebration pool`);
  return pool.pulse;
}




export function disposeCelebrations(scene: THREE.Scene): void {
  const existing = (scene as unknown as PoolCarrier).userData[POOL_KEY];
  if (existing instanceof CelebrationPool) existing.dispose();
  delete (scene as unknown as PoolCarrier).userData[POOL_KEY];
}

