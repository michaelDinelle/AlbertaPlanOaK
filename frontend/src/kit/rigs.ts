
import * as THREE from 'three';
import { TOKENS } from './tokens';

export interface RigHandle {
  readonly camera: THREE.PerspectiveCamera;
  update(dt: number): void;
  resize(aspect: number): void;
  snap(): void;
  dispose(): void;
}

interface BaseCameraOptions {
  fov?: number;
  near?: number;
  far?: number;
  aspect?: number;
  name?: string;
}

function makeCamera(opts: BaseCameraOptions, name: string): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(
    opts.fov ?? TOKENS.fovHero,
    opts.aspect ?? 16 / 9,
    opts.near ?? 0.1,
    opts.far ?? 500,
  );
  camera.name = opts.name ?? name;
  return camera;
}

export function damp(rate: number, dt: number): number {
  if (!(rate > 0) || !(dt > 0)) return rate > 0 ? 0 : 1;
  return 1 - Math.exp(-rate * dt);
}


export interface OrbitState {
  azimuth: number;
  polar: number;
  distance: number;
}

export interface OrbitLimits {
  minPolar: number;
  maxPolar: number;
  minAzimuth: number;
  maxAzimuth: number;
  minDistance: number;
  maxDistance: number;
}

export const DEFAULT_ORBIT_LIMITS: OrbitLimits = Object.freeze({
  minPolar: 0.18,
  maxPolar: 1.36,
  minAzimuth: -Infinity,
  maxAzimuth: Infinity,
  minDistance: 2,
  maxDistance: 140,
});

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function clampOrbit(state: OrbitState, limits: OrbitLimits): OrbitState {
  state.polar = clamp(state.polar, limits.minPolar, limits.maxPolar);
  state.azimuth = clamp(state.azimuth, limits.minAzimuth, limits.maxAzimuth);
  state.distance = clamp(state.distance, limits.minDistance, limits.maxDistance);
  return state;
}

export function orbitPosition(state: OrbitState, target: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
  const sp = Math.sin(state.polar);
  return out.set(
    target.x + state.distance * sp * Math.sin(state.azimuth),
    target.y + state.distance * Math.cos(state.polar),
    target.z + state.distance * sp * Math.cos(state.azimuth),
  );
}

export interface OrbitRigOptions extends BaseCameraOptions {
  canvas?: HTMLCanvasElement | null;
  target?: readonly [number, number, number];
  distance?: number;
  azimuth?: number;
  polar?: number;
  limits?: Partial<OrbitLimits>;
  damping?: number;
  rotateSpeed?: number;
  zoomSpeed?: number;
}

export interface OrbitRig extends RigHandle {
  readonly target: THREE.Vector3;
  readonly state: OrbitState;
  readonly desired: OrbitState;
  readonly limits: OrbitLimits;
  orbit(dAzimuth: number, dPolar: number): void;
  zoom(delta: number): void;
  setDesired(azimuth: number, polar: number, distance: number): void;
  setTarget(x: number, y: number, z: number): void;
}

export function makeOrbitRig(opts: OrbitRigOptions = {}): OrbitRig {
  const camera = makeCamera(opts, 'oakOrbitRig');
  const limits: OrbitLimits = { ...DEFAULT_ORBIT_LIMITS, ...opts.limits };
  const target = new THREE.Vector3(...(opts.target ?? [0, 0, 0]));
  const desired: OrbitState = clampOrbit(
    {
      azimuth: opts.azimuth ?? 0,
      polar: opts.polar ?? 0.85,
      distance: opts.distance ?? 12,
    },
    limits,
  );
  const state: OrbitState = { ...desired };
  const rate = opts.damping ?? 8;
  const rotateSpeed = opts.rotateSpeed ?? 0.006;
  const zoomSpeed = opts.zoomSpeed ?? 0.0016;
  const scratch = new THREE.Vector3();

  const apply = (): void => {
    orbitPosition(state, target, scratch);
    camera.position.copy(scratch);
    camera.lookAt(target);
  };
  apply();

  const canvas = opts.canvas ?? null;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let activePointer = -1;

  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;
    dragging = true;
    activePointer = e.pointerId;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas?.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (!dragging || e.pointerId !== activePointer) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    rig.orbit(-dx * rotateSpeed, -dy * rotateSpeed);
  };
  const onPointerUp = (e: PointerEvent): void => {
    if (e.pointerId !== activePointer) return;
    dragging = false;
    activePointer = -1;
    canvas?.releasePointerCapture?.(e.pointerId);
  };
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    rig.zoom(e.deltaY * zoomSpeed * desired.distance);
  };

  const home: OrbitState = { ...desired };
  const rig: OrbitRig = {
    camera,
    target,
    state,
    desired,
    limits,
    orbit(dAzimuth, dPolar) {
      desired.azimuth += dAzimuth;
      desired.polar += dPolar;
      clampOrbit(desired, limits);
    },
    zoom(delta) {
      desired.distance += delta;
      clampOrbit(desired, limits);
    },
    setDesired(azimuth, polar, distance) {
      desired.azimuth = azimuth;
      desired.polar = polar;
      desired.distance = distance;
      clampOrbit(desired, limits);
    },
    setTarget(x, y, z) {
      target.set(x, y, z);
    },
    update(dt) {
      const k = damp(rate, dt);
      state.azimuth += (desired.azimuth - state.azimuth) * k;
      state.polar += (desired.polar - state.polar) * k;
      state.distance += (desired.distance - state.distance) * k;
      clampOrbit(state, limits);
      apply();
    },
    resize(aspect) {
      camera.aspect = aspect > 0 ? aspect : 1;
      camera.updateProjectionMatrix();
    },
    snap() {
      state.azimuth = desired.azimuth;
      state.polar = desired.polar;
      state.distance = desired.distance;
      apply();
    },
    dispose() {
      if (!canvas) return;
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
    },
  };

  if (canvas) {
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', () => {
      desired.azimuth = home.azimuth;
      desired.polar = home.polar;
      desired.distance = home.distance;
    });
  }

  return rig;
}

export const OrbitRig = makeOrbitRig;


export interface TrackRigOptions extends BaseCameraOptions {
  canvas?: HTMLCanvasElement | null;
  offset?: readonly [number, number, number];
  lookOffset?: readonly [number, number, number];
  lookAhead?: number;
  maxLookAhead?: number;
  damping?: number;
  lookDamping?: number;
  velocityDamping?: number;
}

export interface TrackRig extends RigHandle {
  readonly target: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  setTarget(x: number, y: number, z: number): void;
  setVelocity(x: number, y: number, z: number): void;
  snap(): void;
}

export function makeTrackRig(opts: TrackRigOptions = {}): TrackRig {
  const camera = makeCamera(opts, 'oakTrackRig');
  const offset = new THREE.Vector3(...(opts.offset ?? [0, 1.6, 6]));
  const lookOffset = new THREE.Vector3(...(opts.lookOffset ?? [0, 0.4, 0]));
  const target = new THREE.Vector3();
  const previousTarget = new THREE.Vector3();
  const velocity = new THREE.Vector3();
  const rawVelocity = new THREE.Vector3();
  const lead = new THREE.Vector3();
  const desiredPos = new THREE.Vector3();
  const desiredLook = new THREE.Vector3();
  const currentLook = new THREE.Vector3();

  const posRate = opts.damping ?? 4;
  const lookRate = opts.lookDamping ?? 6;
  const velRate = opts.velocityDamping ?? 5;
  const lookAhead = opts.lookAhead ?? 0.35;
  const maxLookAhead = opts.maxLookAhead ?? 3;

  let primed = false;
  let externalVelocity = false;

  let userYaw = 0;
  let userPitch = 0;
  let userZoom = 1;
  const orbited = new THREE.Vector3();
  const Y_AXIS = new THREE.Vector3(0, 1, 0);
  const PITCH_MIN = -0.5;
  const PITCH_MAX = 0.9;
  const ZOOM_MIN = 0.45;
  const ZOOM_MAX = 2.8;

  const computeDesired = (): void => {
    lead.copy(velocity).multiplyScalar(lookAhead);
    const len = lead.length();
    if (len > maxLookAhead) lead.multiplyScalar(maxLookAhead / len);
    orbited.copy(offset);
    if (userPitch !== 0) {
      const px = -offset.z;
      const pz = offset.x;
      const norm = Math.hypot(px, pz) || 1;
      orbited.applyAxisAngle(scratchAxis.set(px / norm, 0, pz / norm), userPitch);
    }
    orbited.applyAxisAngle(Y_AXIS, userYaw).multiplyScalar(userZoom);
    const minY = Math.min(offset.y, 0.3);
    if (orbited.y < minY) orbited.y = minY;
    desiredPos.copy(target).add(orbited).add(lead);
    desiredLook.copy(target).add(lookOffset).add(lead);
  };
  const scratchAxis = new THREE.Vector3();

  const canvas = opts.canvas ?? null;
  if (canvas) {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let activePointer = -1;
    canvas.addEventListener('pointerdown', (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragging = true;
      activePointer = e.pointerId;
      lastX = e.clientX;
      lastY = e.clientY;
      try { canvas.setPointerCapture(e.pointerId); } catch {  }
    });
    canvas.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging || e.pointerId !== activePointer) return;
      userYaw -= (e.clientX - lastX) * 0.006;
      userPitch = Math.min(Math.max(userPitch + (e.clientY - lastY) * 0.004, PITCH_MIN), PITCH_MAX);
      lastX = e.clientX;
      lastY = e.clientY;
    });
    const endDrag = (e: PointerEvent): void => {
      if (e.pointerId !== activePointer) return;
      dragging = false;
      activePointer = -1;
      try { canvas.releasePointerCapture(e.pointerId); } catch {  }
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('lostpointercapture', () => { dragging = false; activePointer = -1; });
    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      userZoom = Math.min(Math.max(userZoom * (1 + e.deltaY * 0.0011), ZOOM_MIN), ZOOM_MAX);
    }, { passive: false });
    canvas.addEventListener('dblclick', () => {
      userYaw = 0;
      userPitch = 0;
      userZoom = 1;
    });
  }

  const rig: TrackRig = {
    camera,
    target,
    velocity,
    setTarget(x, y, z) {
      target.set(x, y, z);
      if (!primed) {
        previousTarget.copy(target);
        primed = true;
      }
    },
    setVelocity(x, y, z) {
      externalVelocity = true;
      rawVelocity.set(x, y, z);
    },
    update(dt) {
      if (externalVelocity) {
        if (dt > 0) velocity.lerp(rawVelocity, damp(velRate, dt));
      } else if (dt > 0) {
        rawVelocity.copy(target).sub(previousTarget).divideScalar(dt);
        velocity.lerp(rawVelocity, damp(velRate, dt));
      }
      previousTarget.copy(target);
      computeDesired();
      camera.position.lerp(desiredPos, damp(posRate, dt));
      currentLook.lerp(desiredLook, damp(lookRate, dt));
      camera.lookAt(currentLook);
    },
    resize(aspect) {
      camera.aspect = aspect > 0 ? aspect : 1;
      camera.updateProjectionMatrix();
    },
    snap() {
      previousTarget.copy(target);
      velocity.set(0, 0, 0);
      computeDesired();
      camera.position.copy(desiredPos);
      currentLook.copy(desiredLook);
      camera.lookAt(currentLook);
      primed = true;
    },
    dispose() {
    },
  };

  rig.snap();
  return rig;
}

export const TrackRig = makeTrackRig;


export interface FrameRigOptions extends BaseCameraOptions {
  position: readonly [number, number, number];
  lookAt?: readonly [number, number, number];
  canvas?: HTMLCanvasElement | null;
  parallax?: number;
  parallaxY?: number;
  damping?: number;
}

export interface FrameRig extends RigHandle {
  readonly base: THREE.Vector3;
  readonly lookAt: THREE.Vector3;
  setPointer(nx: number, ny: number): void;
  setBase(x: number, y: number, z: number): void;
}

export function makeFrameRig(opts: FrameRigOptions): FrameRig {
  const camera = makeCamera(opts, 'oakFrameRig');
  const base = new THREE.Vector3(...opts.position);
  const lookAt = new THREE.Vector3(...(opts.lookAt ?? [0, 0, 0]));
  const parallax = opts.parallax ?? 0.18;
  const parallaxY = opts.parallaxY ?? 0.5;
  const rate = opts.damping ?? 3;
  const desiredOffset = new THREE.Vector2(0, 0);
  const offset = new THREE.Vector2(0, 0);
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const forward = new THREE.Vector3();
  const scratch = new THREE.Vector3();

  const apply = (): void => {
    forward.copy(lookAt).sub(base);
    if (forward.lengthSq() < 1e-12) forward.set(0, 0, -1);
    forward.normalize();
    right.crossVectors(forward, THREE.Object3D.DEFAULT_UP);
    if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
    right.normalize();
    up.crossVectors(right, forward).normalize();
    scratch
      .copy(base)
      .addScaledVector(right, offset.x * parallax)
      .addScaledVector(up, offset.y * parallax * parallaxY);
    camera.position.copy(scratch);
    camera.lookAt(lookAt);
  };
  apply();

  const canvas = opts.canvas ?? null;
  const onPointerMove = (e: PointerEvent): void => {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    rig.setPointer(((e.clientX - rect.left) / rect.width) * 2 - 1, -(((e.clientY - rect.top) / rect.height) * 2 - 1));
  };
  const onPointerLeave = (): void => rig.setPointer(0, 0);

  const rig: FrameRig = {
    camera,
    base,
    lookAt,
    setPointer(nx, ny) {
      desiredOffset.set(clamp(nx, -1, 1), clamp(ny, -1, 1));
    },
    setBase(x, y, z) {
      base.set(x, y, z);
    },
    update(dt) {
      const k = damp(rate, dt);
      offset.x += (desiredOffset.x - offset.x) * k;
      offset.y += (desiredOffset.y - offset.y) * k;
      apply();
    },
    resize(aspect) {
      camera.aspect = aspect > 0 ? aspect : 1;
      camera.updateProjectionMatrix();
    },
    snap() {
      offset.copy(desiredOffset);
      apply();
    },
    dispose() {
      if (!canvas) return;
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerleave', onPointerLeave);
    },
  };

  if (canvas) {
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerleave', onPointerLeave);
  }

  return rig;
}

export const FrameRig = makeFrameRig;
