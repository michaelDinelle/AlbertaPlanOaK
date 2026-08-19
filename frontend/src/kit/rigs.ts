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
    opts.far ?? 900,
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
  panSpeed?: number;
  follow?: number;
  lookAhead?: number;
}

export interface OrbitRig extends RigHandle {
  readonly target: THREE.Vector3;
  readonly state: OrbitState;
  readonly desired: OrbitState;
  readonly limits: OrbitLimits;
  orbit(dAzimuth: number, dPolar: number): void;
  zoom(delta: number): void;
  pan(dx: number, dy: number): void;
  setDesired(azimuth: number, polar: number, distance: number): void;
  setTarget(x: number, y: number, z: number): void;
  follow(x: number, y: number, z: number): void;
}

export function orbitFromPose(
  position: readonly [number, number, number],
  lookAt: readonly [number, number, number],
): OrbitState {
  const dx = position[0] - lookAt[0];
  const dy = position[1] - lookAt[1];
  const dz = position[2] - lookAt[2];
  const distance = Math.hypot(dx, dy, dz) || 1;
  return {
    azimuth: Math.atan2(dx, dz),
    polar: Math.acos(Math.min(1, Math.max(-1, dy / distance))),
    distance,
  };
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
  const rate = opts.damping ?? 5;
  const rotateSpeed = opts.rotateSpeed ?? 0.004;
  const zoomSpeed = opts.zoomSpeed ?? 0.0012;
  const panSpeed = opts.panSpeed ?? 0.0022;
  const followRate = opts.follow ?? 0;
  const scratch = new THREE.Vector3();
  const panOffset = new THREE.Vector3();
  const followTarget = new THREE.Vector3(...(opts.target ?? [0, 0, 0]));
  const anchor = new THREE.Vector3(...(opts.target ?? [0, 0, 0]));
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const forward = new THREE.Vector3();
  let followPrimed = followRate <= 0;

  const apply = (): void => {
    target.copy(anchor).add(panOffset);
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

  let panning = false;
  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0 && e.button !== 1 && e.button !== 2) return;
    dragging = true;
    panning = e.button !== 0 || e.shiftKey;
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
    if (panning) rig.pan(dx, dy);
    else rig.orbit(-dx * rotateSpeed, -dy * rotateSpeed);
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
    pan(dx, dy) {
      forward.copy(camera.position).sub(target).normalize();
      right.crossVectors(THREE.Object3D.DEFAULT_UP, forward);
      if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
      right.normalize();
      up.crossVectors(forward, right).normalize();
      const k = panSpeed * state.distance;
      panOffset.addScaledVector(right, dx * k).addScaledVector(up, dy * k);
    },
    follow(x, y, z) {
      followTarget.set(x, y, z);
      if (!followPrimed) {
        anchor.copy(followTarget);
        followPrimed = true;
      }
    },
    setDesired(azimuth, polar, distance) {
      desired.azimuth = azimuth;
      desired.polar = polar;
      desired.distance = distance;
      clampOrbit(desired, limits);
    },
    setTarget(x, y, z) {
      anchor.set(x, y, z);
      followTarget.set(x, y, z);
    },
    update(dt) {
      const k = damp(rate, dt);
      if (followRate > 0) anchor.lerp(followTarget, damp(followRate, dt));
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
      if (followRate > 0) anchor.copy(followTarget);
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
    canvas.addEventListener('contextmenu', (e: Event) => e.preventDefault());
    canvas.addEventListener('dblclick', () => {
      desired.azimuth = home.azimuth;
      desired.polar = home.polar;
      desired.distance = home.distance;
      panOffset.set(0, 0, 0);
    });
  }

  return rig;
}
