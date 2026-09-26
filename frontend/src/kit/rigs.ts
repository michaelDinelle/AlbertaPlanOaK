import * as THREE from 'three/webgpu';
import { clampRange } from './mix';
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
  maxDistance: 36,
});

const PAN_SPEED = 0.0022;

export function clampOrbit(state: OrbitState, limits: OrbitLimits): OrbitState {
  state.polar = clampRange(state.polar, limits.minPolar, limits.maxPolar);
  state.azimuth = clampRange(state.azimuth, limits.minAzimuth, limits.maxAzimuth);
  state.distance = clampRange(state.distance, limits.minDistance, limits.maxDistance);
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
  floor?: (x: number, z: number) => number;
}

export interface OrbitRig extends RigHandle {
  readonly target: THREE.Vector3;
  readonly state: OrbitState;
  readonly desired: OrbitState;
  readonly limits: OrbitLimits;
  reset(): void;
  orbit(dAzimuth: number, dPolar: number): void;
  zoom(delta: number): void;
  pan(dx: number, dy: number): void;
  setDesired(azimuth: number, polar: number, distance: number): void;
  setTarget(x: number, y: number, z: number): void;
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
  const scratch = new THREE.Vector3();
  const panOffset = new THREE.Vector3();
  const anchor = new THREE.Vector3(...(opts.target ?? [0, 0, 0]));
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const forward = new THREE.Vector3();

  const apply = (): void => {
    target.copy(anchor).add(panOffset);
    orbitPosition(state, target, scratch);
    if (opts.floor) scratch.y = Math.max(scratch.y, opts.floor(scratch.x, scratch.z));
    camera.position.copy(scratch);
    camera.lookAt(target);
  };
  apply();

  const canvas = opts.canvas ?? null;
  const pointers = new Map<number, { x: number; y: number }>();
  let panning = false;
  let pinchSpan = 0;

  const pinch = (): { span: number; x: number; y: number } => {
    const [a, b] = [...pointers.values()];
    if (!a || !b) return { span: 0, x: 0, y: 0 };
    return {
      span: Math.hypot(a.x - b.x, a.y - b.y),
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
    };
  };

  const onPointerDown = (e: PointerEvent): void => {
    if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 1 && e.button !== 2) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    panning = e.pointerType === 'mouse' && (e.button !== 0 || e.shiftKey);
    if (pointers.size === 2) pinchSpan = pinch().span;
    canvas?.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent): void => {
    const previous = pointers.get(e.pointerId);
    if (!previous) return;
    if (pointers.size === 2) {
      const before = pinch();
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const after = pinch();
      if (pinchSpan > 0 && after.span > 0) {
        desired.distance *= pinchSpan / after.span;
        clampOrbit(desired, limits);
      }
      pinchSpan = after.span;
      rig.pan(after.x - before.x, after.y - before.y);
      return;
    }
    const dx = e.clientX - previous.x;
    const dy = e.clientY - previous.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size !== 1) return;
    if (panning) rig.pan(dx, dy);
    else rig.orbit(-dx * rotateSpeed, -dy * rotateSpeed);
  };
  const onPointerUp = (e: PointerEvent): void => {
    if (!pointers.delete(e.pointerId)) return;
    pinchSpan = pointers.size === 2 ? pinch().span : 0;
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
    reset() {
      Object.assign(desired, home);
      panOffset.set(0, 0, 0);
    },
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
      const k = PAN_SPEED * state.distance;
      panOffset.addScaledVector(right, dx * k).addScaledVector(up, dy * k);
    },
    setDesired(azimuth, polar, distance) {
      desired.azimuth = azimuth;
      desired.polar = polar;
      desired.distance = distance;
      clampOrbit(desired, limits);
    },
    setTarget(x, y, z) {
      anchor.set(x, y, z);
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
      canvas.removeEventListener('dblclick', rig.reset);
    },
  };

  if (canvas) {
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e: Event) => e.preventDefault());
    canvas.addEventListener('dblclick', rig.reset);
  }

  return rig;
}
