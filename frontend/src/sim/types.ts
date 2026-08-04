// FROZEN Phase-0 contract: the sim-core accuracy interfaces.
// sim/ is pure TypeScript — zero DOM and zero Three.js imports. It is the single
// source of geometric/semantic truth for scenes, tests, and the theater host.
// Additive changes only after Phase 0.

import type { EnvId } from '@shared/envs';

// ---------- events (derived ONLY from state deltas by sim-core) ----------

export type SimEvent =
  | { type: 'goal' }                                  // +1 reward event (goal/treasure/summit/success)
  | { type: 'done' }                                  // env emitted done=1 (reset/failure/choice)
  | { type: 'key'; index: number }                    // key picked up (index 0 for single-key envs)
  | { type: 'door'; index: number; open: boolean }    // door state change
  | { type: 'cue'; value: 0 | 1 }                     // MemoryChain cue (re)drawn at pos 0
  | { type: 'teleport' }                              // discontinuous jump (episode reset / respawn)
  | { type: 'resurface' };                            // DeepSea family wrap back to (0,0)

// ---------- frames ----------

export interface EnvFrame<S = unknown> {
  /** record row index; -1 for the synthesized deterministic init frame */
  step: number;
  /** 6 floats as recorded (discrete envs: ctrl[0] = action index) */
  ctrl: Float32Array;
  done: number;
  /** 9 floats, post-step, per-env packing (init frame: synthesized) */
  pose: Float32Array;
  /** env-specific decoded semantic state (per-env interface, cast at use site) */
  state: S;
}

// ---------- geometry (pure data + functions; kit converts to Three) ----------

export interface GridCell { x: number; y: number }

export interface GridGeometry {
  w: number;
  h: number;
  /** w*h row-major occupancy, 1 = wall. For 1-D corridors h === 1. */
  walls: Uint8Array;
  /** initial goal cell (the LIVE goal lives in frame state for respawning envs) */
  goal?: GridCell;
  doors?: GridCell[];
  /** initial key cells (MultiKeyCorridor: regenerated via ctrl replay per respawn) */
  keys?: GridCell[];
  hallways?: GridCell[];
  /** DeepSea/LargeDeepSea per-cell action-flip mask, w*h row-major (1 = right on action 1) */
  actionMask?: Uint8Array;
  /** ProceduralMaze carved solution path — DEBUG OVERLAY ONLY, never shown in art */
  carvedPath?: GridCell[];
}

/** FK output point names, in order; 20 floats = 10 points × [x, z] */
export const CHEETAH_FK_POINTS = [
  'o1', 'bt', 'ft', 'head', 'bs', 'bf', 'toe_b', 'fs', 'ff', 'toe_f',
] as const;

export interface CheetahGeometry {
  /** planar FK: qpos9 -> 20 floats ([x,z] per CHEETAH_FK_POINTS entry). Allocation-free with `out`. */
  fk(qpos: ArrayLike<number>, out?: Float32Array): Float32Array;
  /** capsule radius (all body geoms) */
  radius: number;
  headRadius: number;
}

export interface ClassicGeometry {
  /** MountainCar terrain: sin(3x)*0.45+0.55 */
  heightAt?(x: number): number;
  /** named world anchors ([x, y, z]): pivot points, track ends, goal markers, failure bounds */
  anchors: Record<string, readonly [number, number, number]>;
}

export interface EnvGeometry {
  kind: 'grid' | 'cheetah' | 'classic';
  grid?: GridGeometry;
  cheetah?: CheetahGeometry;
  classic?: ClassicGeometry;
}

// ---------- decoders ----------

export interface EnvDecoder<S = unknown, St = unknown> {
  envId: EnvId;
  /** true when hidden state must be replayed from ctrl (MultiKeyCorridor) */
  needsReplay: boolean;
  /** regenerate hidden static state from the lane seed (Philox walls/masks; key layout) */
  initStatic(seed: number): St;
  /** ground-truth geometry for scenes/overlays/tests */
  geometry(st: St, seed: number): EnvGeometry;
  /** synthesized frame −1 from the env's deterministic init (records are post-step) */
  initFrame(seed: number, st: St): EnvFrame<S>;
  /** decode record row i; prev-threaded so replay-dependent envs can advance hidden state */
  decodeFrame(i: number, raw16: Float32Array, st: St, prev: EnvFrame<S>): EnvFrame<S>;
  /** events fired on the prev -> next transition; pure function of the two frames + static */
  deriveEvents(prev: EnvFrame<S>, next: EnvFrame<S>, st: St): SimEvent[];
}

// ---------- decoded recording (theater input) ----------

export interface DecodedRecording<S = unknown> {
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: EnvGeometry;
  /** frames[0] is the synthesized init frame (step −1); frames[i] = record row i−1 */
  frames: EnvFrame<S>[];
  /**
   * events[i] fired on the transition INTO frames[i]; events[0] = [].
   *
   * READ-ONLY: entries (and the SimEvent objects inside them) must never be
   * mutated. Empty entries are the shared frozen `NO_EVENTS` singleton
   * (`src/sim/events.ts`) — every decoder returns that one object rather than a
   * fresh `[]`, so `events[i] === NO_EVENTS` is a valid "nothing fired" test and
   * a `push` into an empty entry throws in strict mode. Consumers that need to
   * accumulate events copy into their own buffer.
   */
  events: SimEvent[][];
  /**
   * Replay health, present only for `needsReplay` decoders (MultiKeyCorridor):
   * the key CELLS are not recorded, so the decoder replays them from ctrl and
   * cross-checks every recorded pose field per frame. A disagreement means the
   * replay has desynced from the record and everything the replay owns (key
   * positions) is suspect from that step on.
   * `firstDisagreeStep` is a record row index (frames[i+1].step === i).
   */
  replay?: { firstDisagreeStep: number | null; disagreeCount: number };
  /**
   * Count of decoded record rows carrying a non-finite value in any of their 16
   * floats (ctrl / done / pose). Non-zero means the recording itself is
   * damaged: `Math.trunc(NaN) & 3 === 0` would otherwise decode as a silent,
   * plausible-looking action.
   */
  nonFiniteRows?: number;
}

export interface RawRecording {
  envId: EnvId;
  seed: number;
  /** flat LE f32[n][16]: ctrl[6], done, pose[9] */
  data: Float32Array;
  steps: number;
}
