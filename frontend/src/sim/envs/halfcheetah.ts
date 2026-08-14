
import { cheetahGeometry, ROOT_Z0 } from '../fk/cheetah';
import { STREAM_RESET, boxMuller, philoxWords, uniform } from '../philox';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const HC_TIMESTEP = 0.01;
export const HC_FRAME_SKIP = 5;
export const HC_RESET_NOISE = 0.1;
export const CHEETAH_DT = 0.05;
export const CHEETAH_TELEPORT_DX = 10;

export const QPOS_LEN = 9;


export interface CheetahStatic {
  seed: number;
  resetIndex: number;
  initQpos: Float32Array;
  initQvel: Float32Array;
}

export interface CheetahState {
  qpos: Float32Array;
  rootX: number;
  rootZ: number;
  pitch: number;
  velX: number;
}


export function cheetahReset(
  laneSeed: number,
  resetIndex = 0,
): { qpos: Float32Array; qvel: Float32Array } {
  const qpos = new Float32Array(QPOS_LEN);
  const qvel = new Float32Array(QPOS_LEN);

  const rawPos = philoxWords(laneSeed, STREAM_RESET, resetIndex, 12, 0);
  for (let i = 0; i < QPOS_LEN; i++) {
    qpos[i] = uniform(rawPos[i] ?? 0, -HC_RESET_NOISE, HC_RESET_NOISE);
  }

  const rawVel = philoxWords(laneSeed, STREAM_RESET, resetIndex, 12, 3);
  const z = new Float32Array(12);
  for (let p = 0; p < 6; p++) {
    const [a, b] = boxMuller(rawVel[2 * p] ?? 0, rawVel[2 * p + 1] ?? 0);
    z[2 * p] = a;
    z[2 * p + 1] = b;
  }
  for (let i = 0; i < QPOS_LEN; i++) qvel[i] = Math.fround(HC_RESET_NOISE * (z[i] ?? 0));

  return { qpos, qvel };
}


const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'cheetah' as const,
  cheetah: cheetahGeometry,
});

function makeState(pose: Float32Array, velX: number): CheetahState {
  return {
    qpos: pose,
    rootX: pose[0] ?? 0,
    rootZ: ROOT_Z0 + (pose[1] ?? 0),
    pitch: pose[2] ?? 0,
    velX,
  };
}

export const halfCheetahDecoder: EnvDecoder<CheetahState, CheetahStatic> = {
  envId: 'HalfCheetah-v4',

  initStatic(seed: number): CheetahStatic {
    const { qpos, qvel } = cheetahReset(seed, 0);
    return { seed, resetIndex: 0, initQpos: qpos, initQvel: qvel };
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(_seed: number, st: CheetahStatic): EnvFrame<CheetahState> {
    const pose = Float32Array.from(st.initQpos);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, st.initQvel[0] ?? 0),
    };
  },

  decodeFrame(
    i: number,
    raw16: Float32Array,
    _st: CheetahStatic,
    prev: EnvFrame<CheetahState>,
  ): EnvFrame<CheetahState> {
    const pose = raw16.subarray(7, 16);
    const dx = (pose[0] ?? 0) - (prev.pose[0] ?? 0);
    const velX = Math.abs(dx) > CHEETAH_TELEPORT_DX ? 0 : dx / CHEETAH_DT;
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, velX),
    };
  },

  deriveEvents(
    prev: EnvFrame<CheetahState>,
    next: EnvFrame<CheetahState>,
  ): SimEvent[] {
    const teleport = Math.abs((next.pose[0] ?? 0) - (prev.pose[0] ?? 0)) > CHEETAH_TELEPORT_DX;
    if (next.done === 0 && !teleport) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'done' });
    if (teleport) out.push({ type: 'teleport' });
    return out;
  },
};
