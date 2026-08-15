
import { cheetahGeometry } from '../fk/cheetah';
import { STREAM_RESET, boxMuller, philoxWords, uniform } from '../philox';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const HC_RESET_NOISE = 0.1;
export const CHEETAH_DT = 0.05;
export const CHEETAH_TELEPORT_DX = 10;

export const QPOS_LEN = 9;

export const CHEETAH_ROOT_X = 0;
export const CHEETAH_ROOT_Z = 1;
export const CHEETAH_PITCH = 2;


export interface CheetahStatic {
  seed: number;
  resetIndex: number;
  initQpos: Float32Array;
  initQvel: Float32Array;
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

export const halfCheetahDecoder: EnvDecoder<CheetahStatic> = {
  envId: 'HalfCheetah-v4',

  initStatic(seed: number): CheetahStatic {
    const { qpos, qvel } = cheetahReset(seed, 0);
    return { seed, resetIndex: 0, initQpos: qpos, initQvel: qvel };
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(_seed: number, st: CheetahStatic): EnvFrame {
    const pose = Float32Array.from(st.initQpos);
    return {
      step: -1,
      done: 0,
      pose,
    };
  },


  deriveEvents(
    prev: EnvFrame,
    next: EnvFrame,
  ): SimEvent[] {
    const teleport = Math.abs((next.pose[0] ?? 0) - (prev.pose[0] ?? 0)) > CHEETAH_TELEPORT_DX;
    if (next.done === 0 && !teleport) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'done' });
    if (teleport) out.push({ type: 'teleport' });
    return out;
  },
};
