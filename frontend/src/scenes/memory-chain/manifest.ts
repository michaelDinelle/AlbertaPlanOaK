
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { MEMORYCHAIN_END, MEMORYCHAIN_L, type MemoryChainState } from '@sim/envs/memorychain';

const THUMB_POS = 0;
const THUMB_CUE = 1;

export function memoryChainPose(pos: number, cue: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = MEMORYCHAIN_END;
  p[2] = cue;
  return p;
}

export function memoryChainDerived(pos: number, cue: number, lastA = 0): MemoryChainState {
  return { pos, cue, lastA, len: MEMORYCHAIN_L };
}

export const manifest: SceneManifest = {
  envId: 'MemoryChain-cont-v0',
  slug: 'memory-chain',
  title: 'The Firefly Cave',
  logline:
    'A thirty-two-stone passage over still black water, lined with fireflies, where a ' +
    'crystal burns cyan or magenta only while the wisp stands beside it — and thirty-one ' +
    'stones later two dim mouths in the dark ask which colour it was.',
  accent: 'memory-chain',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: memoryChainPose(THUMB_POS, THUMB_CUE),
    derived: memoryChainDerived(THUMB_POS, THUMB_CUE),
  },
  postConfig: {
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.cave.fog,
  },
  budget: { drawCalls: 45, triangles: 75000 },
};

export default manifest;
