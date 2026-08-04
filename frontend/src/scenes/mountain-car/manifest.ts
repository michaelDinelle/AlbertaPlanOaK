// Alpine Valley — scene manifest (MountainCar-cont-v0).
//
// The thumbFrame is a real, physically-reachable pose: mid-slope on the right flank with
// genuine rightward momentum (pos -0.35, vel +0.04 of a 0.07 max). It reads as a climb,
// not a parked cart, and it is the pose the gallery thumbnail renders.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { MC_GOAL_POS, mountainCarHeight } from '@sim/envs/mountaincar';

const THUMB_POS = -0.35;
const THUMB_VEL = 0.04;

/** pose = [pos, vel, goalMarker, 0…] — the record-v3 MountainCar packing. */
function pose(pos: number, vel: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = vel;
  p[2] = MC_GOAL_POS;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'MountainCar-cont-v0',
  slug: 'mountain-car',
  title: 'Alpine Valley',
  logline:
    'A dusk alpine basin whose ground IS sin(3x)·0.45+0.55: a wooden cart rocks the valley for ' +
    'momentum while the prayer-flag cairn waits on the summit at x = 0.5.',
  accent: 'mountain-car',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_POS, THUMB_VEL),
    derived: {
      pos: THUMB_POS,
      vel: THUMB_VEL,
      goal: MC_GOAL_POS,
      height: mountainCarHeight(THUMB_POS),
      action: 2,
      force: 1,
    },
  },
  postConfig: {
    bloomStrength: 0.44,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.dusk.fog,
  },
  // Measured at hero on Apple Silicon (harness HUD, shadow pass included): 36 draws /
  // 173k tris idle, 41 draws at the peak of a celebration with the ground-truth overlay
  // on. Declared with a little headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 60, triangles: 200000 },
};

export default manifest;
