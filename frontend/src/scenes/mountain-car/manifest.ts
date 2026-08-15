
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { MC_GOAL_POS, mountainCarHeight } from '@sim/envs/mountaincar';

const THUMB_POS = -0.35;
const THUMB_VEL = 0.04;

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
};

export default manifest;
