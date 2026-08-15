
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import type { KeyCorridorState } from '@sim/envs/keycorridor';

const THUMB_X = 2;
const THUMB_Y = 4;
const THUMB_GX = 6;
const THUMB_GY = 2;
const THUMB_KX = 3;
const THUMB_KY = 2;

function pose(
  x: number,
  y: number,
  gx: number,
  gy: number,
  kx: number,
  ky: number,
  hasKey: boolean,
  doorOpen: boolean,
): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = kx;
  p[5] = ky;
  p[6] = hasKey ? 1 : 0;
  p[7] = doorOpen ? 1 : 0;
  return p;
}

const thumbDerived: KeyCorridorState = {
  x: THUMB_X,
  y: THUMB_Y,
  gx: THUMB_GX,
  gy: THUMB_GY,
  kx: THUMB_KX,
  ky: THUMB_KY,
  hasKey: false,
  doorOpen: false,
};

export const manifest: SceneManifest = {
  envId: 'KeyCorridor-cont-v0',
  slug: 'key-corridor',
  title: 'The Ivy Gate',
  logline:
    'A morning garden split by one ivy-clad wall: a wild moss side where an amber seed ' +
    'waits in its stone dish, a swept gravel court where a lantern shrine burns, and ' +
    'between them a single oak door that the ivy visibly unpicks itself from when the key ' +
    'turns — and sews shut again the moment the shrine is reached.',
  accent: 'key-corridor',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_Y, THUMB_GX, THUMB_GY, THUMB_KX, THUMB_KY, false, false),
    derived: thumbDerived,
  },
  postConfig: {
    bloomStrength: 0.3,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
};

export default manifest;
