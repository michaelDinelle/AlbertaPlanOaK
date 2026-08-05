
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { fourRoomsRoom } from '@sim/envs/fourrooms';

const THUMB_X = 4;
const THUMB_Y = 4;
const THUMB_GX = 7;
const THUMB_GY = 7;

function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'FourRooms-cont-v0',
  slug: 'four-rooms',
  title: 'Four Seasons Courtyard',
  logline:
    'A noon walled garden where the four rooms are literally four seasons — blossom, lawn, ' +
    'leaf-fall and frost — joined by stone moon-gates around a sundial, with a lantern-flower ' +
    'burning on whichever cell the goal has respawned to.',
  accent: 'four-rooms',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_Y, THUMB_GX, THUMB_GY),
    derived: {
      x: THUMB_X,
      y: THUMB_Y,
      gx: THUMB_GX,
      gy: THUMB_GY,
      room: fourRoomsRoom(THUMB_X, THUMB_Y),
    },
  },
  postConfig: {
    bloomStrength: 0.45,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
  budget: { drawCalls: 70, triangles: 200000 },
};

export default manifest;
