// Four Seasons Courtyard — scene manifest (FourRooms-cont-v0).
//
// The thumbFrame stands the wisp on (4, 4) — the spring room's inner corner, one cell
// diagonally off the sundial — with the lantern-flower on (7, 7) in the winter room. The
// diagonal runs through the courtyard's heart, so the gallery card shows the sundial, two
// moon gates, and blossom against frost: the whole idea of the scene in one still.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { fourRoomsRoom } from '@sim/envs/fourrooms';

const THUMB_X = 4;
const THUMB_Y = 4;
const THUMB_GX = 7;
const THUMB_GY = 7;

/** pose = [x, y, gx, gy, 0…] — the record-v3 FourRooms packing. */
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
  // Measured on Apple Silicon from the harness HUD, shadow pass included:
  //   hero    46 draws / 154.5k tris idle · 47 through a celebration · 50 with the
  //           ground-truth overlay on · ~170–180 fps at 1600×900 DPR 2
  //   medium  45 draws /  38.6k tris · ~205 fps
  //   thumb   26 draws /   9.5k tris · ~240 fps  (STYLE.md thumb cap: 40 / 50k)
  // Declared with headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 70, triangles: 200000 },
};

export default manifest;
