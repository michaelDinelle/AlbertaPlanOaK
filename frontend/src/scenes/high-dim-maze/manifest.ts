// The Slot Canyons — scene manifest (HighDimMaze-cont-v0).
//
// The thumbFrame stands the wisp on (4, 9): one cell EAST of the low arch at (3, 9),
// facing east (lastDir = 3), having just come through it into the wide middle slot — with
// the oasis burning away up slot C at (9, 4). One still that carries the whole scene: the
// three canyons, an arch you can see the agent has just used, and the goal two turns away.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

const THUMB_X = 4;
const THUMB_Y = 9;
const THUMB_GX = 9;
const THUMB_GY = 4;
/** 3 = right: the direction the agent last moved, and the way a dash would go */
const THUMB_DIR = 3;

/** pose = [x, y, gx, gy, lastDir, 0…] — the record-v3 HighDimMaze packing. */
function pose(x: number, y: number, gx: number, gy: number, lastDir: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = lastDir;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'HighDimMaze-cont-v0',
  slug: 'high-dim-maze',
  title: 'The Slot Canyons',
  logline:
    'An afternoon in three terracotta slot canyons, stitched by two natural stone arches ' +
    'exactly where the maze puts its gaps — a scarf-tailed wisp reading the wind down the ' +
    'shaded corridors toward an oasis burning in its alcove.',
  accent: 'high-dim-maze',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_Y, THUMB_GX, THUMB_GY, THUMB_DIR),
    derived: {
      x: THUMB_X,
      y: THUMB_Y,
      gx: THUMB_GX,
      gy: THUMB_GY,
      lastDir: THUMB_DIR,
    },
  },
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.afternoon.fog,
  },
  // Measured headlessly off the live scene graph (every visible Mesh/Points/Line, with
  // instance counts, plus the shadow-casting subset the depth pass re-issues):
  //   hero    37 draws + 9 shadow casters /  96k tris  (up to +8 ripple meshes and +5
  //           overlay lines while a celebration or the ground truth is on screen)
  //   medium  34 draws + 9 shadow casters /  28k tris
  //   thumb   24 draws + 0 shadow casters /  10k tris   (STYLE.md thumb cap: 40 / 50k)
  // Declared with headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 70, triangles: 150000 },
};

export default manifest;
