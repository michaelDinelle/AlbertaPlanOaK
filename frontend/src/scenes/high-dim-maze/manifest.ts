
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

const THUMB_X = 4;
const THUMB_Y = 9;
const THUMB_GX = 9;
const THUMB_GY = 4;
const THUMB_DIR = 3;

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
  budget: { drawCalls: 70, triangles: 150000 },
};

export default manifest;
