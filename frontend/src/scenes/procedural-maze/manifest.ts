
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { PROCMAZE_GOAL, PROCMAZE_START } from '@sim/envs/proceduralmaze';

function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'ProceduralMaze-cont-v0',
  slug: 'procedural-maze',
  title: 'The Overgrown Ruins',
  logline:
    'A noon stone ruin whose every wall is masonry the seed itself laid — mossy blocks, ' +
    'fallen lintels, ivy and broken corner towers — with a shaft of sunlight breaking ' +
    'through the ceiling onto whichever cell the goal has respawned to.',
  accent: 'procedural-maze',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    derived: {
      x: PROCMAZE_START.x,
      y: PROCMAZE_START.y,
      gx: PROCMAZE_GOAL.x,
      gy: PROCMAZE_GOAL.y,
    },
  },
  postConfig: {
    bloomStrength: 0.44,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
  budget: { drawCalls: 80, triangles: 200000 },
};

export default manifest;
