// The Overgrown Ruins — scene manifest (ProceduralMaze-cont-v0).
//
// The thumbFrame is the env's own deterministic init, and that is not laziness: this is
// the ONE env whose layout is regenerated per lane, so the only two cells legal on every
// possible maze are the endpoints the generator carves — (1,1) and (9,9). A thumbnail
// that picked a pretty mid-maze cell would be standing inside a wall on some other seed.
// The wisp enters at the far corner, the sunbeam falls on the near one and leans north-
// west across the whole ruin: the diagonal reads the maze, the beam and the scale of the
// masonry in one still, on every seed the gallery could ever be built from.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { PROCMAZE_GOAL, PROCMAZE_START } from '@sim/envs/proceduralmaze';

/** pose = [x, y, gx, gy, 0…] — the record-v3 ProceduralMaze packing. */
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
  // Measured on Apple Silicon from the harness HUD, shadow pass included, at 1600×900:
  //   seed 96 (69 wall cells, 5 doorways)   hero 53 draws / 118.5k tris · ~215 fps
  //                                         medium 53 draws /  37.1k tris · ~240 fps
  //                                         thumb  28 draws /  10.9k tris · ~240 fps
  //   seed 97 (69 wall cells, 3 doorways)   hero 49 draws / 122.6k tris · ~225 fps
  //   with the ground-truth overlay on: +1 draw.
  // RE-MEASURED off the same HUD after the framing / masonry / ivy repairs, seed 96:
  //   hero 53 draws / 119.4k tris · ~127 fps   (with the overlay on: 58 draws)
  //   medium 53 draws / 37.2k tris · ~212 fps
  //   thumb  28 draws / 10.9k tris · ~240 fps
  // The repairs moved stone and pixels, never object count — pulling the camera back to
  // 19.0 cannot add a draw, because every scatter in this ruin is a single InstancedMesh.
  // The declared budget carries headroom for the densest maze this env can generate: the
  // courses are INSTANCED, so extra wall cells cost triangles and not draws, and the only
  // per-seed draw count is the doorways (capped at 9 by `archBudget`). Thumb measures 28
  // draws / 10.9k tris against STYLE.md's 40 / 50k thumb cap.
  budget: { drawCalls: 80, triangles: 200000 },
};

export default manifest;
