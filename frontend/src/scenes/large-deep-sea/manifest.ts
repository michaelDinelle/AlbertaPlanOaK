// The Abyss — scene manifest (LargeDeepSea-cont-v0).
//
// The thumbFrame puts the wisp at (r = 11, c = 6): deep enough that the shelf under it has
// gone violet-black and the dive bubble is burning at nearly full gain, high enough that
// the sunlit top rows still hold the far edge of the card. The sunken wreck at (15, 15)
// glows in the corner it will never be reached in — which is the whole story of this lane.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { LARGE_DEEPSEA_N } from '@sim/envs/largedeepsea';

const THUMB_R = 11;
const THUMB_C = 6;

/** pose = [r, c, 15, 15, 0…] — the record-v3 LargeDeepSea packing. */
export function abyssPose(r: number, c: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = r;
  p[1] = c;
  p[2] = LARGE_DEEPSEA_N - 1;
  p[3] = LARGE_DEEPSEA_N - 1;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'LargeDeepSea-cont-v0',
  slug: 'large-deep-sea',
  title: 'The Abyss',
  logline:
    'The dive past the reef: sixteen terraced shelves stepping down into a violet-black trench, ' +
    'lit only by anglerfish lures and drifting glow-jellies, with a sunken wreck burning through ' +
    'its ribs at the one corner the agent has never reached.',
  accent: 'large-deep-sea',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: abyssPose(THUMB_R, THUMB_C),
    derived: { r: THUMB_R, c: THUMB_C, lastA: 1 },
  },
  postConfig: {
    // An abyss is black except where something is alive, so the bloom carries the scene —
    // but it is the one knob that turns a single over-threshold pixel into a 300 px glare,
    // and this scene stacks four additive emitters (wisp core, wisp halo, bubble shell rim,
    // bubble core) on top of shelf rows that already run at ~11 albedo. 0.62 blew the
    // shallow end out every fifteen steps; 0.48 keeps the biolume glowing and the terraces
    // readable underneath it.
    bloomStrength: 0.48,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.abyss.fog,
  },
  // Measured on Apple Silicon from the harness HUD at 1600×900 DPR 2, shadow pass
  // included (read over CDP: the --virtual-time-budget screenshot path never advances
  // performance.now() far enough for the harness's 0.25 s stats timer to fire, so those
  // shots always report an empty HUD):
  //   hero    32 draws /  67.1k tris · 47 total · 36 draws with the ground-truth overlay
  //   medium  30 draws /  29.8k tris · 45 total
  //   thumb   19 draws /  12.1k tris · 19 total   (STYLE.md thumb cap: 40 / 50k)
  // Declared with ~1.7× headroom, far inside STYLE.md's 300 / 500k hero cap. The scene is
  // cheap because everything repeated is instanced: one draw for 136 terraces, one for 136
  // coral whips, one for 46 jellies.
  //
  // Re-measured after the `diveSilt` puff landed (the draw that renders the env's
  // per-right-move cost): hero 31 → 32 and medium 29 → 30, thumb UNCHANGED at 19 because
  // the kit builds an empty, undrawn particle field at thumb tier. It is a THREE.Points
  // field either way, so it costs one draw call and zero triangles.
  // Frame rate is deliberately not quoted here any more: the re-measurement ran on a
  // machine building fourteen scenes at once (hero read 50–55 fps under that load, medium
  // 139, thumb 239), so the number would say more about the load than about the scene.
  budget: { drawCalls: 55, triangles: 110000 },
};

export default manifest;
