// The Long Night Road — scene manifest (DelayedChain-cont-v0).
//
// THE THUMB FRAME IS THE WALL VIEW, and the wall view of this env is the WHOLE road: the
// camera (index.ts, CAM_*_THUMB) stands 30 units off the corridor's own axis and holds all
// thirty-two cells, near end to cairn, with the aurora over the far end. So the pose here
// is not a close-up — it is a POSITION ON THAT ROAD. Cell 27 walking east (lastA = 1)
// puts the wisp's spark five cells short of the pay-off, high up the receding line of dark
// lanterns and under the curtain: one still that says how far along this stream is, which
// is the only question a wall of streams asks.
//
// (Superseded, kept as the lesson: the first framing cropped to the last four cells —
// handsome, and wrong, because a random-walking DelayedChain agent spends nearly all of
// its 100k steps outside that crop.)

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L } from '@sim/envs/delayedchain';

const THUMB_POS = 27;
const THUMB_LAST_A = 1;

/** pose = [pos, 31, lastA, 0…] — the record-v3 DelayedChain packing. */
export function delayedChainPose(pos: number, lastA: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = DELAYEDCHAIN_END;
  p[2] = lastA;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'DelayedChain-cont-v0',
  slug: 'delayed-chain',
  title: 'The Long Night Road',
  logline:
    'A thirty-two-cell moorland path under a night sky, walked by lantern light that is not ' +
    'lit yet: the aurora hangs over the far end alone, and when the wisp finally reaches it ' +
    'every lantern on the road ignites in a wave running back from the end to the start.',
  accent: 'delayed-chain',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: delayedChainPose(THUMB_POS, THUMB_LAST_A),
    derived: { pos: THUMB_POS, lastA: THUMB_LAST_A, len: DELAYEDCHAIN_L },
  },
  postConfig: {
    // A night palette is nearly all shadow, so the bloom has almost nothing to grab
    // except the lantern flames, the wisp and the aurora's ribbon cores — which is
    // exactly the list this scene wants glowing. It can afford to be generous.
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
  // Measured on Apple Silicon from the harness HUD, shadow pass included, at 1600×900
  // DPR 2 (headless Chrome, real GPU):
  //   hero    38 draws / 154.5k tris idle · 40 / 161.7k through an ignition wave · 40 with
  //           the ground-truth overlay on · 90–120 fps
  //   medium  37 draws /  61.2k tris · ~176 fps
  //   thumb   26 draws /  19.6k tris · 240 fps  (STYLE.md thumb cap: 40 / 50k)
  // Every tier is now ONE DRAW and one icosphere (80 tris at hero/medium, 20 at thumb)
  // UNDER those figures: the kit wisp halo is hidden rather than dimmed (see index.ts), and
  // a hidden mesh is a draw call that never issues. Nothing else in the repair pass adds
  // geometry. Declared with headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 70, triangles: 190000 },
};

export default manifest;
