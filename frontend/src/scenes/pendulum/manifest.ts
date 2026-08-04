// The Lantern Bell — scene manifest (Pendulum-cont-v0).
//
// The thumbFrame is a real, physically-reachable pose: the lantern a little past the
// bottom of its arc with gentle rightward momentum and a light push. Its shaped cost is
// 0.30 out of a possible 16.27, so the pond is warm and the koi have risen — the scene at
// its most characteristic, with enough tilt that it reads as a pendulum rather than a
// hanging prop.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { PD_MAX_TORQUE, type PendulumState } from '@sim/envs/pendulum';

const THUMB_TH = 0.42;
const THUMB_THDOT = 1.12;
const THUMB_TORQUE = 0.55;

/** pose = [th, thdot, torque, 0…] — the record-v3 Pendulum packing. */
export function pendulumPose(th: number, thDot: number, torque: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = th;
  p[1] = thDot;
  p[2] = torque;
  return p;
}

/** The decoder's own PendulumState shape, so a fixture plays exactly like a record row. */
export function pendulumDerived(th: number, thDot: number, torque: number, action: number): PendulumState {
  return {
    th,
    thDot,
    torque,
    action,
    effort: Math.min(1, Math.abs(torque) / PD_MAX_TORQUE),
  };
}

export const manifest: SceneManifest = {
  envId: 'Pendulum-cont-v0',
  slug: 'pendulum',
  title: 'The Lantern Bell',
  logline:
    'A koi pond under a torii at night: the lantern hangs from the beam at the exact angle θ, ' +
    "and the shaped reward IS the light — the env's own cost θ² + 0.1·θ̇² + 0.001·u² drives the " +
    'flame, the pond glow and how high the koi rise, so a better-scored state is never darker.',
  accent: 'pendulum',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pendulumPose(THUMB_TH, THUMB_THDOT, THUMB_TORQUE),
    derived: pendulumDerived(THUMB_TH, THUMB_THDOT, THUMB_TORQUE, THUMB_TORQUE / PD_MAX_TORQUE),
  },
  postConfig: {
    // A night frame with one warm HDR emissive in the middle of it: the bloom is the
    // lantern's halo on the water, so it carries more strength than the dusk scenes.
    bloomStrength: 0.54,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
  // Counted on the built scene graph in its busiest state (lantern lit, full torque, curl
  // live) — every visible renderable plus one shadow-pass draw per caster, censused
  // headlessly so the figure is reproducible: hero 28 draws / 70.1k tris, medium 26 /
  // 24.0k, thumb 15 / 6.3k. Declared with headroom, far inside STYLE.md's 300 / 500k hero
  // and 40 / 50k thumb caps.
  budget: { drawCalls: 40, triangles: 110000 },
};

export default manifest;
