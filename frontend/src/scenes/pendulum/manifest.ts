
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { PD_MAX_TORQUE, type PendulumState } from '@sim/envs/pendulum';

const THUMB_TH = 0.42;
const THUMB_THDOT = 1.12;
const THUMB_TORQUE = 0.55;

export function pendulumPose(th: number, thDot: number, torque: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = th;
  p[1] = thDot;
  p[2] = torque;
  return p;
}

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
    bloomStrength: 0.54,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
  budget: { drawCalls: 40, triangles: 110000 },
};

export default manifest;
