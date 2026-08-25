import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { blendTokens } from '@kit/mix';
import { TOKENS } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';

export const SWIM_Y = 0;
export const FLOOR_Y = -2.4;

export interface WorldHandle extends WorldCore {
  setOrigin(x: number, z: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, {
    paletteName: 'underwater',
    name: 'swimLagoon',
    extent: 72,
    relief: 0.5,
    groundLift: FLOOR_Y,
    shadowRadius: 4.6,
  });
  const accent = TOKENS.accents['swimmer'] ?? TOKENS.celebrationGold;

  const plankton: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'fireflies',
    count: 340,
    bounds: { x: 0, y: SWIM_Y - 0.6, z: 0, halfX: 9, halfY: 1.6, halfZ: 9 },
    seed: ctx.seed ^ 0x33c1,
    color: blendTokens(accent, g.palette.hemiSky, 0.4),
    size: 0.3,
    intensity: 1.5,
    opacity: 0.6,
    swirl: 0.22,
    name: 'lagoonPlankton',
  });
  g.w.add(plankton.points);
  g.w.size(g.w.time(g.w.own(plankton)));

  return g.w.seal<WorldHandle>({
    setOrigin(x, z) {
      g.setOrigin(x, z);
      plankton.points.position.set(x, 0, z);
    },
  });
}
