import { acrobotDecoder } from '@sim/envs/acrobot';
import { walkDecoder } from '@sim/envs/walk';
import { deepseaDecoder } from '@sim/envs/deepsea';
import { hallDecoder } from '@sim/envs/hall';
import { jbwDecoder } from '@sim/envs/jbw';
import { mountainCarDecoder } from '@sim/envs/mountainCar';
import { pendulumDecoder } from '@sim/envs/pendulum';
import type { EnvDescriptor } from '../data/envs';
import type { EnvDecoder } from '@sim/types';
import type { SceneFactory } from '@kit/types';

export interface SceneRow {
  decoder: EnvDecoder;
  load: () => Promise<{ default: SceneFactory }>;
}

export const SCENES: Record<string, SceneRow | undefined> = {
  acrobot: { decoder: acrobotDecoder, load: () => import('./acrobot/index') },
  walk: { decoder: walkDecoder, load: () => import('./walk/index') },
  deepsea: { decoder: deepseaDecoder, load: () => import('./deepsea/index') },
  hall: { decoder: hallDecoder, load: () => import('./hall/index') },
  jbw: { decoder: jbwDecoder, load: () => import('./jbw/index') },
  mountainCar: { decoder: mountainCarDecoder, load: () => import('./mountainCar/index') },
  pendulum: { decoder: pendulumDecoder, load: () => import('./pendulum/index') },
};

export const sceneOf = (env: EnvDescriptor): SceneRow | null => SCENES[env.family] ?? null;
export const posterOf = (family: string): string => `/posters/${family}.webp`;
