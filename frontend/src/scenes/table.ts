import { fourRoomsDecoder } from '@sim/envs/fourrooms';
import { deepSeaDecoder } from '@sim/envs/deepsea';
import { acrobotDecoder } from '@sim/envs/acrobot';
import { mountainCarDecoder } from '@sim/envs/mountaincar';
import { pendulumDecoder } from '@sim/envs/pendulum';
import { chainDecoder } from '@sim/envs/chain';
import { tMazeDecoder } from '@sim/envs/tmaze';
import { jellyBeanWorldDecoder } from '@sim/envs/jellybeanworld';
import type { EnvDescriptor } from '../data/envs';
import type { EnvDecoder } from '@sim/types';
import type { SceneFactory } from '@kit/types';
import type { Palette, PaletteName } from '@kit/tokens';

export type PaletteToken = readonly [PaletteName, keyof Palette];

export interface FogRecipe {
  from: PaletteToken;
  blend?: readonly (readonly [PaletteToken, number])[];
}

export interface SceneRow {
  slug: string;
  poster: string;
  bloom: number;
  burst?: number;
  fog: FogRecipe;
  decoder: EnvDecoder;
  controlHz: number;
  discretePose: boolean;
  wrapMask: number;
  load: () => Promise<{ default: SceneFactory }>;
}

export const SCENES: Record<string, SceneRow | undefined> = {
  hall: { decoder: fourRoomsDecoder, controlHz: 8, discretePose: true, wrapMask: 0, slug: 'four-rooms', poster: '/posters/four-rooms.webp', bloom: 0.45, burst: 110, fog: { from: ['noon', 'fog'] }, load: () => import('./four-rooms/index') },
  deepsea: { decoder: deepSeaDecoder, controlHz: 8, discretePose: true, wrapMask: 0, slug: 'deep-sea', poster: '/posters/deep-sea.webp', bloom: 0.52, burst: 96, fog: { from: ['underwater', 'fog'] }, load: () => import('./deep-sea/index') },
  acrobot: { decoder: acrobotDecoder, controlHz: 5, discretePose: false, wrapMask: 3, slug: 'acrobot', poster: '/posters/acrobot.webp', bloom: 0.46, burst: 110, fog: { from: ['dawn', 'fog'] }, load: () => import('./acrobot/index') },
  mountainCar: { decoder: mountainCarDecoder, controlHz: 30, discretePose: false, wrapMask: 0, slug: 'mountain-car', poster: '/posters/mountain-car.webp', bloom: 0.44, burst: 84, fog: { from: ['dusk', 'fog'] }, load: () => import('./mountain-car/index') },
  pendulum: { decoder: pendulumDecoder, controlHz: 20, discretePose: false, wrapMask: 1, slug: 'pendulum', poster: '/posters/pendulum.webp', bloom: 0.54, fog: { from: ['night', 'fog'] }, load: () => import('./pendulum/index') },
  walk: { decoder: chainDecoder, controlHz: 8, discretePose: true, wrapMask: 0, slug: 'chain', poster: '/posters/chain.webp', bloom: 0.34, burst: 120, fog: { from: ['afternoon', 'fog'] }, load: () => import('./chain/index') },
  tmaze: { decoder: tMazeDecoder, controlHz: 8, discretePose: true, wrapMask: 0, slug: 't-maze', poster: '/posters/t-maze.svg', bloom: 0.42, burst: 100, fog: { from: ['afternoon', 'fog'] }, load: () => import('./t-maze/index') },
  jbw: { decoder: jellyBeanWorldDecoder, controlHz: 8, discretePose: true, wrapMask: 0, slug: 'jelly-bean-world', poster: '/posters/jelly-bean-world.webp', bloom: 0.38, burst: 108, fog: { from: ['afternoon', 'fog'], blend: [[['dusk', 'fog'], 0.45]] }, load: () => import('./jelly-bean-world/index') },
};

export const sceneOf = (env: EnvDescriptor): SceneRow | null => SCENES[env.family] ?? null;
