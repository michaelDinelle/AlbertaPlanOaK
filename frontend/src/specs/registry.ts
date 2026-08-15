import type { EnvCatalog } from '@sim/catalog';
import type { SceneFactory } from '@kit/types';
import { makeScene } from '../engine/scene';
import type { EnvSpec } from '../engine/spec';

export const SPECS: Record<string, () => Promise<{ default: EnvSpec }>> = {};

export async function specFactory(envId: string, catalog: EnvCatalog): Promise<SceneFactory | null> {
  const load = SPECS[envId];
  if (load === undefined) return null;
  const spec = (await load()).default;
  return () => makeScene(spec, catalog);
}
