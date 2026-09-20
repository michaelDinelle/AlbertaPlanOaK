export interface ScentParameters {
  decay: number;
  diffusion: number;
  lifetime: number;
  values: number[][];
}

export interface EnvDescriptor {
  envId: string;
  family: string;
  instance: number;
  obs: number;
  actions: number;
  box: number;
  seedStart: number;
  seedEnd: number;
  pose: number;
  replay: boolean;
  parameters: Record<string, unknown>;
  grid?: { w: number; h: number; walls: number[] };
  scent?: ScentParameters;
}

export function parameter(env: EnvDescriptor, key: string): number {
  const value = env.parameters[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${env.envId} lacks parameter ${key}`);
  return value;
}

export function configurationLabel(env: EnvDescriptor, family: readonly EnvDescriptor[]): string {
  if (env.family === 'blocking') {
    const phases = env.parameters.phase as NonNullable<EnvDescriptor['grid']>[];
    if (!Array.isArray(phases) || phases.length !== 2 || phases.some(grid => !grid || !Array.isArray(grid.walls) || grid.walls.length !== grid.w * grid.h)) throw new Error('BlockingMaze layout parameters are missing');
    const [first, second] = phases as [NonNullable<EnvDescriptor['grid']>, NonNullable<EnvDescriptor['grid']>];
    if (first.w !== second.w || first.h !== second.h) throw new Error('BlockingMaze layout dimensions differ');
    return `${first.w} × ${first.h} · ${second.walls.some((wall, i) => wall && !first.walls[i]) ? 'barrier shifts' : 'shortcut opens'}`;
  }
  const keys = Object.keys(env.parameters).filter(key => typeof env.parameters[key] === 'number' && !(env.family === 'deepsea' && key === 'mapWords' || env.family === 'jbw' && ['cells', 'slot'].includes(key)) && family.some(other => other.parameters[key] !== env.parameters[key]));
  return keys.map(key => key === 'extent' ? `size ${parameter(env, key) + 1}` : key === 'cueBits' ? `choices ${2 ** parameter(env, key)}` : key === 'positionOnly' ? parameter(env, key) ? 'positions only' : 'full observations' : `${key.replace(/[A-Z]/g, letter => ` ${letter.toLowerCase()}`)} ${Number(parameter(env, key).toPrecision(6))}`).join(' · ');
}

export interface EnvFamily {
  id: string;
  name: string;
  configurations: EnvDescriptor[];
}

export function recordedFamilies(catalog: readonly EnvDescriptor[], ids: readonly string[]): EnvFamily[] {
  const descriptors = new Map(catalog.map(env => [env.envId, env]));
  const groups = new Map<string, EnvFamily>();
  for (const id of ids) {
    const env = descriptors.get(id);
    if (env === undefined) throw new Error(`Recording configuration ${id} is missing from the catalog`);
    let family = groups.get(env.family);
    if (family === undefined) {
      const name = { deepsea: 'DeepSea', tmaze: 'TMaze', hall: 'FourRoomsHall', walk: 'Chain', jbw: 'JellyBeanWorld', blocking: 'BlockingMaze' }[env.family] ?? env.envId.replace(/-cont-v0$/, '').split('-')[0]!;
      family = { id: env.family, name, configurations: [] };
      groups.set(env.family, family);
    }
    family.configurations.push(env);
  }
  return [...groups.values()];
}

export function configurationName(env: EnvDescriptor, family: EnvFamily): string {
  const label = configurationLabel(env, family.configurations);
  return label ? `${family.name} · ${label}` : family.name;
}
