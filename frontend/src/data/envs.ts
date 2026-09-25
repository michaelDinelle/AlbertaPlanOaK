export interface ScentParameters {
  decay: number;
  diffusion: number;
  lifetime: number;
  values: number[][];
}

export interface EnvDescriptor {
  envId: string;
  category: string;
  instance: number;
  obs: number;
  actions: number;
  box: number;
  seedStart: number;
  seedEnd: number;
  words: number;
  view: number;
  parameters: Record<string, unknown>;
  constants: Record<string, CatalogValue>;
  grid?: { w: number; h: number; walls: number[] };
  goal?: [number, number];
  hazards?: [number, number][];
  fences?: [number, number, number, number][];
  scent?: ScentParameters;
}

export type CatalogRecord = Record<string, number | number[]>;
export type CatalogValue = number | number[] | CatalogRecord | CatalogRecord[];

export function parameter(env: EnvDescriptor, key: string): number {
  const value = env.parameters[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${env.envId} lacks parameter ${key}`);
  return value;
}

export function constant(env: EnvDescriptor, key: string): number {
  const value = env.constants[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${env.envId} lacks constant ${key}`);
  return value;
}

export function constantList(env: EnvDescriptor, key: string): number[] {
  const value = env.constants[key];
  if (!Array.isArray(value) || value.some(item => typeof item !== 'number')) throw new Error(`${env.envId} lacks constant list ${key}`);
  return value as number[];
}

export function constantRecord(env: EnvDescriptor, key: string): CatalogRecord {
  const value = env.constants[key];
  if (typeof value !== 'object' || Array.isArray(value)) throw new Error(`${env.envId} lacks constant record ${key}`);
  return value;
}

export function constantRecords(env: EnvDescriptor, key: string): CatalogRecord[] {
  const value = env.constants[key];
  if (!Array.isArray(value) || value.some(item => typeof item !== 'object')) throw new Error(`${env.envId} lacks constant records ${key}`);
  return value as CatalogRecord[];
}

export interface Environment {
  name: string;
  category: string;
  configurations: EnvDescriptor[];
}

export function recordedEnvironments(catalog: readonly EnvDescriptor[], ids: readonly string[]): Environment[] {
  const descriptors = new Map(catalog.map(env => [env.envId, env]));
  const groups = new Map<string, Environment>();
  for (const id of ids) {
    const env = descriptors.get(id);
    if (env === undefined) throw new Error(`Recording configuration ${id} is missing from the catalog`);
    const name = environmentName(env);
    let environment = groups.get(name);
    if (environment === undefined) {
      environment = { name, category: env.category, configurations: [] };
      groups.set(name, environment);
    }
    environment.configurations.push(env);
  }
  return [...groups.values()];
}

export function configurationName(env: EnvDescriptor): string {
  return env.envId.replace(/-cont-v0$/, '');
}

export function environmentName(env: EnvDescriptor): string {
  return configurationName(env).split('-')[0]!;
}
