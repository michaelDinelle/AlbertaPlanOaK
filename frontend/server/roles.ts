
import { join } from 'node:path';
import type { RolesResponse } from '@shared/types';

export type RunRole = 'bestKernel' | 'candidate' | 'parent';
export const RUN_ROLES: readonly RunRole[] = ['bestKernel', 'candidate', 'parent'] as const;

const ALIASES: Record<RunRole | 'featured', readonly string[]> = {
  bestKernel: ['keeper'],
  candidate: [],
  parent: [],
  featured: [],
};

const EMPTY: RolesResponse = { bestKernel: null, candidate: null, parent: null, featured: null };

function pick(obj: Record<string, unknown>, key: RunRole | 'featured'): string | null {
  for (const name of [key, ...ALIASES[key]]) {
    const v = obj[name];
    if (typeof v === 'string' && v.length > 0) return v;
  }
  return null;
}

export async function readRoles(benchRoot: string): Promise<RolesResponse> {
  try {
    const raw = await Bun.file(join(benchRoot, 'ROLES.json')).text();
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return { ...EMPTY };
    const obj = parsed as Record<string, unknown>;
    return {
      bestKernel: pick(obj, 'bestKernel'),
      candidate: pick(obj, 'candidate'),
      parent: pick(obj, 'parent'),
      featured: pick(obj, 'featured'),
    };
  } catch {
    return { ...EMPTY };
  }
}

export function roleMap(roles: RolesResponse): Map<string, RunRole> {
  const m = new Map<string, RunRole>();
  for (const role of RUN_ROLES) {
    const id = roles[role];
    if (id !== null && !m.has(id)) m.set(id, role);
  }
  return m;
}
