// ROLES.json at the bench root names the runs the lab currently cares about.
// Values are run-root directory NAMES (not paths). Unknown/extra keys are ignored.
//
// v2 (rolesSchemaVersion 2) renamed `keeper` -> `bestKernel`. Both spellings are read
// for one migration cycle: a bench root that has not been rewritten yet still resolves
// its best kernel, and `bestKernel` wins when a half-migrated file carries both.
// `featured` is reserved for a showcase run and is simply null until one is named.

import { join } from 'node:path';
import type { RolesResponse } from '@shared/types';

/**
 * The roles a run can wear IN THE INDEX. `featured` is deliberately not one of them:
 * it is a presentation slot on the roles endpoint, not a lineage position, and nothing
 * claims it yet.
 */
export type RunRole = 'bestKernel' | 'candidate' | 'parent';
/** display/priority order — the best kernel is the run every table is compared against */
export const RUN_ROLES: readonly RunRole[] = ['bestKernel', 'candidate', 'parent'] as const;

/** legacy spellings accepted per role, tried in order after the v2 name */
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

/** Read ROLES.json. Missing file / malformed JSON both answer all-nulls (never throws). */
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

/** Invert a RolesResponse into runId -> role. A run named twice keeps the first role. */
export function roleMap(roles: RolesResponse): Map<string, RunRole> {
  const m = new Map<string, RunRole>();
  for (const role of RUN_ROLES) {
    const id = roles[role];
    if (id !== null && !m.has(id)) m.set(id, role);
  }
  return m;
}
