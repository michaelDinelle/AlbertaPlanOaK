// ROLES.json at the bench root names the three runs the lab currently cares about.
// Values are run-root directory NAMES (not paths). Unknown/extra keys are ignored.

import { join } from 'node:path';
import type { RolesResponse } from '@shared/types';

export type RunRole = 'candidate' | 'parent' | 'keeper';
export const RUN_ROLES: readonly RunRole[] = ['candidate', 'parent', 'keeper'] as const;

const EMPTY: RolesResponse = { candidate: null, parent: null, keeper: null };

function pick(obj: Record<string, unknown>, key: RunRole): string | null {
  const v = obj[key];
  return typeof v === 'string' && v.length > 0 ? v : null;
}

/** Read ROLES.json. Missing file / malformed JSON both answer all-nulls (never throws). */
export async function readRoles(benchRoot: string): Promise<RolesResponse> {
  try {
    const raw = await Bun.file(join(benchRoot, 'ROLES.json')).text();
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return { ...EMPTY };
    const obj = parsed as Record<string, unknown>;
    return {
      candidate: pick(obj, 'candidate'),
      parent: pick(obj, 'parent'),
      keeper: pick(obj, 'keeper'),
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
