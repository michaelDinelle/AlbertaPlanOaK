import type { RunDetail, RunIndexEntry, RunMeta } from '../data/types';
import { isEnvId, type EnvId } from '../data/envs';

export function runEnvs(meta: Pick<RunMeta, 'envSchedule'>): EnvId[] {
  const out: EnvId[] = [];
  for (const entry of meta.envSchedule) {
    if (!isEnvId(entry.envId)) throw new Error(`run schedule names unknown env ${entry.envId}`);
    if (!out.includes(entry.envId)) out.push(entry.envId);
  }
  return out;
}

export interface EnvLanes {
  lanes: number[];
  seeds: number[];
}

export function envLanes(meta: Pick<RunMeta, 'envSchedule' | 'seeds'>, envId: string): EnvLanes {
  const laneOfSeed = new Map<number, number>();
  meta.seeds.forEach((seed, lane) => laneOfSeed.set(seed, lane));
  const lanes: number[] = [];
  const seeds: number[] = [];
  for (const entry of meta.envSchedule) {
    if (entry.envId !== envId) continue;
    for (let seed = entry.seedStart; seed <= entry.seedEnd; seed++) {
      const lane = laneOfSeed.get(seed);
      if (lane !== undefined) {
        lanes.push(lane);
        seeds.push(seed);
      }
    }
  }
  return { lanes, seeds };
}

export function sortRuns(entries: readonly RunIndexEntry[]): RunIndexEntry[] {
  return [...entries].sort((a, b) => {
    if (a.inFlight !== b.inFlight) return a.inFlight ? -1 : 1;
    if (a.role !== b.role) return a.role === 'best' ? -1 : 1;
    const delta = Date.parse(b.startedAt) - Date.parse(a.startedAt);
    return delta !== 0 ? delta : a.id.localeCompare(b.id);
  });
}

export function resolveFeaturedRun(entries: readonly RunIndexEntry[]): { runId: string | null } {
  const best = entries.find((entry) => entry.role === 'best');
  if (best !== undefined) return { runId: best.id };
  const oak = sortRuns(entries.filter((entry) => entry.policy === 'oak'));
  return { runId: (oak.find((entry) => entry.complete) ?? oak[0])?.id ?? null };
}

export type RunLabelSource = Pick<RunIndexEntry, 'id' | 'startedAt' | 'role' | 'policy'>;

export function runShortId(id: string): string {
  return id.slice(0, 7);
}

export function runLabel(entry: RunLabelSource): string {
  const parts = [runDateLabel(entry.startedAt)];
  if (entry.role === 'best') parts.push('best');
  if (entry.policy !== 'oak') parts.push(entry.policy);
  parts.push(runShortId(entry.id));
  return parts.join(' · ');
}

export function runLabelFor(
  id: string,
  entries: readonly RunIndexEntry[] | undefined,
  loaded: RunDetail | null,
): string {
  const entry = entries?.find((run) => run.id === id);
  if (entry !== undefined) return runLabel(entry);
  if (loaded !== null && loaded.id === id) return runLabel({ id, startedAt: loaded.meta.startedAt, role: null, policy: loaded.meta.policy });
  return runShortId(id);
}

export function runDateLabel(iso: string, now: Date = new Date()): string {
  const date = new Date(Date.parse(iso));
  const monthDay = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return date.getFullYear() === now.getFullYear() ? monthDay : `${monthDay} ${date.getFullYear()}`;
}
