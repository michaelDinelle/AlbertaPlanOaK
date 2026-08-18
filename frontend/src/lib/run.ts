import type { RunDetail, RunIndexEntry, RunMeta } from '@shared/types';
import { ENV_IDS, isEnvId, type EnvId } from '@shared/envs';

export interface EnvShare {
  envId: EnvId;
  lanes: number;
  fraction: number;
}

export function envShares(meta: Pick<RunMeta, 'envSchedule'>): EnvShare[] {
  const counts = new Map<EnvId, number>();
  for (const entry of meta.envSchedule) {
    if (!isEnvId(entry.envId)) continue;
    const lanes = Math.max(0, entry.seedEnd - entry.seedStart + 1);
    counts.set(entry.envId, (counts.get(entry.envId) ?? 0) + lanes);
  }
  const total = [...counts.values()].reduce((sum, n) => sum + n, 0) || 1;
  return ENV_IDS.filter((id) => counts.has(id)).map((envId) => {
    const lanes = counts.get(envId) ?? 0;
    return { envId, lanes, fraction: lanes / total };
  });
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

export function runEnvs(
  meta: Pick<RunMeta, 'envSchedule'>,
  recordings: readonly { envId: string }[],
): EnvId[] {
  const seen = new Set<EnvId>();
  for (const share of envShares(meta)) seen.add(share.envId);
  for (const rec of recordings) if (isEnvId(rec.envId)) seen.add(rec.envId);
  return ENV_IDS.filter((id) => seen.has(id));
}

export function sortRuns(entries: readonly RunIndexEntry[]): RunIndexEntry[] {
  return [...entries].sort((a, b) => {
    if (a.inFlight !== b.inFlight) return a.inFlight ? -1 : 1;
    if (a.role !== b.role) return a.role === 'keeper' ? -1 : 1;
    const ta = a.startedAt ? Date.parse(a.startedAt) : 0;
    const tb = b.startedAt ? Date.parse(b.startedAt) : 0;
    if (tb !== ta) return tb - ta;
    return a.id.localeCompare(b.id);
  });
}

export type FeaturedSource = 'param' | 'keeper' | 'newest' | null;

export interface FeaturedPick {
  runId: string | null;
  source: FeaturedSource;
}

export function resolveFeaturedRun(
  entries: readonly RunIndexEntry[],
  requested?: string,
): FeaturedPick {
  const byId = new Map<string, RunIndexEntry>();
  for (const entry of entries) byId.set(entry.id, entry);

  if (requested !== undefined && byId.has(requested)) return { runId: requested, source: 'param' };

  const keeper = entries.find((entry) => entry.role === 'keeper');
  if (keeper?.hasRecordings === true) return { runId: keeper.id, source: 'keeper' };

  const started = (entry: RunIndexEntry): number => (entry.startedAt ? Date.parse(entry.startedAt) : 0);
  const newest = (list: readonly RunIndexEntry[]): RunIndexEntry | null =>
    [...list].sort((a, b) => {
      const ta = started(a);
      const tb = started(b);
      if (Number.isFinite(tb) && Number.isFinite(ta) && tb !== ta) return tb - ta;
      return a.id.localeCompare(b.id);
    })[0] ?? null;

  const playable = entries.filter((entry) => entry.policy === 'oak' && entry.hasRecordings);
  const complete = newest(playable.filter((entry) => entry.complete));
  if (complete) return { runId: complete.id, source: 'newest' };
  const any = newest(playable);
  if (any) return { runId: any.id, source: 'newest' };
  return { runId: null, source: null };
}

export type RunLabelSource = Pick<RunIndexEntry, 'id' | 'startedAt' | 'role' | 'ablationLabel' | 'policy'>;

export function runShortId(id: string): string {
  return id.slice(id.lastIndexOf('-') + 1).slice(0, 7);
}

export function runLabelSource(
  id: string,
  meta: Pick<RunMeta, 'startedAt' | 'ablationLabel' | 'policy'>,
): RunLabelSource {
  return {
    id,
    startedAt: meta.startedAt ?? null,
    role: null,
    ablationLabel: meta.ablationLabel,
    policy: meta.policy,
  };
}

export function runLabel(entry: RunLabelSource): string {
  const parts = [runDateLabel(entry.startedAt)];
  if (entry.role === 'keeper') parts.push('keeper');
  if (entry.policy !== 'oak') parts.push(entry.policy);
  if (entry.ablationLabel !== 'none') parts.push(entry.ablationLabel);
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
  if (loaded !== null && loaded.id === id) return runLabel(runLabelSource(loaded.id, loaded.meta));
  return runShortId(id);
}

export function runDateLabel(iso: string | null, now: Date = new Date()): string {
  if (!iso) return 'undated';
  const at = Date.parse(iso);
  if (!Number.isFinite(at)) return 'undated';
  const date = new Date(at);
  const monthDay = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return date.getFullYear() === now.getFullYear() ? monthDay : `${monthDay} ${date.getFullYear()}`;
}
