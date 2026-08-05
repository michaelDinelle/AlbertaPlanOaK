// Run discovery + the index cache.
//
// A run root is any directory ONE level under the bench root whose meta.json carries
// `fleetSchemaVersion`. Per-lane recording meta.json files live two levels deeper and
// lack that key, so they can never be mistaken for a run root.
//
// Completeness is measured, never trusted: `.done` is written by the launcher and lies
// about interrupted runs, so we read the LAST 256 KB of fleet.jsonl instead (rows are
// step-ordered and the kind:"final" block is the last thing written). That window also
// yields maxStep.
//
// The expensive half (tail scan + recording scan + summary probe) is memoized on
// (meta.json mtime, fleet.jsonl mtime+size, summary.json mtime, recordings signature) and
// persisted to frontend/.cache; the cheap half (inFlight, role) is recomputed on every
// request. Every input the derived record REPORTS is in that key — omitting summary.json
// or the recordings tree makes hasSummary/hasRecordings/recordings permanently stale for
// a run whose fleet.jsonl stopped changing (a summary.json written after the run ends is
// the normal case).
//
// A single unreadable root (EACCES, a fleet.jsonl that is a directory, a run deleted
// mid-scan) is SKIPPED, never fatal: /api/runs must keep answering for the other 600.

import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import type {
  AggregateRow,
  DeathRow,
  RunDetail,
  RunIndexEntry,
  RunMeta,
  RunSummary,
} from '@shared/types';
import { FLEET_TAIL_BYTES, INDEX_TTL_MS, IN_FLIGHT_MS } from './config';
import { parseJsonl, parseJsonlDetailed, readTail, statOrNull } from './jsonl';
import { roleMap, readRoles, type RunRole } from './roles';
import { adaptSummary } from './summaryAdapter';
import {
  recordingsSignature,
  scanRecordings,
  toRecordingRefs,
  type RecordingEntry,
} from './recordings';

/** floor between run-index cache writes (the file is ~240 KB at 600 runs) */
const PERSIST_MIN_INTERVAL_MS = 10_000;

// ---------- derived-per-root cache record ----------

interface CachedRun {
  /**
   * `${metaMtime}:${fleetMtime}:${fleetSize}:${summaryMtime}:${recordingsSignature}` —
   * every input of the derived record below, so no field can go stale.
   */
  key: string;
  id: string;
  envIds: string[];
  batch: number;
  steps: number;
  policies: string[];
  ablationLabel: string;
  startedAt: string | null;
  complete: boolean;
  maxStep: number;
  hasSummary: boolean;
  hasRecordings: boolean;
}

// `v` bumps whenever DISCOVERY SEMANTICS change (not just file shapes): a derived
// record is memoized on file mtimes alone, so a code change that alters what a scan
// would find must invalidate persisted records — the files it mis-scanned never
// change again. v1 -> v2: RECORD_VERSION_SERVABLE gained 4; v1 caches persisted
// hasRecordings:false for v4-only roots forever.
interface CacheFile {
  v: 2;
  benchRoot: string;
  runs: Record<string, CachedRun>;
}

export interface ResolvedRun {
  id: string;
  dir: string;
  meta: RunMeta;
  entry: RunIndexEntry;
}

// ---------- meta.json ----------

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

async function readRunMeta(path: string): Promise<RunMeta | null> {
  try {
    const parsed: unknown = JSON.parse(await Bun.file(path).text());
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    const o = parsed as Record<string, unknown>;
    // The one discriminator that separates a run root from a per-lane recording dir.
    if (typeof o['fleetSchemaVersion'] !== 'number') return null;
    return parsed as RunMeta;
  } catch {
    return null;
  }
}

function envIdsOf(meta: RunMeta): string[] {
  const sched = meta.envSchedule;
  if (Array.isArray(sched) && sched.length > 0) {
    const seen: string[] = [];
    for (const s of sched) {
      if (s !== null && typeof s === 'object' && typeof s.envId === 'string' && !seen.includes(s.envId)) {
        seen.push(s.envId);
      }
    }
    if (seen.length > 0) return seen;
  }
  return typeof meta.envId === 'string' && meta.envId.length > 0 ? [meta.envId] : [];
}

function ablationLabelOf(meta: RunMeta): string {
  if (typeof meta.ablationLabel === 'string' && meta.ablationLabel.length > 0) return meta.ablationLabel;
  const abl = asStringArray(meta.ablation);
  return abl.length > 0 ? abl.join(',') : 'none';
}

// ---------- completeness (tail probe) ----------

export interface FleetTailProbe {
  hasFinal: boolean;
  maxStep: number;
  rowsScanned: number;
}

/**
 * Scan the tail window of fleet.jsonl for the completeness signals. Reading a bounded
 * window is what makes indexing ~600 runs (up to 175 MB each) a sub-second operation.
 */
export async function probeFleetTail(
  fleetPath: string,
  bytes = FLEET_TAIL_BYTES,
): Promise<FleetTailProbe> {
  const tail = await readTail(fleetPath, bytes);
  if (tail === null) return { hasFinal: false, maxStep: 0, rowsScanned: 0 };
  const { rows } = parseJsonlDetailed<{ kind?: unknown; step?: unknown }>(tail.text);
  let hasFinal = false;
  let maxStep = 0;
  for (const r of rows) {
    if (r.kind === 'final') hasFinal = true;
    if (typeof r.step === 'number' && Number.isFinite(r.step) && r.step > maxStep) maxStep = r.step;
  }
  return { hasFinal, maxStep, rowsScanned: rows.length };
}

// ---------- the index ----------

export interface RunIndexOptions {
  /** absolute path to the fleet-run store */
  benchRoot: string;
  /** where to persist the derived cache; null disables persistence (tests) */
  cacheFile?: string | null;
  /** rescan throttle */
  ttlMs?: number;
}

export class RunIndex {
  readonly benchRoot: string;
  private readonly cacheFile: string | null;
  private readonly ttlMs: number;

  private cache = new Map<string, CachedRun>();
  private recordings = new Map<string, RecordingEntry[]>();
  private runs = new Map<string, ResolvedRun>();
  private order: string[] = [];

  private loadedCache = false;
  private lastScanAt = 0;
  private lastPersistAt = 0;
  private inFlightScan: Promise<void> | null = null;
  private dirty = false;
  /** roots already reported as skipped, so a permanently broken one logs once, not per scan */
  private readonly loggedSkips = new Set<string>();

  constructor(opts: RunIndexOptions) {
    this.benchRoot = opts.benchRoot;
    this.cacheFile = opts.cacheFile === undefined ? null : opts.cacheFile;
    this.ttlMs = opts.ttlMs ?? INDEX_TTL_MS;
  }

  /** Newest-first list of every discovered run. */
  async list(): Promise<RunIndexEntry[]> {
    await this.ensureFresh();
    return this.order.map((id) => this.runs.get(id)?.entry).filter((e): e is RunIndexEntry => e !== undefined);
  }

  /**
   * Resolve a request-supplied id against the discovered set. This is the ONLY way a
   * request reaches the filesystem: the id is a map key, never a path component.
   */
  async resolve(id: string): Promise<ResolvedRun | undefined> {
    await this.ensureFresh();
    const hit = this.runs.get(id);
    if (hit !== undefined) return hit;
    // A run created since the last scan deserves one rescan — but an unknown id is also
    // the cheapest thing an attacker (or a stale bookmark on a refresh loop) can send, so
    // the miss-rescan obeys the same TTL as everything else. Inside the window, a miss is
    // simply a miss.
    if (Date.now() - this.lastScanAt < this.ttlMs) return undefined;
    await this.rescan();
    return this.runs.get(id);
  }

  async recordingsFor(id: string): Promise<RecordingEntry[]> {
    await this.ensureFresh();
    return this.recordings.get(id) ?? [];
  }

  private async ensureFresh(): Promise<void> {
    if (Date.now() - this.lastScanAt < this.ttlMs && this.order.length > 0) return;
    await this.rescan();
  }

  private async rescan(): Promise<void> {
    if (this.inFlightScan !== null) return this.inFlightScan;
    const p = this.scan().finally(() => {
      this.inFlightScan = null;
    });
    this.inFlightScan = p;
    return p;
  }

  private async scan(): Promise<void> {
    await this.loadCache();

    let dirNames: string[];
    try {
      const ents = await readdir(this.benchRoot, { withFileTypes: true });
      dirNames = ents.filter((e) => e.isDirectory()).map((e) => e.name);
    } catch {
      dirNames = [];
    }

    const roles = roleMap(await readRoles(this.benchRoot));
    const now = Date.now();
    const nextRuns = new Map<string, ResolvedRun>();
    const nextCache = new Map<string, CachedRun>();
    const seenRecordings = new Map<string, RecordingEntry[]>();

    for (const id of dirNames) {
      try {
        const dir = join(this.benchRoot, id);
        const metaPath = join(dir, 'meta.json');
        const metaStat = await statOrNull(metaPath);
        if (metaStat === null) continue;
        const meta = await readRunMeta(metaPath);
        if (meta === null) continue;

        const fleetPath = join(dir, 'fleet.jsonl');
        const fleetStat = await statOrNull(fleetPath);
        // Tape-era roots (METRICS LAW 2026-08-05) write fleet.jsonl only at
        // the end (final rows); mid-run liveness comes from aggregate.jsonl,
        // which every era appends per heartbeat.
        const aggStat = await statOrNull(join(dir, 'aggregate.jsonl'));
        const fleetMtime = Math.max(fleetStat?.mtimeMs ?? 0, aggStat?.mtimeMs ?? 0);
        const fleetSize = (fleetStat?.size ?? 0) + (aggStat?.size ?? 0);
        const summaryStat = await statOrNull(join(dir, 'summary.json'));
        const summaryMtime = summaryStat === null ? -1 : summaryStat.mtimeMs;
        const recSig = await recordingsSignature(dir);
        const key = `${metaStat.mtimeMs}:${fleetMtime}:${fleetSize}:${summaryMtime}:${recSig}`;

        const prior = this.cache.get(id);
        let derived: CachedRun;
        if (prior !== undefined && prior.key === key) {
          derived = prior;
          const priorRecs = this.recordings.get(id);
          if (priorRecs !== undefined) seenRecordings.set(id, priorRecs);
          else if (derived.hasRecordings) {
            seenRecordings.set(id, await scanRecordings(dir, { root: this.benchRoot }));
          }
        } else {
          const probe = fleetStat === null
            ? { hasFinal: false, maxStep: 0, rowsScanned: 0 }
            : await probeFleetTail(fleetPath);
          // Mid-run progress for tape-era roots: fleet.jsonl is empty until
          // the final rows land, so the step high-water comes from the
          // aggregate heartbeat stream.
          if (!probe.hasFinal && aggStat !== null) {
            const aggProbe = await probeFleetTail(join(dir, 'aggregate.jsonl'));
            probe.maxStep = Math.max(probe.maxStep, aggProbe.maxStep);
          }
          const recs = await scanRecordings(dir, { root: this.benchRoot });
          seenRecordings.set(id, recs);
          const steps = typeof meta.steps === 'number' ? meta.steps : 0;
          derived = {
            key,
            id,
            envIds: envIdsOf(meta),
            batch: typeof meta.batch === 'number' ? meta.batch : 0,
            steps,
            policies: asStringArray(meta.policies),
            ablationLabel: ablationLabelOf(meta),
            startedAt: typeof meta.startedAt === 'string' ? meta.startedAt : null,
            complete: probe.hasFinal && probe.maxStep >= steps && steps > 0,
            maxStep: probe.maxStep,
            hasSummary: summaryStat !== null,
            hasRecordings: recs.length > 0,
          };
          this.dirty = true;
        }

        nextCache.set(id, derived);
        const inFlight =
          !derived.complete && fleetMtime > 0 && now - fleetMtime < IN_FLIGHT_MS;
        const role: RunRole | null = roles.get(id) ?? null;
        nextRuns.set(id, {
          id,
          dir,
          meta,
          entry: {
            id,
            envIds: derived.envIds,
            batch: derived.batch,
            steps: derived.steps,
            policies: derived.policies,
            ablationLabel: derived.ablationLabel,
            startedAt: derived.startedAt,
            complete: derived.complete,
            maxStep: derived.maxStep,
            inFlight,
            hasSummary: derived.hasSummary,
            hasRecordings: derived.hasRecordings,
            role,
          },
        });
      } catch (err) {
        // One broken root is data, not an outage: drop it from this index build and keep
        // going. Logged once per id for the life of the process.
        if (!this.loggedSkips.has(id)) {
          this.loggedSkips.add(id);
          const message = err instanceof Error ? err.message : String(err);
          console.warn(`[oak-frontend] skipping unreadable run root ${id}: ${message}`);
        }
      }
    }

    if (nextCache.size !== this.cache.size) this.dirty = true;

    this.cache = nextCache;
    this.recordings = seenRecordings;
    this.runs = nextRuns;
    this.order = [...nextRuns.keys()].sort((a, b) => {
      const ea = nextRuns.get(a)?.entry;
      const eb = nextRuns.get(b)?.entry;
      const sa = ea?.startedAt ?? '';
      const sb = eb?.startedAt ?? '';
      if (sa !== sb) return sa < sb ? 1 : -1; // newest first
      return a < b ? 1 : a > b ? -1 : 0;
    });
    this.lastScanAt = Date.now();
    void this.persist();
  }

  // ---------- disk cache ----------

  private async loadCache(): Promise<void> {
    if (this.loadedCache) return;
    this.loadedCache = true;
    if (this.cacheFile === null) return;
    try {
      const parsed: unknown = JSON.parse(await Bun.file(this.cacheFile).text());
      if (parsed === null || typeof parsed !== 'object') return;
      const f = parsed as Partial<CacheFile>;
      if (f.v !== 2 || f.benchRoot !== this.benchRoot || typeof f.runs !== 'object' || f.runs === null) return;
      for (const [id, rec] of Object.entries(f.runs)) {
        if (rec !== null && typeof rec === 'object' && typeof (rec as CachedRun).key === 'string') {
          this.cache.set(id, rec as CachedRun);
        }
      }
    } catch {
      /* a cold or corrupt cache just means a full scan */
    }
  }

  private async persist(): Promise<void> {
    if (this.cacheFile === null || !this.dirty) return;
    // An in-flight run re-derives every rescan; without a floor that would rewrite the
    // whole index file every TTL for the life of a 40-minute Tier-L.
    const now = Date.now();
    if (this.lastPersistAt !== 0 && now - this.lastPersistAt < PERSIST_MIN_INTERVAL_MS) return;
    this.lastPersistAt = now;
    this.dirty = false;
    const payload: CacheFile = {
      v: 2,
      benchRoot: this.benchRoot,
      runs: Object.fromEntries(this.cache),
    };
    try {
      await ensureCacheDir(this.cacheFile);
      await writeFile(this.cacheFile, JSON.stringify(payload));
    } catch {
      /* cache is an optimization, never a hard dependency */
    }
  }
}

/** Create the cache directory for `file` and drop a self-ignoring .gitignore into it. */
export async function ensureCacheDir(file: string): Promise<void> {
  const dir = dirname(file);
  if (dir.length === 0 || dir === file) return;
  await mkdir(dir, { recursive: true });
  const marker = join(dir, '.gitignore');
  if ((await statOrNull(marker)) === null) {
    await writeFile(marker, '*\n').catch(() => undefined);
  }
}

// ---------- run detail ----------

/**
 * Read + parse a JSON file, keeping "there is no file" and "the file is not JSON"
 * APART. Collapsing them into one `null` makes every downstream empty state a
 * guess, and the wrong guess sends the user to re-run the tool that wrote the
 * broken file. `missing` is a normal state; `unreadable` is a defect worth naming.
 */
type JsonRead<T> =
  | { value: T; error: null }
  | { value: null; error: 'missing' | 'unreadable' };

async function readJsonFile<T>(path: string): Promise<JsonRead<T>> {
  let text: string;
  try {
    text = await Bun.file(path).text();
  } catch {
    return { value: null, error: 'missing' };
  }
  try {
    return { value: JSON.parse(text) as T, error: null };
  } catch {
    return { value: null, error: 'unreadable' };
  }
}

/** Whole-file read that answers `null` for missing OR unreadable (directory, EACCES). */
async function readTextOrNull(path: string): Promise<string | null> {
  try {
    return await Bun.file(path).text();
  } catch {
    return null;
  }
}

export async function buildRunDetail(run: ResolvedRun, recs: RecordingEntry[]): Promise<RunDetail> {
  // Every summary leaves this function in v2 vocabulary, whichever schema is on disk —
  // the ~600 historical roots are never regenerated, and the client speaks one language.
  const summaryRead = await readJsonFile<RunSummary>(join(run.dir, 'summary.json'));
  const summary = adaptSummary(summaryRead.value);
  // A present-but-unparseable summary is reported as such; missing stays plain null.
  const summaryError = summaryRead.error === 'unreadable' ? 'unreadable' : null;

  const aggText = await readTextOrNull(join(run.dir, 'aggregate.jsonl'));
  const aggregate: AggregateRow[] = aggText === null ? [] : parseJsonl<AggregateRow>(aggText);

  const deathsText = await readTextOrNull(join(run.dir, 'deaths.jsonl'));
  const deathsCount = deathsText === null ? 0 : parseJsonl<DeathRow>(deathsText).length;

  return {
    id: run.id,
    meta: run.meta,
    summary,
    summaryError,
    aggregate,
    deathsCount,
    recordings: toRecordingRefs(recs),
  };
}

// ---------- standalone helper used by the tests + the SSE status event ----------

export interface Completeness {
  complete: boolean;
  maxStep: number;
  hasFinal: boolean;
  inFlight: boolean;
}

export async function runCompleteness(runDir: string, declaredSteps: number, now = Date.now()): Promise<Completeness> {
  const fleetPath = join(runDir, 'fleet.jsonl');
  const st = await statOrNull(fleetPath);
  if (st === null) return { complete: false, maxStep: 0, hasFinal: false, inFlight: false };
  const probe = await probeFleetTail(fleetPath);
  const complete = probe.hasFinal && probe.maxStep >= declaredSteps && declaredSteps > 0;
  return {
    complete,
    maxStep: probe.maxStep,
    hasFinal: probe.hasFinal,
    inFlight: !complete && now - st.mtimeMs < IN_FLIGHT_MS,
  };
}
