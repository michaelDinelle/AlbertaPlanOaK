// Per-lane recording discovery: <run>/<envId>/seed<N>/{actions.bin,meta.json}.
// Only recordVersion 3 is servable (16 float32 per step: ctrl[6], done, pose[9]).
// The returned entries carry the resolved absolute paths — request params are matched
// against THIS map, never joined into a path.
//
// SYMLINK INVARIANT: every discovered path is lstat'ed and rejected if it is a link, and
// each lane directory's realpath must still sit under the containment root. `stat` follows
// links, so without this a `seed0/actions.bin -> /etc/passwd` planted in a bench root would
// be served verbatim by the byte-range endpoint.

import { readdir, realpath } from 'node:fs/promises';
import { join, sep } from 'node:path';
import { ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef } from '@shared/types';
import { RECORD_V3_BYTES_PER_STEP, RECORD_VERSION_SERVABLE } from './config';
import { lstatOrNull, statOrNull } from './jsonl';

export interface RecordingEntry {
  envId: EnvId;
  seed: number;
  /** actions.bin byteLength / 64 */
  steps: number;
  actDim: number;
  recordVersion: number;
  actionsPath: string;
  metaPath: string;
  bytes: number;
}

const SEED_DIR = /^seed(\d+)$/;
const ENV_ORDER = new Map<string, number>(ENV_IDS.map((id, i) => [id, i]));

async function readLaneMeta(path: string): Promise<RecordingMeta | null> {
  try {
    const parsed: unknown = JSON.parse(await Bun.file(path).text());
    if (parsed === null || typeof parsed !== 'object') return null;
    const o = parsed as Record<string, unknown>;
    if (typeof o['recordVersion'] !== 'number') return null;
    return {
      envId: typeof o['envId'] === 'string' ? o['envId'] : '',
      seed: typeof o['seed'] === 'number' ? o['seed'] : -1,
      actDim: typeof o['actDim'] === 'number' ? o['actDim'] : 0,
      recordVersion: o['recordVersion'],
    };
  } catch {
    return null;
  }
}

/** List the env-id directories directly under a run root (dirents only — never links). */
async function envDirsOf(runDir: string): Promise<string[]> {
  try {
    const ents = await readdir(runDir, { withFileTypes: true });
    return ents
      .filter((e) => e.isDirectory() && !e.isSymbolicLink() && isEnvId(e.name))
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

/** True iff `path` resolves (through every symlink) to `root` or something under it. */
async function containedIn(path: string, root: string): Promise<boolean> {
  try {
    const real = await realpath(path);
    return real === root || real.startsWith(root + sep);
  } catch {
    return false;
  }
}

export interface ScanOptions {
  /**
   * Containment root every lane directory must resolve inside. Defaults to `runDir`;
   * the index passes the bench root so a run-root-relative link is still caught.
   */
  root?: string;
}

/**
 * Scan one run root for servable recordings. Directories that are not one of the 16
 * known env ids are skipped outright, which is also what keeps this cheap for the ~90%
 * of run roots that have no recordings at all (one readdir and done).
 */
export async function scanRecordings(runDir: string, opts: ScanOptions = {}): Promise<RecordingEntry[]> {
  const top = await envDirsOf(runDir);
  if (top.length === 0) return [];

  // Resolved once: the containment root itself may sit behind a link (/tmp -> /private/tmp
  // on macOS), and comparing a resolved path against an unresolved prefix rejects
  // everything.
  let root: string;
  try {
    root = await realpath(opts.root ?? runDir);
  } catch {
    return [];
  }

  const out: RecordingEntry[] = [];
  for (const envName of top) {
    if (!isEnvId(envName)) continue;
    const envDir = join(runDir, envName);
    let seedDirs: string[];
    try {
      const ents = await readdir(envDir, { withFileTypes: true });
      seedDirs = ents
        .filter((e) => e.isDirectory() && !e.isSymbolicLink() && SEED_DIR.test(e.name))
        .map((e) => e.name);
    } catch {
      continue;
    }
    for (const seedName of seedDirs) {
      const m = SEED_DIR.exec(seedName);
      if (m === null) continue;
      const seedText = m[1];
      if (seedText === undefined) continue;
      const seed = Number(seedText);
      if (!Number.isInteger(seed) || seed < 0) continue;

      const laneDir = join(envDir, seedName);
      const laneStat = await lstatOrNull(laneDir);
      if (laneStat === null || laneStat.isSymbolicLink || !laneStat.isDirectory) continue;
      if (!(await containedIn(laneDir, root))) continue;

      const metaPath = join(laneDir, 'meta.json');
      const actionsPath = join(laneDir, 'actions.bin');
      const metaStat = await lstatOrNull(metaPath);
      if (metaStat === null || metaStat.isSymbolicLink || !metaStat.isFile) continue;
      const laneMeta = await readLaneMeta(metaPath);
      if (laneMeta === null || laneMeta.recordVersion !== RECORD_VERSION_SERVABLE) continue;
      const st = await lstatOrNull(actionsPath);
      if (st === null || st.isSymbolicLink || !st.isFile) continue;
      if (st.size < RECORD_V3_BYTES_PER_STEP) continue;

      out.push({
        envId: envName,
        seed,
        steps: Math.floor(st.size / RECORD_V3_BYTES_PER_STEP),
        actDim: laneMeta.actDim,
        recordVersion: laneMeta.recordVersion,
        actionsPath,
        metaPath,
        bytes: st.size,
      });
    }
  }

  out.sort((a, b) => {
    const ea = ENV_ORDER.get(a.envId) ?? 999;
    const eb = ENV_ORDER.get(b.envId) ?? 999;
    return ea === eb ? a.seed - b.seed : ea - eb;
  });
  return out;
}

/**
 * Cheap fingerprint of a run's recording tree, for the run-index memo key.
 *
 * One readdir of the run root plus one readdir + stat per env directory (0 for the ~90%
 * of roots with no recordings, ≤16 otherwise) — deliberately NOT a per-lane walk, which
 * would be 128 stats per root per rescan. It changes when an env directory appears or
 * disappears, when lanes are added or removed inside one, and when a lane directory is
 * renamed. Growth INSIDE an existing lane file is caught by the fleet.jsonl mtime+size
 * already in the key: a run that is still recording is also still writing heartbeats.
 */
export async function recordingsSignature(runDir: string): Promise<string> {
  const envDirs = await envDirsOf(runDir);
  if (envDirs.length === 0) return '';
  const parts: string[] = [];
  for (const name of envDirs) {
    const envDir = join(runDir, name);
    const st = await statOrNull(envDir);
    let seeds = 0;
    try {
      const ents = await readdir(envDir, { withFileTypes: true });
      for (const e of ents) if (e.isDirectory() && SEED_DIR.test(e.name)) seeds++;
    } catch {
      /* unreadable env dir contributes its name + mtime only */
    }
    parts.push(`${name}:${Math.trunc(st?.mtimeMs ?? 0)}:${seeds}`);
  }
  return parts.join(',');
}

export function toRecordingRefs(entries: readonly RecordingEntry[]): RecordingRef[] {
  return entries.map((e) => ({ envId: e.envId, seed: e.seed, steps: e.steps }));
}

/** Resolve request params against a discovered list. Unknown env/seed answers undefined. */
export function findRecording(
  entries: readonly RecordingEntry[],
  envId: string,
  seed: number,
): RecordingEntry | undefined {
  if (!isEnvId(envId) || !Number.isInteger(seed)) return undefined;
  return entries.find((e) => e.envId === envId && e.seed === seed);
}
