// Per-lane recording discovery: <run>/<envId>/seed<N>/{actions.bin,meta.json}.
// Two record versions are servable:
//   * v3 — flat 16 float32 per step (ctrl[6], done, pose[9]); steps = byteLength / 64.
//   * v4 — the chunked codec of `src/sim/RECORD_V4_SPEC.md`; steps comes from the
//     finalize-time `meta.json.steps`, and for an in-flight or torn lane (no `steps`)
//     from a walk of the chunk HEADERS only, memoized on the file's mtime+size.
// The returned entries carry the resolved absolute paths — request params are matched
// against THIS map, never joined into a path.
//
// SYMLINK INVARIANT: every discovered path is lstat'ed and rejected if it is a link, and
// each lane directory's realpath must still sit under the containment root. `stat` follows
// links, so without this a `seed0/actions.bin -> /etc/passwd` planted in a bench root would
// be served verbatim by the byte-range endpoint.

import { open, readdir, realpath } from 'node:fs/promises';
import { join, sep } from 'node:path';
import { ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef } from '@shared/types';
import {
  RECORD_V3_BYTES_PER_STEP,
  RECORD_V4_CHUNK_HEADER_BYTES,
  RECORD_V4_CHUNK_MAGIC,
  RECORD_V4_FILE_HEADER_BYTES,
  RECORD_V4_FILE_MAGIC,
  RECORD_V4_MAX_CHUNKS,
  RECORD_VERSION_SERVABLE,
} from './config';
import { lstatOrNull, statOrNull, type FileStat } from './jsonl';

export interface RecordingEntry {
  envId: EnvId;
  seed: number;
  /** decoded record rows: v3 = byteLength / 64, v4 = Σ chunk stepCount */
  steps: number;
  actDim: number;
  recordVersion: number;
  actionsPath: string;
  metaPath: string;
  bytes: number;
  /** v4 only: the codec descriptor, verbatim from the lane meta.json */
  codec?: unknown;
  /** v4 only: the codec revision the lane was written with */
  codecVersion?: number;
  /** v4 only: finalize-time chunk table — [stepStart, stepCount, byteOffset, byteLength] */
  chunks?: [number, number, number, number][];
}

const SEED_DIR = /^seed(\d+)$/;
const ENV_ORDER = new Map<string, number>(ENV_IDS.map((id, i) => [id, i]));

/** `meta.chunks` narrowed to the 4-tuple table, or undefined when absent/malformed. */
function readChunkTable(value: unknown): [number, number, number, number][] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined;
  const out: [number, number, number, number][] = [];
  for (const row of value) {
    if (!Array.isArray(row) || row.length < 4) return undefined;
    const [a, b, c, d] = row;
    if (
      typeof a !== 'number' || typeof b !== 'number' ||
      typeof c !== 'number' || typeof d !== 'number'
    ) {
      return undefined;
    }
    out.push([a, b, c, d]);
  }
  return out;
}

async function readLaneMeta(path: string): Promise<RecordingMeta | null> {
  try {
    const parsed: unknown = JSON.parse(await Bun.file(path).text());
    if (parsed === null || typeof parsed !== 'object') return null;
    const o = parsed as Record<string, unknown>;
    if (typeof o['recordVersion'] !== 'number') return null;
    const meta: RecordingMeta = {
      envId: typeof o['envId'] === 'string' ? o['envId'] : '',
      seed: typeof o['seed'] === 'number' ? o['seed'] : -1,
      actDim: typeof o['actDim'] === 'number' ? o['actDim'] : 0,
      recordVersion: o['recordVersion'],
    };
    if (o['codec'] !== undefined) meta.codec = o['codec'];
    if (typeof o['codecVersion'] === 'number') meta.codecVersion = o['codecVersion'];
    if (typeof o['steps'] === 'number') meta.steps = o['steps'];
    const chunks = readChunkTable(o['chunks']);
    if (chunks !== undefined) meta.chunks = chunks;
    return meta;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// v4 step counting
// ---------------------------------------------------------------------------

/**
 * Header-only walk of a v4 actions.bin: 32 bytes read per chunk, no payload and
 * therefore no crc — the client decoder is the authority on chunk validity, this
 * is the directory listing's step count. Stops at the first header that does not
 * continue the file (foreign magic, empty chunk, payload past EOF), which is
 * exactly the torn tail an in-flight or killed run leaves behind.
 */
async function scanV4StepsUncached(path: string, size: number): Promise<number> {
  if (size < RECORD_V4_FILE_HEADER_BYTES + RECORD_V4_CHUNK_HEADER_BYTES) return 0;
  // Nothing here may throw: one unreadable lane must not take down the run's whole
  // recording scan (the error-isolation rule the rest of this module already follows
  // via lstatOrNull). `open` itself can fail — EACCES, or a lane deleted mid-walk.
  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(path, 'r');
  } catch {
    return 0;
  }
  const header = Buffer.allocUnsafe(RECORD_V4_CHUNK_HEADER_BYTES);
  try {
    const first = await handle.read(header, 0, RECORD_V4_FILE_HEADER_BYTES, 0);
    if (first.bytesRead < RECORD_V4_FILE_HEADER_BYTES) return 0;
    if (header.readUInt32LE(0) !== RECORD_V4_FILE_MAGIC) return 0;
    if (header.readUInt32LE(4) !== 4) return 0;

    let steps = 0;
    let offset = RECORD_V4_FILE_HEADER_BYTES;
    for (let seen = 0; seen < RECORD_V4_MAX_CHUNKS; seen++) {
      if (offset + RECORD_V4_CHUNK_HEADER_BYTES > size) break;
      const got = await handle.read(header, 0, RECORD_V4_CHUNK_HEADER_BYTES, offset);
      if (got.bytesRead < RECORD_V4_CHUNK_HEADER_BYTES) break;
      if (header.readUInt32LE(0) !== RECORD_V4_CHUNK_MAGIC) break;
      const stepStart = header.readUInt32LE(4);
      const stepCount = header.readUInt32LE(8);
      const compBytes = header.readUInt32LE(16);
      // stepCount/compBytes of 0 cannot advance the walk; a stepStart that skips
      // is the torn edge. Either way: stop, and keep what came before.
      if (stepCount === 0 || compBytes === 0) break;
      if (stepStart !== steps) break;
      const next = offset + RECORD_V4_CHUNK_HEADER_BYTES + compBytes;
      if (next > size) break;
      steps += stepCount;
      offset = next;
    }
    return steps;
  } catch {
    return 0;
  } finally {
    await handle.close().catch(() => undefined);
  }
}

/**
 * mtime+size-keyed memo over the header walk, mirroring the run-index cache key
 * discipline: a lane that is still being written re-scans, a finalized one never
 * does. Only reached when `meta.json` carries no `steps`.
 */
const v4StepsMemo = new Map<string, { key: string; steps: number }>();
/** memo ceiling — a bench root with thousands of in-flight lanes must not grow it forever */
const V4_STEPS_MEMO_MAX = 4096;

async function scanV4Steps(path: string, st: FileStat): Promise<number> {
  const key = `${Math.trunc(st.mtimeMs)}:${st.size}`;
  const hit = v4StepsMemo.get(path);
  if (hit !== undefined && hit.key === key) return hit.steps;
  const steps = await scanV4StepsUncached(path, st.size);
  if (v4StepsMemo.size >= V4_STEPS_MEMO_MAX) v4StepsMemo.clear();
  v4StepsMemo.set(path, { key, steps });
  return steps;
}

/** Test seam: drop the memo so a rewritten fixture is re-scanned. */
export function clearV4StepsCache(): void {
  v4StepsMemo.clear();
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
      if (laneMeta === null || !RECORD_VERSION_SERVABLE.has(laneMeta.recordVersion)) continue;
      const st = await lstatOrNull(actionsPath);
      if (st === null || st.isSymbolicLink || !st.isFile) continue;

      const v4 = laneMeta.recordVersion === 4;
      if (st.size < (v4 ? RECORD_V4_FILE_HEADER_BYTES : RECORD_V3_BYTES_PER_STEP)) continue;

      // v4 steps: the finalize-time hint when it is a sane positive integer,
      // otherwise a header walk of the file itself (in-flight / torn lanes).
      let steps: number;
      if (!v4) {
        steps = Math.floor(st.size / RECORD_V3_BYTES_PER_STEP);
      } else if (
        typeof laneMeta.steps === 'number' &&
        Number.isSafeInteger(laneMeta.steps) &&
        laneMeta.steps > 0
      ) {
        steps = laneMeta.steps;
      } else {
        steps = await scanV4Steps(actionsPath, st);
      }

      out.push({
        envId: envName,
        seed,
        steps,
        actDim: laneMeta.actDim,
        recordVersion: laneMeta.recordVersion,
        actionsPath,
        metaPath,
        bytes: st.size,
        ...(v4 && laneMeta.codec !== undefined ? { codec: laneMeta.codec } : {}),
        ...(v4 && laneMeta.codecVersion !== undefined ? { codecVersion: laneMeta.codecVersion } : {}),
        ...(v4 && laneMeta.chunks !== undefined ? { chunks: laneMeta.chunks } : {}),
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
