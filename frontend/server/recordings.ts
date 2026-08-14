import { open, readdir, realpath } from 'node:fs/promises';
import { join, sep } from 'node:path';
import { ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingRef } from '@shared/types';
import { FILE_HEADER_BYTES, FILE_MAGIC, RECORD_ENV_KINDS } from '@sim/record';
import { lstatOrNull } from './fs';

export interface RecordingEntry {
  envId: EnvId;
  seed: number;
  steps: number;
  bytes: number;
  actionsPath: string;
}

const SEED_DIR = /^seed(\d+)$/;
const ENV_ORDER = new Map<string, number>(ENV_IDS.map((id, i) => [id, i]));

async function laneSteps(path: string, envId: EnvId): Promise<number> {
  const handle = await open(path, 'r');
  const header = Buffer.allocUnsafe(FILE_HEADER_BYTES);
  try {
    const got = await handle.read(header, 0, FILE_HEADER_BYTES, 0);
    if (got.bytesRead < FILE_HEADER_BYTES) return 0;
    if (header.readUInt32LE(0) !== FILE_MAGIC) {
      throw new Error(`${path}: not a recording (bad magic)`);
    }
    const envKind = header.readUInt32LE(4);
    if (RECORD_ENV_KINDS[envKind] !== envId) {
      throw new Error(`${path}: header envKind ${envKind} contradicts directory env ${envId}`);
    }
    return header.readUInt32LE(8);
  } finally {
    await handle.close().catch(() => undefined);
  }
}

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

async function containedIn(path: string, root: string): Promise<boolean> {
  try {
    const real = await realpath(path);
    return real === root || real.startsWith(root + sep);
  } catch {
    return false;
  }
}

export interface ScanOptions {
  root?: string;
}

export async function scanRecordings(runDir: string, opts: ScanOptions = {}): Promise<RecordingEntry[]> {
  const top = await envDirsOf(runDir);
  if (top.length === 0) return [];

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
    const scanned = await Promise.all(seedDirs.map(async (seedName): Promise<RecordingEntry | null> => {
      const m = SEED_DIR.exec(seedName);
      const seedText = m?.[1];
      if (seedText === undefined) return null;
      const seed = Number(seedText);
      if (!Number.isInteger(seed) || seed < 0) return null;

      const laneDir = join(envDir, seedName);
      const laneStat = await lstatOrNull(laneDir);
      if (laneStat === null || laneStat.isSymbolicLink || !laneStat.isDirectory) return null;
      if (!(await containedIn(laneDir, root))) return null;

      const actionsPath = join(laneDir, 'actions.bin');
      const st = await lstatOrNull(actionsPath);
      if (st === null || st.isSymbolicLink || !st.isFile) return null;
      if (st.size < FILE_HEADER_BYTES) return null;

      let steps: number;
      try {
        steps = await laneSteps(actionsPath, envName);
      } catch (err) {
        console.warn(`[oak-frontend] ${err instanceof Error ? err.message : String(err)}`);
        return null;
      }
      if (steps === 0) return null;

      return { envId: envName, seed, steps, bytes: st.size, actionsPath };
    }));
    for (const entry of scanned) if (entry !== null) out.push(entry);
  }

  out.sort((a, b) => {
    const ea = ENV_ORDER.get(a.envId) ?? 999;
    const eb = ENV_ORDER.get(b.envId) ?? 999;
    return ea === eb ? a.seed - b.seed : ea - eb;
  });
  return out;
}

export function toRecordingRefs(entries: readonly RecordingEntry[]): RecordingRef[] {
  return entries.map((e) => ({
    envId: e.envId,
    seed: e.seed,
    steps: e.steps,
    bytes: e.bytes,
  }));
}

export function findRecording(
  entries: readonly RecordingEntry[],
  envId: string,
  seed: number,
): RecordingEntry | undefined {
  if (!isEnvId(envId) || !Number.isInteger(seed)) return undefined;
  return entries.find((e) => e.envId === envId && e.seed === seed);
}
