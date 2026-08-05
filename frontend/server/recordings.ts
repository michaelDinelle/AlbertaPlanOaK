
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
  steps: number;
  actDim: number;
  recordVersion: number;
  actionsPath: string;
  metaPath: string;
  bytes: number;
  codec?: unknown;
  codecVersion?: number;
  chunks?: [number, number, number, number][];
}

const SEED_DIR = /^seed(\d+)$/;
const ENV_ORDER = new Map<string, number>(ENV_IDS.map((id, i) => [id, i]));

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


async function scanV4StepsUncached(path: string, size: number): Promise<number> {
  if (size < RECORD_V4_FILE_HEADER_BYTES + RECORD_V4_CHUNK_HEADER_BYTES) return 0;
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

const v4StepsMemo = new Map<string, { key: string; steps: number }>();
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

export function clearV4StepsCache(): void {
  v4StepsMemo.clear();
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
    }
    parts.push(`${name}:${Math.trunc(st?.mtimeMs ?? 0)}:${seeds}`);
  }
  return parts.join(',');
}

export function toRecordingRefs(entries: readonly RecordingEntry[]): RecordingRef[] {
  return entries.map((e) => ({ envId: e.envId, seed: e.seed, steps: e.steps }));
}

export function findRecording(
  entries: readonly RecordingEntry[],
  envId: string,
  seed: number,
): RecordingEntry | undefined {
  if (!isEnvId(envId) || !Number.isInteger(seed)) return undefined;
  return entries.find((e) => e.envId === envId && e.seed === seed);
}
