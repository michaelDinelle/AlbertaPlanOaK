import { join } from 'node:path';
import { ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingRef } from '@shared/types';
import { RECORD_ENV_KINDS } from '@sim/record';
import { lstatOrNull } from './fs';

const PACK_MAGIC = 0x504b_414f;
const IDX_MAGIC = 0x494b_414f;

export interface RecordingEntry {
  envId: EnvId;
  seed: number;
  bytes: number;
  lane: number;
}

export interface PoseIndex {
  packPath: string;
  segments: { off: number; len: number }[][];
  entries: RecordingEntry[];
  idxMtimeMs: number;
  idxSize: number;
}

const ENV_ORDER = new Map<string, number>(ENV_IDS.map((id, i) => [id, i]));

export async function readPoseIndex(runDir: string): Promise<PoseIndex | null> {
  const packPath = join(runDir, 'pose.pack');
  const idxPath = join(runDir, 'pose.idx');
  const packStat = await lstatOrNull(packPath);
  const idxStat = await lstatOrNull(idxPath);
  if (packStat === null || packStat.isSymbolicLink || !packStat.isFile) return null;
  if (idxStat === null || idxStat.isSymbolicLink || !idxStat.isFile) return null;

  const packHead = Buffer.from(await Bun.file(packPath).slice(0, 8).arrayBuffer());
  if (packHead.byteLength < 8) return null;
  if (packHead.readUInt32LE(0) !== PACK_MAGIC) {
    throw new Error(`${packPath}: not a pose pack (bad magic)`);
  }
  const laneCount = packHead.readUInt32LE(4);
  const laneTable = Buffer.from(await Bun.file(packPath).slice(8, 8 + laneCount * 12).arrayBuffer());
  if (laneTable.byteLength < laneCount * 12) return null;

  const idx = Buffer.from(await Bun.file(idxPath).arrayBuffer());
  if (idx.byteLength < 8 || idx.readUInt32LE(0) !== IDX_MAGIC) {
    throw new Error(`${idxPath}: not a pose index (bad magic)`);
  }
  if (idx.readUInt32LE(4) !== laneCount) {
    throw new Error(`${idxPath}: lane count contradicts ${packPath}`);
  }
  const segments: { off: number; len: number }[][] = Array.from({ length: laneCount }, () => []);
  let at = 8;
  while (at + 16 <= idx.byteLength) {
    const chunkCount = idx.readUInt32LE(at + 4);
    let off = Number(idx.readBigUInt64LE(at + 8));
    at += 16;
    if (at + chunkCount * 8 > idx.byteLength) break;
    for (let c = 0; c < chunkCount; c++) {
      const lane = idx.readUInt32LE(at);
      const len = idx.readUInt32LE(at + 4);
      at += 8;
      const perLane = segments[lane];
      if (perLane === undefined) throw new Error(`${idxPath}: lane ${lane} out of range`);
      perLane.push({ off, len });
      off += len;
    }
  }

  const entries: RecordingEntry[] = [];
  for (let l = 0; l < laneCount; l++) {
    const envKind = laneTable.readUInt32LE(l * 12);
    const envId = RECORD_ENV_KINDS[envKind];
    if (envId === undefined || !isEnvId(envId)) {
      throw new Error(`${packPath}: lane ${l} has unknown envKind ${envKind}`);
    }
    const body = segments[l]!.reduce((n, s) => n + s.len, 0);
    entries.push({ envId, seed: Number(laneTable.readBigInt64LE(l * 12 + 4)), bytes: body, lane: l });
  }
  entries.sort((a, b) => {
    const ea = ENV_ORDER.get(a.envId) ?? 999;
    const eb = ENV_ORDER.get(b.envId) ?? 999;
    return ea === eb ? a.seed - b.seed : ea - eb;
  });

  return { packPath, segments, entries, idxMtimeMs: idxStat.mtimeMs, idxSize: idxStat.size };
}

export function toRecordingRefs(entries: readonly RecordingEntry[]): RecordingRef[] {
  return entries.map((e) => ({ envId: e.envId, seed: e.seed, bytes: e.bytes }));
}

export function findRecording(
  entries: readonly RecordingEntry[],
  envId: string,
  seed: number,
): RecordingEntry | undefined {
  if (!isEnvId(envId) || !Number.isInteger(seed)) return undefined;
  return entries.find((e) => e.envId === envId && e.seed === seed);
}
