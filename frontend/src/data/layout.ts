export const POLICIES = ['oak', 'ppo', 'random'] as const;
export type Policy = (typeof POLICIES)[number];
export type Baseline = Exclude<Policy, 'oak'>;

export interface Meta {
  seeds: number[];
  steps: number;
  batch: number;
  window: number;
  policy: Policy;
  preset: string;
  startedAt: string;
  finishedAt?: string;
  stepsPerSecond?: number;
  envSchedule: { envId: string; seedStart: number; seedEnd: number }[];
  envCatalog: unknown;
}

export type Baselines = Partial<Record<Baseline, Record<string, string>>>;

export const ENGINE_FILES = {
  meta: 'meta.json',
  ends: 'ends.u32',
  rewardSum: 'series/rewardSum.f32',
  hash: 'hash.u64',
  health: 'health.u32',
} as const;

export const JUDGE_FILE = 'baselines.json';
export const HEALTH_ALIVE = 0xffffffff;

export interface RunIndexEntry {
  id: string;
  envIds: string[];
  steps: number;
  startedAt: string;
  inFlight: boolean;
  policy: Policy;
  preset: string;
  role: 'best' | null;
}

export interface Recording {
  id: string;
  meta: Meta;
  baselines: Baselines;
  windows: number;
  lanes: number;
  ends: Uint32Array;
  rewardSum: Float32Array;
  hash: BigUint64Array;
  health: Uint32Array;
}

const HEADER_BYTES = 16;
const pad8 = (n: number): number => (n + 7) & ~7;

export function packRecording(meta: Meta, baselines: Baselines, ends: Uint32Array, rewardSum: Float32Array, hash: BigUint64Array, health: Uint32Array): ArrayBuffer {
  const lanes = meta.batch;
  const windows = Math.min(ends.length, Math.floor(rewardSum.length / lanes));
  if (hash.length !== lanes || health.length !== lanes) throw new Error(`recording holds ${hash.length} hashes and ${health.length} health words for ${lanes} lanes`);
  const metaBytes = new TextEncoder().encode(JSON.stringify(meta));
  const baselineBytes = new TextEncoder().encode(JSON.stringify(baselines));
  const sections = [
    metaBytes,
    baselineBytes,
    new Uint8Array(ends.buffer, ends.byteOffset, windows * 4),
    new Uint8Array(rewardSum.buffer, rewardSum.byteOffset, windows * lanes * 4),
    new Uint8Array(hash.buffer, hash.byteOffset, lanes * 8),
    new Uint8Array(health.buffer, health.byteOffset, lanes * 4),
  ];
  const out = new Uint8Array(HEADER_BYTES + sections.reduce((n, s) => n + pad8(s.byteLength), 0));
  new Uint32Array(out.buffer, 0, 4).set([metaBytes.byteLength, baselineBytes.byteLength, windows, lanes]);
  let at = HEADER_BYTES;
  for (const section of sections) {
    out.set(section, at);
    at += pad8(section.byteLength);
  }
  return out.buffer;
}

export function unpackRecording(id: string, buffer: ArrayBuffer): Recording {
  const [metaLength, baselineLength, windows, lanes] = new Uint32Array(buffer, 0, 4) as unknown as [number, number, number, number];
  let at = HEADER_BYTES;
  const text = (length: number): string => {
    const s = new TextDecoder().decode(new Uint8Array(buffer, at, length));
    at += pad8(length);
    return s;
  };
  const meta = JSON.parse(text(metaLength)) as Meta;
  const baselines = JSON.parse(text(baselineLength)) as Baselines;
  const ends = new Uint32Array(buffer, at, windows);
  at += pad8(windows * 4);
  const rewardSum = new Float32Array(buffer, at, windows * lanes);
  at += pad8(windows * lanes * 4);
  const hash = new BigUint64Array(buffer, at, lanes);
  at += pad8(lanes * 8);
  const health = new Uint32Array(buffer, at, lanes);
  return { id, meta, baselines, windows, lanes, ends, rewardSum, hash, health };
}
