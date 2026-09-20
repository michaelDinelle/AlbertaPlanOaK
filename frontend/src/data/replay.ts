import type { EnvDescriptor } from './envs';
import type { Policy } from './layout';

export const TAPE_MAGIC = 'OAKTAPE\0';
export const HASH_MAGIC = 'OAKHASH\0';
export const HEADER_BYTES = 32;
export const VERDICT_BYTES = 32;

export interface ReplayDescriptor {
  environment: EnvDescriptor;
  seed: number;
  horizon: number;
  policy: Policy;
  preset: string;
  lane: number;
  controls: number;
  actionMap?: number[];
}

export interface TapeHeader {
  descriptor: ReplayDescriptor;
  offset: number;
  width: number;
  frames: number;
  blockFrames: number;
}

export function readHeader(bytes: Uint8Array): TapeHeader | null {
  if (bytes.length < HEADER_BYTES) return null;
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== TAPE_MAGIC) throw new Error('Invalid replay tape');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = view.getUint32(8, true);
  if (length > 1024 * 1024) throw new Error('Invalid replay descriptor length');
  const offset = HEADER_BYTES + Math.ceil(length / 4) * 4;
  if (bytes.length < offset) return null;
  const descriptor = JSON.parse(new TextDecoder().decode(bytes.subarray(HEADER_BYTES, HEADER_BYTES + length))) as ReplayDescriptor;
  const width = view.getUint32(12, true), frames = view.getUint32(16, true), blockFrames = view.getUint32(20, true);
  const env = descriptor.environment;
  if (!env.replay || width !== env.pose + env.obs + descriptor.controls + 2 || frames !== descriptor.horizon + 1 || !blockFrames || blockFrames > 65536) throw new Error('Inconsistent replay descriptor');
  return { descriptor, offset, width, frames, blockFrames };
}

export function readVerdict(bytes: Uint8Array, header: TapeHeader): { identical: boolean; computed: bigint; recorded: bigint; health: number } {
  if (bytes.length !== VERDICT_BYTES || new TextDecoder().decode(bytes.subarray(0, 8)) !== HASH_MAGIC) throw new Error('Replay ended without a reward-hash verdict');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const computed = view.getBigUint64(8, true), recorded = view.getBigUint64(16, true);
  if (view.getUint32(24, true) !== header.frames) throw new Error('Incomplete replay verdict');
  return { identical: computed === recorded, computed, recorded, health: view.getUint32(28, true) };
}
