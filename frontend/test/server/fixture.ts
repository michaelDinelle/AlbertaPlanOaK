import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { OFF_DONE, OFF_POSE, RECORD_FLOATS } from '@sim/record';
import { encodeRecord } from '../helpers/encode';
import { tapeBytes, type TapeRowSpec } from '../helpers/tape';

export const ENV = 'FourRooms-cont-v0';

export function laneRows(steps: number): Float32Array {
  const rows = new Float32Array(steps * RECORD_FLOATS);
  for (let i = 0; i < steps; i++) {
    const at = i * RECORD_FLOATS;
    rows[at] = i % 4;
    rows[at + OFF_DONE] = i === steps - 1 ? 1 : 0;
    rows[at + OFF_POSE] = i % 200;
    rows[at + OFF_POSE + 1] = 7;
    rows[at + OFF_POSE + 2] = 3;
    rows[at + OFF_POSE + 3] = 1;
  }
  return rows;
}

export function laneBytes(envId: string, steps: number, chunkSteps = 4): Uint8Array {
  return encodeRecord(envId, laneRows(steps), steps, chunkSteps).bytes;
}

export function runMeta(steps: number, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    batch: 2,
    seeds: [0, 1],
    steps,
    ablationLabel: 'none',
    envSchedule: [{ envId: ENV, seedStart: 0, seedEnd: 1 }],
    ...extra,
  });
}

export function completeTape(laneCount: number, steps: number): Uint8Array {
  const half = Math.floor(steps / 2);
  const rows: TapeRowSpec[] = [
    { stepStart: 0, stepCount: half, values: {} },
    { stepStart: half, stepCount: steps - half, values: {} },
  ];
  return tapeBytes(laneCount, rows);
}

export interface LaneFile {
  env: string;
  seed: number;
  bytes: Uint8Array;
}

export interface RunSpec {
  meta?: string;
  tape?: Uint8Array;
  summary?: string;
  lanes?: LaneFile[];
}

export async function writeRun(benchRoot: string, id: string, spec: RunSpec): Promise<string> {
  const dir = join(benchRoot, id);
  await mkdir(dir, { recursive: true });
  if (spec.meta !== undefined) await writeFile(join(dir, 'meta.json'), spec.meta);
  if (spec.tape !== undefined) await writeFile(join(dir, 'tape.bin'), spec.tape);
  if (spec.summary !== undefined) await writeFile(join(dir, 'summary.json'), spec.summary);
  for (const lane of spec.lanes ?? []) {
    const laneDir = join(dir, lane.env, `seed${lane.seed}`);
    await mkdir(laneDir, { recursive: true });
    await writeFile(join(laneDir, 'actions.bin'), lane.bytes);
  }
  return dir;
}
