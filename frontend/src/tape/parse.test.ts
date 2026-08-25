import { expect, test } from 'bun:test';
import { appendRows, parseTape, type Tape } from './parse';
import { syntheticTapeBytes, type SyntheticRow } from './synthetic';

const FIELDS = ['rewardSum', 'rho'];

const rows: SyntheticRow[] = [
  { start: 0, count: 100, wall: 1, values: [[1, 2], [0.1, 0.2]] },
  { start: 100, count: 100, wall: 2, values: [[3, 4], [0.3, 0.4]] },
  { start: 200, count: 100, wall: 3, values: [[5, 6], [0.5, 0.6]] },
];

function logicalRegion(tape: Tape) {
  const valuesPerRow = tape.fields.length * tape.laneCount;
  return {
    rows: tape.rows,
    byteLength: tape.byteLength,
    stepStart: [...tape.stepStart.subarray(0, tape.rows)],
    stepEnd: [...tape.stepEnd.subarray(0, tape.rows)],
    wallSeconds: [...tape.wallSeconds.subarray(0, tape.rows)],
    data: [...tape.data.subarray(0, tape.rows * valuesPerRow)],
  };
}

test('a whole-buffer parse equals split-feed appends', async () => {
  const bytes = await syntheticTapeBytes(2, FIELDS, rows);
  const whole = await parseTape(bytes);

  const headerEnd = whole.headerBytes;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const row1End = headerEnd + 16 + dv.getUint32(headerEnd + 12, true);
  const row2End = row1End + 16 + dv.getUint32(row1End + 12, true);
  let split = await parseTape(bytes.subarray(0, row1End));
  split = await appendRows(split, bytes.subarray(row1End, row2End));
  split = await appendRows(split, bytes.subarray(row2End));

  expect(logicalRegion(split)).toEqual(logicalRegion(whole));
  expect(split.fields).toEqual(whole.fields);
  expect(split.laneCount).toBe(whole.laneCount);
});

test('an append shorter than one row changes nothing', async () => {
  const bytes = await syntheticTapeBytes(2, FIELDS, rows);
  const whole = await parseTape(bytes);
  const nudged = await appendRows(whole, new Uint8Array(5));
  expect(nudged.rows).toBe(whole.rows);
  expect(nudged.byteLength).toBe(whole.byteLength);
});

test('parsing a header-only tape yields zero rows', async () => {
  const bytes = await syntheticTapeBytes(2, FIELDS, []);
  const empty = await parseTape(bytes);
  expect(empty.rows).toBe(0);
  expect(empty.byteLength).toBe(empty.headerBytes);
});
