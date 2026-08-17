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
    etag: tape.etag,
    stepStart: [...tape.stepStart.subarray(0, tape.rows)],
    stepEnd: [...tape.stepEnd.subarray(0, tape.rows)],
    wallSeconds: [...tape.wallSeconds.subarray(0, tape.rows)],
    data: [...tape.data.subarray(0, tape.rows * valuesPerRow)],
  };
}

test('a whole-buffer parse equals split-feed appends', () => {
  const bytes = syntheticTapeBytes(2, FIELDS, rows);
  const whole = parseTape(bytes, 'etag-final');

  const headerEnd = whole.headerBytes;
  const rowBytes = whole.rowBytes;
  let split = parseTape(bytes.subarray(0, headerEnd + rowBytes), null);
  split = appendRows(split, bytes.subarray(headerEnd + rowBytes, headerEnd + 2 * rowBytes), null);
  split = appendRows(split, bytes.subarray(headerEnd + 2 * rowBytes), 'etag-final');

  expect(logicalRegion(split)).toEqual(logicalRegion(whole));
  expect(split.fields).toEqual(whole.fields);
  expect(split.laneCount).toBe(whole.laneCount);
});

test('an append shorter than one row changes nothing but the etag', () => {
  const bytes = syntheticTapeBytes(2, FIELDS, rows);
  const whole = parseTape(bytes, 'a');
  const nudged = appendRows(whole, new Uint8Array(5), 'b');
  expect(nudged.rows).toBe(whole.rows);
  expect(nudged.byteLength).toBe(whole.byteLength);
  expect(nudged.etag).toBe('b');
});

test('parsing a header-only tape yields zero rows', () => {
  const bytes = syntheticTapeBytes(2, FIELDS, []);
  const empty = parseTape(bytes, null);
  expect(empty.rows).toBe(0);
  expect(empty.byteLength).toBe(empty.headerBytes);
});
