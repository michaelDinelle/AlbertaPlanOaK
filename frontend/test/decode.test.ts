/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import type { EnvId } from '@shared/envs';
import { ENV_IDS } from '@shared/envs';
import { DecodeError, decodeRecording, eventTimeline, lensRecording } from '@sim/decode';
import { NO_EVENTS } from '@sim/events';
import { RECORD_BYTES, RECORD_FLOATS, RecordParseError } from '@sim/record';
import { parseRawTrace, rawRecordingFromFloats } from './helpers/rawTrace';
import { decoderFor } from '@sim/registry';
import type { EnvDecoder, EnvFrame, RawRecording, SimEvent } from '@sim/types';

import { acrobotDecoder } from '@sim/envs/acrobot';
import { cartPoleDecoder } from '@sim/envs/cartpole';
import { halfCheetahDecoder } from '@sim/envs/halfcheetah';
import { mountainCarDecoder } from '@sim/envs/mountaincar';
import { pendulumDecoder } from '@sim/envs/pendulum';

const APPROX_TOL = 1e-12;

interface TraceRow {
  step: number;
  done: number;
  pose: number[];
  state: Record<string, number>;
}

interface TraceDoc {
  version: number;
  envId: EnvId;
  seed: number;
  kind: string;
  bin: string;
  windowStart: number;
  steps: number;
  initIsPredecessor: boolean;
  init: { step: number; done: number; ctrl: number[]; pose: number[]; state: Record<string, number> };
  ctrlHead: number[][];
  rows: TraceRow[];
  doneRows: number[];
  doneEvents: string[];
  approxFields: string[];
  oneStepParity: { steps: number; maxAbsErr: number; doneMismatch: number } | null;
}

function loadDoc(name: string): TraceDoc {
  return JSON.parse(
    readFileSync(new URL(`./fixtures/traces/${name}.json`, import.meta.url), 'utf8'),
  ) as TraceDoc;
}

function loadRaw(doc: TraceDoc): RawRecording {
  const buf = readFileSync(new URL(`./fixtures/traces/${doc.bin}`, import.meta.url));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  return parseRawTrace(ab, { envId: doc.envId, seed: doc.seed });
}

function same(a: number, b: number): boolean {
  return a === b || (Number.isNaN(a) && Number.isNaN(b));
}

function eventTypes(evs: readonly SimEvent[]): string[] {
  return evs.map((e) => e.type);
}

function checkTrace<S, St>(decoder: EnvDecoder<S, St>, name: string): string[] {
  const doc = loadDoc(name);
  const raw = loadRaw(doc);
  const bad: string[] = [];

  if (raw.steps !== doc.steps) bad.push(`${name}: raw.steps ${raw.steps} != ${doc.steps}`);
  const dec = decodeRecording(raw, decoder);

  if (dec.frames.length !== doc.steps + 1) {
    bad.push(`${name}: frames ${dec.frames.length} != ${doc.steps + 1}`);
    return bad;
  }
  if (dec.events.length !== dec.frames.length) {
    bad.push(`${name}: events ${dec.events.length} != frames ${dec.frames.length}`);
  }

  const approx = new Set(doc.approxFields);
  const cmp = (label: string, got: number, want: number, field: string): void => {
    const ok = approx.has(field) ? Math.abs(got - want) <= APPROX_TOL : same(got, want);
    if (!ok) bad.push(`${label} ${field}: got ${got} want ${want}`);
  };

  const checkFrame = (
    label: string,
    frame: EnvFrame<S>,
    step: number,
    done: number,
    pose: number[],
    state: Record<string, number>,
  ): void => {
    if (frame.step !== step) bad.push(`${label}: step ${frame.step} != ${step}`);
    if (!same(frame.done, done)) bad.push(`${label}: done ${frame.done} != ${done}`);
    if (frame.pose.length !== 9) bad.push(`${label}: pose length ${frame.pose.length} != 9`);
    for (let j = 0; j < pose.length; j++) {
      cmp(label, frame.pose[j] ?? NaN, pose[j] ?? NaN, `pose[${j}]`);
    }
    const got = frame.state as unknown as Record<string, number>;
    const wantKeys = Object.keys(state);
    for (const k of wantKeys) cmp(label, got[k] ?? NaN, state[k] ?? NaN, k);
    for (const k of wantKeys) {
      if (!(k in got)) bad.push(`${label}: state is missing field ${k}`);
    }
  };

  const init = dec.frames[0];
  if (!init) {
    bad.push(`${name}: no init frame`);
    return bad;
  }
  checkFrame(`${name} init`, init, doc.init.step, doc.init.done, doc.init.pose, doc.init.state);
  if (init.ctrl.length !== 6 || init.ctrl.some((v) => v !== 0)) {
    bad.push(`${name} init: ctrl is not six zeros (${Array.from(init.ctrl).join(',')})`);
  }
  if ((dec.events[0] ?? []).length !== 0) bad.push(`${name}: events[0] is not empty`);

  for (let i = 0; i < doc.rows.length; i++) {
    const want = doc.rows[i];
    const frame = dec.frames[i + 1];
    if (!want || !frame) {
      bad.push(`${name}: missing row/frame ${i}`);
      continue;
    }
    checkFrame(`${name} row ${i}`, frame, want.step, want.done, want.pose, want.state);
  }

  for (let i = 0; i < doc.ctrlHead.length; i++) {
    const want = doc.ctrlHead[i];
    const frame = dec.frames[i + 1];
    if (!want || !frame) continue;
    if (frame.ctrl.length !== 6) bad.push(`${name} row ${i}: ctrl length ${frame.ctrl.length}`);
    for (let u = 0; u < want.length; u++) {
      cmp(`${name} row ${i}`, frame.ctrl[u] ?? NaN, want[u] ?? NaN, `ctrl[${u}]`);
    }
  }

  const fired = dec.events
    .map((e, i) => (e.length > 0 ? i : -1))
    .filter((i) => i >= 0);
  const expectFired = doc.doneRows.map((r) => r + 1);
  if (fired.join(',') !== expectFired.join(',')) {
    bad.push(`${name}: event frames [${fired.join(',')}] != [${expectFired.join(',')}]`);
  }
  for (const r of doc.doneRows) {
    const got = eventTypes(dec.events[r + 1] ?? []);
    if (got.join(',') !== doc.doneEvents.join(',')) {
      bad.push(`${name} row ${r}: events [${got.join(',')}] != [${doc.doneEvents.join(',')}]`);
    }
  }

  return bad;
}

const HEAD_CASES: [string, () => string[]][] = [
  ['HalfCheetah-v4', () => checkTrace(halfCheetahDecoder, 'HalfCheetah-v4')],
  ['Acrobot-cont-v0', () => checkTrace(acrobotDecoder, 'Acrobot-cont-v0')],
  ['MountainCar-cont-v0', () => checkTrace(mountainCarDecoder, 'MountainCar-cont-v0')],
  ['CartPole-cont-v0', () => checkTrace(cartPoleDecoder, 'CartPole-cont-v0')],
  ['Pendulum-cont-v0', () => checkTrace(pendulumDecoder, 'Pendulum-cont-v0')],
];

const EVENT_CASES: [string, () => string[]][] = [
  ['Acrobot-cont-v0.events', () => checkTrace(acrobotDecoder, 'Acrobot-cont-v0.events')],
  ['MountainCar-cont-v0.events', () => checkTrace(mountainCarDecoder, 'MountainCar-cont-v0.events')],
];

describe('decodeRecording — trace goldens', () => {
  for (const [name, run] of [...HEAD_CASES, ...EVENT_CASES]) {
    test(name, () => {
      const bad = run();
      expect(bad.slice(0, 12)).toEqual([]);
      expect(bad.length).toBe(0);
    });
  }
});

describe('trace fixtures carry their fp64 certification', () => {
  for (const name of ['Acrobot-cont-v0', 'MountainCar-cont-v0', 'CartPole-cont-v0',
    'Pendulum-cont-v0']) {
    test(`${name} one-step parity`, () => {
      const doc = loadDoc(name);
      expect(doc.oneStepParity).not.toBeNull();
      const p = doc.oneStepParity;
      if (!p) return;
      expect(p.steps).toBe(doc.steps);
      expect(p.doneMismatch).toBe(0);
      expect(p.maxAbsErr).toBeLessThan(1e-3);
    });
  }
});

describe('decode driver contract', () => {
  test('rejects a decoder/recording envId mismatch', () => {
    const doc = loadDoc('CartPole-cont-v0');
    const raw = loadRaw(doc);
    expect(() => decodeRecording(raw, pendulumDecoder)).toThrow(DecodeError);
  });

  test('rejects a recording that claims more steps than it holds', () => {
    const raw = rawRecordingFromFloats(
      'CartPole-cont-v0',
      48,
      new Float32Array(RECORD_FLOATS * 2),
    );
    expect(() => decodeRecording({ ...raw, steps: 5 }, cartPoleDecoder)).toThrow(DecodeError);
  });

  test('lensRecording serves the same frames and events as decodeRecording', () => {
    const doc = loadDoc('MountainCar-cont-v0');
    const raw = loadRaw(doc);
    const dec = decodeRecording(raw, mountainCarDecoder);
    const lens = lensRecording(raw, mountainCarDecoder);
    expect(lens.frameCount).toBe(dec.frames.length);
    expect(lens.steps).toBe(doc.steps);
    for (let i = 0; i < dec.frames.length; i++) {
      expect({ i, step: lens.frameAt(i).step }).toEqual({ i, step: dec.frames[i]?.step ?? NaN });
      expect({ i, n: lens.eventsAt(i).length }).toEqual({ i, n: dec.events[i]?.length ?? -1 });
    }
    expect(lens.frameAt(0).step).toBe(-1);
    expect(lens.frameAt(doc.steps).step).toBe(doc.steps - 1);
  });

  test('eventTimeline flattens to the done rows', () => {
    const doc = loadDoc('CartPole-cont-v0');
    const dec = decodeRecording(loadRaw(doc), cartPoleDecoder);
    const tl = eventTimeline(dec);
    expect(tl.length).toBe(doc.doneRows.length * doc.doneEvents.length);
    expect(tl.map((e) => e.step)).toEqual(
      doc.doneRows.flatMap((r) => doc.doneEvents.map(() => r)),
    );
    expect(tl.map((e): string => e.event.type)).toEqual(
      doc.doneRows.flatMap(() => doc.doneEvents),
    );
  });

  test('geometry is a stable singleton per decoder', () => {
    const doc = loadDoc('Acrobot-cont-v0');
    const raw = loadRaw(doc);
    const a = decodeRecording(raw, acrobotDecoder);
    const b = decodeRecording(raw, acrobotDecoder);
    expect(a.geometry).toBe(b.geometry);
    expect(a.geometry.kind).toBe('classic');
  });

  test('controlHz / discretePose come from the env table', () => {
    const doc = loadDoc('HalfCheetah-v4');
    const dec = decodeRecording(loadRaw(doc), halfCheetahDecoder);
    expect(dec.envId).toBe('HalfCheetah-v4');
    expect(dec.seed).toBe(doc.seed);
    expect(dec.controlHz).toBe(20);
    expect(dec.discretePose).toBe(false);
  });
});

describe('parseRawTrace accepts buffer VIEWS, not just bare ArrayBuffers', () => {
  const META = { envId: 'CartPole-cont-v0', seed: 48 };

  function reference(): Float32Array {
    const doc = loadDoc('CartPole-cont-v0');
    return loadRaw(doc).data;
  }

  test('a readFileSync Buffer decodes to the same floats as its slice', () => {
    const doc = loadDoc('CartPole-cont-v0');
    const b = readFileSync(new URL(`./fixtures/traces/${doc.bin}`, import.meta.url));
    const pooled = parseRawTrace(b, META);
    const want = reference();
    expect(pooled.steps).toBe(doc.steps);
    expect(pooled.data.length).toBe(want.length);
    expect(Array.from(pooled.data)).toEqual(Array.from(want));
  });

  test('an offset view over a padded buffer reads only its own window', () => {
    const want = reference();
    const bytes = want.length * 4;
    for (const pad of [8, 64, 2, 6]) {
      const padded = new ArrayBuffer(pad + bytes + 32);
      new Uint8Array(padded).fill(0xab);
      new Uint8Array(padded, pad, bytes).set(
        new Uint8Array(want.buffer, want.byteOffset, bytes),
      );
      const view = new Uint8Array(padded, pad, bytes);
      const got = parseRawTrace(view, META);
      expect({ pad, steps: got.steps }).toEqual({ pad, steps: want.length / RECORD_FLOATS });
      expect({ pad, data: Array.from(got.data) }).toEqual({ pad, data: Array.from(want) });
    }
  });

  test('a DataView and a Float32Array view work the same way', () => {
    const want = reference();
    const bytes = want.length * 4;
    const padded = new ArrayBuffer(16 + bytes);
    new Uint8Array(padded, 16, bytes).set(new Uint8Array(want.buffer, want.byteOffset, bytes));
    const asDataView = parseRawTrace(new DataView(padded, 16, bytes), META);
    const asF32 = parseRawTrace(new Float32Array(padded, 16, want.length), META);
    expect(Array.from(asDataView.data)).toEqual(Array.from(want));
    expect(Array.from(asF32.data)).toEqual(Array.from(want));
  });

  test('the row-alignment check uses the VIEW length, not the whole buffer', () => {
    const padded = new ArrayBuffer(RECORD_BYTES * 4);
    const view = new Uint8Array(padded, 0, RECORD_BYTES * 3 + 8);
    expect(() => parseRawTrace(view, META)).toThrow(RecordParseError);
  });
});

describe('unvalidated envIds are rejected with a typed error', () => {
  const bogus = 'NotAnEnv-v9' as EnvId;

  test('rawRecordingFromFloats refuses an unknown envId', () => {
    expect(() => rawRecordingFromFloats(bogus, 0, new Float32Array(RECORD_FLOATS)))
      .toThrow(RecordParseError);
    expect(() => rawRecordingFromFloats(bogus, 0, new Float32Array(RECORD_FLOATS)))
      .toThrow(/NotAnEnv-v9/);
  });

  test('decodeRecording refuses an unknown envId instead of TypeErroring on ENVS', () => {
    const rogue = { ...cartPoleDecoder, envId: bogus } as EnvDecoder<unknown, unknown>;
    const raw: RawRecording = {
      envId: bogus,
      seed: 0,
      data: new Float32Array(RECORD_FLOATS * 2),
      steps: 2,
    };
    expect(() => decodeRecording(raw, rogue)).toThrow(DecodeError);
    expect(() => decodeRecording(raw, rogue)).toThrow(/NotAnEnv-v9/);
  });
});

describe('the empty event list is one shared frozen singleton', () => {
  test('NO_EVENTS is frozen and empty', () => {
    expect(NO_EVENTS.length).toBe(0);
    expect(Object.isFrozen(NO_EVENTS)).toBe(true);
    expect(() => (NO_EVENTS as SimEvent[]).push({ type: 'goal' })).toThrow();
    expect(NO_EVENTS.length).toBe(0);
  });

  for (const envId of ENV_IDS) {
    test(`${envId} returns the singleton for quiet frames`, () => {
      const doc = JSON.parse(
        readFileSync(new URL(`./fixtures/traces/${envId}.json`, import.meta.url), 'utf8'),
      ) as { seed: number };
      const b = readFileSync(new URL(`./fixtures/traces/${envId}.bin`, import.meta.url));
      const raw = parseRawTrace(b, { envId, seed: doc.seed });
      const dec = decodeRecording(raw, decoderFor(envId));
      let quiet = 0;
      for (let i = 0; i < dec.events.length; i++) {
        const evs = dec.events[i];
        if (evs && evs.length === 0) {
          quiet++;
          expect({ i, singleton: evs === NO_EVENTS }).toEqual({ i, singleton: true });
        }
      }
      expect(quiet).toBeGreaterThan(0);
    });
  }
});

describe('non-finite record rows abort the decode', () => {
  function cartPoleRaw(): RawRecording {
    const doc = loadDoc('CartPole-cont-v0');
    const raw = loadRaw(doc);
    return { ...raw, data: Float32Array.from(raw.data) };
  }

  test('a clean recording decodes every row', () => {
    const raw = cartPoleRaw();
    const dec = decodeRecording(raw, cartPoleDecoder);
    expect(dec.frames.length).toBe(raw.steps + 1);
  });

  test('NaN / Infinity anywhere in a row throws DecodeError', () => {
    for (const [row, slot, value] of [
      [3, 0, NaN],
      [9, 6, Infinity],
      [9, 8, NaN],
      [20, 15, -Infinity],
    ] as const) {
      const raw = cartPoleRaw();
      raw.data[row * RECORD_FLOATS + slot] = value;
      expect(() => decodeRecording(raw, cartPoleDecoder)).toThrow(DecodeError);
      expect(() => decodeRecording(raw, cartPoleDecoder)).toThrow(new RegExp(`row ${row} `));
    }
  });

  test('lensRecording rejects the damaged row too', () => {
    const raw = cartPoleRaw();
    raw.data[30 * RECORD_FLOATS] = NaN;
    expect(() => lensRecording(raw, cartPoleDecoder)).toThrow(DecodeError);
    expect(() => lensRecording(raw, cartPoleDecoder)).toThrow(/row 30 /);
  });
});
