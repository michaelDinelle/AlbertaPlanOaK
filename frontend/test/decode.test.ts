/// <reference types="bun" />
// End-to-end decode driver tests: every classic-control + HalfCheetah trace
// golden is run through `decodeRecording` and compared, field by field,
// against the Python-generated expectation.
//
// Oracles (test/fixtures/gen_classic.py):
//   * the trace `.bin` is a verbatim cut of a real fleet recording
//     (.oak-benchmarks/frontend-demo-16env-20260803)
//   * the trace `.json` is generated from that same `.bin` by an independent
//     Python decode, and every classic row additionally passed an F1-style
//     fp64 one-step parity check against tools/physics_parity.py before the
//     fixture was written (`oneStepParity` in each file)
// Regenerate with:
//   /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py --only classic
//
// Exactness: pose fields and every state field that is a copy of the pose (or
// exact arithmetic on it) are compared with `===`, which is bit equality for
// float32-sourced doubles. Fields listed in the fixture's `approxFields` are
// transcendental (sin/cos), where numpy's libm and V8's may differ in the last
// ulp; those use a 1e-12 absolute tolerance.

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import type { EnvId } from '@shared/envs';
import { ENV_IDS } from '@shared/envs';
import { DecodeError, decodeRecording, eventTimeline, walkRecording } from '@sim/decode';
import { NO_EVENTS } from '@sim/events';
import { RECORD_BYTES, RECORD_FLOATS, RecordParseError, parseActionsBin, rawRecordingFromFloats }
  from '@sim/record';
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
  recordVersion: number;
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
  return parseActionsBin(ab, {
    envId: doc.envId,
    seed: doc.seed,
    actDim: 6,
    recordVersion: doc.recordVersion,
  });
}

/** float64 equality that treats -0 and 0 as the same value (they carry no meaning here). */
function same(a: number, b: number): boolean {
  return a === b || (Number.isNaN(a) && Number.isNaN(b));
}

function eventTypes(evs: readonly SimEvent[]): string[] {
  return evs.map((e) => e.type);
}

/**
 * Decode one trace fixture and compare every frame against the expectation.
 * Returns the mismatch list (empty = pass) so the caller can assert on it.
 */
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
    // every expected field must exist on the decoded state
    for (const k of wantKeys) {
      if (!(k in got)) bad.push(`${label}: state is missing field ${k}`);
    }
  };

  // frame 0 is the decoder's synthesized init frame
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

  // ctrl views: the first 8 rows are pinned in full
  for (let i = 0; i < doc.ctrlHead.length; i++) {
    const want = doc.ctrlHead[i];
    const frame = dec.frames[i + 1];
    if (!want || !frame) continue;
    if (frame.ctrl.length !== 6) bad.push(`${name} row ${i}: ctrl length ${frame.ctrl.length}`);
    for (let u = 0; u < want.length; u++) {
      cmp(`${name} row ${i}`, frame.ctrl[u] ?? NaN, want[u] ?? NaN, `ctrl[${u}]`);
    }
  }

  // events fire exactly on the done rows, with the env's documented types
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

// windows cut around the first `done` of envs whose first 200 rows have none
const EVENT_CASES: [string, () => string[]][] = [
  ['Acrobot-cont-v0.events', () => checkTrace(acrobotDecoder, 'Acrobot-cont-v0.events')],
  ['MountainCar-cont-v0.events', () => checkTrace(mountainCarDecoder, 'MountainCar-cont-v0.events')],
];

describe('decodeRecording — trace goldens', () => {
  for (const [name, run] of [...HEAD_CASES, ...EVENT_CASES]) {
    test(name, () => {
      const bad = run();
      // slice: a systematic error would otherwise print thousands of lines
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

  test('maxSteps truncates frames and events together', () => {
    const doc = loadDoc('CartPole-cont-v0');
    const raw = loadRaw(doc);
    const dec = decodeRecording(raw, cartPoleDecoder, { maxSteps: 17 });
    expect(dec.frames.length).toBe(18);
    expect(dec.events.length).toBe(18);
    expect(dec.frames[17]?.step).toBe(16);
    // maxSteps 0 still yields the init frame
    expect(decodeRecording(raw, cartPoleDecoder, { maxSteps: 0 }).frames.length).toBe(1);
  });

  test('walkRecording visits the same frames as decodeRecording', () => {
    const doc = loadDoc('MountainCar-cont-v0');
    const raw = loadRaw(doc);
    const dec = decodeRecording(raw, mountainCarDecoder);
    const seen: number[] = [];
    let events = 0;
    walkRecording(raw, mountainCarDecoder, (frame, evs, index) => {
      seen.push(frame.step);
      events += evs.length;
      expect(index).toBe(seen.length - 1);
    });
    expect(seen.length).toBe(dec.frames.length);
    expect(seen[0]).toBe(-1);
    expect(seen[seen.length - 1]).toBe(doc.steps - 1);
    expect(events).toBe(dec.events.reduce((n, e) => n + e.length, 0));
  });

  test('walkRecording stops early on false', () => {
    const doc = loadDoc('Pendulum-cont-v0');
    const raw = loadRaw(doc);
    let n = 0;
    walkRecording(raw, pendulumDecoder, () => {
      n++;
      return n < 5;
    });
    expect(n).toBe(5);
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

// The driver threads `prev` into decodeFrame so replay-dependent envs
// (MultiKeyCorridor) can advance hidden state as they go. None of the five
// envs in this track need that, so it is exercised with a synthetic decoder —
// a grid-shaped stand-in whose state depends on the whole action prefix.
describe('decode driver threads hidden replay state', () => {
  interface ReplayState { pos: number; visits: number }
  interface ReplayStatic { walls: Uint8Array }

  const replayDecoder: EnvDecoder<ReplayState, ReplayStatic> = {
    envId: 'FourRooms-cont-v0',
    needsReplay: true,
    initStatic: () => ({ walls: new Uint8Array([0, 1, 0, 0]) }),
    geometry: (st) => ({
      kind: 'grid',
      grid: { w: 2, h: 2, walls: st.walls, goal: { x: 1, y: 1 } },
    }),
    initFrame: () => ({
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose: new Float32Array(9),
      state: { pos: 0, visits: 0 },
    }),
    decodeFrame: (i, raw16, _st, prev) => {
      const pos = prev.state.pos + (raw16[0] ?? 0);
      const pose = new Float32Array(9);
      pose[0] = pos;
      return {
        step: i,
        ctrl: raw16.subarray(0, 6),
        done: raw16[6] ?? 0,
        pose,
        state: { pos, visits: prev.state.visits + 1 },
      };
    },
    deriveEvents: (prev, next) => (next.state.pos !== prev.state.pos ? [{ type: 'goal' }] : []),
  };

  test('prev-threaded state accumulates across the whole pass', () => {
    const n = 6;
    const data = new Float32Array(RECORD_FLOATS * n);
    for (let i = 0; i < n; i++) data[i * RECORD_FLOATS] = i % 2; // 0,1,0,1,0,1
    const raw = rawRecordingFromFloats('FourRooms-cont-v0', 8, data);
    const dec = decodeRecording(raw, replayDecoder);

    expect(dec.frames.length).toBe(n + 1);
    expect(dec.frames.map((f) => f.state.pos)).toEqual([0, 0, 1, 1, 2, 2, 3]);
    expect(dec.frames[n]?.state.visits).toBe(n);
    expect(dec.geometry.kind).toBe('grid');
    expect(Array.from(dec.geometry.grid?.walls ?? [])).toEqual([0, 1, 0, 0]);
    // events fire only on the transitions that moved
    expect(dec.events.map((e) => e.length)).toEqual([0, 0, 1, 0, 1, 0, 1]);
  });
});

// ---------------------------------------------------------------------------
// Hardening regressions (sim verifier wave 1)
// ---------------------------------------------------------------------------

// `readFileSync` hands back a view into Node/Bun's shared 64 KiB Buffer POOL:
// `.buffer` is that pool, and the file lives at `.byteOffset`. A parser that
// does `new Float32Array(buf)` therefore decodes whatever was pooled before it.
describe('parseActionsBin accepts buffer VIEWS, not just bare ArrayBuffers', () => {
  const META = { envId: 'CartPole-cont-v0', seed: 48, actDim: 6, recordVersion: 3 };

  function reference(): Float32Array {
    const doc = loadDoc('CartPole-cont-v0');
    return loadRaw(doc).data;
  }

  // the shape of the bug: a readFileSync Buffer handed straight in, with no
  // defensive `.buffer.slice(byteOffset, ...)` at the call site
  test('a readFileSync Buffer decodes to the same floats as its slice', () => {
    const doc = loadDoc('CartPole-cont-v0');
    const b = readFileSync(new URL(`./fixtures/traces/${doc.bin}`, import.meta.url));
    const pooled = parseActionsBin(b, META);
    const want = reference();
    expect(pooled.steps).toBe(doc.steps);
    expect(pooled.data.length).toBe(want.length);
    expect(Array.from(pooled.data)).toEqual(Array.from(want));
  });

  test('an offset view over a padded buffer reads only its own window', () => {
    const want = reference();
    const bytes = want.length * 4;
    for (const pad of [8, 64, 2, 6]) {
      // pad 2 / 6 are NOT 4-byte aligned: the parser must copy, not throw
      const padded = new ArrayBuffer(pad + bytes + 32);
      new Uint8Array(padded).fill(0xab); // poison the neighbours
      new Uint8Array(padded, pad, bytes).set(
        new Uint8Array(want.buffer, want.byteOffset, bytes),
      );
      const view = new Uint8Array(padded, pad, bytes);
      const got = parseActionsBin(view, META);
      expect({ pad, steps: got.steps }).toEqual({ pad, steps: want.length / RECORD_FLOATS });
      expect({ pad, data: Array.from(got.data) }).toEqual({ pad, data: Array.from(want) });
    }
  });

  test('a DataView and a Float32Array view work the same way', () => {
    const want = reference();
    const bytes = want.length * 4;
    const padded = new ArrayBuffer(16 + bytes);
    new Uint8Array(padded, 16, bytes).set(new Uint8Array(want.buffer, want.byteOffset, bytes));
    const asDataView = parseActionsBin(new DataView(padded, 16, bytes), META);
    const asF32 = parseActionsBin(new Float32Array(padded, 16, want.length), META);
    expect(Array.from(asDataView.data)).toEqual(Array.from(want));
    expect(Array.from(asF32.data)).toEqual(Array.from(want));
  });

  test('the row-alignment check uses the VIEW length, not the whole buffer', () => {
    const padded = new ArrayBuffer(RECORD_BYTES * 4);
    // a 3.5-row window inside a perfectly row-aligned buffer must be rejected
    const view = new Uint8Array(padded, 0, RECORD_BYTES * 3 + 8);
    expect(() => parseActionsBin(view, META)).toThrow(RecordParseError);
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
    // a decoder that agrees with the bogus id gets past the mismatch check and
    // used to reach `ENVS[envId].controlHz` -> "cannot read properties of undefined"
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

describe('walkRecording honours DecodeOptions', () => {
  test('maxSteps clamps the walk exactly like decodeRecording', () => {
    const doc = loadDoc('MountainCar-cont-v0');
    const raw = loadRaw(doc);
    for (const maxSteps of [0, 1, 7, doc.steps, doc.steps + 50, -3, 4.9]) {
      const dec = decodeRecording(raw, mountainCarDecoder, { maxSteps });
      const seen: number[] = [];
      const evs: number[] = [];
      walkRecording(
        raw,
        mountainCarDecoder,
        (frame, events) => {
          seen.push(frame.step);
          evs.push(events.length);
        },
        { maxSteps },
      );
      expect({ maxSteps, n: seen.length }).toEqual({ maxSteps, n: dec.frames.length });
      expect({ maxSteps, steps: seen }).toEqual({
        maxSteps,
        steps: dec.frames.map((f) => f.step),
      });
      expect({ maxSteps, evs }).toEqual({ maxSteps, evs: dec.events.map((e) => e.length) });
    }
  });

  test('an omitted opts argument still walks the whole recording', () => {
    const doc = loadDoc('Pendulum-cont-v0');
    let n = 0;
    walkRecording(loadRaw(doc), pendulumDecoder, () => {
      n++;
    });
    expect(n).toBe(doc.steps + 1);
  });
});

describe('the empty event list is one shared frozen singleton', () => {
  test('NO_EVENTS is frozen and empty', () => {
    expect(NO_EVENTS.length).toBe(0);
    expect(Object.isFrozen(NO_EVENTS)).toBe(true);
    expect(() => (NO_EVENTS as SimEvent[]).push({ type: 'goal' })).toThrow();
    expect(NO_EVENTS.length).toBe(0);
  });

  // every decoder — a fresh [] per quiet frame is 100k allocations per lane and
  // breaks `events[i] === NO_EVENTS` as a "nothing fired" test
  for (const envId of ENV_IDS) {
    test(`${envId} returns the singleton for quiet frames`, () => {
      const doc = JSON.parse(
        readFileSync(new URL(`./fixtures/traces/${envId}.json`, import.meta.url), 'utf8'),
      ) as { seed: number };
      const b = readFileSync(new URL(`./fixtures/traces/${envId}.bin`, import.meta.url));
      const raw = parseActionsBin(b, { envId, seed: doc.seed, actDim: 6, recordVersion: 3 });
      const dec = decodeRecording(raw, decoderFor(envId));
      let quiet = 0;
      for (let i = 0; i < dec.events.length; i++) {
        const evs = dec.events[i];
        if (evs && evs.length === 0) {
          quiet++;
          expect({ i, singleton: evs === NO_EVENTS }).toEqual({ i, singleton: true });
        }
      }
      expect(quiet).toBeGreaterThan(0); // the assertion above must have bitten
    });
  }
});

describe('non-finite record rows are counted, not silently decoded', () => {
  function cartPoleRaw(): RawRecording {
    const doc = loadDoc('CartPole-cont-v0');
    const raw = loadRaw(doc);
    return { ...raw, data: Float32Array.from(raw.data) };
  }

  test('a clean recording reports zero damaged rows and no replay block', () => {
    const dec = decodeRecording(cartPoleRaw(), cartPoleDecoder);
    expect(dec.nonFiniteRows).toBe(0);
    expect(dec.replay).toBeUndefined();
  });

  test('NaN / Infinity anywhere in a row counts that row once', () => {
    const raw = cartPoleRaw();
    raw.data[3 * RECORD_FLOATS + 0] = NaN; // ctrl[0]
    raw.data[9 * RECORD_FLOATS + 6] = Infinity; // done
    raw.data[9 * RECORD_FLOATS + 8] = NaN; // pose[1] of the SAME row
    raw.data[20 * RECORD_FLOATS + 15] = -Infinity; // pose[8]
    const dec = decodeRecording(raw, cartPoleDecoder);
    expect(dec.nonFiniteRows).toBe(3);
    // a non-replay decoder still decodes every row (per-decoder NaN hardening
    // is deliberately out of scope) — the count is the tripwire
    expect(dec.frames.length).toBe(raw.steps + 1);
  });

  test('the count respects maxSteps', () => {
    const raw = cartPoleRaw();
    raw.data[30 * RECORD_FLOATS] = NaN;
    expect(decodeRecording(raw, cartPoleDecoder, { maxSteps: 30 }).nonFiniteRows).toBe(0);
    expect(decodeRecording(raw, cartPoleDecoder, { maxSteps: 31 }).nonFiniteRows).toBe(1);
  });
});
