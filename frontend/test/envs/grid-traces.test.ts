/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import type { EnvId } from '@shared/envs';
import { decodeRecording } from '@sim/decode';
import { RECORD_FLOATS, parseActionsBin, rawRecordingFromFloats } from '@sim/record';
import type { EnvDecoder, SimEvent } from '@sim/types';
import { fourRoomsDecoder } from '@sim/envs/fourrooms';
import { deepSeaDecoder } from '@sim/envs/deepsea';
import { largeDeepSeaDecoder } from '@sim/envs/largedeepsea';
import { keyCorridorDecoder } from '@sim/envs/keycorridor';
import { multiKeyCorridorDecoder } from '@sim/envs/multikeycorridor';
import { chainDecoder } from '@sim/envs/chain';
import { memoryChainDecoder } from '@sim/envs/memorychain';
import { delayedChainDecoder } from '@sim/envs/delayedchain';
import { proceduralMazeDecoder } from '@sim/envs/proceduralmaze';
import { highDimMazeDecoder } from '@sim/envs/highdimmaze';
import { poMazeDecoder } from '@sim/envs/pomaze';

const DECODERS: Record<string, EnvDecoder<unknown, unknown>> = {
  'FourRooms-cont-v0': fourRoomsDecoder,
  'DeepSea-cont-v0': deepSeaDecoder,
  'LargeDeepSea-cont-v0': largeDeepSeaDecoder,
  'KeyCorridor-cont-v0': keyCorridorDecoder,
  'MultiKeyCorridor-cont-v0': multiKeyCorridorDecoder,
  'Chain-cont-v0': chainDecoder,
  'MemoryChain-cont-v0': memoryChainDecoder,
  'DelayedChain-cont-v0': delayedChainDecoder,
  'ProceduralMaze-cont-v0': proceduralMazeDecoder,
  'HighDimMaze-cont-v0': highDimMazeDecoder,
  'POMaze-cont-v0': poMazeDecoder,
};

const GRID_ENVS = Object.keys(DECODERS);

interface TraceFixture {
  version: number;
  envId: string;
  seed: number;
  steps: number;
  eventCounts: Record<string, number>;
  rewardSum: number;
  frames: Record<string, unknown>[];
  events: SimEvent[][];
  rewards: number[];
}

interface EventCase {
  envId: string;
  seed: number;
  label: string;
  steps: number;
  eventCounts: Record<string, number>;
  rows: number[][];
  frames: Record<string, unknown>[];
  events: SimEvent[][];
}

function fixtureUrl(name: string): URL {
  return new URL(`../fixtures/${name}`, import.meta.url);
}

function loadJson<T>(name: string): T {
  return JSON.parse(readFileSync(fixtureUrl(name), 'utf8')) as T;
}

function loadBin(name: string): ArrayBuffer {
  const b = readFileSync(fixtureUrl(name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

function decoderFor(envId: string): EnvDecoder<unknown, unknown> {
  const d = DECODERS[envId];
  if (!d) throw new Error(`no decoder for ${envId}`);
  return d;
}

function expectGolden(
  envId: string,
  seed: number,
  data: Float32Array,
  frames: Record<string, unknown>[],
  events: SimEvent[][],
): void {
  const raw = rawRecordingFromFloats(envId as EnvId, seed, data);
  const dec = decodeRecording(raw, decoderFor(envId));
  expect(dec.frames.length).toBe(frames.length);
  expect(dec.events.length).toBe(events.length);
  expect(dec.frames[0]?.step).toBe(-1);
  expect(dec.events[0]).toEqual([]);
  for (let i = 0; i < frames.length; i++) {
    expect({ i, state: dec.frames[i]?.state }).toEqual({ i, state: frames[i] });
    expect({ i, events: dec.events[i] }).toEqual({ i, events: events[i] ?? [] });
    expect(dec.frames[i]?.step).toBe(i - 1);
  }
}

describe('real fleet-lane traces (200 steps each)', () => {
  for (const envId of GRID_ENVS) {
    const fx = loadJson<TraceFixture>(`traces/${envId}.json`);

    test(`${envId}: 200 rows decode to the reference state stream`, () => {
      expect(fx.envId).toBe(envId);
      expect(fx.steps).toBe(200);
      const buf = loadBin(`traces/${envId}.bin`);
      expect(buf.byteLength).toBe(200 * RECORD_FLOATS * 4);
      const raw = parseActionsBin(buf, {
        envId,
        seed: fx.seed,
        actDim: 6,
        recordVersion: 3,
      });
      expect(raw.steps).toBe(200);
      expectGolden(envId, fx.seed, raw.data, fx.frames, fx.events);
    });

    test(`${envId}: event derivation matches the reference tally`, () => {
      const buf = loadBin(`traces/${envId}.bin`);
      const raw = parseActionsBin(buf, { envId, seed: fx.seed, actDim: 6, recordVersion: 3 });
      const dec = decodeRecording(raw, decoderFor(envId));
      const counts: Record<string, number> = {};
      for (const evs of dec.events) for (const e of evs) counts[e.type] = (counts[e.type] ?? 0) + 1;
      expect(counts).toEqual(fx.eventCounts);
    });
  }

  test('maxSteps decodes a prefix of the same stream', () => {
    const envId = 'Chain-cont-v0';
    const fx = loadJson<TraceFixture>(`traces/${envId}.json`);
    const raw = parseActionsBin(loadBin(`traces/${envId}.bin`), {
      envId,
      seed: fx.seed,
      actDim: 6,
      recordVersion: 3,
    });
    const dec = decodeRecording(raw, decoderFor(envId), { maxSteps: 50 });
    expect(dec.frames.length).toBe(51);
    for (let i = 0; i < 51; i++) expect(dec.frames[i]?.state).toEqual(fx.frames[i]);
  });
});

describe('scripted event cases (every event type, every env)', () => {
  const fx = loadJson<{ cases: EventCase[] }>('grid_event_cases.json');

  test('covers all 11 gridworlds', () => {
    expect(fx.cases.map((c) => c.envId).sort()).toEqual([...GRID_ENVS].sort());
  });

  for (const c of fx.cases) {
    test(`${c.envId}: ${c.label}`, () => {
      const data = new Float32Array(c.rows.length * RECORD_FLOATS);
      for (let i = 0; i < c.rows.length; i++) data.set(c.rows[i] ?? [], i * RECORD_FLOATS);
      expectGolden(c.envId, c.seed, data, c.frames, c.events);
    });
  }

  test('goal / teleport / done fire on every respawning env', () => {
    for (const c of fx.cases) {
      expect(c.eventCounts['goal'] ?? 0).toBeGreaterThan(0);
    }
    for (const envId of [
      'KeyCorridor-cont-v0',
      'MultiKeyCorridor-cont-v0',
      'ProceduralMaze-cont-v0',
      'HighDimMaze-cont-v0',
      'POMaze-cont-v0',
      'Chain-cont-v0',
      'DelayedChain-cont-v0',
      'MemoryChain-cont-v0',
    ]) {
      const c = fx.cases.find((x) => x.envId === envId)!;
      expect(c.eventCounts['teleport'] ?? 0).toBeGreaterThan(0);
      expect(c.eventCounts['done'] ?? 0).toBeGreaterThan(0);
    }
  });

  test('key + door fire on both corridor envs', () => {
    for (const envId of ['KeyCorridor-cont-v0', 'MultiKeyCorridor-cont-v0']) {
      const c = fx.cases.find((x) => x.envId === envId)!;
      expect(c.eventCounts['key'] ?? 0).toBeGreaterThan(0);
      expect(c.eventCounts['door'] ?? 0).toBeGreaterThan(0);
    }
  });

  test('resurface fires on the DeepSea family, and only there', () => {
    for (const c of fx.cases) {
      const want = c.envId === 'DeepSea-cont-v0' || c.envId === 'LargeDeepSea-cont-v0';
      expect((c.eventCounts['resurface'] ?? 0) > 0).toBe(want);
    }
  });

  test('MemoryChain: a wrong choice is done + cue without goal', () => {
    const c = fx.cases.find((x) => x.envId === 'MemoryChain-cont-v0')!;
    expect(c.eventCounts['done']).toBe(4);
    expect(c.eventCounts['cue']).toBe(4);
    expect(c.eventCounts['goal']).toBe(2);
    const data = new Float32Array(c.rows.length * RECORD_FLOATS);
    for (let i = 0; i < c.rows.length; i++) data.set(c.rows[i] ?? [], i * RECORD_FLOATS);
    const dec = decodeRecording(
      rawRecordingFromFloats(c.envId as EnvId, c.seed, data),
      decoderFor(c.envId),
    );
    const wrong = dec.events.filter(
      (evs) => evs.some((e) => e.type === 'done') && !evs.some((e) => e.type === 'goal'),
    );
    expect(wrong.length).toBe(2);
    for (const evs of wrong) expect(evs.some((e) => e.type === 'cue')).toBe(true);
  });

  test('DeepSea: a failed dive resurfaces without a goal', () => {
    for (const envId of ['DeepSea-cont-v0', 'LargeDeepSea-cont-v0']) {
      const c = fx.cases.find((x) => x.envId === envId)!;
      const data = new Float32Array(c.rows.length * RECORD_FLOATS);
      for (let i = 0; i < c.rows.length; i++) data.set(c.rows[i] ?? [], i * RECORD_FLOATS);
      const dec = decodeRecording(
        rawRecordingFromFloats(envId as EnvId, c.seed, data),
        decoderFor(envId),
      );
      const surfaces = dec.events.filter((evs) => evs.some((e) => e.type === 'resurface'));
      const dry = surfaces.filter((evs) => !evs.some((e) => e.type === 'goal'));
      expect(surfaces.length).toBe(c.eventCounts['resurface'] ?? 0);
      expect(dry.length).toBe(1);
      const wet = surfaces.find((evs) => evs.some((e) => e.type === 'goal'))!;
      expect(wet[0]).toEqual({ type: 'goal' });
      expect(wet[1]).toEqual({ type: 'resurface' });
    }
  });
});

describe('init frames', () => {
  test('every decoder synthesizes step -1 from the deterministic init', () => {
    const expected: Record<string, Record<string, unknown>> = {};
    for (const envId of GRID_ENVS) {
      expected[envId] = loadJson<TraceFixture>(`traces/${envId}.json`).frames[0] ?? {};
    }
    for (const envId of GRID_ENVS) {
      const d = decoderFor(envId);
      const seed = loadJson<TraceFixture>(`traces/${envId}.json`).seed;
      const f = d.initFrame(seed, d.initStatic(seed));
      expect({ envId, state: f.state }).toEqual({ envId, state: expected[envId] });
      expect(f.step).toBe(-1);
      expect(f.done).toBe(0);
      expect(f.ctrl.length).toBe(6);
      expect(f.pose.length).toBe(9);
      expect(Array.from(f.ctrl)).toEqual([0, 0, 0, 0, 0, 0]);
    }
  });

  test('the init pose uses the same packing as the record rows', () => {
    for (const envId of GRID_ENVS) {
      const fx = loadJson<TraceFixture>(`traces/${envId}.json`);
      const d = decoderFor(envId);
      const init = d.initFrame(fx.seed, d.initStatic(fx.seed));
      const raw = parseActionsBin(loadBin(`traces/${envId}.bin`), {
        envId,
        seed: fx.seed,
        actDim: 6,
        recordVersion: 3,
      });
      const dec = decodeRecording(raw, d);
      expect(Array.from(dec.frames[0]?.pose ?? [])).toEqual(Array.from(init.pose));
    }
  });
});
