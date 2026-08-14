/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import type { EnvId } from '@shared/envs';
import { DecodeError, decodeRecording } from '@sim/decode';
import { RECORD_FLOATS } from '@sim/record';
import { parseRawTrace } from '../helpers/rawTrace';
import type { DecodedRecording, GridCell, RawRecording } from '@sim/types';
import {
  MKC_GOAL,
  MKC_KEYS,
  MKC_ROOM_CELLS,
  MKC_START,
  mkcIsWall,
  mkcRoomCells,
  mkcStep,
  multiKeyCorridorDecoder,
} from '@sim/envs/multikeycorridor';
import type { MultiKeyCorridorState } from '@sim/envs/multikeycorridor';

interface ReplayStep {
  action: number;
  reward: number;
  done: number;
  x: number;
  y: number;
  gx: number;
  gy: number;
  keys: GridCell[];
  flags: number;
  respawnIdx: number;
}

interface ReplayFixture {
  version: number;
  laneSeed: number;
  completions: number;
  steps: ReplayStep[];
}

function fixtureUrl(name: string): URL {
  return new URL(`../fixtures/${name}`, import.meta.url);
}

const replay = JSON.parse(readFileSync(fixtureUrl('mkc_replay.json'), 'utf8')) as ReplayFixture;

const MKC_ENV = 'MultiKeyCorridor-cont-v0' as EnvId;

function mkcRaw(): RawRecording {
  const b = readFileSync(fixtureUrl(`traces/${MKC_ENV}.bin`));
  const raw = parseRawTrace(b, { envId: MKC_ENV, seed: 80 });
  return { ...raw, data: Float32Array.from(raw.data) };
}

function disagreeFrames(dec: DecodedRecording<MultiKeyCorridorState>): number[] {
  const out: number[] = [];
  for (let i = 0; i < dec.frames.length; i++) {
    if (dec.frames[i]?.state.poseAgrees === false) out.push(i);
  }
  return out;
}

describe('mkcStep replay', () => {
  test('fixture exercises real respawns', () => {
    expect(replay.steps.length).toBe(500);
    expect(replay.completions).toBeGreaterThanOrEqual(3);
    expect(replay.steps.some((s) => (s.flags & 4) !== 0)).toBe(true);
  });

  test('500 scripted steps reproduce gridworld_ref exactly, keys included', () => {
    const s = {
      x: MKC_START.x,
      y: MKC_START.y,
      keys: MKC_KEYS.map((k) => ({ ...k })),
      gx: MKC_GOAL.x,
      gy: MKC_GOAL.y,
      flags: 0,
      respawnIdx: 0,
    };
    for (let i = 0; i < replay.steps.length; i++) {
      const want = replay.steps[i]!;
      const done = mkcStep(replay.laneSeed, s, want.action);
      expect({
        i,
        x: s.x,
        y: s.y,
        gx: s.gx,
        gy: s.gy,
        keys: s.keys,
        flags: s.flags,
        respawnIdx: s.respawnIdx,
        done,
      }).toEqual({
        i,
        x: want.x,
        y: want.y,
        gx: want.gx,
        gy: want.gy,
        keys: want.keys,
        flags: want.flags,
        respawnIdx: want.respawnIdx,
        done: want.done,
      });
    }
  });

  test('the replayed respawn draws land in the right rooms', () => {
    const rooms = [0, 1, 2, 3].map(mkcRoomCells);
    const inRoom = (c: GridCell, r: number): boolean =>
      rooms[r]!.some((q) => q.x === c.x && q.y === c.y);
    for (const s of replay.steps) {
      for (let j = 0; j < 3; j++) expect(inRoom(s.keys[j]!, j)).toBe(true);
      expect(inRoom({ x: s.gx, y: s.gy }, 3)).toBe(true);
      expect(mkcIsWall(s.x, s.y)).toBe(false);
    }
  });

  test('a locked door never lets the agent through without its key', () => {
    let prev = { x: MKC_START.x, y: MKC_START.y, flags: 0 };
    for (const s of replay.steps) {
      const doorIdx = [4, 8, 12].indexOf(prev.x);
      if (doorIdx >= 0 && prev.y === 3) {
        expect((prev.flags & (8 << doorIdx)) !== 0).toBe(true);
      }
      const entered = [4, 8, 12].indexOf(s.x);
      if (entered >= 0 && s.y === 3 && s.done === 0) {
        expect((s.flags & (1 << entered)) !== 0).toBe(true);
        expect((s.flags & (8 << entered)) !== 0).toBe(true);
      }
      prev = { x: s.x, y: s.y, flags: s.flags };
    }
  });

  test('room index arithmetic matches the reference cell order', () => {
    expect(mkcRoomCells(0).length).toBe(MKC_ROOM_CELLS);
    expect(mkcRoomCells(0)[0]).toEqual({ x: 1, y: 1 });
    expect(mkcRoomCells(0)[14]).toEqual({ x: 3, y: 5 });
    expect(mkcRoomCells(3)[0]).toEqual({ x: 13, y: 1 });
    expect(mkcRoomCells(3)[14]).toEqual({ x: 15, y: 5 });
  });
});

describe('MultiKeyCorridor decoder', () => {
  test('poseAgrees holds for every frame of the real 200-step lane', () => {
    const envId = 'MultiKeyCorridor-cont-v0';
    const b = readFileSync(fixtureUrl(`traces/${envId}.bin`));
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
    expect(buf.byteLength).toBe(200 * RECORD_FLOATS * 4);
    const raw = parseRawTrace(buf, { envId, seed: 80 });
    const dec = decodeRecording(raw, multiKeyCorridorDecoder);
    expect(dec.frames.length).toBe(201);
    for (let i = 0; i < dec.frames.length; i++) {
      const st = dec.frames[i]!.state;
      expect({ i, agrees: st.poseAgrees }).toEqual({ i, agrees: true });
      if (i > 0) {
        const pose = dec.frames[i]!.pose;
        expect(pose[0]).toBe(st.x);
        expect(pose[1]).toBe(st.y);
        expect(pose[4]).toBe(st.flags & 7);
        expect(pose[5]).toBe((st.flags >> 3) & 7);
      }
    }
  });

  test('a corrupted pose is reported, not silently absorbed', () => {
    const envId = 'MultiKeyCorridor-cont-v0';
    const b = readFileSync(fixtureUrl(`traces/${envId}.bin`));
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
    const raw = parseRawTrace(buf, { envId, seed: 80 });
    const data = Float32Array.from(raw.data);
    data[10 * RECORD_FLOATS + 7] = 99;
    const dec = decodeRecording(
      { envId: envId as EnvId, seed: 80, data, steps: raw.steps },
      multiKeyCorridorDecoder,
    );
    const flags = dec.frames.map((f) => (f.state as MultiKeyCorridorState).poseAgrees);
    expect(flags[11]).toBe(false);
    expect(flags.filter((v) => v === false).length).toBe(1);
  });

  describe('pose disagreement is tracked per frame', () => {
    test('a clean lane reports no disagreement at all', () => {
      const dec = decodeRecording(mkcRaw(), multiKeyCorridorDecoder);
      expect(disagreeFrames(dec)).toEqual([]);
    });

    test('the corrupted pose row shows up as one disagreement at its own frame', () => {
      const raw = mkcRaw();
      raw.data[10 * RECORD_FLOATS + 7] = 99;
      const dec = decodeRecording(raw, multiKeyCorridorDecoder);
      expect(disagreeFrames(dec)).toEqual([11]);
    });

    test('an offset recording desyncs and says so', () => {
      const src = mkcRaw();
      const skip = 20;
      const data = src.data.slice(skip * RECORD_FLOATS);
      const dec = decodeRecording(
        { envId: MKC_ENV, seed: 80, data, steps: src.steps - skip },
        multiKeyCorridorDecoder,
      );
      expect(disagreeFrames(dec).length).toBeGreaterThan(0);
    });

    test('a truncated window only sees the disagreements inside it', () => {
      const raw = mkcRaw();
      raw.data[10 * RECORD_FLOATS + 7] = 99;
      const window = (steps: number) =>
        decodeRecording(
          { ...raw, data: raw.data.subarray(0, steps * RECORD_FLOATS), steps },
          multiKeyCorridorDecoder,
        );
      expect(disagreeFrames(window(10))).toEqual([]);
      expect(disagreeFrames(window(11))).toEqual([11]);
    });
  });

  describe('a non-finite row aborts the decode', () => {
    const BAD_ROW = 10;

    function nanRaw(offset: number): RawRecording {
      const raw = mkcRaw();
      raw.data[BAD_ROW * RECORD_FLOATS + offset] = NaN;
      return raw;
    }

    test('a non-finite ctrl throws DecodeError at the damaged row', () => {
      expect(() => decodeRecording(nanRaw(0), multiKeyCorridorDecoder)).toThrow(DecodeError);
      expect(() => decodeRecording(nanRaw(0), multiKeyCorridorDecoder))
        .toThrow(new RegExp(`row ${BAD_ROW} `));
    });

    test('a non-finite POSE throws too, not just ctrl', () => {
      expect(() => decodeRecording(nanRaw(7), multiKeyCorridorDecoder)).toThrow(DecodeError);
    });

    test('rows before the damage stay decodable when the window excludes it', () => {
      const raw = nanRaw(0);
      const head = {
        ...raw,
        data: raw.data.subarray(0, BAD_ROW * RECORD_FLOATS),
        steps: BAD_ROW,
      };
      const dec = decodeRecording(head, multiKeyCorridorDecoder);
      expect(dec.frames.length).toBe(BAD_ROW + 1);
      expect(disagreeFrames(dec)).toEqual([]);
    });
  });

  test('the init frame is the documented deterministic reset', () => {
    const st = multiKeyCorridorDecoder.initStatic(80);
    const f = multiKeyCorridorDecoder.initFrame(80, st);
    expect(f.state.x).toBe(1);
    expect(f.state.y).toBe(1);
    expect(f.state.gx).toBe(15);
    expect(f.state.gy).toBe(1);
    expect(f.state.keys).toEqual([
      { x: 3, y: 5 },
      { x: 7, y: 5 },
      { x: 11, y: 5 },
    ]);
    expect(f.state.hasKey).toEqual([false, false, false]);
    expect(f.state.doorOpen).toEqual([false, false, false]);
    expect(f.state.respawnIdx).toBe(0);
    expect(Array.from(f.pose)).toEqual([1, 1, 15, 1, 0, 0, 0, 0, 0]);
  });
});
