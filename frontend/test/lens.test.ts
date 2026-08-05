/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { decodeRecording, lensRecording, NO_EVENTS, LENS_BLOCK } from '../src/sim/decode';
import { parseActionsBin } from '../src/sim/record';
import { decoders } from '../src/sim/registry';
import type { RecordingMeta } from '@shared/types';

const ROOT = '/Users/studio/OaKMeta/.oak-benchmarks/frontend-demo-16env-20260803';

function load(envId: string, seed: number) {
  const dir = `${ROOT}/${envId}/seed${seed}`;
  const meta = JSON.parse(readFileSync(`${dir}/meta.json`, 'utf8')) as RecordingMeta;
  const buf = readFileSync(`${dir}/actions.bin`);
  const raw = parseActionsBin(buf, meta);
  const decoder = decoders[meta.envId as keyof typeof decoders]!;
  return { raw, decoder };
}

const CASES: [string, number][] = [
  ['HalfCheetah-v4', 0],
  ['MultiKeyCorridor-cont-v0', 80],
  ['DeepSea-cont-v0', 16],
];

describe('RecordingLens ≡ decodeRecording', () => {
  for (const [envId, seed] of CASES) {
    test(`${envId} seed ${seed}: frames + events identical, random access order`, () => {
      const { raw, decoder } = load(envId, seed);
      const full = decodeRecording(raw, decoder);
      const lens = lensRecording(raw, decoder);
      expect(lens.frameCount).toBe(full.frames.length);
      expect(lens.nonFiniteRows).toBe(full.nonFiniteRows ?? 0);
      expect(lens.replay ?? null).toEqual(full.replay ?? null);

      const probes = [
        lens.frameCount - 1, 0, 1,
        LENS_BLOCK, LENS_BLOCK - 1, LENS_BLOCK + 1,
        7 * LENS_BLOCK + 3, 2 * LENS_BLOCK,
        Math.floor(lens.frameCount / 2),
      ].filter((i) => i >= 0 && i < lens.frameCount);
      for (const i of probes) {
        const a = lens.frameAt(i);
        const b = full.frames[i]!;
        expect(a.step).toBe(b.step);
        expect([...a.pose]).toEqual([...b.pose]);
        expect([...a.ctrl]).toEqual([...b.ctrl]);
        expect(a.done).toBe(b.done);
        expect(JSON.stringify(a.state)).toBe(JSON.stringify(b.state));
      }

      let eventFrames = 0;
      for (const [i, evs] of lens.eventEntries()) {
        eventFrames++;
        expect(evs).toEqual(full.events[i]!);
      }
      let denseEventFrames = 0;
      for (let i = 0; i < full.events.length; i++) {
        if (full.events[i]! !== NO_EVENTS && full.events[i]!.length > 0) denseEventFrames++;
      }
      expect(eventFrames).toBe(denseEventFrames);
      for (let k = 0; k < 1000; k++) {
        const i = (k * 97 + 13) % lens.frameCount;
        expect(lens.eventsAt(i)).toEqual(full.events[i] ?? NO_EVENTS);
      }
    });
  }
});
