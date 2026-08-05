/// <reference types="bun" />

import { describe, expect, it } from 'bun:test';
import { statOrNull } from '../../server/jsonl';
import { createApp } from '../../server/app';

const BENCH = '/Users/studio/OaKMeta/.oak-benchmarks';
const hasFloors = (await statOrNull(`${BENCH}/FLOORS.json`)) !== null;

describe('/api/floors', () => {
  it.skipIf(!hasFloors)('serves FLOORS.json verbatim with the schema fields', async () => {
    const app = createApp({ benchRoot: BENCH, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    const res = await app.fetch(new Request('http://x/api/floors'));
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      floorsSchemaVersion: number;
      sourceRun: string;
      floors: Record<string, { zero: number | null; random: number | null; floor: number }>;
    };
    expect(body.floorsSchemaVersion).toBe(1);
    expect(Object.keys(body.floors).length).toBeGreaterThanOrEqual(16);
    for (const f of Object.values(body.floors)) {
      const candidates = [f.zero, f.random].filter((v): v is number => v !== null);
      expect(f.floor).toBe(Math.max(...candidates));
    }
  });

  it('404s when no floors have been extracted', async () => {
    const app = createApp({
      benchRoot: '/tmp/definitely-not-a-bench-root',
      cacheFile: null,
      fleetCacheDir: null,
      ttlMs: 0,
    });
    const res = await app.fetch(new Request('http://x/api/floors'));
    expect(res.status).toBe(404);
  });
});
