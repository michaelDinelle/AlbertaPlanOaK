import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import index from './index.html';
import { ENGINE_FILES, JUDGE_FILE, packRecording, type Baselines, type Meta, type RunIndexEntry } from './src/data/layout';

const BENCH = resolve(import.meta.dir, '../.oak-benchmarks');
const PUBLIC = resolve(import.meta.dir, 'public');
const BEST = resolve(import.meta.dir, '../best');
const ENGINE = resolve(import.meta.dir, '../.build/oak');

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const bytes = (path: string): ArrayBuffer => {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

const META_FIELDS = ['seeds', 'steps', 'batch', 'window', 'policy', 'preset', 'startedAt', 'envSchedule'] as const;

function meta(root: string): Meta {
  const m = JSON.parse(readFileSync(join(root, ENGINE_FILES.meta), 'utf8')) as Meta;
  for (const field of META_FIELDS) if (!(field in m)) throw new HttpError(500, `${root}/${ENGINE_FILES.meta} lacks ${field}`);
  return m;
}

const recorded = (root: string): boolean => Object.values(ENGINE_FILES).every((file) => existsSync(join(root, file)));

function runRoot(id: string): string {
  if (id === '' || id.includes('/') || id.startsWith('.')) throw new HttpError(400, `bad run id ${id}`);
  const root = join(BENCH, id);
  if (!recorded(root)) throw new HttpError(404, `unknown run ${id}`);
  return root;
}

function runs(): RunIndexEntry[] {
  const best = existsSync(BEST) ? readFileSync(BEST, 'utf8').trim() : '';
  return readdirSync(BENCH)
    .filter((id) => !id.startsWith('.') && recorded(join(BENCH, id)))
    .map((id): RunIndexEntry => {
      const m = meta(join(BENCH, id));
      return {
        id,
        envIds: m.envSchedule.map((x) => x.envId),
        steps: m.steps,
        startedAt: m.startedAt,
        inFlight: m.finishedAt === undefined,
        policy: m.policy,
        preset: m.preset,
        role: id === best ? 'best' : null,
      };
    })
    .sort((a, b) => a.startedAt !== b.startedAt ? (a.startedAt > b.startedAt ? -1 : 1) : a.id > b.id ? -1 : 1);
}

function snapshot(root: string): ArrayBuffer {
  const m = meta(root);
  const baselines = existsSync(join(root, JUDGE_FILE)) ? (JSON.parse(readFileSync(join(root, JUDGE_FILE), 'utf8')) as Baselines) : {};
  const ends = new Uint32Array(bytes(join(root, ENGINE_FILES.ends)));
  const rewardSum = new Float32Array(bytes(join(root, ENGINE_FILES.rewardSum)));
  const hash = new BigUint64Array(bytes(join(root, ENGINE_FILES.hash)));
  const health = new Uint32Array(bytes(join(root, ENGINE_FILES.health)));
  return packRecording(m, baselines, ends, rewardSum, hash, health);
}

function replay(id: string, lane: string): Response {
  if (!/^\d+$/.test(lane)) throw new HttpError(400, `bad lane ${lane}`);
  const root = runRoot(id);
  const rows = join(root, 'replay', `${lane}.f32`);
  const verdict = join(root, 'replay', `${lane}.json`);
  if (!existsSync(rows) || !existsSync(verdict)) {
    mkdirSync(join(root, 'replay'), { recursive: true });
    const run = Bun.spawnSync([ENGINE, 'replay', root, lane], { cwd: resolve(import.meta.dir, '..') });
    if (run.exitCode !== 0 && run.exitCode !== 2) throw new HttpError(500, `replay failed: ${run.stderr.toString()}`);
    writeFileSync(verdict, JSON.stringify({ identical: run.exitCode === 0 }));
  }
  const identical = (JSON.parse(readFileSync(verdict, 'utf8')) as { identical: boolean }).identical;
  return new Response(Bun.file(rows), { headers: { 'X-Oak-Identical': identical ? '1' : '0' } });
}

const workerBuild = await Bun.build({
  entrypoints: [resolve(import.meta.dir, 'src/scenes/jelly-bean-world/scentfield.worker.ts')],
  target: 'browser',
  minify: true,
});
if (!workerBuild.success) throw new AggregateError(workerBuild.logs, 'worker bundle failed');
const workerOf = (name: string): Blob => {
  const out = workerBuild.outputs.find((o) => o.path.endsWith(`/${name}.js`));
  if (out === undefined) throw new Error(`worker bundle lacks ${name}`);
  return out;
};
const scentfieldWorker = workerOf('scentfield.worker');

async function publicFile(req: Request): Promise<Response> {
  const rel = decodeURIComponent(new URL(req.url).pathname).split('/').filter((s) => s !== '' && s !== '..').join('/');
  const file = Bun.file(join(PUBLIC, rel));
  if (!(await file.exists())) throw new HttpError(404, `no file ${rel}`);
  return new Response(file);
}

const server = Bun.serve({
  development: { hmr: true, console: true },
  routes: {
    '/': index,
    '/api/runs': () => Response.json(runs()),
    '/api/runs/:id': (req) => new Response(snapshot(runRoot(req.params.id))),
    '/api/runs/:id/replay/:lane': (req) => replay(req.params.id, req.params.lane),
    '/scentfield.worker.js': () => new Response(scentfieldWorker),
    '/posters/*': publicFile,
    '/terrain/*': publicFile,
    '/textures/*': publicFile,
  },
  fetch: () => new Response('unknown endpoint', { status: 404 }),
  error: (error) => new Response(String(error), { status: error instanceof HttpError ? error.status : 500 }),
});
console.log(`serving ${BENCH} on ${server.url}`);
