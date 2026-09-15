import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import index from './index.html';

const BENCH = resolve(import.meta.dir, '../.oak-benchmarks');
const PUBLIC = resolve(import.meta.dir, 'public');
const BEST = resolve(import.meta.dir, '../best');

interface Meta {
  seeds: number[];
  steps: number;
  batch: number;
  policy: string;
  preset: string;
  startedAt: string;
  finishedAt?: string;
  envSchedule: { envId: string; seedStart: number; seedEnd: number }[];
  envCatalog: unknown;
}

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const bytes = (path: string): ArrayBuffer => {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

const META_FIELDS = ['seeds', 'steps', 'batch', 'policy', 'preset', 'startedAt', 'envSchedule', 'envCatalog'];

function meta(root: string): Meta {
  const m = JSON.parse(readFileSync(join(root, 'meta.json'), 'utf8')) as Meta;
  for (const field of META_FIELDS) if (!(field in m)) throw new HttpError(500, `${root}/meta.json lacks ${field}`);
  return m;
}

const ends = (root: string): Uint32Array => existsSync(join(root, 'ends.u32')) ? new Uint32Array(bytes(join(root, 'ends.u32'))) : new Uint32Array(0);

function runRoot(id: string): string {
  const [name, policy, ...rest] = id.split(':');
  if (!name || rest.length > 0 || name.includes('/') || name.startsWith('.') || (policy !== undefined && policy !== 'ppo' && policy !== 'random')) {
    throw new HttpError(400, `bad run id ${id}`);
  }
  const root = policy === undefined ? join(BENCH, name) : join(BENCH, name, 'baselines', policy);
  if (!existsSync(join(root, 'meta.json'))) throw new HttpError(404, `unknown run ${id}`);
  return root;
}

function runs(): unknown[] {
  const best = existsSync(BEST) ? readFileSync(BEST, 'utf8').trim() : '';
  const ids = readdirSync(BENCH).filter((n) => n !== 'baselines' && !n.startsWith('.') && existsSync(join(BENCH, n, 'meta.json')));
  return ids
    .map((id) => {
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
        windows: ends(join(BENCH, id)).length,
      };
    })
    .filter((x) => x.windows > 0)
    .sort((a, b) => a.startedAt !== b.startedAt ? (a.startedAt > b.startedAt ? -1 : 1) : a.id > b.id ? -1 : 1)
    .map(({ windows: _, ...entry }) => entry);
}

function detail(id: string): unknown {
  const root = runRoot(id);
  const m = meta(root);
  const baselines: Record<string, string> = {};
  if (m.policy === 'oak') {
    for (const policy of ['ppo', 'random']) {
      const b = join(root, 'baselines', policy);
      if (existsSync(join(b, 'meta.json')) && meta(b).finishedAt !== undefined) baselines[policy] = `${id}:${policy}`;
    }
  }
  return { id, meta: m, baselines };
}

function series(root: string, col: string): Blob {
  if (!readdirSync(join(root, 'series')).includes(`${col}.f32`)) throw new HttpError(404, `no column ${col}`);
  const e = bytes(join(root, 'ends.u32'));
  const w = bytes(join(root, 'wall.f32'));
  const v = bytes(join(root, `series/${col}.f32`));
  const W = e.byteLength / 4;
  const L = meta(root).seeds.length;
  if (w.byteLength !== W * 4 || v.byteLength !== W * L * 4) throw new HttpError(500, `${col}: ${W} windows, wall ${w.byteLength} B, values ${v.byteLength} B`);
  return new Blob([new Uint32Array([W, L]), e, w, v]);
}

async function slice(req: Request, path: string, what: string): Promise<Response> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new HttpError(404, `no ${what}`);
  const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.get('range') ?? 'bytes=0-');
  if (range === null) throw new HttpError(400, 'range must be bytes=N-[M]');
  const from = Number(range[1]);
  const to = range[2] === '' ? file.size : Math.min(file.size, Number(range[2]) + 1);
  return new Response(file.slice(from, to), { status: from > 0 || to < file.size ? 206 : 200 });
}

function poses(req: Request, id: string, lane: string): Promise<Response> {
  if (!/^\d+$/.test(lane)) throw new HttpError(400, `bad lane ${lane}`);
  return slice(req, join(runRoot(id), 'poses', lane), `lane ${lane}`);
}


const workerBuild = await Bun.build({
  entrypoints: [resolve(import.meta.dir, 'src/scenes/jelly-bean-world/scentfield.worker.ts'), resolve(import.meta.dir, 'src/theater/tape.worker.ts')],
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
const tapeWorker = workerOf('tape.worker');

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
    '/api/runs/:id': (req) => Response.json(detail(req.params.id)),
    '/api/runs/:id/series/:col': (req) => new Response(series(runRoot(req.params.id), req.params.col)),
    '/api/runs/:id/poses/:lane': (req) => poses(req, req.params.id, req.params.lane),
    '/scentfield.worker.js': () => new Response(scentfieldWorker),
    '/tape.worker.js': () => new Response(tapeWorker),
    '/posters/*': publicFile,
    '/terrain/*': publicFile,
    '/textures/*': publicFile,
  },
  fetch: () => new Response('unknown endpoint', { status: 404 }),
  error: (error) => new Response(String(error), { status: error instanceof HttpError ? error.status : 500 }),
});
console.log(`serving ${BENCH} on ${server.url}`);
