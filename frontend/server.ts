import { createHash, randomUUID } from 'node:crypto';
import { watch } from 'node:fs';
import { open, readFile, readdir, mkdir, rename, rm, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import index from './index.html';
import wgsl from './wgsl';
import type { EnvDescriptor } from './src/data/envs';
import { ENGINE_FILES, JUDGE_FILE, RECORDED_ENGINE, packRecording, type Baselines, type Meta, type RunIndexEntry } from './src/data/layout';
import { readHeader, readVerdict, VERDICT_BYTES } from './src/data/replay';

const REPO = resolve(import.meta.dir, '..');
const BENCH = join(REPO, '.oak-benchmarks');
const PUBLIC = resolve(import.meta.dir, 'public');
const BEST = join(REPO, 'best');
const contract = await readFile(join(import.meta.dir, 'src/data/replay.ts'));

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const hash = (data: string | Uint8Array): string => createHash('sha256').update(data).digest('hex');
const missing = (error: unknown): boolean => (error as NodeJS.ErrnoException).code === 'ENOENT';
const present = async (path: string): Promise<boolean> => stat(path).then(() => true, error => { if (missing(error)) return false; throw error; });
const meta = async (root: string): Promise<Meta> => {
  const m = JSON.parse(await readFile(join(root, ENGINE_FILES.meta), 'utf8')) as Meta;
  for (const field of ['seeds', 'steps', 'batch', 'window', 'policy', 'startedAt', 'envSchedule'] as const) {
    if (!(field in m)) throw new HttpError(500, `${root}/meta.json lacks ${field}`);
  }
  if (m.seeds.length !== m.batch || m.envSchedule.flatMap(e => Array.from({ length: e.seedEnd - e.seedStart + 1 }, (_, i) => e.seedStart + i)).some((seed, i) => seed !== m.seeds[i])) throw new HttpError(500, 'Recording lane schedule is inconsistent');
  return m;
};

async function runRoot(id: string): Promise<string> {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(id)) throw new HttpError(400, 'Invalid recording ID');
  const root = join(BENCH, id);
  if (!(await present(join(root, ENGINE_FILES.meta)))) throw new HttpError(404, `Unknown recording ${id}`);
  return root;
}

interface RecordedEngine {
  path: string;
  identity: string;
  catalog: EnvDescriptor[];
}

const engines = new Map<string, { stamp: string; engine: Promise<RecordedEngine> }>();
async function recordedEngine(root: string): Promise<RecordedEngine> {
  const path = join(root, RECORDED_ENGINE);
  const info = await stat(path).catch(error => { if (missing(error)) throw new HttpError(500, `${root} carries no recorded engine`); throw error; });
  const stamp = `${info.mtimeMs}/${info.size}`, hit = engines.get(root);
  if (hit?.stamp === stamp) return hit.engine;
  const engine = (async () => {
    const process = Bun.spawn([path, 'envs'], { cwd: REPO, stdout: 'pipe', stderr: 'pipe' });
    const [text, errors, exit, binary] = await Promise.all([new Response(process.stdout).text(), new Response(process.stderr).text(), process.exited, readFile(path)]);
    if (exit) throw new Error(`Environment catalog failed: ${errors}`);
    const catalog = JSON.parse(text) as EnvDescriptor[];
    if (!catalog.length || new Set(catalog.map(e => e.envId)).size !== catalog.length) throw new Error('Invalid environment catalog');
    return { path, identity: hash(Buffer.concat([binary, contract])), catalog };
  })();
  engines.set(root, { stamp, engine });
  engine.catch(() => { if (engines.get(root)?.engine === engine) engines.delete(root); });
  return engine;
}

const bestId = async (): Promise<string> => (await readFile(BEST, 'utf8')).trim();

const metas = new Map<string, { stamp: number; meta: Promise<Meta> }>();
async function cachedMeta(root: string): Promise<Meta | null> {
  const stamp = await stat(join(root, ENGINE_FILES.meta)).then(info => info.mtimeMs, error => { if (missing(error)) return null; throw error; });
  if (stamp === null) return null;
  const hit = metas.get(root);
  if (hit?.stamp === stamp) return hit.meta;
  const pending = meta(root);
  metas.set(root, { stamp, meta: pending });
  pending.catch(() => { if (metas.get(root)?.meta === pending) metas.delete(root); });
  return pending;
}

async function runs(): Promise<RunIndexEntry[]> {
  const best = await bestId(), horizon = (await meta(join(BENCH, best))).steps;
  const entries = await Promise.all((await readdir(BENCH, { withFileTypes: true })).filter(entry => entry.isDirectory() && !entry.name.startsWith('.')).map(async directory => {
    const id = directory.name;
    const m = await cachedMeta(join(BENCH, id));
    if (!m || m.policy !== 'oak' || m.steps !== horizon || m.timingOnly) return null;
    return { id, envIds: m.envSchedule.map(e => e.envId), steps: m.steps, startedAt: m.startedAt, inFlight: m.finishedAt === undefined, policy: m.policy, role: id === best ? 'best' : null } satisfies RunIndexEntry;
  }));
  return entries.filter((entry): entry is NonNullable<typeof entry> => entry !== null).sort((a, b) => b.startedAt.localeCompare(a.startedAt) || b.id.localeCompare(a.id));
}

const snapshots = new Map<string, { stamp: number; data: Promise<ArrayBuffer> }>();
async function snapshot(root: string): Promise<ArrayBuffer> {
  const stamp = (await stat(join(root, ENGINE_FILES.meta))).mtimeMs;
  if (snapshots.get(root)?.stamp === stamp) return snapshots.get(root)!.data;
  const data = (async () => {
    const [m, baselineText, ends, sums, hashes, health] = await Promise.all([
      meta(root), readFile(join(root, JUDGE_FILE), 'utf8').catch(error => { if (missing(error)) return '{}'; throw error; }),
      ...[ENGINE_FILES.ends, ENGINE_FILES.rewardSum, ENGINE_FILES.hash, ENGINE_FILES.health].map(file => readFile(join(root, file))),
    ]);
    const bytes = (b: Buffer): ArrayBuffer => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
    return packRecording(m, JSON.parse(baselineText) as Baselines, new Uint32Array(bytes(ends!)), new Float32Array(bytes(sums!)), new BigUint64Array(bytes(hashes!)), new Uint32Array(bytes(health!)));
  })();
  snapshots.set(root, { stamp, data });
  data.catch(() => { if (snapshots.get(root)?.data === data) snapshots.delete(root); });
  return data;
}

interface Generation {
  path: string;
  size: number;
  done: boolean;
  error: Error | null;
  changed: Promise<void>;
  wake(): void;
}

const generations = new Map<string, Promise<Generation>>();
const CHUNK_BYTES = 256 * 1024;

async function verify(path: string, expected: bigint) {
  const file = await open(path, 'r');
  try {
    const { size } = await file.stat();
    const bytes = Buffer.alloc(Math.min(size, 1024 * 1024 + 32));
    await file.read(bytes, 0, bytes.length, 0);
    const header = readHeader(bytes);
    if (!header || size !== header.offset + header.frames * header.width * 4 + VERDICT_BYTES) throw new Error('Incomplete replay tape');
    const footer = Buffer.alloc(VERDICT_BYTES);
    await file.read(footer, 0, footer.length, size - footer.length);
    const verdict = readVerdict(footer, header);
    if (!verdict.identical || verdict.recorded !== expected) throw new Error('Replay reward hash differs from the recording');
    return size;
  } finally { await file.close(); }
}

async function generate(key: string, path: string, root: string, engine: string, lane: number, expected: bigint): Promise<Generation> {
  await mkdir(join(root, 'replay'), { recursive: true });
  const temporary = `${path}.${randomUUID()}.partial`;
  const file = await open(temporary, 'wx');
  const process = Bun.spawn([engine, 'replay', root, String(lane)], { cwd: REPO, stdout: 'pipe', stderr: 'pipe' });
  let wake = () => {};
  const changed = () => new Promise<void>(resolve => { wake = resolve; });
  const job: Generation = { path: temporary, size: 0, done: false, error: null, changed: changed(), wake() { wake(); job.changed = changed(); } };
  const errors = new Response(process.stderr).text();
  void (async () => {
    try {
      for await (const chunk of process.stdout) {
        await file.writeFile(chunk);
        job.size += chunk.byteLength;
        job.wake();
      }
      const [exit, message] = await Promise.all([process.exited, errors]);
      if (exit !== 0) throw new Error(exit === 2 ? 'Replay reward hash differs from the recording' : message.trim() || 'Replay generation interrupted');
      await file.sync();
      await verify(temporary, expected);
      await rename(temporary, path);
      job.path = path;
    } catch (error) {
      job.error = error instanceof Error ? error : new Error(String(error));
      process.kill();
      await process.exited;
      await rm(temporary, { force: true });
    } finally {
      await file.close();
      job.done = true;
      job.wake();
      const pending = generations.get(key);
      if (pending && await pending === job) generations.delete(key);
    }
  })();
  return job;
}

async function replay(req: Request, id: string, laneText: string): Promise<Response> {
  if (!/^\d+$/.test(laneText)) throw new HttpError(400, 'Invalid lane');
  const root = await runRoot(id), m = await meta(root), lane = Number(laneText);
  if (!Number.isSafeInteger(lane) || lane >= m.batch) throw new HttpError(400, 'Lane out of range');
  if (!m.finishedAt) throw new HttpError(409, 'Recording is still in progress');
  const seed = m.seeds[lane]!, entry = m.envSchedule.find(e => seed >= e.seedStart && seed <= e.seedEnd)!;
  const built = await recordedEngine(root), descriptor = built.catalog.find(e => e.envId === entry.envId);
  if (!descriptor) throw new HttpError(422, 'Environment is not in the catalog');
  const hashes = await readFile(join(root, ENGINE_FILES.hash));
  if (hashes.length !== m.batch * 8) throw new HttpError(500, 'Recording hashes are incomplete');
  const expected = hashes.readBigUInt64LE(lane * 8);
  const key = hash(JSON.stringify([id, lane, m, expected.toString(), built.identity, descriptor]));
  const path = join(root, 'replay', `${lane}-${key}.tape`);
  const headers = { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store', 'Accept-Ranges': 'bytes' };
  if (await present(path)) {
    const size = (await stat(path)).size;
    const range = req.headers.get('Range');
    let first = 0, last = size - 1;
    if (range !== null) {
      const match = /^bytes=(\d+)-(\d*)$/.exec(range);
      if (!match) throw new HttpError(416, 'Invalid replay range');
      first = Number(match[1]); last = match[2] ? Number(match[2]) : size - 1;
      if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first > last || last >= size) return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${size}` } });
    }
    return new Response(Bun.file(path).slice(first, last + 1), { status: range ? 206 : 200, headers: { ...headers, 'Content-Length': String(last - first + 1), 'X-Oak-Cached': '1', ...(range ? { 'Content-Range': `bytes ${first}-${last}/${size}` } : {}) } });
  }
  if (!generations.has(key)) {
    const pending = generate(key, path, root, built.path, lane, expected);
    generations.set(key, pending);
    pending.catch(() => generations.delete(key));
  }
  const job = await generations.get(key)!;
  let cursor = 0, closed = false;
  const readAt = async (bytes: Uint8Array, at: number) => {
    const file = await open(job.path, 'r').catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error; return open(path, 'r'); });
    try { return (await file.read(bytes, 0, bytes.length, at)).bytesRead; } finally { await file.close(); }
  };
  const detach = async () => {
    if (closed) return;
    closed = true;
    req.signal.removeEventListener('abort', abort);
    job.wake();
  };
  const abort = () => { void detach(); };
  req.signal.addEventListener('abort', abort, { once: true });
  if (req.signal.aborted) await detach();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        while (!closed && cursor === job.size && !job.done) await job.changed;
        if (closed) { controller.close(); return; }
        if (cursor < job.size) {
          const bytes = new Uint8Array(Math.min(CHUNK_BYTES, job.size - cursor));
          const bytesRead = await readAt(bytes, cursor);
          if (!bytesRead) throw new Error('Replay cache write was incomplete');
          cursor += bytesRead;
          controller.enqueue(bytes.subarray(0, bytesRead));
        } else {
          if (job.error) throw job.error;
          controller.close();
          await detach();
        }
      } catch (error) { controller.error(error); await detach(); }
    },
    cancel: detach,
  });
  return new Response(body, { headers: { ...headers, 'X-Oak-Cached': '0' } });
}

type WorkerFiles = Map<string, Blob>;
let workers: Promise<WorkerFiles> | null = null, previousWorkers: WorkerFiles = new Map();
const buildWorkers = async (): Promise<WorkerFiles> => {
  const build = await Bun.build({ entrypoints: ['src/scenes/jbw/scentfield.worker.ts', 'src/theater/tape.worker.ts'].map(path => resolve(import.meta.dir, path)), target: 'browser', naming: { entry: '[name].[ext]' }, splitting: true, minify: true, plugins: [wgsl] });
  if (!build.success) throw new AggregateError(build.logs, 'Worker bundle failed');
  return new Map(build.outputs.map(output => [output.path.split('/').at(-1)!, output]));
};
const workerFile = async (name: string): Promise<Blob | undefined> => {
  workers ??= buildWorkers();
  return (await workers).get(name) ?? previousWorkers.get(name);
};
watch(resolve(import.meta.dir, 'src'), { recursive: true }, () => {
  if (workers === null) return;
  const stale = workers;
  workers = null;
  void stale.then(files => { previousWorkers = files; }, () => {});
});
await workerFile('tape.worker.js');

async function publicFile(req: Request): Promise<Response> {
  const rel = decodeURIComponent(new URL(req.url).pathname).split('/').filter(s => s !== '' && s !== '..').join('/');
  const file = Bun.file(join(PUBLIC, rel));
  if (!(await file.exists())) throw new HttpError(404, `No file ${rel}`);
  return new Response(file);
}

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  development: { hmr: true, console: true },
  idleTimeout: 0,
  routes: {
    '/': index,
    '/api/catalog': async () => Response.json((await recordedEngine(join(BENCH, await bestId()))).catalog),
    '/api/runs': async () => Response.json(await runs()),
    '/api/runs/:id': async req => new Response(await snapshot(await runRoot(req.params.id))),
    '/api/runs/:id/replay/:lane': req => replay(req, req.params.id, req.params.lane),
    '/workers/:name': async req => { const file = await workerFile(req.params.name); return file ? new Response(file) : new Response('Unknown worker', { status: 404 }); },
    '/posters/*': async req => publicFile(req).catch(error => { if (error instanceof HttpError && error.status === 404) return new Response(null, { status: 204 }); throw error; }),
    '/api/posters/:name': { POST: async req => { await Bun.write(join(PUBLIC, 'posters', `${req.params.name.replace(/[^\w-]/g, '')}.webp`), await req.arrayBuffer()); return new Response(null, { status: 204 }); } },
    '/terrain/*': publicFile,
    '/textures/*': publicFile,
  },
  fetch: () => new Response('Unknown endpoint', { status: 404 }),
  error: error => new Response(String(error), { status: error instanceof HttpError ? error.status : 500 }),
});
console.log(`serving ${BENCH} on ${server.url}`);
