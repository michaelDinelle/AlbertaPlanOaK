import { createHash, randomUUID } from 'node:crypto';
import { open, readFile, readdir, mkdir, rename, rm, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import index from './index.html';
import type { EnvDescriptor } from './src/data/envs';
import { ENGINE_FILES, JUDGE_FILE, packRecording, type Baselines, type Meta, type RunIndexEntry } from './src/data/layout';
import { readHeader, readVerdict, VERDICT_BYTES } from './src/data/replay';

const REPO = resolve(import.meta.dir, '..');
const BENCH = join(REPO, '.oak-benchmarks');
const PUBLIC = resolve(import.meta.dir, 'public');
const BEST = join(REPO, 'best');
const ENGINE = join(REPO, '.build/oak');
const contract = await readFile(join(import.meta.dir, 'src/data/replay.ts'));

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const hash = (data: string | Uint8Array): string => createHash('sha256').update(data).digest('hex');
const missing = (error: unknown): boolean => (error as NodeJS.ErrnoException).code === 'ENOENT';
const present = async (path: string): Promise<boolean> => stat(path).then(() => true, error => { if (missing(error)) return false; throw error; });
const meta = async (root: string): Promise<Meta> => {
  const m = JSON.parse(await readFile(join(root, ENGINE_FILES.meta), 'utf8')) as Meta;
  for (const field of ['seeds', 'steps', 'batch', 'window', 'policy', 'preset', 'startedAt', 'envSchedule'] as const) {
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

let engineStamp = '';
let engineLoad: Promise<{ identity: string; catalog: EnvDescriptor[] }>;
async function engine() {
  const info = await stat(ENGINE), stamp = `${info.mtimeMs}/${info.size}`;
  if (stamp !== engineStamp) {
    engineStamp = stamp;
    engineLoad = (async () => {
      const process = Bun.spawn([ENGINE, 'envs'], { cwd: REPO, stdout: 'pipe', stderr: 'pipe' });
      const [text, errors, exit, binary] = await Promise.all([new Response(process.stdout).text(), new Response(process.stderr).text(), process.exited, readFile(ENGINE)]);
      if (exit) throw new Error(`Environment catalog failed: ${errors}`);
      const catalog = JSON.parse(text) as EnvDescriptor[];
      if (!catalog.length || new Set(catalog.map(e => e.envId)).size !== catalog.length) throw new Error('Invalid environment catalog');
      return { identity: hash(Buffer.concat([binary, contract])), catalog };
    })();
  }
  return engineLoad;
}

async function runs(): Promise<RunIndexEntry[]> {
  const best = (await readFile(BEST, 'utf8')).trim(), horizon = (await meta(join(BENCH, best))).steps;
  const entries = await Promise.all((await readdir(BENCH, { withFileTypes: true })).filter(entry => entry.isDirectory() && !entry.name.startsWith('.')).map(async directory => {
    const id = directory.name;
    const root = join(BENCH, id);
    if (!(await present(join(root, ENGINE_FILES.meta)))) return null;
    const m = await meta(root);
    if (m.policy !== 'oak' || m.steps !== horizon || m.timingOnly) return null;
    return { id, envIds: m.envSchedule.map(e => e.envId), steps: m.steps, startedAt: m.startedAt, inFlight: m.finishedAt === undefined, policy: m.policy, preset: m.preset, role: id === best ? 'best' : null } satisfies RunIndexEntry;
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
  clients: number;
  done: boolean;
  error: Error | null;
  changed: Promise<void>;
  wake(): void;
  process: ReturnType<typeof Bun.spawn>;
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

async function generate(key: string, path: string, root: string, lane: number, expected: bigint): Promise<Generation> {
  await mkdir(join(root, 'replay'), { recursive: true });
  const temporary = `${path}.${randomUUID()}.partial`;
  const file = await open(temporary, 'wx');
  const process = Bun.spawn([ENGINE, 'replay', root, String(lane)], { cwd: REPO, stdout: 'pipe', stderr: 'pipe' });
  let wake = () => {};
  const changed = () => new Promise<void>(resolve => { wake = resolve; });
  const job: Generation = { path: temporary, size: 0, clients: 0, done: false, error: null, changed: changed(), wake() { wake(); job.changed = changed(); }, process };
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
  const built = await engine(), descriptor = built.catalog.find(e => e.envId === entry.envId);
  if (!descriptor?.replay) throw new HttpError(422, 'Visualization unavailable');
  const hashes = await readFile(join(root, ENGINE_FILES.hash));
  if (hashes.length !== m.batch * 8) throw new HttpError(500, 'Recording hashes are incomplete');
  const expected = hashes.readBigUInt64LE(lane * 8);
  const key = hash(JSON.stringify([id, lane, m, expected.toString(), built.identity, descriptor]));
  const path = join(root, 'replay', `${lane}-${key}.tape`);
  const headers = { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store', 'Accept-Ranges': 'bytes' };
  if (await present(path)) {
    let size: number;
    try { size = await verify(path, expected); }
    catch (error) { await rm(path, { force: true }); throw error; }
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
    const pending = generate(key, path, root, lane, expected);
    generations.set(key, pending);
    pending.catch(() => generations.delete(key));
  }
  const job = await generations.get(key)!;
  job.clients++;
  const file = await open(job.path, 'r');
  let cursor = 0, closed = false;
  const detach = async () => {
    if (closed) return;
    closed = true;
    req.signal.removeEventListener('abort', abort);
    if (--job.clients === 0 && !job.done) { generations.delete(key); job.process.kill(); }
    job.wake();
    await file.close();
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
          const { bytesRead } = await file.read(bytes, 0, bytes.length, cursor);
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

const workerBuild = await Bun.build({ entrypoints: ['src/scenes/jelly-bean-world/scentfield.worker.ts', 'src/theater/tape.worker.ts'].map(path => resolve(import.meta.dir, path)), target: 'browser', naming: { entry: '[name].[ext]' }, splitting: true, minify: true });
if (!workerBuild.success) throw new AggregateError(workerBuild.logs, 'Worker bundle failed');
const workers = new Map(workerBuild.outputs.map(output => [output.path.split('/').at(-1)!, output]));

async function publicFile(req: Request): Promise<Response> {
  const rel = decodeURIComponent(new URL(req.url).pathname).split('/').filter(s => s !== '' && s !== '..').join('/');
  const file = Bun.file(join(PUBLIC, rel));
  if (!(await file.exists())) throw new HttpError(404, `No file ${rel}`);
  return new Response(file);
}

const server = Bun.serve({
  development: { hmr: true, console: true },
  idleTimeout: 0,
  routes: {
    '/': index,
    '/api/catalog': async () => Response.json((await engine()).catalog),
    '/api/runs': async () => Response.json(await runs()),
    '/api/runs/:id': async req => new Response(await snapshot(await runRoot(req.params.id))),
    '/api/runs/:id/replay/:lane': req => replay(req, req.params.id, req.params.lane),
    '/workers/:name': req => { const file = workers.get(req.params.name); return file ? new Response(file) : new Response('Unknown worker', { status: 404 }); },
    '/posters/*': publicFile,
    '/terrain/*': publicFile,
    '/textures/*': publicFile,
  },
  fetch: () => new Response('Unknown endpoint', { status: 404 }),
  error: error => new Response(String(error), { status: error instanceof HttpError ? error.status : 500 }),
});
console.log(`serving ${BENCH} on ${server.url}`);
