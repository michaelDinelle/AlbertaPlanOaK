import { open } from 'node:fs/promises';
import { join, normalize, resolve, sep } from 'node:path';
import { DIST_DIR, HOSTNAME, benchRootFromEnv, portFromEnv } from './config';
import { corsHeaders } from './cors';
import { statOrNull } from './fs';
import { findRecording, type PoseIndex } from './recordings';
import { RunIndex, buildRunDetail } from './runIndex';

function json(body: unknown, init: { status?: number; cache?: string } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': init.cache ?? 'no-store',
    },
  });
}

function fail(status: number, error: string): Response {
  return json({ error }, { status });
}

function decodeSegments(pathname: string): string[] | null {
  const parts = pathname.split('/').filter((s) => s.length > 0);
  const out: string[] = [];
  for (const p of parts) {
    let d: string;
    try {
      d = decodeURIComponent(p);
    } catch {
      return null;
    }
    if (d.includes('/') || d.includes('\\') || d.includes('\0') || d === '..' || d === '.') return null;
    out.push(d);
  }
  return out;
}

type RangeSpec = { start: number; end: number } | 'unsatisfiable' | 'empty' | null;

export function parseRangeHeader(header: string | null, size: number): RangeSpec {
  if (header === null) return null;
  const m = /^bytes=(.*)$/.exec(header.trim());
  if (m === null) return null;
  const spec = m[1];
  if (spec === undefined || spec.includes(',')) return null;
  const dash = spec.indexOf('-');
  if (dash < 0) return null;
  const rawStart = spec.slice(0, dash).trim();
  const rawEnd = spec.slice(dash + 1).trim();

  if (!/^\d+$/.test(rawStart)) return null;
  const start = Number(rawStart);
  if (start === size && rawEnd.length === 0) return 'empty';
  if (start >= size) return 'unsatisfiable';
  if (rawEnd.length === 0) return { start, end: size - 1 };
  if (!/^\d+$/.test(rawEnd)) return null;
  const end = Math.min(Number(rawEnd), size - 1);
  if (end < start) return 'unsatisfiable';
  return { start, end };
}

function conditional(
  req: Request, etag: string | null, size: number, contentType: string,
): { done: Response } | { status: 200 | 206; start: number; end: number; headers: Record<string, string> } {
  const base: Record<string, string> = {
    'content-type': contentType,
    'accept-ranges': 'bytes',
    'cache-control': 'private, max-age=0, must-revalidate',
  };
  if (etag !== null) {
    base['etag'] = etag;
    const inm = req.headers.get('if-none-match');
    if (inm !== null) {
      const tokens = inm.split(',').map((t) => t.trim());
      if (tokens.includes('*') || tokens.some((t) => t.replace(/^W\//, '') === etag)) {
        return { done: new Response(null, { status: 304, headers: base }) };
      }
    }
  }
  const ifRange = req.headers.get('if-range');
  const rangeValid = ifRange === null || (etag !== null && ifRange.trim() === etag);
  const range = rangeValid ? parseRangeHeader(req.headers.get('range'), size) : null;
  if (range === 'empty') {
    return {
      done: new Response(null, { status: 204, headers: { ...base, 'content-range': `bytes */${size}` } }),
    };
  }
  if (range === 'unsatisfiable') {
    return {
      done: new Response(null, { status: 416, headers: { ...base, 'content-range': `bytes */${size}` } }),
    };
  }
  const start = range?.start ?? 0;
  const end = range?.end ?? size - 1;
  const headers: Record<string, string> = { ...base, 'content-length': String(end - start + 1) };
  if (range !== null) headers['content-range'] = `bytes ${start}-${end}/${size}`;
  const status = range !== null ? 206 as const : 200 as const;
  if (req.method === 'HEAD') return { done: new Response(null, { status, headers }) };
  return { status, start, end, headers };
}

async function serveStatic(req: Request, path: string, contentType: string): Promise<Response> {
  const st = await statOrNull(path);
  if (st === null) return fail(404, 'not found');
  const etag = `"${Math.trunc(st.mtimeMs).toString(36)}-${st.size.toString(36)}"`;
  const cond = conditional(req, etag, st.size, contentType);
  if ('done' in cond) return cond.done;
  const body = cond.status === 206 ? Bun.file(path).slice(cond.start, cond.end + 1) : Bun.file(path);
  return new Response(body, { status: cond.status, headers: cond.headers });
}

async function servePose(req: Request, index: PoseIndex, lane: number): Promise<Response> {
  const segments = index.segments[lane];
  if (segments === undefined) return fail(404, 'unknown recording');
  const size = segments.reduce((n, s) => n + s.len, 0);
  const cond = conditional(req, null, size, 'application/octet-stream');
  if ('done' in cond) return cond.done;
  const { start, end } = cond;
  const length = end - start + 1;
  const out = Buffer.allocUnsafe(length);
  let wrote = 0;
  const handle = await open(index.packPath, 'r');
  try {
    let virt = 0;
    for (const seg of segments) {
      const from = Math.max(start, virt);
      const to = Math.min(end, virt + seg.len - 1);
      if (to >= from) {
        const got = await handle.read(out, wrote, to - from + 1, seg.off + (from - virt));
        if (got.bytesRead !== to - from + 1) {
          throw new Error(`${index.packPath}: short read at ${seg.off + (from - virt)}`);
        }
        wrote += got.bytesRead;
      }
      virt += seg.len;
      if (virt > end) break;
    }
  } finally {
    await handle.close().catch(() => undefined);
  }
  return new Response(out, { status: cond.status, headers: cond.headers });
}

export function createServer() {
  const benchRoot = benchRootFromEnv();
  const distDir = resolve(DIST_DIR);
  const production = process.env['NODE_ENV'] === 'production';

  const index = new RunIndex({ benchRoot });

  async function handleApi(req: Request, segs: string[]): Promise<Response> {
    const head = segs[1];

    if (head === 'health' && segs.length === 2) {
      return json({ ok: true, benchRoot, production });
    }

    if (head !== 'runs') return fail(404, 'unknown endpoint');

    if (segs.length === 2) {
      return json(await index.list());
    }

    const id = segs[2];
    if (id === undefined) return fail(404, 'unknown run');
    const run = await index.resolve(id);
    if (run === undefined) return fail(404, 'unknown run');

    if (segs.length === 3) {
      const recs = await index.recordingsFor(run.id);
      return json(await buildRunDetail(run, recs));
    }

    const sub = segs[3];

    if (sub === 'tape' && segs.length === 4) {
      return serveStatic(req, join(run.dir, 'tape.bin'), 'application/octet-stream');
    }

    if (sub === 'reward' && segs.length === 6) {
      const envId = segs[4];
      const part = segs[5];
      if (envId === undefined || !/^[A-Za-z0-9._-]+$/.test(envId) || (part !== 'pack' && part !== 'idx')) {
        return fail(404, 'unknown reward stream');
      }
      return serveStatic(req, join(run.dir, 'reward', `${envId}.${part}`), 'application/octet-stream');
    }

    if (sub === 'recordings' && segs.length === 6) {
      const envId = segs[4];
      const seedText = segs[5];
      if (envId === undefined || seedText === undefined || !/^\d+$/.test(seedText)) {
        return fail(404, 'unknown recording');
      }
      const pose = await index.poseIndexFor(run.id);
      if (pose === null) return fail(404, 'unknown recording');
      const rec = findRecording(pose.entries, envId, Number(seedText));
      if (rec === undefined) return fail(404, 'unknown recording');
      return servePose(req, pose, rec.lane);
    }

    return fail(404, 'unknown endpoint');
  }

  const CONTENT_TYPES: Record<string, string> = {
    html: 'text/html; charset=utf-8',
    js: 'text/javascript; charset=utf-8',
    css: 'text/css; charset=utf-8',
    json: 'application/json; charset=utf-8',
    svg: 'image/svg+xml',
    png: 'image/png',
    webp: 'image/webp',
    ico: 'image/x-icon',
    woff2: 'font/woff2',
    map: 'application/json; charset=utf-8',
  };

  function contentTypeFor(path: string): string {
    const dot = path.lastIndexOf('.');
    const ext = dot < 0 ? '' : path.slice(dot + 1).toLowerCase();
    return CONTENT_TYPES[ext] ?? 'application/octet-stream';
  }

  async function handleStatic(req: Request, url: URL): Promise<Response> {
    if (!production) {
      return new Response(
        'oak-frontend API server. Client dev server runs on :5173 (vite, /api proxied here).\n',
        { status: 200, headers: { 'content-type': 'text/plain; charset=utf-8' } },
      );
    }
    let rel: string;
    try {
      rel = decodeURIComponent(url.pathname);
    } catch {
      return fail(400, 'bad path');
    }
    const target = resolve(distDir, `.${normalize(rel)}`);
    if (target !== distDir && !target.startsWith(distDir + sep)) return fail(403, 'forbidden');

    const st = await statOrNull(target);
    if (st !== null && !rel.endsWith('/')) {
      return serveStatic(req, target, contentTypeFor(target));
    }
    if (rel.startsWith('/assets/') || /\.[a-z0-9]{1,8}$/i.test(rel)) {
      return fail(404, 'not found');
    }
    const indexPath = resolve(distDir, 'index.html');
    if ((await statOrNull(indexPath)) === null) return fail(404, 'no client build (run `bun run build`)');
    return serveStatic(req, indexPath, 'text/html; charset=utf-8');
  }

  function withCors(req: Request, res: Response): Response {
    const headers = corsHeaders(req, production);
    for (const [k, v] of Object.entries(headers)) res.headers.set(k, v);
    return res;
  }

  async function route(req: Request): Promise<Response> {
    if (req.method === 'OPTIONS') return new Response(null, { status: 204 });
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return json({ error: 'method not allowed' }, { status: 405 });
    }
    const url = new URL(req.url);
    const segs = decodeSegments(url.pathname);
    if (segs === null) return fail(400, 'bad path');

    if (segs[0] === 'api') {
      try {
        return await handleApi(req, segs);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.error(`[oak-frontend] ${url.pathname}: ${message}`);
        return fail(500, 'internal error');
      }
    }
    return handleStatic(req, url);
  }

  async function fetchHandler(req: Request): Promise<Response> {
    return withCors(req, await route(req));
  }

  const server = Bun.serve({
    hostname: HOSTNAME,
    port: portFromEnv(),
    idleTimeout: 0,
    fetch: fetchHandler,
  });
  return { server, benchRoot };
}
