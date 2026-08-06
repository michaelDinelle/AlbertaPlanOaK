
import { normalize, resolve, sep } from 'node:path';
import type { RolesResponse } from '@shared/types';
import {
  DIST_DIR,
  HOSTNAME,
  benchRootFromEnv,
  fleetCacheDir,
  portFromEnv,
  runIndexCacheFile,
} from './config';
import { corsHeaders } from './cors';
import { deriveCurves, parseCurvesQuery } from './curves';
import { deriveFleetColumnar, parseFleetQuery } from './fleetDerive';
import { statOrNull } from './jsonl';
import { createLiveResponse } from './live';
import { findRecording } from './recordings';
import { RunIndex, buildRunDetail } from './runIndex';
import { readRoles } from './roles';
import { deriveWall, parseWallRequest, wallCacheFor } from './wall';

export interface AppOptions {
  benchRoot?: string;
  cacheFile?: string | null;
  fleetCacheDir?: string | null;
  distDir?: string;
  production?: boolean;
  ttlMs?: number;
}

export interface App {
  fetch: (req: Request) => Promise<Response>;
  index: RunIndex;
  benchRoot: string;
}

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


type RangeSpec = { start: number; end: number } | 'unsatisfiable' | null;

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

  if (rawStart.length === 0) {
    if (!/^\d+$/.test(rawEnd)) return null;
    const n = Number(rawEnd);
    if (n === 0) return 'unsatisfiable';
    const start = Math.max(0, size - n);
    return { start, end: size - 1 };
  }
  if (!/^\d+$/.test(rawStart)) return null;
  const start = Number(rawStart);
  if (start >= size) return 'unsatisfiable';
  if (rawEnd.length === 0) return { start, end: size - 1 };
  if (!/^\d+$/.test(rawEnd)) return null;
  const end = Math.min(Number(rawEnd), size - 1);
  if (end < start) return 'unsatisfiable';
  return { start, end };
}

async function serveStatic(
  req: Request,
  path: string,
  contentType: string,
): Promise<Response> {
  const st = await statOrNull(path);
  if (st === null) return fail(404, 'not found');

  const etag = `"${Math.trunc(st.mtimeMs).toString(36)}-${st.size.toString(36)}"`;
  const base: Record<string, string> = {
    'content-type': contentType,
    'accept-ranges': 'bytes',
    etag,
    'cache-control': 'private, max-age=0, must-revalidate',
  };

  const inm = req.headers.get('if-none-match');
  if (inm !== null) {
    const tokens = inm.split(',').map((t) => t.trim());
    if (tokens.includes('*') || tokens.some((t) => t.replace(/^W\//, '') === etag)) {
      return new Response(null, { status: 304, headers: base });
    }
  }

  const range = parseRangeHeader(req.headers.get('range'), st.size);
  if (range === 'unsatisfiable') {
    return new Response(null, {
      status: 416,
      headers: { ...base, 'content-range': `bytes */${st.size}` },
    });
  }

  if (range !== null) {
    const length = range.end - range.start + 1;
    const headers = {
      ...base,
      'content-range': `bytes ${range.start}-${range.end}/${st.size}`,
      'content-length': String(length),
    };
    if (req.method === 'HEAD') return new Response(null, { status: 206, headers });
    return new Response(Bun.file(path).slice(range.start, range.end + 1), { status: 206, headers });
  }

  const headers = { ...base, 'content-length': String(st.size) };
  if (req.method === 'HEAD') return new Response(null, { status: 200, headers });
  return new Response(Bun.file(path), { status: 200, headers });
}


export function createApp(opts: AppOptions = {}): App {
  const benchRoot = opts.benchRoot ?? benchRootFromEnv();
  const cacheFile = opts.cacheFile === undefined ? runIndexCacheFile(benchRoot) : opts.cacheFile;
  const fleetCache = opts.fleetCacheDir === undefined ? fleetCacheDir(benchRoot) : opts.fleetCacheDir;
  const wallCache = wallCacheFor(benchRoot, opts.fleetCacheDir);
  const distDir = resolve(opts.distDir ?? DIST_DIR);
  const production = opts.production ?? process.env['NODE_ENV'] === 'production';

  const index = new RunIndex({ benchRoot, cacheFile, ttlMs: opts.ttlMs });

  async function handleApi(req: Request, url: URL, segs: string[]): Promise<Response> {
    const head = segs[1];

    if (head === 'health' && segs.length === 2) {
      return json({ ok: true, benchRoot, production });
    }

    if (head === 'roles' && segs.length === 2) {
      const roles: RolesResponse = await readRoles(benchRoot);
      return json(roles);
    }

    if (head === 'floors' && segs.length === 2) {
      const floorsPath = resolve(benchRoot, 'FLOORS.json');
      if ((await statOrNull(floorsPath)) === null) return fail(404, 'no floors extracted');
      try {
        return json(JSON.parse(await Bun.file(floorsPath).text()));
      } catch {
        return fail(500, 'internal error');
      }
    }

    if (head !== 'runs') return fail(404, 'unknown endpoint');

    if (segs.length === 2) {
      return json(await index.list(), { cache: 'no-store' });
    }

    const id = segs[2];
    if (id === undefined) return fail(404, 'unknown run');
    const run = await index.resolve(id);
    if (run === undefined) return fail(404, 'unknown run');

    if (segs.length === 3) {
      const recs = await index.recordingsFor(run.id);
      return json(await buildRunDetail(run, recs), {
        cache: run.entry.inFlight ? 'no-store' : 'private, max-age=5',
      });
    }

    const sub = segs[3];

    if (sub === 'fleet' && segs.length === 4) {
      const parsed = parseFleetQuery(url);
      if ('error' in parsed) return fail(400, parsed.error);
      const data = await deriveFleetColumnar({
        id: run.id,
        dir: run.dir,
        query: parsed.query,
        inFlight: run.entry.inFlight,
        cacheDir: fleetCache,
      });
      return json(data, { cache: run.entry.inFlight ? 'no-store' : 'private, max-age=30' });
    }

    if (sub === 'curves' && segs.length === 4) {
      const parsed = parseCurvesQuery(url);
      if ('error' in parsed) return fail(400, parsed.error);
      const data = await deriveCurves({
        id: run.id,
        dir: run.dir,
        query: parsed.query,
        inFlight: run.entry.inFlight,
        cacheDir: fleetCache,
      });
      return json(data, { cache: run.entry.inFlight ? 'no-store' : 'private, max-age=30' });
    }

    if (sub === 'wall' && segs.length === 5) {
      const parsed = parseWallRequest(segs[4], url);
      if ('error' in parsed) return fail(parsed.status, parsed.error);
      const recs = await index.recordingsFor(run.id);
      const { body, header } = await deriveWall({
        id: run.id,
        dir: run.dir,
        envId: parsed.envId,
        stride: parsed.stride,
        recordings: recs,
        inFlight: run.entry.inFlight,
        cacheDir: wallCache,
      });
      return new Response(req.method === 'HEAD' ? null : body, {
        status: 200,
        headers: {
          'content-type': 'application/octet-stream',
          'content-length': String(body.byteLength),
          'x-wall-header-bytes': String(header.headerBytes),
          'x-wall-binary-bytes': String(header.binaryBytes),
          'cache-control': run.entry.inFlight ? 'no-store' : 'private, max-age=30',
        },
      });
    }

    if (sub === 'live' && segs.length === 4) {
      return createLiveResponse(req, { id: run.id, dir: run.dir, steps: run.entry.steps });
    }

    if (sub === 'recordings' && segs.length === 6) {
      const envId = segs[4];
      const seedText = segs[5];
      if (envId === undefined || seedText === undefined) return fail(404, 'not found');
      if (!/^\d+$/.test(seedText)) return fail(404, 'unknown recording');
      const recs = await index.recordingsFor(run.id);
      const rec = findRecording(recs, envId, Number(seedText));
      if (rec === undefined) return fail(404, 'unknown recording');
      return json({
        envId: rec.envId,
        seed: rec.seed,
        steps: rec.steps,
        actDim: rec.actDim,
        recordVersion: rec.recordVersion,
        bytes: rec.bytes,
        ...(rec.codec !== undefined ? { codec: rec.codec } : {}),
        ...(rec.codecVersion !== undefined ? { codecVersion: rec.codecVersion } : {}),
        ...(rec.chunks !== undefined ? { chunks: rec.chunks } : {}),
      });
    }

    if (sub === 'recordings' && segs.length === 7) {
      const envId = segs[4];
      const seedText = segs[5];
      const file = segs[6];
      if (envId === undefined || seedText === undefined || file === undefined) return fail(404, 'not found');
      if (!/^\d+$/.test(seedText)) return fail(404, 'unknown recording');
      const recs = await index.recordingsFor(run.id);
      const rec = findRecording(recs, envId, Number(seedText));
      if (rec === undefined) return fail(404, 'unknown recording');
      if (file === 'actions.bin') return serveStatic(req, rec.actionsPath, 'application/octet-stream');
      if (file === 'meta.json') return serveStatic(req, rec.metaPath, 'application/json; charset=utf-8');
      return fail(404, 'not found');
    }

    if (sub === 'recordings' && segs.length === 4) {
      const recs = await index.recordingsFor(run.id);
      return json(recs.map((r) => ({ envId: r.envId, seed: r.seed, steps: r.steps })));
    }

    return fail(404, 'unknown endpoint');
  }

  const CONTENT_TYPES: Record<string, string> = {
    html: 'text/html; charset=utf-8',
    js: 'text/javascript; charset=utf-8',
    mjs: 'text/javascript; charset=utf-8',
    css: 'text/css; charset=utf-8',
    json: 'application/json; charset=utf-8',
    svg: 'image/svg+xml',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    webp: 'image/webp',
    ico: 'image/x-icon',
    woff: 'font/woff',
    woff2: 'font/woff2',
    map: 'application/json; charset=utf-8',
    bin: 'application/octet-stream',
    glb: 'model/gltf-binary',
    ktx2: 'image/ktx2',
    wasm: 'application/wasm',
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
        return await handleApi(req, url, segs);
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

  return { fetch: fetchHandler, index, benchRoot };
}

export interface ServeOptions extends AppOptions {
  port?: number;
  hostname?: string;
}

export function createServer(opts: ServeOptions = {}) {
  const app = createApp(opts);
  const server = Bun.serve({
    hostname: opts.hostname ?? HOSTNAME,
    port: opts.port ?? portFromEnv(),
    idleTimeout: 0,
    fetch: app.fetch,
  });
  return { server, app };
}
