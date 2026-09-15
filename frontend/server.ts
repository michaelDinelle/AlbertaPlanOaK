import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { inflateRawSync } from 'node:zlib';

export const BENCH = resolve(import.meta.dir, process.argv[2] ?? '../.oak-benchmarks');
const DIST = resolve(import.meta.dir, 'dist');
const BEST = resolve(import.meta.dir, '../best');
const FRAME_FLOATS = 11;
const CHUNK_MAGIC = 0x524b4843;
const LPC_ORDER = 8;
const LPC_SHIFT = 14;

interface Field { slot: number; wrap: boolean; scale: number; anchor: boolean }
interface Meta {
  seeds: number[];
  steps: number;
  batch: number;
  policy: string;
  preset: string;
  startedAt: string;
  finishedAt?: string;
  envSchedule: { envId: string; seedStart: number; seedEnd: number }[];
  envCatalog: { envId: string; fields: Field[] }[];
}

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const bytes = (path: string): ArrayBuffer => {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

function readRange(path: string, offset: number, length: number): ArrayBuffer {
  const out = new Uint8Array(length);
  const fd = openSync(path, 'r');
  const got = readSync(fd, out, 0, length, offset);
  closeSync(fd);
  if (got !== length) throw new HttpError(500, `${path}: read ${got} of ${length} bytes at ${offset}`);
  return out.buffer;
}

const META_FIELDS = ['seeds', 'steps', 'batch', 'policy', 'preset', 'startedAt', 'envSchedule', 'envCatalog'];

function meta(root: string): Meta {
  const m = JSON.parse(readFileSync(join(root, 'meta.json'), 'utf8')) as Meta;
  for (const field of META_FIELDS) if (!(field in m)) throw new HttpError(500, `${root}/meta.json lacks ${field}`);
  return m;
}

const ends = (root: string): Uint32Array => existsSync(join(root, 'ends.u32')) ? new Uint32Array(bytes(join(root, 'ends.u32'))) : new Uint32Array(0);

function int(v: string | null, what: string): number {
  const n = Number(v);
  if (v === null || v === '' || !Number.isInteger(n) || n < 0) throw new HttpError(400, `${what} must be a non-negative integer`);
  return n;
}

export function runRoot(id: string): string {
  const [name, policy, ...rest] = id.split(':');
  if (!name || rest.length > 0 || name.includes('/') || name.startsWith('.') || (policy !== undefined && policy !== 'ppo' && policy !== 'random')) {
    throw new HttpError(400, `bad run id ${id}`);
  }
  const root = policy === undefined ? join(BENCH, name) : join(BENCH, name, 'baselines', policy);
  if (!existsSync(join(root, 'meta.json'))) throw new HttpError(404, `unknown run ${id}`);
  return root;
}

function laneEnv(m: Meta, lane: number): { envId: string; width: number; column: number; fields: Field[] } {
  let at = 0;
  for (const e of m.envSchedule) {
    const width = e.seedEnd - e.seedStart + 1;
    if (lane < at + width) {
      const entry = m.envCatalog.find((c) => c.envId === e.envId);
      if (entry === undefined) throw new HttpError(500, `catalog has no entry for ${e.envId}`);
      return { envId: e.envId, width, column: lane - at, fields: entry.fields };
    }
    at += width;
  }
  throw new HttpError(404, `no lane ${lane}`);
}

function runs(): unknown[] {
  const best = existsSync(BEST) ? readFileSync(BEST, 'utf8').trim() : '';
  const ids = readdirSync(BENCH).filter((n) => n !== 'baselines' && !n.startsWith('.') && existsSync(join(BENCH, n, 'meta.json')));
  const entries = ids.map((id) => {
    const root = join(BENCH, id);
    const m = meta(root);
    const e = ends(root);
    return {
      id,
      envIds: m.envSchedule.map((x) => x.envId),
      steps: m.steps,
      startedAt: m.startedAt,
      complete: m.finishedAt !== undefined && (e[e.length - 1] ?? 0) >= m.steps,
      inFlight: m.finishedAt === undefined,
      policy: m.policy,
      preset: m.preset,
      role: id === best ? 'best' : null,
      windows: e.length,
    };
  });
  return entries
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

const exact = (x: number): number => {
  if (!Number.isSafeInteger(x)) throw new HttpError(500, `pose integer ${x} exceeds 2^53`);
  return x;
};
const unzig = (z: number): number => (exact(z) % 2 ? -(z + 1) / 2 : z / 2);

function decodeColumns(p: Uint8Array, live: number, fields: Field[]): { doneAt: number; columns: Float64Array[] } {
  const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
  let off = 0;
  const take = (n: number, what: string): number => {
    if (off + n > p.length) throw new HttpError(500, `chunk payload is ${p.length} bytes, too short for ${what} at ${off} (+${n})`);
    const at = off;
    off += n;
    return at;
  };
  const doneAt = take((live + 7) >> 3, 'the done bitstream');
  const metaAt = take(fields.length, 'the column meta bytes');
  const columns = fields.map((f, k) => {
    const m = p[metaAt + k]!;
    const pred = m & 3;
    let anchored = f.anchor ? dv.getFloat64(take(8, `anchor seed of column ${k}`), true) : 0;
    const coefs: number[] = [];
    if (pred === 2) for (let j = 0; j < LPC_ORDER; j++) coefs.push(dv.getInt32(take(4, `LPC coefficients of column ${k}`), true));
    const res = new Float64Array(live);
    if (m & 4) {
      const riceK = p[take(1, `Rice parameter of column ${k}`)]!;
      let bit = off * 8;
      const readBit = (): number => {
        if (bit >> 3 >= p.length) throw new HttpError(500, `column ${k} ran past the payload`);
        return (p[bit >> 3]! >> (bit++ & 7)) & 1;
      };
      for (let i = 0; i < live; i++) {
        let unary = 0;
        while (readBit() === 0) unary++;
        let low = 0;
        for (let j = 0; j < riceK; j++) low += readBit() * 2 ** j;
        res[i] = unzig(unary * 2 ** riceK + low);
      }
      off = (bit + 7) >> 3;
    } else {
      for (let i = 0; i < live; i++) {
        let raw = 0;
        let mul = 1;
        for (;;) {
          const b = p[take(1, `column ${k}`)]!;
          raw += (b & 0x7f) * mul;
          if ((b & 0x80) === 0) break;
          mul *= 128;
        }
        res[i] = unzig(raw);
      }
    }
    const q = new Float64Array(live);
    if (pred === 2) {
      for (let i = 0; i < live; i++) {
        let acc = 0;
        if (i >= LPC_ORDER) {
          for (let j = 0; j < LPC_ORDER; j++) acc = exact(acc + exact(coefs[j]! * q[i - 1 - j]!));
          q[i] = exact(Math.floor(acc / 2 ** LPC_SHIFT) + res[i]!);
        } else {
          q[i] = exact((i === 0 ? 0 : q[i - 1]!) + res[i]!);
        }
      }
    } else {
      let value = 0;
      let delta = 0;
      for (let i = 0; i < live; i++) {
        delta = pred === 1 && i >= 2 ? delta + res[i]! : res[i]!;
        value = i === 0 ? delta : value + delta;
        q[i] = exact(value);
      }
    }
    const out = new Float64Array(live);
    if (f.anchor) {
      let prev = 0;
      for (let i = 0; i < live; i++) {
        anchored += exact(q[i]! - prev) * f.scale;
        prev = q[i]!;
        out[i] = anchored;
      }
    } else {
      for (let i = 0; i < live; i++) out[i] = q[i]! * f.scale;
    }
    return out;
  });
  if (off !== p.length) throw new HttpError(500, `chunk payload has ${p.length - off} trailing bytes`);
  return { doneAt, columns };
}

export function frames(root: string, lane: number, from: number, n: number): Float32Array<ArrayBuffer> {
  const m = meta(root);
  const env = laneEnv(m, lane);
  const e = ends(root);
  const file = new Uint8Array(bytes(join(root, 'poses', String(lane))));
  const dv = new DataView(file.buffer);
  const rewardPath = join(root, 'rewards', `${env.envId}.f32`);
  const end = Math.min(from + n, m.steps);
  const out = new Float32Array(Math.max(0, end - from) * FRAME_FLOATS);
  let off = 0;
  let written = 0;
  for (let k = 0; off < file.byteLength; k++) {
    if (off + 20 > file.byteLength || dv.getUint32(off, true) !== CHUNK_MAGIC) throw new HttpError(500, `lane ${lane} chunk ${k} at ${off} is not a record chunk`);
    const live = dv.getUint32(off + 4, true);
    const payloadBytes = dv.getUint32(off + 8, true);
    const storedBytes = dv.getUint32(off + 12, true);
    const flags = dv.getUint32(off + 16, true);
    const start = k === 0 ? 0 : e[k - 1];
    if (start === undefined || off + 20 + storedBytes > file.byteLength) throw new HttpError(500, `lane ${lane} chunk ${k} has no window or is truncated`);
    if (start >= end) break;
    if (start + live > from) {
      const stored = file.subarray(off + 20, off + 20 + storedBytes);
      const payload = flags >>> 31 ? new Uint8Array(inflateRawSync(stored)) : stored;
      if (payload.length !== payloadBytes) throw new HttpError(500, `chunk payload is ${payload.length} bytes, header claims ${payloadBytes}`);
      const { doneAt, columns } = decodeColumns(payload, live, env.fields);
      const lo = Math.max(from, start) - start;
      const hi = Math.min(end, start + live) - start;
      const reward = new Float32Array(readRange(rewardPath, (start + lo) * env.width * 4, (hi - lo) * env.width * 4));
      for (let i = lo; i < hi; i++, written++) {
        const row = written * FRAME_FLOATS;
        out[row] = (payload[doneAt + (i >> 3)]! >> (i & 7)) & 1;
        env.fields.forEach((f, c) => { out[row + 1 + f.slot] = columns[c]![i]!; });
        out[row + FRAME_FLOATS - 1] = reward[(i - lo) * env.width + env.column]!;
      }
    }
    off += 20 + storedBytes;
  }
  return out.subarray(0, written * FRAME_FLOATS);
}

function staticFile(pathname: string): Response {
  const rel = decodeURIComponent(pathname).split('/').filter((s) => s !== '' && s !== '..').join('/');
  let path = join(DIST, rel);
  if (!existsSync(path) || statSync(path).isDirectory()) path = join(DIST, 'index.html');
  return new Response(Bun.file(path));
}

const binary = (body: Float32Array<ArrayBuffer>): Response => body.length === 0 ? new Response(null, { status: 204 }) : new Response(body);

function route(url: URL): Response {
  const s = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  if (s[0] !== 'api') return staticFile(url.pathname);
  const q = (k: string): number => int(url.searchParams.get(k), k);
  if (s.length === 2 && s[1] === 'runs') return Response.json(runs());
  if (s.length === 3 && s[1] === 'runs') return Response.json(detail(s[2]!));
  if (s.length === 5 && s[1] === 'runs' && s[3] === 'series') return new Response(series(runRoot(s[2]!), s[4]!));
  if (s.length === 5 && s[1] === 'runs' && s[3] === 'frames') return binary(frames(runRoot(s[2]!), int(s[4]!, 'lane'), q('from'), q('n')));
  throw new HttpError(404, 'unknown endpoint');
}

if (import.meta.main) {
  Bun.serve({
    port: 5174,
    fetch(req) {
      try {
        return route(new URL(req.url));
      } catch (error) {
        return new Response(String(error), { status: error instanceof HttpError ? error.status : 500 });
      }
    },
  });
  console.log(`serving ${BENCH} on http://127.0.0.1:5174`);
}
