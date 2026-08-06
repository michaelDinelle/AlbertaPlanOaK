
import { lstat, stat } from 'node:fs/promises';

export interface JsonlParse<T> {
  rows: T[];
  skipped: number;
  truncatedTail: boolean;
}

export function parseJsonlDetailed<T>(text: string): JsonlParse<T> {
  const rows: T[] = [];
  let skipped = 0;
  let truncatedTail = false;
  if (text.length === 0) return { rows, skipped, truncatedTail };

  const terminated = text.endsWith('\n');
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (raw === undefined) continue;
    const line = raw.trim();
    if (line.length === 0) continue;
    try {
      rows.push(JSON.parse(line) as T);
    } catch {
      skipped++;
      if (i === lines.length - 1 && !terminated) truncatedTail = true;
    }
  }
  return { rows, skipped, truncatedTail };
}

export function parseJsonl<T>(text: string): T[] {
  return parseJsonlDetailed<T>(text).rows;
}

export interface FileStat {
  size: number;
  mtimeMs: number;
}

export async function statOrNull(path: string): Promise<FileStat | null> {
  try {
    const s = await stat(path);
    return { size: s.size, mtimeMs: s.mtimeMs };
  } catch {
    return null;
  }
}

export interface LinkStat extends FileStat {
  isSymbolicLink: boolean;
  isDirectory: boolean;
  isFile: boolean;
}

export async function lstatOrNull(path: string): Promise<LinkStat | null> {
  try {
    const s = await lstat(path);
    return {
      size: s.size,
      mtimeMs: s.mtimeMs,
      isSymbolicLink: s.isSymbolicLink(),
      isDirectory: s.isDirectory(),
      isFile: s.isFile(),
    };
  } catch {
    return null;
  }
}

export interface TailRead {
  text: string;
  size: number;
  mtimeMs: number;
  windowed: boolean;
}

export async function readTail(path: string, bytes: number): Promise<TailRead | null> {
  const s = await statOrNull(path);
  if (s === null) return null;
  const start = Math.max(0, s.size - bytes);
  let text: string;
  try {
    text = await Bun.file(path).slice(start, s.size).text();
  } catch {
    return null;
  }
  if (start === 0) return { text, size: s.size, mtimeMs: s.mtimeMs, windowed: false };
  const nl = text.indexOf('\n');
  return {
    text: nl >= 0 ? text.slice(nl + 1) : '',
    size: s.size,
    mtimeMs: s.mtimeMs,
    windowed: true,
  };
}

export async function readJsonlFrom<T>(
  path: string,
  offset: number,
): Promise<{ rows: T[]; nextOffset: number; size: number; reset: boolean } | null> {
  const s = await statOrNull(path);
  if (s === null) return null;
  let from = offset;
  let reset = false;
  if (from > s.size) {
    from = 0;
    reset = true;
  }
  if (from === s.size) return { rows: [], nextOffset: from, size: s.size, reset };

  let buf: Uint8Array;
  try {
    buf = new Uint8Array(await Bun.file(path).slice(from, s.size).arrayBuffer());
  } catch {
    return { rows: [], nextOffset: from, size: s.size, reset };
  }
  let lastNl = -1;
  for (let i = buf.length - 1; i >= 0; i--) {
    if (buf[i] === 0x0a) {
      lastNl = i;
      break;
    }
  }
  if (lastNl < 0) return { rows: [], nextOffset: from, size: s.size, reset };
  const text = new TextDecoder().decode(buf.subarray(0, lastNl + 1));
  return {
    rows: parseJsonl<T>(text),
    nextOffset: from + lastNl + 1,
    size: s.size,
    reset,
  };
}

export async function streamJsonl<T>(
  path: string,
  onRow: (row: T) => void,
): Promise<{ rows: number; skipped: number }> {
  const s = await statOrNull(path);
  if (s === null) return { rows: 0, skipped: 0 };

  const decoder = new TextDecoder();
  let remainder = '';
  let rows = 0;
  let skipped = 0;

  const flush = (line: string): void => {
    const t = line.trim();
    if (t.length === 0) return;
    try {
      onRow(JSON.parse(t) as T);
      rows++;
    } catch {
      skipped++;
    }
  };

  try {
    const reader = Bun.file(path).stream().getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (value !== undefined) {
          remainder += decoder.decode(value, { stream: true });
          let start = 0;
          let nl = remainder.indexOf('\n', start);
          while (nl >= 0) {
            flush(remainder.slice(start, nl));
            start = nl + 1;
            nl = remainder.indexOf('\n', start);
          }
          remainder = start > 0 ? remainder.slice(start) : remainder;
        }
        if (done) break;
      }
    } finally {
      reader.releaseLock();
    }
  } catch {
    return { rows, skipped };
  }
  remainder += decoder.decode();
  flush(remainder);
  return { rows, skipped };
}


const SEGMENT = /^[A-Za-z_][A-Za-z0-9_]*$/;
const BANNED = new Set(['__proto__', 'constructor', 'prototype']);

export function parseFieldPath(field: string): string[] | null {
  const segs = field.split('.');
  if (segs.length === 0 || segs.length > 4) return null;
  for (const seg of segs) {
    if (!SEGMENT.test(seg) || BANNED.has(seg)) return null;
  }
  return segs;
}

export function pluckNumber(row: unknown, path: readonly string[]): number | null {
  let cur: unknown = row;
  for (const seg of path) {
    if (cur === null || typeof cur !== 'object') return null;
    cur = (cur as Record<string, unknown>)[seg];
  }
  if (typeof cur === 'number') return Number.isFinite(cur) ? cur : null;
  if (typeof cur === 'boolean') return cur ? 1 : 0;
  return null;
}
