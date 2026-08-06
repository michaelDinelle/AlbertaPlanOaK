
import { join } from 'node:path';
import type { AggregateRow, LiveHello, LiveStatus } from '@shared/types';
import { IN_FLIGHT_MS, LIVE_KEEPALIVE_POLLS, LIVE_POLL_MS } from './config';
import { readJsonlFrom, statOrNull } from './jsonl';
import { probeFleetTail } from './runIndex';

export interface LiveTarget {
  id: string;
  dir: string;
  steps: number;
}

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function createLiveResponse(req: Request, target: LiveTarget, pollMs = LIVE_POLL_MS): Response {
  const aggPath = join(target.dir, 'aggregate.jsonl');
  const fleetPath = join(target.dir, 'fleet.jsonl');
  const encoder = new TextEncoder();

  let timer: ReturnType<typeof setInterval> | null = null;
  let closed = false;
  let offset = 0;
  let fleetKey = '';
  let lastStatus = '';
  let polls = 0;
  let onAbort: (() => void) | null = null;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const write = (chunk: string): boolean => {
        if (closed) return false;
        try {
          controller.enqueue(encoder.encode(chunk));
          return true;
        } catch {
          teardown();
          return false;
        }
      };
      const send = (event: string, data: unknown): boolean => write(sseFrame(event, data));

      const finish = (): void => {
        if (closed) return;
        teardown();
        try {
          controller.close();
        } catch {
        }
      };

      const readStatus = async (): Promise<LiveStatus | null> => {
        const st = await statOrNull(fleetPath);
        const ag = await statOrNull(aggPath);
        if (st === null && ag === null) return { maxStep: 0, inFlight: false, complete: false };
        const key = `${st?.mtimeMs ?? 0}:${st?.size ?? 0}:${ag?.mtimeMs ?? 0}:${ag?.size ?? 0}`;
        if (key === fleetKey && lastStatus !== '') return null;
        fleetKey = key;
        const probe = st === null
          ? { hasFinal: false, maxStep: 0, rowsScanned: 0 }
          : await probeFleetTail(fleetPath);
        if (!probe.hasFinal && ag !== null) {
          const aggProbe = await probeFleetTail(aggPath);
          probe.maxStep = Math.max(probe.maxStep, aggProbe.maxStep);
        }
        const complete = probe.hasFinal && probe.maxStep >= target.steps && target.steps > 0;
        const fresh = Math.max(st?.mtimeMs ?? 0, ag?.mtimeMs ?? 0);
        return {
          maxStep: probe.maxStep,
          inFlight: !complete && Date.now() - fresh < IN_FLIGHT_MS,
          complete,
        };
      };

      const pushStatus = async (): Promise<boolean> => {
        const status = await readStatus();
        if (status === null) return false;
        const enc = JSON.stringify(status);
        if (enc === lastStatus) return false;
        lastStatus = enc;
        send('status', status);
        return status.complete;
      };

      const poll = async (): Promise<void> => {
        if (closed) return;
        polls++;
        const read = await readJsonlFrom<AggregateRow>(aggPath, offset);
        if (read !== null) {
          if (read.reset) offset = 0;
          offset = read.nextOffset;
          for (const row of read.rows) {
            if (!send('row', row)) return;
          }
        }
        const done = await pushStatus();
        if (done) {
          send('done', { id: target.id });
          finish();
          return;
        }
        if (polls % LIVE_KEEPALIVE_POLLS === 0) write(': keepalive\n\n');
      };

      const boot = async (): Promise<void> => {
        const initial = await readJsonlFrom<AggregateRow>(aggPath, 0);
        const rows = initial?.rows ?? [];
        offset = initial?.nextOffset ?? 0;
        const hello: LiveHello = { rows };
        if (!send('hello', hello)) return;
        const complete = await pushStatus();
        if (complete) {
          send('done', { id: target.id });
          finish();
          return;
        }
        timer = setInterval(() => {
          void poll();
        }, pollMs);
      };

      onAbort = (): void => finish();
      req.signal.addEventListener('abort', onAbort, { once: true });
      if (req.signal.aborted) {
        finish();
        return;
      }
      void boot();
    },
    cancel() {
      teardown();
    },
  });

  function teardown(): void {
    if (closed) return;
    closed = true;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    if (onAbort !== null) {
      try {
        req.signal.removeEventListener('abort', onAbort);
      } catch {
      }
      onAbort = null;
    }
  }

  return new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}
