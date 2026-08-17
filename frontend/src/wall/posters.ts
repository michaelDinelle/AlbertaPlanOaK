import { useEffect, useState } from 'react';
import { z } from 'zod';

export const SCENES_VERSION = 1;
const SETTLE_FRAMES = 30;
const WAIT_FRAMES = 600;
const POSTER_W = 320;
const POSTER_H = 180;

const posterSchema = z.object({
  envId: z.string(),
  version: z.number(),
  capturedAt: z.number(),
  blob: z.instanceof(Blob),
});

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('oak-posters', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('posters', { keyPath: 'envId' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('poster store unavailable'));
  });
}

async function withStore<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    const request = work(db.transaction('posters', mode).objectStore('posters'));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('poster store failed'));
  }).finally(() => db.close());
}

export async function readPoster(envId: string): Promise<Blob | null> {
  const raw = await withStore('readonly', (store) => store.get(envId)).catch(() => null);
  const parsed = posterSchema.safeParse(raw);
  if (!parsed.success || parsed.data.version !== SCENES_VERSION) return null;
  return parsed.data.blob;
}

const listeners = new Set<() => void>();

let generation = 0;

export function capturePoster(getStage: () => { envId: string; canvas: HTMLCanvasElement } | null): void {
  const token = ++generation;
  let settle = SETTLE_FRAMES;
  let patience = WAIT_FRAMES;
  const tick = (): void => {
    if (token !== generation) return;
    const stage = getStage();
    if (stage === null || stage.canvas.width === 0) {
      if (--patience > 0) requestAnimationFrame(tick);
      return;
    }
    if (--settle > 0) {
      requestAnimationFrame(tick);
      return;
    }
    const target = document.createElement('canvas');
    target.width = POSTER_W;
    target.height = POSTER_H;
    const ctx = target.getContext('2d');
    if (ctx === null) return;
    const { width, height } = stage.canvas;
    const scale = Math.max(POSTER_W / width, POSTER_H / height);
    const cw = POSTER_W / scale;
    const ch = POSTER_H / scale;
    ctx.drawImage(stage.canvas, (width - cw) / 2, (height - ch) / 2, cw, ch, 0, 0, POSTER_W, POSTER_H);
    target.toBlob(
      (blob) => {
        if (blob === null) return;
        void withStore('readwrite', (store) =>
          store.put({ envId: stage.envId, version: SCENES_VERSION, capturedAt: Date.now(), blob }),
        )
          .then(() => {
            for (const listener of listeners) listener();
          })
          .catch(() => {});
      },
      'image/webp',
      0.85,
    );
  };
  requestAnimationFrame(tick);
}

export function usePosterUrl(envId: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (envId === null) {
      setUrl(null);
      return;
    }
    let alive = true;
    let objectUrl: string | null = null;
    const load = (): void => {
      void readPoster(envId).then((blob) => {
        if (!alive) return;
        if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
        objectUrl = blob === null ? null : URL.createObjectURL(blob);
        setUrl(objectUrl);
      });
    };
    listeners.add(load);
    load();
    return () => {
      alive = false;
      listeners.delete(load);
      if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
    };
  }, [envId]);
  return url;
}
