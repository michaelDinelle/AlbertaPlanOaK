import type uPlot from 'uplot';

export const ENV_SYNC_KEY = 'env';

export function cursorSync(key: string = ENV_SYNC_KEY): uPlot.Cursor.Sync {
  return { key, setSeries: false };
}

export function syncedCursor(key: string = ENV_SYNC_KEY): uPlot.Cursor {
  return { y: false, drag: { x: false, y: false, setScale: false }, sync: cursorSync(key) };
}
