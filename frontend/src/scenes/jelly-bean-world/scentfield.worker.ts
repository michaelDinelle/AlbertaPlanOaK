import {
  bundleKernel,
  buildScentKernel,
  cellKey,
  renderTile,
  TILE_BYTES,
  type KernelBundle,
  type TilePalette,
} from './fieldmath';

export interface FieldInit {
  op: 'init';
  seed: number;
  palette: TilePalette;
}

export interface FieldAppend {
  op: 'log';
  from: number;
  steps: Int32Array;
  xs: Int32Array;
  ys: Int32Array;
}

export interface FieldWantTile {
  key: string;
  tileX: number;
  tileY: number;
  stamp: string;
}

export interface FieldWant {
  op: 'want';
  playhead: number;
  tiles: FieldWantTile[];
}

export type FieldRequest = FieldInit | FieldAppend | FieldWant;

export interface FieldTile {
  op: 'tile';
  key: string;
  tileX: number;
  tileY: number;
  stamp: string;
  rgba: ArrayBuffer;
}

const SLICE_MS = 8;

let seed = 0;
let palette: TilePalette | null = null;
let kernel: KernelBundle | null = null;
const collected = new Map<number, number>();
let logCount = 0;
let queue: FieldWantTile[] = [];
let cursor = 0;
let playhead = 0;
let draining = false;

const wake = new MessageChannel();
wake.port1.onmessage = () => {
  draining = false;
  drain();
};

function drain(): void {
  if (draining) return;
  draining = true;
  const started = performance.now();
  while (cursor < queue.length) {
    const tile = queue[cursor];
    cursor += 1;
    if (!tile || !kernel || !palette) continue;
    const out = new Uint8Array(TILE_BYTES);
    renderTile(seed, tile.tileX, tile.tileY, playhead, collected, kernel, palette, out);
    const reply: FieldTile = {
      op: 'tile',
      key: tile.key,
      tileX: tile.tileX,
      tileY: tile.tileY,
      stamp: tile.stamp,
      rgba: out.buffer,
    };
    postMessage(reply, { transfer: [out.buffer] });
    if (performance.now() - started > SLICE_MS) {
      wake.port2.postMessage(0);
      return;
    }
  }
  draining = false;
}

onmessage = (event: MessageEvent<FieldRequest>) => {
  const msg = event.data;
  if (msg.op === 'init') {
    seed = msg.seed;
    palette = msg.palette;
    kernel ??= bundleKernel(buildScentKernel());
    collected.clear();
    logCount = 0;
    queue = [];
    cursor = 0;
    return;
  }
  if (msg.op === 'log') {
    if (msg.from !== logCount) {
      collected.clear();
      logCount = 0;
      if (msg.from !== 0) return;
    }
    for (let i = 0; i < msg.steps.length; i++) {
      collected.set(cellKey(msg.xs[i] ?? 0, msg.ys[i] ?? 0), msg.steps[i] ?? 0);
    }
    logCount += msg.steps.length;
    return;
  }
  playhead = msg.playhead;
  queue = msg.tiles;
  cursor = 0;
  drain();
};
