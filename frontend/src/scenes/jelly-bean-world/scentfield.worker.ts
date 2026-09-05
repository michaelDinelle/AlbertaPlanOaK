import { bundleKernel, buildScentKernel, cellKey, renderTile } from './fieldmath';

export interface FieldJob { id: number; x: number; y: number; step: number; stamp: string }
export interface FieldTile extends FieldJob { rgba: ArrayBuffer }
export type FieldRequest = { op: 'init'; seed: number } | { op: 'log'; from: number; steps: Int32Array; xs: Int32Array; ys: Int32Array } | { op: 'tile'; job: FieldJob };

let seed = 0, count = 0;
const collected = new Map<number, number>();
const kernel = bundleKernel(buildScentKernel());
onmessage = (event: MessageEvent<FieldRequest>) => {
  const message = event.data;
  if (message.op === 'init') { seed = message.seed; return; }
  if (message.op === 'log') {
    if (message.from === 0) { collected.clear(); count = 0; }
    if (message.from !== count) throw new Error('Scent collection log is not contiguous');
    for (let i = 0; i < message.steps.length; i++) collected.set(cellKey(message.xs[i]!, message.ys[i]!), message.steps[i]!);
    count += message.steps.length;
    return;
  }
  const job = message.job, rgba = renderTile(seed, job.x, job.y, job.step, collected, kernel).buffer;
  postMessage({ ...job, rgba } satisfies FieldTile, { transfer: [rgba] });
};
