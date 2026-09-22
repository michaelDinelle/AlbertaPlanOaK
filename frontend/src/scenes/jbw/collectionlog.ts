import { SIM_EVENT_TYPES, type EventStream } from '@sim/types';
import { cellKey, type FieldLog } from './fieldmath';

export class CollectionLog {
  version = 0;

  private stream: EventStream | null = null;
  private taken = 0;
  private steps = new Int32Array(1024);
  private xs = new Int32Array(1024);
  private ys = new Int32Array(1024);
  private count = 0;
  private readonly cells = new Map<number, number>();
  private readonly view: FieldLog & { steps: Int32Array; xs: Int32Array; ys: Int32Array };

  constructor() {
    this.view = { count: 0, steps: this.steps, xs: this.xs, ys: this.ys };
  }

  get log(): FieldLog {
    return this.view;
  }

  get collected(): ReadonlyMap<number, number> {
    return this.cells;
  }

  sync(stream: EventStream): boolean {
    let changed = false;
    if (stream !== this.stream) {
      this.stream = stream;
      this.taken = 0;
      this.count = 0;
      this.cells.clear();
      this.version += 1;
      changed = true;
    }
    const collection = SIM_EVENT_TYPES.indexOf('collection');
    while (stream.blocks[this.taken] !== undefined) {
      const block = stream.blocks[this.taken++]!;
      for (let at = 0; at < block.steps.length; at++) {
        if (block.kinds[at] !== collection) continue;
        if (this.count === this.steps.length) {
          const grow = (data: Int32Array): Int32Array<ArrayBuffer> => { const next = new Int32Array(data.length * 2); next.set(data); return next; };
          this.steps = grow(this.steps); this.xs = grow(this.xs); this.ys = grow(this.ys);
        }
        const x = block.cellX[at]!, y = block.cellY[at]!, step = block.steps[at]!;
        this.steps[this.count] = step; this.xs[this.count] = x; this.ys[this.count] = y;
        this.count++; this.cells.set(cellKey(x, y), step); changed = true;
      }
    }
    if (changed) {
      this.view.steps = this.steps;
      this.view.xs = this.xs;
      this.view.ys = this.ys;
      this.view.count = this.count;
    }
    return changed;
  }

  depleted(x: number, y: number, step: number): boolean {
    const at = this.cells.get(cellKey(x, y));
    return at !== undefined && at <= step;
  }
}
