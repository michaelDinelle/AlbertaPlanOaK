import type { EventStream } from '@sim/types';
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
    const collection = stream.types.indexOf('collection');
    while (this.taken < stream.count) {
      const at = this.taken;
      this.taken += 1;
      if (collection < 0 || (stream.kinds[at] ?? 0) !== collection) continue;
      if (this.count === this.steps.length) {
        const grown = this.count * 2;
        const steps = new Int32Array(grown);
        const xs = new Int32Array(grown);
        const ys = new Int32Array(grown);
        steps.set(this.steps);
        xs.set(this.xs);
        ys.set(this.ys);
        this.steps = steps;
        this.xs = xs;
        this.ys = ys;
      }
      const x = stream.cellX[at] ?? 0;
      const y = stream.cellY[at] ?? 0;
      const step = stream.steps[at] ?? 0;
      this.steps[this.count] = step;
      this.xs[this.count] = x;
      this.ys[this.count] = y;
      this.count += 1;
      this.cells.set(cellKey(x, y), step);
      changed = true;
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
