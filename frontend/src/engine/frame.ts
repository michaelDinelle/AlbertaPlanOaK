import type { EnvEntry, NamedSlot } from '@sim/catalog';
import type { LaneGeometry } from '@sim/geometry';
import type { FrameState } from '@kit/types';

export interface SemanticFrame {
  readonly entry: EnvEntry;
  readonly geometry: LaneGeometry;
  readonly state: FrameState;
  readonly step: number;
  readonly done: boolean;
  has(slot: NamedSlot): boolean;
  get(slot: NamedSlot): number;
  cell(slot: 'agent' | 'goal' | 'key', out: { x: number; y: number }): { x: number; y: number };
  bit(slot: 'gates' | 'tokens' | 'carry', index: number): boolean;
  effort(): number;
}

const CELL_SLOTS: Record<'agent' | 'goal' | 'key', readonly [NamedSlot, NamedSlot]> = {
  agent: ['x', 'y'],
  goal: ['gx', 'gy'],
  key: ['kx', 'ky'],
};

export function makeSemanticFrame(entry: EnvEntry, geometry: LaneGeometry): {
  frame: SemanticFrame;
  bind(state: FrameState): void;
} {
  let current: FrameState | null = null;

  const read = (slot: NamedSlot): number => {
    const at = entry.slotAt[slot];
    if (at === undefined || current === null) return Number.NaN;
    return current.pose[at] ?? Number.NaN;
  };

  const clampX = (v: number): number => {
    if (!Number.isFinite(v)) return 0;
    const n = Math.round(v);
    return n < 0 ? 0 : n > geometry.w - 1 ? geometry.w - 1 : n;
  };
  const clampY = (v: number): number => {
    if (!Number.isFinite(v)) return 0;
    const n = Math.round(v);
    return n < 0 ? 0 : n > geometry.h - 1 ? geometry.h - 1 : n;
  };

  const frame: SemanticFrame = {
    entry,
    geometry,
    get state() {
      if (current === null) throw new Error('semantic frame read before bind');
      return current;
    },
    get step() {
      return current?.step ?? -1;
    },
    get done() {
      return current?.done === true;
    },
    has(slot) {
      return entry.slotAt[slot] !== undefined;
    },
    get(slot) {
      return read(slot);
    },
    cell(which, out) {
      const pair = CELL_SLOTS[which];
      out.x = clampX(read(pair[0]));
      out.y = clampY(read(pair[1]));
      return out;
    },
    bit(slot, index) {
      const v = read(slot);
      if (!Number.isFinite(v)) return false;
      return ((Math.round(v) >> index) & 1) === 1;
    },
    effort() {
      const e = read('effort');
      if (Number.isFinite(e)) return e;
      const ctrl = current?.ctrl[0];
      return ctrl === undefined || !Number.isFinite(ctrl) ? 0 : ctrl;
    },
  };

  return {
    frame,
    bind(state) {
      current = state;
    },
  };
}
