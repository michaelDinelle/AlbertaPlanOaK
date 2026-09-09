import type { Policy } from '../data/types';
import { currentPosition, subscribePosition } from '../playback/store';
import type { CumulativeReward } from './derive';
import { penCurves, type PenCurves } from './pen';
import type { RewardsCache } from '../data/rewards';

export type PenKey = Policy | 'best';
export const PEN_KEYS: readonly PenKey[] = ['oak', 'ppo', 'random', 'best'];
const DETAIL_STEPS = 65536;
const DETAIL_LOOKAHEAD = 4096;
const AXIS_INTERVAL_MS = 250;
const AXIS_TOLERANCE = 0.04;

export interface PenSource {
  cumulative: CumulativeReward;
  stream: RewardsCache | null;
}

export interface PenInputs {
  active: boolean;
  win: number;
  samples: number;
  primary: PenKey;
  sources: Record<PenKey, PenSource | null>;
  heroLane: number;
}

export interface PenFrame {
  index: number;
  step: number;
  pens: Record<PenKey, PenCurves | null>;
  xs: Float64Array;
  laneValues: Float64Array[];
}

export interface AxisState {
  xMax: number;
  yMin: number;
  yMax: number;
}

const EMPTY_SOURCES: Record<PenKey, PenSource | null> = { oak: null, ppo: null, random: null, best: null };

function detailAnchor(cumulative: CumulativeReward, head: number): number {
  const target = Math.max(0, head - DETAIL_STEPS);
  let anchor = 0;
  for (const end of cumulative.x) {
    if (end <= target) anchor = end;
    else break;
  }
  return anchor;
}

class PenStore {
  private inputs: PenInputs = { active: false, win: 1, samples: 256, primary: 'oak', sources: EMPTY_SOURCES, heroLane: -1 };
  private frameValue: PenFrame | null = null;
  private frameIndex = 0;
  private laneBuffers: Float64Array[] = [];
  private xsBuffer = new Float64Array(0);
  private axisValue: AxisState = { xMax: 1, yMin: -1, yMax: 1 };
  private axisAt = 0;
  private pending = 0;
  private unsubscribePosition: (() => void) | null = null;
  private streamUnsubscribes: (() => void)[] = [];
  private readonly frameListeners = new Set<() => void>();
  private readonly axisListeners = new Set<() => void>();

  setInputs(inputs: PenInputs): void {
    this.inputs = inputs;
    for (const off of this.streamUnsubscribes) off();
    this.streamUnsubscribes = [];
    if (inputs.active) {
      for (const key of PEN_KEYS) {
        const stream = inputs.sources[key]?.stream ?? null;
        if (stream !== null) this.streamUnsubscribes.push(stream.subscribe(() => this.schedule()));
      }
      if (this.unsubscribePosition === null) this.unsubscribePosition = subscribePosition(() => this.schedule());
      this.schedule();
    } else {
      this.unsubscribePosition?.();
      this.unsubscribePosition = null;
      if (this.pending !== 0) cancelAnimationFrame(this.pending);
      this.pending = 0;
      if (this.frameValue !== null) {
        this.frameValue = null;
        for (const listener of this.frameListeners) listener();
      }
    }
  }

  frame(): PenFrame | null {
    return this.frameValue;
  }

  axis(): AxisState {
    return this.axisValue;
  }

  subscribeFrame(listener: () => void): () => void {
    this.frameListeners.add(listener);
    return () => {
      this.frameListeners.delete(listener);
    };
  }

  subscribeAxis(listener: () => void): () => void {
    this.axisListeners.add(listener);
    return () => {
      this.axisListeners.delete(listener);
    };
  }

  private schedule(): void {
    if (this.pending !== 0 || !this.inputs.active) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      this.compute();
    });
  }

  private compute(): void {
    const inputs = this.inputs;
    const step = Math.max(0, currentPosition());
    const pens = { oak: null, ppo: null, random: null, best: null } as Record<PenKey, PenCurves | null>;
    for (const key of PEN_KEYS) {
      const source = inputs.sources[key];
      if (source === null) continue;
      const anchor = detailAnchor(source.cumulative, step);
      source.stream?.ensure(anchor, step + DETAIL_LOOKAHEAD);
      pens[key] = penCurves(source.cumulative, inputs.win, step, inputs.samples, source.stream?.detail(anchor) ?? null);
    }
    const primary = pens[inputs.primary];
    const sampleCount = primary === null ? 0 : primary.x.length;
    if (this.xsBuffer.length !== sampleCount) this.xsBuffer = new Float64Array(sampleCount);
    const xs = this.xsBuffer;
    for (let i = 0; i < sampleCount; i++) xs[i] = primary!.x[i] ?? 0;
    const laneCount = primary === null ? 0 : primary.perLane.length;
    if (this.laneBuffers.length !== laneCount) this.laneBuffers = Array.from({ length: laneCount }, () => new Float64Array(sampleCount));
    for (let li = 0; li < laneCount; li++) {
      let buffer = this.laneBuffers[li]!;
      if (buffer.length !== sampleCount) {
        buffer = new Float64Array(sampleCount);
        this.laneBuffers[li] = buffer;
      }
      const lane = primary!.perLane[li]!;
      for (let i = 0; i < sampleCount; i++) {
        const value = lane[i];
        buffer[i] = value === null || value === undefined ? NaN : value;
      }
    }
    this.frameIndex += 1;
    this.frameValue = { index: this.frameIndex, step, pens, xs, laneValues: this.laneBuffers };
    for (const listener of this.frameListeners) listener();
    const now = performance.now();
    if (now - this.axisAt >= AXIS_INTERVAL_MS) {
      this.axisAt = now;
      this.updateAxis(step, pens, inputs);
    }
  }

  private updateAxis(step: number, pens: Record<PenKey, PenCurves | null>, inputs: PenInputs): void {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    const take = (values: readonly (number | null)[] | undefined) => {
      if (values === undefined) return;
      for (const value of values) {
        if (value === null || !Number.isFinite(value)) continue;
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
    };
    for (const key of PEN_KEYS) {
      const pen = pens[key];
      if (pen === null) continue;
      take(pen.iqm);
      if (key === inputs.primary) {
        take(pen.p10);
        take(pen.p90);
        if (inputs.heroLane >= 0) take(pen.perLane[inputs.heroLane]);
      }
    }
    const current = this.axisValue;
    let next: AxisState;
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      next = { ...current, xMax: Math.max(1, step) };
    } else {
      const pad = Math.max((hi - lo) * 0.08, Math.abs(hi) * 1e-3, 1e-6);
      next = { xMax: Math.max(1, step), yMin: lo - pad, yMax: hi + pad };
    }
    const range = Math.max(current.yMax - current.yMin, 1e-12);
    const yMoved = Math.abs(next.yMin - current.yMin) > range * AXIS_TOLERANCE || Math.abs(next.yMax - current.yMax) > range * AXIS_TOLERANCE;
    const xMoved = Math.abs(next.xMax - current.xMax) > Math.max(1, current.xMax) * AXIS_TOLERANCE;
    if (!yMoved && !xMoved) return;
    this.axisValue = { xMax: next.xMax, yMin: yMoved ? next.yMin : current.yMin, yMax: yMoved ? next.yMax : current.yMax };
    for (const listener of this.axisListeners) listener();
  }
}

export const penStore = new PenStore();
