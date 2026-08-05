
import { MAX_CATCHUP_FRAMES, TIERS } from '@kit/harness';
import type { QualityTier } from '@kit/types';
import type { SimEvent } from '@sim/types';
import { describeEvent } from './loader';


export const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4, 8];

export function nearestSpeedIndex(speed: number): number {
  let best = 0;
  let bestError = Infinity;
  for (let i = 0; i < SPEEDS.length; i++) {
    const candidate = SPEEDS[i] ?? 1;
    const error = Math.abs(Math.log(candidate) - Math.log(speed > 0 ? speed : 1));
    if (error < bestError) {
      bestError = error;
      best = i;
    }
  }
  return best;
}

export function nextSpeed(speed: number, direction: 1 | -1): number {
  const index = nearestSpeedIndex(speed);
  const next = Math.min(SPEEDS.length - 1, Math.max(0, index + direction));
  return SPEEDS[next] ?? 1;
}


export function cycleTier(tier: QualityTier): QualityTier {
  const index = TIERS.indexOf(tier);
  return TIERS[(index + 1) % TIERS.length] ?? 'hero';
}

export function lowerTier(tier: QualityTier): QualityTier | null {
  const index = TIERS.indexOf(tier);
  if (index < 0 || index >= TIERS.length - 1) return null;
  return TIERS[index + 1] ?? null;
}


export const REFRESH_CAP_MS = 1000 / 60;

export interface TierGovernorOptions {
  tier?: QualityTier;
  budgetMs?: number;
  sustainSeconds?: number;
  window?: number;
  minSamples?: number;
  percentile?: number;
  maxSampleMs?: number;
  vsyncFactor?: number;
}

export interface TierGovernor {
  readonly tier: QualityTier;
  readonly p95: number;
  readonly threshold: number;
  readonly overSeconds: number;
  readonly drops: number;
  readonly starving: boolean;
  sample(frameMs: number): QualityTier | null;
  adopt(tier: QualityTier): void;
  reset(): void;
}

export function makeTierGovernor(options: TierGovernorOptions = {}): TierGovernor {
  const budgetMs = options.budgetMs ?? 20;
  const sustainSeconds = options.sustainSeconds ?? 2;
  const window = Math.max(8, Math.floor(options.window ?? 120));
  const minSamples = Math.max(2, Math.floor(options.minSamples ?? Math.min(30, window)));
  const percentile = options.percentile ?? 0.95;
  const maxSampleMs = options.maxSampleMs ?? 250;
  const vsyncFactor = options.vsyncFactor ?? 1.5;

  const ring = new Float64Array(window);
  const scratch = new Float64Array(window);
  let count = 0;
  let head = 0;

  let tier: QualityTier = options.tier ?? 'hero';
  let p95 = 0;
  let threshold = 0;
  let overSeconds = 0;
  let drops = 0;
  let starving = false;

  const clear = (): void => {
    count = 0;
    head = 0;
    p95 = 0;
    threshold = 0;
    overSeconds = 0;
  };

  const measure = (): { p95: number; floor: number } => {
    const n = Math.min(count, window);
    for (let i = 0; i < n; i++) scratch[i] = ring[i] ?? 0;
    const view = scratch.subarray(0, n);
    view.sort();
    const index = Math.min(n - 1, Math.max(0, Math.ceil(percentile * n) - 1));
    return { p95: view[index] ?? 0, floor: view[0] ?? 0 };
  };

  const governor: TierGovernor = {
    get tier() {
      return tier;
    },
    get p95() {
      return p95;
    },
    get threshold() {
      return threshold;
    },
    get overSeconds() {
      return overSeconds;
    },
    get drops() {
      return drops;
    },
    get starving() {
      return starving;
    },
    sample(frameMs) {
      if (!Number.isFinite(frameMs) || frameMs <= 0 || frameMs > maxSampleMs) return null;
      ring[head] = frameMs;
      head = (head + 1) % window;
      count++;
      if (count < minSamples) return null;

      const stats = measure();
      p95 = stats.p95;
      threshold = Math.max(budgetMs, vsyncFactor * Math.min(stats.floor, REFRESH_CAP_MS));
      if (p95 <= threshold) {
        overSeconds = 0;
        return null;
      }
      overSeconds += frameMs / 1000;
      if (overSeconds < sustainSeconds) return null;

      const next = lowerTier(tier);
      if (!next) {
        overSeconds = sustainSeconds;
        starving = true;
        return null;
      }
      tier = next;
      drops++;
      clear();
      return tier;
    },
    adopt(next) {
      tier = next;
      starving = false;
      clear();
    },
    reset() {
      clear();
    },
  };
  return governor;
}


export interface FrameCursor {
  lastIndex: number;
  hasPrevious: boolean;
}

export function makeFrameCursor(): FrameCursor {
  return { lastIndex: -1, hasPrevious: false };
}

export interface CursorStep {
  index: number;
  sequential: boolean;
  from: number;
  to: number;
}

export function advanceCursor(cursor: FrameCursor, position: number, frameCount: number): CursorStep {
  const last = Math.max(0, frameCount - 1);
  const index = Math.min(last, Math.max(0, Math.floor(position)));
  const delta = index - cursor.lastIndex;
  const sequential = cursor.hasPrevious && delta >= 0 && delta <= MAX_CATCHUP_FRAMES;
  const step: CursorStep = {
    index,
    sequential,
    from: sequential ? cursor.lastIndex + 1 : index + 1,
    to: sequential ? index : index,
  };
  cursor.lastIndex = index;
  cursor.hasPrevious = true;
  return step;
}

export function collectEvents(
  eventsAt: (i: number) => readonly SimEvent[],
  from: number,
  to: number,
  out: SimEvent[],
  outFrames?: number[],
): SimEvent[] {
  out.length = 0;
  if (outFrames) outFrames.length = 0;
  for (let i = from; i <= to; i++) {
    const frameEvents = eventsAt(i);
    for (const event of frameEvents) {
      out.push(event);
      outFrames?.push(i);
    }
  }
  return out;
}


export type TransportAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'speed'; direction: 1 | -1 }
  | { kind: 'debug' }
  | { kind: 'tier' }
  | { kind: 'seek'; position: 'start' | 'end' };

export function keyAction(key: string, shift = false): TransportAction | null {
  switch (key) {
    case ' ':
    case 'Spacebar':
    case 'k':
    case 'K':
      return { kind: 'toggle' };
    case 'ArrowRight':
    case 'l':
    case 'L':
      return { kind: 'step', frames: shift ? 10 : 1 };
    case 'ArrowLeft':
    case 'j':
    case 'J':
      return { kind: 'step', frames: shift ? -10 : -1 };
    case '[':
    case '{':
      return { kind: 'speed', direction: -1 };
    case ']':
    case '}':
      return { kind: 'speed', direction: 1 };
    case 'g':
    case 'G':
      return { kind: 'debug' };
    case 't':
    case 'T':
      return { kind: 'tier' };
    case 'Home':
      return { kind: 'seek', position: 'start' };
    case 'End':
      return { kind: 'seek', position: 'end' };
    default:
      return null;
  }
}


export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
  const whole = Math.floor(seconds);
  const tenths = Math.floor((seconds - whole) * 10);
  const s = whole % 60;
  const m = Math.floor(whole / 60) % 60;
  const h = Math.floor(whole / 3600);
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m);
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}.${tenths}` : `${mm}:${ss}.${tenths}`;
}

export function clampPosition(position: number, frameCount: number): number {
  const last = Math.max(0, frameCount - 1);
  if (!Number.isFinite(position)) return 0;
  return position < 0 ? 0 : position > last ? last : position;
}

export function stepPosition(position: number, frames: number, frameCount: number): number {
  const base = Math.floor(clampPosition(position, frameCount));
  return clampPosition(base + frames, frameCount);
}


export interface TickerEntry {
  step: number;
  label: string;
  type: SimEvent['type'];
}

export const TICKER_LIMIT = 8;

export function pushTicker(
  ticker: readonly TickerEntry[],
  events: readonly SimEvent[],
  steps: number | readonly number[],
  limit: number = TICKER_LIMIT,
): TickerEntry[] {
  if (events.length === 0) return ticker as TickerEntry[];
  const next = [...ticker];
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (!event) continue;
    const step = typeof steps === 'number' ? steps : steps[i] ?? -1;
    next.push({ step, label: describeEvent(event), type: event.type });
  }
  return next.length > limit ? next.slice(next.length - limit) : next;
}
