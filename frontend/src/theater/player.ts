// theater/player — transport policy and the frame governor.
//
// Everything the Theater decides WITHOUT a GPU lives here: what a key press
// means, what the next speed is, how a scrub position maps to a frame, whether
// the last two seconds of frame times justify dropping a quality tier, and what
// the event ticker should say. `sceneHost` owns the WebGL; this file owns the
// rules, so all of it is unit-testable.
//
// The playback clock itself is NOT re-implemented: `kit/harness.makePlaybackClock`
// is the proven one and `sceneHost` uses it directly. Same for the scrub-vs-catchup
// rule — `MAX_CATCHUP_FRAMES` is imported, never re-guessed.

import { MAX_CATCHUP_FRAMES, TIERS } from '@kit/harness';
import type { QualityTier } from '@kit/types';
import type { SimEvent } from '@sim/types';
import { describeEvent } from './loader';

// ---------------------------------------------------------------------------
// speed
// ---------------------------------------------------------------------------

/** Transport speeds. The brief's range: 0.25× to 8×. */
export const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4, 8];

/** Index of the listed speed closest to `speed` (multiplicatively). */
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

/** One step along the speed ladder, clamped at both ends. */
export function nextSpeed(speed: number, direction: 1 | -1): number {
  const index = nearestSpeedIndex(speed);
  const next = Math.min(SPEEDS.length - 1, Math.max(0, index + direction));
  return SPEEDS[next] ?? 1;
}

// ---------------------------------------------------------------------------
// tiers
// ---------------------------------------------------------------------------

/** hero -> medium -> thumb -> hero; the `t` key. */
export function cycleTier(tier: QualityTier): QualityTier {
  const index = TIERS.indexOf(tier);
  return TIERS[(index + 1) % TIERS.length] ?? 'hero';
}

/** The next tier DOWN, or null at the bottom. */
export function lowerTier(tier: QualityTier): QualityTier | null {
  const index = TIERS.indexOf(tier);
  if (index < 0 || index >= TIERS.length - 1) return null;
  return TIERS[index + 1] ?? null;
}

// ---------------------------------------------------------------------------
// the tier governor (STYLE.md: quality-tier selection + frame governor is a HOST duty)
// ---------------------------------------------------------------------------

/**
 * One 60 Hz refresh. The display's own interval is estimated from the FASTEST
 * frame in the window, capped here so a scene that never once hits refresh
 * cannot talk its way into an arbitrarily lenient threshold.
 */
export const REFRESH_CAP_MS = 1000 / 60;

export interface TierGovernorOptions {
  tier?: QualityTier;
  /** frame-time ceiling in ms; 20 ms ≈ 50 fps */
  budgetMs?: number;
  /** how long p95 must stay over budget before a drop */
  sustainSeconds?: number;
  /** rolling sample window */
  window?: number;
  /** samples required before the window is judged at all */
  minSamples?: number;
  /** the percentile watched */
  percentile?: number;
  /** samples above this are discarded (tab hidden, GC pause, breakpoint) */
  maxSampleMs?: number;
  /** missed-vsync multiple; 0 disables the vsync-relative floor */
  vsyncFactor?: number;
}

export interface TierGovernor {
  readonly tier: QualityTier;
  /** last computed percentile in ms; 0 until the window is warm */
  readonly p95: number;
  /** the ms a frame must exceed to count as over budget; 0 until warm */
  readonly threshold: number;
  /** consecutive seconds spent over budget */
  readonly overSeconds: number;
  /** how many automatic drops have happened */
  readonly drops: number;
  /** true once the bottom tier is over budget too — nothing left to give */
  readonly starving: boolean;
  /** feed one rendered frame (ms). Returns the tier to rebuild at, or null to hold. */
  sample(frameMs: number): QualityTier | null;
  /** the user picked a tier: adopt it and start the window over */
  adopt(tier: QualityTier): void;
  /** forget the window (after a rebuild — those frames are warmup, not steady state) */
  reset(): void;
}

/**
 * Drop a tier when p95 frame time stays over budget for `sustainSeconds`.
 *
 * Four guards keep it from firing on noise:
 *   * p95, not mean — a single 60 ms hitch every second is a hitch, not a budget failure.
 *   * a warm window (`minSamples`) — the frames right after a scene build are
 *     shader compiles and texture uploads, never steady state.
 *   * `maxSampleMs` — a backgrounded tab produces one 4000 ms "frame"; counting
 *     it would drop the tier of every scene the moment you switch away and back.
 *   * a VSYNC-RELATIVE floor. The sample is a frame INTERVAL, so a perfectly
 *     healthy vsync-locked page already measures p95 ≈ 17–18.6 ms (measured) —
 *     only ~7% under a bare 20 ms rule. The real question is "are we missing
 *     refreshes", so the threshold is `max(budgetMs, vsyncFactor × refresh)`
 *     with `refresh` the window's fastest frame capped at one 60 Hz interval:
 *     ~25 ms on a 60 Hz display, still exactly 20 ms at 120 Hz.
 *
 * It never raises a tier. Automatic quality that oscillates is worse than
 * automatic quality that only ever gives up: the way back up is the `t` key.
 */
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

  /** p95 and the fastest frame, from one sort of the window. */
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
        // Nothing left to drop. Pin the accumulator so it cannot grow without
        // bound and flag the condition for the HUD.
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

// ---------------------------------------------------------------------------
// frame cursor — sequential playback vs a scrub
// ---------------------------------------------------------------------------

export interface FrameCursor {
  /** frame index rendered last tick; -1 before the first */
  lastIndex: number;
  hasPrevious: boolean;
}

export function makeFrameCursor(): FrameCursor {
  return { lastIndex: -1, hasPrevious: false };
}

export interface CursorStep {
  index: number;
  /** true when the jump is small enough to walk — events are delivered only then */
  sequential: boolean;
  /** inclusive range of frames whose events fire this tick (empty when scrubbed) */
  from: number;
  to: number;
}

/**
 * Advance the cursor to `position`, using the harness's own catch-up rule:
 * up to MAX_CATCHUP_FRAMES frames are WALKED (so 8× playback still fires every
 * event), and anything further is a scrub — snap, no celebrations.
 */
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

/**
 * Collect the events of frames [from, to] into `out` (cleared first).
 *
 * `outFrames`, when given, receives the FRAME each event came from — one entry
 * per event, parallel to `out`. A tick at 8× walks up to twelve frames, so
 * without it every event in a catch-up burst would be attributed to whichever
 * frame the tick happened to land on.
 */
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

// ---------------------------------------------------------------------------
// keyboard
// ---------------------------------------------------------------------------

export type TransportAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'speed'; direction: 1 | -1 }
  | { kind: 'debug' }
  | { kind: 'tier' }
  | { kind: 'seek'; position: 'start' | 'end' };

/**
 * The Theater's keymap: space play/pause, arrows step, `[` `]` speed, `g` debug,
 * `t` tier. Shift+arrow steps ten frames; Home/End jump the ends.
 */
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

// ---------------------------------------------------------------------------
// readouts
// ---------------------------------------------------------------------------

/** Playback seconds as `m:ss.s` (or `h:mm:ss.s` past an hour). */
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

/** Clamp a scrub value into a valid frame position. */
export function clampPosition(position: number, frameCount: number): number {
  const last = Math.max(0, frameCount - 1);
  if (!Number.isFinite(position)) return 0;
  return position < 0 ? 0 : position > last ? last : position;
}

/** Frame-accurate step: floor to a frame first, then move (so `+1` from 4.7 is 5). */
export function stepPosition(position: number, frames: number, frameCount: number): number {
  const base = Math.floor(clampPosition(position, frameCount));
  return clampPosition(base + frames, frameCount);
}

// ---------------------------------------------------------------------------
// event ticker
// ---------------------------------------------------------------------------

export interface TickerEntry {
  step: number;
  label: string;
  type: SimEvent['type'];
}

export const TICKER_LIMIT = 8;

/**
 * Append events to the ticker, newest LAST, capped at `limit`.
 *
 * `steps` is either one step for the whole delivery or a per-event array — a
 * catch-up tick can carry events from several different steps and printing them
 * all under the last one would misdate the record.
 *
 * Returns a NEW array (the view re-renders from it; mutation would defeat that),
 * except for an empty delivery, which returns the same list so an idle frame
 * costs no re-render.
 */
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
