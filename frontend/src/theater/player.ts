
import { MAX_CATCHUP_FRAMES } from './playbackSource';
import type { SimEvent } from '@sim/types';


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

