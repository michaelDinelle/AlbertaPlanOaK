import type { EventStream } from '@sim/types';

export const TICK_KINDS = ['goal', 'collection', 'door', 'key', 'cue', 'teleport', 'resurface', 'done'] as const;
export type TickKind = (typeof TICK_KINDS)[number];

export interface Tick {
  kind: TickKind;
  step: number;
  position: number;
  count: number;
  label: string;
}

export function ticksOf(events: EventStream, totalSteps: number, buckets = 180): Tick[] {
  const ranks = TICK_KINDS.length;
  const first = new Int32Array(buckets).fill(-1);
  const tally = new Int32Array(buckets * ranks);
  const rankOf = events.types.map((type) => (TICK_KINDS as readonly string[]).indexOf(type));
  for (let i = 0; i < events.count; i++) {
    const rank = rankOf[events.kinds[i] ?? 0] ?? -1;
    if (rank < 0) continue;
    const step = events.steps[i] ?? 0;
    const slot = Math.min(buckets - 1, Math.max(0, Math.round(Math.min(1, (step + 1) / Math.max(1, totalSteps)) * (buckets - 1))));
    if (first[slot] === -1) first[slot] = step;
    tally[slot * ranks + rank] = (tally[slot * ranks + rank] ?? 0) + 1;
  }
  const out: Tick[] = [];
  for (let slot = 0; slot < buckets; slot++) {
    const step = first[slot] ?? -1;
    if (step < 0) continue;
    const parts: string[] = [];
    let count = 0;
    let head: TickKind | null = null;
    for (let r = 0; r < ranks; r++) {
      const n = tally[slot * ranks + r] ?? 0;
      if (n === 0) continue;
      head ??= TICK_KINDS[r]!;
      count += n;
      parts.push(n > 1 ? `${n}× ${TICK_KINDS[r]}` : TICK_KINDS[r]!);
    }
    if (head !== null) out.push({ kind: head, step, position: slot / (buckets - 1), count, label: `${parts.join(' · ')} — step ${step}` });
  }
  return out;
}
