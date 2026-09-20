import { SIM_EVENT_TYPES, type EventBlock } from '@sim/types';

export const TICK_KINDS = ['goal', 'collection', 'door', 'key', 'cue', 'teleport', 'resurface', 'done'] as const;
export type TickKind = (typeof TICK_KINDS)[number];
export interface Tick { kind: TickKind; step: number; position: number; count: number; label: string }

export function tickAccumulator(totalSteps: number, buckets = 180) {
  const first = new Int32Array(buckets).fill(-1), ranks = TICK_KINDS.length, tally = new Int32Array(buckets * ranks);
  return {
    add(block: EventBlock) {
      for (let i = 0; i < block.steps.length; i++) {
        const type = SIM_EVENT_TYPES[block.kinds[i]!]!, rank = (TICK_KINDS as readonly string[]).indexOf(type);
        if (rank < 0) continue;
        const step = block.steps[i]!, slot = Math.min(buckets - 1, Math.floor(step / Math.max(1, totalSteps) * buckets));
        if (first[slot] === -1 || first[slot]! > step) first[slot] = step;
        tally[slot * ranks + rank]!++;
      }
    },
    ticks(): Tick[] {
      const out: Tick[] = [];
      for (let slot = 0; slot < buckets; slot++) {
        if (first[slot] === -1) continue;
        const parts: string[] = []; let count = 0, head: TickKind | null = null;
        for (let r = 0; r < ranks; r++) {
          const n = tally[slot * ranks + r]!;
          if (!n) continue;
          head ??= TICK_KINDS[r]!; count += n; parts.push(`${n}× ${TICK_KINDS[r]}`);
        }
        out.push({ kind: head!, step: first[slot]!, position: slot / (buckets - 1), count, label: `${parts.join(' · ')} — step ${first[slot]}` });
      }
      return out;
    },
  };
}
