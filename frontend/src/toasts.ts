import { toast } from 'sonner';
import type { RunIndexEntry } from './data/types';
import { runLabel } from './lib/run';
import { STR } from './strings';

let known: Map<string, RunIndexEntry> | null = null;

export function watchRuns(entries: readonly RunIndexEntry[] | undefined): void {
  if (entries === undefined) return;
  const next = new Map(entries.map((entry) => [entry.id, entry]));
  if (known !== null) {
    for (const entry of entries) {
      const before = known.get(entry.id);
      if (before === undefined) {
        toast(STR.toasts.appeared(runLabel(entry)));
        continue;
      }
      if (before.inFlight && !entry.inFlight) toast.success(STR.toasts.finished(runLabel(entry)));
      if (!before.hasSummary && entry.hasSummary) toast(STR.toasts.scored(runLabel(entry)));
    }
  }
  known = next;
}
