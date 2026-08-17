import type { RunIndexEntry } from '@shared/types';
import { formatStep } from '../../lib/format';
import { navigate } from '../../url/useUrlState';
import { RunSwitcher } from './RunSwitcher';

export function TopBar({
  entries,
  entry,
}: {
  entries: readonly RunIndexEntry[] | undefined;
  entry: RunIndexEntry | null;
}) {
  return (
    <header className="glass sticky top-0 z-40 flex h-12 items-center gap-4 rounded-none border-x-0 border-t-0 px-4">
      <button
        type="button"
        onClick={() => navigate({})}
        className="flex items-baseline gap-1.5 rounded-sm px-1 hover:opacity-80"
        title="OaK Preserve"
      >
        <span className="size-2.5 self-center rounded-full bg-accent shadow-[0_0_8px_var(--color-accent)]" />
        <span className="font-mono text-sm font-semibold tracking-wide">OaK</span>
        <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">Preserve</span>
      </button>
      <RunSwitcher entries={entries ?? []} entry={entry} />
      {entry?.inFlight === true && (
        <span className="flex items-center gap-1.5 text-2xs uppercase tracking-[0.14em] text-good">
          <span className="size-1.5 animate-pulse rounded-full bg-good" />
          live
        </span>
      )}
      <span className="ml-auto hidden font-mono text-2xs text-ink-faint sm:block">
        {entry === null ? '' : `${entry.envIds.length} envs · ${formatStep(entry.steps)} steps`}
      </span>
    </header>
  );
}
