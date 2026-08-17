import { useState } from 'react';
import { ChevronsUpDown } from 'lucide-react';
import { clsx } from 'clsx';
import type { RunIndexEntry } from '@shared/types';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/ui/popover';
import { formatAgo, formatStep } from '../../lib/format';
import { runLabel, sortRuns } from '../../lib/run';
import { selectRun } from '../../state/selectionStore';

function Badge({ tone, children }: { tone: 'keeper' | 'quiet' | 'warn'; children: string }) {
  return (
    <span
      className={clsx(
        'rounded-xs border px-1 py-px font-mono text-2xs leading-none',
        tone === 'keeper' && 'border-accent/40 bg-accent/10 text-accent',
        tone === 'warn' && 'border-bad/40 bg-bad/10 text-bad',
        tone === 'quiet' && 'border-line text-ink-dim',
      )}
    >
      {children}
    </span>
  );
}

export function RunSwitcher({
  entries,
  entry,
}: {
  entries: readonly RunIndexEntry[];
  entry: RunIndexEntry | null;
}) {
  const [open, setOpen] = useState(false);
  const [needle, setNeedle] = useState('');
  const query = needle.trim().toLowerCase();
  const rows = sortRuns(
    entries.filter((run) => query === '' || runLabel(run).toLowerCase().includes(query) || run.id.includes(query)),
  ).slice(0, 40);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className="flex items-center gap-2 rounded-sm border border-line bg-sunken/60 px-2.5 py-1 font-mono text-xs hover:border-line-strong hover:bg-hover"
        title="Switch run"
      >
        <span>{entry === null ? '—' : runLabel(entry)}</span>
        <ChevronsUpDown className="size-3 text-ink-faint" />
      </PopoverTrigger>
      <PopoverContent className="w-88 p-1.5">
        <input
          type="search"
          value={needle}
          onChange={(event) => setNeedle(event.target.value)}
          placeholder="find a run…"
          aria-label="Find a run"
          className="mb-1.5 w-full rounded-sm border border-line bg-sunken px-2 py-1 font-mono text-xs outline-none placeholder:text-ink-faint focus:border-accent"
        />
        <div className="flex max-h-80 flex-col gap-px overflow-y-auto">
          {rows.map((run) => (
            <button
              key={run.id}
              type="button"
              onClick={() => {
                setOpen(false);
                selectRun(run.id);
              }}
              className={clsx(
                'flex items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-hover',
                run.id === entry?.id && 'bg-hover/60',
              )}
            >
              {run.inFlight && <span className="size-1.5 animate-pulse rounded-full bg-good" />}
              <span className="font-mono">{runLabel(run)}</span>
              {run.role === 'keeper' && <Badge tone="keeper">keeper</Badge>}
              {run.ablationLabel !== 'none' && <Badge tone="warn">{`ablate ${run.ablationLabel}`}</Badge>}
              {!run.complete && <Badge tone="warn">incomplete</Badge>}
              <span className="ml-auto whitespace-nowrap font-mono text-2xs text-ink-faint">
                {run.envIds.length} envs · {formatStep(run.steps)} · {formatAgo(run.startedAt)}
              </span>
            </button>
          ))}
          {rows.length === 0 && (
            <div className="px-2 py-3 text-center text-2xs text-ink-faint">no runs match</div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
