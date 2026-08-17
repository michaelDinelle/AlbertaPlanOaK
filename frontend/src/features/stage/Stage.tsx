import { useStageMeta } from '../../theater/controller';
import { TheaterStage } from '../../theater/TheaterStage';

export function Stage() {
  const meta = useStageMeta();
  return (
    <div className="relative min-h-0 flex-1 overflow-hidden rounded-md bg-[#05070c]">
      <TheaterStage />
      {meta.status === 'ready' && meta.envName !== null && (
        <div className="veil pointer-events-none absolute left-3 top-3 flex items-baseline gap-2 rounded-sm px-2.5 py-1.5">
          <b className="font-mono text-sm">{meta.envName}</b>
          <span className="font-mono text-2xs text-ink-dim">
            seed {meta.seed}
            {meta.promoted ? ' · promoted' : ' · leader'}
            {meta.policy === 'oak' ? '' : ` · ${meta.policy}`}
          </span>
        </div>
      )}
      {meta.note !== null && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div className="veil flex max-w-md flex-col gap-1.5 rounded-md px-5 py-4">
            <div className="font-mono text-sm">{meta.note}</div>
            {meta.noteDetail !== null && <div className="text-xs text-ink-dim">{meta.noteDetail}</div>}
            {meta.status === 'loading' && (
              <span className="mt-1 size-1.5 animate-pulse rounded-full bg-accent" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
