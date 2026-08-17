import type { RunDetail } from '@shared/types';
import { clsx } from 'clsx';

export function Scoreboard({ detail }: { detail: RunDetail | null }) {
  const rows: { name: string; token: string; present: boolean }[] = [
    { name: 'OaK', token: 'var(--color-oak)', present: detail !== null },
    { name: 'PPO', token: 'var(--color-ppo)', present: detail?.baselines.ppo !== undefined },
    { name: 'random', token: 'var(--color-random)', present: detail?.baselines.random !== undefined },
  ];
  return (
    <div className="veil rounded-md px-3 py-2.5">
      <div className="mb-1.5 text-2xs uppercase tracking-[0.16em] text-ink-faint">Scoreboard</div>
      <div className="flex flex-col gap-1">
        {rows.map((row) => (
          <div
            key={row.name}
            className={clsx('flex items-center gap-2 font-mono text-xs', !row.present && 'opacity-40')}
          >
            <span className="size-2 rounded-full" style={{ background: row.token }} />
            <span>{row.name}</span>
            <span className="ml-auto tabular-nums text-ink-faint">—</span>
          </div>
        ))}
      </div>
      <div className="mt-2 text-2xs text-ink-faint">statistics land with summary v2</div>
    </div>
  );
}
