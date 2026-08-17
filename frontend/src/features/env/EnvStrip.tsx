import { useEffect, useState, type CSSProperties } from 'react';
import { clsx } from 'clsx';
import { ENVS, isEnvId, type EnvId } from '@shared/envs';
import type { RunDetail } from '@shared/types';
import { formatCount } from '../../lib/format';
import { seedsFor } from '../../theater/loader';
import { selectEnv, useSelection } from '../../state/selectionStore';

export function EnvStrip({ detail, envList }: { detail: RunDetail | null; envList: readonly EnvId[] }) {
  const sel = useSelection();
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    const measure = (): void => {
      setStuck(window.scrollY > 1);
    };
    window.addEventListener('scroll', measure, { passive: true });
    measure();
    return () => {
      window.removeEventListener('scroll', measure);
    };
  }, []);

  const recordings = detail?.recordings ?? [];

  return (
    <nav
      className={clsx(
        'sticky top-12 z-30 px-4 py-2 transition-colors duration-150',
        stuck && 'glass rounded-none border-x-0 border-t-0',
      )}
      aria-label="Environment"
    >
      <div className="hidden flex-wrap gap-1.5 sm:flex" role="tablist">
        {envList.map((id) => {
          const info = ENVS[id];
          const count = seedsFor(recordings, id).length;
          const on = id === sel.envId;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-current={on ? 'true' : undefined}
              onClick={() => selectEnv(id)}
              style={{ '--env-accent': `var(--color-env-${info.slug})` } as CSSProperties}
              title={
                count > 0
                  ? `${info.name} — ${formatCount(count)} recorded seed${count === 1 ? '' : 's'}`
                  : `${info.name} — no recordings in this run`
              }
              className={clsx(
                'rounded-sm border px-2 py-0.5 font-mono text-2xs leading-4 transition-colors',
                on
                  ? 'border-(--env-accent) bg-(--env-accent)/12 text-(--env-accent)'
                  : 'border-line text-ink-dim hover:border-line-strong hover:text-ink',
                count === 0 && 'opacity-45',
              )}
            >
              {info.name}
            </button>
          );
        })}
      </div>
      <select
        aria-label="Environment"
        value={sel.envId ?? ''}
        onChange={(event) => {
          if (isEnvId(event.target.value)) selectEnv(event.target.value);
        }}
        className="w-full rounded-sm border border-line bg-sunken px-2 py-1.5 font-mono text-xs sm:hidden"
      >
        {envList.map((id) => (
          <option key={id} value={id}>
            {ENVS[id].name}
          </option>
        ))}
      </select>
    </nav>
  );
}
