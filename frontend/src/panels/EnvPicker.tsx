import { useMemo, useState } from 'react';
import type { EnvId } from '../data/envs';
import type { RunDetail } from '../data/oak';
import { deliveredRate, envLanes } from '../derive';
import { glass } from '../glass/glass';
import { SCENES, nameOf } from '../scenes/table';
import { selectEnv, store } from '../store';

const SORTS = ['name', 'seeds', 'reward'] as const;
type Sort = (typeof SORTS)[number];

const gradient = (slug: string): string => `linear-gradient(140deg, color-mix(in oklab, var(--color-env-${slug}) 45%, var(--color-poster-base)) 0%, var(--color-poster-base) 85%)`;

function Tile({ env, seeds, current, onPick }: { env: EnvId; seeds: number; current: boolean; onPick: (env: EnvId) => void }) {
  const slug = SCENES[env].slug;
  return (
    <button
      type="button"
      onClick={() => onPick(env)}
      title={`Watch ${nameOf(env)} — ${seeds > 0 ? `${seeds} seeds` : 'no seeds'}`}
      className={`group relative aspect-video overflow-hidden rounded-lg text-left transition-transform hover:scale-[1.02] ${current ? 'ring-2 ring-accent' : ''}`}
      style={{ background: gradient(slug) }}
    >
      <img src={`/posters/${slug}.webp`} alt="" draggable={false} className={`absolute inset-0 h-full w-full object-cover ${seeds === 0 ? 'opacity-45' : ''}`} />
      <span className="absolute inset-x-0 bottom-0 flex items-baseline gap-2 bg-gradient-to-t from-black/75 to-transparent px-2.5 pb-1.5 pt-5">
        <span className="text-xs font-medium text-ink">{nameOf(env)}</span>
        <span className="ml-auto font-mono text-2xs text-ink-dim">{seeds} seeds</span>
      </span>
    </button>
  );
}

function Wall({ envs, detail, onClose }: { envs: EnvId[]; detail: RunDetail | null; onClose: () => void }) {
  const { envId, tapes } = store.use();
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<Sort>('name');
  const tape = detail === null ? null : tapes[detail.id] ?? null;
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const seeds = (env: EnvId): number => (detail === null ? 0 : envLanes(detail.meta, env).seeds.length);
    const reward = (env: EnvId): number => {
      if (detail === null || tape === null) return -Infinity;
      const lanes = envLanes(detail.meta, env).lanes;
      return lanes.reduce((sum, lane) => sum + (deliveredRate(tape, lane) ?? NaN), 0) / lanes.length;
    };
    return envs
      .filter((env) => q === '' || nameOf(env).toLowerCase().includes(q))
      .sort((a, b) => (sort === 'name' ? nameOf(a).localeCompare(nameOf(b)) : sort === 'seeds' ? seeds(b) - seeds(a) : reward(b) - reward(a)))
      .map((env) => ({ env, seeds: seeds(env) }));
  }, [envs, query, sort, detail, tape]);
  const pick = (env: EnvId): void => { selectEnv(env); onClose(); };
  return (
    <div className="fixed inset-0 z-40 bg-black/50" onPointerDown={onClose}>
      <div ref={glass} className="glass absolute inset-5 flex flex-col overflow-hidden rounded-xl" onPointerDown={(event) => event.stopPropagation()}>
        <div className="flex items-center gap-3 px-5 pb-3 pt-4">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-dim">Environments</h2>
          <input
            type="search"
            autoFocus
            value={query}
            placeholder="Search environments"
            aria-label="Search environments"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') onClose();
              else if (event.key === 'Enter' && rows[0] !== undefined) pick(rows[0].env);
            }}
            className="w-64 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-ink outline-none transition-colors placeholder:text-ink-faint hover:border-white/25 focus:border-accent/50"
          />
          <button
            type="button"
            onClick={() => setSort(SORTS[(SORTS.indexOf(sort) + 1) % SORTS.length]!)}
            title="Change sort"
            className="flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-1 text-xs text-ink-dim hover:border-white/25 hover:text-ink"
          >
            ⇅ {sort}
          </button>
          <button type="button" onClick={onClose} title="Close" className="ml-auto flex size-7 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15">
            ✕
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
          {rows.length === 0 ? (
            <div className="py-8 text-center text-xs text-ink-dim">No environment matches “{query.trim()}”</div>
          ) : (
            <div className="grid grid-cols-4 gap-3">
              {rows.map(({ env, seeds }) => <Tile key={env} env={env} seeds={seeds} current={env === envId} onPick={pick} />)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function EnvPicker({ envs, detail, seed }: { envs: EnvId[]; detail: RunDetail | null; seed: number | null }) {
  const { envId, seed: picked } = store.use();
  const [open, setOpen] = useState(false);
  const current = envId === null ? null : SCENES[envId];
  return (
    <>
      {open && <Wall envs={envs} detail={detail} onClose={() => setOpen(false)} />}
      <button ref={glass} type="button" onClick={() => setOpen(true)} title="Choose an environment" className="glass flex items-center gap-2.5 rounded-lg p-2 pr-3.5 hover:bg-white/10">
        <span className="relative h-10 w-[71px] shrink-0 overflow-hidden rounded-md ring-1 ring-white/10" style={{ background: current === null ? undefined : gradient(current.slug) }}>
          {current !== null && <img src={`/posters/${current.slug}.webp`} alt="" draggable={false} className="absolute inset-0 h-full w-full object-cover" />}
        </span>
        <span className="flex flex-col items-start">
          <span className="text-xs font-medium text-ink">{envId === null ? 'no environment yet' : nameOf(envId)}</span>
          <span className="font-mono text-2xs text-ink-dim">{seed === null ? '—' : `seed ${seed} · ${picked === null ? 'top reward' : 'your pick'}`}</span>
        </span>
      </button>
    </>
  );
}
