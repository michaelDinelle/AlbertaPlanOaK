import { useMemo, useState } from 'react';
import { configurationLabel, type EnvDescriptor, type EnvFamily } from '../data/envs';
import type { Recording } from '../data/layout';
import { deliveredRate, envLanes } from '../derive';
import { sceneOf } from '../scenes/table';
import { selectEnv, store } from '../store';

const SORTS = ['name', 'seeds', 'reward'] as const;
type Sort = (typeof SORTS)[number];
const gradient = (slug: string): string => `linear-gradient(140deg, color-mix(in oklab, var(--color-env-${slug}) 45%, var(--color-poster-base)) 0%, var(--color-poster-base) 85%)`;

function Tile({ descriptor, label, detail, seeds, current, onPick }: { descriptor: EnvDescriptor; label: string; detail: string; seeds: number; current: boolean; onPick: () => void }) {
  const scene = sceneOf(descriptor);
  return <button type="button" onClick={onPick} title={`${label} — ${seeds} seeds`} className={`group relative aspect-video overflow-hidden rounded-lg text-left transition-transform hover:scale-[1.02] ${current ? 'ring-2 ring-accent' : ''}`} style={{ background: scene === null ? 'var(--color-poster-base)' : gradient(scene.slug) }}>
    {scene !== null && <img src={scene.poster} alt="" draggable={false} className="absolute inset-0 h-full w-full object-cover" />}
    <span className="absolute inset-x-0 bottom-0 flex items-baseline gap-2 bg-gradient-to-t from-black/75 to-transparent px-2.5 pb-1.5 pt-5">
      <span className="min-w-0 text-xs font-medium text-ink">{label}{detail && <span className="block text-2xs font-normal text-ink-dim">{detail}</span>}{scene === null && <span className="block text-2xs font-normal text-ink-dim">Visualization unavailable</span>}</span>
      <span className="ml-auto shrink-0 whitespace-nowrap font-mono text-2xs text-ink-dim">{seeds} seeds</span>
    </span>
  </button>;
}

function Wall({ families, run, onClose }: { families: EnvFamily[]; run: Recording | null; onClose: () => void }) {
  const { envId } = store.use();
  const [query, setQuery] = useState(''), [sort, setSort] = useState<Sort>('name'), [familyId, setFamilyId] = useState<string | null>(null);
  const family = families.find(entry => entry.id === familyId);
  const seeds = (env: EnvDescriptor): number => run === null ? 0 : envLanes(run.meta, env.envId).seeds.length;
  const pick = (env: EnvDescriptor) => { selectEnv(env.envId); onClose(); };
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const reward = (env: EnvDescriptor) => {
      if (run === null) return -Infinity;
      const lanes = envLanes(run.meta, env.envId).lanes;
      return lanes.reduce((sum, lane) => sum + (deliveredRate(run, lane) ?? NaN), 0) / lanes.length;
    };
    const options = family ? family.configurations.map(env => ({ id: env.envId, descriptor: env, label: configurationLabel(env, family.configurations) || family.name, detail: '', seeds: seeds(env), selected: env.envId === envId, reward: reward(env), pick: () => pick(env) })) : families.map(group => ({ id: group.id, descriptor: group.configurations[0]!, label: group.name, detail: group.configurations.length > 1 ? `${group.configurations.length} recorded configurations` : '', seeds: group.configurations.reduce((sum, env) => sum + seeds(env), 0), selected: group.configurations.some(env => env.envId === envId), reward: 0, pick: () => { if (group.configurations.length === 1) pick(group.configurations[0]!); else { setFamilyId(group.id); setQuery(''); setSort('name'); } } }));
    return options.filter(row => `${row.label} ${row.detail}`.toLowerCase().includes(q)).sort((a, b) => sort === 'seeds' ? b.seeds - a.seeds : sort === 'reward' ? b.reward - a.reward : a.label.localeCompare(b.label, undefined, { numeric: true }));
  }, [families, family, query, sort, run, envId]);
  const back = () => { setFamilyId(null); setQuery(''); setSort('name'); };
  return <div className="fixed inset-0 z-40 bg-black/50" onPointerDown={onClose}>
    <div className="glass absolute inset-5 flex flex-col overflow-hidden rounded-xl" role="dialog" aria-label="Environment gallery" onPointerDown={event => event.stopPropagation()}>
      <div className="flex shrink-0 items-center gap-3 px-5 pb-3 pt-4">
        {family && <button type="button" onClick={back} aria-label="Back to environments" className="rounded-md px-2 py-1 text-xs text-ink-dim hover:bg-white/10 hover:text-ink">←</button>}
        <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-dim">{family?.name ?? 'Environments'}</h2>
        <input type="search" autoFocus value={query} placeholder={family ? 'Search configurations' : 'Search environments'} aria-label={family ? 'Search configurations' : 'Search environments'} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') family ? back() : onClose(); else if (event.key === 'Enter') rows[0]?.pick(); }} className="w-64 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-ink outline-none transition-colors placeholder:text-ink-faint hover:border-white/25 focus:border-accent/50" />
        <button type="button" onClick={() => { const sorts = family ? SORTS : SORTS.slice(0, 2); setSort(sorts[(sorts.indexOf(sort) + 1) % sorts.length]!); }} title="Change sort" className="flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-1 text-xs text-ink-dim hover:border-white/25 hover:text-ink">⇅ {sort}</button>
        <button type="button" onClick={onClose} title="Close" className="ml-auto flex size-7 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15">✕</button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
        {rows.length === 0 ? <div className="py-8 text-center text-xs text-ink-dim">No match for “{query.trim()}”</div> : <div className="grid grid-cols-4 gap-3">{rows.map(row => <Tile key={row.id} descriptor={row.descriptor} label={row.label} detail={row.detail} seeds={row.seeds} current={row.selected} onPick={row.pick} />)}</div>}
      </div>
    </div>
  </div>;
}

export function EnvPicker({ families, run, seed }: { families: EnvFamily[]; run: Recording | null; seed: number | null }) {
  const { envId } = store.use();
  const [open, setOpen] = useState(false);
  const family = families.find(entry => entry.configurations.some(env => env.envId === envId));
  const descriptor = family?.configurations.find(env => env.envId === envId);
  const scene = descriptor === undefined ? null : sceneOf(descriptor);
  const label = descriptor && family ? configurationLabel(descriptor, family.configurations) : '';
  return <>
    {open && <Wall families={families} run={run} onClose={() => setOpen(false)} />}
    <button type="button" onClick={() => setOpen(true)} title="Choose an environment" className="glass flex items-center gap-2.5 rounded-lg p-2 pr-3.5 hover:bg-white/10">
      <span className="relative h-10 w-[71px] shrink-0 overflow-hidden rounded-md ring-1 ring-white/10" style={{ background: scene === null ? undefined : gradient(scene.slug) }}>{scene !== null && <img src={scene.poster} alt="" draggable={false} className="absolute inset-0 h-full w-full object-cover" />}</span>
      <span className="flex min-w-0 flex-col items-start"><span className="text-xs font-medium text-ink">{family?.name ?? 'no environment yet'}</span><span className="max-w-64 truncate font-mono text-2xs text-ink-dim" title={label}>{label && `${label} · `}{seed === null ? '—' : `seed ${seed}`}</span></span>
    </button>
  </>;
}
