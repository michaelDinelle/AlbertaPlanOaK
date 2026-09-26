import { useMemo, useState, type SyntheticEvent } from 'react';
import { configurationName, type EnvDescriptor, type Environment } from '../data/envs';
import type { Recording } from '../data/layout';
import { deliveredRate, envLanes } from '../derive';
import { posterOf } from '../scenes/table';
import { selectEnv, store } from '../store';

export const CATEGORIES = ['control', 'navigation', 'memory', 'decision', 'prediction', 'world', 'games', 'puzzles'] as const;
const SORTS = ['name', 'seeds', 'reward'] as const;
type Sort = (typeof SORTS)[number];
const gradient = (name: string): string => `linear-gradient(140deg, color-mix(in oklab, var(--color-env-${name}) 45%, var(--color-poster-base)) 0%, var(--color-poster-base) 85%)`;
const hideMissing = (event: SyntheticEvent<HTMLImageElement>): void => { event.currentTarget.hidden = true; };

function Poster({ name, className }: { name: string; className: string }) {
  return <span className={`${className} overflow-hidden`} style={{ background: gradient(name) }}><img src={posterOf(name)} alt="" draggable={false} onError={hideMissing} className="absolute inset-0 h-full w-full object-cover" /></span>;
}

interface Row { id: string; environment: string; label: string; detail: string; terms: string; seeds: number; selected: boolean; reward: number; pick: () => void }

function Tile({ row }: { row: Row }) {
  return <button type="button" onClick={row.pick} title={`${row.label} — ${row.seeds} seeds`} className={`group relative aspect-video overflow-hidden rounded-lg text-left transition-transform hover:scale-[1.02] ${row.selected ? 'ring-2 ring-accent' : ''}`}>
    <Poster name={row.environment} className="absolute inset-0 block" />
    <span className="absolute inset-x-0 bottom-0 flex items-baseline gap-2 bg-gradient-to-t from-black/75 to-transparent px-2.5 pb-1.5 pt-5">
      <span className="min-w-0 text-xs font-medium text-ink">{row.label}{row.detail && <span className="block text-2xs font-normal text-ink-dim">{row.detail}</span>}</span>
      <span className="ml-auto shrink-0 whitespace-nowrap font-mono text-2xs text-ink-dim">{row.seeds} seeds</span>
    </span>
  </button>;
}

function Wall({ environments, run, onClose }: { environments: Environment[]; run: Recording | null; onClose: () => void }) {
  const { envId } = store.use();
  const [query, setQuery] = useState(''), [sort, setSort] = useState<Sort>('name'), [opened, setOpened] = useState<string | null>(null);
  const environment = environments.find(entry => entry.name === opened);
  const seeds = (env: EnvDescriptor): number => run === null ? 0 : envLanes(run.meta, env.envId).seeds.length;
  const pick = (env: EnvDescriptor) => { selectEnv(env.envId); onClose(); };
  const sections = useMemo(() => {
    const q = query.trim().toLowerCase();
    const reward = (env: EnvDescriptor) => {
      if (run === null) return -Infinity;
      const lanes = envLanes(run.meta, env.envId).lanes;
      return lanes.reduce((sum, lane) => sum + (deliveredRate(run, lane) ?? NaN), 0) / lanes.length;
    };
    const order = (rows: Row[]) => rows.filter(row => row.terms.toLowerCase().includes(q)).sort((a, b) => sort === 'seeds' ? b.seeds - a.seeds : sort === 'reward' ? b.reward - a.reward : a.label.localeCompare(b.label, undefined, { numeric: true }));
    if (environment) {
      return [{ title: environment.name, rows: order(environment.configurations.map(env => ({ id: env.envId, environment: environment.name, label: configurationName(env), detail: '', terms: configurationName(env), seeds: seeds(env), selected: env.envId === envId, reward: reward(env), pick: () => pick(env) }))) }];
    }
    return CATEGORIES.map(category => ({ title: category, rows: order(environments.filter(group => group.category === category).map(group => ({
      id: group.name, environment: group.name, label: group.name, detail: group.configurations.length > 1 ? `${group.configurations.length} recorded configurations` : '',
      terms: [group.name, ...group.configurations.map(configurationName)].join(' '),
      seeds: group.configurations.reduce((sum, env) => sum + seeds(env), 0), selected: group.configurations.some(env => env.envId === envId), reward: 0,
      pick: () => { if (group.configurations.length === 1) pick(group.configurations[0]!); else { setOpened(group.name); setQuery(''); setSort('name'); } },
    }))) })).filter(section => section.rows.length > 0);
  }, [environments, environment, query, sort, run, envId]);
  const first = sections[0]?.rows[0];
  const back = () => { setOpened(null); setQuery(''); setSort('name'); };
  return <div className="fixed inset-0 z-40 bg-black/50" onPointerDown={onClose}>
    <div className="glass absolute inset-5 flex flex-col overflow-hidden rounded-xl" role="dialog" aria-label="Environment gallery" onPointerDown={event => event.stopPropagation()}>
      <div className="flex shrink-0 items-center gap-3 px-5 pb-3 pt-4">
        {environment && <button type="button" onClick={back} aria-label="Back to environments" className="rounded-md px-2 py-1 text-xs text-ink-dim hover:bg-white/10 hover:text-ink">←</button>}
        <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-dim">{environment?.name ?? 'Environments'}</h2>
        <input type="search" autoFocus value={query} placeholder={environment ? 'Search configurations' : 'Search environments'} aria-label={environment ? 'Search configurations' : 'Search environments'} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') environment ? back() : onClose(); else if (event.key === 'Enter') first?.pick(); }} className="w-64 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-ink outline-none transition-colors placeholder:text-ink-faint hover:border-white/25 focus:border-accent/50" />
        <button type="button" onClick={() => { const sorts = environment ? SORTS : SORTS.slice(0, 2); setSort(sorts[(sorts.indexOf(sort) + 1) % sorts.length]!); }} title="Change sort" className="flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-1 text-xs text-ink-dim hover:border-white/25 hover:text-ink">⇅ {sort}</button>
        <button type="button" onClick={onClose} title="Close" className="ml-auto flex size-7 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15">✕</button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
        {sections.length === 0 ? <div className="py-8 text-center text-xs text-ink-dim">No match for “{query.trim()}”</div> : sections.map(section => <section key={section.title} className="mb-5">
          {!environment && <h3 className="mb-2 flex items-baseline gap-2 text-2xs font-semibold uppercase tracking-wider text-ink-dim"><span className="h-px w-4 bg-emblem/70" />{section.title}<span className="font-mono font-normal normal-case tracking-normal text-ink-faint">{section.rows.length}</span></h3>}
          <div className="grid grid-cols-4 gap-3">{section.rows.map(row => <Tile key={row.id} row={row} />)}</div>
        </section>)}
      </div>
    </div>
  </div>;
}

export function EnvPicker({ environments, run, seed }: { environments: Environment[]; run: Recording | null; seed: number | null }) {
  const { envId } = store.use();
  const [open, setOpen] = useState(false);
  const environment = environments.find(entry => entry.configurations.some(env => env.envId === envId));
  const descriptor = environment?.configurations.find(env => env.envId === envId);
  const label = descriptor && environment && environment.configurations.length > 1 ? configurationName(descriptor) : '';
  return <>
    {open && <Wall environments={environments} run={run} onClose={() => setOpen(false)} />}
    <button type="button" onClick={() => setOpen(true)} title="Choose an environment" className="glass flex items-center gap-2.5 rounded-lg p-2 pr-3.5 hover:bg-white/10">
      {environment ? <Poster name={environment.name} className="relative block h-10 w-[71px] shrink-0 rounded-md ring-1 ring-white/10" /> : <span className="block h-10 w-[71px] shrink-0 rounded-md ring-1 ring-white/10" />}
      <span className="flex min-w-0 flex-col items-start"><span className="text-xs font-medium text-ink">{environment?.name ?? 'no environment yet'}</span><span className="max-w-64 truncate font-mono text-2xs text-ink-dim" title={label}>{label && `${label} · `}{seed === null ? '—' : `seed ${seed}`}</span></span>
    </button>
  </>;
}
