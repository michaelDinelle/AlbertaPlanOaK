import { useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { ArrowUpDown, X } from 'lucide-react';
import { VList } from 'virtua';
import type { EnvId } from '@shared/envs';
import { SCENES, nameOf } from '../scenes/table';
import type { RunDetail } from '@shared/types';
import { Frost } from '../glass/Panel';
import { Label, TXT } from '../panels/charts';
import { selectEnv, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { seedsFor } from '../theater/loader';
import { envGradient, posterUrl } from './EnvTileButton';

const COLUMNS = 4;

const WALL_SORTS = ['name', 'seeds', 'reward'] as const;

type WallSort = (typeof WALL_SORTS)[number];

function Tile({
  envId,
  seedCount,
  current,
  onPick,
}: {
  envId: EnvId;
  seedCount: number;
  current: boolean;
  onPick: (envId: EnvId) => void;
}) {
  const info = SCENES[envId];
  return (
    <button
      type="button"
      onClick={() => onPick(envId)}
      title={`${STR.wall.watchEnv(nameOf(envId))} — ${seedCount > 0 ? STR.wall.seeds(seedCount) : STR.wall.none}`}
      className={clsx(
        'group relative aspect-video overflow-hidden rounded-lg text-left transition-transform hover:scale-[1.02]',
        current && 'ring-2 ring-accent',
      )}
      style={{ background: envGradient(info.slug) }}
    >
      <img
        src={posterUrl(info.slug)}
        alt=""
        draggable={false}
        className={clsx('absolute inset-0 h-full w-full object-cover', seedCount === 0 && 'opacity-45')}
      />
      <span className="absolute inset-x-0 bottom-0 flex items-baseline gap-2 bg-gradient-to-t from-black/75 to-transparent px-2.5 pb-1.5 pt-5">
        <span className="text-xs font-medium text-ink">{nameOf(envId)}</span>
        <span className={clsx('ml-auto', TXT.dataDim)}>{STR.wall.seeds(seedCount)}</span>
      </span>
    </button>
  );
}

export function EnvWall({
  detail,
  envList,
  onClose,
}: {
  detail: RunDetail | null;
  envList: readonly EnvId[];
  onClose: () => void;
}) {
  const sel = useSelection();
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<WallSort>('name');
  const recordings = detail?.recordings ?? [];

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const shown = envList.filter((id) => q === '' || nameOf(id).toLowerCase().includes(q));
    const seedCount = (id: EnvId): number => seedsFor(recordings, id).length;
    const reward = (id: EnvId): number => detail?.summary?.envs[id]?.finalRate.iqmZ ?? -Infinity;
    const sorted = [...shown].sort((a, b) =>
      sort === 'name'
        ? nameOf(a).localeCompare(nameOf(b))
        : sort === 'seeds'
          ? seedCount(b) - seedCount(a)
          : reward(b) - reward(a),
    );
    const out: EnvId[][] = [];
    for (let i = 0; i < sorted.length; i += COLUMNS) out.push(sorted.slice(i, i + COLUMNS));
    return out;
  }, [envList, query, sort, detail, recordings]);

  const pick = (envId: EnvId): void => {
    selectEnv(envId);
    onClose();
  };

  return (
    <div className="absolute inset-0 z-40 bg-black/50" onPointerDown={onClose}>
      <Frost className="absolute inset-5 flex flex-col overflow-hidden">
        <div
          className="flex items-center gap-3 px-5 pb-3 pt-4"
          onPointerDown={(event) => event.stopPropagation()}
        >
          <h2><Label>{STR.wall.title}</Label></h2>
          <input
            type="search"
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={STR.wall.search}
            className="w-64 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-ink outline-none transition-colors placeholder:text-ink-faint hover:border-white/25 focus:border-accent/50"
          />
          <button
            type="button"
            onClick={() => setSort(WALL_SORTS[(WALL_SORTS.indexOf(sort) + 1) % WALL_SORTS.length] ?? 'name')}
            title={STR.wall.sort}
            className={clsx('flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-1 hover:border-white/25 hover:text-ink', TXT.bodyDim)}
          >
            <ArrowUpDown className="size-2.5" />
            {STR.wall.sortBy[sort]}
          </button>
          <button
            type="button"
            onClick={onClose}
            title={STR.wall.close}
            className="ml-auto flex size-7 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 px-5 pb-5" onPointerDown={(event) => event.stopPropagation()}>
          {rows.length === 0 ? (
            <div className={clsx('py-8 text-center', TXT.bodyDim)}>{STR.wall.noMatch(query.trim())}</div>
          ) : (
            <VList style={{ height: '100%' }}>
              {rows.map((row) => (
                <div key={row[0] ?? 'row'} className="grid grid-cols-4 gap-3 pb-3">
                  {row.map((envId) => (
                    <Tile
                      key={envId}
                      envId={envId}
                      seedCount={seedsFor(recordings, envId).length}
                      current={envId === sel.envId}
                      onPick={pick}
                    />
                  ))}
                </div>
              ))}
            </VList>
          )}
        </div>
      </Frost>
    </div>
  );
}
