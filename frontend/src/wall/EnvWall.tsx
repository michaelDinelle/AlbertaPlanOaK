import { useMemo } from 'react';
import { clsx } from 'clsx';
import { VList } from 'virtua';
import { ENVS, type EnvId } from '@shared/envs';
import type { RunDetail, RunIndexEntry } from '@shared/types';
import { Frost } from '../glass/Panel';
import { Label, TXT } from '../panels/charts';
import { runLabel, sortRuns } from '../lib/run';
import { selectEnv, selectRun, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { seedsFor } from '../theater/loader';
import { envGradient, posterUrl } from './EnvTileButton';

const COLUMNS = 4;

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
  const info = ENVS[envId];
  return (
    <button
      type="button"
      onClick={() => onPick(envId)}
      title={`${STR.wall.watchEnv(info.name)} — ${seedCount > 0 ? STR.wall.seeds(seedCount) : STR.wall.none}`}
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
        <span className="text-xs font-medium text-ink">{info.name}</span>
        <span className={clsx('ml-auto', TXT.dataDim)}>{STR.wall.seeds(seedCount)}</span>
      </span>
    </button>
  );
}

export function EnvWall({
  entries,
  entry,
  detail,
  envList,
  onClose,
}: {
  entries: readonly RunIndexEntry[] | undefined;
  entry: RunIndexEntry | null;
  detail: RunDetail | null;
  envList: readonly EnvId[];
  onClose: () => void;
}) {
  const sel = useSelection();
  const runs = useMemo(() => sortRuns(entries ?? []).slice(0, 16), [entries]);
  const recordings = detail?.recordings ?? [];
  const rows = useMemo(() => {
    const out: EnvId[][] = [];
    for (let i = 0; i < envList.length; i += COLUMNS) out.push(envList.slice(i, i + COLUMNS));
    return out;
  }, [envList]);

  const pick = (envId: EnvId): void => {
    selectEnv(envId);
    onClose();
  };

  return (
    <div className="absolute inset-0 z-40 bg-black/50" onPointerDown={onClose}>
      <Frost className="absolute inset-5 flex flex-col overflow-hidden">
        <div className="flex flex-col gap-2 px-5 pb-3 pt-4" onPointerDown={(event) => event.stopPropagation()}>
          <div className="flex items-baseline gap-3">
            <h2><Label>{STR.wall.title}</Label></h2>
            <span className={TXT.bodyDim}>{STR.wall.hint}</span>
          </div>
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
            <span className="mr-1"><Label>{STR.wall.runs}</Label></span>
            {runs.map((run) => (
              <button
                key={run.id}
                type="button"
                onClick={() => selectRun(run.id)}
                title={STR.wall.switchRun(runLabel(run))}
                className={clsx(
                  'flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-2xs transition-colors',
                  run.id === entry?.id
                    ? 'border-accent/50 bg-accent/15 text-accent'
                    : 'border-white/10 text-ink-dim hover:border-white/25 hover:text-ink',
                )}
              >
                {run.inFlight && <span className="size-1.5 motion-safe:animate-pulse rounded-full bg-good" title={STR.wall.live} />}
                {runLabel(run)}
              </button>
            ))}
          </div>
        </div>
        <div className="min-h-0 flex-1 px-5 pb-5" onPointerDown={(event) => event.stopPropagation()}>
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
        </div>
      </Frost>
    </div>
  );
}
