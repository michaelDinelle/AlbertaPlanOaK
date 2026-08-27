import { SCENES, nameOf } from '../scenes/table';
import { Panel } from '../glass/Panel';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useStageMeta } from '../theater/controller';

export function envGradient(slug: string): string {
  return `linear-gradient(140deg, color-mix(in oklab, var(--color-env-${slug}) 45%, var(--color-poster-base)) 0%, var(--color-poster-base) 85%)`;
}

export function posterUrl(slug: string): string {
  return `/posters/${slug}.webp`;
}

export function EnvTileButton({ onOpen }: { onOpen: () => void }) {
  const sel = useSelection();
  const meta = useStageMeta();
  const info = sel.envId === null ? null : SCENES[sel.envId];
  return (
    <Panel>
      <button
        type="button"
        onClick={onOpen}
        title={STR.tile.open}
        className="flex items-center gap-2.5 rounded-lg p-2 pr-3.5 transition-colors hover:bg-white/10"
      >
        <span
          className="relative h-10 w-[71px] shrink-0 overflow-hidden rounded-md ring-1 ring-white/10"
          style={{ background: info !== null ? envGradient(info.slug) : undefined }}
        >
          {info !== null && (
            <img
              src={posterUrl(info.slug)}
              alt=""
              draggable={false}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
        </span>
        <span className="flex flex-col items-start">
          <span className="text-xs font-medium text-ink">{sel.envId === null ? STR.tile.none : nameOf(sel.envId)}</span>
          <span className="font-mono text-2xs text-ink-dim">
            {meta.seed !== null
              ? `${STR.tile.seed(meta.seed)} · ${meta.promoted ? STR.tile.yourPick : STR.tile.bestSeed}`
              : '—'}
          </span>
        </span>
      </button>
    </Panel>
  );
}
