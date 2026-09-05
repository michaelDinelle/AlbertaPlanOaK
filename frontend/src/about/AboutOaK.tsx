import { useEffect, useState } from 'react';
import { clsx } from 'clsx';
import { ArrowRight, ArrowUpRight, Play, X } from 'lucide-react';
import { Panel } from '../glass/Panel';
import { Label, TXT } from '../panels/charts';
import { STR } from '../strings';
import { CycleDiagram } from './CycleDiagram';
import { AUTHOR_LINKS } from './links';
import {
  ALBERTA_PLAN,
  STAGE_ORDER,
  TALKS,
  type Stage,
  type StageKey,
  type Talk,
} from './content';

function Section({ children, hint }: { children: string; hint?: string }) {
  return (
    <div className="mb-2.5 flex items-center gap-2">
      <Label hint={hint}>{children}</Label>
      <span className="h-px flex-1 bg-white/12" />
    </div>
  );
}

function CodeChip({ children }: { children: string }) {
  return (
    <span className="rounded-sm border border-white/15 bg-white/8 px-1.5 py-0.5 font-mono text-2xs text-ink">
      {children}
    </span>
  );
}

function watchUrl(id: string): string {
  return `https://www.youtube.com/watch?v=${id}`;
}

function StageDetail({ stage, next }: { stage: Stage; next: Stage }) {
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
        <span className="font-mono text-base text-accent">{stage.badge}</span>
        <h3 className="text-base text-ink">{stage.name.join(' ')}</h3>
        <span
          title={STR.about.handoff(next.name.join(' '))}
          className="ml-auto flex cursor-help items-center gap-1.5 text-2xs text-ink-dim"
        >
          {stage.handoff}
          <ArrowRight className="size-3" />
          <span className="font-mono text-accent">{next.badge}</span>
        </span>
      </div>
      <p className="text-sm leading-relaxed text-ink">{stage.plan}</p>
      <div className="mt-4">
        <div className="flex flex-wrap gap-1.5">
          {stage.symbols.map((symbol) => (
            <CodeChip key={symbol}>{symbol}</CodeChip>
          ))}
        </div>
      </div>
    </div>
  );
}

function TalkCard({ talk, onPlay }: { talk: Talk; onPlay: () => void }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <button
        type="button"
        onClick={onPlay}
        title={STR.about.play(talk.title)}
        className="group relative block aspect-video overflow-hidden rounded-md border border-white/15 bg-black/50"
      >
        <img
          src={`https://i.ytimg.com/vi/${talk.id}/hqdefault.jpg`}
          alt=""
          draggable={false}
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover opacity-85 transition-opacity group-hover:opacity-100"
        />
        <span className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent" />
        <span className="absolute inset-0 flex items-center justify-center">
          <span className="flex size-9 items-center justify-center rounded-full border border-white/30 bg-black/50 transition-transform group-hover:scale-110">
            <Play className="ml-0.5 size-4 fill-ink text-ink" />
          </span>
        </span>
      </button>
      <a
        href={watchUrl(talk.id)}
        target="_blank"
        rel="noreferrer"
        title={STR.about.watch(talk.title)}
        className="flex items-start gap-1 text-xs leading-snug text-ink transition-colors hover:text-accent"
      >
        <span className="min-w-0">{talk.title}</span>
        <ArrowUpRight className="mt-0.5 size-3 shrink-0 text-ink-dim" />
      </a>
      <span className={TXT.dataDim}>{talk.venue}</span>
    </div>
  );
}

function VideoLightbox({ talk, onClose }: { talk: Talk; onClose: () => void }) {
  return (
    <div
      className="absolute inset-0 z-10 flex items-center justify-center bg-black/85 p-4 sm:p-8"
      onPointerDown={(event) => {
        event.stopPropagation();
        onClose();
      }}
    >
      <div className="w-[min(1100px,100%)]" onPointerDown={(event) => event.stopPropagation()}>
        <div className="relative aspect-video overflow-hidden rounded-lg border border-white/15 bg-black">
          <iframe
            src={`https://www.youtube-nocookie.com/embed/${talk.id}?autoplay=1&rel=0`}
            title={talk.title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; picture-in-picture"
            allowFullScreen
            className="absolute inset-0 h-full w-full border-0"
          />
        </div>
        <div className="mt-3 flex items-start gap-4">
          <div className="min-w-0">
            <div className="text-sm text-ink">{talk.title}</div>
            <div className={clsx('mt-0.5', TXT.dataDim)}>{talk.venue}</div>
          </div>
          <a
            href={watchUrl(talk.id)}
            target="_blank"
            rel="noreferrer"
            className="ml-auto flex shrink-0 items-center gap-1 text-xs text-ink-dim transition-colors hover:text-accent"
          >
            {STR.about.youtube}
            <ArrowUpRight className="size-3" />
          </a>
          <button
            type="button"
            onClick={onClose}
            title={STR.about.closeVideo}
            className="flex size-8 shrink-0 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
          >
            <X className="size-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

export function AboutOaK({ onClose }: { onClose: () => void }) {
  const [selected, setSelected] = useState<StageKey>('fc');
  const [playing, setPlaying] = useState<Talk | null>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return;
      if (playing !== null) setPlaying(null);
      else onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [playing, onClose]);

  return (
    <div className="absolute inset-0 z-50" onPointerDown={onClose}>
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-(--gutter)">
        <Panel className="flex max-h-full w-[min(1040px,100%)] flex-col overflow-hidden">
          <div className="flex min-h-0 flex-1 flex-col" onPointerDown={(event) => event.stopPropagation()}>
            <header className="flex items-start gap-6 px-6 pb-4 pt-5">
              <div className="min-w-0">
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-xl text-accent">{STR.app}</span>
                  <h2 className="text-base text-ink">{STR.about.title}</h2>
                </div>
                <p className="mt-1 text-xs text-ink-dim">{STR.about.subtitle}</p>
              </div>
              <div className="ml-auto flex shrink-0 items-center gap-2">
                <span className="hidden text-xs text-ink sm:block">{STR.about.by}</span>
                {AUTHOR_LINKS.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    target="_blank"
                    rel="noreferrer"
                    title={STR.about.follow(link.label)}
                    className="flex size-9 items-center justify-center rounded-md border border-white/15 bg-white/8 text-ink transition-colors hover:border-accent/50 hover:bg-accent/15 hover:text-accent"
                  >
                    <link.Mark className="size-4" />
                  </a>
                ))}
                <span className="mx-1 h-7 w-px bg-white/15" />
                <button
                  type="button"
                  onClick={onClose}
                  title={STR.about.close}
                  className="flex size-9 items-center justify-center rounded-md text-ink transition-colors hover:bg-white/10 active:bg-white/15"
                >
                  <X className="size-4" />
                </button>
              </div>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
              <a
                href={ALBERTA_PLAN.href}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-3 rounded-md border border-accent/30 bg-accent/12 px-3.5 py-2.5 transition-colors hover:border-accent/60 hover:bg-accent/20"
              >
                <span className="text-sm text-ink">{ALBERTA_PLAN.title}</span>
                <span className="font-mono text-2xs text-ink-dim">{ALBERTA_PLAN.cite}</span>
                <ArrowUpRight className="ml-auto size-4 shrink-0 text-accent" />
              </a>

              <div className="mt-5 grid gap-7 lg:grid-cols-[minmax(0,400px)_minmax(0,1fr)]">
                <div className="min-w-0">
                  <Section hint={STR.about.cycleHint}>{STR.about.cycle}</Section>
                  <CycleDiagram selected={selected} onSelect={setSelected} />
                </div>

                <div className="flex min-w-0 flex-col gap-6">
                  <div className="grid">
                    {STAGE_ORDER.map((each, index) => (
                      <div
                        key={each.key}
                        aria-hidden={each.key !== selected}
                        className={clsx(
                          'col-start-1 row-start-1 transition-opacity duration-200 motion-reduce:transition-none',
                          each.key === selected ? 'opacity-100' : 'pointer-events-none opacity-0',
                        )}
                      >
                        <StageDetail
                          stage={each}
                          next={STAGE_ORDER[(index + 1) % STAGE_ORDER.length] ?? each}
                        />
                      </div>
                    ))}
                  </div>

                  <div>
                    <Section>{STR.about.talks}</Section>
                    <div className="grid gap-4 sm:grid-cols-3">
                      {TALKS.map((talk) => (
                        <TalkCard key={talk.id} talk={talk} onPlay={() => setPlaying(talk)} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              <p className="mt-6 text-2xs text-ink-dim">
                Rockies scenery adapted from Canadian elevation data via <a href="https://registry.opendata.aws/terrain-tiles/" target="_blank" rel="noreferrer" className="underline">Mapzen</a>. Contains information licensed under the <a href="https://open.canada.ca/en/open-government-licence-canada" target="_blank" rel="noreferrer" className="underline">Open Government Licence – Canada</a>.
              </p>
            </div>
          </div>
        </Panel>
      </div>
      {playing !== null && <VideoLightbox talk={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}
