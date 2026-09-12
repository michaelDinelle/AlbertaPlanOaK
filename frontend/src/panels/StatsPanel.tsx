import { clsx } from 'clsx';
import { PanelRightClose } from 'lucide-react';
import type { Policy, RunDetail } from '../data/types';
import { Panel } from '../glass/Panel';
import { Hotkey } from './charts';
import { STR } from '../strings';
import { ComputeView } from './ComputeView';
import { HealthView } from './HealthView';
import { RewardView } from './RewardView';
import { StandingsView } from './StandingsView';
import { AUTHOR_LINKS } from './links';

export type PanelView = keyof typeof STR.panels;

export const PANEL_VIEWS = Object.keys(STR.panels) as PanelView[];

export function StatsPanel({
  view,
  setView,
  runs,
  best,
  active,
  onHide,
}: {
  view: PanelView;
  setView: (view: PanelView) => void;
  runs: Record<Policy, RunDetail | null>;
  best: RunDetail | null;
  active: RunDetail | null;
  onHide: () => void;
}) {
  return (
    <Panel className="flex max-h-full w-full flex-col overflow-hidden py-2.5">
      <header className="mx-3 mb-2 flex shrink-0 items-center gap-3 border-b border-white/10 pb-2">
        <span className="min-w-0 text-xs text-ink-dim">{STR.author.by}</span>
        <nav aria-label={STR.author.links} className="ml-auto flex items-center gap-1">
          {AUTHOR_LINKS.map(({ href, label, Mark }) => (
            <a
              key={href}
              href={href}
              target="_blank"
              rel="noreferrer"
              aria-label={STR.author.follow(label)}
              title={STR.author.follow(label)}
              className="flex size-8 items-center justify-center rounded-md text-ink-dim transition-colors hover:bg-white/10 hover:text-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <Mark className="size-3.5" />
            </a>
          ))}
        </nav>
      </header>
      <div className="flex items-center gap-1 px-2 pb-2">
        {PANEL_VIEWS.map((id, index) => (
          <Hotkey key={id} label={STR.palette.show(STR.panels[id])} keyName={String(index + 1)}>
            <button
              type="button"
              onClick={() => setView(id)}
              className={clsx(
                'rounded-md px-2.5 py-1 text-xs transition-colors active:bg-white/15',
                view === id ? 'bg-accent/20 text-accent' : 'text-ink-dim hover:bg-white/10 hover:text-ink',
              )}
            >
              {STR.panels[id]}
            </button>
          </Hotkey>
        ))}
        <span className="ml-auto flex items-center gap-1">
          <Hotkey label={STR.bar.hidePanel} keyName="c" align="end">
            <button
              type="button"
              onClick={onHide}
              aria-label={STR.bar.hidePanel}
              className="flex size-6 shrink-0 items-center justify-center rounded-md text-ink-faint transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
            >
              <PanelRightClose className="size-3.5" />
            </button>
          </Hotkey>
        </span>
      </div>
      {view === 'reward' && <RewardView runs={runs} best={best} active={active} />}
      {view === 'standings' && <StandingsView runs={runs} />}
      {view === 'compute' && <ComputeView runs={runs} />}
      {view === 'health' && <HealthView runs={runs} active={active} />}
    </Panel>
  );
}
