import { useMemo, type ReactNode } from 'react';
import { Command } from 'cmdk';
import type { EnvId } from '../data/envs';
import { nameOf, slugOf } from '../scenes/table';
import type { Policy, RunDetail, RunIndexEntry } from '../data/types';
import { Panel } from '../glass/Panel';
import { runLabel, sortRuns } from '../lib/run';
import { setTransport, transportSnapshot } from '../playback/store';
import {
  cycleSortMode,
  promote,
  selectEnv,
  selectPolicy,
  selectRun,
  toggleDebug,
  useSelection,
} from '../state/selectionStore';
import { STR } from '../strings';
import { seedsFor } from '../theater/loader';
import { SPEEDS } from '../url/codec';
import { PANEL_VIEWS, type PanelView } from './StatsPanel';

function Shortcut({ children }: { children: string }) {
  return (
    <kbd className="ml-auto rounded-sm border border-white/10 bg-white/5 px-1.5 font-mono text-2xs leading-4 text-ink-faint">
      {children}
    </kbd>
  );
}

function Item({
  onPick,
  hint,
  shortcut,
  children,
}: {
  onPick: () => void;
  hint: string;
  shortcut?: string;
  children: ReactNode;
}) {
  return (
    <Command.Item
      onSelect={onPick}
      title={hint}
      className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-ink-dim transition-colors hover:bg-white/10 data-[selected=true]:bg-white/10 data-[selected=true]:text-ink"
    >
      {children}
      {shortcut !== undefined && <Shortcut>{shortcut}</Shortcut>}
    </Command.Item>
  );
}

export function CommandPalette({
  entries,
  detail,
  envList,
  setView,
  openWall,
  panelHidden,
  togglePanel,
  fullscreenTarget,
  onClose,
}: {
  entries: readonly RunIndexEntry[] | undefined;
  detail: RunDetail | null;
  envList: readonly EnvId[];
  setView: (view: PanelView) => void;
  openWall: () => void;
  panelHidden: boolean;
  togglePanel: () => void;
  fullscreenTarget: { current: HTMLElement | null };
  onClose: () => void;
}) {
  const sel = useSelection();
  const runs = useMemo(() => sortRuns(entries ?? []).slice(0, 12), [entries]);
  const seeds = useMemo(
    () => (detail === null || sel.envId === null ? [] : seedsFor(detail.recordings, sel.envId)),
    [detail, sel.envId],
  );
  const policies: Policy[] = ['oak', ...(['ppo', 'random'] as const).filter((p) => detail?.baselines[p] !== undefined)];
  const done = (act: () => void) => () => {
    act();
    onClose();
  };
  const groupClass =
    '[&_[cmdk-group-heading]]:px-2.5 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:pt-2.5 [&_[cmdk-group-heading]]:text-2xs [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-label [&_[cmdk-group-heading]]:text-ink-faint';

  return (
    <div
      className="absolute inset-0 z-50 flex justify-center bg-black/40 pt-[16vh]"
      onPointerDown={onClose}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onClose();
      }}
    >
      <div className="h-fit w-[600px] max-w-[calc(100vw-2rem)]" onPointerDown={(event) => event.stopPropagation()}>
        <Panel className="overflow-hidden">
          <Command label={STR.app}>
            <Command.Input
              autoFocus
              placeholder={STR.palette.placeholder}
              title={STR.palette.placeholder}
              className="w-full border-b border-white/10 bg-transparent px-3.5 py-3 text-sm text-ink outline-none placeholder:text-ink-faint"
            />
            <Command.List className={`max-h-[46vh] overflow-y-auto p-1.5 ${groupClass}`}>
              <Command.Empty>
                <div className="px-2.5 py-4 text-center text-xs text-ink-faint">{STR.palette.empty}</div>
              </Command.Empty>
              <Command.Group heading={STR.palette.environments}>
                {envList.map((envId) => (
                  <Item key={envId} onPick={done(() => selectEnv(envId))} hint={STR.wall.watchEnv(nameOf(envId))} shortcut="[ ]">
                    <span
                      className="size-2 rounded-full"
                      style={{ background: `var(--color-env-${slugOf(envId)})` }}
                    />
                    {nameOf(envId)}
                  </Item>
                ))}
              </Command.Group>
              <Command.Group heading={STR.palette.runs}>
                {runs.map((run) => (
                  <Item key={run.id} onPick={done(() => selectRun(run.id))} hint={STR.run.switchTo(runLabel(run))}>
                    {run.inFlight && <span className="size-1.5 motion-safe:animate-pulse rounded-full bg-good" />}
                    {runLabel(run)}
                  </Item>
                ))}
              </Command.Group>
              <Command.Group heading={STR.palette.seeds}>
                {seeds.map((seed) => (
                  <Item key={seed} onPick={done(() => promote(seed, () => true))} hint={STR.seeds.promote(seed)}>
                    {STR.palette.seed(seed)}
                  </Item>
                ))}
              </Command.Group>
              <Command.Group heading={STR.palette.policies}>
                {policies.map((policy) => (
                  <Item key={policy} onPick={done(() => selectPolicy(policy))} hint={STR.bar.watch(STR.policy[policy])} shortcut="p">
                    {STR.palette.watch(STR.policy[policy])}
                  </Item>
                ))}
              </Command.Group>
              <Command.Group heading={STR.palette.panels}>
                {PANEL_VIEWS.map((view, index) => (
                  <Item
                    key={view}
                    onPick={done(() => setView(view))}
                    hint={STR.palette.showKey(STR.panels[view], String(index + 1))}
                    shortcut={String(index + 1)}
                  >
                    {STR.palette.show(STR.panels[view])}
                  </Item>
                ))}
                <Item onPick={done(openWall)} hint={STR.tile.open}>
                  {STR.palette.wall}
                </Item>
              </Command.Group>
              <Command.Group heading={STR.palette.actions}>
                <Item
                  onPick={done(() => setTransport({ playing: !transportSnapshot().playing }))}
                  hint={STR.palette.playPause}
                  shortcut="space"
                >
                  {STR.palette.playPause}
                </Item>
                <Item
                  onPick={done(() => {
                    const at = (SPEEDS as readonly number[]).indexOf(transportSnapshot().speed);
                    setTransport({ speed: SPEEDS[(at + 1) % SPEEDS.length] ?? 1 });
                  })}
                  hint={STR.palette.nextSpeed}
                >
                  {STR.palette.nextSpeed}
                </Item>
                <Item onPick={done(toggleDebug)} hint={STR.bar.debug} shortcut="g">
                  {STR.palette.debug}
                </Item>
                <Item onPick={done(cycleSortMode)} hint={STR.seeds.sort} shortcut="s">
                  {STR.palette.sort}
                </Item>
                <Item
                  onPick={done(togglePanel)}
                  hint={panelHidden ? STR.bar.showPanel : STR.bar.hidePanel}
                  shortcut="c"
                >
                  {panelHidden ? STR.palette.showPanel : STR.palette.hidePanel}
                </Item>
                <Item
                  onPick={done(() => {
                    if (document.fullscreenElement !== null) void document.exitFullscreen();
                    else fullscreenTarget.current?.requestFullscreen().catch(() => {});
                  })}
                  hint={STR.bar.fullscreen}
                  shortcut="f"
                >
                  {STR.palette.fullscreen}
                </Item>
              </Command.Group>
            </Command.List>
          </Command>
        </Panel>
      </div>
    </div>
  );
}
