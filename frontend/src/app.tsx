import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { ENVS } from '@shared/envs';
import { baselineDetailQuery, runDetailQuery, runsQuery } from './data/queries';
import { EnvStrip } from './features/env/EnvStrip';
import { Scoreboard } from './features/stage/Scoreboard';
import { SeedRail } from './features/stage/SeedRail';
import { Stage } from './features/stage/Stage';
import { Transport } from './features/stage/Transport';
import { TopBar } from './features/topbar/TopBar';
import { cycleEnv, keyAction } from './lib/keymap';
import { resolveFeaturedRun, runEnvs } from './lib/run';
import { setTransport, transportSnapshot } from './playback/store';
import {
  adoptDefaultEnv,
  adoptFromUrl,
  cycleSortMode,
  electLeader,
  resolveRun,
  selectEnv,
  selectionSnapshot,
  toggleDebug,
  unpromote,
  useSelection,
} from './state/selectionStore';
import { setLive, useTapeProgress } from './tape/store';
import { setTheaterInputs, stepBySteps } from './theater/controller';
import { seedsFor } from './theater/loader';
import { urlState } from './url/useUrlState';

function readPane(key: string, fallback: number): number {
  const raw = Number(typeof localStorage === 'undefined' ? NaN : localStorage.getItem(key));
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

function savePane(key: string, value: number): void {
  try {
    localStorage.setItem(key, String(value));
  } catch {}
}

function SectionHead({ label, accent }: { label: string; accent?: string }) {
  return (
    <div className="mb-2.5 flex items-center gap-2">
      {accent !== undefined && <span className="size-2 rounded-full" style={{ background: accent }} />}
      <h2 className="text-2xs uppercase tracking-[0.2em] text-ink-dim">{label}</h2>
      <div className="h-px flex-1 bg-line" />
    </div>
  );
}

function PlaceholderCard({
  title,
  className,
  style,
}: {
  title: string;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={clsx('flex flex-col rounded-md border border-dashed border-line bg-raised/40 px-4 py-3', className)}
      style={style}
    >
      <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{title}</div>
      <div className="flex flex-1 items-center justify-center py-5 font-mono text-2xs text-ink-faint">
        statistics land with summary v2
      </div>
    </div>
  );
}

function NoticeCard({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-md border border-line bg-raised/60 px-5 py-4">
      <div className="font-mono text-sm">{title}</div>
      {children !== undefined && <div className="mt-1 text-xs text-ink-dim">{children}</div>}
    </div>
  );
}

export function App() {
  const sel = useSelection();
  const runsQ = useQuery(runsQuery());
  const entries = runsQ.data;
  const entry = useMemo(() => entries?.find((run) => run.id === sel.runId) ?? null, [entries, sel.runId]);

  const detailQ = useQuery({
    ...runDetailQuery(sel.runId ?? '', entry?.inFlight === true),
    enabled: sel.runId !== null,
  });
  const detail = detailQ.data ?? null;

  const pointer = sel.policy === 'oak' ? undefined : detail?.baselines[sel.policy];
  const baselineQ = useQuery({ ...baselineDetailQuery(pointer ?? ''), enabled: pointer !== undefined });
  const active = sel.policy === 'oak' ? detail : pointer === undefined ? null : baselineQ.data ?? null;
  const baselinePending =
    sel.policy !== 'oak' && (detail === null ? detailQ.isPending : pointer !== undefined && baselineQ.isPending);

  const envList = useMemo(() => (detail === null ? [] : runEnvs(detail.meta, detail.recordings)), [detail]);
  const progress = useTapeProgress(sel.runId);
  const stride = Math.max(1, progress?.windowSteps ?? 1);

  useEffect(() => {
    const onPop = (): void => adoptFromUrl(urlState());
    window.addEventListener('popstate', onPop);
    return () => {
      window.removeEventListener('popstate', onPop);
    };
  }, []);

  useEffect(() => {
    if (sel.runId !== null || entries === undefined) return;
    const pick = resolveFeaturedRun(entries);
    if (pick.runId !== null) resolveRun(pick.runId);
  }, [entries, sel.runId]);

  useEffect(() => {
    if (detail === null || sel.envId !== null) return;
    const first = envList.find((id) => seedsFor(detail.recordings, id).length > 0) ?? envList[0];
    if (first !== undefined) adoptDefaultEnv(first);
  }, [detail, envList, sel.envId]);

  useEffect(() => {
    if (active === null || sel.envId === null || baselinePending) return;
    electLeader(seedsFor(active.recordings, sel.envId));
  }, [active, sel.envId, baselinePending]);

  useEffect(() => {
    setTheaterInputs({
      active,
      policy: sel.policy,
      baselinePointer: pointer,
      baselinePending,
      live: sel.policy === 'oak' && entry?.inFlight === true,
    });
  }, [active, sel.policy, pointer, baselinePending, entry]);

  const inFlight = entry?.inFlight === true;
  useEffect(() => {
    const runId = sel.runId;
    if (runId === null) return;
    setLive(runId, inFlight);
    return () => {
      setLive(runId, false);
    };
  }, [sel.runId, inFlight]);

  const stageColRef = useRef<HTMLDivElement | null>(null);
  const stageRowRef = useRef<HTMLDivElement | null>(null);
  const keyContext = useRef({ envList, stride });
  keyContext.current = { envList, stride };

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const node = event.target;
      if (
        node instanceof HTMLInputElement
        || node instanceof HTMLSelectElement
        || node instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const action = keyAction(event.key, event.shiftKey);
      if (action === null) return;
      event.preventDefault();
      switch (action.kind) {
        case 'toggle':
          setTransport({ playing: !transportSnapshot().playing });
          break;
        case 'step':
          stepBySteps(action.frames * keyContext.current.stride);
          break;
        case 'env': {
          const next = cycleEnv(keyContext.current.envList, selectionSnapshot().envId, action.direction);
          if (next !== null) selectEnv(next);
          break;
        }
        case 'sort':
          cycleSortMode();
          break;
        case 'debug':
          toggleDebug();
          break;
        case 'fullscreen':
          if (document.fullscreenElement !== null) void document.exitFullscreen();
          else stageColRef.current?.requestFullscreen().catch(() => {});
          break;
        case 'escape':
          unpromote();
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
    };
  }, []);

  const [railW, setRailW] = useState(() => readPane('oakWallRailW', 380));
  const railWRef = useRef(railW);
  railWRef.current = railW;
  const railDragging = useRef(false);

  const [chartsH, setChartsH] = useState(() => readPane('oakWallScienceH', 210));
  const chartsHRef = useRef(chartsH);
  chartsHRef.current = chartsH;
  const chartsDrag = useRef<{ startY: number; startH: number } | null>(null);

  const envName = sel.envId === null ? null : ENVS[sel.envId].name;
  const envAccent = sel.envId === null ? undefined : `var(--color-env-${ENVS[sel.envId].slug})`;

  return (
    <div className="min-h-screen">
      <TopBar entries={entries} entry={entry} />
      <EnvStrip detail={detail} envList={envList} />
      <main className="flex flex-col px-4 pb-12 pt-1">
        {runsQ.isError && (
          <NoticeCard title="Could not reach the API">
            {runsQ.error instanceof Error ? runsQ.error.message : String(runsQ.error)} — the dev server proxies
            /api to the frontend server; start it with `bun run dev`.
          </NoticeCard>
        )}
        {!runsQ.isError && entries !== undefined && sel.runId === null && (
          <NoticeCard title="No run to show yet">
            The stage plays a fleet's recorded seeds and the server found no science run with a tape.
          </NoticeCard>
        )}
        {sel.runId !== null && detailQ.isError && (
          <NoticeCard title={`Could not load ${sel.runId}`}>
            {detailQ.error instanceof Error ? detailQ.error.message : String(detailQ.error)}
          </NoticeCard>
        )}

        <section ref={stageRowRef} className="flex h-[62vh] min-h-96">
          <div ref={stageColRef} className="flex min-w-0 flex-1 flex-col bg-ground">
            <Stage />
            <Transport detail={detail} stride={stride} fullscreenTarget={stageColRef} />
          </div>
          <div
            role="separator"
            aria-label="Resize the seed panel"
            title="Drag to resize the seed panel"
            className="mx-1.5 w-1 cursor-col-resize self-stretch rounded-full bg-line/40 hover:bg-line-strong"
            onPointerDown={(event) => {
              railDragging.current = true;
              try {
                event.currentTarget.setPointerCapture(event.pointerId);
              } catch {}
            }}
            onPointerMove={(event) => {
              if (!railDragging.current || stageRowRef.current === null) return;
              const rect = stageRowRef.current.getBoundingClientRect();
              const width = rect.right - event.clientX;
              setRailW(Math.min(Math.max(Math.round(width), 260), Math.round(rect.width * 0.55)));
            }}
            onPointerUp={() => {
              if (!railDragging.current) return;
              railDragging.current = false;
              savePane('oakWallRailW', railWRef.current);
            }}
            onPointerCancel={() => {
              railDragging.current = false;
            }}
          />
          <aside className="flex min-h-0 flex-col gap-2" style={{ width: railW }}>
            <Scoreboard detail={detail} />
            <SeedRail detail={detail} active={active} />
          </aside>
        </section>

        <div
          role="separator"
          aria-label="Resize the chart row"
          title="Drag to resize the charts"
          className="my-3 h-1 cursor-row-resize rounded-full bg-line/40 hover:bg-line-strong"
          onPointerDown={(event) => {
            chartsDrag.current = { startY: event.clientY, startH: chartsHRef.current };
            try {
              event.currentTarget.setPointerCapture(event.pointerId);
            } catch {}
          }}
          onPointerMove={(event) => {
            const drag = chartsDrag.current;
            if (drag === null) return;
            setChartsH(Math.min(Math.max(Math.round(drag.startH + event.clientY - drag.startY), 140), 460));
          }}
          onPointerUp={() => {
            if (chartsDrag.current === null) return;
            chartsDrag.current = null;
            savePane('oakWallScienceH', chartsHRef.current);
          }}
          onPointerCancel={() => {
            chartsDrag.current = null;
          }}
        />

        <section>
          <SectionHead label={envName === null ? 'This environment' : `${envName} — this environment`} accent={envAccent} />
          <div className="grid gap-3 lg:grid-cols-4">
            <PlaceholderCard title="Reward rate" className="lg:col-span-4" style={{ height: chartsH }} />
            <PlaceholderCard title="Reward per head" />
            <PlaceholderCard title="ρ̄ vs measured" />
            <PlaceholderCard title="Survival" />
            <PlaceholderCard title="Forgetting" />
          </div>
        </section>

        <section className="mt-6">
          <SectionHead label="Across the suite" />
          <div className="grid gap-3 lg:grid-cols-4">
            <PlaceholderCard title="Forest plot" className="lg:col-span-2" />
            <PlaceholderCard title="Performance profiles" />
            <PlaceholderCard title="Compute frontier" />
            <PlaceholderCard title="Diagnostics" className="lg:col-span-4" />
          </div>
        </section>
      </main>
    </div>
  );
}
