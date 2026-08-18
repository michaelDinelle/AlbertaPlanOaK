import { useEffect, useRef, useState, type ReactNode } from 'react';
import { clsx } from 'clsx';
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Maximize,
  Minimize,
  Pause,
  Play,
  ScanEye,
} from 'lucide-react';
import type { Policy, RunDetail, RunIndexEntry } from '@shared/types';
import { Lens } from '../glass/Panel';
import { Label, POLICY_ORDER } from './charts';
import { formatCount, formatStep } from '../lib/format';
import { runLabelFor } from '../lib/run';
import { setTransport, useTransport } from '../playback/store';
import { selectPolicy, toggleDebug, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import {
  currentStep,
  scrubToFrame,
  seekToTick,
  stepBySteps,
  subscribeStageFrame,
  useStageMeta,
} from '../theater/controller';
import { formatClock } from '../theater/player';
import { SPEEDS } from '../url/codec';
import { navigate } from '../url/useUrlState';

function BarButton({
  on,
  disabled,
  title,
  onClick,
  children,
}: {
  on?: boolean;
  disabled?: boolean;
  title: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={clsx(
        'flex h-7 min-w-7 items-center justify-center rounded-md px-1.5 font-mono text-2xs transition-colors active:bg-white/15',
        disabled === true ? 'pointer-events-none text-ink-faint' : on === true ? 'bg-accent/20 text-accent' : 'text-ink hover:bg-white/10',
      )}
    >
      {children}
    </button>
  );
}

function Divider() {
  return <div className="mx-0.5 h-5 w-px bg-white/10" />;
}

const POLICY_ACTIVE: Record<Policy, string> = {
  oak: 'bg-oak/25 text-ink ring-1 ring-inset ring-oak/50',
  ppo: 'bg-ppo/25 text-ink ring-1 ring-inset ring-ppo/50',
  random: 'bg-random/25 text-ink ring-1 ring-inset ring-random/50',
};

export function PolicySwitch({
  runs,
  entries,
}: {
  runs: Record<Policy, RunDetail | null>;
  entries: readonly RunIndexEntry[] | undefined;
}) {
  const sel = useSelection();
  const labelOf = (policy: Policy): string | null => {
    const id = policy === 'oak' ? runs.oak?.id : runs.oak?.baselines[policy];
    return id === undefined ? null : runLabelFor(id, entries, runs[policy]);
  };
  return (
    <Lens className="flex h-12 items-center gap-1 px-2">
      <span className="px-1.5"><Label>{STR.bar.policyLabel}</Label></span>
      {POLICY_ORDER.map((policy) => {
        const run = labelOf(policy);
        const absent = run === null;
        const on = sel.policy === policy;
        return (
          <button
            key={policy}
            type="button"
            aria-disabled={absent}
            aria-pressed={on}
            title={
              absent
                ? policy === 'oak' ? STR.notices.noRun : STR.bar.noBaseline(STR.policy[policy])
                : STR.bar.watchRun(STR.policy[policy], run)
            }
            onClick={absent ? undefined : () => selectPolicy(policy)}
            className={clsx(
              'flex h-8 items-center rounded-md px-3 text-xs font-medium transition-colors',
              absent
                ? 'cursor-default text-ink-faint'
                : on
                  ? POLICY_ACTIVE[policy]
                  : 'text-ink-dim hover:bg-white/10 hover:text-ink active:bg-white/15',
            )}
          >
            {STR.policy[policy]}
          </button>
        );
      })}
    </Lens>
  );
}

export function PlaybackBar({
  stride,
  live,
  fullscreenTarget,
}: {
  stride: number;
  live: boolean;
  fullscreenTarget: { current: HTMLElement | null };
}) {
  const meta = useStageMeta();
  const transport = useTransport();
  const sel = useSelection();
  const [fullscreen, setFullscreen] = useState(false);
  const scrubbing = useRef(false);
  const lastReadoutAt = useRef(0);
  const fillRef = useRef<HTMLDivElement | null>(null);
  const headRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const readoutRef = useRef<HTMLDivElement | null>(null);

  useEffect(
    () =>
      subscribeStageFrame((frame) => {
        const frames = frame.frameCount - 1;
        const ratio = frames > 0 ? Math.min(1, Math.max(0, frame.position / frames)) : 0;
        if (fillRef.current !== null) fillRef.current.style.transform = `translateY(-50%) scaleX(${ratio.toFixed(5)})`;
        if (headRef.current !== null) headRef.current.style.transform = `translateX(${(ratio * 100).toFixed(3)}%)`;
        if (!scrubbing.current && inputRef.current !== null) inputRef.current.value = String(frame.position);
        const now = performance.now();
        if (now - lastReadoutAt.current < 150 || readoutRef.current === null) return;
        lastReadoutAt.current = now;
        readoutRef.current.textContent = STR.bar.readout(
          frame.step < 0 ? STR.bar.start : formatCount(frame.step),
          formatStep(frame.totalSteps),
          formatClock(frame.simTime),
        );
      }),
    [],
  );

  useEffect(() => {
    const onChange = (): void => setFullscreen(document.fullscreenElement !== null);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const idle = meta.status !== 'ready';
  const frames = Math.max(1, meta.frameCount - 1);
  const loadedPct = meta.totalSteps > 0 ? Math.min(100, (meta.loadedSteps / meta.totalSteps) * 100) : 0;
  const jump = Math.max(1, stride);
  let busiest = 1;
  for (const tick of meta.ticks) if (tick.count > busiest) busiest = tick.count;

  const endScrub = (): void => {
    if (!scrubbing.current) return;
    scrubbing.current = false;
    navigate({ at: Math.floor(currentStep()) });
  };

  const cycleSpeed = (): void => {
    const at = SPEEDS.indexOf(transport.speed as (typeof SPEEDS)[number]);
    const speed = SPEEDS[(at + 1) % SPEEDS.length] ?? 1;
    setTransport({ speed });
    navigate({ speed });
  };

  return (
    <Lens className={clsx('flex h-12 items-center gap-1 px-2', idle && 'opacity-80')}>
      <BarButton
        title={transport.playing ? STR.bar.pause : STR.bar.play}
        disabled={idle}
        onClick={() => setTransport({ playing: !transport.playing })}
      >
        {transport.playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
      </BarButton>
      <BarButton title={STR.bar.farBack} disabled={idle} onClick={() => stepBySteps(-200 * jump)}>
        <ChevronsLeft className="size-3.5" />
      </BarButton>
      <BarButton title={STR.bar.back} disabled={idle} onClick={() => stepBySteps(-20 * jump)}>
        <ChevronLeft className="size-3.5" />
      </BarButton>
      <BarButton title={STR.bar.ahead} disabled={idle} onClick={() => stepBySteps(20 * jump)}>
        <ChevronRight className="size-3.5" />
      </BarButton>
      <BarButton title={STR.bar.farAhead} disabled={idle} onClick={() => stepBySteps(200 * jump)}>
        <ChevronsRight className="size-3.5" />
      </BarButton>

      <div
        className="relative mx-2 h-7 min-w-32 flex-1 rounded-md transition-colors hover:bg-white/5 has-[:focus-visible]:ring-1 has-[:focus-visible]:ring-accent"
        title={STR.bar.scrub}
      >
        <div className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-white/15" />
        <div
          className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-ink-faint/40"
          style={{ width: `${loadedPct.toFixed(2)}%` }}
        />
        <div
          ref={fillRef}
          className="absolute inset-x-0 top-1/2 h-[3px] origin-left rounded-full bg-accent"
          style={{ transform: 'translateY(-50%) scaleX(0)', willChange: 'transform' }}
        />
        <div className="pointer-events-none absolute inset-0">
          {meta.ticks.map((tick) => {
            const density = Math.sqrt(tick.count / busiest);
            const height = 6 + 8 * density;
            return (
              <button
                key={`${tick.kind}-${tick.frame}`}
                type="button"
                title={STR.bar.seekTo(tick.label)}
                aria-label={STR.bar.seekTo(tick.label)}
                onClick={() => seekToTick(tick)}
                className="pointer-events-auto absolute top-1/2 w-px -translate-x-1/2 bg-seed transition-transform hover:scale-x-[3]"
                style={{
                  left: `${(tick.position * 100).toFixed(3)}%`,
                  opacity: (0.22 + 0.62 * density).toFixed(3),
                  height: `${height.toFixed(1)}px`,
                  marginTop: `${(-height / 2).toFixed(1)}px`,
                }}
              />
            );
          })}
        </div>
        <div ref={headRef} className="pointer-events-none absolute inset-0" style={{ willChange: 'transform' }}>
          <div className="absolute left-0 top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-[0_0_6px_var(--color-accent)]" />
        </div>
        <input
          ref={inputRef}
          type="range"
          min="0"
          max={String(frames)}
          step="any"
          defaultValue="0"
          disabled={idle}
          aria-label={STR.bar.scrub}
          title={STR.bar.scrub}
          className="absolute inset-0 w-full cursor-pointer appearance-none opacity-0"
          onPointerDown={() => {
            scrubbing.current = true;
          }}
          onPointerUp={endScrub}
          onPointerCancel={endScrub}
          onBlur={endScrub}
          onInput={(event) => scrubToFrame(Number(event.currentTarget.value))}
        />
      </div>

      <div ref={readoutRef} className="whitespace-nowrap font-mono text-2xs text-ink">
        {STR.bar.idle}
      </div>

      <Divider />
      <BarButton title={STR.bar.speed(transport.speed)} onClick={cycleSpeed}>
        ×{transport.speed}
      </BarButton>

      <Divider />
      <BarButton on={sel.debug} title={STR.bar.debug} onClick={toggleDebug}>
        <ScanEye className="size-3.5" />
      </BarButton>
      <BarButton
        on={fullscreen}
        title={fullscreen ? STR.bar.windowed : STR.bar.fullscreen}
        onClick={() => {
          if (document.fullscreenElement !== null) void document.exitFullscreen();
          else fullscreenTarget.current?.requestFullscreen().catch(() => {});
        }}
      >
        {fullscreen ? <Minimize className="size-3.5" /> : <Maximize className="size-3.5" />}
      </BarButton>
      {live && <span className="mx-1 size-1.5 motion-safe:animate-pulse rounded-full bg-good" title={STR.bar.live} />}
    </Lens>
  );
}
