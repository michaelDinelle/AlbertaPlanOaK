import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
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
import type { Policy, RunDetail } from '@shared/types';
import { formatCount, formatStep } from '../../lib/format';
import { setTransport, useTransport } from '../../playback/store';
import { selectPolicy, toggleDebug, useSelection } from '../../state/selectionStore';
import {
  currentStep,
  scrubToFrame,
  seekToTick,
  stepBySteps,
  subscribeStageFrame,
  useStageMeta,
} from '../../theater/controller';
import { formatClock } from '../../theater/player';
import { SPEEDS } from '../../url/codec';
import { navigate } from '../../url/useUrlState';

const POLICY_LABEL: Record<Policy, string> = { oak: 'oak', ppo: 'ppo', random: 'rand' };

function TBtn({
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
        'flex h-7 min-w-7 items-center justify-center rounded-sm px-1.5 font-mono text-2xs transition-colors',
        on ? 'bg-accent/15 text-accent' : 'text-ink-dim hover:bg-hover hover:text-ink',
        disabled === true && 'pointer-events-none opacity-35',
      )}
    >
      {children}
    </button>
  );
}

function Sep() {
  return <div className="mx-0.5 h-5 w-px bg-line" />;
}

export function Transport({
  detail,
  stride,
  fullscreenTarget,
}: {
  detail: RunDetail | null;
  stride: number;
  fullscreenTarget: RefObject<HTMLElement | null>;
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
        const frames = Math.max(1, frame.frameCount - 1);
        const percent = `${((frame.position / frames) * 100).toFixed(3)}%`;
        if (fillRef.current !== null) fillRef.current.style.width = percent;
        if (headRef.current !== null) headRef.current.style.left = percent;
        if (!scrubbing.current && inputRef.current !== null) inputRef.current.value = String(frame.position);
        const now = performance.now();
        if (now - lastReadoutAt.current < 150 || readoutRef.current === null) return;
        lastReadoutAt.current = now;
        readoutRef.current.textContent =
          `step ${frame.step < 0 ? 'init' : formatCount(frame.step)} / ${formatStep(frame.totalSteps)}`
          + ` · ${formatClock(frame.simTime)}`;
      }),
    [],
  );

  useEffect(() => {
    const onChange = (): void => {
      setFullscreen(document.fullscreenElement !== null);
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => {
      document.removeEventListener('fullscreenchange', onChange);
    };
  }, []);

  const idle = meta.status !== 'ready';
  const frames = Math.max(1, meta.frameCount - 1);
  const loadedPct = meta.totalSteps > 0 ? Math.min(100, (meta.loadedSteps / meta.totalSteps) * 100) : 0;
  const jump = Math.max(1, stride);
  const baselines = (['ppo', 'random'] as const).filter((p) => detail?.baselines[p] !== undefined);
  let busiest = 1;
  for (const tick of meta.ticks) if (tick.count > busiest) busiest = tick.count;

  const endScrub = (): void => {
    if (!scrubbing.current) return;
    scrubbing.current = false;
    navigate({ at: Math.floor(currentStep()) });
  };

  const toggleFullscreen = (): void => {
    if (document.fullscreenElement !== null) void document.exitFullscreen();
    else fullscreenTarget.current?.requestFullscreen().catch(() => {});
  };

  return (
    <div className={clsx('glass mt-2 flex items-center gap-1 rounded-md px-2 py-1.5', idle && 'opacity-80')}>
      <TBtn
        title={transport.playing ? 'Pause (space)' : 'Play (space)'}
        disabled={idle}
        onClick={() => setTransport({ playing: !transport.playing })}
      >
        {transport.playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
      </TBtn>
      <TBtn title="Back 200 windows (shift+←)" disabled={idle} onClick={() => stepBySteps(-200 * jump)}>
        <ChevronsLeft className="size-3.5" />
      </TBtn>
      <TBtn title="Back 20 windows (←)" disabled={idle} onClick={() => stepBySteps(-20 * jump)}>
        <ChevronLeft className="size-3.5" />
      </TBtn>
      <TBtn title="Forward 20 windows (→)" disabled={idle} onClick={() => stepBySteps(20 * jump)}>
        <ChevronRight className="size-3.5" />
      </TBtn>
      <TBtn title="Forward 200 windows (shift+→)" disabled={idle} onClick={() => stepBySteps(200 * jump)}>
        <ChevronsRight className="size-3.5" />
      </TBtn>

      <div className="relative mx-2 h-7 min-w-24 flex-1">
        <div className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-line" />
        <div
          className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-ink-faint/40"
          style={{ width: `${loadedPct.toFixed(2)}%` }}
        />
        <div
          ref={fillRef}
          className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-accent"
          style={{ width: '0%' }}
        />
        <div className="pointer-events-none absolute inset-0">
          {meta.ticks.map((tick) => {
            const density = Math.sqrt(tick.count / busiest);
            const height = 6 + 8 * density;
            return (
              <button
                key={`${tick.kind}-${tick.frame}`}
                type="button"
                title={tick.label}
                aria-label={`Seek to ${tick.label}`}
                onClick={() => seekToTick(tick)}
                className="pointer-events-auto absolute top-1/2 w-px -translate-x-1/2 bg-seed"
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
        <div
          ref={headRef}
          className="pointer-events-none absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-[0_0_6px_var(--color-accent)]"
          style={{ left: '0%' }}
        />
        <input
          ref={inputRef}
          type="range"
          min="0"
          max={String(frames)}
          step="any"
          defaultValue="0"
          disabled={idle}
          aria-label="Scrub the run"
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

      <div ref={readoutRef} className="whitespace-nowrap font-mono text-2xs text-ink-dim">
        {idle ? 'no recording' : 'step — / —'}
      </div>

      <Sep />
      <div className="hidden items-center lg:flex">
        {SPEEDS.map((preset) => (
          <TBtn
            key={preset}
            on={transport.speed === preset}
            title={`×${preset} real time (${preset} × the environment's own clock)`}
            onClick={() => {
              setTransport({ speed: preset });
              navigate({ speed: preset });
            }}
          >
            ×{preset}
          </TBtn>
        ))}
      </div>

      {baselines.length > 0 && (
        <>
          <Sep />
          {(['oak' as Policy, ...baselines]).map((p) => (
            <TBtn
              key={p}
              on={sel.policy === p}
              title={p === 'oak' ? 'The OaK fleet' : `The ${p} baseline fleet`}
              onClick={() => selectPolicy(p)}
            >
              {POLICY_LABEL[p]}
            </TBtn>
          ))}
        </>
      )}

      <Sep />
      <TBtn on={sel.debug} title="Ground-truth overlay (g)" onClick={toggleDebug}>
        <ScanEye className="size-3.5" />
      </TBtn>
      <TBtn
        on={fullscreen}
        title={fullscreen ? 'Exit fullscreen (f or Esc)' : 'Fullscreen (f)'}
        onClick={toggleFullscreen}
      >
        {fullscreen ? <Minimize className="size-3.5" /> : <Maximize className="size-3.5" />}
      </TBtn>
    </div>
  );
}
