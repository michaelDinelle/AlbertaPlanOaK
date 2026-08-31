import { clsx } from 'clsx';
import { STAGE_ORDER, type StageKey } from './content';

const CX = 232;
const CY = 226;
const RADIUS = 130;
const NODE = 37;
const STEP = (Math.PI * 2) / STAGE_ORDER.length;

function angleAt(index: number): number {
  return -Math.PI / 2 + index * STEP;
}

function polar(radius: number, angle: number): readonly [number, number] {
  return [CX + radius * Math.cos(angle), CY + radius * Math.sin(angle)];
}

function connector(index: number): string {
  const gap = Math.asin((NODE + 8) / RADIUS);
  const [x1, y1] = polar(RADIUS, angleAt(index) + gap);
  const [x2, y2] = polar(RADIUS, angleAt(index + 1) - gap);
  return `M ${x1} ${y1} A ${RADIUS} ${RADIUS} 0 0 1 ${x2} ${y2}`;
}

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} viewBox="0 0 9 9" refX="7.5" refY="4.5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M0.5 0.5 L8.5 4.5 L0.5 8.5 z" fill={color} />
    </marker>
  );
}

export function CycleDiagram({
  selected,
  onSelect,
}: {
  selected: StageKey;
  onSelect: (key: StageKey) => void;
}) {
  const selectedIndex = STAGE_ORDER.findIndex((stage) => stage.key === selected);
  return (
    <svg viewBox="0 0 464 404" role="presentation" className="w-full select-none">
      <defs>
        <Arrow id="oak-cycle-arrow" color="var(--color-ink-dim)" />
        <Arrow id="oak-cycle-arrow-live" color="var(--color-accent)" />
        <radialGradient id="oak-cycle-core" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--color-accent)" stopOpacity="0.13" />
          <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
        </radialGradient>
      </defs>

      <circle cx={CX} cy={CY} r={RADIUS - NODE - 10} fill="url(#oak-cycle-core)" />
      <circle
        cx={CX}
        cy={CY}
        r={RADIUS}
        fill="none"
        stroke="var(--color-line)"
        strokeWidth="1"
        strokeDasharray="2 6"
        opacity="0.55"
      />

      {STAGE_ORDER.map((stage, index) => {
        const live = index === selectedIndex;
        return (
          <path
            key={`edge-${stage.key}`}
            d={connector(index)}
            fill="none"
            stroke={live ? 'var(--color-accent)' : 'var(--color-ink-dim)'}
            strokeWidth={live ? 1.8 : 1.1}
            strokeLinecap="round"
            opacity={live ? 1 : 0.6}
            markerEnd={live ? 'url(#oak-cycle-arrow-live)' : 'url(#oak-cycle-arrow)'}
            className="transition-all duration-300"
          />
        );
      })}

      <text
        x={CX}
        y={CY - 14}
        textAnchor="middle"
        className="fill-ink font-mono text-[12px] uppercase"
        style={{ letterSpacing: '0.16em' }}
      >
        one stream
      </text>
      <text x={CX} y={CY + 4} textAnchor="middle" className="fill-ink-dim text-[12px]">
        every step, identically
      </text>
      <text x={CX} y={CY + 21} textAnchor="middle" className="fill-ink-dim text-[12px]">
        no episodes, no discounting
      </text>

      {STAGE_ORDER.map((stage, index) => {
        const angle = angleAt(index);
        const [x, y] = polar(RADIUS, angle);
        const [lx, ly] = polar(RADIUS + NODE + 21, angle);
        const live = stage.key === selected;
        const top = stage.name.length - 1;
        return (
          <g
            key={stage.key}
            onClick={() => onSelect(stage.key)}
            onMouseEnter={() => onSelect(stage.key)}
            className="cursor-pointer"
            tabIndex={0}
            role="button"
            aria-label={stage.name.join(' ')}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') onSelect(stage.key);
            }}
          >
            <circle cx={x} cy={y} r={NODE + 9} fill="transparent" />
            <circle
              cx={x}
              cy={y}
              r={NODE}
              fill={live ? 'color-mix(in oklab, var(--color-accent) 17%, transparent)' : 'oklch(100% 0 0 / 0.04)'}
              stroke={live ? 'var(--color-accent)' : 'var(--color-line)'}
              strokeWidth={live ? 1.6 : 1}
              className="transition-all duration-300"
            />
            <text
              x={x}
              y={y + 7}
              textAnchor="middle"
              className={clsx('font-mono text-[20px] transition-colors duration-300', live ? 'fill-accent' : 'fill-ink')}
            >
              {stage.badge}
            </text>
            {stage.name.map((line, row) => (
              <text
                key={line}
                x={lx}
                y={ly + (row - top) * 13 + 4}
                textAnchor="middle"
                className={clsx('text-[12px] transition-colors duration-300', live ? 'fill-ink' : 'fill-ink-dim')}
              >
                {line}
              </text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
