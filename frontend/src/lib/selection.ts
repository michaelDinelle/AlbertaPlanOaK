import { clampPosition } from '../theater/player';

export interface HeroSelection {
  leader: number | null;
  promoted: number | null;
}

export function heroSeed(selection: HeroSelection): number | null {
  return selection.promoted ?? selection.leader;
}

export function promoteSeed(
  selection: HeroSelection,
  seed: number,
  playable: (seed: number) => boolean,
): HeroSelection {
  if (!playable(seed)) return selection;
  if (selection.promoted === seed) return selection;
  return { ...selection, promoted: seed };
}

export function escapeSelection(selection: HeroSelection): HeroSelection {
  if (selection.promoted === null) return selection;
  return { ...selection, promoted: null };
}

export function setLeader(
  selection: HeroSelection,
  leader: number | null,
  playable: (seed: number) => boolean,
): HeroSelection {
  const promoted = selection.promoted !== null && playable(selection.promoted) ? selection.promoted : null;
  return { leader, promoted };
}

export const SORT_MODES = ['rate', 'seed', 'retention'] as const;

export type SortMode = (typeof SORT_MODES)[number];

export function isSortMode(value: string): value is SortMode {
  return (SORT_MODES as readonly string[]).includes(value);
}

export function nextSort(sort: SortMode): SortMode {
  const index = SORT_MODES.indexOf(sort);
  return SORT_MODES[(index + 1) % SORT_MODES.length] ?? 'rate';
}

export interface SeedOrderKey {
  seed: number;
  rate: number | null;
  retention: number | null;
}

export function sortPermutation(rows: readonly SeedOrderKey[], sort: SortMode): number[] {
  const order = rows.map((_, index) => index);
  const direction = sort === 'seed' ? 1 : -1;
  const keyOf = (row: SeedOrderKey): number | null =>
    sort === 'seed' ? row.seed : sort === 'retention' ? row.retention : row.rate;
  order.sort((a, b) => {
    const rowA = rows[a];
    const rowB = rows[b];
    if (rowA === undefined || rowB === undefined) return a - b;
    const keyA = keyOf(rowA);
    const keyB = keyOf(rowB);
    const nullA = keyA === null || !Number.isFinite(keyA);
    const nullB = keyB === null || !Number.isFinite(keyB);
    if (nullA && nullB) return a - b;
    if (nullA) return 1;
    if (nullB) return -1;
    if (keyA !== keyB) return (keyA - keyB) * direction;
    return a - b;
  });
  return order;
}

export function frameFromStep(step: number, totalSteps: number, frameCount: number): number {
  const scale = totalSteps > 0 ? (frameCount - 1) / totalSteps : 1;
  return clampPosition(step * scale, frameCount);
}

export function stepFromFrame(frame: number, totalSteps: number, frameCount: number): number {
  const frames = frameCount - 1;
  return frames > 0 ? (frame * totalSteps) / frames : 0;
}
