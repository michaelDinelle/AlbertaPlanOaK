// #/runs — the run gallery.
//
// It used to be the home page; `#/` is the Wall now, and this is the index you
// go to when you want the ~600 roots rather than one run playing. `GALLERY_PATH`
// is the single constant every "back to the list" link in the app reads, so the
// route can move again without leaving orphan links behind.
//
// The signature element is the env spectrum: a fixed-order strip of accent
// cells, one per env in the run, widths proportional to seed share. It uses the
// same per-env accents the 3D scenes use (kit/tokens.ts), so a run's identity in
// the gallery matches its identity in the theater. The strip is an accent, never
// the only encoding — the env names are always printed beside it.

import type { ViewMount } from '../router';
import type { RunDetail, RunIndexEntry } from '@shared/types';
import { TOKENS } from '@kit/tokens';
import { ENVS, isEnvId, type EnvId } from '@shared/envs';
import {
  EMPTY_FILTER,
  aggregateSeries,
  envIdsInIndex,
  envIdsOf,
  envName,
  envShares,
  envSlug,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatStep,
  loadRunDetail,
  matchesFilter,
  policiesIn,
  primaryResult,
  roles as rolesResource,
  runIndex,
  sortRuns,
  type EnvShare,
  type GalleryFilter,
} from '../state';
import {
  actionButton,
  badge,
  el,
  emptyState,
  errorState,
  eyebrow,
  spinner,
  viewShell,
} from '../components/table';
import { sparkline, type SparklineHandle } from '../components/sparkline';

/** Where the gallery lives. The Wall owns `#/`. */
export const GALLERY_PATH = '#/runs';

export function accentCss(slug: string): string {
  const value = TOKENS.accents[slug];
  if (value === undefined) return '#5c6a80';
  return `#${value.toString(16).padStart(6, '0')}`;
}

/**
 * The strip: one cell per env, ordered by the canonical env table.
 *
 * Two inputs, deliberately unequal in what they claim:
 *   * `EnvShare[]` — TRUE apportionment from `envShares(meta)` (i.e. from
 *     `meta.envSchedule`): cell widths are the real seed fractions and each cell
 *     names its seed count.
 *   * a run-index entry — IDENTITY ONLY: equal-width cells and no seed numbers,
 *     because the index carries `envIds` and `batch` but never the split between
 *     them. Inventing `batch / envIds.length` there printed confident fiction on
 *     every mixed run (the default schedule is 1024/128/64/64/64/64, not 6×235).
 *     Callers that later hydrate the detail swap the strip for the share form.
 */
export function envSpectrum(source: Pick<RunIndexEntry, 'envIds'> | readonly EnvShare[]): HTMLElement {
  let shares: readonly EnvShare[] | null = null;
  let ids: EnvId[];
  if ('envIds' in source) {
    ids = envIdsOf(source);
  } else {
    shares = source;
    ids = source.map((share) => share.envId);
  }
  const strip = el('div', {
    class: 'spectrum',
    ariaLabel: ids.length > 0 ? `Environments: ${ids.map((id) => envName(id)).join(', ')}` : 'No environments',
    role: 'img',
  });
  if (ids.length === 0) {
    strip.append(el('span', { class: 'spectrum-cell', style: { flexGrow: '1', background: '#232c3d' } }));
    return strip;
  }
  ids.forEach((envId, index) => {
    const info = ENVS[envId];
    const share = shares?.[index];
    strip.append(
      el('span', {
        class: 'spectrum-cell',
        title: share ? `${info.name} — ${formatCount(share.lanes)} seeds` : info.name,
        style: {
          flexGrow: String(share ? Math.max(0.05, share.fraction) : 1),
          background: accentCss(info.slug),
        },
      }),
    );
  });
  return strip;
}

/**
 * Which filters are actually narrowing the list, in plain words, each with the
 * edit that would undo it.
 *
 * The zero-match state has to answer "why is this empty?", and with six
 * independent controls (search, env, and four toggles) the honest answer is the
 * list of what is currently on — a user who set `recordings` twenty runs ago and
 * then typed a name sees an empty grid and blames the search box.
 */
export interface FilterFacet {
  key: keyof GalleryFilter;
  label: string;
}

export function activeFacets(filter: GalleryFilter): FilterFacet[] {
  const facets: FilterFacet[] = [];
  const search = filter.search.trim();
  if (search) facets.push({ key: 'search', label: `name “${search}”` });
  if (filter.envId) facets.push({ key: 'envId', label: envName(filter.envId) });
  if (filter.completeOnly) facets.push({ key: 'completeOnly', label: 'complete' });
  if (filter.recordingsOnly) facets.push({ key: 'recordingsOnly', label: 'recordings' });
  if (filter.inFlightOnly) facets.push({ key: 'inFlightOnly', label: 'running' });
  if (filter.rolesOnly) facets.push({ key: 'rolesOnly', label: 'roles' });
  return facets;
}

// The gallery PAGE is retired: the run picker on the wall header carries its
// search/roles/live/recordings affordances, and #/runs now lands on the wall.
// The helpers above (accentCss, envSpectrum, theaterHref, roleBadge…) live on —
// they are imported by the science module and the live view.
export const mount: ViewMount = (host) => {
  window.location.replace('#/');
  void host;
  return { destroy() {} };
};
