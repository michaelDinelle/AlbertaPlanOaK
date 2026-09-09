import { queryOptions } from '@tanstack/react-query';
import { fetchRunDetail, fetchRuns } from './oak';

export const RUN_POLL_MS = 5000;

export function runsQuery() {
  return queryOptions({
    queryKey: ['runs'],
    queryFn: ({ signal }) => fetchRuns(signal),
    refetchInterval: (query) =>
      query.state.data?.some((entry) => entry.inFlight) === true ? RUN_POLL_MS : false,
  });
}

export function runDetailQuery(id: string, inFlight: boolean) {
  return queryOptions({
    queryKey: ['run', id],
    queryFn: ({ signal }) => fetchRunDetail(id, signal),
    refetchInterval: inFlight ? RUN_POLL_MS : false,
  });
}

export function baselineDetailQuery(id: string) {
  return queryOptions({
    queryKey: ['baseline', id],
    queryFn: ({ signal }) => fetchRunDetail(id, signal),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
