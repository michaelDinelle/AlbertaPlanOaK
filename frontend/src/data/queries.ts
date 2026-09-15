import { queryOptions } from '@tanstack/react-query';
import { fetchRunDetail, fetchRuns } from './oak';

export function runsQuery() {
  return queryOptions({
    queryKey: ['runs'],
    queryFn: ({ signal }) => fetchRuns(signal),
    refetchInterval: (query) => (query.state.data?.some((entry) => entry.inFlight) === true ? 5000 : false),
  });
}

export function runDetailQuery(id: string) {
  return queryOptions({
    queryKey: ['run', id],
    queryFn: ({ signal }) => fetchRunDetail(id, signal),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
