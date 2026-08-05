// #/theater — RETIRED. The Wall is the one way to watch: promoting a seed on the
// wall IS the theater (full-fidelity hero, event timeline, debug overlay). These
// routes survive only as redirects so old deep links keep meaning something.

import type { ViewMount } from '../router';

export const mount: ViewMount = (el, params) => {
  const runId = params['id'];
  const envId = params['envId'];
  const seed = params['seed'];
  const q = new URLSearchParams();
  if (runId) q.set('run', runId);
  if (envId) q.set('env', envId);
  if (seed) q.set('seed', seed);
  const target = q.size > 0 ? `#/?${q.toString()}` : '#/';
  // replace() keeps Back working (the retired URL never enters history)
  window.location.replace(target);
  void el;
  return { destroy() {} };
};
