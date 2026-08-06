
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
  window.location.replace(target);
  void el;
  return { destroy() {} };
};
