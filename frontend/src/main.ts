import { startRouter, type Route } from './router';

// `#/` is the WALL — one run, one environment, every recorded seed at once. The
// run gallery moved to `#/runs`.
//
// THE RULE THAT FOLLOWS FROM THAT, and it is a rule rather than a claim, because a
// stale claim is how the last one went wrong: a link whose label promises a LIST OF
// RUNS ("Browse every run", "Back to the gallery", "Find a run with …") must point
// at `#/runs`. `#/` is a link to one run's population, and sending a reader who
// asked for the list to a single environment's stage is a broken promise, not a
// redirect. Only navigation that means "home / the wall" may target `#/`.
const routes: Route[] = [
  { pattern: '#/', view: () => import('./views/wall') },
  { pattern: '#/runs', view: () => import('./views/gallery') },
  { pattern: '#/run/:id', view: () => import('./views/run') },
  { pattern: '#/watch', view: () => import('./views/watch') },
  { pattern: '#/watch/:id', view: () => import('./views/watch') },
  { pattern: '#/theater', view: () => import('./views/theater') },
  { pattern: '#/theater/:id/:envId/:seed', view: () => import('./views/theater') },
  { pattern: '#/harness/:slug', view: () => import('./views/harness') },
];

const app = document.getElementById('app');
if (!app) throw new Error('#app missing');
startRouter(app, routes, '#/');
