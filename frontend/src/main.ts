import { startRouter, type Route } from './router';

const routes: Route[] = [
  { pattern: '#/', view: () => import('./views/wall') },
  { pattern: '#/runs', view: () => import('./views/wall') },
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
