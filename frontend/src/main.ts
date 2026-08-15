import { startRouter, type Route } from './router';

const routes: Route[] = [
  { pattern: '#/', view: () => import('./views/wall') },
  { pattern: '#/runs', view: () => import('./views/wall') },
];

const app = document.getElementById('app');
if (!app) throw new Error('#app missing');
startRouter(app, routes, '#/');
