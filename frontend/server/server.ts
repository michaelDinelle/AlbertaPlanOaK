// Entry point. Starts the API (and, under NODE_ENV=production, the built client) on
// 127.0.0.1:5174 — override with OAK_FRONTEND_PORT. Fleet-run store: OAK_BENCH_DIR,
// defaulting to <repo>/.oak-benchmarks.
//
// dev.ts imports this module for its side effect, so starting on import is intentional;
// the handler itself lives in app.ts and can be mounted without binding a port.

import { createServer } from './app';

const { server, app } = createServer();

const mode = process.env['NODE_ENV'] === 'production' ? 'production' : 'development';
console.log(`[oak-frontend] API on http://${server.hostname}:${server.port} (${mode})`);
console.log(`[oak-frontend] bench root: ${app.benchRoot}`);

// Warm the run index so the first /api/runs is fast; failures are non-fatal.
void app.index
  .list()
  .then((runs) => console.log(`[oak-frontend] indexed ${runs.length} run roots`))
  .catch((err: unknown) => console.error('[oak-frontend] index warmup failed:', err));

const shutdown = (): void => {
  void server.stop(true);
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export { server };
