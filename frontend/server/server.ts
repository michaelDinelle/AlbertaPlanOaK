
import { createServer } from './app';

const { server, app } = createServer();

const mode = process.env['NODE_ENV'] === 'production' ? 'production' : 'development';
console.log(`[oak-frontend] API on http://${server.hostname}:${server.port} (${mode})`);
console.log(`[oak-frontend] bench root: ${app.benchRoot}`);

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
