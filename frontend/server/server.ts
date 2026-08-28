import { createServer } from './app';

const { server, benchRoot } = createServer();

const mode = process.env['NODE_ENV'] === 'production' ? 'production' : 'development';
console.log(`[oak-frontend] API on http://${server.hostname}:${server.port} (${mode})`);
console.log(`[oak-frontend] bench root: ${benchRoot}`);

const shutdown = (): void => {
  void server.stop(true);
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export { server };
