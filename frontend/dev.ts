// Dev entry: start the API server in-process, then spawn Vite.
// Usage: bun dev.ts  (API on :5174, Vite on :5173 with /api proxy)
import './server/server.ts';

const vite = Bun.spawn(['bunx', 'vite'], {
  cwd: import.meta.dir,
  stdout: 'inherit',
  stderr: 'inherit',
});

process.on('SIGINT', () => {
  vite.kill();
  process.exit(0);
});
