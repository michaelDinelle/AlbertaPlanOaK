import './server/server.ts';

const vite = Bun.spawn(['bunx', 'vite', '--host'], {
  cwd: import.meta.dir,
  stdout: 'inherit',
  stderr: 'inherit',
});

process.on('SIGINT', () => {
  vite.kill();
  process.exit(0);
});
