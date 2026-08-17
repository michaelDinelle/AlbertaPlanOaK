import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { ENV_IDS, ENVS } from '../shared/envs';

const HELP = [
  'Regenerates the 16 environment poster images the app ships as static files.',
  'Start the dev stack first (bun run dev), then run:',
  '  bun scripts/posters.ts [origin]',
  'origin defaults to http://localhost:5173. Posters land in frontend/public/posters/<slug>.webp;',
  'commit the changed files. Set CHROME to use a different Chrome binary.',
].join('\n');

const CDP_PORT = 9411;
const VIEW_W = 1280;
const VIEW_H = 720;
const POSTER_W = 320;
const CAPTURE_AT_STEP = 200000;
const SETTLE_MS = 3000;
const READY_TIMEOUT_MS = 30000;

const origin = process.argv[2] ?? 'http://localhost:5173';
if (origin === '--help' || origin === '-h') {
  console.log(HELP);
  process.exit(0);
}

const chromeBin = process.env['CHROME'] ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const profileDir = join(tmpdir(), 'oak-posters-chrome');
const outDir = resolve(import.meta.dir, '../public/posters');

interface CdpClient {
  send(method: string, params?: object): Promise<any>;
  close(): void;
}

async function connect(): Promise<CdpClient> {
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const list = (await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json()) as {
        type: string;
        webSocketDebuggerUrl: string;
      }[];
      const page = list.find((t) => t.type === 'page');
      if (page !== undefined) {
        const ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((res, rej) => {
          ws.onopen = res;
          ws.onerror = rej;
        });
        const pending = new Map<number, (v: any) => void>();
        let nextId = 1;
        ws.onmessage = (event) => {
          const msg = JSON.parse(String(event.data));
          if (msg.id !== undefined) pending.get(msg.id)?.(msg);
        };
        return {
          send: (method, params = {}) =>
            new Promise((res) => {
              const id = nextId++;
              pending.set(id, res);
              ws.send(JSON.stringify({ id, method, params }));
            }),
          close: () => ws.close(),
        };
      }
    } catch {}
    await new Promise((res) => setTimeout(res, 250));
  }
  throw new Error(`no Chrome page on CDP port ${CDP_PORT}`);
}

async function evaluate(cdp: CdpClient, expression: string): Promise<unknown> {
  const res = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  const failure = res.result?.exceptionDetails;
  if (failure !== undefined) {
    throw new Error(
      `page evaluation failed: ${failure.exception?.description ?? failure.text ?? 'unknown page error'}`,
    );
  }
  return res.result?.result?.value;
}

async function waitForStage(cdp: CdpClient, envName: string): Promise<void> {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  for (;;) {
    const ready = await evaluate(
      cdp,
      `(() => {
        const canvas = document.querySelector('canvas.theater-canvas');
        const tile = [...document.querySelectorAll('button')].find((el) => el.title === ${JSON.stringify('Choose an environment')});
        return canvas !== null && tile !== undefined && tile.textContent.includes(${JSON.stringify(envName)});
      })()`,
    ).catch(() => false);
    if (ready === true) return;
    if (Date.now() > deadline) throw new Error(`${envName} never reached a ready stage`);
    await new Promise((res) => setTimeout(res, 250));
  }
}

const chrome = Bun.spawn(
  [
    chromeBin,
    '--headless=new',
    `--user-data-dir=${profileDir}`,
    `--remote-debugging-port=${CDP_PORT}`,
    '--no-first-run',
    '--no-default-browser-check',
    `--window-size=${VIEW_W},${VIEW_H}`,
    '--force-device-scale-factor=1',
    '--hide-scrollbars',
    'about:blank',
  ],
  { stdout: 'ignore', stderr: 'ignore' },
);

try {
  const cdp = await connect();
  await cdp.send('Page.enable');
  await mkdir(outDir, { recursive: true });
  for (const envId of ENV_IDS) {
    const info = ENVS[envId];
    await cdp.send('Page.navigate', { url: `${origin}/?env=${envId}&at=${CAPTURE_AT_STEP}` });
    await waitForStage(cdp, info.name);
    await new Promise((res) => setTimeout(res, SETTLE_MS));
    await evaluate(
      cdp,
      `(() => {
        const root = document.querySelector('#app > div');
        if (root === null) throw new Error('app root missing');
        [...root.children].slice(1).forEach((child) => { child.style.display = 'none'; });
        return true;
      })()`,
    );
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'webp',
      quality: 85,
      clip: { x: 0, y: 0, width: VIEW_W, height: VIEW_H, scale: POSTER_W / VIEW_W },
    });
    const file = resolve(outDir, `${info.slug}.webp`);
    await Bun.write(file, Buffer.from(shot.result.data, 'base64'));
    console.log(`${info.slug}.webp`);
  }
  cdp.close();
} finally {
  chrome.kill();
  await rm(profileDir, { recursive: true, force: true });
}
