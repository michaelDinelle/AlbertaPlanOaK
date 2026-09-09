import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { ENV_IDS } from '../src/data/envs';
import { SCENES, nameOf } from '../src/scenes/table';

const origin = process.argv[2] ?? 'http://localhost:5173';
const only = process.argv[3] ?? '';
const run = process.argv[4] ?? '';

const CDP_PORT = 9411;
const VIEW_W = 1280;
const VIEW_H = 720;
const POSTER_W = 320;
const CAPTURE_AT_STEP = Number(process.argv[5] ?? '200000');
const SETTLE_MS = 3000;
const READY_TIMEOUT_MS = 120000;

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
        const pending = new Map<number, { resolve(value: any): void; reject(error: Error): void }>();
        let nextId = 1;
        ws.onmessage = (event) => {
          const msg = JSON.parse(String(event.data));
          if (msg.id !== undefined) {
            const request = pending.get(msg.id);
            pending.delete(msg.id);
            if (msg.error !== undefined) request?.reject(new Error(msg.error.message));
            else request?.resolve(msg);
          }
        };
        ws.onclose = () => {
          for (const request of pending.values()) request.reject(new Error('capture browser disconnected'));
          pending.clear();
        };
        return {
          send: (method, params = {}) =>
            new Promise((resolve, reject) => {
              const id = nextId++;
              const timer = setTimeout(() => {
                pending.delete(id);
                reject(new Error(`${method} timed out`));
              }, READY_TIMEOUT_MS);
              pending.set(id, {
                resolve: (value) => { clearTimeout(timer); resolve(value); },
                reject: (error) => { clearTimeout(timer); reject(error); },
              });
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
      `(async () => {
        const { stageMetaSnapshot } = await import('/src/theater/controller.ts');
        const { currentPosition, setTransport } = await import('/src/playback/store.ts');
        const meta = stageMetaSnapshot();
        if (meta.status === 'error') throw new Error(meta.noteDetail ?? meta.note);
        const canvas = document.querySelector('canvas.theater-canvas');
        const tile = [...document.querySelectorAll('button')].find((el) => el.title === ${JSON.stringify('Choose an environment')});
        const ready = canvas !== null && canvas.width > 0 && meta.status === 'ready'
          && meta.envName === ${JSON.stringify(envName)} && meta.loadedSteps >= ${CAPTURE_AT_STEP}
          && currentPosition() >= ${CAPTURE_AT_STEP}
          && tile !== undefined && tile.textContent.includes(${JSON.stringify(envName)});
        if (ready) setTransport({ playing: false });
        return ready;
      })()`,
    ).catch(() => false);
    if (ready === true) return;
    if (Date.now() > deadline) throw new Error(`${envName} never reached a ready stage`);
    await new Promise((res) => setTimeout(res, 250));
  }
}

for (const envId of ENV_IDS) {
  if (only !== '' && envId !== only) continue;
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
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: VIEW_W, height: VIEW_H, deviceScaleFactor: 1, mobile: false,
    });
    await mkdir(outDir, { recursive: true });
    const info = SCENES[envId];
    const runQuery = run !== '' ? `run=${run}&` : '';
    await cdp.send('Page.navigate', { url: `${origin}/?${runQuery}env=${envId}&at=${CAPTURE_AT_STEP}` });
    await waitForStage(cdp, nameOf(envId));
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
    cdp.close();
  } finally {
    chrome.kill();
    await chrome.exited;
    await rm(profileDir, { recursive: true, force: true });
  }
}
