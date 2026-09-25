import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ORIGIN = 'http://localhost:3000';
const VIEW = { width: 1280, height: 720 };
const CELL = { width: 640, height: 360, columns: 3 };
const SETTLE_FRAMES = 1;

interface Shot { env: string; step: number; seed?: number; debug: boolean; dpr: number; width: number; height: number; format: string; frames?: number }
interface Taken { url: string; spread: number; frameMs: { median: number; p95: number; max: number } | null }
type Send = (method: string, params?: object, sessionId?: string) => Promise<Record<string, any>>;

const events = new Set<(message: Record<string, any>) => void>();
const FAILURE = /(Could not [^\n]*|No scene is registered[^\n]*|WebGPU is unavailable[^\n]*|Scene initialization failed[^\n]*|Replay [^\n]*(?:differs|incomplete|interrupted|changed)[^\n]*|Unknown recording[^\n]*|Invalid [^\n]*)/;

async function launch(dpr: number) {
  const profile = mkdtempSync(join(tmpdir(), 'oak-capture-'));
  const chrome = Bun.spawn([CHROME, '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', '--hide-scrollbars',
    `--window-size=${VIEW.width},${VIEW.height}`, `--force-device-scale-factor=${dpr}`, 'about:blank'], { stdout: 'ignore', stderr: 'ignore' });
  const portFile = join(profile, 'DevToolsActivePort');
  let port = '', path = '';
  for (let i = 0; i < 400 && !path; i++) {
    if (existsSync(portFile)) [port = '', path = ''] = readFileSync(portFile, 'utf8').split('\n');
    if (!path) await Bun.sleep(25);
  }
  if (!path) throw new Error('Chrome did not open its debugging port');
  const socket = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((done, fail) => { socket.addEventListener('open', done, { once: true }); socket.addEventListener('error', fail, { once: true }); });
  let next = 0;
  const pending = new Map<number, (message: Record<string, any>) => void>();
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(String(event.data));
    if (message.id === undefined) { for (const listen of events) listen(message); return; }
    pending.get(message.id)?.(message);
    pending.delete(message.id);
  });
  socket.addEventListener('close', () => {
    for (const settle of pending.values()) settle({ error: { message: 'Chrome closed its debugging socket' } });
    pending.clear();
  });
  const send: Send = (method, params = {}, sessionId) => new Promise((done, fail) => {
    const id = ++next, timer = setTimeout(() => { pending.delete(id); fail(new Error(`${method}: Chrome did not answer within 20 s`)); }, 20_000);
    pending.set(id, (message) => { clearTimeout(timer); if (message.error) fail(new Error(`${method}: ${message.error.message}`)); else done(message.result); });
    socket.send(JSON.stringify({ id, method, params, sessionId }));
  });
  return { send, close() { socket.close(); chrome.kill(); rmSync(profile, { recursive: true, force: true }); } };
}

async function evaluate(send: Send, sessionId: string, expression: string) {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  return result.result.value;
}

const dataUrlOf = (blob: string) => `new Promise((done, fail) => { const reader = new FileReader(); reader.onload = () => done(reader.result); reader.onerror = () => fail(reader.error); reader.readAsDataURL(${blob}); })`;

interface Page { targetId: string; sessionId: string; errors: string[]; listen: (message: Record<string, any>) => void }
interface Worker { browser: Awaited<ReturnType<typeof launch>> }

const catalog: { envId: string; seedStart: number }[] = await (await fetch(`${ORIGIN}/api/catalog`)).json();
const seedStart = new Map(catalog.map((entry) => [entry.envId, entry.seedStart]));

async function openPage(send: Send, dpr: number): Promise<Page> {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const page: Page = { targetId, sessionId, errors: [], listen: (message) => {
    if (message.sessionId !== sessionId) return;
    const details = message.method === 'Runtime.exceptionThrown' ? message.params.exceptionDetails : null;
    const text = details ? details.exception?.description ?? details.text
      : message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error' ? message.params.args.map((arg: Record<string, any>) => arg.value ?? arg.description ?? '').join(' ') : null;
    if (text && !page.errors.includes(text)) page.errors.push(text);
  } };
  events.add(page.listen);
  await send('Runtime.enable', {}, sessionId);
  await send('Emulation.setDeviceMetricsOverride', { width: VIEW.width, height: VIEW.height, deviceScaleFactor: dpr, mobile: false }, sessionId);
  await send('Page.enable', {}, sessionId);
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `{ const Socket = WebSocket; window.WebSocket = Object.assign(function (url, protocols) { return new Socket(String(url).includes('/_bun/hmr') ? 'ws://127.0.0.1:1' : url, protocols); }, { prototype: Socket.prototype, CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 }); }` }, sessionId);
  return page;
}

async function shoot(worker: Worker, shot: Shot): Promise<Taken> {
  const seed = shot.seed ?? seedStart.get(shot.env);
  if (seed === undefined) throw new Error(`${shot.env} is not an envId in the catalog`);
  const send = worker.browser.send, page = await openPage(send, shot.dpr), { sessionId } = page;
  const query = new URLSearchParams({ env: shot.env, step: String(shot.step), seed: String(seed), ...(shot.debug ? { debug: '1' } : {}) });
  try {
    await send('Page.navigate', { url: `${ORIGIN}/?${query}` }, sessionId);
    const deadline = Date.now() + 180_000;
    for (let styled = false; ;) {
      const state = await evaluate(send, sessionId, `(() => {
        if (document.title === 'Bun - Build Failed') {
          const encoded = /atob\\("([^"]+)"\\)/.exec([...document.scripts].map((script) => script.textContent).join(' '));
          const decoded = encoded ? atob(encoded[1]) : '';
          const file = (/src\\/[\\w\\/.-]+\\.tsx?/.exec(decoded) ?? ['an unknown file'])[0], message = (decoded.match(/[\\x20-\\x7e]{6,}/g) ?? []).filter((part) => !part.includes('src/'))[0] ?? '';
          return { step: -1, shown: false, loading: false, panel: false, failure: 'bundle build failed in ' + file + ': ' + message };
        }
        const scrub = document.querySelector('input[aria-label="Drag anywhere in the run"]'), text = document.body?.innerText ?? '';
        const failure = text.match(${FAILURE}), card = document.querySelector('button[title="Choose an environment"]')?.textContent ?? '';
        return { step: scrub ? Number(scrub.value) : -1, shown: (/\\bseed (\\d+)/.exec(card) ?? [])[1] === ${JSON.stringify(String(seed))},
          loading: text.includes('Loading scene and replay') || /\\bbuffering\\b/.test(text), panel: document.querySelector('aside') !== null, failure: failure ? failure[1] : null };
      })()`);
      if (state.failure) throw new Error(`${shot.env}: ${state.failure}${page.errors.length ? ` | ${page.errors[0]!.split('\n')[0]}` : ''}`);
      if (!styled && state.step >= 0) {
        await evaluate(send, sessionId, `document.getElementById('capture-glass') ?? document.head.append(Object.assign(document.createElement('style'), { id: 'capture-glass', textContent: '.glass { display: none !important; }' }))`);
        if (state.panel) await evaluate(send, sessionId, `window.dispatchEvent(new KeyboardEvent('keydown', { key: 'c' }))`);
        styled = true;
      }
      if (state.step === Math.floor(shot.step) && !state.loading && state.shown) break;
      if (Date.now() > deadline) throw new Error(`${shot.env} never reached step ${shot.step}`);
      await Bun.sleep(100);
    }
    const taken: Taken = await evaluate(send, sessionId, `new Promise((done, fail) => {
      setTimeout(() => fail(new Error('the scene stopped rendering frames')), 15000);
      let frames = ${SETTLE_FRAMES}, last = 0;
      const deltas = [], timing = ${shot.frames ?? 0};
      if (timing) document.querySelector('button[aria-label^="Play"]')?.click();
      const grab = (now) => {
        if (last) deltas.push(now - last);
        last = now;
        if (--frames > 0 || deltas.length < ${SETTLE_FRAMES} + timing - 1) { requestAnimationFrame(grab); return; }
        const source = document.querySelector('canvas.theater-canvas'), out = new OffscreenCanvas(${shot.width} || source.width, ${shot.height} || source.height);
        out.getContext('2d').drawImage(source, 0, 0, out.width, out.height);
        const probe = new OffscreenCanvas(32, 18).getContext('2d');
        probe.drawImage(source, 0, 0, 32, 18);
        const px = probe.getImageData(0, 0, 32, 18).data;
        let sum = 0, square = 0;
        for (let i = 0; i < px.length; i += 4) { const l = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]; sum += l; square += l * l; }
        const n = px.length / 4, spread = Math.sqrt(Math.max(0, square / n - (sum / n) ** 2));
        const timed = deltas.slice(${SETTLE_FRAMES}).sort((a, b) => a - b);
        const frameMs = timing ? { median: timed[timed.length >> 1], p95: timed[Math.floor(timed.length * 0.95)], max: timed[timed.length - 1] } : null;
        if (timing) document.querySelector('button[aria-label^="Pause"]')?.click();
        out.convertToBlob({ type: '${shot.format}', quality: 0.92 }).then((blob) => ${dataUrlOf('blob')}.then((url) => done({ url, spread, frameMs }), fail), fail);
      };
      requestAnimationFrame(grab);
    })`);
    for (const error of page.errors.slice(0, 3)) console.log(`page error at ${shot.env}@${shot.step}: ${error.split('\n')[0]!.slice(0, 300)}`);
    return taken;
  } finally {
    events.delete(page.listen);
    await send('Target.closeTarget', { targetId: page.targetId }).catch(() => {});
  }
}

async function tile(send: Send, urls: readonly (string | null)[], labels: readonly string[]): Promise<string> {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  try {
    return await evaluate(send, sessionId, `(async () => {
      const urls = ${JSON.stringify(urls)}, labels = ${JSON.stringify(labels)}, { width, height, columns: most } = ${JSON.stringify(CELL)}, columns = Math.min(most, urls.length);
      const sheet = new OffscreenCanvas(columns * width, Math.ceil(urls.length / columns) * height), g = sheet.getContext('2d');
      g.font = '13px monospace';
      for (const [i, url] of urls.entries()) {
        const x = (i % columns) * width, y = Math.floor(i / columns) * height;
        if (url) g.drawImage(await createImageBitmap(await (await fetch(url)).blob()), x, y, width, height);
        else { g.fillStyle = '#5a0f14'; g.fillRect(x, y, width, height); }
        g.fillStyle = 'rgba(0, 0, 0, 0.72)';
        g.fillRect(x, y, g.measureText(labels[i]).width + 12, 20);
        g.fillStyle = '#fff';
        g.fillText(labels[i], x + 6, y + 14);
      }
      const blob = await sheet.convertToBlob({ type: 'image/png' });
      return await ${dataUrlOf('blob')};
    })()`);
  } finally {
    await send('Target.closeTarget', { targetId });
  }
}

const save = (url: string, out: string) => writeFileSync(out, Buffer.from(url.slice(url.indexOf(',') + 1), 'base64'));
const formatOf = (out: string) => out.endsWith('.png') ? 'image/png' : 'image/webp';

const USAGE = 'usage: bun capture.ts <envId> <step> <out.webp|out.png> [seed] [--debug] [--dpr 2] [--size 1280x720] | bun capture.ts sheet <out.png> <cell>... [--dpr 2] [--jobs 2] | bun capture.ts perf <cell>... | bun capture.ts posters   (cell: <envId>@<step>[~seed][+debug]; more than 9 cells write <out>-01.png, <out>-02.png, ...)';
const FLAT = 4;
const STALL = /did not answer|closed its debugging socket|No target with given id|Failed to open a new tab/;
const cellOf = (spec: string): Omit<Shot, 'dpr' | 'width' | 'height' | 'format'> => {
  const match = /^([^@~+]+)@(\d+(?:\.\d+)?)(?:~(\d+))?(\+debug)?$/.exec(spec);
  if (!match) throw new Error(`${spec} is not <envId>@<step>[~seed][+debug]`);
  return { env: match[1]!, step: Number(match[2]), seed: match[3] === undefined ? undefined : Number(match[3]), debug: match[4] !== undefined };
};
const args = Bun.argv.slice(2), flag = (name: string) => { const at = args.indexOf(name); return at < 0 ? undefined : args.splice(at, 2)[1]; };
const debug = args.includes('--debug'), dpr = Number(flag('--dpr') ?? 1), jobs = Math.max(1, Number(flag('--jobs') ?? 2)), [width, height] = (flag('--size') ?? '0x0').split('x').map(Number);
const positional = args.filter((arg) => arg !== '--debug');
const workers: Worker[] = [];
const pool = async (n: number) => {
  const fresh = await Promise.all(Array.from({ length: Math.max(0, n - workers.length) }, async () => ({ browser: await launch(dpr) })));
  workers.push(...fresh);
  return workers.slice(0, n);
};
const attempt = async (worker: Worker, shot: Shot): Promise<Taken> => {
  try {
    return await shoot(worker, shot);
  } catch (error) {
    if (STALL.test(String(error))) {
      worker.browser.close();
      worker.browser = await launch(dpr);
    }
    throw error;
  }
};
const each = async <T>(items: readonly T[], n: number, work: (worker: Worker, item: T, index: number) => Promise<void>) => {
  let next = 0;
  await Promise.all((await pool(Math.min(n, items.length))).map(async (worker) => {
    while (next < items.length) { const index = next++; await work(worker, items[index]!, index); }
  }));
};
try {
  if (positional[0] === 'posters') {
    const [worker] = await pool(1), send = worker!.browser.send, page = await openPage(send, dpr);
    await send('Page.navigate', { url: `${ORIGIN}/?posters=1` }, page.sessionId);
    for (let title = ''; !title.startsWith('posters:'); await Bun.sleep(250)) title = await evaluate(send, page.sessionId, 'document.title');
    console.log(await evaluate(send, page.sessionId, 'document.title'));
    for (const error of page.errors.slice(0, 3)) console.log(`page error: ${error.split('\n')[0]!.slice(0, 300)}`);
  } else if (positional[0] === 'sheet') {
    const [, out, ...specs] = positional;
    if (!out || !specs.length) throw new Error(USAGE);
    const urls: (string | null)[] = specs.map(() => null), lines = specs.map(() => '');
    await each(specs, jobs, async (worker, spec, index) => {
      const began = performance.now(), took = () => `${((performance.now() - began) / 1000).toFixed(1)} s`;
      try {
        const taken = await attempt(worker, { ...cellOf(spec), dpr, width: 0, height: 0, format: 'image/png' });
        urls[index] = taken.url;
        lines[index] = `cell ${spec}: spread ${taken.spread.toFixed(1)}${taken.spread < FLAT ? ' FLAT' : ''} (${took()})`;
      } catch (error) {
        lines[index] = `cell ${spec}: FAILED ${String(error).split('\n')[0]!.slice(0, 240)} (${took()})`;
      }
    });
    for (const line of lines) console.log(line);
    for (let k = 0; k < specs.length; k += 9) {
      const file = specs.length > 9 ? resolve(out).replace(/\.png$/, `-${String(k / 9 + 1).padStart(2, '0')}.png`) : resolve(out);
      save(await tile(workers[0]!.browser.send, urls.slice(k, k + 9), specs.slice(k, k + 9)), file);
      console.log(file);
    }
  } else if (positional[0] === 'perf') {
    const specs = positional.slice(1);
    if (!specs.length) throw new Error(USAGE);
    await each(specs, 1, async (worker, spec) => {
      try {
        const { frameMs } = await attempt(worker, { ...cellOf(spec), dpr, width: 64, height: 36, format: 'image/png', frames: 240 });
        console.log(`perf ${spec}: median ${frameMs!.median.toFixed(2)} ms  p95 ${frameMs!.p95.toFixed(2)} ms  max ${frameMs!.max.toFixed(2)} ms`);
      } catch (error) {
        console.log(`perf ${spec}: FAILED ${String(error).split('\n')[0]!.slice(0, 240)}`);
      }
    });
  } else {
    const [env, step, out, seed] = positional;
    if (!env || !step || !out) throw new Error(USAGE);
    const [worker] = await pool(1);
    save((await attempt(worker!, { env, step: Number(step), seed: seed === undefined ? undefined : Number(seed), debug, dpr, width: width!, height: height!, format: formatOf(out) })).url, resolve(out));
    console.log(resolve(out));
  }
} finally {
  for (const worker of workers) worker.browser.close();
}
