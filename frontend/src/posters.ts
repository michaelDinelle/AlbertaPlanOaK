import * as THREE from 'three/webgpu';
import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { celebrationPulse } from '@kit/celebrate';
import { buildComposer } from '@kit/post';
import { TOKENS } from '@kit/tokens';
import type { FrameState, SceneSize } from '@kit/types';
import { blockOf, deriveBlock, frameOf, type Block } from '@sim/events';
import { emptyStream, type EnvDecoder, type EnvFrame } from '@sim/types';
import { environmentName, type EnvDescriptor } from './data/envs';
import { fetchCatalog, fetchRecording, fetchRuns } from './data/oak';
import { HEADER_BYTES, readHeader } from './data/replay';
import { envLanes } from './derive';
import { sceneOf } from './scenes/table';

const STEP = 500_000;
const HISTORY_BLOCKS = 16;
const CHUNK_BLOCKS = 16;
const PROBE_BYTES = 32768;
const SIZE: SceneSize = { width: 1280, height: 720, dpr: 1 };
const POSTER = { width: 512, height: 288 };

async function bytesOf(url: string, first: number, last: number): Promise<ArrayBuffer> {
  const response = await fetch(url, { headers: { Range: `bytes=${first}-${last}` } });
  if (response.status !== 206) { await response.body?.cancel(); throw new Error(`replay is not cached yet (${response.status}); open it in the theater once`); }
  return response.arrayBuffer();
}

async function tapeAt(url: string, decoder: EnvDecoder) {
  let probe = await bytesOf(url, 0, PROBE_BYTES - 1);
  const view = new DataView(probe), offset = HEADER_BYTES + Math.ceil(view.getUint32(8, true) / 4) * 4, row = view.getUint32(12, true) * 4;
  if (probe.byteLength < offset + row) probe = await bytesOf(url, 0, offset + row - 1);
  const header = readHeader(new Uint8Array(probe))!, frames = header.blockFrames;
  const initial = frameOf(blockOf(probe.slice(offset, offset + row)), 0, 0, header);
  const target = Math.floor(STEP / frames), keep = Math.max(0, target - HISTORY_BLOCKS);
  const events = emptyStream(frames), blocks = new Map<number, Block>();
  for (let first = decoder.sequential ? 0 : keep; first <= target; first += CHUNK_BLOCKS) {
    const last = Math.min(target, first + CHUNK_BLOCKS - 1), startRow = Math.max(0, first * frames - 1), endRow = Math.min(header.frames, (last + 1) * frames);
    const bytes = await bytesOf(url, header.offset + startRow * row, header.offset + endRow * row - 1);
    for (let index = first; index <= last; index++) {
      const from = index * frames, at = from - startRow, count = Math.min(frames, header.frames - from);
      const prev = from > 0 ? frameOf(blockOf(bytes.slice((at - 1) * row, at * row)), 0, from - 1, header) : null;
      const block = blockOf(bytes.slice(at * row, (at + count) * row)), derived = deriveBlock(block, from, header, decoder, prev);
      events.blocks[index] = derived;
      events.count += derived.steps.length;
      if (index >= keep) blocks.set(index, block);
    }
  }
  const frameAt = (i: number): EnvFrame => {
    const step = Math.max(0, Math.min(STEP, Math.floor(i))), block = blocks.get(Math.floor(step / frames));
    if (step === 0) return initial;
    return block ? frameOf(block, step % frames, step, header) : frameAt(STEP);
  };
  return { header, initial, events, frameAt };
}

export async function renderPosters(): Promise<void> {
  const failed: string[] = [], began = performance.now();
  let written = 0;
  try {
    const [catalog, runs] = await Promise.all([fetchCatalog(), fetchRuns()]);
    const best = runs.find((run) => run.role === 'best')!, recording = await fetchRecording(best.id);
    const canvas = document.body.appendChild(document.createElement('canvas'));
    const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
    renderer.setClearColor(0x05070c, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    await Promise.all([renderer.init(), MeshoptSimplifier.ready]);
    renderer.setPixelRatio(SIZE.dpr);
    renderer.setSize(SIZE.width, SIZE.height, false);
    const out = new OffscreenCanvas(POSTER.width, POSTER.height), draw = out.getContext('2d')!;
    const first = new Map<string, EnvDescriptor>();
    for (const environment of catalog) if (!first.has(environmentName(environment))) first.set(environmentName(environment), environment);
    for (const [name, environment] of first) {
      try {
        const spec = sceneOf(environment), decoder = spec.decoder, { lanes, seeds } = envLanes(recording.meta, environment.envId);
        const [loaded, tape] = await Promise.all([spec.load(), tapeAt(`/api/runs/${encodeURIComponent(best.id)}/replay/${lanes[0]!}`, decoder)]);
        const module = loaded.default();
        await module.init({ renderer, canvas, size: SIZE, tokens: TOKENS, envId: environment.envId, environment, geometry: decoder.geometry(tape.header.descriptor, tape.initial), seed: seeds[0]!, time: () => STEP / module.controlHz });
        const composer = buildComposer(renderer, module.scene, module.camera, module.post);
        module.resize(SIZE);
        const frame = tape.frameAt(STEP), pose = new Float32Array(decoder.slots), scratch = new Float32Array(decoder.slots);
        decoder.decode(frame, pose);
        const state: FrameState = { pose, observations: frame.observations, reward: frame.reward, reset: frame.reset, done: frame.done !== 0, step: frame.step, simTime: frame.step / module.controlHz };
        module.update({ state, prev: null, events: [], eventLog: tape.events, dtReal: 0, scrubbed: true, position: STEP, controlHz: module.controlHz, frameAt: tape.frameAt, poseAt: (i) => { decoder.decode(tape.frameAt(i), scratch); return scratch; } });
        await composer.compile();
        composer.setBloomPulse(celebrationPulse(module.scene).bloom);
        composer.setGlass([]);
        composer.render();
        draw.drawImage(canvas, 0, 0, POSTER.width, POSTER.height);
        module.dispose();
        composer.dispose();
        await fetch(`/api/posters/${name}`, { method: 'POST', body: await out.convertToBlob({ type: 'image/webp', quality: 0.92 }) });
        written++;
      } catch (error) {
        failed.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  } catch (error) {
    failed.push(error instanceof Error ? error.message : String(error));
  }
  document.title = `posters: ${written} written in ${((performance.now() - began) / 1000).toFixed(1)} s${failed.length ? `; failed ${failed.join(' | ')}` : ''}`;
}
