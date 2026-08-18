import { fetchTape, type Tape } from './parse';
import {
  buildSeedRows,
  nextRowDueMs,
  rhoBinned,
  survivalSeries,
  tapeWindowSteps,
  windowedRates,
} from './derive';
import type { DeriveMode, DeriveResults, DeriveSpec, TapeProgress, WorkerRequest, WorkerResponse } from './store';

const tapes = new Map<string, Tape>();
const inflight = new Map<string, Promise<Tape>>();

function progressOf(tape: Tape): TapeProgress {
  return {
    rows: tape.rows,
    byteLength: tape.byteLength,
    lastStep: tape.rows > 0 ? tape.stepEnd[tape.rows - 1] ?? 0 : 0,
    nextRowDueMs: nextRowDueMs(tape),
    windowSteps: tapeWindowSteps(tape),
  };
}

function tailTape(runId: string): Promise<Tape> {
  const running = inflight.get(runId);
  if (running !== undefined) return running;
  const promise = fetchTape(runId, tapes.get(runId) ?? null).then(
    (tape) => {
      tapes.set(runId, tape);
      inflight.delete(runId);
      return tape;
    },
    (error: unknown) => {
      inflight.delete(runId);
      throw error;
    },
  );
  inflight.set(runId, promise);
  return promise;
}

function ensureTape(runId: string): Promise<Tape> {
  const have = tapes.get(runId);
  if (have !== undefined) return Promise.resolve(have);
  return tailTape(runId);
}

function derive(tape: Tape, spec: DeriveSpec): DeriveResults[DeriveMode] {
  switch (spec.mode) {
    case 'rates':
    case 'windowedRates':
      return windowedRates(tape, spec.lanes, spec.win);
    case 'rho':
      return rhoBinned(tape, spec.lanes);
    case 'survival':
      return survivalSeries(tape, spec.lanes);
    case 'rows':
      return buildSeedRows(
        tape,
        { envSchedule: spec.envSchedule, seeds: spec.seeds, steps: spec.steps },
        spec.envId,
        new Set(spec.recorded),
      );
  }
}

async function handle(request: WorkerRequest): Promise<WorkerResponse> {
  if (request.op === 'tail') {
    const tape = await tailTape(request.runId);
    return { id: request.id, ok: true, progress: progressOf(tape) };
  }
  const tape = await ensureTape(request.runId);
  return { id: request.id, ok: true, progress: progressOf(tape), result: derive(tape, request.spec) };
}

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse): void;
};

scope.onmessage = (event) => {
  void handle(event.data).then(
    (response) => scope.postMessage(response),
    (error: unknown) =>
      scope.postMessage({
        id: event.data.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }),
  );
};
