import type { Page } from '@playwright/test';
import publishedModel from '../public/models/deltrel/manifest.json';
import { parseDeltrelAiDecision, type DeltrelAiDecision } from '../src/lib/deltrel/ai/decision';
import type { DeltrelNetworkHead, DeltrelNetworkOutput } from '../src/lib/deltrel/ai/network-output';
import { predictionsFromNetworkOutput } from '../src/lib/deltrel/ai/predictions';
import type { DeltrelAiRequest } from '../src/lib/deltrel/ai/protocol';
import {
  DELTREL_AI_WORKER_PROTOCOL_VERSION,
  parseWorkerCommand,
  type DeltrelAiWorkerCommand,
  type DeltrelAiWorkerEvent,
} from '../src/lib/deltrel/ai/worker-protocol';

function gate() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function head(shape: number[], mask: boolean[], sigmoid = false, probabilities?: number[]): DeltrelNetworkHead {
  const width = shape.at(-1)!;
  const values = probabilities ?? mask.map((active, i) => active
    ? sigmoid ? 0.5 : 1 / mask.slice(Math.floor(i / width) * width, (Math.floor(i / width) + 1) * width).filter(Boolean).length
    : 0);
  return {
    shape, mask, activation: sigmoid ? 'sigmoid' : 'softmax', applicable: true,
    probabilities: values,
    logits: mask.map((active, i) => active ? probabilities ? Math.log(values[i]) : 0 : null),
  };
}

function decision(command: Extract<DeltrelAiWorkerCommand, { type: 'choose' }>, modelVersion: string): DeltrelAiDecision {
  const { request } = command;
  const state = request.state;
  const n = state.stones.length;
  const empty = state.stones.map((stone) => stone === -1);
  const all = (length: number) => Array<boolean>(length).fill(true);
  const networkOutput: DeltrelNetworkOutput = {
    schemaVersion: 1, perspective: state.toMove, nodeCount: n, auxiliaryStatus: 'ready',
    heads: {
      policy: head([n], empty), outcome: head([2], all(2), false, [0.2, 0.8]),
      scoreMargin: head([303], all(303)), ownership: head([n, 3], all(n * 3)),
      alive: head([n], all(n), true), softPolicy: head([n], empty),
      opponentReply: { ...head([n + 1], [...empty, true]), applicable: empty.filter(Boolean).length > state.movesLeft },
      secondStone: { ...head([n], empty), applicable: state.mode === 'double' && !state.opening && state.movesLeft === 2 },
      finalShores: head([2, 51], Array.from({ length: 102 }, (_, i) => i % 51 <= state.rings * 5)),
      finalNetworks: head([2, 26], Array.from({ length: 52 }, (_, i) => i % 26 <= Math.floor(state.rings * 5 / 2))),
      finalCapes: head([2, 6], all(12)),
    },
  };
  const action = { type: 'place' as const, node: request.legalActions[0] };
  const search = command.search ?? { simulations: 512, maxConsidered: 16 };
  return parseDeltrelAiDecision(request, {
    response: {
      schema: request.schema, version: request.version, requestId: request.requestId,
      rulesHash: request.rulesHash, stateHash: request.stateHash, action,
    },
    analysis: {
      perspective: state.toMove, stateHash: request.stateHash,
      outcome: { loss: 0.2, win: 0.8 }, modelValue: 0.6, searchValue: 0.6, rootValue: 0.2,
      swapRecommended: false, expectedMargin: 0,
      rootActions: [action], rootPolicy: [1], rootQ: [0.2], rootVisits: [search.simulations],
      modelVersion, modelStep: 42, modelIdentity: null, ...search,
      timingMs: { queue: 0, modelLoad: 0, inferenceSearch: 1, total: 1 },
      networkOutput, predictions: predictionsFromNetworkOutput(networkOutput, state, false),
    },
  });
}

/** Deterministic UI fixture: preserve the real worker protocol and cancellation lifecycle. */
export async function installAiWorkerFixture(page: Page, options: {
  modelVersion?: string;
  holdPreparation?: boolean;
  holdMoves?: boolean;
  heldCalls?: number[];
} = {}) {
  const modelVersion = options.modelVersion ?? 'inspection-fixture';
  const release = { ...publishedModel, model_version: modelVersion };
  const readyInfo = { modelVersion, bytes: release.artifacts.onnx.bytes, backend: 'wasm' as const, cached: true };
  const preparation = gate(), moves = gate();
  const calls = new Map((options.heldCalls ?? []).map((call) => [call, gate()]));
  const requests: DeltrelAiRequest[] = [];
  const deliveries = new Map<string, ReturnType<typeof gate>>();
  const discardedRequests = new Set<string>();
  if (!options.holdPreparation) preparation.resolve();
  if (!options.holdMoves) moves.resolve();

  // Local-engine scenarios must not depend on the configured cloud service.
  // Cloud-specific tests install their own health and move routes instead.
  await page.route('**/v2/health', route => route.fulfill({
    status: 503,
    json: { error: { code: 'deltrel_ai_unavailable', message: 'Cloud AI is offline.', retryable: true } },
  }));
  await page.route('**/models/deltrel/manifest.json', route => route.fulfill({ json: release }));
  await page.exposeFunction('__deltrelAiFixtureDelivered', (taskId: string, discarded: boolean) => {
    if (discarded) discardedRequests.add(taskId);
    deliveries.get(taskId)?.resolve();
  });
  await page.exposeFunction('__deltrelAiFixture', async (value: unknown): Promise<DeltrelAiWorkerEvent | null> => {
    const command = parseWorkerCommand(value);
    if (command.type === 'cancel') return null;
    if (command.type === 'prepare') {
      await preparation.promise;
      return { type: 'prepared', taskId: command.taskId, release, info: readyInfo };
    }
    requests.push(command.request);
    deliveries.set(command.taskId, gate());
    const pending = calls.get(requests.length);
    await moves.promise;
    if (pending) await pending.promise;
    return { type: 'result', taskId: command.taskId, decision: decision(command, modelVersion) };
  });
  await page.addInitScript(({ protocolVersion, readyInfo, release }) => {
    const bridge = window as unknown as {
      __deltrelAiFixture(command: DeltrelAiWorkerCommand): Promise<DeltrelAiWorkerEvent | null>;
      __deltrelAiFixtureDelivered(taskId: string, discarded: boolean): Promise<void>;
    };
    class FixtureWorker extends EventTarget {
      private terminated = false;
      constructor() {
        super();
        queueMicrotask(() => this.emit({ type: 'ready', protocolVersion }));
      }
      private emit(event: DeltrelAiWorkerEvent) {
        if (!this.terminated) this.dispatchEvent(new MessageEvent('message', { data: event }));
      }
      postMessage(command: DeltrelAiWorkerCommand) {
        if (command.type === 'choose') {
          this.emit({ type: 'prepared', taskId: command.taskId, info: readyInfo, release });
          this.emit({ type: 'search-progress', taskId: command.taskId,
            progress: { completedSimulations: 1, totalSimulations: command.search?.simulations ?? 512 } });
        }
        void bridge.__deltrelAiFixture(command).then(event => {
          if (event) this.emit(event);
          if (event?.type === 'result') return bridge.__deltrelAiFixtureDelivered(event.taskId, this.terminated);
        }).catch(error => this.emit({ type: 'error', taskId: command.taskId,
          error: { code: 'internal', message: String(error), retryable: false } }));
      }
      terminate() { this.terminated = true; }
    }
    Object.defineProperty(window, 'Worker', { configurable: true, writable: true, value: FixtureWorker });
  }, { protocolVersion: DELTREL_AI_WORKER_PROTOCOL_VERSION, readyInfo, release });
  return {
    requests,
    discardedRequests,
    releasePreparation: preparation.resolve,
    releaseMoves: moves.resolve,
    async releaseCall(call: number) {
      const request = requests[call - 1];
      if (!request) throw new Error(`AI call ${call} has not started.`);
      calls.get(call)?.resolve();
      await deliveries.get(request.requestId)?.promise;
    },
  };
}
