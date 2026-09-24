import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameConfig } from '../../game';
import { DEFAULT_LOCAL_AI_TIMEOUT_MS, LocalDeltrelAiClient, type LocalAiRequestOptions } from '../local-client';
import {
  buildAiRequest,
  makeAiResponse,
  type DeltrelAiRequest,
} from '../protocol';
import publishedRelease from '../../../../../public/models/deltrel/manifest.json';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

function decisionFor(
  request: DeltrelAiRequest,
  search = { simulations: 1, maxConsidered: 1 },
) {
  return {
    response: makeAiResponse(request, { type: 'place' as const, node: 0 }),
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: { loss: 0.5, win: 0.5 },
      modelValue: 0,
      searchValue: 0,
      rootValue: 0,
      swapRecommended: false,
      expectedMargin: 0,
      rootActions: [{ type: 'place' as const, node: 0 }],
      rootPolicy: [1],
      rootQ: [0],
      rootVisits: [search.simulations],
      modelVersion: 'browser-test',
      modelStep: null,
      modelIdentity: 'browser-test',
      simulations: search.simulations,
      maxConsidered: search.maxConsidered,
      timingMs: {
        queue: 0,
        modelLoad: 0,
        inferenceSearch: 1,
        total: 1,
      },
    },
  };
}

class FakeWorker extends EventTarget {
  readonly messages: unknown[] = [];
  terminated = false;

  postMessage(message: unknown): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(message: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: message }));
  }
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('pinned browser champion', () => {
  it('keeps the verified release when a worker is disposed and recreated', async () => {
    vi.stubGlobal('Worker', class {});
    const workers: FakeWorker[] = [];
    const client = new LocalDeltrelAiClient(() => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker as unknown as Worker;
    });
    const info = { modelVersion: publishedRelease.model_version, bytes: publishedRelease.artifacts.onnx.bytes,
      backend: 'wasm', cached: true };
    const preparation = client.prepare();
    workers[0].emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    const preparedTask = (workers[0].messages[0] as { taskId: string }).taskId;
    workers[0].emit({ type: 'prepared', taskId: preparedTask, info, release: publishedRelease });
    await preparation;
    client.dispose();

    const request = buildAiRequest(config, [], 'pinned-release-search');
    const pending = client.request(request);
    workers[1].emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    expect(workers[1].messages[0]).toMatchObject({ type: 'choose', release: publishedRelease });
    const decision = decisionFor(request);
    decision.analysis.modelVersion = publishedRelease.model_version;
    workers[1].emit({ type: 'result', taskId: request.requestId, decision });
    await expect(pending).resolves.toMatchObject({ analysis: { modelVersion: publishedRelease.model_version } });

    const retry = client.prepare();
    await Promise.resolve();
    const retryCommand = workers[1].messages.at(-1) as { taskId: string };
    expect(retryCommand).toMatchObject({ type: 'prepare', release: publishedRelease });
    workers[1].emit({ type: 'prepared', taskId: retryCommand.taskId, info, release: publishedRelease });
    await retry;
    client.dispose();
  });

  it('rejects a different champion returned during a pinned session', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const preparation = client.prepare();
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    worker.emit({ type: 'prepared', taskId: (worker.messages[0] as { taskId: string }).taskId,
      info: { modelVersion: publishedRelease.model_version, bytes: publishedRelease.artifacts.onnx.bytes, backend: 'wasm', cached: true },
      release: publishedRelease });
    await preparation;
    const request = buildAiRequest(config, [], 'wrong-release-search');
    const pending = client.request(request);
    const rejection = expect(pending).rejects.toMatchObject({ code: 'protocol' });
    await Promise.resolve();
    worker.emit({ type: 'result', taskId: request.requestId, decision: decisionFor(request) });
    await rejection;
    client.dispose();
  });
});

describe('local search progress watchdog', () => {
  async function start(options: LocalAiRequestOptions = { search: { simulations: 64, maxConsidered: 8 } }) {
    vi.useFakeTimers();
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'progress-search');
    const onSearchProgress = vi.fn();
    const result = client.request(request, { ...options, onSearchProgress });
    const rejected = result.catch(error => error);
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    const progress = (completedSimulations: number, totalSimulations = 64, taskId = request.requestId) =>
      worker.emit({ type: 'search-progress', taskId, progress: { completedSimulations, totalSimulations } });
    return { worker, client, request, result, rejected, progress, onSearchProgress };
  }

  it('allows a real search to exceed the old deadline while completed simulations keep increasing', async () => {
    const f = await start();
    for (const completed of [1, 2, 63, 64]) {
      await vi.advanceTimersByTimeAsync(DEFAULT_LOCAL_AI_TIMEOUT_MS - 1);
      expect(f.worker.terminated).toBe(false);
      f.progress(completed);
    }
    f.worker.emit({ type: 'result', taskId: f.request.requestId,
      decision: decisionFor(f.request, { simulations: 64, maxConsidered: 8 }) });
    await expect(f.result).resolves.toMatchObject({ analysis: { simulations: 64 } });
    expect(f.onSearchProgress.mock.calls.map(([progress]) => progress.completedSimulations)).toEqual([1, 2, 63, 64]);
    f.client.dispose();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('still expires a stalled search despite duplicate, regressing, and unrelated progress', async () => {
    const f = await start();
    await vi.advanceTimersByTimeAsync(80_000);
    f.progress(2);
    await vi.advanceTimersByTimeAsync(80_000);
    f.progress(2);
    f.progress(1);
    f.progress(3, 64, 'some-other-request');
    await vi.advanceTimersByTimeAsync(9_999);
    expect(f.worker.terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await expect(f.rejected).resolves.toMatchObject({ code: 'timeout', retryable: true });
    expect(f.worker.terminated).toBe(true);
    expect(f.onSearchProgress).toHaveBeenCalledExactlyOnceWith({ completedSimulations: 2, totalSimulations: 64 });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not let preparation messages prolong a move without any completed simulation', async () => {
    const f = await start();
    await vi.advanceTimersByTimeAsync(89_000);
    f.worker.emit({ type: 'progress', taskId: f.request.requestId,
      progress: { phase: 'initializing', loadedBytes: 128, totalBytes: 128, modelVersion: 'test-model', cached: true } });
    f.worker.emit({ type: 'prepared', taskId: f.request.requestId,
      info: { modelVersion: 'test-model', bytes: 128, backend: 'wasm', cached: true } });
    await vi.advanceTimersByTimeAsync(1_000);
    await expect(f.rejected).resolves.toMatchObject({ code: 'timeout' });
    expect(f.onSearchProgress).not.toHaveBeenCalled();
    expect(f.worker.terminated).toBe(true);
  });

  it('preserves an explicit absolute deadline even while progress is advancing', async () => {
    const f = await start({ timeoutMs: 100, search: { simulations: 64, maxConsidered: 8 } });
    await vi.advanceTimersByTimeAsync(50);
    f.progress(1);
    await vi.advanceTimersByTimeAsync(49);
    f.progress(2);
    expect(f.worker.terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await expect(f.rejected).resolves.toMatchObject({ code: 'timeout' });
    expect(f.onSearchProgress).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('requires one stable total and binds it to an explicitly requested budget', async () => {
    const explicit = await start();
    explicit.progress(1, 32);
    await expect(explicit.rejected).resolves.toMatchObject({ code: 'protocol' });
    expect(explicit.onSearchProgress).not.toHaveBeenCalled();
    expect(explicit.worker.terminated).toBe(true);

    const implicit = await start({});
    implicit.progress(1, 8);
    implicit.progress(2, 16);
    await expect(implicit.rejected).resolves.toMatchObject({ code: 'protocol' });
    expect(implicit.onSearchProgress).toHaveBeenCalledExactlyOnceWith({ completedSimulations: 1, totalSimulations: 8 });
    expect(implicit.worker.terminated).toBe(true);
  });

  it('rejects a final result whose effort disagrees with the progress or requested candidate budget', async () => {
    for (const search of [{ simulations: 32, maxConsidered: 8 }, { simulations: 64, maxConsidered: 4 }]) {
      const f = await start();
      f.progress(1);
      f.worker.emit({ type: 'result', taskId: f.request.requestId, decision: decisionFor(f.request, search) });
      await expect(f.rejected).resolves.toMatchObject({ code: 'protocol' });
      f.client.dispose();
    }
  });

  it('requires a final result after full progress and cannot be kept alive by repeated completion messages', async () => {
    const f = await start();
    f.progress(64);
    await vi.advanceTimersByTimeAsync(89_000);
    f.progress(64);
    await vi.advanceTimersByTimeAsync(1_000);
    await expect(f.rejected).resolves.toMatchObject({ code: 'timeout' });
    expect(f.onSearchProgress).toHaveBeenCalledTimes(1);
    expect(f.worker.terminated).toBe(true);
  });

  it('terminates promptly on cancellation after progress and discards late events', async () => {
    const controller = new AbortController();
    const f = await start({ signal: controller.signal, search: { simulations: 64, maxConsidered: 8 } });
    await vi.advanceTimersByTimeAsync(80_000);
    f.progress(1);
    controller.abort();
    await expect(f.rejected).resolves.toMatchObject({ code: 'cancelled' });
    f.progress(2);
    expect(f.onSearchProgress).toHaveBeenCalledTimes(1);
    expect(f.worker.terminated).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('local worker construction lifecycle', () => {
  const readyInfo = { modelVersion: 'champion-browser', bytes: 128, backend: 'wasm', cached: false } as const;

  it('prepares without choosing a move and keeps progress nonterminal', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const status = vi.fn();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker, status);
    const result = client.prepare();
    const resolved = vi.fn();
    void result.then(resolved);
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    expect(worker.messages).toEqual([{ type: 'prepare', taskId: 'prepare-1' }]);
    const progress = { phase: 'downloading', loadedBytes: 64, totalBytes: 128, modelVersion: 'champion-browser', cached: false };
    worker.emit({ type: 'progress', taskId: 'prepare-1', progress });
    await Promise.resolve();
    expect(status).toHaveBeenLastCalledWith(progress);
    expect(resolved).not.toHaveBeenCalled();
    worker.emit({ type: 'prepared', taskId: 'prepare-1', info: readyInfo });
    await expect(result).resolves.toEqual(readyInfo);
    expect(status).toHaveBeenLastCalledWith({ phase: 'ready', info: readyInfo });
    client.dispose();
  });

  it('cancels preparation, ignores late progress, and permits retry in a fresh worker', async () => {
    vi.stubGlobal('Worker', class {});
    const workers = [new FakeWorker(), new FakeWorker()];
    let index = 0;
    const status = vi.fn();
    const client = new LocalDeltrelAiClient(() => workers[index++] as unknown as Worker, status);
    const controller = new AbortController();
    const first = client.prepare({ signal: controller.signal });
    const cancelled = expect(first).rejects.toMatchObject({ code: 'cancelled' });
    workers[0].emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    controller.abort();
    await cancelled;
    expect(workers[0].terminated).toBe(true);
    expect(status).toHaveBeenLastCalledWith({ phase: 'idle' });
    workers[0].emit({ type: 'prepared', taskId: 'prepare-1', info: readyInfo });
    expect(status).toHaveBeenLastCalledWith({ phase: 'idle' });
    const retry = client.prepare();
    workers[1].emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    workers[1].emit({ type: 'prepared', taskId: 'prepare-2', info: { ...readyInfo, cached: true } });
    await expect(retry).resolves.toMatchObject({ cached: true });
    client.dispose();
  });

  it('does not consume a playing request when its model becomes prepared', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'playing-while-loading');
    const result = client.request(request);
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    worker.emit({ type: 'prepared', taskId: request.requestId, info: readyInfo });
    worker.emit({ type: 'result', taskId: request.requestId, decision: decisionFor(request) });
    await expect(result).resolves.toEqual(decisionFor(request));
    client.dispose();
  });

  it('rejects preparation errors and timeouts without leaving its timer or worker pending', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const status = vi.fn();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker, status);
    const first = client.prepare();
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    worker.emit({ type: 'error', taskId: 'prepare-1', error: { code: 'network', message: 'Disconnected', retryable: true } });
    await expect(first).rejects.toMatchObject({ code: 'network' });
    expect(status).toHaveBeenLastCalledWith({ phase: 'error', message: 'Disconnected', retryable: true });
    const retry = client.prepare({ timeoutMs: 50 });
    const rejection = expect(retry).rejects.toMatchObject({ code: 'timeout', retryable: true });
    await vi.advanceTimersByTimeAsync(50);
    await rejection;
    expect(worker.terminated).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('waits for the worker handshake before choosing and disposes cleanly', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-handshake');

    const result = client.request(request);
    expect(worker.messages).toEqual([]);
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    expect(worker.messages).toEqual([
      { type: 'choose', taskId: request.requestId, request, search: null },
    ]);

    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: decisionFor(request),
    });
    await expect(result).resolves.toEqual(decisionFor(request));
    client.dispose();
    expect(worker.terminated).toBe(true);
  });

  it('terminates a blocked worker on cancellation and recreates it', async () => {
    vi.stubGlobal('Worker', class {});
    const workers = [new FakeWorker(), new FakeWorker()];
    let nextWorker = 0;
    const client = new LocalDeltrelAiClient(
      () => workers[nextWorker++] as unknown as Worker,
    );
    const abort = new AbortController();
    const firstRequest = buildAiRequest(config, [], 'local-cancel');
    const first = client.request(firstRequest, { signal: abort.signal });
    const active = workers[0];
    active.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    abort.abort();
    await expect(first).rejects.toMatchObject({ code: 'cancelled' });
    expect(active.terminated).toBe(true);

    const secondRequest = buildAiRequest(config, [], 'local-recreated');
    const second = client.request(secondRequest);
    const replacement = workers[1];
    expect(replacement).not.toBe(active);
    replacement.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    replacement.emit({
      type: 'result',
      taskId: secondRequest.requestId,
      decision: decisionFor(secondRequest),
    });
    await expect(second).resolves.toEqual(decisionFor(secondRequest));
    client.dispose();
  });

  it('terminates a worker when local inference times out', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-timeout');
    const result = client.request(request, { timeoutMs: 100 });
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();

    const rejection = expect(result).rejects.toMatchObject({
      code: 'timeout',
      retryable: true,
    });
    await vi.advanceTimersByTimeAsync(100);
    await rejection;
    expect(worker.terminated).toBe(true);
  });

  it('sends validated per-request budgets and returns compact worker analysis', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-budget');
    const search = { simulations: 4_096, maxConsidered: 256 };
    const result = client.request(request, { search });
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    expect(worker.messages).toEqual([
      { type: 'choose', taskId: request.requestId, request, search },
    ]);
    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: decisionFor(request, search),
    });
    await expect(result).resolves.toMatchObject({
      analysis: {
        simulations: 4_096,
        maxConsidered: 256,
        modelIdentity: 'browser-test',
      },
    });
    await expect(
      client.request(buildAiRequest(config, [], 'invalid-budget'), {
        search: { simulations: 536_870_912, maxConsidered: 8 },
      }),
    ).rejects.toMatchObject({ code: 'protocol' });
    client.dispose();
  });

  it('rejects a worker decision with stale nested state identity', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalDeltrelAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-stale');
    const result = client.request(request);
    worker.emit({ type: 'ready', protocolVersion: 5 });
    await Promise.resolve();
    const stale = decisionFor(request);
    stale.response.stateHash = 'zobrist64:0000000000000000';
    stale.analysis.stateHash = 'zobrist64:0000000000000000';
    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: stale,
    });
    await expect(result).rejects.toMatchObject({ code: 'stale' });
    client.dispose();
  });
});
