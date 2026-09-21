import { DeltrelAiError } from './errors';
import {
  parseDeltrelAiDecision,
  responseFromDeltrelAiDecision,
  type DeltrelAiDecision,
  type DeltrelAiSearchBudget,
} from './decision';
import type { DeltrelAiRequest, DeltrelAiResponse } from './protocol';
import {
  parseBrowserSearchBudget,
  parseWorkerEvent,
  type DeltrelAiWorkerCommand,
  type DeltrelAiWorkerEvent,
} from './worker-protocol';

interface PendingRequest {
  request: DeltrelAiRequest;
  resolve: (decision: DeltrelAiDecision) => void;
  reject: (error: DeltrelAiError) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  timeout?: ReturnType<typeof setTimeout>;
}

type WorkerFactory = () => Worker;

export interface LocalAiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  search?: DeltrelAiSearchBudget;
}

export const DEFAULT_LOCAL_AI_TIMEOUT_MS = 90_000;
export const LOCAL_AI_HANDSHAKE_TIMEOUT_MS = 5_000;
export const LOCAL_AI_IDLE_TIMEOUT_MS = 60_000;

function createLocalWorker(): Worker {
  return new Worker(new URL('../../../workers/deltrel-ai.worker.ts', import.meta.url), {
    type: 'module',
    name: 'deltrel-local-ai',
  });
}

export class LocalDeltrelAiClient {
  private worker: Worker | null = null;
  private readonly pending = new Map<string, PendingRequest>();
  private readyPromise: Promise<Worker> | null = null;
  private readyResolve: ((worker: Worker) => void) | null = null;
  private readyReject: ((error: DeltrelAiError) => void) | null = null;
  private handshakeTimeout: ReturnType<typeof setTimeout> | null = null;
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly workerFactory: WorkerFactory = createLocalWorker) {}

  request(
    request: DeltrelAiRequest,
    options: LocalAiRequestOptions = {},
  ): Promise<DeltrelAiDecision> {
    const { signal } = options;
    if (signal?.aborted) {
      return Promise.reject(new DeltrelAiError('cancelled', 'AI request cancelled.'));
    }
    if (this.pending.has(request.requestId)) {
      return Promise.reject(new DeltrelAiError('protocol', 'Duplicate local AI request id.'));
    }

    const timeoutMs = options.timeoutMs ?? DEFAULT_LOCAL_AI_TIMEOUT_MS;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      return Promise.reject(new DeltrelAiError('protocol', 'Local AI timeout is invalid.'));
    }
    let search: DeltrelAiSearchBudget | null = null;
    try {
      if (options.search !== undefined) {
        search = parseBrowserSearchBudget(options.search);
      }
    } catch (error) {
      return Promise.reject(
        error instanceof DeltrelAiError
          ? error
          : new DeltrelAiError('protocol', 'Local AI search budget is invalid.', false, error),
      );
    }

    return new Promise<DeltrelAiDecision>((resolve, reject) => {
      const pending: PendingRequest = {
        request,
        resolve,
        reject: (error) => reject(error),
        signal,
      };
      if (signal) {
        pending.abortListener = () => {
          if (!this.pending.has(request.requestId)) return;
          this.resetWorker(new DeltrelAiError('cancelled', 'AI request cancelled.'));
        };
        signal.addEventListener('abort', pending.abortListener, { once: true });
      }
      pending.timeout = setTimeout(() => {
        if (!this.pending.has(request.requestId)) return;
        this.resetWorker(new DeltrelAiError('timeout', 'Local AI timed out.', true));
      }, timeoutMs);
      this.pending.set(request.requestId, pending);
      this.clearIdleTimeout();
      let ready: Promise<Worker>;
      try {
        ready = this.ensureWorker();
      } catch (error) {
        this.resetWorker(
          error instanceof DeltrelAiError
            ? error
            : new DeltrelAiError(
                'unavailable',
                'Local AI worker is unavailable.',
                true,
                error,
              ),
        );
        return;
      }
      void ready
        .then((worker) => {
          if (!this.pending.has(request.requestId)) return;
          const command: DeltrelAiWorkerCommand = {
            type: 'choose',
            taskId: request.requestId,
            request,
            search,
          };
          worker.postMessage(command);
        })
        .catch((error) => {
          if (!this.pending.has(request.requestId)) return;
          this.resetWorker(
            error instanceof DeltrelAiError
              ? error
              : new DeltrelAiError(
                  'unavailable',
                  'Local AI worker is unavailable.',
                  true,
                  error,
                ),
          );
        });
    });
  }

  dispose(): void {
    this.resetWorker(new DeltrelAiError('cancelled', 'Local AI disposed.'));
  }

  private ensureWorker(): Promise<Worker> {
    if (this.worker && this.readyPromise) return this.readyPromise;
    if (typeof Worker === 'undefined') {
      throw new DeltrelAiError('unavailable', 'Local AI requires browser Web Worker support.');
    }
    const worker = this.workerFactory();
    worker.addEventListener('message', this.onMessage);
    worker.addEventListener('error', this.onWorkerError);
    this.worker = worker;
    this.readyPromise = new Promise<Worker>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.handshakeTimeout = setTimeout(() => {
      this.resetWorker(
        new DeltrelAiError('unavailable', 'Local AI worker failed to start.', true),
      );
    }, LOCAL_AI_HANDSHAKE_TIMEOUT_MS);
    return this.readyPromise;
  }

  private readonly onMessage = (event: MessageEvent<unknown>) => {
    let message: DeltrelAiWorkerEvent;
    try {
      message = parseWorkerEvent(event.data);
    } catch (error) {
      this.resetWorker(
        error instanceof DeltrelAiError
          ? error
          : new DeltrelAiError('protocol', 'Local AI worker protocol failed.', false, error),
      );
      return;
    }
    if (message.type === 'ready') {
      if (!this.worker || !this.readyResolve) return;
      if (this.handshakeTimeout) clearTimeout(this.handshakeTimeout);
      this.handshakeTimeout = null;
      const resolve = this.readyResolve;
      this.readyResolve = null;
      this.readyReject = null;
      resolve(this.worker);
      return;
    }
    const pending = this.pending.get(message.taskId);
    if (!pending) return;
    this.pending.delete(message.taskId);
    this.cleanPending(pending);

    if (message.type === 'error') {
      pending.reject(
        new DeltrelAiError(
          message.error.code,
          message.error.message,
          message.error.retryable,
        ),
      );
      this.scheduleIdleDisposal();
      return;
    }
    try {
      pending.resolve(parseDeltrelAiDecision(pending.request, message.decision));
    } catch (error) {
      pending.reject(
        error instanceof DeltrelAiError
          ? error
          : new DeltrelAiError('protocol', 'Local AI returned an invalid action.', false, error),
      );
    }
    this.scheduleIdleDisposal();
  };

  private readonly onWorkerError = (event: ErrorEvent) => {
    this.resetWorker(
      new DeltrelAiError(
        'internal',
        event.message || 'Local AI worker crashed.',
        true,
        event.error,
      ),
    );
  };

  private removeAbortListener(pending: PendingRequest): void {
    if (pending.signal && pending.abortListener) {
      pending.signal.removeEventListener('abort', pending.abortListener);
    }
  }

  private cleanPending(pending: PendingRequest): void {
    this.removeAbortListener(pending);
    if (pending.timeout) clearTimeout(pending.timeout);
  }

  private clearIdleTimeout(): void {
    if (this.idleTimeout) clearTimeout(this.idleTimeout);
    this.idleTimeout = null;
  }

  private scheduleIdleDisposal(): void {
    if (this.pending.size > 0 || !this.worker) return;
    this.clearIdleTimeout();
    this.idleTimeout = setTimeout(() => this.dispose(), LOCAL_AI_IDLE_TIMEOUT_MS);
  }

  private resetWorker(error: DeltrelAiError): void {
    this.clearIdleTimeout();
    if (this.handshakeTimeout) clearTimeout(this.handshakeTimeout);
    this.handshakeTimeout = null;
    this.readyReject?.(error);
    this.readyResolve = null;
    this.readyReject = null;
    this.readyPromise = null;
    for (const pending of this.pending.values()) {
      this.cleanPending(pending);
      pending.reject(error);
    }
    this.pending.clear();
    if (this.worker) {
      this.worker.removeEventListener('message', this.onMessage);
      this.worker.removeEventListener('error', this.onWorkerError);
      this.worker.terminate();
      this.worker = null;
    }
  }
}

const localClient = new LocalDeltrelAiClient();

export function requestLocalAiAction(
  request: DeltrelAiRequest,
  options: LocalAiRequestOptions = {},
): Promise<DeltrelAiResponse> {
  return requestLocalAiDecision(request, options).then(responseFromDeltrelAiDecision);
}

export function requestLocalAiDecision(
  request: DeltrelAiRequest,
  options: LocalAiRequestOptions = {},
): Promise<DeltrelAiDecision> {
  return localClient.request(request, options);
}

export function disposeLocalAiClient(): void {
  localClient.dispose();
}
