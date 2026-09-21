import { DeltrelAiError } from './errors';
import {
  parseDeltrelAiDecision,
  responseFromDeltrelAiDecision,
  type DeltrelAiDecision,
  type DeltrelAiSearchBudget,
} from './decision';
import type { DeltrelAiRequest, DeltrelAiResponse } from './protocol';
import { publishLocalAiStatus, type LocalAiReadyInfo, type LocalAiStatus } from './local-ai-status';
export { getLocalAiStatus, getServerLocalAiStatus, subscribeLocalAiStatus } from './local-ai-status';
export type { LocalAiProgress, LocalAiReadyInfo, LocalAiStatus } from './local-ai-status';
import {
  parseBrowserSearchBudget,
  parseWorkerEvent,
  type DeltrelAiWorkerCommand,
  type DeltrelAiWorkerEvent,
  type LocalAiSearchProgress,
} from './worker-protocol';
export type { LocalAiSearchProgress } from './worker-protocol';

interface PendingRequest {
  request: DeltrelAiRequest;
  resolve: (decision: DeltrelAiDecision) => void;
  reject: (error: DeltrelAiError) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  timeout?: ReturnType<typeof setTimeout>;
  timeoutMs: number;
  inactivityTimeout: boolean;
  completedSimulations: number;
  totalSimulations: number | null;
  requestedSearch: DeltrelAiSearchBudget | null;
  onSearchProgress?: (progress: LocalAiSearchProgress) => void;
}

type WorkerFactory = () => Worker;

interface PendingPreparation {
  resolve: (info: LocalAiReadyInfo) => void;
  reject: (error: DeltrelAiError) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  timeout?: ReturnType<typeof setTimeout>;
}
export interface LocalAiPreparationOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface LocalAiRequestOptions {
  signal?: AbortSignal;
  /** Explicit absolute deadline. By default, only 90 seconds without search progress times out. */
  timeoutMs?: number;
  search?: DeltrelAiSearchBudget;
  onSearchProgress?: (progress: LocalAiSearchProgress) => void;
}

export const DEFAULT_LOCAL_AI_TIMEOUT_MS = 90_000;
export const LOCAL_AI_HANDSHAKE_TIMEOUT_MS = 5_000;
export const LOCAL_AI_IDLE_TIMEOUT_MS = 60_000;
export const LOCAL_AI_PREPARATION_TIMEOUT_MS = 10 * 60_000;

function createLocalWorker(): Worker {
  return new Worker(new URL('../../../workers/deltrel-ai.worker.ts', import.meta.url), {
    type: 'module',
    name: 'deltrel-local-ai',
  });
}

export class LocalDeltrelAiClient {
  private worker: Worker | null = null;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly preparations = new Map<string, PendingPreparation>();
  private preparationSequence = 0;
  private readyPromise: Promise<Worker> | null = null;
  private readyResolve: ((worker: Worker) => void) | null = null;
  private readyReject: ((error: DeltrelAiError) => void) | null = null;
  private handshakeTimeout: ReturnType<typeof setTimeout> | null = null;
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly workerFactory: WorkerFactory = createLocalWorker,
    private readonly onStatus: (status: LocalAiStatus) => void = () => {},
  ) {}

  prepare(options: LocalAiPreparationOptions = {}): Promise<LocalAiReadyInfo> {
    const { signal, timeoutMs = LOCAL_AI_PREPARATION_TIMEOUT_MS } = options;
    if (signal?.aborted) return Promise.reject(new DeltrelAiError('cancelled', 'AI preparation cancelled.'));
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return Promise.reject(new DeltrelAiError('protocol', 'Local AI timeout is invalid.'));
    let taskId: string;
    do { taskId = `prepare-${++this.preparationSequence}`; } while (this.pending.has(taskId));
    return new Promise((resolve, reject) => {
      const pending: PendingPreparation = { resolve, reject, signal };
      pending.abortListener = () => {
        if (this.preparations.has(taskId)) this.resetWorker(new DeltrelAiError('cancelled', 'AI preparation cancelled.'));
      };
      signal?.addEventListener('abort', pending.abortListener, { once: true });
      pending.timeout = setTimeout(() => {
        if (this.preparations.has(taskId)) this.resetWorker(new DeltrelAiError('timeout', 'Preparing the AI took too long. Please retry.', true));
      }, timeoutMs);
      this.preparations.set(taskId, pending);
      this.clearIdleTimeout();
      this.onStatus({ phase: 'checking', loadedBytes: 0, totalBytes: null, modelVersion: null, cached: false });
      try {
        void this.ensureWorker().then((worker) => {
          if (this.preparations.has(taskId)) worker.postMessage({ type: 'prepare', taskId } satisfies DeltrelAiWorkerCommand);
        }).catch((error) => {
          if (this.preparations.has(taskId)) this.resetWorker(error instanceof DeltrelAiError ? error : new DeltrelAiError('unavailable', 'Browser AI could not start. Please retry.', true, error));
        });
      } catch (error) {
        this.resetWorker(error instanceof DeltrelAiError ? error : new DeltrelAiError('unavailable', 'Browser AI could not start. Please retry.', true, error));
      }
    });
  }

  request(
    request: DeltrelAiRequest,
    options: LocalAiRequestOptions = {},
  ): Promise<DeltrelAiDecision> {
    const { signal } = options;
    if (signal?.aborted) {
      return Promise.reject(new DeltrelAiError('cancelled', 'AI request cancelled.'));
    }
    if (this.pending.has(request.requestId) || this.preparations.has(request.requestId)) {
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
        timeoutMs,
        inactivityTimeout: options.timeoutMs === undefined,
        completedSimulations: 0,
        totalSimulations: search?.simulations ?? null,
        requestedSearch: search,
        onSearchProgress: options.onSearchProgress,
      };
      if (signal) {
        pending.abortListener = () => {
          if (!this.pending.has(request.requestId)) return;
          this.resetWorker(new DeltrelAiError('cancelled', 'AI request cancelled.'));
        };
        signal.addEventListener('abort', pending.abortListener, { once: true });
      }
      this.pending.set(request.requestId, pending);
      this.armRequestTimeout(pending);
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
    if (!this.pending.has(message.taskId) && !this.preparations.has(message.taskId)) return;
    if (message.type === 'search-progress') {
      const pending = this.pending.get(message.taskId);
      if (!pending || (pending.totalSimulations !== null &&
          pending.totalSimulations !== message.progress.totalSimulations)) {
        this.resetWorker(new DeltrelAiError('protocol', 'Local AI progress does not match its search budget.'));
        return;
      }
      // A duplicate, regressing, stale, or arbitrary heartbeat cannot keep a
      // stalled search alive. Strict protocol bounds also cap possible renewals.
      if (message.progress.completedSimulations <= pending.completedSimulations) return;
      pending.totalSimulations = message.progress.totalSimulations;
      pending.completedSimulations = message.progress.completedSimulations;
      if (pending.inactivityTimeout) this.armRequestTimeout(pending);
      pending.onSearchProgress?.({ ...message.progress });
      return;
    }
    if (message.type === 'progress') {
      this.onStatus(message.progress);
      return;
    }
    if (message.type === 'prepared') {
      this.onStatus({ phase: 'ready', info: message.info });
      const preparation = this.preparations.get(message.taskId);
      if (preparation) {
        this.preparations.delete(message.taskId);
        this.cleanPending(preparation);
        preparation.resolve(message.info);
        this.scheduleIdleDisposal();
      }
      return;
    }
    const preparation = this.preparations.get(message.taskId);
    if (preparation) {
      this.preparations.delete(message.taskId);
      this.cleanPending(preparation);
      const error = message.type === 'error'
        ? new DeltrelAiError(message.error.code, message.error.message, message.error.retryable)
        : new DeltrelAiError('protocol', 'Browser AI preparation returned an unexpected result.');
      this.onStatus({ phase: 'error', message: error.message, retryable: error.retryable });
      preparation.reject(error);
      this.scheduleIdleDisposal();
      return;
    }
    const pending = this.pending.get(message.taskId);
    if (!pending) return;
    this.pending.delete(message.taskId);
    this.cleanPending(pending);

    if (message.type === 'error') {
      this.onStatus({ phase: 'error', message: message.error.message, retryable: message.error.retryable });
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
      const decision = parseDeltrelAiDecision(pending.request, message.decision);
      if ((pending.totalSimulations !== null && decision.analysis.simulations !== pending.totalSimulations) ||
          (pending.requestedSearch && decision.analysis.maxConsidered !== pending.requestedSearch.maxConsidered)) {
        throw new DeltrelAiError('protocol', 'Local AI result does not match its search budget.');
      }
      pending.resolve(decision);
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

  private removeAbortListener(pending: PendingPreparation | PendingRequest): void {
    if (pending.signal && pending.abortListener) {
      pending.signal.removeEventListener('abort', pending.abortListener);
    }
  }

  private armRequestTimeout(pending: PendingRequest): void {
    if (pending.timeout) clearTimeout(pending.timeout);
    pending.timeout = setTimeout(() => {
      if (this.pending.get(pending.request.requestId) !== pending) return;
      this.resetWorker(new DeltrelAiError('timeout', pending.inactivityTimeout
        ? 'Local AI stopped making search progress. Please retry or use a lower strength.'
        : 'Local AI timed out.', true));
    }, pending.timeoutMs);
  }

  private cleanPending(pending: PendingPreparation | PendingRequest): void {
    this.removeAbortListener(pending);
    if (pending.timeout) clearTimeout(pending.timeout);
  }

  private clearIdleTimeout(): void {
    if (this.idleTimeout) clearTimeout(this.idleTimeout);
    this.idleTimeout = null;
  }

  private scheduleIdleDisposal(): void {
    if (this.pending.size > 0 || this.preparations.size > 0 || !this.worker) return;
    this.clearIdleTimeout();
    this.idleTimeout = setTimeout(() => this.dispose(), LOCAL_AI_IDLE_TIMEOUT_MS);
  }

  private resetWorker(error: DeltrelAiError): void {
    if (this.preparations.size > 0 || this.pending.size > 0) {
      this.onStatus(error.code === 'cancelled' ? { phase: 'idle' }
        : { phase: 'error', message: error.message, retryable: error.retryable });
    }
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
    for (const pending of this.preparations.values()) {
      this.cleanPending(pending);
      pending.reject(error);
    }
    this.preparations.clear();
    if (this.worker) {
      this.worker.removeEventListener('message', this.onMessage);
      this.worker.removeEventListener('error', this.onWorkerError);
      this.worker.terminate();
      this.worker = null;
    }
  }
}

const localClient = new LocalDeltrelAiClient(createLocalWorker, publishLocalAiStatus);

export function prepareLocalAi(options: LocalAiPreparationOptions = {}): Promise<LocalAiReadyInfo> {
  return localClient.prepare(options);
}

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
