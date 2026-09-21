/** Download progress describes model bytes; initialization is a separate stage. */
export interface LocalAiProgress {
  phase: 'checking' | 'downloading' | 'verifying' | 'initializing';
  loadedBytes: number;
  totalBytes: number | null;
  modelVersion: string | null;
  cached: boolean;
}

export interface LocalAiReadyInfo {
  modelVersion: string;
  bytes: number;
  backend: 'webgpu' | 'wasm';
  cached: boolean;
}

export type LocalAiStatus =
  | { phase: 'idle' }
  | LocalAiProgress
  | { phase: 'ready'; info: LocalAiReadyInfo }
  | { phase: 'error'; message: string; retryable: boolean };

const idle: LocalAiStatus = { phase: 'idle' };
let status: LocalAiStatus = idle;
const listeners = new Set<() => void>();

export const getLocalAiStatus = (): LocalAiStatus => status;
export const getServerLocalAiStatus = (): LocalAiStatus => idle;
export function subscribeLocalAiStatus(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
export function publishLocalAiStatus(next: LocalAiStatus): void {
  status = next;
  for (const listener of listeners) listener();
}
