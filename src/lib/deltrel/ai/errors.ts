export type DeltrelAiErrorCode =
  | 'unavailable'
  | 'timeout'
  | 'network'
  | 'protocol'
  | 'stale'
  | 'illegal'
  | 'cancelled'
  | 'internal';

export class DeltrelAiError extends Error {
  readonly code: DeltrelAiErrorCode;
  readonly retryable: boolean;

  constructor(code: DeltrelAiErrorCode, message: string, retryable = false, cause?: unknown) {
    super(message, { cause });
    this.name = 'DeltrelAiError';
    this.code = code;
    this.retryable = retryable;
  }
}

export function asDeltrelAiError(error: unknown): DeltrelAiError {
  if (error instanceof DeltrelAiError) return error;
  if (
    typeof DOMException !== 'undefined' &&
    error instanceof DOMException &&
    error.name === 'AbortError'
  ) {
    return new DeltrelAiError('cancelled', 'AI request cancelled.');
  }
  return new DeltrelAiError(
    'internal',
    error instanceof Error ? error.message : 'AI request failed.',
    true,
    error,
  );
}
