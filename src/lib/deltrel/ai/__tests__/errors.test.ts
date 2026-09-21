import { describe, expect, it } from 'vitest';
import { DeltrelAiError, asDeltrelAiError } from '../errors';

describe('AI error normalization', () => {
  it('preserves typed errors without wrapping them', () => {
    const error = new DeltrelAiError('network', 'offline', true);
    expect(asDeltrelAiError(error)).toBe(error);
  });

  it('maps AbortError to a non-retryable cancellation', () => {
    const normalized = asDeltrelAiError(new DOMException('aborted', 'AbortError'));
    expect(normalized).toMatchObject({
      name: 'DeltrelAiError',
      code: 'cancelled',
      retryable: false,
      message: 'AI request cancelled.',
    });
  });

  it('retains Error messages and unknown causes for diagnostics', () => {
    const cause = new TypeError('bad tensor');
    const normalized = asDeltrelAiError(cause);
    expect(normalized).toMatchObject({
      code: 'internal',
      retryable: true,
      message: 'bad tensor',
      cause,
    });

    const unknown = { detail: 'opaque' };
    const fallback = asDeltrelAiError(unknown);
    expect(fallback.message).toBe('AI request failed.');
    expect(fallback.cause).toBe(unknown);
  });
});

