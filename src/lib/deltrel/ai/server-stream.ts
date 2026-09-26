import { DeltrelAiError } from './errors';

export const SERVER_SEARCH_STREAM_CONTENT_TYPE = 'application/x-ndjson';
export const MAX_SERVER_SEARCH_STREAM_BYTES = 4 * 1024 * 1024;
export const MAX_SERVER_SEARCH_EVENT_BYTES = 1024 * 1024;
const MAX_SERVER_SEARCH_EVENTS = 20_000;

export type ServerSearchEvent =
  | { type: 'progress'; completed_simulations: number; total_simulations: number }
  | { type: 'result'; result: Record<string, unknown> }
  | { type: 'error'; error: Record<string, unknown> & { code: string; message: string } };

export function isServerSearchStream(contentType: string | null): boolean {
  return contentType?.split(';', 1)[0].trim().toLowerCase() === SERVER_SEARCH_STREAM_CONTENT_TYPE;
}

export function acceptsServerSearchStream(accept: string | null): boolean {
  return accept?.split(',').some((item) => isServerSearchStream(item.trim())) ?? false;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function invalid(message = 'Cloud AI returned an invalid progress stream.'): never {
  throw new DeltrelAiError('protocol', message);
}

/** Parse incrementally, retaining at most one bounded event and checking the whole stream. */
export async function* readServerSearchEvents(
  response: Response,
  expectedSimulations: number | undefined,
  signal?: AbortSignal,
): AsyncGenerator<ServerSearchEvent> {
  const declared = response.headers.get('Content-Length');
  if (declared !== null && (!Number.isSafeInteger(Number(declared)) ||
      Number(declared) < 0 || Number(declared) > MAX_SERVER_SEARCH_STREAM_BYTES)) {
    void response.body?.cancel().catch(() => {});
    invalid('Cloud AI progress stream exceeds its size limit.');
  }
  if (!response.body) invalid('Cloud AI returned an empty progress stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  const encoder = new TextEncoder();
  let pending = '';
  let bytes = 0;
  let events = 0;
  let completed = -1;
  let total = expectedSimulations;
  let terminal = false;
  let ended = false;
  const abort = () => { void reader.cancel(signal?.reason).catch(() => {}); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    while (true) {
      signal?.throwIfAborted();
      const chunk = await reader.read();
      signal?.throwIfAborted();
      if (chunk.done) {
        ended = true;
        try { pending += decoder.decode(); } catch { invalid(); }
        if (pending.length !== 0 || !terminal) invalid('Cloud AI progress stream ended before a complete result.');
        return;
      }
      bytes += chunk.value.byteLength;
      if (bytes > MAX_SERVER_SEARCH_STREAM_BYTES) invalid('Cloud AI progress stream exceeds its size limit.');
      try { pending += decoder.decode(chunk.value, { stream: true }); } catch { invalid(); }
      let newline: number;
      while ((newline = pending.indexOf('\n')) !== -1) {
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        if (encoder.encode(line).byteLength > MAX_SERVER_SEARCH_EVENT_BYTES || ++events > MAX_SERVER_SEARCH_EVENTS) {
          invalid('Cloud AI progress stream exceeds its size limit.');
        }
        if (terminal) invalid('Cloud AI returned data after its final result.');
        let event: unknown;
        try { event = JSON.parse(line); } catch { invalid(); }
        if (!record(event)) invalid();
        if (event.type === 'progress') {
          const next = event.completed_simulations;
          const budget = event.total_simulations;
          if (Object.keys(event).length !== 3 || typeof next !== 'number' || !Number.isSafeInteger(next) ||
              typeof budget !== 'number' || !Number.isSafeInteger(budget) || budget <= 0 ||
              next < 0 || next > budget || next < completed || (total !== undefined && budget !== total)) invalid();
          total = budget;
          completed = next;
          yield { type: 'progress', completed_simulations: next, total_simulations: budget };
        } else if (event.type === 'result' && Object.keys(event).length === 2 && record(event.result)) {
          terminal = true;
          yield { type: 'result', result: event.result };
        } else if (event.type === 'error' && Object.keys(event).length === 2 && record(event.error) &&
            typeof event.error.code === 'string' && event.error.code.length > 0 && event.error.code.length <= 128 &&
            typeof event.error.message === 'string' && event.error.message.length <= 2048) {
          terminal = true;
          yield { type: 'error', error: { ...event.error, code: event.error.code, message: event.error.message } };
        } else invalid();
      }
      if (encoder.encode(pending).byteLength > MAX_SERVER_SEARCH_EVENT_BYTES) {
        invalid('Cloud AI progress stream exceeds its size limit.');
      }
    }
  } finally {
    signal?.removeEventListener('abort', abort);
    if (!ended) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
