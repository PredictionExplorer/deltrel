import { DeltrelAiError } from './errors';
import type { DeltrelBrowserModelManifest } from './manifest';
import type { LocalAiProgress } from './local-ai-status';

export const MODEL_CACHE_NAME = 'deltrel-models-v1';
type ProgressListener = (progress: LocalAiProgress) => void;

export async function modelSha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return `sha256:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

async function optionalCache(): Promise<Cache | null> {
  try { return typeof caches === 'undefined' ? null : await caches.open(MODEL_CACHE_NAME); }
  catch { return null; } // Private browsing and storage quotas must not prevent play.
}

async function readBoundedModel(
  response: Response, expectedBytes: number, signal: AbortSignal,
  onBytes: (loaded: number) => void,
): Promise<ArrayBuffer> {
  const data = new Uint8Array(expectedBytes);
  let offset = 0;
  if (!response.body) throw new DeltrelAiError('network', 'The model download was empty.', true);
  const reader = response.body.getReader();
  const cancelRead = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener('abort', cancelRead, { once: true });
  try {
    for (;;) {
      signal.throwIfAborted();
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      if (offset + value.byteLength > expectedBytes) {
        throw new DeltrelAiError('protocol', 'The model download exceeds its verified size.');
      }
      data.set(value, offset);
      offset += value.byteLength;
      onBytes(offset);
    }
    if (offset !== expectedBytes) throw new DeltrelAiError('network', 'The model download was interrupted. Please retry.', true);
    return data.buffer;
  } catch (error) {
    signal.throwIfAborted();
    if (error instanceof DeltrelAiError) throw error;
    throw new DeltrelAiError('network', 'The model download was interrupted. Please retry.', true, error);
  } finally {
    signal.removeEventListener('abort', cancelRead);
    // Teed cache responses may wait for their sibling; cleanup must not block retry.
    cancelRead();
    reader.releaseLock();
  }
}

/** Persist only complete, verified bytes. A broken or denied cache is recoverable. */
export async function downloadBrowserModel(
  manifest: DeltrelBrowserModelManifest, signal: AbortSignal, onProgress: ProgressListener,
): Promise<{ bytes: ArrayBuffer; cached: boolean }> {
  const { model, modelVersion } = manifest;
  const cache = await optionalCache();
  // A content digest in the key prevents a reused model URL from selecting old weights.
  const cacheKey = `${model.url}?sha256=${model.sha256.slice(7)}`;
  const report = (phase: LocalAiProgress['phase'], loadedBytes: number, cached = false) => {
    onProgress({ phase, loadedBytes, totalBytes: model.bytes, modelVersion, cached });
  };
  signal.throwIfAborted();
  let stored: Response | undefined;
  try { stored = await cache?.match(cacheKey); } catch { /* Storage is optional. */ }
  if (stored) {
    try {
      report('verifying', 0, true);
      const bytes = await readBoundedModel(stored, model.bytes, signal, () => {});
      if (await modelSha256(bytes) === model.sha256) {
        signal.throwIfAborted();
        report('verifying', model.bytes, true);
        return { bytes, cached: true };
      }
    } catch { signal.throwIfAborted(); }
    await cache?.delete(cacheKey).catch(() => false);
  }
  report('downloading', 0);
  let response: Response;
  try { response = await fetch(model.url, { cache: 'no-cache', signal }); }
  catch (error) {
    signal.throwIfAborted();
    throw new DeltrelAiError('network', 'The AI model could not be downloaded. Check your connection and retry.', true, error);
  }
  if (!response.ok) {
    throw new DeltrelAiError('network', `The AI model download failed (HTTP ${response.status}). Please retry.`, true);
  }
  const bytes = await readBoundedModel(response, model.bytes, signal, (loaded) => report('downloading', loaded));
  report('verifying', model.bytes);
  if (await modelSha256(bytes) !== model.sha256) {
    throw new DeltrelAiError('protocol', 'The downloaded model failed its integrity check. Please retry.', true);
  }
  signal.throwIfAborted();
  try { await cache?.put(cacheKey, new Response(bytes)); } catch { /* Play from memory if storage is full. */ }
  return { bytes, cached: false };
}
