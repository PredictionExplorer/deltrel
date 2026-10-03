import { DeltrelAiError } from './errors';
import { modelSha256 } from './model-download';
import { DELTREL_RUNTIME_ARTIFACTS } from './runtime-channel';

/** Read only one of the two build-pinned runtime artifacts, within its exact size. */
export async function downloadRuntimeArtifact(
  kind: keyof typeof DELTREL_RUNTIME_ARTIFACTS,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  if (kind !== 'module' && kind !== 'binary') {
    throw new DeltrelAiError('protocol', 'Unknown browser engine artifact.');
  }
  const artifact = DELTREL_RUNTIME_ARTIFACTS[kind];
  signal.throwIfAborted();
  const response = await fetch(artifact.url, { cache: 'force-cache', signal });
  if (!response.ok || !response.body) {
    throw new DeltrelAiError('unavailable', 'The browser engine could not be downloaded.', true);
  }
  const bytes = new Uint8Array(artifact.bytes);
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener('abort', cancel, { once: true });
  let offset = 0;
  try {
    for (;;) {
      signal.throwIfAborted();
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      if (offset + value.byteLength > bytes.byteLength) {
        throw new DeltrelAiError('protocol', 'The browser engine exceeds its verified size.');
      }
      bytes.set(value, offset);
      offset += value.byteLength;
    }
    if (offset !== bytes.byteLength || await modelSha256(bytes.buffer) !== artifact.sha256) {
      throw new DeltrelAiError('protocol', 'The browser engine failed its integrity check.');
    }
    signal.throwIfAborted();
    return bytes.buffer;
  } finally {
    signal.removeEventListener('abort', cancel);
    cancel();
    reader.releaseLock();
  }
}
