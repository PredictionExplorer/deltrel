import { webcrypto, createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { downloadRuntimeArtifact } from '../runtime-download';
import { DELTREL_RUNTIME_ARTIFACTS, DELTREL_RUNTIME_CHANNEL_PATH, DELTREL_RUNTIME_ID } from '../runtime-channel';
import { DELTREL_BROWSER_MODEL_MANIFEST_PATH, parseDeltrelBrowserModelManifest } from '../manifest';
import oldRelease from '../../../../../public/models/deltrel/manifest.json';

afterEach(() => vi.unstubAllGlobals());

describe('fixed browser execution channel', () => {
  it('uses a runtime-bound manifest and accepts no executable URL overrides', () => {
    expect(DELTREL_BROWSER_MODEL_MANIFEST_PATH).toBe(`/models/deltrel/manifest-runtime-${DELTREL_RUNTIME_ID}.json`);
    expect(DELTREL_RUNTIME_CHANNEL_PATH).not.toBe('/models/deltrel/manifest.json');
    expect(() => parseDeltrelBrowserModelManifest({ ...oldRelease, wasm: { moduleUrl: 'https://other.test/code.js' } })).toThrow();
    const parsed = parseDeltrelBrowserModelManifest(oldRelease);
    expect(parsed.wasm.moduleUrl).toBe(DELTREL_RUNTIME_ARTIFACTS.module.url);
    expect(parsed.wasm.binaryUrl).toBe(DELTREL_RUNTIME_ARTIFACTS.binary.url);
  });

  it.each(['module', 'binary'] as const)('verifies every %s byte before returning it', async kind => {
    vi.stubGlobal('crypto', webcrypto);
    const artifact = DELTREL_RUNTIME_ARTIFACTS[kind];
    const bytes = new Uint8Array(readFileSync(resolve('public', artifact.url.slice(1))));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(artifact.sha256.slice(7));
    const fetch = vi.fn().mockResolvedValue(new Response(bytes));
    vi.stubGlobal('fetch', fetch);
    const signal = new AbortController().signal;
    expect(new Uint8Array(await downloadRuntimeArtifact(kind, signal))).toEqual(bytes);
    expect(fetch).toHaveBeenCalledWith(artifact.url, { cache: 'force-cache', signal });
  });

  it.each(['module', 'binary'] as const)('rejects same-size tampered %s', async kind => {
    vi.stubGlobal('crypto', webcrypto);
    const artifact = DELTREL_RUNTIME_ARTIFACTS[kind];
    const bytes = new Uint8Array(readFileSync(resolve('public', artifact.url.slice(1))));
    bytes[0] ^= 1;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(bytes)));
    await expect(downloadRuntimeArtifact(kind, new AbortController().signal)).rejects.toMatchObject({ code: 'protocol' });
  });

  it.each([-1, 1])('rejects runtime size drift of %i bytes', async difference => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new Uint8Array(DELTREL_RUNTIME_ARTIFACTS.module.bytes + difference))));
    await expect(downloadRuntimeArtifact('module', new AbortController().signal)).rejects.toMatchObject({ code: 'protocol' });
  });

  it.each([404, 204])('rejects unavailable/empty responses (%i)', async status => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status })));
    await expect(downloadRuntimeArtifact('binary', new AbortController().signal)).rejects.toMatchObject({ code: 'unavailable' });
  });

  it('rejects unknown names and already-aborted work without fetching', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const abort = new AbortController();
    await expect(downloadRuntimeArtifact('__proto__' as never, abort.signal)).rejects.toMatchObject({ code: 'protocol' });
    abort.abort();
    await expect(downloadRuntimeArtifact('module', abort.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('cancels an in-flight bounded reader on abort', async () => {
    const abort = new AbortController();
    let readStarted!: () => void;
    const started = new Promise<void>(resolve => { readStarted = resolve; });
    const cancel = vi.fn();
    const stream = new ReadableStream({ pull() { readStarted(); }, cancel });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(stream)));
    const pending = downloadRuntimeArtifact('binary', abort.signal);
    const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    await started;
    abort.abort();
    await rejected;
    expect(cancel).toHaveBeenCalled();
  });
});
