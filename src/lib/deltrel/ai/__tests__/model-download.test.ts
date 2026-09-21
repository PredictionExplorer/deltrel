import { createHash } from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DeltrelBrowserModelManifest } from '../manifest';
import type { LocalAiProgress } from '../local-ai-status';
import { downloadBrowserModel, MODEL_CACHE_NAME, modelSha256 } from '../model-download';

const modelBytes = Uint8Array.from([0, 97, 115, 109, 1, 2, 3, 4]);
const digest = `sha256:${createHash('sha256').update(modelBytes).digest('hex')}`;
const manifest = {
  modelVersion: 'browser-test-v1',
  model: { url: '/models/deltrel/browser-test-v1.onnx', bytes: modelBytes.length, sha256: digest },
} as DeltrelBrowserModelManifest;
const cacheKey = `${manifest.model.url}?sha256=${digest.slice(7)}`;

function responseFromChunks(chunks: Uint8Array[]): Response {
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  }));
}

function setup(initial?: Response) {
  const entries = new Map<string, Response>();
  if (initial) entries.set(cacheKey, initial);
  const cache = {
    match: vi.fn(async (key: string) => entries.get(key)?.clone()),
    delete: vi.fn(async (key: string) => entries.delete(key)),
    put: vi.fn(async (key: string, response: Response) => { entries.set(key, response.clone()); }),
  };
  const open = vi.fn(async () => cache);
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal('caches', { open });
  vi.stubGlobal('fetch', fetch);
  const abort = new AbortController();
  const progress: LocalAiProgress[] = [];
  const download = () => downloadBrowserModel(manifest, abort.signal, (value) => progress.push(value));
  return { cache, open, fetch, abort, progress, download, entries };
}

afterEach(() => vi.unstubAllGlobals());

describe('verified browser model downloads', () => {
  it('computes the standard SHA-256 tag from model bytes', async () => {
    expect(await modelSha256(modelBytes.buffer)).toBe(digest);
    expect(await modelSha256(new TextEncoder().encode('abc').buffer)).toBe(
      'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
  });

  it('streams byte progress and caches only the complete verified model', async () => {
    const f = setup();
    const response = responseFromChunks([modelBytes.slice(0, 2), modelBytes.slice(2, 5), modelBytes.slice(5)]);
    f.fetch.mockResolvedValue(response);
    const result = await f.download();
    expect(new Uint8Array(result.bytes)).toEqual(modelBytes);
    expect(result.cached).toBe(false);
    expect(f.open).toHaveBeenCalledWith(MODEL_CACHE_NAME);
    expect(f.fetch).toHaveBeenCalledWith(manifest.model.url, { cache: 'no-cache', signal: f.abort.signal });
    expect(f.progress.map(({ phase, loadedBytes }) => [phase, loadedBytes])).toEqual([
      ['downloading', 0], ['downloading', 2], ['downloading', 5], ['downloading', 8], ['verifying', 8],
    ]);
    expect(f.progress.every(p => p.totalBytes === 8 && p.modelVersion === manifest.modelVersion && !p.cached)).toBe(true);
    expect(f.cache.put).toHaveBeenCalledOnce();
    expect(f.cache.put.mock.calls[0][0]).toBe(cacheKey);
    expect(new Uint8Array(await f.entries.get(cacheKey)!.arrayBuffer())).toEqual(modelBytes);
    expect(response.body?.locked).toBe(false);
  });

  it('revalidates cached bytes and skips the network for a verified cache hit', async () => {
    const f = setup(new Response(modelBytes));
    const result = await f.download();
    expect(result.cached).toBe(true);
    expect(new Uint8Array(result.bytes)).toEqual(modelBytes);
    expect(f.fetch).not.toHaveBeenCalled();
    expect(f.cache.put).not.toHaveBeenCalled();
    expect(f.cache.delete).not.toHaveBeenCalled();
    expect(f.progress.map(p => [p.phase, p.loadedBytes, p.cached])).toEqual([
      ['verifying', 0, true], ['verifying', 8, true],
    ]);
  });

  it.each([
    ['wrong hash', Uint8Array.from([9, 9, 9, 9, 9, 9, 9, 9])],
    ['truncated', modelBytes.slice(0, 4)],
    ['oversize', new Uint8Array(12)],
  ])('repairs a %s cached model with a verified network response', async (_, corrupt) => {
    const f = setup(new Response(corrupt));
    f.fetch.mockResolvedValue(new Response(modelBytes));
    const result = await f.download();
    expect(result.cached).toBe(false);
    expect(new Uint8Array(result.bytes)).toEqual(modelBytes);
    expect(f.cache.delete).toHaveBeenCalledWith(cacheKey);
    expect(f.fetch).toHaveBeenCalledOnce();
    expect(f.cache.put).toHaveBeenCalledOnce();
    const second = await f.download();
    expect(second.cached).toBe(true);
    expect(f.fetch).toHaveBeenCalledOnce();
  });

  it.each([
    ['wrong hash', new Uint8Array(8), /integrity check/, 'protocol'],
    ['oversize', new Uint8Array(9), /exceeds its verified size/, 'protocol'],
    ['truncated', modelBytes.slice(0, 7), /interrupted/, 'network'],
  ])('rejects a %s network response without poisoning the cache', async (_, data, message, code) => {
    const f = setup();
    const response = responseFromChunks([data]);
    f.fetch.mockResolvedValue(response);
    await expect(f.download()).rejects.toMatchObject({ code, message: expect.stringMatching(message) });
    expect(f.cache.put).not.toHaveBeenCalled();
    expect(f.entries.size).toBe(0);
    expect(response.body?.locked).toBe(false);
  });

  it.each(['absent', 'denied', 'lookup', 'quota'] as const)('still plays when cache storage is %s', async failure => {
    const f = setup();
    if (failure === 'absent') vi.stubGlobal('caches', undefined);
    if (failure === 'denied') f.open.mockRejectedValue(new DOMException('Denied', 'SecurityError'));
    if (failure === 'lookup') f.cache.match.mockRejectedValue(new DOMException('Denied', 'SecurityError'));
    if (failure === 'quota') f.cache.put.mockRejectedValue(new DOMException('Full', 'QuotaExceededError'));
    f.fetch.mockResolvedValue(new Response(modelBytes));
    expect(new Uint8Array((await f.download()).bytes)).toEqual(modelBytes);
  });

  it('recovers even when removing a corrupt cache entry is denied', async () => {
    const f = setup(new Response(new Uint8Array(8)));
    f.cache.delete.mockRejectedValue(new DOMException('Denied', 'SecurityError'));
    f.fetch.mockResolvedValue(new Response(modelBytes));
    expect(new Uint8Array((await f.download()).bytes)).toEqual(modelBytes);
  });

  it.each(['transport', 'http', 'empty'] as const)('reports a recoverable %s failure', async failure => {
    const f = setup();
    if (failure === 'transport') f.fetch.mockRejectedValue(new TypeError('offline'));
    if (failure === 'http') f.fetch.mockResolvedValue(new Response('', { status: 503 }));
    if (failure === 'empty') f.fetch.mockResolvedValue(new Response(null));
    await expect(f.download()).rejects.toMatchObject({ code: 'network', retryable: true });
    expect(f.cache.put).not.toHaveBeenCalled();
  });

  it('does not fetch or publish progress for an already-cancelled download', async () => {
    const f = setup();
    f.abort.abort();
    await expect(f.download()).rejects.toMatchObject({ name: 'AbortError' });
    expect(f.fetch).not.toHaveBeenCalled();
    expect(f.progress).toEqual([]);
  });

  it('cancels between streamed chunks without storing partial bytes', async () => {
    const f = setup();
    const response = responseFromChunks([modelBytes.slice(0, 4), modelBytes.slice(4)]);
    f.fetch.mockResolvedValue(response);
    await expect(downloadBrowserModel(manifest, f.abort.signal, progress => {
      if (progress.phase === 'downloading' && progress.loadedBytes > 0) f.abort.abort();
    })).rejects.toMatchObject({ name: 'AbortError' });
    expect(f.cache.put).not.toHaveBeenCalled();
    expect(response.body?.locked).toBe(false);
  });

  it('does not retry a cancelled cache read over the network', async () => {
    const f = setup(new Response(modelBytes));
    await expect(downloadBrowserModel(manifest, f.abort.signal, progress => {
      if (progress.phase === 'verifying') f.abort.abort();
    })).rejects.toMatchObject({ name: 'AbortError' });
    expect(f.fetch).not.toHaveBeenCalled();
    expect(f.cache.put).not.toHaveBeenCalled();
  });
});
