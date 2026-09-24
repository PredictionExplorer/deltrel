import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkLocalAiCapability,
  checkAiCapabilities,
  checkServerAiCapability,
  localBrowserCapabilityIssue,
} from '../capabilities';
import { DELTREL_FEATURE_SCHEMA_HASH } from '../protocol';
import publishedManifest from '../../../../../public/models/deltrel/manifest.json';
import * as localClient from '../local-client';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AI capability preflight', () => {
  it('uses the verified pinned release without network access after preparation', async () => {
    vi.stubGlobal('Worker', class {});
    vi.spyOn(localClient, 'getPinnedLocalAiRelease').mockReturnValue(publishedManifest);
    const fetchMock = vi.fn(() => Promise.reject(new Error('offline')));
    vi.stubGlobal('fetch', fetchMock);
    await expect(checkAiCapabilities()).resolves.toMatchObject({ local: {
      status: 'available', browserModel: { modelVersion: publishedManifest.model_version },
      search: { default: { simulations: 512, maxConsidered: 16 } },
    } });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('advertises native custom limits separately from the published search defaults and presets', async () => {
    vi.stubGlobal('Worker', class {});
    vi.stubGlobal('fetch', vi.fn(async (_url: string, options?: RequestInit) =>
      options?.method === 'HEAD' ? new Response(null) : Response.json(publishedManifest)));
    await expect(checkLocalAiCapability()).resolves.toMatchObject({
      status: 'available',
      search: {
        default: { simulations: 512, maxConsidered: 16 },
        maximum: { simulations: 536_870_911, maxConsidered: 4_294_967_295 },
        presets: { maximum: { simulations: 4096, maxConsidered: 64 } },
      },
    });
  });

  it('checks only the published browser release for public gameplay', async () => {
    vi.stubGlobal('Worker', class {});
    const fetchMock = vi.fn(async (url: string, options?: RequestInit) => {
      expect(url).not.toContain('/v2/');
      expect(options?.cache).toBe('no-store');
      return options?.method === 'HEAD' ? new Response(null) : Response.json(publishedManifest);
    });
    vi.stubGlobal('fetch', fetchMock);
    await expect(checkAiCapabilities()).resolves.toMatchObject({
      local: { status: 'available' }, server: { status: 'unavailable', code: 'browser_only' },
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('does not send an already-cancelled health check', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const signal = AbortSignal.abort();
    await expect(checkServerAiCapability(signal)).resolves.toMatchObject({ status: 'unavailable' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('stops an unbounded chunked capability response at the byte limit', async () => {
    const cancel = vi.fn();
    let chunks = 0;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new ReadableStream({
      pull(controller) {
        chunks++;
        controller.enqueue(new Uint8Array(64 * 1024));
      },
      cancel,
    }), { headers: { 'Content-Type': 'application/json' } })));
    await expect(checkServerAiCapability()).resolves.toMatchObject({ status: 'unavailable' });
    expect(cancel).toHaveBeenCalledOnce();
    expect(chunks).toBeLessThanOrEqual(6);
  });

  it.each(['-1', 'NaN', '262145'])('rejects an invalid declared response length %s without reading it', async (length) => {
    const cancel = vi.fn();
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new ReadableStream({ cancel }), {
      headers: { 'Content-Type': 'application/json', 'Content-Length': length },
    })));
    await expect(checkServerAiCapability()).resolves.toMatchObject({ status: 'unavailable' });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('validates the server health contract through the same-origin route', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('/v2/health');
      return Response.json({
        status: 'ok',
        service_version: '1.0.0',
        api_schema_version: 3,
        model: { ready: true, model_version: 'test', model_step: 1 },
        rules: {
          schema_id: 'deltrel.rules.v3',
          version: 3,
          hash: 'fnv1a64:46e4fbcff4e17fd3',
        },
        features: {
          schema_id: 'deltrel.model-features.external.v3',
          version: 4,
          hash: DELTREL_FEATURE_SCHEMA_HASH,
        },
        actions: {
          schema_id: 'deltrel.action-layout.nodes-only.v1',
        },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    await expect(checkServerAiCapability()).resolves.toEqual({
      status: 'available',
      label: 'Server AI',
    });
  });

  it('reports incompatible server health as permanently unavailable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 3,
          model: { ready: true },
          rules: {
            schema_id: 'deltrel.rules.v3',
            hash: 'fnv1a64:wrong',
          },
          features: {},
          actions: {},
        }),
      ),
    );
    await expect(checkServerAiCapability()).resolves.toMatchObject({
      status: 'unavailable',
      code: 'server_incompatible',
      retryable: false,
    });
  });

  it('preserves optional server device, champion, and actual search metadata', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 3,
          device: 'mps',
          model: {
            ready: true,
            model_version: 'champion-v7',
            model_step: 700,
            model_identity: `sha256-${'a'.repeat(64)}`,
            role: 'champion',
          },
          search: {
            defaults: { simulations: 512, max_considered: 16 },
            maximums: { simulations: 4_096, max_considered: 64 },
            presets: {
              quick: { simulations: 128, max_considered: 8 },
              strong: { simulations: 512, max_considered: 16 },
              maximum: { simulations: 4_096, max_considered: 64 },
            },
          },
          rules: {
            schema_id: 'deltrel.rules.v3',
            hash: 'fnv1a64:46e4fbcff4e17fd3',
          },
          features: {
            schema_id: 'deltrel.model-features.external.v3',
            version: 4,
            hash: DELTREL_FEATURE_SCHEMA_HASH,
          },
          actions: {
            schema_id: 'deltrel.action-layout.nodes-only.v1',
          },
        }),
      ),
    );
    await expect(checkServerAiCapability()).resolves.toEqual({
      status: 'available',
      label: 'Server AI',
      device: 'mps',
      champion: {
        role: 'champion',
        modelVersion: 'champion-v7',
        modelStep: 700,
        modelIdentity: `sha256-${'a'.repeat(64)}`,
      },
      search: {
        default: { simulations: 512, maxConsidered: 16 },
        maximum: { simulations: 4_096, maxConsidered: 64 },
        presets: {
          quick: { simulations: 128, maxConsidered: 8 },
          strong: { simulations: 512, maxConsidered: 16 },
          maximum: { simulations: 4_096, maxConsidered: 64 },
        },
      },
    });
  });

  it('rejects malformed optional server limits instead of clamping them', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 3,
          device: 'mps',
          model: { ready: true },
          search: {
            defaults: { simulations: 512, max_considered: 16 },
            maximums: { simulations: 128, max_considered: 8 },
            presets: {
              strong: { simulations: 512, max_considered: 16 },
            },
          },
          rules: {
            schema_id: 'deltrel.rules.v3',
            hash: 'fnv1a64:46e4fbcff4e17fd3',
          },
          features: {
            schema_id: 'deltrel.model-features.external.v3',
            version: 4,
            hash: DELTREL_FEATURE_SCHEMA_HASH,
          },
          actions: {
            schema_id: 'deltrel.action-layout.nodes-only.v1',
          },
        }),
      ),
    );
    await expect(checkServerAiCapability()).resolves.toMatchObject({
      status: 'unavailable',
      code: 'server_incompatible',
      retryable: false,
    });
  });

  it('guards local AI before loading BigInt-dependent modules', async () => {
    expect(localBrowserCapabilityIssue()).toMatch(/Web Workers/i);

    vi.stubGlobal('Worker', class {});
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('{}', { status: 404 })),
    );
    await expect(checkLocalAiCapability()).resolves.toMatchObject({
      status: 'unavailable',
      code: 'local_assets_missing',
      retryable: false,
    });
  });
});
