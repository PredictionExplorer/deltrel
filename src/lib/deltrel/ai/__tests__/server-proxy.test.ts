import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DELTREL_AI_PROXY_ANALYZE_PATH,
  DELTREL_AI_PROXY_HEALTH_PATH,
  DELTREL_AI_PROXY_MOVE_PATH,
  DELTREL_AI_PROXY_REQUEST_BYTES,
  proxyDeltrelAiRequest,
  resolveDeltrelAiUpstreamUrl,
} from '../server-proxy';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('same-origin deltrelserve proxy', () => {
  it('uses only the fixed private target and forwards identity/auth server-side', async () => {
    const upstreamPayload = { schema_version: 2, request_id: 'proxy-request' };
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe('https://private.example/base/v2/move');
      expect(init?.cache).toBe('no-store');
      expect(init?.redirect).toBe('error');
      const headers = new Headers(init?.headers);
      expect(headers.get('X-Request-ID')).toBe('proxy-request');
      expect(headers.get('Authorization')).toBe('Bearer private-token');
      return Response.json(upstreamPayload, {
        headers: { 'X-Request-ID': 'proxy-request' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/move?target=https://evil.example', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': 'proxy-request',
        },
        body: JSON.stringify({ schema_version: 2 }),
      }),
      DELTREL_AI_PROXY_MOVE_PATH,
      {
        serverUrl: 'https://private.example/base',
        bearerToken: 'private-token',
      },
    );

    expect(response.status).toBe(200);
    expect(response.headers.get('Cache-Control')).toContain('no-store');
    await expect(response.json()).resolves.toEqual(upstreamPayload);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('proxies health to the fixed v2 endpoint', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('https://private.example/v2/health');
      return Response.json({ status: 'ok' });
    });
    vi.stubGlobal('fetch', fetchMock);
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/health'),
      DELTREL_AI_PROXY_HEALTH_PATH,
      { serverUrl: 'https://private.example/v2/move' },
    );
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: 'ok' });
  });

  it('proxies analysis to its fixed v2 endpoint', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('https://private.example/v2/analyze');
      return Response.json({ schema_version: 2 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      }),
      DELTREL_AI_PROXY_ANALYZE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(response.status).toBe(200);
  });

  it('returns structured unavailable and strict input failures', async () => {
    const unavailable = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/health'),
      DELTREL_AI_PROXY_HEALTH_PATH,
      { serverUrl: undefined },
    );
    expect(unavailable.status).toBe(503);
    await expect(unavailable.json()).resolves.toMatchObject({
      error: { code: 'deltrel_ai_unavailable', retryable: false },
    });

    const wrongType = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/move', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: '{}',
      }),
      DELTREL_AI_PROXY_MOVE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(wrongType.status).toBe(415);

    const tooLarge = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': String(DELTREL_AI_PROXY_REQUEST_BYTES + 1),
        },
        body: '{}',
      }),
      DELTREL_AI_PROXY_MOVE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(tooLarge.status).toBe(413);
  });

  it('normalizes every documented upstream URL form with path prefixes', () => {
    const serverUrls = [
      ['base', 'https://private.example/prefix'],
      ['/v2', 'https://private.example/prefix/v2'],
      ['/v2/move', 'https://private.example/prefix/v2/move'],
      ['/v2/analyze', 'https://private.example/prefix/v2/analyze'],
      ['/v2/health', 'https://private.example/prefix/v2/health'],
      ['/healthz', 'https://private.example/prefix/healthz'],
    ] as const;

    for (const [label, serverUrl] of serverUrls) {
      expect(
        resolveDeltrelAiUpstreamUrl(serverUrl, '/v2/move'),
        `${label} -> move`,
      ).toBe('https://private.example/prefix/v2/move');
      expect(
        resolveDeltrelAiUpstreamUrl(serverUrl, '/v2/health'),
        `${label} -> health`,
      ).toBe('https://private.example/prefix/v2/health');
    }
  });

  it('rejects private endpoint credentials', () => {
    expect(() =>
      resolveDeltrelAiUpstreamUrl('https://user:secret@private.example', '/v2/move'),
    ).toThrow(/without credentials/i);
    expect(() =>
      resolveDeltrelAiUpstreamUrl('https://private.example/v1/move', '/v2/move'),
    ).toThrow(/v2 API/i);
  });

  it('does not dispatch a search after the caller has already cancelled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    controller.abort();
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/health', { signal: controller.signal }),
      DELTREL_AI_PROXY_HEALTH_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(response.status).toBe(499);
    expect(fetchMock).not.toHaveBeenCalled();
    await expect(response.json()).resolves.toMatchObject({
      error: { code: 'deltrel_ai_cancelled', retryable: false },
    });
  });

  it.each(['timeout', 'cancel'] as const)('preserves %s while reading upstream response bytes', async (reason) => {
    const caller = new AbortController();
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init: RequestInit) => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          init.signal!.addEventListener('abort', () => controller.error(init.signal!.reason), { once: true });
          controller.enqueue(new TextEncoder().encode('{'));
          if (reason === 'cancel') queueMicrotask(() => caller.abort());
        },
      });
      return new Response(stream, { headers: { 'Content-Type': 'application/json' } });
    }));
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/health', { signal: caller.signal }),
      DELTREL_AI_PROXY_HEALTH_PATH,
      { serverUrl: 'https://private.example', healthTimeoutMs: 10 },
    );
    expect(response.status).toBe(reason === 'timeout' ? 503 : 499);
    await expect(response.json()).resolves.toMatchObject({
      error: {
        code: reason === 'timeout' ? 'deltrel_ai_timeout' : 'deltrel_ai_cancelled',
        retryable: reason === 'timeout',
      },
    });
  });

  it('preserves cancellation while reading the incoming request body', async () => {
    const caller = new AbortController();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const init: RequestInit & { duplex: 'half' } = {
      method: 'POST', duplex: 'half', signal: caller.signal,
      headers: { 'Content-Type': 'application/json' },
      body: new ReadableStream({
        start(controller) {
          caller.signal.addEventListener('abort', () => controller.error(caller.signal.reason));
          queueMicrotask(() => caller.abort());
        },
      }),
    };
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/move', init),
      DELTREL_AI_PROXY_MOVE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(response.status).toBe(499);
    expect(fetchMock).not.toHaveBeenCalled();
    await expect(response.json()).resolves.toMatchObject({ error: { code: 'deltrel_ai_cancelled' } });
  });

  it.each([
    new Headers({ 'Content-Type': 'text/html' }),
    new Headers({ 'Content-Type': 'application/json', 'Content-Length': String(2 * 1024 * 1024) }),
  ])('cancels a rejected upstream body instead of leaving the connection open', async (headers) => {
    const cancel = vi.fn();
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new ReadableStream({ cancel }), { headers })));
    const response = await proxyDeltrelAiRequest(
      new Request('https://public.example/v2/health'),
      DELTREL_AI_PROXY_HEALTH_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(response.status).toBe(502);
    expect(cancel).toHaveBeenCalledOnce();
  });
});
