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
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('streaming same-origin AI proxy', () => {
  const progress = { type: 'progress', completed_simulations: 2, total_simulations: 4 };
  const result = { type: 'result', result: { request_id: 'stream-request', action: 0 } };
  const encode = (event: unknown) => new TextEncoder().encode(`${JSON.stringify(event)}\n`);
  const decode = (bytes: Uint8Array | undefined) => JSON.parse(new TextDecoder().decode(bytes));
  const request = (signal?: AbortSignal) => new Request('https://public.example/v2/move', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson, application/json', 'X-Request-ID': 'stream-request' },
    body: JSON.stringify({ search: { simulations: 4 } }),
  });
  const streaming = (body: ReadableStream<Uint8Array> | string) => new Response(body, {
    headers: { 'Content-Type': 'application/x-ndjson', Authorization: 'Bearer private-token' },
  });

  it('forwards progress before the upstream result exists and keeps auth private', async () => {
    vi.useFakeTimers();
    let output!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({ start(controller) { output = controller; } });
    vi.stubGlobal('fetch', vi.fn(async (_url: unknown, init: RequestInit) => {
      expect(new Headers(init.headers).get('Accept')).toContain('application/x-ndjson');
      expect(new Headers(init.headers).get('Authorization')).toBe('Bearer private-token');
      return streaming(body);
    }));
    const response = await proxyDeltrelAiRequest(request(), DELTREL_AI_PROXY_MOVE_PATH, {
      serverUrl: 'https://private.example', bearerToken: 'private-token',
    });
    expect(response.headers.get('Content-Type')).toContain('application/x-ndjson');
    expect(response.headers.get('Cache-Control')).toContain('no-store');
    expect(response.headers.has('Authorization')).toBe(false);
    const reader = response.body!.getReader();
    output.enqueue(encode(progress));
    expect(decode((await reader.read()).value)).toEqual(progress);
    output.enqueue(encode(result));
    output.close();
    expect(decode((await reader.read()).value)).toEqual(result);
    expect((await reader.read()).done).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('allows JSON fallback without a second upstream request', async () => {
    const fetchMock = vi.fn(async () => Response.json(result.result));
    vi.stubGlobal('fetch', fetchMock);
    const response = await proxyDeltrelAiRequest(request(), DELTREL_AI_PROXY_MOVE_PATH, { serverUrl: 'https://private.example' });
    expect(response.headers.get('Content-Type')).toContain('application/json');
    await expect(response.json()).resolves.toEqual(result.result);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it.each(['timeout', 'cancel'] as const)('preserves %s after returning the stream headers', async (reason) => {
    vi.useFakeTimers();
    const caller = new AbortController();
    const cancel = vi.fn();
    let upstreamSignal!: AbortSignal;
    vi.stubGlobal('fetch', vi.fn(async (_url: unknown, init: RequestInit) => {
      upstreamSignal = init.signal!;
      return streaming(new ReadableStream({
        start(controller) { controller.enqueue(encode(progress)); }, cancel,
      }));
    }));
    const response = await proxyDeltrelAiRequest(request(caller.signal), DELTREL_AI_PROXY_MOVE_PATH, {
      serverUrl: 'https://private.example', moveTimeoutMs: 10,
    });
    const reader = response.body!.getReader();
    expect(decode((await reader.read()).value)).toEqual(progress);
    if (reason === 'cancel') caller.abort();
    else await vi.advanceTimersByTimeAsync(10);
    expect(decode((await reader.read()).value)).toMatchObject({
      type: 'error', error: { code: reason === 'timeout' ? 'deltrel_ai_timeout' : 'deltrel_ai_cancelled' },
    });
    expect((await reader.read()).done).toBe(true);
    expect(upstreamSignal.aborted).toBe(true);
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('propagates downstream reader cancellation and clears the deadline', async () => {
    vi.useFakeTimers();
    const cancel = vi.fn();
    let upstreamSignal!: AbortSignal;
    vi.stubGlobal('fetch', vi.fn(async (_url: unknown, init: RequestInit) => {
      upstreamSignal = init.signal!;
      return streaming(new ReadableStream({
        start(controller) { controller.enqueue(encode(progress)); }, cancel,
      }));
    }));
    const response = await proxyDeltrelAiRequest(request(), DELTREL_AI_PROXY_MOVE_PATH, {
      serverUrl: 'https://private.example', moveTimeoutMs: 10,
    });
    const reader = response.body!.getReader();
    await reader.read();
    await reader.cancel();
    expect(upstreamSignal.aborted).toBe(true);
    expect(cancel).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each([
    ['invalid JSON', '{broken}\n'],
    ['truncated response', JSON.stringify(progress) + '\n'],
    ['wrong budget', JSON.stringify({ ...progress, total_simulations: 8 }) + '\n'],
    ['oversized event', 'x'.repeat(1024 * 1024 + 1)],
    ['private upstream error', JSON.stringify({ type: 'error', error: {
      code: 'internal_error', message: 'private-token at https://private.example', details: ['private-token'],
    } }) + '\n'],
  ])('ends %s safely without exposing private details', async (_name, body) => {
    vi.stubGlobal('fetch', vi.fn(async () => streaming(body)));
    const response = await proxyDeltrelAiRequest(request(), DELTREL_AI_PROXY_MOVE_PATH, {
      serverUrl: 'https://private.example', bearerToken: 'private-token',
    });
    const text = await response.text();
    expect(text).not.toContain('private-token');
    expect(text).not.toContain('https://private.example');
    const events = text.trim().split('\n').map((line) => JSON.parse(line));
    expect(events.at(-1)).toMatchObject({ type: 'error', error: { request_id: 'stream-request' } });
  });

  it.each([
    ['analysis_timeout', 'deltrel_ai_timeout', true],
    ['client_disconnected', 'deltrel_ai_cancelled', false],
    ['service_busy', 'deltrel_ai_unavailable', true],
  ])('maps backend %s to a safe public stream error', async (upstreamCode, code, retryable) => {
    vi.stubGlobal('fetch', vi.fn(async () => streaming(JSON.stringify({
      type: 'error', error: { code: upstreamCode, message: 'Private error details.' },
    }) + '\n')));
    const response = await proxyDeltrelAiRequest(request(), DELTREL_AI_PROXY_MOVE_PATH, {
      serverUrl: 'https://private.example',
    });
    const event = JSON.parse((await response.text()).trim());
    expect(event).toMatchObject({ type: 'error', error: { code, retryable, request_id: 'stream-request' } });
    expect(event.error.message).not.toContain('Private');
  });
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
