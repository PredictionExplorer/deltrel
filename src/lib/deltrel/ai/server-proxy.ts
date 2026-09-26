import {
  acceptsServerSearchStream,
  isServerSearchStream,
  readServerSearchEvents,
  SERVER_SEARCH_STREAM_CONTENT_TYPE,
  type ServerSearchEvent,
} from './server-stream';

export const DELTREL_AI_PROXY_MOVE_PATH = '/v2/move' as const;
export const DELTREL_AI_PROXY_ANALYZE_PATH = '/v2/analyze' as const;
export const DELTREL_AI_PROXY_HEALTH_PATH = '/v2/health' as const;
export const DELTREL_AI_PROXY_REQUEST_BYTES = 64 * 1024;
export const DELTREL_AI_PROXY_RESPONSE_BYTES = 1024 * 1024;
// The cloud profile has a 180-second total admission/search budget. Allow transport
// headroom so its structured result or timeout arrives before our deadline.
export const DELTREL_AI_PROXY_MOVE_TIMEOUT_MS = 190_000;
export const DELTREL_AI_PROXY_HEALTH_TIMEOUT_MS = 5_000;
const REQUEST_BODY_TIMEOUT_MS = 5_000;

const REQUEST_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export interface DeltrelAiProxyConfig {
  serverUrl: string | undefined;
  bearerToken?: string;
  moveTimeoutMs?: number;
  healthTimeoutMs?: number;
}

class BodyLimitError extends Error {}

function noStoreHeaders(requestId: string, contentType = 'application/json'): HeadersInit {
  return {
    'Cache-Control': 'no-store, max-age=0',
    'Content-Type': `${contentType}; charset=utf-8`,
    Pragma: 'no-cache',
    'X-Content-Type-Options': 'nosniff',
    'X-Request-ID': requestId,
  };
}

function publicStreamError(requestId: string, code: string): ServerSearchEvent {
  const timedOut = code === 'deltrel_ai_timeout' || code === 'analysis_timeout' || code === 'timeout';
  const cancelled = code === 'deltrel_ai_cancelled' || code === 'client_disconnected';
  const invalid = code === 'invalid_upstream_response';
  const busy = code === 'service_busy' || code === 'deltrel_ai_busy';
  return {
    type: 'error',
    error: {
      code: timedOut ? 'deltrel_ai_timeout' : cancelled ? 'deltrel_ai_cancelled'
        : invalid ? 'invalid_upstream_response' : busy ? 'deltrel_ai_busy' : 'deltrel_ai_unavailable',
      message: timedOut ? 'Cloud AI timed out.' : cancelled ? 'AI request cancelled.'
        : invalid ? 'Cloud AI returned an invalid response.' : busy ? 'Cloud AI is busy. Try again shortly.' : 'Cloud AI is unavailable.',
      retryable: !cancelled,
      request_id: requestId,
    },
  };
}

function streamUpstreamResponse(
  upstream: Response,
  requestId: string,
  simulations: number | undefined,
  controller: AbortController,
  abortCode: () => string,
  cleanup: () => void,
): Response {
  const events = readServerSearchEvents(upstream, simulations, controller.signal);
  const encoder = new TextEncoder();
  let ended = false;
  let abort: () => void;
  const finish = () => {
    controller.signal.removeEventListener('abort', abort);
    cleanup();
  };
  const fail = (output: ReadableStreamDefaultController<Uint8Array>, code: string) => {
    if (ended) return;
    ended = true;
    // Never expose upstream exceptions, auth failures, or private service details.
    output.enqueue(encoder.encode(`${JSON.stringify(publicStreamError(requestId, code))}\n`));
    output.close();
    controller.abort();
    void events.return(undefined).catch(() => {});
    if (upstream.body && !upstream.body.locked) void upstream.body.cancel().catch(() => {});
    finish();
  };
  const body = new ReadableStream<Uint8Array>({
    start(output) {
      abort = () => fail(output, abortCode());
      controller.signal.addEventListener('abort', abort, { once: true });
      if (controller.signal.aborted) abort();
    },
    async pull(output) {
      try {
        const next = await events.next();
        if (ended) return;
        if (next.done) {
          ended = true;
          output.close();
          finish();
          return;
        }
        const event = next.value.type === 'error'
          ? publicStreamError(requestId, next.value.error.code) : next.value;
        output.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      } catch {
        fail(output, controller.signal.aborted ? abortCode() : 'invalid_upstream_response');
      }
    },
    cancel(reason) {
      if (ended) return;
      ended = true;
      controller.abort(reason);
      void events.return(undefined).catch(() => {});
      if (upstream.body && !upstream.body.locked) void upstream.body.cancel(reason).catch(() => {});
      finish();
    },
  });
  return new Response(body, {
    headers: { ...noStoreHeaders(requestId, SERVER_SEARCH_STREAM_CONTENT_TYPE), 'X-Accel-Buffering': 'no' },
  });
}

function proxyError(
  requestId: string,
  status: number,
  code: string,
  message: string,
  retryable: boolean,
  retryAfter?: string,
): Response {
  return Response.json(
    {
      error: {
        code,
        message,
        retryable,
        request_id: requestId,
      },
    },
    { status, headers: { ...noStoreHeaders(requestId), ...(retryAfter ? { 'Retry-After': retryAfter } : {}) } },
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function retryAfterSeconds(response: Response): string | undefined {
  const value = response.headers.get('Retry-After');
  if (!value) return undefined;
  const seconds = /^\d+$/.test(value) ? Number(value) : Math.ceil((Date.parse(value) - Date.now()) / 1000);
  return Number.isFinite(seconds) && seconds >= 0 ? String(Math.min(seconds, 300)) : undefined;
}

function publicHealth(payload: unknown): unknown {
  if (!isRecord(payload)) return payload;
  // Model diagnostics can contain local checkpoint paths and reload exceptions.
  // Only capability metadata belongs on the public health endpoint.
  const keys = ['status', 'service_version', 'api_schema_version', 'network_output_schema_version',
    'server_config_schema_version', 'model_schema_version', 'device', 'rules', 'features',
    'actions', 'search', 'capacity', 'variants', 'outcomes'] as const;
  const result: Record<string, unknown> = {};
  for (const key of keys) if (key in payload) result[key] = payload[key];
  const model = payload.model;
  if (isRecord(model)) {
    result.model = Object.fromEntries(['ready', 'role', 'model_version', 'model_step', 'model_identity']
      .filter((key) => key in model).map((key) => [key, model[key]]));
  }
  return result;
}

function upstreamError(requestId: string, response: Response, payload?: unknown): Response {
  const error = isRecord(payload) && isRecord(payload.error) ? payload.error : undefined;
  const code = error?.code;
  const busy = response.status === 429 || code === 'service_busy' || code === 'deltrel_ai_busy';
  const timeout = response.status === 504 || code === 'analysis_timeout' || code === 'deltrel_ai_timeout';
  const retryAfter = retryAfterSeconds(response);
  if (busy || timeout || response.status >= 500 || response.status === 401 || response.status === 403) {
    return proxyError(requestId, busy && response.status === 429 ? 429 : 503,
      busy ? 'deltrel_ai_busy' : timeout ? 'deltrel_ai_timeout' : 'deltrel_ai_unavailable',
      busy ? 'Cloud AI is busy. Try again shortly.' : timeout ? 'Cloud AI timed out.' : 'Cloud AI is unavailable.',
      true, retryAfter);
  }
  // Preserve only the precise safe signal used to negotiate older services.
  // Arbitrary validation messages/details may include private implementation data.
  const details = error?.details;
  if ((response.status === 400 || response.status === 422) && code === 'invalid_request' &&
      Array.isArray(details) && details.length === 1 && isRecord(details[0]) &&
      details[0].type === 'extra_forbidden' && Array.isArray(details[0].location) &&
      details[0].location.length === 2 && details[0].location[0] === 'body' && details[0].location[1] === 'include_network_output') {
    return Response.json({ error: { code: 'invalid_request', message: 'Cloud AI does not support network diagnostics.',
      retryable: false, request_id: requestId,
      details: [{ type: 'extra_forbidden', location: ['body', 'include_network_output'] }],
    } }, { status: response.status, headers: noStoreHeaders(requestId) });
  }
  return proxyError(requestId, response.status >= 400 ? response.status : 502,
    'invalid_request', 'Cloud AI could not accept this request.', false);
}

function isCrossOriginBrowserRequest(request: Request): boolean {
  const site = request.headers.get('Sec-Fetch-Site');
  if (site === 'cross-site' || site === 'same-site') return true;
  const origin = request.headers.get('Origin');
  if (origin === null) return false;
  const url = new URL(request.url);
  // Next may normalize Request.url to its internal listener hostname. Host is
  // the browser's actual authority and must be preserved by the reverse proxy.
  // Only the scheme comes from X-Forwarded-Proto (overwritten by trusted proxies);
  // X-Forwarded-Host must never widen the set of accepted browser origins.
  const host = request.headers.get('Host') ?? url.host;
  const forwardedProtocol = request.headers.get('X-Forwarded-Proto');
  const protocol = forwardedProtocol === null ? url.protocol : `${forwardedProtocol}:`;
  if ((protocol !== 'http:' && protocol !== 'https:') || /[\s,\\/?#@]/.test(host)) return true;
  try {
    const external = new URL(`${protocol}//${host}`);
    return external.origin !== origin;
  } catch {
    return true;
  }
}

function requestIdFor(request: Request): string {
  const supplied = request.headers.get('X-Request-ID');
  if (supplied && REQUEST_ID.test(supplied)) return supplied;
  return crypto.randomUUID().replaceAll('-', '');
}

function isJsonContentType(value: string | null): boolean {
  return value?.split(';', 1)[0].trim().toLowerCase() === 'application/json';
}

function endpointBase(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '');
  for (const suffix of [
    '/v2/move',
    '/v2/analyze',
    '/v2/health',
    '/healthz',
    '/v2',
  ]) {
    if (normalized.endsWith(suffix)) return normalized.slice(0, -suffix.length);
  }
  return normalized;
}

export function resolveDeltrelAiUpstreamUrl(serverUrl: string, endpoint: string): string {
  let target: URL;
  try {
    target = new URL(serverUrl);
  } catch (error) {
    throw new Error('DELTREL_AI_SERVER_URL must be an absolute URL', { cause: error });
  }
  if (
    (target.protocol !== 'http:' && target.protocol !== 'https:') ||
    target.username ||
    target.password ||
    target.search ||
    target.hash
  ) {
    throw new Error('DELTREL_AI_SERVER_URL must be an HTTP(S) URL without credentials');
  }
  if (
    /(?:^|\/)v\d+(?:\/|$)/.test(target.pathname) &&
    !/(?:^|\/)v2(?:\/|$)/.test(target.pathname)
  ) {
    throw new Error('DELTREL_AI_SERVER_URL must use the v2 API');
  }
  target.pathname = `${endpointBase(target.pathname)}${endpoint}`;
  return target.toString();
}

async function readLimitedBody(
  body: ReadableStream<Uint8Array> | null,
  contentLength: string | null,
  maximum: number,
  signal: AbortSignal,
): Promise<Uint8Array> {
  signal.throwIfAborted();
  if (contentLength) {
    const declared = Number(contentLength);
    if (!Number.isSafeInteger(declared) || declared < 0 || declared > maximum) {
      await body?.cancel();
      throw new BodyLimitError('body exceeds limit');
    }
  }
  if (!body) return new Uint8Array();

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  const abort = () => { void reader.cancel(signal.reason).catch(() => {}); };
  signal.addEventListener('abort', abort, { once: true });
  try {
    while (true) {
      signal.throwIfAborted();
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      total += value.byteLength;
      if (total > maximum) {
        await reader.cancel();
        throw new BodyLimitError('body exceeds limit');
      }
      chunks.push(value);
    }
  } finally {
    signal.removeEventListener('abort', abort);
    reader.releaseLock();
  }

  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

export async function proxyDeltrelAiRequest(
  request: Request,
  endpoint:
    | typeof DELTREL_AI_PROXY_MOVE_PATH
    | typeof DELTREL_AI_PROXY_ANALYZE_PATH
    | typeof DELTREL_AI_PROXY_HEALTH_PATH,
  config: DeltrelAiProxyConfig = {
    serverUrl: process.env.DELTREL_AI_SERVER_URL,
    bearerToken: process.env.DELTREL_AI_BEARER_TOKEN,
  },
): Promise<Response> {
  const requestId = requestIdFor(request);
  if (request.signal.aborted) {
    return proxyError(requestId, 499, 'deltrel_ai_cancelled', 'AI request cancelled.', false);
  }
  // Browser origin checks prevent cross-site pages from spending search capacity.
  // This is not authentication: scripts and API clients can omit these headers.
  if (endpoint !== DELTREL_AI_PROXY_HEALTH_PATH && isCrossOriginBrowserRequest(request)) {
    return proxyError(requestId, 403, 'cross_origin_request', 'Cloud AI requests must come from this site.', false);
  }
  if (!config.serverUrl?.trim()) {
    return proxyError(
      requestId,
      503,
      'deltrel_ai_unavailable',
      'Cloud AI is not configured.',
      false,
    );
  }

  let target: string;
  try {
    target = resolveDeltrelAiUpstreamUrl(config.serverUrl, endpoint);
  } catch {
    return proxyError(
      requestId,
      503,
      'deltrel_ai_unavailable',
      'Cloud AI configuration is invalid.',
      false,
    );
  }

  let body: Uint8Array | undefined;
  let simulations: number | undefined;
  if (endpoint !== DELTREL_AI_PROXY_HEALTH_PATH) {
    if (!isJsonContentType(request.headers.get('Content-Type'))) {
      return proxyError(
        requestId,
        415,
        'invalid_content_type',
        'Expected application/json.',
        false,
      );
    }
    const bodyDeadline = AbortSignal.timeout(REQUEST_BODY_TIMEOUT_MS);
    try {
      body = await readLimitedBody(
        request.body,
        request.headers.get('Content-Length'),
        DELTREL_AI_PROXY_REQUEST_BYTES,
        AbortSignal.any([request.signal, bodyDeadline]),
      );
      const parsed = JSON.parse(new TextDecoder().decode(body));
      if (typeof parsed?.search?.simulations === 'number') simulations = parsed.search.simulations;
    } catch (error) {
      if (request.signal.aborted) {
        return proxyError(requestId, 499, 'deltrel_ai_cancelled', 'AI request cancelled.', false);
      }
      if (bodyDeadline.aborted) {
        return proxyError(requestId, 408, 'request_timeout', 'Request body timed out.', true);
      }
      return proxyError(
        requestId,
        error instanceof BodyLimitError ? 413 : 400,
        error instanceof BodyLimitError ? 'request_too_large' : 'invalid_json',
        error instanceof BodyLimitError ? 'Request body is too large.' : 'Request body is invalid JSON.',
        false,
      );
    }
  }

  const timeoutMs =
    endpoint === DELTREL_AI_PROXY_HEALTH_PATH
      ? (config.healthTimeoutMs ?? DELTREL_AI_PROXY_HEALTH_TIMEOUT_MS)
      : (config.moveTimeoutMs ?? DELTREL_AI_PROXY_MOVE_TIMEOUT_MS);
  const controller = new AbortController();
  let timedOut = false;
  let streaming = false;
  const wantsProgress = endpoint !== DELTREL_AI_PROXY_HEALTH_PATH &&
    acceptsServerSearchStream(request.headers.get('Accept'));
  const abortFromClient = () => controller.abort(request.signal.reason);
  request.signal.addEventListener('abort', abortFromClient, { once: true });
  if (request.signal.aborted) abortFromClient();
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const cleanup = () => {
    clearTimeout(timeout);
    request.signal.removeEventListener('abort', abortFromClient);
  };

  try {
    controller.signal.throwIfAborted();
    const headers = new Headers({
      Accept: wantsProgress ? `${SERVER_SEARCH_STREAM_CONTENT_TYPE}, application/json` : 'application/json',
      'X-Request-ID': requestId,
    });
    if (body) headers.set('Content-Type', 'application/json');
    if (config.bearerToken) headers.set('Authorization', `Bearer ${config.bearerToken}`);
    const upstreamBody = body ? Uint8Array.from(body).buffer : undefined;

    const upstream = await fetch(target, {
      method: endpoint === DELTREL_AI_PROXY_HEALTH_PATH ? 'GET' : 'POST',
      body: upstreamBody,
      headers,
      cache: 'no-store',
      redirect: 'error',
      signal: controller.signal,
    });
    const upstreamRequestId = upstream.headers.get('X-Request-ID');
    const responseRequestId =
      upstreamRequestId && REQUEST_ID.test(upstreamRequestId) ? upstreamRequestId : requestId;
    if (wantsProgress && upstream.ok && isServerSearchStream(upstream.headers.get('Content-Type'))) {
      // Ownership of the deadline and cancellation listener lasts through EOF,
      // including time after the route handler has returned the first bytes.
      streaming = true;
      return streamUpstreamResponse(
        upstream, responseRequestId, simulations, controller,
        () => request.signal.aborted ? 'deltrel_ai_cancelled'
          : timedOut ? 'deltrel_ai_timeout' : 'deltrel_ai_unavailable',
        cleanup,
      );
    }
    if (!isJsonContentType(upstream.headers.get('Content-Type'))) {
      await upstream.body?.cancel();
      if (!upstream.ok) return upstreamError(requestId, upstream);
      return proxyError(
        requestId,
        502,
        'invalid_upstream_response',
        'Cloud AI returned an invalid response.',
        true,
      );
    }

    let payload: unknown;
    try {
      const responseBody = await readLimitedBody(
        upstream.body,
        upstream.headers.get('Content-Length'),
        DELTREL_AI_PROXY_RESPONSE_BYTES,
        controller.signal,
      );
      payload = JSON.parse(new TextDecoder().decode(responseBody));
    } catch (error) {
      if (controller.signal.aborted) throw error;
      if (!upstream.ok) return upstreamError(requestId, upstream);
      return proxyError(
        requestId,
        502,
        'invalid_upstream_response',
        'Cloud AI returned an invalid response.',
        true,
      );
    }

    if (!upstream.ok) return upstreamError(requestId, upstream, payload);
    return Response.json(endpoint === DELTREL_AI_PROXY_HEALTH_PATH ? publicHealth(payload) : payload, {
      status: upstream.status,
      headers: noStoreHeaders(responseRequestId),
    });
  } catch {
    if (request.signal.aborted) {
      return proxyError(requestId, 499, 'deltrel_ai_cancelled', 'AI request cancelled.', false);
    }
    return proxyError(
      requestId,
      503,
      timedOut ? 'deltrel_ai_timeout' : 'deltrel_ai_unavailable',
      timedOut ? 'Cloud AI timed out.' : 'Cloud AI is unavailable.',
      true,
    );
  } finally {
    if (!streaming) cleanup();
  }
}
