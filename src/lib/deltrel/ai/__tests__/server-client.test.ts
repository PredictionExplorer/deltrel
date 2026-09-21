import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameConfig } from '../../game';
import { buildAiRequest, makeAiResponse } from '../protocol';
import {
  DEFAULT_SERVER_AI_MAX_CONSIDERED,
  DEFAULT_SERVER_AI_SIMULATIONS,
  configuredServerAiUrl,
  configuredServerHealthUrl,
  deterministicServerSeed,
  parseAnalyzeResponse,
  requestServerAiAction,
  requestServerAiDecision,
  resolveServerSearchBudget,
  resolveDeltrelAiHealthUrl,
  resolveDeltrelAiMoveUrl,
  toAnalyzeRequest,
} from '../server-client';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};
const request = buildAiRequest(config, [], 'deltrelserve-test');

function representativeAnalyzeResponse(
  target = request,
  variant: { mode: 'classic' | 'double'; handicap: number; pie: boolean } = {
    mode: 'double',
    handicap: 1,
    pie: false,
  },
  swapAvailable = false,
  swapRecommended = false,
) {
  const score = new Array<number>(303).fill(0);
  score[151] = 1;
  return {
    schema_version: 3,
    request_id: target.requestId,
    action: { code: 0, kind: 'place', node: 0 },
    root_actions: [
      { code: 0, kind: 'place', node: 0 },
      { code: 1, kind: 'place', node: 1 },
    ],
    root_policy: [0.75, 0.25],
    root_q: [0.2, -0.1],
    root_visits: [3, 1],
    outcome: { loss: 0.2, win: 0.8 },
    value: 0.6,
    search_value: 0.3,
    root_value: swapRecommended ? -0.4 : 0.25,
    variant,
    swap_available: swapAvailable,
    swap_recommended: swapRecommended,
    history_known: true,
    score_belief: {
      support_min: -151,
      support_max: 151,
      expected_margin: 0,
      probabilities: score,
    },
    model_version: 'fake-v2',
    model_step: 5,
    timing_ms: {
      queue: 0,
      model_reload: 0,
      inference_search: 1,
      total: 1,
    },
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('deltrelserve v3 adapter', () => {
  it('accepts optional final predictions and rejects inconsistent player identity or ranges', () => {
    const predictions = {
      perspective: 0,
      final_basis: 'official_end',
      final_counts: [
        { player: 0, shores: 12, networks: 2, corners: 3.5, corner_bonus_probability: 0.8 },
        { player: 1, shores: 8, networks: 3, corners: 1.5, corner_bonus_probability: 0.2 },
      ],
      opponent_reply: { player: 1, kind: 'place', node: 2, probability: 0.25 },
      second_stone: null,
    };
    const payload = { ...representativeAnalyzeResponse(), predictions };
    const decision = parseAnalyzeResponse(request, payload);
    expect(decision.analysis.predictions).toMatchObject({
      perspective: 0, finalBasis: 'official_end',
      finalCounts: [{ player: 0, shores: 12 }, { player: 1, shores: 8 }],
      opponentReply: { player: 1, node: 2 }, secondStone: null,
    });
    expect(parseAnalyzeResponse(request, { ...payload, predictions: null }).analysis.predictions).toBeNull();
    expect(parseAnalyzeResponse(request, representativeAnalyzeResponse()).analysis.predictions).toBeUndefined();
    const malformed = [
      { ...predictions, perspective: 1 },
      { ...predictions, final_counts: predictions.final_counts.slice(0, 1) },
      { ...predictions, final_counts: [{ ...predictions.final_counts[0], shores: 21 }, predictions.final_counts[1]] },
      { ...predictions, final_counts: [{ ...predictions.final_counts[0], networks: NaN }, predictions.final_counts[1]] },
      { ...predictions, opponent_reply: { ...predictions.opponent_reply, player: 0 } },
      { ...predictions, opponent_reply: { ...predictions.opponent_reply, node: 274 } },
      { ...predictions, second_stone: { player: 0, kind: 'place', node: 2, probability: 0.5 } },
      { ...predictions, unexpected: true },
    ];
    for (const invalid of malformed) {
      expect(() => parseAnalyzeResponse(request, { ...payload, predictions: invalid })).toThrow(/predictions/i);
    }
  });

  it('converts semantic state to strict snake_case with the variant and history', () => {
    const wire = toAnalyzeRequest(request, {
      simulations: 4,
      maxConsidered: 2,
    });
    expect(wire).toEqual({
      schema_version: 3,
      rules_hash: 'fnv1a64:46e4fbcff4e17fd3',
      rings: 4,
      stones: new Array(50).fill(-1),
      to_move: 0,
      moves_left: 1,
      opening: true,
      terminal: false,
      mode: 'double',
      handicap: 1,
      pie: false,
      swap_available: false,
      swapped: false,
      history: {
        current_turn: [],
        previous_turn: [],
        own_previous_turn: [],
        handicap_stones: [],
      },
      pda: 0,
      include_predictions: true,
      search: {
        simulations: 4,
        max_considered: 2,
        seed: deterministicServerSeed(request.stateHash),
      },
    });
    expect(Number.isSafeInteger(wire.search.seed)).toBe(true);
    expect(JSON.parse(JSON.stringify(wire))).toEqual(wire);

    const pieRequest = buildAiRequest(
      { ...config, pieRule: true },
      [{ type: 'place', node: 7 }],
      'deltrelserve-pie',
    );
    expect(toAnalyzeRequest(pieRequest, { simulations: 4, maxConsidered: 2 })).toMatchObject({
      pie: true,
      swap_available: true,
      to_move: 1,
      history: { previous_turn: [7], handicap_stones: [7] },
    });
  });

  it('dispatches a recommended swap and rejects variant drift', () => {
    const pieRequest = buildAiRequest(
      { ...config, pieRule: true },
      [{ type: 'place', node: 7 }],
      'deltrelserve-swap',
    );
    const pieVariant = { mode: 'double' as const, handicap: 1, pie: true };
    const swapped = parseAnalyzeResponse(
      pieRequest,
      representativeAnalyzeResponse(pieRequest, pieVariant, true, true),
      pieRequest.requestId,
      { simulations: 4, maxConsidered: 2 },
    );
    expect(swapped.response.action).toEqual({ type: 'swap' });
    expect(swapped.analysis).toMatchObject({ swapRecommended: true, rootValue: -0.4 });
    expect(() => parseAnalyzeResponse(pieRequest, {
      ...representativeAnalyzeResponse(pieRequest, pieVariant, true, true),
      predictions: {
        perspective: 1,
        final_basis: 'official_end',
        final_counts: [
          { player: 0, shores: 12, networks: 2, corners: 3, corner_bonus_probability: 0.8 },
          { player: 1, shores: 8, networks: 3, corners: 2, corner_bonus_probability: 0.2 },
        ],
        opponent_reply: { player: 0, kind: 'place', node: 2, probability: 0.2 },
        second_stone: { player: 1, kind: 'place', node: 3, probability: 0.3 },
      },
    })).toThrow(/second stone after a pie swap/i);
    const kept = parseAnalyzeResponse(
      pieRequest,
      representativeAnalyzeResponse(pieRequest, pieVariant, true, false),
      pieRequest.requestId,
      { simulations: 4, maxConsidered: 2 },
    );
    expect(kept.response.action).toEqual({ type: 'place', node: 0 });
    expect(kept.analysis.swapRecommended).toBe(false);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...representativeAnalyzeResponse(),
        variant: { mode: 'classic', handicap: 1, pie: false },
      }),
    ).toThrow(/variant metadata/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...representativeAnalyzeResponse(),
        swap_recommended: true,
      }),
    ).toThrow(/variant metadata/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...representativeAnalyzeResponse(),
        history_known: false,
      }),
    ).toThrow(/variant metadata/i);
    expect(() =>
      parseAnalyzeResponse(request, { ...representativeAnalyzeResponse(), schema_version: 2 }),
    ).toThrow(/schema is incompatible/i);
  });

  it('validates binary outcomes and maps a placement response', () => {
    const decision = parseAnalyzeResponse(
      request,
      representativeAnalyzeResponse(),
      request.requestId,
      { simulations: 4, maxConsidered: 2 },
    );
    expect(decision.response).toEqual(
      makeAiResponse(request, { type: 'place', node: 0 }),
    );
    expect(decision.analysis).toMatchObject({
      perspective: 0,
      stateHash: request.stateHash,
      outcome: { loss: 0.2, win: 0.8 },
      modelValue: 0.6,
      searchValue: 0.3,
      rootValue: 0.25,
      swapRecommended: false,
      expectedMargin: 0,
      rootActions: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
      rootPolicy: [0.75, 0.25],
      rootQ: [0.2, -0.1],
      rootVisits: [3, 1],
      modelVersion: 'fake-v2',
      modelStep: 5,
      simulations: 4,
      maxConsidered: 2,
      timingMs: { queue: 0, modelLoad: 0, inferenceSearch: 1, total: 1 },
    });
  });

  it('rejects removed action and outcome shapes', () => {
    expect(() =>
      parseAnalyzeResponse(request, {
        ...representativeAnalyzeResponse(),
        action: { code: -1, kind: 'pass', node: null },
      }),
    ).toThrow(/valid atomic action|disagree/i);

    const response = representativeAnalyzeResponse();
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        outcome: { ...response.outcome, draw: 0 },
      }),
    ).toThrow(/outcome belief/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        value: 0,
      }),
    ).toThrow(/value belief/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        root_q: [Number.NaN, -0.1],
      }),
    ).toThrow(/root Q/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        score_belief: {
          ...response.score_belief,
          expected_margin: 1,
        },
      }),
    ).toThrow(/expected score margin/i);
    expect(() =>
      parseAnalyzeResponse(
        request,
        response,
        request.requestId,
        { simulations: 5, maxConsidered: 2 },
      ),
    ).toThrow(/visits/i);
  });

  it('rejects inconsistent, illegal, or stale actions', () => {
    const inconsistent = {
      ...representativeAnalyzeResponse(),
      action: { code: 0, kind: 'place', node: 1 },
    };
    expect(() => parseAnalyzeResponse(request, inconsistent)).toThrow(
      /code, kind, and node disagree/i,
    );

    const illegalAction = { code: 50, kind: 'place', node: 50 };
    const illegal = {
      ...representativeAnalyzeResponse(),
      action: illegalAction,
      root_actions: [illegalAction],
      root_policy: [1],
      root_q: [0],
      root_visits: [1],
    };
    expect(() => parseAnalyzeResponse(request, illegal)).toThrow(
      /illegal action/i,
    );
    expect(() =>
      parseAnalyzeResponse(
        request,
        representativeAnalyzeResponse(),
        'different-request',
      ),
    ).toThrow(/identity/i);
  });

  it('bounds defaults and rejects malformed explicit budgets', () => {
    expect(
      resolveServerSearchBudget(
        {},
        { simulations: '128', maxConsidered: '8' },
      ),
    ).toEqual({ simulations: 128, maxConsidered: 8 });
    expect(
      resolveServerSearchBudget(
        {},
        { simulations: '999999', maxConsidered: 'invalid' },
      ),
    ).toEqual({
      simulations: DEFAULT_SERVER_AI_SIMULATIONS,
      maxConsidered: DEFAULT_SERVER_AI_MAX_CONSIDERED,
    });
    expect(() =>
      toAnalyzeRequest(request, { simulations: 0, maxConsidered: 8 }),
    ).toThrow(/simulations/i);
    expect(() =>
      toAnalyzeRequest(request, { simulations: 8, maxConsidered: 129 }),
    ).toThrow(/max-considered/i);
  });

  it('normalizes v2 move, analyze, base, and health URLs', () => {
    expect(resolveDeltrelAiMoveUrl('https://ai.example')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveDeltrelAiMoveUrl('https://ai.example/proxy/')).toBe(
      'https://ai.example/proxy/v2/move',
    );
    expect(resolveDeltrelAiMoveUrl('https://ai.example/v2/move')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveDeltrelAiMoveUrl('https://ai.example/v2/analyze')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveDeltrelAiMoveUrl('https://ai.example/v2/health')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveDeltrelAiMoveUrl('/deltrelserve')).toBe('/deltrelserve/v2/move');
    expect(resolveDeltrelAiHealthUrl('https://ai.example/base')).toBe(
      'https://ai.example/base/v2/health',
    );
    expect(() => resolveDeltrelAiMoveUrl('https://ai.example/v1/move')).toThrow(
      /v2 API/,
    );
  });

  it('defaults browser traffic to the same-origin v2 proxy', () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_URL', '');
    expect(configuredServerAiUrl()).toBe('/v2/move');
    expect(configuredServerHealthUrl()).toBe('/v2/health');
  });

  it('posts schema v3 with request identity and no browser bearer secret', async () => {
    const fetchMock = vi.fn(
      async (url: string | URL | Request, init?: RequestInit) => {
        expect(String(url)).toBe('https://ai.example/v2/move');
        const headers = new Headers(init?.headers);
        expect(headers.get('X-Request-ID')).toBe(request.requestId);
        expect(headers.has('Authorization')).toBe(false);
        const body = JSON.parse(String(init?.body));
        expect(body).toMatchObject({
          schema_version: 3,
          to_move: 0,
          mode: 'double',
          handicap: 1,
          pie: false,
          pda: 0,
          search: { simulations: 4, max_considered: 2 },
        });
        expect(body.pass_streak).toBeUndefined();
        return new Response(JSON.stringify(representativeAnalyzeResponse()), {
          status: 200,
          headers: { 'X-Request-ID': request.requestId },
        });
      },
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      requestServerAiDecision(request, {
        url: 'https://ai.example',
        search: { simulations: 4, maxConsidered: 2 },
      }),
    ).resolves.toMatchObject({
      response: makeAiResponse(request, { type: 'place', node: 0 }),
      analysis: { simulations: 4, maxConsidered: 2, modelVersion: 'fake-v2' },
    });
    await expect(
      requestServerAiAction(request, {
        url: 'https://ai.example',
        search: { simulations: 4, maxConsidered: 2 },
      }),
    ).resolves.toEqual(makeAiResponse(request, { type: 'place', node: 0 }));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
  it('opts into diagnostics and retries only an explicit unknown-field rejection', async () => {
    const rejected = {error: {code: 'invalid_request', message: 'unknown field', details: [
      {type: 'extra_forbidden', location: ['body', 'include_network_output']},
    ]}};
    const calls: Record<string, unknown>[] = [];
    const fetchMock = vi.fn(async (_url: unknown, init: RequestInit) => {
      calls.push(JSON.parse(String(init.body)));
      return calls.length === 1 ? new Response(JSON.stringify(rejected), {status: 422})
        : new Response(JSON.stringify(representativeAnalyzeResponse()), {status: 200});
    });
    vi.stubGlobal('fetch', fetchMock);
    await expect(requestServerAiDecision(request, {includeNetworkOutput: true,
      search: {simulations: 4, maxConsidered: 2}})).resolves.toBeDefined();
    expect(calls).toHaveLength(2);
    expect(calls[0].include_network_output).toBe(true);
    expect(calls[1].include_network_output).toBeUndefined();
    expect(fetchMock.mock.calls[0][1].signal).toBe(fetchMock.mock.calls[1][1].signal);
    expect(toAnalyzeRequest(request).include_network_output).toBeUndefined();
  });

  it('does not retry ordinary validation or inference failures', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({error: {
      code: 'invalid_request', message: 'invalid state', details: [
        {type: 'extra_forbidden', location: ['body', 'include_network_output']},
        {type: 'value_error', location: ['body', 'stones']},
      ],
    }}), {status: 422}));
    vi.stubGlobal('fetch', fetchMock);
    await expect(requestServerAiDecision(request, {includeNetworkOutput: true})).rejects.toThrow('invalid state');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('retains caller cancellation across the compatibility retry', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn(async (_url: unknown, init: RequestInit) => {
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({error: {
        code: 'invalid_request', details: [{type: 'extra_forbidden', location: ['body', 'include_network_output']}],
      }}), {status: 422});
      controller.abort();
      expect(init.signal?.aborted).toBe(true);
      throw new DOMException('aborted', 'AbortError');
    });
    vi.stubGlobal('fetch', fetchMock);
    await expect(requestServerAiDecision(request, {includeNetworkOutput: true, signal: controller.signal})).rejects.toMatchObject({code: 'cancelled'});
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('bounds diagnostic responses before parsing JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('x'.repeat(1024 * 1024 + 1))));
    await expect(requestServerAiDecision(request, {includeNetworkOutput: true})).rejects.toThrow(/size limit/);
  });

  it('shares the original timeout across an older-service compatibility retry', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const started: number[] = [];
    const fetchMock = vi.fn(async (_url: unknown, init: RequestInit) => {
      started.push(Date.now());
      if (started.length === 1) {
        await new Promise(resolve => setTimeout(resolve, 7));
        return new Response(JSON.stringify({error: {code: 'invalid_request', details: [
          {type: 'extra_forbidden', location: ['body', 'include_network_output']},
        ]}}), {status: 422});
      }
      return new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), {once: true});
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const expected = expect(requestServerAiDecision(request, {includeNetworkOutput: true, timeoutMs: 10})).rejects.toMatchObject({code: 'timeout'});
    await vi.advanceTimersByTimeAsync(10);
    await expected;
    expect(started).toEqual([0, 7]);
    expect(Date.now()).toBe(10);
  });

});
