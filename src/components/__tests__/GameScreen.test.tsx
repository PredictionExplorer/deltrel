import { StrictMode } from 'react';
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  DeltrelAiAnalysis,
  DeltrelAiDecision,
} from '@/lib/deltrel/ai/decision';
import { DeltrelAiError } from '@/lib/deltrel/ai/errors';
import {
  acceptAiResponse,
  makeAiResponse,
  type AtomicGameAction,
  type DeltrelAiRequest,
} from '@/lib/deltrel/ai/protocol';
import { prepareLocalAi, requestLocalAiDecision } from '@/lib/deltrel/ai/local-client';
import { checkAiCapabilities, type AiCapabilities } from '@/lib/deltrel/ai/capabilities';
import { publishLocalAiStatus } from '@/lib/deltrel/ai/local-ai-status';
import { requestServerAiDecision } from '@/lib/deltrel/ai/server-client';
import { getBoard, parseLabel } from '@/lib/deltrel/board';
import {
  DEFAULT_AI_SEARCH_SETTINGS,
  useAppStore,
  type AppState,
} from '@/lib/store';
import { GameScreen } from '../GameScreen';

vi.mock('@/lib/deltrel/ai/protocol', async () => {
  const actual = await vi.importActual<typeof import('@/lib/deltrel/ai/protocol')>(
    '@/lib/deltrel/ai/protocol',
  );
  return { ...actual, acceptAiResponse: vi.fn(actual.acceptAiResponse) };
});

vi.mock('@/lib/deltrel/ai/server-client', async () => {
  const actual = await vi.importActual<typeof import('@/lib/deltrel/ai/server-client')>(
    '@/lib/deltrel/ai/server-client',
  );
  return { ...actual, requestServerAiDecision: vi.fn() };
});

vi.mock('@/lib/deltrel/ai/local-client', async () => {
  const actual = await vi.importActual<typeof import('@/lib/deltrel/ai/local-client')>(
    '@/lib/deltrel/ai/local-client',
  );
  return { ...actual, requestLocalAiDecision: vi.fn(), prepareLocalAi: vi.fn() };
});

vi.mock('@/lib/deltrel/ai/capabilities', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/capabilities')>('@/lib/deltrel/ai/capabilities'),
  checkAiCapabilities: vi.fn(),
}));

const browserReady = { modelVersion: 'browser-champion', bytes: 37_577_312, backend: 'wasm' as const, cached: true };
const browserCapabilities: AiCapabilities = {
  server: { status: 'available', label: 'Server AI' },
  local: {
    status: 'available', label: 'Browser AI',
    browserModel: { modelVersion: browserReady.modelVersion, bytes: browserReady.bytes, sha256: 'a'.repeat(64) },
    search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 64, maxConsidered: 8 }, presets: {} },
  },
};

const config = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['Ada', 'Grace'],
} as const;

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeDecision(
  request: DeltrelAiRequest,
  action: AtomicGameAction = { type: 'place', node: 0 },
  overrides: Partial<DeltrelAiAnalysis> = {},
): DeltrelAiDecision {
  const anchor = action.type === 'place' ? action.node : 0;
  const rootActions = overrides.rootActions ?? [
    { type: 'place', node: anchor },
    { type: 'place', node: anchor + 1 },
    { type: 'place', node: anchor + 2 },
  ];
  const rootVisits = overrides.rootVisits ?? [6, 3, 1];
  const simulations =
    overrides.simulations ??
    rootVisits.reduce((total, visits) => total + visits, 0);
  return {
    response: makeAiResponse(request, action),
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: { loss: 0.25, win: 0.75 },
      modelValue: 0.5,
      searchValue: 0.4,
      rootValue: 0.4,
      swapRecommended: action.type === 'swap',
      expectedMargin: 2.5,
      rootActions,
      rootPolicy: rootActions.map((_, index) => (index === 0 ? 0.6 : 0.2)),
      rootQ: rootActions.map((_, index) => 0.3 - index * 0.2),
      rootVisits,
      modelVersion: 'test-champion',
      modelStep: 700,
      modelIdentity: 'sha256-test',
      simulations,
      maxConsidered: 16,
      timingMs: {
        queue: 1,
        modelLoad: 2,
        inferenceSearch: 9,
        total: 12,
      },
      ...overrides,
    },
  };
}

function resetPlayingStore(overrides: Partial<AppState> = {}) {
  useAppStore.setState({
    phase: 'playing',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: ['server', 'human'],
    aiSearchSettings: {
      server: { ...DEFAULT_AI_SEARCH_SETTINGS.server },
      local: { ...DEFAULT_AI_SEARCH_SETTINGS.local },
    },
    aiPaused: false,
    log: [],
    redoStack: [],
    reviewing: false,
    earlyOutcome: null,
    clinchAcknowledgement: null,
    ...overrides,
  });
}

function PhaseHarness() {
  const phase = useAppStore((state) => state.phase);
  return phase === 'playing' ? <GameScreen /> : <p>Setup is ready</p>;
}

async function completedEstimate() {
  const panel = screen.getByRole('region', { name: 'Engine estimate' });
  await within(panel).findByText(/Expected final points come from/);
  return panel;
}

beforeEach(() => {
  localStorage.clear();
  vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '0');
  resetPlayingStore();
  vi.mocked(requestServerAiDecision).mockReset();
  vi.mocked(requestLocalAiDecision).mockReset();
  publishLocalAiStatus({ phase: 'ready', info: browserReady });
  vi.mocked(checkAiCapabilities).mockReset();
  vi.mocked(checkAiCapabilities).mockResolvedValue(browserCapabilities);
  vi.mocked(prepareLocalAi).mockReset();
  vi.mocked(prepareLocalAi).mockImplementation(async () => {
    publishLocalAiStatus({ phase: 'ready', info: browserReady });
    return browserReady;
  });
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
});

describe('GameScreen AI lifecycle', () => {
  it('does not download or request a restored browser turn until preparation is explicitly requested', async () => {
    publishLocalAiStatus({ phase: 'idle' });
    resetPlayingStore({ controllers: ['local', 'human'] });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestLocalAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    const prepare = await screen.findByRole('button', { name: 'Prepare browser AI' });
    expect(prepareLocalAi).not.toHaveBeenCalled();
    expect(requestLocalAiDecision).not.toHaveBeenCalled();
    await user.click(prepare);
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    expect(prepareLocalAi).toHaveBeenCalledOnce();
    expect(useAppStore.getState().log).toEqual([]);
  });

  it('keeps an authorized browser request alive through cache initialization and cancels it deliberately', async () => {
    resetPlayingStore({ controllers: ['local', 'human'] });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestLocalAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    const signal = vi.mocked(requestLocalAiDecision).mock.calls[0][1]!.signal!;
    act(() => publishLocalAiStatus({ phase: 'initializing', loadedBytes: browserReady.bytes, totalBytes: browserReady.bytes, modelVersion: browserReady.modelVersion, cached: true }));
    expect(signal.aborted).toBe(false);
    expect(requestLocalAiDecision).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('button', { name: 'Cancel browser AI preparation' }));
    expect(signal.aborted).toBe(true);
    expect(useAppStore.getState().aiPaused).toBe(true);
    expect(useAppStore.getState().log).toEqual([]);
  });

  it('uses browser analysis in a human game when the public site has no online engine', async () => {
    publishLocalAiStatus({ phase: 'idle' });
    resetPlayingStore({ controllers: ['human', 'human'] });
    vi.mocked(checkAiCapabilities).mockResolvedValue({ ...browserCapabilities, server: { status: 'unavailable', label: 'Online AI', code: 'offline', reason: 'No online engine.', retryable: false } });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestLocalAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await user.click(await screen.findByRole('button', { name: 'Prepare browser AI' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze position' })).toBeEnabled());
    expect(requestLocalAiDecision).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    expect(requestServerAiDecision).not.toHaveBeenCalled();
    const [request] = vi.mocked(requestLocalAiDecision).mock.calls[0];
    await act(async () => flight.resolve(makeDecision(request)));
    expect(useAppStore.getState().log).toEqual([]);
    expect(await completedEstimate()).toHaveTextContent('Browser engine');
  });

  it('reuses one logical request when Strict Mode replays effects', async () => {
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);

    render(
      <StrictMode>
        <GameScreen />
      </StrictMode>,
    );

    expect(await screen.findByText('Server AI is thinking…')).toBeInTheDocument();
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];
    expect(options?.signal?.aborted).toBe(false);
    expect(options?.search).toEqual({ simulations: 512, maxConsidered: 16 });

    flight.resolve(makeDecision(request));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    expect(
      screen.getByRole('region', { name: 'Engine estimate' }),
    ).toBeInTheDocument();
    expect(screen.queryByText('Search value')).not.toBeInTheDocument();
  });

  it('aborts on exit and ignores a response that arrives after cancellation', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<PhaseHarness />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];

    await user.click(screen.getByRole('button', { name: 'New game' }));
    expect(options?.signal?.aborted).toBe(true);
    expect(screen.getByText('Setup is ready')).toBeInTheDocument();

    await act(async () => {
      flight.resolve(makeDecision(request));
      await flight.promise;
    });
    expect(acceptAiResponse).not.toHaveBeenCalled();
    expect(useAppStore.getState().log).toEqual([]);
    expect(
      screen.queryByRole('region', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('rejects a stale response without mutating the action log', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    const stale = makeDecision(request);
    flight.resolve({
      ...stale,
      response: { ...stale.response, requestId: 'obsolete-request' },
    });

    await waitFor(() => expect(acceptAiResponse).toHaveBeenCalledOnce());
    expect(useAppStore.getState().log).toEqual([]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Engine estimate' })).toBeInTheDocument();
    expect(screen.queryByText(/Expected final points come from/)).not.toBeInTheDocument();
  });

  it('retries a recoverable error and applies the successful response', async () => {
    const retryFlight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision)
      .mockRejectedValueOnce(new DeltrelAiError('network', 'Server AI is offline.', true))
      .mockReturnValueOnce(retryFlight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Server AI is offline.');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledTimes(2));

    const [retryRequest] = vi.mocked(requestServerAiDecision).mock.calls[1];
    retryFlight.resolve(makeDecision(retryRequest));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
  });

  it('lets the current player take over after an AI error', async () => {
    vi.mocked(requestServerAiDecision).mockRejectedValue(
      new DeltrelAiError('network', 'Server AI is offline.', true),
    );
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByRole('alert');
    await user.click(screen.getByRole('button', { name: 'Take over as human' }));

    expect(useAppStore.getState().controllers).toEqual(['human', 'human']);
    const firstNode = screen.getByRole('button', {
      name: /node G4, empty interior node; ada may place here/i,
    });
    await user.click(firstNode);
    expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]);
  });

  it.each(['0', '1'])('passes server budgets and clamps browser budgets to published limits with devtools=%s', async (devtools) => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', devtools);
    resetPlayingStore({
      aiSearchSettings: {
        server: { simulations: 777, maxConsidered: 21 },
        local: { simulations: 123, maxConsidered: 9 },
      },
    });
    const serverFlight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(serverFlight.promise);
    const first = render(<GameScreen />);

    await screen.findByText(devtools === '1' ? 'Mac engine — current champion is thinking…' : 'Server AI is thinking…');
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    expect(vi.mocked(requestServerAiDecision).mock.calls[0][1]?.search).toEqual({
      simulations: 777,
      maxConsidered: 21,
    });
    first.unmount();

    resetPlayingStore({
      controllers: ['local', 'human'],
      aiSearchSettings: {
        server: { simulations: 777, maxConsidered: 21 },
        local: { simulations: 123, maxConsidered: 9 },
      },
    });
    const localFlight = deferred<DeltrelAiDecision>();
    vi.mocked(requestLocalAiDecision).mockReturnValue(localFlight.promise);
    render(<GameScreen />);

    await screen.findByText(devtools === '1' ? 'Browser AI — trained champion is thinking…' : 'Browser AI is thinking…');
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    expect(vi.mocked(requestLocalAiDecision).mock.calls[0][1]?.search).toEqual({
      simulations: 64,
      maxConsidered: 8,
    });
  });

  it('renders named estimates and top board labels from the accepted perspective', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const { container } = render(<GameScreen />);
    const user = userEvent.setup();

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(
      makeDecision(request, { type: 'place', node: 0 }, {
        outcome: { loss: 0.2, win: 0.8 },
        modelValue: 0.6,
        searchValue: -0.25,
        expectedMargin: 3.5,
        rootActions: [
          { type: 'place', node: 0 },
          { type: 'place', node: 1 },
          { type: 'place', node: 2 },
        ],
        rootPolicy: [0.2, 0.5, 0.3],
        rootQ: [-0.1, 0.7, 0.3],
        rootVisits: [2, 7, 5],
        simulations: 14,
        modelStep: 12_345,
        timingMs: {
          queue: 1,
          modelLoad: 2,
          inferenceSearch: 15,
          total: 18,
        },
      }),
    );

    const panel = await completedEstimate();
    const ada = within(panel).getByRole('article', { name: 'Ada forecast' });
    const grace = within(panel).getByRole('article', { name: 'Grace forecast' });
    expect(within(ada).getByText('80.0%')).toBeInTheDocument();
    expect(within(grace).getByText('20.0%')).toBeInTheDocument();
    expect(within(ada).getByText('12.3')).toBeInTheDocument();
    expect(within(grace).getByText('8.8')).toBeInTheDocument();
    expect(within(panel).getByText('Ada +3.5 expected points')).toBeInTheDocument();
    expect(within(panel).getByText(/Earlier position/)).toBeInTheDocument();
    await user.click(within(panel).getByRole('button', { name: /^Search and model details/ }));
    expect(within(panel).getByText('-0.250')).toBeInTheDocument();
    expect(within(panel).queryByText('37.5%')).not.toBeInTheDocument();
    expect(within(panel).getByText('14')).toBeInTheDocument();
    expect(within(panel).getByText('18 ms')).toBeInTheDocument();
    expect(within(panel).getByText('12,345')).toBeInTheDocument();
    await user.click(within(panel).getByRole('button', { name: /^Search candidates/ }));
    const candidates = within(within(panel).getByRole('table', { name: 'Search candidates' })).getAllByRole('row').slice(1);
    expect(candidates[0]).toHaveTextContent('E4');
    expect(candidates[1]).toHaveTextContent('E5');
    expect(candidates[2]).toHaveTextContent('G4');
    expect((await axe(container)).violations).toEqual([]);
  });

  it('shows official final predictions during normal play without developer settings', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '0');
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const { container } = render(<GameScreen />);
    const user = userEvent.setup();
    await screen.findByText('Server AI is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(makeDecision(request, { type: 'place', node: 0 }, {
      predictions: {
        perspective: 0, finalBasis: 'official_end',
        finalCounts: [
          { player: 0, shores: 12.5, networks: 2.5, corners: 3.2, cornerBonusProbability: 0.8 },
          { player: 1, shores: 7.5, networks: 1.5, corners: 1.8, cornerBonusProbability: 0.2 },
        ],
        opponentReply: { player: 1, kind: 'place', node: 2, probability: 0.25 },
        secondStone: null,
      },
    }));
    const panel = await completedEstimate();
    await user.click(within(panel).getByRole('button', { name: /^Final-count forecasts/ }));
    const table = within(panel).getByRole('table');
    const ada = within(table).getByRole('row', { name: /Ada/ });
    expect(within(ada).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['12.5', '2.5', '3.2', '80.0%']);
    const grace = within(table).getByRole('row', { name: /Grace/ });
    expect(within(grace).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['7.5', '1.5', '1.8', '20.0%']);
    expect(within(panel).getByText(/official final counts/)).toBeInTheDocument();
    expect(within(panel).getByText('Grace · next reply')).toBeInTheDocument();
    expect(within(panel).getByText('E5 · 25.0%')).toBeInTheDocument();
    await user.click(within(panel).getByRole('button', { name: /^Search and model details/ }));
    expect(within(panel).getByText('Search value')).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('maps a second-player analysis to the correct named win estimates', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    resetPlayingStore({
      controllers: ['human', 'server'],
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
    });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    expect(request.state.toMove).toBe(1);
    flight.resolve(
      makeDecision(request, { type: 'place', node: 2 }, {
        outcome: { loss: 0.3, win: 0.7 },
        modelValue: 0.4,
        expectedMargin: -4,
      }),
    );

    const panel = await completedEstimate();
    expect(within(within(panel).getByRole('article', { name: 'Ada forecast' })).getByText('30.0%')).toBeInTheDocument();
    expect(within(within(panel).getByRole('article', { name: 'Grace forecast' })).getByText('70.0%')).toBeInTheDocument();
    expect(within(panel).getByText('Ada +4.0 expected points')).toBeInTheDocument();
    expect(within(panel).getByText(/After move 2 · Grace to move/)).toBeInTheDocument();
  });

  it('restores the exact cached estimate on undo', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(makeDecision(request));
    await completedEstimate();

    await user.click(screen.getByRole('button', { name: 'Undo' }));
    const panel = screen.getByRole('region', { name: 'Engine estimate' });
    expect(within(panel).getByText(/Opening position/)).toBeInTheDocument();
    expect(within(panel).queryByText(/Earlier position/)).not.toBeInTheDocument();
    expect(within(within(panel).getByRole('article', { name: 'Ada forecast' })).getByText('75.0%')).toBeInTheDocument();
  });

  it('retains the previous estimate and expanded details during the next search and an error', async () => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '1');
    resetPlayingStore({ controllers: ['server', 'server'] });
    const first = deferred<DeltrelAiDecision>();
    const second = deferred<DeltrelAiDecision>();
    const user = userEvent.setup();
    vi.mocked(requestServerAiDecision)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [firstRequest] = vi.mocked(requestServerAiDecision).mock.calls[0];
    first.resolve(makeDecision(firstRequest));
    const panel = await completedEstimate();
    await user.click(within(panel).getByRole('button', { name: /^Search candidates/ }));
    await waitFor(() =>
      expect(requestServerAiDecision).toHaveBeenCalledTimes(2),
    );

    second.reject(new DeltrelAiError('timeout', 'Engine timed out.', true));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Engine timed out.',
    );
    expect(within(panel).getByRole('button', { name: /^Search candidates/ })).toHaveAttribute('aria-expanded', 'true');
    expect(within(within(panel).getByRole('article', { name: 'Ada forecast' })).getByText('75.0%')).toBeInTheDocument();
    expect(within(panel).getByText(/Earlier position/)).toBeInTheDocument();
  });

  it.each(['0', '1'])('retries a timeout with less effort with devtools=%s', async (devtools) => {
    vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', devtools);
    const retryFlight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision)
      .mockRejectedValueOnce(new DeltrelAiError('timeout', 'Engine timed out.', true))
      .mockReturnValueOnce(retryFlight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByRole('alert');
    await user.click(screen.getByRole('button', { name: 'Use less effort' }));
    await waitFor(() =>
      expect(requestServerAiDecision).toHaveBeenCalledTimes(2),
    );
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({
      simulations: 256,
      maxConsidered: 8,
    });
    expect(vi.mocked(requestServerAiDecision).mock.calls[1][1]?.search).toEqual({
      simulations: 256,
      maxConsidered: 8,
    });
  });

  it('has no detectable accessibility violations for a human turn', async () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    const { container } = render(<GameScreen />);

    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
    expect(requestServerAiDecision).not.toHaveBeenCalled();
  });

  it('blocks automatic play for the clinch decision and proof view', async () => {
    const user = userEvent.setup();
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    resetPlayingStore({
      controllers: ['server', 'server'],
      log: Array.from({ length: 49 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
    });
    const { container } = render(<GameScreen />);

    const clinch = screen.getByRole('dialog', {
      name: 'Grace cannot be caught',
    });
    expect(requestServerAiDecision).not.toHaveBeenCalled();
    await user.click(
      within(clinch).getByRole('button', { name: 'Continue playing' }),
    );
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledOnce());
    const signal = vi.mocked(requestServerAiDecision).mock.calls[0][1]?.signal;
    expect(signal?.aborted).toBe(false);

    const actionDock = within(
      container.querySelector('[data-action-dock]') as HTMLElement,
    );
    await user.click(
      actionDock.getByRole('button', { name: /^Show proof board/ }),
    );
    await waitFor(() => expect(signal?.aborted).toBe(true));
    expect(useAppStore.getState().log).toHaveLength(49);
  });

  it('does not reopen Rules after an AI finishes the game behind it', async () => {
    const user = userEvent.setup();
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    resetPlayingStore({
      controllers: ['server', 'server'],
      log: Array.from({ length: 49 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
      clinchAcknowledgement: { winner: 1, atLogLength: 49 },
    });
    render(<GameScreen />);

    await screen.findByText(/server ai is thinking/i);
    await user.click(screen.getByRole('button', { name: 'Rules' }));
    expect(
      screen.getByRole('dialog', { name: 'How to play Deltrel' }),
    ).toBeInTheDocument();

    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(makeDecision(request, { type: 'place', node: 49 }));
    const result = await screen.findByRole('dialog', { name: 'Game over' });
    await user.click(
      within(result).getByRole('button', { name: 'Review board' }),
    );

    expect(
      screen.queryByRole('dialog', { name: 'How to play Deltrel' }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole('region', { name: 'Game board' }),
    ).toBeInTheDocument();
  });
});

describe('GameScreen position-scoped engine inspection', () => {
  it('analyzes a human position without applying the suggested move', async () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    const before = useAppStore.getState().log;
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];
    expect(options?.includeNetworkOutput).toBe(true);
    flight.resolve(makeDecision(request));
    const panel = await completedEstimate();
    expect(useAppStore.getState().log).toBe(before);
    expect(within(panel).getByText(/Suggested move/)).toHaveTextContent('G4');
    expect(within(panel).queryByText(/Earlier position/)).not.toBeInTheDocument();
    expect(screen.getByRole('group', { name: /0 of 50 nodes occupied/ })).toBeInTheDocument();
  });

  it('cancels inspection when a human moves and rejects the late result', async () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];
    await user.click(screen.getByRole('button', { name: /^Node E4, empty/ }));
    expect(options?.signal?.aborted).toBe(true);
    await act(async () => { flight.resolve(makeDecision(request)); await flight.promise; });
    expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 1 }]);
    expect(screen.queryByText(/Expected final points come from/)).not.toBeInTheDocument();
  });

  it('keeps reviewed-position estimates fixed while AI-versus-AI play continues', async () => {
    resetPlayingStore({ controllers: ['server', 'server'] });
    const first = deferred<DeltrelAiDecision>();
    const second = deferred<DeltrelAiDecision>();
    const pending = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(pending.promise)
      .mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledOnce());
    const firstRequest = vi.mocked(requestServerAiDecision).mock.calls[0][0];
    first.resolve(makeDecision(firstRequest));
    await completedEstimate();
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledTimes(2));
    await user.click(screen.getByRole('button', { name: 'Jump to the empty board' }));
    const panel = screen.getByRole('region', { name: 'Engine estimate' });
    expect(within(panel).getByText(/Opening position/)).toBeInTheDocument();
    const secondRequest = vi.mocked(requestServerAiDecision).mock.calls[1][0];
    second.resolve(makeDecision(secondRequest, { type: 'place', node: 1 }, {
      outcome: { loss: 0.1, win: 0.9 }, modelValue: 0.8,
    }));
    await waitFor(() => expect(useAppStore.getState().log).toHaveLength(2));
    expect(within(within(panel).getByRole('article', { name: 'Ada forecast' })).getByText('75.0%')).toBeInTheDocument();
    expect(within(panel).getByText(/Opening position/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Step one move forward' }));
    expect(within(panel).getByText(/After move 1 · Grace to move/)).toBeInTheDocument();
    expect(within(within(panel).getByRole('article', { name: 'Grace forecast' })).getByText('90.0%')).toBeInTheDocument();
  });

  it('pauses AI-versus-AI play without losing the latest estimate and resumes the same position', async () => {
    resetPlayingStore({ controllers: ['server', 'server'] });
    const first = deferred<DeltrelAiDecision>();
    const second = deferred<DeltrelAiDecision>();
    const pending = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(pending.promise)
      .mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledOnce());
    first.resolve(makeDecision(vi.mocked(requestServerAiDecision).mock.calls[0][0]));
    await completedEstimate();
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledTimes(2));
    const [secondRequest, secondOptions] = vi.mocked(requestServerAiDecision).mock.calls[1];
    await user.click(screen.getByRole('button', { name: 'Pause AI' }));
    expect(secondOptions?.signal?.aborted).toBe(true);
    expect(useAppStore.getState().aiPaused).toBe(true);
    await act(async () => { second.resolve(makeDecision(secondRequest, { type: 'place', node: 1 })); await second.promise; });
    expect(useAppStore.getState().log).toHaveLength(1);
    expect(screen.getByText(/Expected final points come from/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Resume AI' }));
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledTimes(3));
    expect(vi.mocked(requestServerAiDecision).mock.calls[2][0].stateHash).toBe(secondRequest.stateHash);
  });

  it.each(['local', 'server'] as const)(
    'requires paused play before browser inspection while the next %s turn is running',
    async (nextController) => {
      resetPlayingStore({ controllers: ['local', nextController] });
      const first = deferred<DeltrelAiDecision>();
      const autoplay = deferred<DeltrelAiDecision>();
      const inspection = deferred<DeltrelAiDecision>();
      const resumed = deferred<DeltrelAiDecision>();
      const local = vi.mocked(requestLocalAiDecision);
      const server = vi.mocked(requestServerAiDecision);
      local.mockReturnValue(resumed.promise).mockReturnValueOnce(first.promise);
      if (nextController === 'local') local.mockReturnValueOnce(autoplay.promise);
      else server.mockReturnValue(resumed.promise).mockReturnValueOnce(autoplay.promise);
      local.mockReturnValueOnce(inspection.promise);
      const user = userEvent.setup();
      render(<GameScreen />);
      await waitFor(() => expect(local).toHaveBeenCalledOnce());
      first.resolve(makeDecision(local.mock.calls[0][0]));
      await completedEstimate();
      const automatic = nextController === 'local' ? local : server;
      const autoCallIndex = nextController === 'local' ? 1 : 0;
      await waitFor(() => expect(automatic).toHaveBeenCalledTimes(autoCallIndex + 1));
      const [autoRequest, autoOptions] = automatic.mock.calls[autoCallIndex];
      await user.click(screen.getByRole('button', { name: 'Jump to the empty board' }));
      const analyze = screen.getByRole('button', { name: 'Analyze position' });
      expect(analyze).toBeDisabled();
      expect(screen.getByText('Pause AI to analyze this position with the browser engine.')).toBeInTheDocument();
      await user.click(analyze);
      expect(local).toHaveBeenCalledTimes(nextController === 'local' ? 2 : 1);
      expect(autoOptions?.signal?.aborted).toBe(false);

      await user.click(screen.getByRole('button', { name: 'Pause AI' }));
      expect(useAppStore.getState().aiPaused).toBe(true);
      expect(autoOptions?.signal?.aborted).toBe(true);
      expect(analyze).toBeEnabled();
      await user.click(analyze);
      const inspectionIndex = nextController === 'local' ? 2 : 1;
      await waitFor(() => expect(local).toHaveBeenCalledTimes(inspectionIndex + 1));
      const [inspectionRequest, inspectionOptions] = local.mock.calls[inspectionIndex];
      expect(inspectionRequest.actionLog).toHaveLength(0);
      await user.click(screen.getByRole('button', { name: 'Step one move forward' }));
      expect(inspectionOptions?.signal?.aborted).toBe(true);
      await act(async () => {
        autoplay.resolve(makeDecision(autoRequest, { type: 'place', node: 1 }));
        inspection.resolve(makeDecision(inspectionRequest));
        await Promise.all([autoplay.promise, inspection.promise]);
      });
      expect(useAppStore.getState().log).toHaveLength(1);
      await user.click(screen.getByRole('button', { name: 'Resume AI' }));
      const expectedCalls = nextController === 'local' ? 4 : 2;
      await waitFor(() => expect(automatic).toHaveBeenCalledTimes(expectedCalls));
      expect(automatic.mock.calls.at(-1)?.[0].stateHash).toBe(autoRequest.stateHash);
    },
  );

  it('clears analysis on rematch even when the opening position has the same identity', async () => {
    resetPlayingStore({ config: { ...config, playerNames: [...config.playerNames], pieRule: true }, controllers: ['human', 'human'] });
    const flight = deferred<DeltrelAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    flight.resolve(makeDecision(vi.mocked(requestServerAiDecision).mock.calls[0][0]));
    await completedEstimate();
    await user.click(screen.getByRole('button', { name: 'Resign Ada' }));
    await user.click(within(screen.getByRole('dialog', { name: 'Resign Ada?' })).getByRole('button', { name: 'Resign Ada' }));
    await user.click(within(screen.getByRole('dialog', { name: 'Game over' })).getByRole('button', { name: 'Rematch' }));
    expect(useAppStore.getState().log).toHaveLength(0);
    expect(screen.queryByText(/Expected final points come from/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze position' })).toBeEnabled();
  });
});

describe('GameScreen move review', () => {
  const fourMoves = [0, 1, 2, 3].map((node) => ({
    type: 'place' as const,
    node,
  }));

  it('reviews an earlier position without touching the live log', async () => {
    const user = userEvent.setup();
    resetPlayingStore({ controllers: ['human', 'human'], log: fourMoves });
    render(<GameScreen />);

    const panel = screen.getByRole('region', { name: 'Move history' });
    expect(within(panel).getByText('Live position')).toBeInTheDocument();

    await user.click(
      within(panel).getByRole('button', { name: 'Go to move 2: Grace at E4' }),
    );

    // The board becomes a read-only snapshot of the position after move 2.
    expect(
      screen.getByRole('img', {
        name: /Deltrel board with 4 rings, 2 of 50 nodes occupied/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText('Reviewing move 2 of 4')).toBeInTheDocument();
    expect(screen.getByText('Position at move 2')).toBeInTheDocument();
    expect(useAppStore.getState().log).toHaveLength(4);
    expect(useAppStore.getState().redoStack).toHaveLength(0);

    // Arrow keys step through the history.
    await user.keyboard('{ArrowLeft}');
    expect(screen.getByText('Reviewing move 1 of 4')).toBeInTheDocument();
    await user.keyboard('{ArrowRight}');
    expect(screen.getByText('Reviewing move 2 of 4')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Back to live' }));
    expect(
      screen.getByRole('group', {
        name: /Deltrel board with 4 rings, 4 of 50 nodes occupied/i,
      }),
    ).toBeInTheDocument();
    expect(useAppStore.getState().log).toHaveLength(4);
  });

  it('keeps the finished pair highlighted after the turn ends', () => {
    resetPlayingStore({ controllers: ['human', 'human'], log: fourMoves });
    const { container } = render(<GameScreen />);

    // Grace's completed pair (nodes 1 and 2) stays ringed while Ada plays.
    expect(
      container.querySelector('[data-last-turn-move="1"]'),
    ).toBeInTheDocument();
    expect(
      container.querySelector('[data-move-badge="2"][data-move-order="2"]'),
    ).toBeInTheDocument();
    // Ada's in-progress stone is the pulsing last move, numbered 1 of 2.
    expect(container.querySelector('[data-last-move="3"]')).toBeInTheDocument();
    expect(
      container.querySelector('[data-move-badge="3"][data-move-order="1"]'),
    ).toBeInTheDocument();
  });

  it('rewinds the live game to the reviewed position with play-from-here', async () => {
    const user = userEvent.setup();
    resetPlayingStore({ controllers: ['human', 'human'], log: fourMoves });
    render(<GameScreen />);

    const panel = screen.getByRole('region', { name: 'Move history' });
    await user.click(
      within(panel).getByRole('button', { name: 'Go to move 2: Grace at E4' }),
    );
    await user.click(
      within(panel).getByRole('button', { name: 'Play from here' }),
    );

    expect(useAppStore.getState().log).toEqual(fourMoves.slice(0, 2));
    expect(useAppStore.getState().redoStack).toEqual([
      fourMoves[3],
      fourMoves[2],
    ]);
    expect(useAppStore.getState().aiPaused).toBe(true);
    // Back on the live position, ready to branch.
    expect(
      screen.getByRole('group', {
        name: /Deltrel board with 4 rings, 2 of 50 nodes occupied/i,
      }),
    ).toBeInTheDocument();
    expect(within(panel).getByText('Live position')).toBeInTheDocument();
  });

  it('exits review when undo shortens the log past the viewed ply', async () => {
    const user = userEvent.setup();
    resetPlayingStore({ controllers: ['human', 'human'], log: fourMoves });
    render(<GameScreen />);

    const panel = screen.getByRole('region', { name: 'Move history' });
    await user.click(
      within(panel).getByRole('button', { name: 'Go to move 2: Grace at E4' }),
    );
    await user.click(screen.getByRole('button', { name: 'Undo' }));

    expect(useAppStore.getState().log).toHaveLength(3);
    expect(within(panel).getByText('Live position')).toBeInTheDocument();
    expect(
      screen.getByRole('group', {
        name: /Deltrel board with 4 rings, 3 of 50 nodes occupied/i,
      }),
    ).toBeInTheDocument();
  });
});

describe('GameScreen score guidance', () => {
  it('shows both extreme completion scores throughout play', () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    render(<GameScreen />);

    expect(
      screen.getByRole('heading', { name: 'Completion bounds' }),
    ).toBeInTheDocument();
    expect(screen.getByText('All open → Ada')).toBeInTheDocument();
    expect(screen.getByText('All open → Grace')).toBeInTheDocument();
    expect(
      screen.getByText('The final winner is not clinched yet.'),
    ).toBeInTheDocument();
  });

  it('pauses on a clinch, continues once, and toggles a non-mutating proof', async () => {
    const user = userEvent.setup();
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: Array.from({ length: 49 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
    });
    const { container } = render(<GameScreen />);

    const dialog = screen.getByRole('dialog', {
      name: 'Grace cannot be caught',
    });
    expect(
      screen.getByRole('heading', { name: 'Grace has clinched' }),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText(/even if every remaining open node became ada/i),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByRole('button', { name: 'Continue playing' }),
    ).toHaveFocus();

    await user.click(
      within(dialog).getByRole('button', { name: 'Continue playing' }),
    );
    expect(dialog).not.toBeInTheDocument();
    expect(
      screen.getByRole('region', { name: 'Game board' }),
    ).toHaveFocus();
    expect(screen.getByText(/Grace to play/)).toBeInTheDocument();
    const actionDock = within(
      container.querySelector('[data-action-dock]') as HTMLElement,
    );
    expect(
      actionDock.getByRole('button', { name: /^Show proof board/ }),
    ).toBeInTheDocument();
    expect(actionDock.getByRole('button', { name: 'End game' })).toBeInTheDocument();

    const originalLog = useAppStore.getState().log;
    await user.click(
      actionDock.getByRole('button', { name: /^Show proof board/ }),
    );
    expect(
      screen.getByRole('img', { name: 'Clinch proof board' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('img', { name: 'Clinch proof board' }),
    ).toHaveAccessibleDescription(/proof scenario—not actual moves/i);
    expect(container.querySelectorAll('[data-proof-stone]')).toHaveLength(1);
    expect(
      screen.getByLabelText('Hypothetical clinch proof scores'),
    ).toBeInTheDocument();
    expect(useAppStore.getState().log).toEqual(originalLog);

    await user.click(
      actionDock.getByRole('button', { name: /^Return to live board/ }),
    );
    expect(container.querySelectorAll('[data-proof-stone]')).toHaveLength(0);
    expect(
      screen.getByRole('group', { name: /Deltrel board with 4 rings/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('dialog', { name: 'Grace cannot be caught' }),
    ).not.toBeInTheDocument();

    await user.click(actionDock.getByRole('button', { name: 'End game' }));
    const endConfirmation = screen.getByRole('dialog', {
      name: 'End this clinched game?',
    });
    expect(
      within(endConfirmation).getByRole('button', { name: 'Keep playing' }),
    ).toHaveFocus();
    await user.click(
      within(endConfirmation).getByRole('button', { name: 'Keep playing' }),
    );
    expect(useAppStore.getState().earlyOutcome).toBeNull();
  });

  it('ends a clinched game without presenting a projected score as final', async () => {
    const user = userEvent.setup();
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: Array.from({ length: 49 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
    });
    render(<GameScreen />);

    const clinchDialog = screen.getByRole('dialog', {
      name: 'Grace cannot be caught',
    });
    await user.click(
      within(clinchDialog).getByRole('button', { name: 'End game now' }),
    );

    const result = screen.getByRole('dialog', { name: 'Game over' });
    expect(
      within(result).getByRole('heading', { name: 'Grace wins' }),
    ).toBeInTheDocument();
    expect(within(result).getByText(/no final score was recorded/i)).toBeInTheDocument();
    expect(screen.queryByText('Final score')).not.toBeInTheDocument();
    expect(useAppStore.getState().earlyOutcome).toEqual({
      reason: 'clinch',
      winner: 1,
      loser: 0,
      emptyNodes: 1,
    });

    await user.click(
      within(result).getByRole('button', { name: 'Review proof' }),
    );
    expect(
      screen.getByRole('img', { name: 'Clinch proof board' }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Rules' }));
    expect(
      screen.getByRole('dialog', { name: 'How to play Deltrel' }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Close rules' }));
    expect(
      within(screen.getByRole('navigation', { name: 'Game' })).getByRole(
        'button',
        { name: 'Result' },
      ),
    ).toBeInTheDocument();
  });

  it('keeps a normally completed board undoable during review', async () => {
    const user = userEvent.setup();
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: Array.from({ length: 50 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
    });
    const { container } = render(<GameScreen />);

    const result = screen.getByRole('dialog', { name: 'Game over' });
    await user.click(
      within(result).getByRole('button', { name: 'Review board' }),
    );
    const undoButton = within(
      container.querySelector('[data-action-dock]') as HTMLElement,
    ).getByRole('button', { name: 'Undo' });
    expect(undoButton).toBeEnabled();
    await user.click(undoButton);
    expect(useAppStore.getState().log).toHaveLength(49);
  });

  it('confirms a named resignation and applies the human-owned-side policy', async () => {
    const user = userEvent.setup();
    resetPlayingStore({ controllers: ['human', 'human'] });
    const { rerender } = render(<GameScreen />);

    await user.click(screen.getByRole('button', { name: 'Resign Ada' }));
    const resignation = screen.getByRole('dialog', { name: 'Resign Ada?' });
    expect(
      within(resignation).getByRole('button', { name: 'Keep playing' }),
    ).toHaveFocus();
    await user.click(
      within(resignation).getByRole('button', { name: 'Resign Ada' }),
    );
    expect(
      screen.getByRole('heading', { name: 'Grace wins' }),
    ).toBeInTheDocument();
    expect(useAppStore.getState().earlyOutcome).toEqual({
      reason: 'resignation',
      winner: 1,
      loser: 0,
    });

    resetPlayingStore({
      controllers: ['human', 'server'],
      log: [{ type: 'place', node: 0 }],
    });
    vi.mocked(requestServerAiDecision).mockReturnValue(
      deferred<DeltrelAiDecision>().promise,
    );
    rerender(<GameScreen />);
    expect(screen.getByRole('button', { name: 'Resign Ada' })).toBeInTheDocument();

    resetPlayingStore({ controllers: ['server', 'local'] });
    rerender(<GameScreen />);
    expect(
      screen.queryByRole('button', { name: /resign/i }),
    ).not.toBeInTheDocument();
  });

  it('suppresses completion guidance while a pie swap can recolor the opening', () => {
    resetPlayingStore({
      controllers: ['human', 'human'],
      config: {
        ...config,
        pieRule: true,
        playerNames: [...config.playerNames],
      },
      log: [{ type: 'place', node: 0 }],
    });
    render(<GameScreen />);

    expect(screen.getByText(/may steal the opening stone/i)).toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Completion bounds' }),
    ).not.toBeInTheDocument();
  });

  it('does not cross rescuable stones in projected opponent territory', () => {
    const board = getBoard(4);
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: ['E4', 'I1', 'G1'].map((label) => ({
        type: 'place' as const,
        node: parseLabel(board, label),
      })),
    });
    const { container } = render(<GameScreen />);

    expect(
      screen.getByText('Current scoring projection'),
    ).toBeInTheDocument();
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);
    expect(
      screen.getByRole('button', {
        name: /node E4, ada stone.*not currently part of a living network/i,
      }),
    ).toBeInTheDocument();
  });

  it('removes and restores a provably dead marker through undo and redo', async () => {
    const user = userEvent.setup();
    const board = getBoard(4);
    const dead = parseLabel(board, 'E1');
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: ['E1', 'F1', 'E2', 'D8', 'E9', 'D2', 'C1'].map(
        (label) => ({
          type: 'place' as const,
          node: parseLabel(board, label),
        }),
      ),
    });
    const { container } = render(<GameScreen />);

    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Redo' }));
    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).toBeInTheDocument();
  });
});
