import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { checkAiCapabilities, type AiCapabilities } from '@/lib/deltrel/ai/capabilities';
import type { DeltrelAiDecision } from '@/lib/deltrel/ai/decision';
import { DeltrelAiError } from '@/lib/deltrel/ai/errors';
import { requestLocalAiDecision } from '@/lib/deltrel/ai/local-client';
import { publishLocalAiStatus } from '@/lib/deltrel/ai/local-ai-status';
import { makeAiResponse, type DeltrelAiRequest } from '@/lib/deltrel/ai/protocol';
import type { GameConfig, Mode } from '@/lib/deltrel/game';
import { useAppStore } from '@/lib/store';
import { GameScreen } from '../GameScreen';
import { analysisFixture } from './engineEstimateFixtures';

vi.mock('@/lib/deltrel/ai/local-client', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/local-client')>('@/lib/deltrel/ai/local-client'),
  requestLocalAiDecision: vi.fn(),
}));
vi.mock('@/lib/deltrel/ai/capabilities', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/capabilities')>('@/lib/deltrel/ai/capabilities'),
  checkAiCapabilities: vi.fn(),
}));

const capabilities: AiCapabilities = {
  server: { status: 'unavailable', label: 'AI', code: 'browser_only', reason: 'AI runs in your browser.', retryable: false },
  local: { status: 'available', label: 'AI' },
};
const config: GameConfig = { rings: 4, mode: 'classic', pieRule: true, handicap: 1, playerNames: ['Ada', 'Grace'] };

function deferred() {
  let resolve!: (decision: DeltrelAiDecision) => void;
  const promise = new Promise<DeltrelAiDecision>(done => { resolve = done; });
  return { promise, resolve };
}

function decisionFor(request: DeltrelAiRequest): DeltrelAiDecision {
  const node = request.legalActions.find(node => node >= 0 && node < request.state.stones.length)!;
  const action = { type: 'place' as const, node };
  return {
    response: makeAiResponse(request, action),
    analysis: analysisFixture({
      perspective: request.state.toMove, stateHash: request.stateHash,
      rootActions: [action], rootVisits: [8], rootPolicy: [1], rootQ: [0.5],
      simulations: 8, maxConsidered: 1, networkOutput: null,
    }),
  };
}

function expectInsightsHidden() {
  expect(screen.queryByRole('region', { name: 'Engine estimate', hidden: true })).not.toBeInTheDocument();
  expect(screen.queryByRole('progressbar', { name: 'AI search progress', hidden: true })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Analyze position|Search candidates|Raw engine output|All network outputs|Final-count forecasts/, hidden: true })).not.toBeInTheDocument();
  expect(screen.queryByRole('textbox', { name: 'Raw engine output', hidden: true })).not.toBeInTheDocument();
  expect(screen.queryByText('75.0%')).not.toBeInTheDocument();
  expect(screen.queryByText(/Inspect the forecasts/)).not.toBeInTheDocument();
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(requestLocalAiDecision).mockReset();
  vi.mocked(checkAiCapabilities).mockResolvedValue(capabilities);
  publishLocalAiStatus({ phase: 'ready', info: { modelVersion: 'champion-test', bytes: 72_474_137, backend: 'wasm', cached: true } });
  useAppStore.getState().startGame(config, ['human', 'local']);
});
afterEach(() => { cleanup(); localStorage.clear(); });

const matches = (['classic', 'double'] as Mode[]).flatMap(mode =>
  ([0, 1] as const).flatMap(aiSeat => [
    ...[4, 6, 8, 10].map(rings => ({ mode, aiSeat, rings, handicap: 1, pieRule: true })),
    ...Array.from({ length: 8 }, (_, i) => ({ mode, aiSeat, rings: 10, handicap: i + 2, pieRule: false })),
  ]),
);

describe('human-versus-AI insight privacy', () => {
  it.each(matches)('hides all engine details in $mode, $rings rings, handicap $handicap, AI seat $aiSeat', async match => {
    const { aiSeat, ...rules } = match;
    useAppStore.getState().startGame({ ...config, ...rules }, aiSeat === 0 ? ['local', 'human'] : ['human', 'local']);
    const openingCount = aiSeat === 1 ? rules.handicap : 0;
    for (let node = 0; node < openingCount; node++) useAppStore.getState().act({ type: 'place', node });
    const first = deferred();
    vi.mocked(requestLocalAiDecision).mockReturnValue(deferred().promise).mockReturnValueOnce(first.promise);
    render(<GameScreen />);
    expectInsightsHidden();
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    const [request, options] = vi.mocked(requestLocalAiDecision).mock.calls[0];
    expect(request.state.toMove).toBe(aiSeat);
    act(() => options!.onSearchProgress?.({ completedSimulations: 256, totalSimulations: 512 }));
    expectInsightsHidden();
    await act(async () => first.resolve(decisionFor(request)));
    await waitFor(() => expect(useAppStore.getState().log).toHaveLength(openingCount + 1));
    expectInsightsHidden();
    act(() => useAppStore.getState().pauseAi());
    expectInsightsHidden();
  });

  it('keeps insight controls out of human turns, history, undo/redo, and ended-game review', async () => {
    const user = userEvent.setup();
    const first = deferred();
    vi.mocked(requestLocalAiDecision).mockReturnValue(first.promise);
    render(<GameScreen />);
    expect(screen.getByRole('region', { name: 'Current player scores' })).toBeInTheDocument();
    expectInsightsHidden();
    expect(requestLocalAiDecision).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /^Node G4, empty/ }));
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    await act(async () => first.resolve(decisionFor(vi.mocked(requestLocalAiDecision).mock.calls[0][0])));
    await waitFor(() => expect(useAppStore.getState().log).toHaveLength(2));
    await user.click(screen.getByRole('button', { name: 'Jump to the empty board' }));
    expectInsightsHidden();
    act(() => useAppStore.getState().undo());
    expectInsightsHidden();
    act(() => useAppStore.getState().redo());
    expectInsightsHidden();
    act(() => useAppStore.getState().resign(0));
    expectInsightsHidden();
    act(() => useAppStore.getState().setReviewing(true));
    expectInsightsHidden();
    expect(screen.getByRole('heading', { name: 'Ada' })).toBeInTheDocument();
    expect(requestLocalAiDecision).toHaveBeenCalledOnce();
  });

  it('keeps forecasts hidden when an AI request fails', async () => {
    useAppStore.getState().startGame(config, ['local', 'human']);
    vi.mocked(requestLocalAiDecision).mockRejectedValue(new DeltrelAiError('network', 'AI is offline.', true));
    render(<GameScreen />);
    expect(await screen.findByRole('alert')).toHaveTextContent('AI is offline.');
    expectInsightsHidden();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled();
  });

  it('preserves pause, resume and human takeover without revealing earlier engine results', async () => {
    useAppStore.getState().startGame(config, ['local', 'human']);
    // Controller-derived protection must remain sticky even with a stale flag.
    useAppStore.setState({ aiInsightsHidden: false });
    const user = userEvent.setup();
    const first = deferred();
    vi.mocked(requestLocalAiDecision).mockReturnValue(deferred().promise).mockReturnValueOnce(first.promise);
    const view = render(<GameScreen />);
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    const [request, options] = vi.mocked(requestLocalAiDecision).mock.calls[0];
    await user.click(screen.getByRole('button', { name: 'Pause AI' }));
    expect(options!.signal!.aborted).toBe(true);
    expectInsightsHidden();
    await act(async () => first.resolve(decisionFor(request)));
    expect(useAppStore.getState().log).toEqual([]);
    await user.click(screen.getByRole('button', { name: 'Resume AI' }));
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledTimes(2));
    await user.click(screen.getByRole('button', { name: 'Pause AI' }));
    await user.click(screen.getByRole('button', { name: 'Take over as human' }));
    expect(useAppStore.getState().controllers).toEqual(['human', 'human']);
    expectInsightsHidden();
    view.unmount();
    render(<GameScreen />);
    expectInsightsHidden();
  });

  it('immediately removes previously displayed self-play forecasts on human takeover', async () => {
    useAppStore.getState().startGame(config, ['local', 'local']);
    const first = deferred();
    const next = deferred();
    vi.mocked(requestLocalAiDecision).mockReturnValue(next.promise).mockReturnValueOnce(first.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    await act(async () => first.resolve(decisionFor(vi.mocked(requestLocalAiDecision).mock.calls[0][0])));
    await waitFor(() => expect(screen.getByText('75.0%')).toBeVisible());
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledTimes(2));
    await user.click(screen.getByRole('button', { name: 'Pause AI' }));
    await user.click(screen.getByRole('button', { name: 'Take over as human' }));
    expectInsightsHidden();
    const [request, options] = vi.mocked(requestLocalAiDecision).mock.calls[1];
    act(() => options!.onSearchProgress?.({ completedSimulations: 512, totalSimulations: 512 }));
    await act(async () => next.resolve(decisionFor(request)));
    expectInsightsHidden();
    expect(useAppStore.getState().log).toHaveLength(1);
  });

  it('cancels read-only analysis when a human-versus-AI match begins', async () => {
    useAppStore.getState().startGame(config, ['human', 'human']);
    const pending = deferred();
    vi.mocked(requestLocalAiDecision).mockReturnValue(pending.promise);
    const user = userEvent.setup();
    render(<GameScreen />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze position' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    await waitFor(() => expect(requestLocalAiDecision).toHaveBeenCalledOnce());
    const [request, options] = vi.mocked(requestLocalAiDecision).mock.calls[0];
    act(() => useAppStore.getState().setPlayerController(1, 'local'));
    expect(options!.signal!.aborted).toBe(true);
    expectInsightsHidden();
    await act(async () => pending.resolve(decisionFor(request)));
    expectInsightsHidden();
    expect(useAppStore.getState().log).toEqual([]);
  });
});
