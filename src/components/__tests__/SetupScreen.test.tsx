import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  checkAiCapabilities,
  type AiCapabilities,
} from '@/lib/deltrel/ai/capabilities';
import {
  DEFAULT_AI_SEARCH_SETTINGS,
  DEFAULT_CONFIG,
  useAppStore,
  type AppState,
} from '@/lib/store';
import { SetupScreen } from '../SetupScreen';
import { prepareLocalAi } from '@/lib/deltrel/ai/local-client';
import { publishLocalAiStatus } from '@/lib/deltrel/ai/local-ai-status';

vi.mock('@/lib/deltrel/ai/local-client', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/local-client')>('@/lib/deltrel/ai/local-client'),
  prepareLocalAi: vi.fn(),
}));

const browserReady = { modelVersion: 'browser-champion', bytes: 37_577_312, backend: 'wasm' as const, cached: false };

vi.mock('@/lib/deltrel/ai/capabilities', async () => {
  const actual = await vi.importActual<typeof import('@/lib/deltrel/ai/capabilities')>(
    '@/lib/deltrel/ai/capabilities',
  );
  return { ...actual, checkAiCapabilities: vi.fn() };
});

const availableCapabilities: AiCapabilities = {
  // A legacy native service must never reappear in player choices.
  server: { status: 'available', label: 'Server AI' },
  local: {
    status: 'available', label: 'AI',
    browserModel: { modelVersion: browserReady.modelVersion, bytes: browserReady.bytes, sha256: 'a'.repeat(64) },
    search: {
      default: { simulations: 512, maxConsidered: 16 },
      maximum: { simulations: 4294967295, maxConsidered: 4294967295 },
      presets: {},
    },
  },
};

function resetStore(overrides: Partial<AppState> = {}) {
  const config = overrides.config ?? DEFAULT_CONFIG;
  useAppStore.setState({
    phase: 'setup',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: ['human', 'human'],
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

beforeEach(() => {
  localStorage.clear();
  vi.stubEnv('NEXT_PUBLIC_DELTREL_AI_DEVTOOLS', '0');
  resetStore();
  vi.mocked(checkAiCapabilities).mockReset();
  vi.mocked(checkAiCapabilities).mockResolvedValue(availableCapabilities);
  publishLocalAiStatus({ phase: 'ready', info: browserReady });
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

describe('SetupScreen', () => {
  it.each([
    ['human', 'human'], ['local', 'local'], ['local', 'local'],
  ] as const)('uses neutral names for legacy quick-start preferences with %s and %s players', async (first, second) => {
    resetStore({ config: { ...DEFAULT_CONFIG, playerNames: ['You', 'Champion'] }, controllers: [first, second] });
    render(<SetupScreen />);
    await waitFor(() => expect(checkAiCapabilities).toHaveBeenCalledOnce());
    expect(screen.getByRole('textbox', { name: 'Player 1 name' })).toHaveValue('Player 1');
    expect(screen.getByRole('textbox', { name: 'Player 2 name' })).toHaveValue('Player 2');
    expect(useAppStore.getState().config.playerNames).toEqual(['You', 'Champion']);
  });


  it('keeps neutral names when switching a prepared quick start to two humans or two computers', async () => {
    publishLocalAiStatus({ phase: 'ready', info: browserReady });
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against AI' }));
    for (const controller of ['human', 'local']) {
      await user.selectOptions(screen.getByRole('combobox', { name: 'Player 1 controller' }), controller);
      await user.selectOptions(screen.getByRole('combobox', { name: 'Player 2 controller' }), controller);
      expect(screen.getByRole('textbox', { name: 'Player 1 name' })).toHaveValue('Player 1');
      expect(screen.getByRole('textbox', { name: 'Player 2 name' })).toHaveValue('Player 2');
    }
    await user.click(screen.getByRole('button', { name: 'Begin the game' }));
    expect(useAppStore.getState().config.playerNames).toEqual(['Player 1', 'Player 2']);
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
  });


  it('supports the complete radio keyboard pattern for handicap selection', async () => {
    resetStore({ config: { ...DEFAULT_CONFIG, rings: 10, pieRule: false, handicap: 2 } });
    const user = userEvent.setup();
    render(<SetupScreen />);
    const two = screen.getByRole('radio', { name: '2 handicap stones' });
    two.focus();
    await user.keyboard('{End}');
    const nine = screen.getByRole('radio', { name: '9 handicap stones' });
    expect(nine).toHaveFocus();
    expect(nine).toHaveAttribute('aria-checked', 'true');
    expect(nine).toHaveAttribute('tabindex', '0');
    expect(two).toHaveAttribute('tabindex', '-1');
    await user.keyboard('{ArrowRight}');
    expect(two).toHaveFocus();
    await user.keyboard('{ArrowUp}');
    expect(nine).toHaveFocus();
    await user.keyboard('{Home}');
    expect(two).toHaveFocus();
    expect(two).toHaveAttribute('aria-checked', 'true');
  });


  it('reuses an already prepared browser model without another preparation request', async () => {
    publishLocalAiStatus({ phase: 'ready', info: { ...browserReady, cached: true } });
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against AI' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Begin the game' })).toBeEnabled());
    expect(prepareLocalAi).not.toHaveBeenCalled();
    expect(screen.getByText('Loaded from this browser’s saved model.')).toBeInTheDocument();
  });


  it.each(['classic', 'double'] as const)(
    'offers only pie even games on smaller boards in %s mode',
    async (mode) => {
      resetStore({ config: { ...DEFAULT_CONFIG, mode, pieRule: false } });
      const user = userEvent.setup();
      render(<SetupScreen />);
      expect(screen.queryByRole('button', { name: 'Standard opening' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Even (pie) opening' })).toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByRole('button', { name: 'Handicap opening' })).toBeDisabled();
      await user.click(screen.getByRole('button', { name: /begin the game/i }));
      expect(useAppStore.getState().config).toMatchObject({ mode, pieRule: true, handicap: 1 });
    },
  );


  it.each(['preset', 'slider'] as const)(
    'resets handicap to an even game when a %s selects a smaller board',
    async (control) => {
      resetStore({ config: { ...DEFAULT_CONFIG, rings: 10, pieRule: false, handicap: 9 } });
      const user = userEvent.setup();
      render(<SetupScreen />);
      expect(screen.getByRole('radio', { name: '9 handicap stones' })).toHaveAttribute('aria-checked', 'true');
      if (control === 'preset') {
        await user.click(screen.getByRole('button', { name: 'Mini, 4 rings' }));
      } else {
        fireEvent.change(screen.getByRole('slider', { name: 'Custom' }), { target: { value: '4' } });
      }
      expect(screen.getByRole('button', { name: 'Handicap opening' })).toBeDisabled();
      expect(screen.queryByRole('radiogroup', { name: 'Handicap stones' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Even (pie) opening' })).toHaveAttribute('aria-pressed', 'true');
      await user.click(screen.getByRole('button', { name: /begin the game/i }));
      expect(useAppStore.getState().config).toMatchObject({ rings: 4, pieRule: true, handicap: 1 });
    },
  );


  it('repairs old small-board handicap preferences without mutating the saved game', async () => {
    const saved = { ...DEFAULT_CONFIG, rings: 4, pieRule: false, handicap: 9 };
    resetStore({ config: saved });
    const user = userEvent.setup();
    render(<SetupScreen />);
    expect(screen.getByRole('button', { name: 'Even (pie) opening' })).toHaveAttribute('aria-pressed', 'true');
    expect(useAppStore.getState().config).toEqual(saved);
    await user.click(screen.getByRole('button', { name: /begin the game/i }));
    expect(useAppStore.getState().config).toMatchObject({ rings: 4, pieRule: true, handicap: 1 });
    expect(saved).toMatchObject({ pieRule: false, handicap: 9 });
  });


  it.each([
    ['classic', 'Even (pie)', true, 1],
    ['classic', 'Handicap', false, 9],
    ['double', 'Even (pie)', true, 1],
    ['double', 'Handicap', false, 9],
  ] as const)('starts %s with %s opening against the champion', async (mode, opening, pieRule, handicap) => {
    vi.mocked(checkAiCapabilities).mockResolvedValue(availableCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against AI' }));
    await user.click(screen.getByRole('button', {
      name: mode === 'classic' ? 'Classic Deltrel, 1 stone per turn' : /Double Deltrel, 2 stones per turn/,
    }));
    if (handicap > 1) {
      await user.click(screen.getByRole('button', { name: 'Full, 10 rings' }));
    }
    await user.click(screen.getByRole('button', { name: `${opening} opening` }));
    if (handicap > 1) {
      await user.click(screen.getByRole('radio', { name: `${handicap} handicap stones` }));
    }
    const rings = handicap > 1 ? 10 : 6;
    const summary = `${mode === 'classic' ? 'Classic' : 'Double'} · ${handicap > 1 ? '9-stone handicap' : opening} · ${rings} rings`;
    expect(screen.getByText(summary)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /begin the game/i }));
    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      controllers: ['human', 'local'],
      config: { mode, rings, pieRule, handicap, playerNames: ['Player 1', 'Player 2'] },
    });
  });


  it('keeps pie and handicap openings mutually exclusive without dropping the champion', async () => {
    vi.mocked(checkAiCapabilities).mockResolvedValue(availableCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against AI' }));
    await user.click(screen.getByRole('button', { name: 'Full, 10 rings' }));
    await user.click(screen.getByRole('button', { name: 'Handicap opening' }));
    await user.click(screen.getByRole('radio', { name: '9 handicap stones' }));
    await user.click(screen.getByRole('button', { name: 'Even (pie) opening' }));
    expect(screen.queryByRole('radiogroup', { name: 'Handicap stones' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /begin the game/i }));
    expect(useAppStore.getState()).toMatchObject({
      controllers: ['human', 'local'],
      config: { pieRule: true, handicap: 1 },
    });
  });


  it('stores trimmed fallback player names when fields are blank', async () => {
    const user = userEvent.setup();
    render(<SetupScreen />);

    const playerOne = screen.getByRole('textbox', { name: 'Player 1 name' });
    const playerTwo = screen.getByRole('textbox', { name: 'Player 2 name' });
    expect(playerOne).toHaveValue('Player 1');
    expect(playerTwo).toHaveValue('Player 2');
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'min',
      '4',
    );
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'max',
      '10',
    );
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'step',
      '2',
    );
    fireEvent.change(screen.getByRole('slider', { name: 'Custom' }), {
      target: { value: '9' },
    });
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveValue('6');

    await user.clear(playerOne);
    await user.clear(playerTwo);
    await user.click(screen.getByRole('button', { name: /begin the game/i }));

    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      controllers: ['human', 'human'],
      config: { playerNames: ['Player 1', 'Player 2'] },
    });
  });


  it('offers only Human and AI for each seat and preserves names when migrating a native game', async () => {
    resetStore({ controllers: ['server', 'local'], config: { ...DEFAULT_CONFIG, playerNames: ['Drift', 'Tide'] } });
    const user = userEvent.setup();
    render(<SetupScreen />);
    await waitFor(() => expect(checkAiCapabilities).toHaveBeenCalledOnce());
    for (const player of [1, 2]) {
      const controller = screen.getByRole('combobox', { name: `Player ${player} controller` });
      expect(within(controller).getAllByRole('option').map(option => option.textContent)).toEqual(['Human', 'AI']);
      expect(controller).toHaveValue('local');
    }
    expect(screen.queryByText('Current champion')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'AI selected' }));
    await user.click(screen.getByRole('button', { name: 'Begin the game' }));
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
    expect(useAppStore.getState().config.playerNames).toEqual(['Drift', 'Tide']);
  });

  it('prepares the published model automatically without changing either player or their search budget', async () => {
    resetStore({ controllers: ['local', 'local'], config: { ...DEFAULT_CONFIG, playerNames: ['Drift', 'Tide'] }, aiSearchSettings: { ...DEFAULT_AI_SEARCH_SETTINGS, local: { simulations: 9999, maxConsidered: 127 } } });
    publishLocalAiStatus({ phase: 'idle' });
    const user = userEvent.setup();
    render(<SetupScreen />);
    expect(screen.getByRole('button', { name: 'Begin the game' })).toBeDisabled();
    await waitFor(() => expect(prepareLocalAi).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Begin the game' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Begin the game' }));
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
    expect(useAppStore.getState().config.playerNames).toEqual(['Drift', 'Tide']);
    expect(useAppStore.getState().aiSearchSettings.local).toEqual({ simulations: 9999, maxConsidered: 127 });
  });

  it('uses champion effort by default and exposes its published version and exact presets', async () => {
    const user = userEvent.setup();
    const { container } = render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against AI' }));
    expect(screen.getByRole('button', { name: 'Standard AI strength' })).toHaveAttribute('aria-pressed', 'true');
    for (const [label, simulations, maxConsidered] of [['Quick', 128, 8], ['Deep', 4096, 64], ['Standard', 512, 16]] as const) {
      await user.click(screen.getByRole('button', { name: `${label} AI strength` }));
      expect(useAppStore.getState().aiSearchSettings.local).toEqual({ simulations, maxConsidered });
    }
    await user.click(screen.getByText('AI details'));
    expect(screen.getByText(browserReady.modelVersion)).toBeVisible();
    await user.click(screen.getByText('Custom search budget'));
    expect((await axe(container)).violations).toEqual([]);
  });

  it('blocks AI while unavailable and allows a human game and an availability retry', async () => {
    resetStore({ controllers: ['local', 'human'] });
    vi.mocked(checkAiCapabilities).mockResolvedValueOnce({ ...availableCapabilities, local: { status: 'unavailable', label: 'AI', code: 'offline', reason: 'The model is unavailable.', retryable: true } });
    const user = userEvent.setup();
    render(<SetupScreen />);
    expect(await screen.findByRole('alert')).toHaveTextContent('AI: The model is unavailable.');
    const begin = screen.getByRole('button', { name: 'Begin the game' });
    expect(begin).toBeDisabled();
    await user.selectOptions(screen.getByRole('combobox', { name: 'Player 1 controller' }), 'human');
    expect(begin).toBeEnabled();
    await user.click(screen.getByRole('button', { name: 'Check AI availability again' }));
    await waitFor(() => expect(checkAiCapabilities).toHaveBeenCalledTimes(2));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Player 1 controller' }), 'local');
    expect(begin).toBeEnabled();
  });

  it('cancels an automatic download and leaves a deliberate retry available', async () => {
    resetStore({ controllers: ['human', 'local'] });
    publishLocalAiStatus({ phase: 'idle' });
    vi.mocked(prepareLocalAi).mockImplementationOnce(({ signal } = {}) => new Promise((_, reject) => {
      publishLocalAiStatus({ phase: 'downloading', loadedBytes: 1, totalBytes: 10, modelVersion: 'browser-champion', cached: false });
      signal?.addEventListener('abort', () => {
        publishLocalAiStatus({ phase: 'idle' });
        reject(new Error('cancelled'));
      }, { once: true });
    }));
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Cancel AI preparation' }));
    expect(screen.getByRole('button', { name: 'Begin the game' })).toBeDisabled();
    expect(useAppStore.getState().phase).toBe('setup');
    await user.click(screen.getByRole('button', { name: 'Retry AI' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Begin the game' })).toBeEnabled());
    expect(prepareLocalAi).toHaveBeenCalledTimes(2);
  });

});
