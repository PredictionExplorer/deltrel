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
} from '@/lib/star/ai/capabilities';
import {
  DEFAULT_AI_SEARCH_SETTINGS,
  DEFAULT_CONFIG,
  useAppStore,
  type AppState,
} from '@/lib/store';
import { SetupScreen } from '../SetupScreen';

vi.mock('@/lib/star/ai/capabilities', async () => {
  const actual = await vi.importActual<typeof import('@/lib/star/ai/capabilities')>(
    '@/lib/star/ai/capabilities',
  );
  return { ...actual, checkAiCapabilities: vi.fn() };
});

const availableCapabilities: AiCapabilities = {
  server: { status: 'available', label: 'Server AI' },
  local: { status: 'available', label: 'Local AI' },
};

const developerCapabilities: AiCapabilities = {
  server: {
    status: 'available',
    label: 'Server AI',
    search: {
      default: { simulations: 512, maxConsidered: 16 },
      maximum: { simulations: 4_096, maxConsidered: 64 },
      presets: {
        quick: { simulations: 128, maxConsidered: 8 },
        strong: { simulations: 512, maxConsidered: 16 },
        maximum: { simulations: 4_096, maxConsidered: 64 },
      },
    },
  },
  local: {
    status: 'available',
    label: 'Local AI',
    search: {
      default: { simulations: 64, maxConsidered: 16 },
      maximum: { simulations: 1_024, maxConsidered: 128 },
      presets: {
        quick: { simulations: 64, maxConsidered: 8 },
        strong: { simulations: 64, maxConsidered: 16 },
        maximum: { simulations: 1_024, maxConsidered: 128 },
      },
    },
  },
};

const championCapabilities: AiCapabilities = {
  ...developerCapabilities,
  server: {
    ...developerCapabilities.server,
    status: 'available',
    champion: {
      role: 'champion',
      modelVersion: 'model-step-185554',
      modelStep: 185554,
      modelIdentity: `sha256-${'a'.repeat(64)}`,
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
  vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '0');
  resetStore();
  vi.mocked(checkAiCapabilities).mockReset();
  vi.mocked(checkAiCapabilities).mockResolvedValue(availableCapabilities);
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
});

describe('SetupScreen', () => {
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
    vi.mocked(checkAiCapabilities).mockResolvedValue(championCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against the champion' }));
    await user.click(screen.getByRole('button', {
      name: mode === 'classic' ? 'Classic *Star, 1 stone per turn' : /Double \*Star, 2 stones per turn/,
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
      controllers: ['human', 'server'],
      config: { mode, rings, pieRule, handicap, playerNames: ['You', 'Champion'] },
    });
  });

  it('shows the full champion identity and ordinary thinking-time choices', async () => {
    vi.mocked(checkAiCapabilities).mockResolvedValue(championCapabilities);
    const user = userEvent.setup();
    const { container } = render(<SetupScreen />);
    expect(await screen.findByText('Ready to play')).toBeInTheDocument();
    expect(screen.getByText(/step 185,554/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Play against the champion' }));
    expect(screen.getByRole('combobox', { name: 'Player 2 controller' })).toHaveValue('server');
    expect(screen.getByRole('button', { name: 'Strong champion search' })).toHaveAttribute('aria-pressed', 'true');
    for (const [label, simulations, maxConsidered] of [
      ['Quick', 128, 8], ['Maximum', 4096, 64], ['Strong', 512, 16],
    ] as const) {
      await user.click(screen.getByRole('button', { name: `${label} champion search` }));
      expect(useAppStore.getState().aiSearchSettings.server).toEqual({ simulations, maxConsidered });
    }
    await user.click(screen.getByText('Champion details'));
    expect(screen.getByText(`sha256-${'a'.repeat(64)}`)).toBeVisible();
    expect(screen.getByText('model-step-185554')).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('keeps pie and handicap openings mutually exclusive without dropping the champion', async () => {
    vi.mocked(checkAiCapabilities).mockResolvedValue(championCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against the champion' }));
    await user.click(screen.getByRole('button', { name: 'Full, 10 rings' }));
    await user.click(screen.getByRole('button', { name: 'Handicap opening' }));
    await user.click(screen.getByRole('radio', { name: '9 handicap stones' }));
    await user.click(screen.getByRole('button', { name: 'Even (pie) opening' }));
    expect(screen.queryByRole('radiogroup', { name: 'Handicap stones' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /begin the game/i }));
    expect(useAppStore.getState()).toMatchObject({
      controllers: ['human', 'server'],
      config: { pieRule: true, handicap: 1 },
    });
  });

  it('does not describe an unverified generic AI server as a ready champion', async () => {
    render(<SetupScreen />);
    expect(await screen.findByText('Not connected')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Play against the champion' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /begin the game/i })).toBeEnabled();
  });

  it('lets an ordinary player repair a saved budget above the server limit', async () => {
    resetStore({ controllers: ['human', 'server'], aiSearchSettings: {
      server: { simulations: 8192, maxConsidered: 64 },
      local: DEFAULT_AI_SEARCH_SETTINGS.local,
    } });
    vi.mocked(checkAiCapabilities).mockResolvedValue(championCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    expect(await screen.findByRole('alert')).toHaveTextContent('saved thinking-time setting');
    expect(screen.getByRole('button', { name: /begin the game/i })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Use recommended thinking time' }));
    expect(screen.getByRole('button', { name: /begin the game/i })).toBeEnabled();
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({ simulations: 512, maxConsidered: 16 });
  });

  it('synchronizes advanced budget drafts after choosing ordinary champion effort', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    vi.mocked(checkAiCapabilities).mockResolvedValue(championCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);
    await user.click(await screen.findByRole('button', { name: 'Play against the champion' }));
    await user.click(screen.getByText('Engine developer settings'));
    await user.click(screen.getByText('Advanced search budget'));
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Simulations' }), { target: { value: '4097' } });
    await waitFor(() => expect(screen.getByRole('button', { name: /begin the game/i })).toBeDisabled());
    await user.click(screen.getByRole('button', { name: 'Play against the champion' }));
    expect(screen.getByRole('button', { name: /begin the game/i })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Quick champion search' }));
    expect(screen.getByRole('spinbutton', { name: 'Simulations' })).toHaveValue(128);
    await waitFor(() => expect(screen.getByRole('button', { name: /begin the game/i })).toBeEnabled());
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

  it('shows controllers for every variant and keeps them across the pie rule', async () => {
    const user = userEvent.setup();
    render(<SetupScreen />);

    const classic = screen.getByRole('button', { name: 'Classic *Star, 1 stone per turn' });
    const double = screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i });
    expect(classic).toHaveAttribute('aria-pressed', 'true');
    expect(
      screen.getByRole('combobox', { name: 'Player 1 controller' }),
    ).toBeInTheDocument();

    await user.click(double);
    const playerOneController = await screen.findByRole('combobox', {
      name: 'Player 1 controller',
    });
    expect(double).toHaveAttribute('aria-pressed', 'true');

    await waitFor(() =>
      expect(
        within(playerOneController).getByRole('option', { name: 'Server AI' }),
      ).toBeEnabled(),
    );
    await user.selectOptions(playerOneController, 'server');
    expect(playerOneController).toHaveValue('server');
    expect(
      screen.queryByText('Engine developer settings'),
    ).not.toBeInTheDocument();

    const pieRule = screen.getByRole('button', { name: 'Even (pie) opening' });
    await user.click(pieRule);
    expect(
      screen.getByRole('combobox', { name: 'Player 1 controller' }),
    ).toHaveValue('server');

    await user.click(screen.getByRole('button', { name: 'Full, 10 rings' }));
    await user.click(screen.getByRole('button', { name: 'Handicap opening' }));
    expect(
      screen.getByRole('combobox', { name: 'Player 1 controller' }),
    ).toHaveValue('server');
  });

  it('blocks setup until a selected controller is ready and supports rechecking', async () => {
    resetStore({
      config: {
        rings: 4,
        mode: 'double',
        pieRule: false,
        playerNames: ['Ada', 'Grace'],
      },
      controllers: ['server', 'human'],
    });
    vi.mocked(checkAiCapabilities).mockResolvedValueOnce({
      server: {
        status: 'unavailable',
        label: 'Server AI',
        code: 'server_unavailable',
        reason: 'Server AI is offline.',
        retryable: true,
      },
      local: { status: 'available', label: 'Local AI' },
    });

    const user = userEvent.setup();
    render(<SetupScreen />);

    const begin = screen.getByRole('button', { name: /begin the game/i });
    expect(begin).toBeDisabled();
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Server AI: Server AI is offline.',
    );
    expect(begin).toBeDisabled();

    await user.click(
      screen.getByRole('button', { name: /check ai availability again/i }),
    );
    await waitFor(() => expect(begin).toBeEnabled());
    expect(checkAiCapabilities).toHaveBeenCalledTimes(2);
  });

  it('offers exact validated developer budgets from capability presets', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    vi.mocked(checkAiCapabilities).mockResolvedValue(developerCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);

    await user.click(
      screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i }),
    );
    expect(
      screen.queryByText('Engine developer settings'),
    ).not.toBeInTheDocument();

    const playerOneController = screen.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await waitFor(() =>
      expect(
        within(playerOneController).getByRole('option', {
          name: 'Mac engine — current champion',
        }),
      ).toBeEnabled(),
    );
    expect(
      within(playerOneController).getByRole('option', {
        name: 'Browser AI — lightweight',
      }),
    ).toBeEnabled();
    await user.selectOptions(playerOneController, 'server');

    const settingsSummary = screen.getByText('Engine developer settings');
    const settingsDetails = settingsSummary.closest('details');
    expect(settingsDetails).not.toHaveAttribute('open');
    await user.click(settingsSummary);

    const quick = screen.getByRole('button', {
      name: /Quick, 128 simulations.*8 candidates/i,
    });
    const strong = screen.getByRole('button', {
      name: /Strong, 512 simulations.*16 candidates/i,
    });
    const maximum = screen.getByRole('button', {
      name: /Maximum, 4,096 simulations.*64 candidates/i,
    });
    expect(strong).toHaveAttribute('aria-pressed', 'true');
    expect(quick).toHaveAttribute('aria-pressed', 'false');
    expect(maximum).toHaveAttribute('aria-pressed', 'false');

    await user.click(quick);
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({
      simulations: 128,
      maxConsidered: 8,
    });
    expect(screen.getByText(/Runs exactly 128 simulations/i)).toBeInTheDocument();

    await user.click(screen.getByText('Advanced search budget'));
    const simulations = screen.getByRole('spinbutton', {
      name: 'Simulations',
    });
    fireEvent.change(simulations, { target: { value: '4097' } });
    expect(simulations).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Simulations must be a whole number from 1 to 4,096.',
    );
    expect(useAppStore.getState().aiSearchSettings.server.simulations).toBe(128);
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /begin the game/i }),
      ).toBeDisabled(),
    );

    fireEvent.change(simulations, { target: { value: '777' } });
    await waitFor(() =>
      expect(useAppStore.getState().aiSearchSettings.server).toEqual({
        simulations: 777,
        maxConsidered: 8,
      }),
    );
    expect(simulations).toHaveAttribute('aria-invalid', 'false');
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /begin the game/i }),
      ).toBeEnabled(),
    );
  });

  it('keeps developer engine controls accessible when expanded', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    vi.mocked(checkAiCapabilities).mockResolvedValue(developerCapabilities);
    const user = userEvent.setup();
    const { container } = render(<SetupScreen />);

    await user.click(
      screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i }),
    );
    const controller = screen.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await waitFor(() =>
      expect(
        within(controller).getByRole('option', {
          name: 'Mac engine — current champion',
        }),
      ).toBeEnabled(),
    );
    await user.selectOptions(controller, 'server');
    await user.click(screen.getByText('Engine developer settings'));
    await user.click(screen.getByText('Advanced search budget'));

    expect((await axe(container)).violations).toEqual([]);
  });

  it('has no detectable accessibility violations', async () => {
    const { container } = render(<SetupScreen />);
    await waitFor(() => expect(checkAiCapabilities).toHaveBeenCalledOnce());

    expect((await axe(container)).violations).toEqual([]);
  });
});
