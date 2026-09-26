import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import fc from 'fast-check';
import {
  APP_STORE_VERSION,
  DEFAULT_AI_SEARCH_SETTINGS,
  DEFAULT_CONFIG,
  migratePersistedState,
  normalizeAiSearchSettings,
  normalizeGameConfig,
  parseAiSearchBudget,
  parseGameAction,
  sanitizePersistedState,
  useAppStore,
} from '../../store';
import { replay, type GameAction, type GameConfig } from '../game';
import { canShowAiInsights, type PlayerControllers } from '../ai/controllers';

const double: GameConfig = {
  rings: 6,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};
const mini: GameConfig = {
  ...double,
  rings: 4,
  playerNames: ['Ada', 'Grace'],
};
const clinchedLog = Array.from({ length: 49 }, (_, node) => ({
  type: 'place' as const,
  node,
}));

beforeEach(() => {
  useAppStore.getState().setSelfPlayAccess(true);
});

afterEach(() => {
  useAppStore.setState({
    phase: 'setup',
    selfPlayAllowed: false,
    selfPlayAccessReady: false,
    config: DEFAULT_CONFIG,
    controllers: ['human', 'human'],
    aiInsightsHidden: false,
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
  });
});

describe('persisted app-state validation', () => {
  it.each([
    { controllers: ['human', 'local'], local: { simulations: 8, maxConsidered: 4 }, expected: { simulations: 544, maxConsidered: 16 } },
    { controllers: ['server', 'human'], local: { simulations: 8, maxConsidered: 4 }, expected: { simulations: 544, maxConsidered: 16 } },
    { controllers: ['server', 'local'], local: { simulations: 9999, maxConsidered: 127 }, expected: { simulations: 4096, maxConsidered: 64 } },
  ])('migrates version 6 controllers and effort without losing a paused game: $controllers', ({ controllers, local, expected }) => {
    const saved = {
      phase: 'playing', config: mini, controllers,
      aiSearchSettings: { server: { simulations: 777, maxConsidered: 21 }, local },
      aiPaused: true,
      log: [{ type: 'place', node: 0 }], redoStack: [{ type: 'place', node: 1 }],
    };
    const result = migratePersistedState(saved, 6);
    expect(result).toMatchObject({
      phase: 'playing', config: mini, aiPaused: true,
      log: saved.log, redoStack: saved.redoStack,
      controllers,
      aiSearchSettings: { server: saved.aiSearchSettings.server, local: expected },
    });
    expect(saved.aiSearchSettings.local).toEqual(local);
  });

  it.each([5, 6, 7, APP_STORE_VERSION])('preserves independent cloud and browser effort from version %i', (version) => {
    const saved = {
      phase: 'playing', config: mini, controllers: ['human', 'server'],
      aiSearchSettings: { server: { simulations: 4096, maxConsidered: 64 }, local: { simulations: 8, maxConsidered: 4 } },
      aiPaused: true, log: [{ type: 'place', node: 0 }], redoStack: [{ type: 'place', node: 1 }],
    };
    expect(migratePersistedState(saved, version)).toMatchObject({
      ...saved,
      aiSearchSettings: { server: saved.aiSearchSettings.server, local: { simulations: 544, maxConsidered: 16 } },
    });
  });

  it.each([7, 8, APP_STORE_VERSION])('upgrades old low search budgets from version %i to Standard', (version) => {
    const local = { simulations: 128, maxConsidered: 8 };
    expect(migratePersistedState({
      phase: 'playing', config: mini, controllers: ['human', 'local'],
      aiSearchSettings: { local }, aiPaused: true, log: [], redoStack: [],
    }, version).aiSearchSettings.local).toEqual({ simulations: 544, maxConsidered: 16 });
  });

  it('rejects saved histories longer than any legal game', () => {
    expect(sanitizePersistedState({
      phase: 'playing', config: mini, controllers: ['local', 'local'],
      log: Array.from({ length: 51 }, (_, node) => ({ type: 'place', node })), redoStack: [],
    })).toMatchObject({ phase: 'setup', log: [], redoStack: [] });
  });

  it.each(['log', 'redoStack'] as const)('rejects sparse and null-filled saved %s arrays', (field) => {
    for (const history of [new Array(3), [null], [undefined]]) {
      expect(sanitizePersistedState({
        phase: 'playing', config: mini, controllers: ['local', 'local'],
        log: [], redoStack: [], [field]: history,
      })).toMatchObject({ phase: 'setup', log: [], redoStack: [] });
    }
  });

  it('rehydrates only replayable strict action logs and redo history', () => {
    expect(
      sanitizePersistedState({
        phase: 'playing',
        config: double,
        controllers: ['local', 'local'],
        aiSearchSettings: {
          server: { simulations: 128, maxConsidered: 8 },
          local: { simulations: 32, maxConsidered: 4 },
        },
        aiPaused: true,
        log: [{ type: 'place', node: 0 }],
        redoStack: [{ type: 'place', node: 1 }],
      }),
    ).toMatchObject({
      phase: 'playing',
      controllers: ['local', 'local'],
      aiSearchSettings: {
        server: { simulations: 128, maxConsidered: 8 },
        local: { simulations: 544, maxConsidered: 16 },
      },
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
    });
  });

  it('returns to setup and clears malformed or illegal history', () => {
    const malformed = sanitizePersistedState({
      phase: 'playing',
      config: double,
      controllers: ['local', 'local'],
      log: [{ type: 'pass', unexpected: true }],
      redoStack: [],
    });
    expect(malformed).toMatchObject({
      phase: 'setup',
      log: [],
      redoStack: [],
      aiPaused: false,
    });

    const illegal = sanitizePersistedState({
      phase: 'playing',
      config: double,
      controllers: ['local', 'local'],
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 0 },
      ],
      redoStack: [],
    });
    expect(illegal.phase).toBe('setup');
    expect(illegal.log).toEqual([]);
  });

  it('rehydrates only outcomes and acknowledgements that match the position', () => {
    const persisted = {
      phase: 'playing',
      config: mini,
      controllers: ['human', 'human'],
      aiPaused: false,
      log: clinchedLog,
      redoStack: [],
      earlyOutcome: {
        reason: 'clinch',
        winner: 1,
        loser: 0,
        emptyNodes: 1,
      },
      clinchAcknowledgement: {
        winner: 1,
        atLogLength: 49,
      },
    };

    expect(sanitizePersistedState(persisted)).toMatchObject({
      earlyOutcome: persisted.earlyOutcome,
      clinchAcknowledgement: persisted.clinchAcknowledgement,
    });
    expect(
      sanitizePersistedState({
        ...persisted,
        log: clinchedLog.slice(0, -1),
        redoStack: [clinchedLog[48]],
        earlyOutcome: null,
      }),
    ).toMatchObject({
      clinchAcknowledgement: persisted.clinchAcknowledgement,
    });
    expect(
      sanitizePersistedState({
        ...persisted,
        earlyOutcome: {
          reason: 'clinch',
          winner: 0,
          loser: 1,
          emptyNodes: 1,
        },
        clinchAcknowledgement: {
          winner: 0,
          atLogLength: 49,
        },
      }),
    ).toMatchObject({
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });
    expect(
      sanitizePersistedState({
        ...persisted,
        earlyOutcome: null,
        clinchAcknowledgement: {
          winner: 1,
          atLogLength: 1,
        },
      }),
    ).toMatchObject({
      clinchAcknowledgement: null,
    });
  });

  it('has no legacy pass parser path', () => {
    expect(parseGameAction({ type: 'pass' })).toBeNull();
    expect(parseGameAction({ type: 'place', node: 2 })).toEqual({
      type: 'place',
      node: 2,
    });
    expect(parseGameAction({ type: 'swap' })).toEqual({ type: 'swap' });
  });

  it('rejects incompatible configs instead of casting persisted values', () => {
    const result = sanitizePersistedState({
      phase: 'playing',
      config: { ...double, rings: '3' },
      controllers: ['local', 'local'],
      log: [],
      redoStack: [],
    });
    expect(result.phase).toBe('setup');
    expect(result.config).toEqual(DEFAULT_CONFIG);
    expect(result.controllers).toEqual(['human', 'human']);
  });

  it('migrates version 5 games without discarding replayable history', () => {
    const migrated = migratePersistedState(
      {
        phase: 'playing',
        config: {
          rings: 8,
          mode: 'double',
          pieRule: false,
          playerNames: ['Ada', 'Grace'],
        },
        controllers: ['local', 'local'],
        aiPaused: true,
        log: [{ type: 'place', node: 0 }],
        redoStack: [{ type: 'place', node: 1 }],
      },
      5,
    );
    expect(migrated).toMatchObject({
      phase: 'playing',
      controllers: ['local', 'local'],
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });
  });

  it('still resets pre-version-5 games while preserving valid preferences', () => {
    const migrated = migratePersistedState(
      {
        config: {
          rings: 8,
          mode: 'double',
          pieRule: false,
          playerNames: ['Ada', 'Grace'],
        },
        controllers: ['local', 'local'],
      },
      4,
    );
    expect(migrated).toEqual({
      phase: 'setup',
      config: {
        rings: 8,
        mode: 'double',
        pieRule: false,
        handicap: 1,
        playerNames: ['Ada', 'Grace'],
      },
      controllers: ['local', 'local'],
      aiInsightsHidden: false,
      aiSearchSettings: DEFAULT_AI_SEARCH_SETTINGS,
      aiPaused: false,
      log: [],
      redoStack: [],
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });
  });

  it('normalizes preferences independently and defaults an invalid ring to 6', () => {
    expect(
      normalizeGameConfig({
        rings: 5,
        mode: 'double',
        pieRule: true,
        playerNames: ['Ada', 17],
      }),
    ).toEqual({
      rings: 6,
      mode: 'double',
      pieRule: true,
      handicap: 1,
      playerNames: ['Ada', 'Player 2'],
    });
    // A handicap survives normalization only without the pie rule.
    expect(
      normalizeGameConfig({
        rings: 4,
        mode: 'classic',
        pieRule: false,
        handicap: 5,
        playerNames: ['Ada', 'Grace'],
      }).handicap,
    ).toBe(5);
    expect(
      normalizeGameConfig({
        rings: 4,
        mode: 'classic',
        pieRule: true,
        handicap: 5,
        playerNames: ['Ada', 'Grace'],
      }).handicap,
    ).toBe(1);
    expect(
      normalizeGameConfig({
        rings: 4,
        mode: 'classic',
        pieRule: false,
        handicap: 12,
        playerNames: ['Ada', 'Grace'],
      }).handicap,
    ).toBe(1);
  });

  it('validates each runtime search budget without clamping', () => {
    expect(parseAiSearchBudget('server', {
      simulations: 4_096,
      maxConsidered: 64,
    })).toEqual({ simulations: 4_096, maxConsidered: 64 });
    expect(parseAiSearchBudget('server', {
      simulations: 16_385,
      maxConsidered: 64,
    })).toBeNull();
    expect(parseAiSearchBudget('local', {
      simulations: 4_096,
      maxConsidered: 256,
    })).toEqual({ simulations: 4_096, maxConsidered: 256 });
    expect(parseAiSearchBudget('local', {
      simulations: 536_870_911,
      maxConsidered: 4_294_967_295,
    })).toEqual({ simulations: 536_870_911, maxConsidered: 4_294_967_295 });

    expect(
      normalizeAiSearchSettings({
        server: { simulations: 256, maxConsidered: 12 },
        local: { simulations: 0, maxConsidered: 8 },
      }),
    ).toEqual({
      server: { simulations: 256, maxConsidered: 12 },
      local: DEFAULT_AI_SEARCH_SETTINGS.local,
    });
  });

  it.each(['simulations', 'maxConsidered'] as const)(
    'rejects invalid custom browser %s instead of coercing or clamping',
    (field) => {
      const overflow = field === 'simulations' ? 536_870_912 : 4_294_967_296;
      for (const value of [0, -1, 1.5, NaN, Infinity, overflow, Number.MAX_SAFE_INTEGER, '4096', null]) {
        expect(parseAiSearchBudget('local', {
          simulations: 4_096,
          maxConsidered: 256,
          [field]: value,
        })).toBeNull();
      }
    },
  );

  it.each(['setup', 'playing'] as const)(
    'normalizes a deeper custom browser search to Deep after restoring %s',
    (phase) => {
      const budget = { simulations: 4_096, maxConsidered: 256 };
      const saved = JSON.parse(JSON.stringify({
        phase,
        config: DEFAULT_CONFIG,
        controllers: ['human', 'local'],
        aiSearchSettings: { ...DEFAULT_AI_SEARCH_SETTINGS, local: budget },
        log: phase === 'playing' ? [{ type: 'place', node: 0 }] : [],
        redoStack: [],
      }));

      expect(sanitizePersistedState(saved)).toMatchObject({
        phase,
        aiSearchSettings: { local: { simulations: 4096, maxConsidered: 64 } },
      });
      expect(migratePersistedState(saved, APP_STORE_VERSION)).toMatchObject({
        phase,
        aiSearchSettings: { local: { simulations: 4096, maxConsidered: 64 } },
      });
    },
  );

  it('updates valid runtime settings and ignores invalid setter values', () => {
    useAppStore.getState().setAiSearchBudget('local', {
      simulations: 4_096,
      maxConsidered: 256,
    });
    expect(useAppStore.getState().aiSearchSettings.local).toEqual({
      simulations: 4_096,
      maxConsidered: 64,
    });

    useAppStore.getState().setAiSearchBudget('local', {
      simulations: 536_870_912,
      maxConsidered: 8,
    });
    expect(useAppStore.getState().aiSearchSettings.local).toEqual({
      simulations: 4_096,
      maxConsidered: 64,
    });
  });

  it('updates cloud effort without changing the browser setting and vice versa', () => {
    const state = useAppStore.getState();
    state.setAiSearchBudget('server', { simulations: 2048, maxConsidered: 32 });
    expect(useAppStore.getState().aiSearchSettings).toEqual({
      server: { simulations: 2048, maxConsidered: 32 }, local: DEFAULT_AI_SEARCH_SETTINGS.local,
    });
    state.setAiSearchBudget('local', { simulations: 4096, maxConsidered: 64 });
    expect(useAppStore.getState().aiSearchSettings).toEqual({
      server: { simulations: 2048, maxConsidered: 32 }, local: { simulations: 4096, maxConsidered: 64 },
    });
    state.setAiSearchBudget('server', { simulations: 16385, maxConsidered: 32 });
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({ simulations: 2048, maxConsidered: 32 });
  });
});

describe('human versus AI insight privacy', () => {
  it.each(['classic', 'double'] as const)(
    'hides AI insights for both human seats across every %s opening',
    (mode) => {
      const openings = [
        { pieRule: true, handicap: 1 },
        ...Array.from({ length: 9 }, (_, index) => ({ pieRule: false, handicap: index + 1 })),
      ];
      for (const opening of openings) {
        for (const controllers of [
          ['human', 'local'], ['local', 'human'],
          ['human', 'server'], ['server', 'human'],
        ] satisfies PlayerControllers[]) {
          useAppStore.getState().startGame({ ...double, rings: 10, mode, ...opening }, controllers);
          expect(useAppStore.getState().aiInsightsHidden).toBe(true);
        }
      }
    },
  );

  it.each([5, 6, 7, APP_STORE_VERSION])(
    'hides version %i saved human-AI games while upgrading unsupported effort',
    (version) => {
      for (const aiInsightsHidden of [undefined, false, true]) {
        const saved = {
          phase: 'playing',
          config: { ...double, handicap: 3 },
          controllers: ['server', 'human'],
          aiInsightsHidden,
          aiSearchSettings: { local: { simulations: 2_048, maxConsidered: 32 } },
          aiPaused: true,
          log: [{ type: 'place', node: 0 }],
          redoStack: [{ type: 'place', node: 1 }],
        };
        const restored = migratePersistedState(saved, version);
        expect(restored).toMatchObject({
          ...saved,
          aiSearchSettings: { local: { simulations: 544, maxConsidered: 16 } },
          controllers: ['server', 'human'],
          aiInsightsHidden: true,
        });
        expect(sanitizePersistedState(saved).aiInsightsHidden).toBe(true);
      }
    },
  );

  it('keeps insights hidden through takeover, history navigation, review, and save restoration', () => {
    useAppStore.getState().startGame(double, ['human', 'local']);
    useAppStore.getState().act({ type: 'place', node: 0 });
    useAppStore.getState().act({ type: 'place', node: 1 });
    useAppStore.getState().setPlayerController(0, 'local');
    useAppStore.getState().pauseAi();
    useAppStore.getState().undo();
    useAppStore.getState().redo();
    useAppStore.getState().rewindTo(1);
    useAppStore.getState().setReviewing(true);
    useAppStore.getState().resign(0);
    const current = useAppStore.getState();
    expect(current.controllers).toEqual(['local', 'local']);
    expect(current.aiInsightsHidden).toBe(true);
    expect(canShowAiInsights(current.controllers, current.aiInsightsHidden)).toBe(false);

    expect(sanitizePersistedState(JSON.parse(JSON.stringify(current)))).toMatchObject({
      aiInsightsHidden: true,
      controllers: ['local', 'local'],
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
    });
  });

  it('protects self-play as soon as a human takes over, including a later second takeover', () => {
    useAppStore.getState().startGame(double, ['local', 'local']);
    expect(useAppStore.getState().aiInsightsHidden).toBe(false);
    useAppStore.getState().setPlayerController(0, 'human');
    expect(useAppStore.getState().aiInsightsHidden).toBe(true);
    useAppStore.getState().setPlayerController(1, 'human');
    const current = useAppStore.getState();
    expect(current.controllers).toEqual(['human', 'human']);
    expect(current.aiInsightsHidden).toBe(true);
    expect(sanitizePersistedState(current).aiInsightsHidden).toBe(true);
  });

  it('recomputes privacy for new games and rematches and clears it on leaving a game', () => {
    useAppStore.getState().startGame(double, ['human', 'local']);
    useAppStore.getState().rematch();
    expect(useAppStore.getState().aiInsightsHidden).toBe(true);
    useAppStore.getState().setPlayerController(0, 'local');
    useAppStore.getState().rematch();
    expect(useAppStore.getState().aiInsightsHidden).toBe(false);
    useAppStore.getState().setPlayerController(0, 'human');
    useAppStore.getState().startGame(double, ['human', 'human']);
    expect(useAppStore.getState().aiInsightsHidden).toBe(false);
    useAppStore.getState().setPlayerController(1, 'local');
    expect(useAppStore.getState().aiInsightsHidden).toBe(true);
    useAppStore.getState().toSetup();
    expect(useAppStore.getState().aiInsightsHidden).toBe(false);
    useAppStore.getState().setPlayerController(0, 'local');
    expect(useAppStore.getState().aiInsightsHidden).toBe(false);
    expect(sanitizePersistedState({ ...useAppStore.getState(), aiInsightsHidden: true }).aiInsightsHidden).toBe(false);
  });
});

describe('history navigation AI pause', () => {
  it('keeps randomized branching, undo, redo, and saved histories consistent', () => {
    fc.assert(fc.property(
      fc.constantFrom('classic' as const, 'double' as const),
      fc.integer({ min: 1, max: 9 }),
      fc.array(fc.record({
        operation: fc.constantFrom('place', 'swap', 'undo', 'redo', 'rewind'),
        selector: fc.nat(),
      }), { minLength: 20, maxLength: 120 }),
      (mode, handicap, commands) => {
        useAppStore.getState().startGame({ ...double, rings: 10, mode, handicap }, ['local', 'local']);
        const config = useAppStore.getState().config;
        let log: GameAction[] = [];
        let redoStack: GameAction[] = [];
        for (const { operation, selector } of commands) {
          const game = replay(config, log);
          const store = useAppStore.getState();
          if (operation === 'place' || operation === 'swap') {
            const node = selector % game.board.n;
            const action: GameAction = operation === 'place' ? { type: 'place', node } : { type: 'swap' };
            store.act(action);
            if (!game.over && (operation === 'place' ? game.stones[node] === -1 : game.canSwap)) {
              log = [...log, action];
              redoStack = [];
            }
          } else if (operation === 'undo') {
            store.undo();
            if (log.length) {
              redoStack.push(log[log.length - 1]);
              log = log.slice(0, -1);
            }
          } else if (operation === 'redo') {
            store.redo();
            if (redoStack.length) log.push(redoStack.pop()!);
          } else {
            const ply = selector % (log.length + 2);
            store.rewindTo(ply);
            if (ply < log.length) {
              redoStack = [...redoStack, ...log.slice(ply).reverse()];
              log = log.slice(0, ply);
            }
          }
          const current = useAppStore.getState();
          expect(current.log).toEqual(log);
          expect(current.redoStack).toEqual(redoStack);
          const restored = sanitizePersistedState(JSON.parse(JSON.stringify(current)));
          expect(restored.phase).toBe('playing');
          expect(restored.log).toEqual(log);
          expect(restored.redoStack).toEqual(redoStack);
          expect(replay(restored.config, restored.log)).toEqual(replay(config, log));
        }
      },
    ), { numRuns: 80 });
  });

  it('rewinds several actions at once exactly like repeated undo', () => {
    const log = [
      { type: 'place' as const, node: 0 },
      { type: 'place' as const, node: 1 },
      { type: 'place' as const, node: 2 },
      { type: 'place' as const, node: 3 },
    ];
    useAppStore.setState({
      phase: 'playing',
      config: double,
      controllers: ['human', 'human'],
      aiPaused: false,
      log,
      redoStack: [],
      earlyOutcome: null,
    });

    useAppStore.getState().rewindTo(1);
    expect(useAppStore.getState()).toMatchObject({
      log: [{ type: 'place', node: 0 }],
      redoStack: [
        { type: 'place', node: 3 },
        { type: 'place', node: 2 },
        { type: 'place', node: 1 },
      ],
      aiPaused: true,
      earlyOutcome: null,
    });

    // Redo restores the original order one action at a time.
    useAppStore.getState().redo();
    useAppStore.getState().redo();
    useAppStore.getState().redo();
    expect(useAppStore.getState().log).toEqual(log);
    expect(useAppStore.getState().redoStack).toEqual([]);
  });

  it('ignores rewind targets outside the current log', () => {
    useAppStore.setState({
      phase: 'playing',
      config: double,
      controllers: ['human', 'human'],
      aiPaused: false,
      log: [{ type: 'place', node: 0 }],
      redoStack: [],
    });

    for (const ply of [-1, 1, 5, 0.5]) {
      useAppStore.getState().rewindTo(ply);
      expect(useAppStore.getState()).toMatchObject({
        log: [{ type: 'place', node: 0 }],
        redoStack: [],
        aiPaused: false,
      });
    }
  });

  it('pauses after undo and preserves redo when a player takes over', () => {
    useAppStore.setState({
      phase: 'playing',
      config: double,
      controllers: ['local', 'local'],
      aiPaused: false,
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
      redoStack: [],
    });

    useAppStore.getState().undo();
    expect(useAppStore.getState()).toMatchObject({
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
    });

    useAppStore.getState().setPlayerController(1, 'human');
    expect(useAppStore.getState()).toMatchObject({
      controllers: ['local', 'human'],
      aiPaused: false,
      redoStack: [{ type: 'place', node: 1 }],
    });
  });
});

describe('gameplay store actions', () => {
  it('ignores stale, malformed, duplicate, and terminal actions without corrupting the game', () => {
    useAppStore.getState().toSetup();
    useAppStore.getState().act({ type: 'place', node: 0 });
    expect(useAppStore.getState().log).toEqual([]);
    useAppStore.getState().startGame(mini, ['human', 'human']);
    for (const action of [null, { type: 'swap' }, { type: 'place', node: -1 }, { type: 'place', node: '0' }, { type: 'place', node: 50 }]) {
      useAppStore.getState().act(action as GameAction);
      expect(useAppStore.getState().log).toEqual([]);
    }
    const action: GameAction = { type: 'place', node: 0 };
    useAppStore.getState().act(action);
    action.node = 1;
    expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]);
    useAppStore.getState().act({ type: 'place', node: 0 });
    expect(useAppStore.getState().log).toHaveLength(1);
    for (let node = 1; node < 50; node++) useAppStore.getState().act({ type: 'place', node });
    const completed = useAppStore.getState();
    expect(replay(completed.config, completed.log).over).toBe(true);
    completed.act({ type: 'place', node: 0 });
    completed.act({ type: 'swap' });
    expect(useAppStore.getState().log).toEqual(completed.log);
  });

  it('cannot create results in setup or replace a resignation with a clinch', () => {
    useAppStore.getState().toSetup();
    useAppStore.getState().resign(0);
    useAppStore.getState().acknowledgeClinch(0);
    useAppStore.getState().endClinchedGame(0);
    expect(useAppStore.getState().earlyOutcome).toBeNull();
    expect(useAppStore.getState().clinchAcknowledgement).toBeNull();
    useAppStore.setState({ phase: 'playing', config: mini, log: clinchedLog });
    useAppStore.getState().resign(1);
    useAppStore.getState().endClinchedGame(1);
    useAppStore.getState().acknowledgeClinch(1);
    expect(useAppStore.getState().earlyOutcome).toEqual({ reason: 'resignation', winner: 0, loser: 1 });
    expect(useAppStore.getState().clinchAcknowledgement).toBeNull();
  });

  it('reopens play for a rematch and clears review mode on redo', () => {
    useAppStore.getState().toSetup();
    useAppStore.getState().rematch();
    expect(useAppStore.getState().phase).toBe('playing');
    useAppStore.getState().act({ type: 'place', node: 0 });
    useAppStore.getState().undo();
    useAppStore.getState().setReviewing(true);
    useAppStore.getState().redo();
    expect(useAppStore.getState().reviewing).toBe(false);
  });

  it.each(['classic', 'double'] as const)(
    'enforces new-game openings on direct %s starts and rematches, preserving old active games',
    (mode) => {
      for (const rings of [4, 6, 8, 10]) {
        for (const handicap of [1, 2, 9]) {
          const config: GameConfig = { ...double, mode, rings, handicap };
          const legacy = sanitizePersistedState({
            phase: 'playing',
            config,
            controllers: ['local', 'local'],
            log: [{ type: 'place', node: 0 }],
            redoStack: [],
          });
          expect(legacy.phase).toBe('playing');
          expect(legacy.config).toEqual(config);
          expect(legacy.log).toEqual([{ type: 'place', node: 0 }]);
          const expectedHandicap = rings === 10 ? handicap : 1;
          const expected = {
            ...config,
            handicap: expectedHandicap,
            pieRule: expectedHandicap === 1,
          };
          useAppStore.setState(legacy);
          useAppStore.getState().rematch();
          expect(useAppStore.getState().config).toEqual(expected);
          expect(useAppStore.getState().log).toEqual([]);
          useAppStore.getState().startGame(config, ['local', 'local']);
          expect(useAppStore.getState().config).toEqual(expected);
          expect(config.pieRule).toBe(false);
          expect(legacy.config).toEqual(config);
        }
      }
    },
  );

  it('starts, acts, redoes, reviews, rematches, and leaves without stale state', () => {
    const store = useAppStore.getState();
    store.startGame(double, ['local', 'local']);
    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      config: { ...double, pieRule: true, handicap: 1 },
      controllers: ['local', 'local'],
      log: [],
      reviewing: false,
    });

    useAppStore.getState().act({ type: 'place', node: 0 });
    useAppStore.getState().act({ type: 'place', node: 1 });
    useAppStore.getState().undo();
    useAppStore.getState().redo();
    expect(useAppStore.getState()).toMatchObject({
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
      redoStack: [],
      aiPaused: true,
    });

    useAppStore.getState().setReviewing(true);
    expect(useAppStore.getState().reviewing).toBe(true);
    useAppStore.getState().rematch();
    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      log: [],
      redoStack: [],
      aiPaused: false,
      reviewing: false,
    });

    useAppStore.getState().toSetup();
    expect(useAppStore.getState()).toMatchObject({
      phase: 'setup',
      log: [],
      redoStack: [],
      aiPaused: false,
      reviewing: false,
    });
  });

  it('refuses to start unsupported board sizes instead of coercing them', () => {
    const before = useAppStore.getState();
    expect(() =>
      before.startGame({ ...double, rings: 9 }, ['human', 'human']),
    ).toThrow(/unsupported configuration/i);
    expect(useAppStore.getState().phase).toBe('setup');
  });

  it('keeps AI controllers for classic games and ignores invalid controller slots', () => {
    const classic: GameConfig = { ...double, mode: 'classic' };
    useAppStore.getState().startGame(classic, ['local', 'local']);
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
    const before = useAppStore.getState();
    before.setPlayerController(2 as 0, 'server');
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
    useAppStore.getState().resumeAi();
    expect(useAppStore.getState().aiPaused).toBe(false);
  });

  it('refuses configurations outside the rules family', () => {
    const oversized: GameConfig = { ...double, handicap: 12 };
    expect(() =>
      useAppStore.getState().startGame(oversized, ['local', 'local']),
    ).toThrow(/unsupported configuration/);
  });

  it('acknowledges, ends, and resets a clinched game without changing its log', () => {
    useAppStore.setState({
      phase: 'playing',
      config: mini,
      controllers: ['human', 'human'],
      log: clinchedLog,
      redoStack: [],
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });

    useAppStore.getState().acknowledgeClinch(1);
    expect(useAppStore.getState().clinchAcknowledgement).toEqual({
      winner: 1,
      atLogLength: 49,
    });

    useAppStore.getState().endClinchedGame(1);
    expect(useAppStore.getState()).toMatchObject({
      log: clinchedLog,
      earlyOutcome: {
        reason: 'clinch',
        winner: 1,
        loser: 0,
        emptyNodes: 1,
      },
    });
    useAppStore.getState().act({ type: 'place', node: 49 });
    expect(useAppStore.getState().log).toEqual(clinchedLog);

    useAppStore.getState().rematch();
    expect(useAppStore.getState()).toMatchObject({
      log: [],
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });
  });

  it('records the named resigning player and clears the outcome on undo', () => {
    useAppStore.setState({
      phase: 'playing',
      config: mini,
      controllers: ['human', 'human'],
      log: [{ type: 'place', node: 0 }],
      redoStack: [],
      earlyOutcome: null,
      clinchAcknowledgement: null,
    });

    useAppStore.getState().resign(0);
    expect(useAppStore.getState().earlyOutcome).toEqual({
      reason: 'resignation',
      winner: 1,
      loser: 0,
    });
    useAppStore.getState().undo();
    expect(useAppStore.getState()).toMatchObject({
      log: [],
      earlyOutcome: null,
    });
  });
});
