// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { GAME_STORAGE_KEY } from '../../persistence';
import { getGameLibraryStatus, listGameRecords, saveGameRecord, subscribeGameLibrary } from '../../game-library';
import { APP_STORE_VERSION, currentGameRecord, DEFAULT_CONFIG, useAppStore } from '../../store';

beforeEach(() => {
  const unsubscribe = subscribeGameLibrary(() => {});
  unsubscribe();
  useAppStore.getState().setSelfPlayAccess(true);
});

afterEach(() => {
  useAppStore.getState().toSetup();
  useAppStore.getState().setSelfPlayAccess(false);
  window.localStorage.clear();
});

it('archives every played game, current settings, and results through rematches and setup', () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'human']);
  const firstId = useAppStore.getState().gameId;
  useAppStore.getState().act({ type: 'place', node: 0 });
  useAppStore.getState().setPlayerController(1, 'server');
  useAppStore.getState().setAiSearchBudget('server', { simulations: 4096, maxConsidered: 64 });
  useAppStore.getState().resign(0);
  const original = listGameRecords().find((game) => game.id === firstId);
  expect(original).toMatchObject({
    controllers: ['human', 'server'],
    aiSearchSettings: { server: { simulations: 4096, maxConsidered: 64 } },
    log: [{ type: 'place', node: 0 }],
    earlyOutcome: { reason: 'resignation', winner: 1, loser: 0 },
  });
  useAppStore.getState().rematch();
  const secondId = useAppStore.getState().gameId;
  expect(secondId).not.toBe(firstId);
  useAppStore.getState().act({ type: 'place', node: 1 });
  useAppStore.getState().toSetup();
  expect(useAppStore.getState().gameId).toBeNull();
  expect(listGameRecords()).toHaveLength(2);
  expect(listGameRecords().find((game) => game.id === firstId)).toEqual(original);
  expect(listGameRecords().find((game) => game.id === secondId)?.log).toEqual([{ type: 'place', node: 1 }]);
});

it('preserves the original result when undoing into a separate variation', () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'human']);
  useAppStore.getState().act({ type: 'place', node: 0 });
  useAppStore.getState().act({ type: 'place', node: 1 });
  useAppStore.getState().resign(1);
  const original = currentGameRecord(useAppStore.getState())!;
  useAppStore.getState().undo();
  const variationId = useAppStore.getState().gameId;
  expect(variationId).not.toBe(original.id);
  useAppStore.getState().undo();
  expect(useAppStore.getState().gameId).toBe(variationId);
  useAppStore.getState().redo();
  useAppStore.getState().act({ type: 'place', node: 2 });
  expect(listGameRecords().find((game) => game.id === original.id)).toEqual(original);
  expect(listGameRecords().find((game) => game.id === variationId)).toMatchObject({
    log: [{ type: 'place', node: 0 }, { type: 'place', node: 2 }],
    earlyOutcome: null,
  });
});

it('preserves a completed full board when rewinding and leaves sharing snapshots detached', () => {
  useAppStore.getState().startGame({ ...DEFAULT_CONFIG, rings: 4 }, ['human', 'human']);
  for (let node = 0; node < 50; node++) useAppStore.getState().act({ type: 'place', node });
  const original = currentGameRecord(useAppStore.getState())!;
  original.config.playerNames[0] = 'Changed only in copy';
  expect(useAppStore.getState().config.playerNames[0]).toBe('Player 1');
  useAppStore.getState().rewindTo(45);
  expect(useAppStore.getState().gameId).not.toBe(original.id);
  expect(listGameRecords().find((game) => game.id === original.id)?.log).toHaveLength(50);
  expect(currentGameRecord(useAppStore.getState())?.log).toHaveLength(45);
});

it('migrates the existing active game into the library once with a stable persisted identity', async () => {
  window.localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify({
    version: 9,
    state: {
      phase: 'playing', config: DEFAULT_CONFIG, controllers: ['human', 'human'],
      log: [{ type: 'place', node: 0 }], redoStack: [], earlyOutcome: null,
    },
  }));
  await useAppStore.persist.rehydrate();
  const first = currentGameRecord(useAppStore.getState())!;
  expect(first.id).toBeTruthy();
  expect(listGameRecords()).toEqual([first]);
  expect(JSON.parse(window.localStorage.getItem(GAME_STORAGE_KEY)!).state.gameId).toBe(first.id);
  await useAppStore.persist.rehydrate();
  expect(currentGameRecord(useAppStore.getState())).toEqual(first);
  expect(listGameRecords()).toEqual([first]);
});

it('forks simultaneous tab changes instead of overwriting the other saved continuation', () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'human']);
  useAppStore.getState().act({ type: 'place', node: 0 });
  const previous = currentGameRecord(useAppStore.getState())!;
  const otherTab = { ...previous, log: [...previous.log, { type: 'place' as const, node: 1 }] };
  saveGameRecord(otherTab);
  useAppStore.getState().act({ type: 'place', node: 2 });
  expect(useAppStore.getState().gameId).not.toBe(previous.id);
  expect(listGameRecords().find((game) => game.id === previous.id)).toEqual(otherTab);
  expect(listGameRecords().find((game) => game.id === useAppStore.getState().gameId)?.log).toEqual([
    { type: 'place', node: 0 }, { type: 'place', node: 2 },
  ]);
});

it('does not overwrite another tab’s saved continuation while hydrating a stale active game', async () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'human']);
  useAppStore.getState().act({ type: 'place', node: 0 });
  const active = currentGameRecord(useAppStore.getState())!;
  const otherTab = { ...active, log: [...active.log, { type: 'place' as const, node: 1 }] };
  saveGameRecord(otherTab);
  await useAppStore.persist.rehydrate();
  expect(currentGameRecord(useAppStore.getState())).toEqual(active);
  expect(listGameRecords().find((game) => game.id === active.id)).toEqual(otherTab);
  useAppStore.getState().act({ type: 'place', node: 2 });
  expect(useAppStore.getState().gameId).not.toBe(active.id);
  expect(listGameRecords().find((game) => game.id === active.id)).toEqual(otherTab);
});

it('does not rewrite the archive for AI pause or board review and reports failed saves', () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'human']);
  const before = currentGameRecord(useAppStore.getState());
  const writes = vi.spyOn(Storage.prototype, 'setItem');
  useAppStore.getState().pauseAi();
  useAppStore.getState().setReviewing(true);
  expect(writes.mock.calls.every(([key]) => key === GAME_STORAGE_KEY)).toBe(true);
  expect(currentGameRecord(useAppStore.getState())).toEqual(before);
  writes.mockImplementation(() => { throw new Error('Full'); });
  expect(() => useAppStore.getState().act({ type: 'place', node: 0 })).not.toThrow();
  expect(useAppStore.getState().log).toHaveLength(1);
  expect(getGameLibraryStatus()).toContain('could not be saved');
  writes.mockRestore();
});

it('persists private match history when the current controllers no longer include a human', async () => {
  useAppStore.getState().startGame(DEFAULT_CONFIG, ['human', 'local']);
  useAppStore.getState().act({ type: 'place', node: 0 });
  useAppStore.getState().setPlayerController(0, 'local');
  useAppStore.getState().pauseAi();
  const rawSaved = window.localStorage.getItem(GAME_STORAGE_KEY)!;
  const saved = JSON.parse(rawSaved);
  expect(saved.version).toBe(APP_STORE_VERSION);
  expect(saved.state.aiInsightsHidden).toBe(true);

  useAppStore.getState().toSetup();
  expect(useAppStore.getState().aiInsightsHidden).toBe(false);
  window.localStorage.setItem(GAME_STORAGE_KEY, rawSaved);
  await useAppStore.persist.rehydrate();
  expect(useAppStore.getState()).toMatchObject({
    phase: 'playing',
    controllers: ['local', 'local'],
    aiInsightsHidden: true,
    aiPaused: true,
    log: [{ type: 'place', node: 0 }],
  });
});
