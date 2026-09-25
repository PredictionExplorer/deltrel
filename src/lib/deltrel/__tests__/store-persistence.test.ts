// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it } from 'vitest';
import { GAME_STORAGE_KEY } from '../../persistence';
import { APP_STORE_VERSION, DEFAULT_CONFIG, useAppStore } from '../../store';

beforeEach(() => {
  useAppStore.getState().setSelfPlayAccess(true);
});

afterEach(() => {
  useAppStore.getState().toSetup();
  useAppStore.getState().setSelfPlayAccess(false);
  window.localStorage.clear();
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
