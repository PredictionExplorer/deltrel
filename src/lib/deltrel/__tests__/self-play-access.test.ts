// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { GAME_STORAGE_KEY } from '../../persistence';
import { DEFAULT_CONFIG, useAppStore } from '../../store';

beforeEach(() => {
  localStorage.clear();
  useAppStore.getState().toSetup();
  useAppStore.getState().setSelfPlayAccess(false);
});

describe('self-play capability', () => {
  it('prevents public starts and controller changes from creating two computers', () => {
    useAppStore.getState().startGame(DEFAULT_CONFIG, ['local', 'local']);
    expect(useAppStore.getState().controllers).toEqual(['human', 'local']);
    useAppStore.getState().setPlayerController(0, 'local');
    expect(useAppStore.getState().controllers).toEqual(['human', 'local']);
  });

  it('allows self-play only for the authorized page and removes access without losing the game', () => {
    useAppStore.getState().setSelfPlayAccess(true);
    useAppStore.getState().startGame(DEFAULT_CONFIG, ['local', 'local']);
    useAppStore.getState().act({ type: 'place', node: 0 });
    expect(useAppStore.getState().controllers).toEqual(['local', 'local']);
    const saved = JSON.parse(localStorage.getItem(GAME_STORAGE_KEY)!);
    expect(saved.state).not.toHaveProperty('selfPlayAllowed');
    expect(saved.state).not.toHaveProperty('selfPlayAccessReady');
    useAppStore.getState().setSelfPlayAccess(false);
    expect(useAppStore.getState()).toMatchObject({ controllers: ['human', 'local'], aiPaused: true, aiInsightsHidden: true, log: [{ type: 'place', node: 0 }] });
  });

  it('ignores forged saved access and prevents rehydration or rematch from restoring public self-play', async () => {
    useAppStore.getState().setSelfPlayAccess(true);
    useAppStore.getState().startGame(DEFAULT_CONFIG, ['local', 'local']);
    const saved = JSON.parse(localStorage.getItem(GAME_STORAGE_KEY)!);
    saved.state.selfPlayAllowed = true;
    useAppStore.getState().setSelfPlayAccess(false);
    localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify(saved));
    await useAppStore.persist.rehydrate();
    expect(useAppStore.getState().selfPlayAllowed).toBe(false);
    expect(useAppStore.getState().controllers).toEqual(['human', 'local']);
    useAppStore.getState().rematch();
    expect(useAppStore.getState().controllers).toEqual(['human', 'local']);
  });
});
