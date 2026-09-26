// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GameRecord } from './deltrel/game-record';

let library: typeof import('./game-library');

function record(id = 'game-one'): GameRecord {
  return {
    id,
    createdAt: '2026-09-26T12:00:00.000Z',
    updatedAt: '2026-09-26T12:01:00.000Z',
    config: { rings: 4, mode: 'classic', pieRule: true, handicap: 1, playerNames: ['Ada', 'Grace'] },
    controllers: ['human', 'server'],
    aiSearchSettings: {
      server: { simulations: 544, maxConsidered: 16 },
      local: { simulations: 544, maxConsidered: 16 },
    },
    log: [{ type: 'place', node: 0 }],
    earlyOutcome: null,
  };
}

beforeEach(async () => {
  window.localStorage.clear();
  vi.resetModules();
  library = await import('./game-library');
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe('local game library', () => {
  it('writes only the changed game and retains independently saved games', () => {
    library.saveGameRecord(record('older'));
    const previous = window.localStorage.getItem(`${library.GAME_LIBRARY_PREFIX}older`);
    const writes = vi.spyOn(Storage.prototype, 'setItem');
    const current = record('current');
    expect(library.saveGameRecord(current)).toBe(true);
    expect(writes).toHaveBeenCalledTimes(1);
    expect(writes.mock.calls[0][0]).toBe(`${library.GAME_LIBRARY_PREFIX}current`);
    expect(window.localStorage.getItem(`${library.GAME_LIBRARY_PREFIX}older`)).toBe(previous);
    current.log.push({ type: 'place', node: 1 });
    // The library retains its own validated snapshot.
    expect(library.listGameRecords().find((game) => game.id === 'current')?.log).toHaveLength(1);
    expect(library.saveGameRecord(current)).toBe(true);
    expect(library.listGameRecords()).toHaveLength(2);
    expect(library.listGameRecords().find((game) => game.id === 'current')?.log).toHaveLength(2);
  });

  it('reloads independent records in recency order and preserves corrupt entries', () => {
    const prefix = library.GAME_LIBRARY_PREFIX;
    const latest = { ...record('new'), updatedAt: '2026-09-26T13:00:00.000Z' };
    window.localStorage.setItem(`${prefix}old`, JSON.stringify({ version: 1, record: record('old') }));
    window.localStorage.setItem(`${prefix}new`, JSON.stringify({ version: 1, record: latest }));
    window.localStorage.setItem(`${prefix}bad`, '{bad JSON');
    window.localStorage.setItem(`${prefix}mismatch`, JSON.stringify({ version: 1, record: record('wrong-id') }));
    window.localStorage.setItem(`${prefix}oversized`, ' '.repeat(library.MAX_STORED_GAME_LENGTH + 1));
    window.localStorage.setItem('unrelated', 'keep');
    expect(library.listGameRecords().map((game) => game.id)).toEqual(['new', 'old']);
    expect(library.getGameLibraryStatus()).toContain('3 saved games could not be read');
    expect(window.localStorage.getItem(`${prefix}bad`)).toBe('{bad JSON');
    expect(window.localStorage.getItem('unrelated')).toBe('keep');
  });

  it('retains failed writes in memory, reports the failure, and retries without evicting old games', () => {
    expect(library.saveGameRecord(record('old'))).toBe(true);
    const writes = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Full', 'QuotaExceededError');
    });
    expect(library.saveGameRecord(record('new'))).toBe(false);
    expect(library.listGameRecords()).toHaveLength(2);
    expect(library.getGameLibraryStatus()).toContain('could not be saved');
    expect(window.localStorage.getItem(`${library.GAME_LIBRARY_PREFIX}old`)).not.toBeNull();
    writes.mockRestore();
    expect(library.retryGameLibrarySaves()).toBe(true);
    expect(library.getGameLibraryStatus()).toBeNull();
    expect(window.localStorage.getItem(`${library.GAME_LIBRARY_PREFIX}new`)).not.toBeNull();
  });

  it('detects silently discarded saves and does not pretend deletion succeeded', () => {
    const saved = record();
    expect(library.saveGameRecord(saved)).toBe(true);
    const remove = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {});
    expect(library.deleteGameRecord(saved.id)).toBe(false);
    expect(library.listGameRecords()).toHaveLength(1);
    expect(library.getGameLibraryStatus()).toContain('could not be removed');
    remove.mockRestore();
    expect(library.deleteGameRecord(saved.id)).toBe(true);
    expect(library.listGameRecords()).toEqual([]);
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});
    expect(library.saveGameRecord(record('discarded'))).toBe(false);
    expect(library.getGameLibraryStatus()).toContain('could not be saved');
  });

  it('handles blocked browser storage and invalid records without throwing', () => {
    const blocked = vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('Blocked', 'SecurityError');
    });
    expect(library.listGameRecords()).toEqual([]);
    expect(library.getGameLibraryStatus()).toContain('unavailable');
    expect(library.saveGameRecord(record())).toBe(false);
    expect(library.listGameRecords()).toHaveLength(1);
    blocked.mockRestore();
    expect(library.retryGameLibrarySaves()).toBe(true);
    expect(library.saveGameRecord({ ...record('invalid'), log: [{ type: 'swap' }] })).toBe(false);
    expect(library.getGameLibraryStatus()).toContain('invalid');
    expect(library.listGameRecords()).toHaveLength(1);
  });

  it('notifies subscribers about saves, deletes, and changes from another tab', () => {
    const listener = vi.fn();
    const unsubscribe = library.subscribeGameLibrary(listener);
    listener.mockClear();
    library.saveGameRecord(record());
    expect(listener).toHaveBeenCalledTimes(1);
    const external = record('external');
    window.localStorage.setItem(`${library.GAME_LIBRARY_PREFIX}external`, JSON.stringify({ version: 1, record: external }));
    window.dispatchEvent(new StorageEvent('storage', { key: `${library.GAME_LIBRARY_PREFIX}external` }));
    expect(library.listGameRecords()).toHaveLength(2);
    const calls = listener.mock.calls.length;
    window.dispatchEvent(new StorageEvent('storage', { key: 'unrelated' }));
    expect(listener).toHaveBeenCalledTimes(calls);
    expect(library.deleteGameRecord('game-one')).toBe(true);
    expect(library.listGameRecords().map((game) => game.id)).toEqual(['external']);
    unsubscribe();
    listener.mockClear();
    library.saveGameRecord(record('later'));
    expect(listener).not.toHaveBeenCalled();
  });
});
