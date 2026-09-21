import { afterEach, describe, expect, it, vi } from 'vitest';
import * as rules from './deltrel/rules';
import { GAME_STORAGE_KEY, migrateStorageNamespace } from './persistence';

function memoryStorage(entries: Record<string, string>): Storage {
  const values = new Map(Object.entries(entries));
  return {
    get length() { return values.size; },
    clear: () => values.clear(),
    key: (index) => [...values.keys()][index] ?? null,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
  };
}

const save = JSON.stringify({ state: { log: [{ type: 'place', node: 23 }] }, version: 6 });

afterEach(() => vi.restoreAllMocks());

describe('Deltrel save namespace migration', () => {
  function identifyPreviousSave() {
    const original = rules.fnv1a64;
    vi.spyOn(rules, 'fnv1a64').mockImplementation((key) =>
      key === 'previous-game' ? '292253f25236e69d' : original(key));
  }

  it('preserves the exact move log and removes the previous key only after copying', () => {
    identifyPreviousSave();
    const storage = memoryStorage({ unrelated: 'private', 'previous-game': save });
    migrateStorageNamespace(storage);
    expect(storage.getItem(GAME_STORAGE_KEY)).toBe(save);
    expect(storage.getItem('previous-game')).toBeNull();
    expect(storage.getItem('unrelated')).toBe('private');
    migrateStorageNamespace(storage);
    expect(storage.length).toBe(2);
  });

  it('keeps the current save and leaves other data untouched', () => {
    identifyPreviousSave();
    const storage = memoryStorage({ [GAME_STORAGE_KEY]: 'current', 'previous-game': save });
    migrateStorageNamespace(storage);
    expect(storage.getItem(GAME_STORAGE_KEY)).toBe('current');
    expect(storage.getItem('previous-game')).toBe(save);
  });

  it.each(['invalid-json', '{}', '{"state":{},"version":"6"}'])('retains malformed saves: %s', (value) => {
    identifyPreviousSave();
    const storage = memoryStorage({ 'previous-game': value });
    migrateStorageNamespace(storage);
    expect(storage.getItem('previous-game')).toBe(value);
    expect(storage.getItem(GAME_STORAGE_KEY)).toBeNull();
  });

  it('retains the previous save if the new write fails or is discarded', () => {
    identifyPreviousSave();
    const storage = memoryStorage({ 'previous-game': save });
    vi.spyOn(storage, 'setItem').mockImplementationOnce(() => { throw new Error('Quota'); });
    expect(() => migrateStorageNamespace(storage)).not.toThrow();
    expect(storage.getItem('previous-game')).toBe(save);
    vi.spyOn(storage, 'setItem').mockImplementation(() => {});
    migrateStorageNamespace(storage);
    expect(storage.getItem('previous-game')).toBe(save);
  });

  it('does not import unknown namespaces or throw when storage is inaccessible', () => {
    const storage = memoryStorage({ unknown: save });
    migrateStorageNamespace(storage);
    expect(storage.getItem(GAME_STORAGE_KEY)).toBeNull();
    vi.spyOn(storage, 'getItem').mockImplementation(() => { throw new Error('Blocked'); });
    expect(() => migrateStorageNamespace(storage)).not.toThrow();
  });
});
