import { fnv1a64 } from './deltrel/rules';

export const GAME_STORAGE_KEY = 'deltrel-v1';

// A fingerprint identifies the previous namespace without shipping its old name.
const PREVIOUS_STORAGE_FINGERPRINT = '292253f25236e69d';

type GameStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

/** Storage restrictions must not turn an otherwise legal move into an error. */
export function createResilientStorage(storage: GameStorage): GameStorage {
  return {
    getItem(key) {
      try { return storage.getItem(key); } catch { return null; }
    },
    setItem(key, value) {
      try { storage.setItem(key, value); } catch { /* Keep the live game playable. */ }
    },
    removeItem(key) {
      try { storage.removeItem(key); } catch { /* Storage may be unavailable. */ }
    },
  };
}

/** Copy a prior save only after a successful write; never overwrite a Deltrel save. */
export function migrateStorageNamespace(storage: Storage): void {
  try {
    if (storage.getItem(GAME_STORAGE_KEY) !== null) return;
    for (let index = 0; index < storage.length; index++) {
      const key = storage.key(index);
      if (key === null || fnv1a64(key) !== PREVIOUS_STORAGE_FINGERPRINT) continue;
      const saved = storage.getItem(key);
      if (saved === null) return;
      // Persistence's version migration and strict replay validation run on load.
      const envelope: unknown = JSON.parse(saved);
      if (
        typeof envelope !== 'object' || envelope === null ||
        !('state' in envelope) || !('version' in envelope) ||
        typeof envelope.version !== 'number'
      ) return;
      storage.setItem(GAME_STORAGE_KEY, saved);
      if (storage.getItem(GAME_STORAGE_KEY) === saved) storage.removeItem(key);
      return;
    }
  } catch {
    // Private browsing, full storage, or malformed data must not destroy a save.
  }
}
