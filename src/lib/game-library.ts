'use client';

import { useSyncExternalStore } from 'react';
import { validateGameRecord, type GameRecord } from './deltrel/game-record';

/** Each game is an independent atomic write; a large library is never rewritten. */
export const GAME_LIBRARY_PREFIX = 'deltrel-game-v1:';
// Far above a complete normal game; bound damaged/untrusted storage before parsing.
export const MAX_STORED_GAME_LENGTH = 1_048_576;

export interface GameLibrarySnapshot {
  records: GameRecord[];
  error: string | null;
}

const serverSnapshot: GameLibrarySnapshot = { records: [], error: null };
let snapshot = serverSnapshot;
let loaded = false;
let readError: string | null = null;
let writeError: string | null = null;
const records = new Map<string, GameRecord>();
// Failed writes remain available to copy/share, and retry on subsequent saves.
const pending = new Map<string, GameRecord>();
const listeners = new Set<() => void>();
const unavailableMessage = 'Local game storage is unavailable. Copy games you want to keep before closing this page.';
const saveFailureMessage = 'Some games could not be saved on this device. They are available in this page for now. Copy them before closing, or free browser storage and retry.';

function storage(): Storage {
  return window.localStorage;
}

function parseStoredRecord(raw: string, key: string): GameRecord | null {
  if (raw.length > MAX_STORED_GAME_LENGTH) return null;
  const envelope: unknown = JSON.parse(raw);
  const record =
    typeof envelope === 'object' && envelope !== null &&
    'version' in envelope && envelope.version === 1 && 'record' in envelope
      ? validateGameRecord(envelope.record)
      : null;
  return record && key === `${GAME_LIBRARY_PREFIX}${record.id}` ? record : null;
}

/** A live game opened in two tabs must fork before overwriting the other line. */
export function hasConflictingSavedGame(value: GameRecord): boolean {
  if (typeof window === 'undefined' || pending.has(value.id)) return false;
  try {
    const key = `${GAME_LIBRARY_PREFIX}${value.id}`;
    const raw = storage().getItem(key);
    if (raw === null) return false;
    const saved = parseStoredRecord(raw, key);
    const expected = validateGameRecord(value);
    return saved !== null && expected !== null && JSON.stringify(saved) !== JSON.stringify(expected);
  } catch {
    // Saving will report unavailable storage through the regular failure path.
    return false;
  }
}

function publish(): void {
  snapshot = {
    records: [...records.values()].sort((a, b) =>
      b.updatedAt.localeCompare(a.updatedAt) || a.id.localeCompare(b.id)),
    error: writeError ?? readError,
  };
  for (const listener of listeners) listener();
}

function refresh(): void {
  if (typeof window === 'undefined') return;
  loaded = true;
  const saved = new Map<string, GameRecord>();
  let damaged = 0;
  try {
    const local = storage();
    for (let index = 0; index < local.length; index++) {
      const key = local.key(index);
      if (!key?.startsWith(GAME_LIBRARY_PREFIX)) continue;
      try {
        const raw = local.getItem(key);
        if (raw === null) continue;
        const record = parseStoredRecord(raw, key);
        if (!record) {
          damaged++;
          continue;
        }
        saved.set(record.id, record);
      } catch {
        // Leave the damaged entry intact; other games remain independently usable.
        damaged++;
      }
    }
    records.clear();
    for (const [id, record] of saved) records.set(id, record);
    readError = damaged > 0
      ? `${damaged} saved ${damaged === 1 ? 'game could' : 'games could'} not be read. Other games are still available; unreadable data has been kept.`
      : null;
  } catch {
    readError = unavailableMessage;
  }
  for (const [id, record] of pending) records.set(id, record);
  publish();
}

function ensureLoaded(): void {
  if (!loaded && typeof window !== 'undefined') refresh();
}

function getSnapshot(): GameLibrarySnapshot {
  ensureLoaded();
  return snapshot;
}

export function listGameRecords(): GameRecord[] {
  return [...getSnapshot().records];
}

export function getGameLibraryStatus(): string | null {
  return getSnapshot().error;
}

/** Explicit retry, also used automatically after every new game change. */
export function retryGameLibrarySaves(): boolean {
  if (typeof window === 'undefined') return false;
  ensureLoaded();
  let failed = false;
  for (const [id, record] of pending) {
    try {
      const local = storage();
      const key = `${GAME_LIBRARY_PREFIX}${id}`;
      const serialized = JSON.stringify({ version: 1, record });
      if (serialized.length > MAX_STORED_GAME_LENGTH) throw new Error('Record exceeds the storage limit');
      local.setItem(key, serialized);
      if (local.getItem(key) !== serialized) throw new Error('Storage did not retain the game');
      pending.delete(id);
    } catch {
      failed = true;
    }
  }
  writeError = failed ? saveFailureMessage : null;
  if (!failed && readError === unavailableMessage) readError = null;
  publish();
  return !failed;
}

export function saveGameRecord(value: GameRecord): boolean {
  if (typeof window === 'undefined') return false;
  ensureLoaded();
  const record = validateGameRecord(value);
  if (!record) {
    writeError = 'This game could not be saved because its record is invalid.';
    publish();
    return false;
  }
  records.set(record.id, record);
  pending.set(record.id, record);
  return retryGameLibrarySaves();
}

export function deleteGameRecord(id: string): boolean {
  if (typeof window === 'undefined') return false;
  ensureLoaded();
  try {
    const local = storage();
    const key = `${GAME_LIBRARY_PREFIX}${id}`;
    local.removeItem(key);
    if (local.getItem(key) !== null) throw new Error('Storage did not remove the game');
    records.delete(id);
    pending.delete(id);
    writeError = pending.size > 0 ? saveFailureMessage : null;
    publish();
    return true;
  } catch {
    writeError = 'This game could not be removed from local storage. Please try again.';
    publish();
    return false;
  }
}

function onStorage(event: StorageEvent): void {
  if (event.key === null || event.key.startsWith(GAME_LIBRARY_PREFIX)) refresh();
}

export function subscribeGameLibrary(listener: () => void): () => void {
  ensureLoaded();
  listeners.add(listener);
  if (listeners.size === 1 && typeof window !== 'undefined') {
    window.addEventListener('storage', onStorage);
    // A different tab may have changed the library while no view was subscribed.
    refresh();
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && typeof window !== 'undefined') {
      window.removeEventListener('storage', onStorage);
    }
  };
}

export function useGameLibrary(): GameLibrarySnapshot {
  return useSyncExternalStore(subscribeGameLibrary, getSnapshot, () => serverSnapshot);
}
