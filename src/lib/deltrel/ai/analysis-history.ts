import { configHandicap, type GameAction, type GameConfig } from '../game';
import type { DeltrelAiAnalysis } from './decision';
import type { AtomicGameAction } from './protocol';

export const MAX_ENGINE_ANALYSES = 64;

/** An accepted estimate belongs to the position before its associated action. */
export interface StoredEngineAnalysis {
  analysis: DeltrelAiAnalysis;
  source: 'server' | 'local';
  ply: number;
  configKey: string;
  prefix: readonly GameAction[];
  action: AtomicGameAction;
  applied: boolean;
}

/** Display names do not change the meaning of a position or its forecast. */
export function analysisConfigKey(config: GameConfig): string {
  return JSON.stringify([
    config.rings,
    config.mode,
    configHandicap(config),
    config.pieRule,
  ]);
}

function validPrefix(entry: StoredEngineAnalysis): boolean {
  return Number.isSafeInteger(entry.ply) && entry.ply >= 0 &&
    entry.ply === entry.prefix.length;
}

/** Keep the latest completed result for each position, in completion order. */
export function recordEngineAnalysis(
  history: readonly StoredEngineAnalysis[],
  entry: StoredEngineAnalysis,
): StoredEngineAnalysis[] {
  if (!validPrefix(entry)) return [...history];
  const retained = history.filter((previous) =>
    previous.configKey !== entry.configKey ||
    previous.analysis.stateHash !== entry.analysis.stateHash,
  );
  return [...retained, {
    ...entry,
    prefix: entry.prefix.map((action) => ({ ...action })),
    action: { ...entry.action },
  }].slice(-MAX_ENGINE_ANALYSES);
}

interface AnalysisSelectionOptions {
  config: GameConfig;
  log: readonly GameAction[];
  positionHash: string | null;
  ply: number;
  allowPrevious: boolean;
}

function matchesBranch(entry: StoredEngineAnalysis, log: readonly GameAction[]): boolean {
  return entry.prefix.every((action, index) => {
    const current = log[index];
    return action.type === 'swap'
      ? current?.type === 'swap'
      : current?.type === 'place' && action.node === current.node;
  });
}

/** Exact history positions never borrow a forecast from another branch. */
export function selectEngineAnalysis(
  history: readonly StoredEngineAnalysis[],
  { config, log, positionHash, ply, allowPrevious }: AnalysisSelectionOptions,
): { entry: StoredEngineAnalysis; isExact: boolean } | null {
  if (!Number.isSafeInteger(ply) || ply < 0 || ply > log.length) return null;
  const configKey = analysisConfigKey(config);
  let previous: StoredEngineAnalysis | null = null;
  for (let index = history.length - 1; index >= 0; index--) {
    const entry = history[index];
    if (
      !validPrefix(entry) || entry.configKey !== configKey ||
      entry.ply > ply || !matchesBranch(entry, log)
    ) continue;
    if (positionHash !== null && entry.analysis.stateHash === positionHash) {
      return { entry, isExact: true };
    }
    // Historical reanalysis can finish after a more recent live position.
    // Prefer the closest earlier position; reverse iteration keeps the latest
    // completed estimate when multiple entries share that ply.
    if (previous === null || entry.ply > previous.ply) previous = entry;
  }
  return allowPrevious && previous ? { entry: previous, isExact: false } : null;
}
