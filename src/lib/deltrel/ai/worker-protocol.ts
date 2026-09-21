import {
  DELTREL_ACTION_LAYOUT_SCHEMA_ID,
  DELTREL_FEATURE_SCHEMA_ID,
  DELTREL_MAX_HANDICAP,
  DELTREL_RULES_HASH,
  DELTREL_RULES_SCHEMA_ID,
} from '../rules';
import { getBoard, isSupportedRings } from '../board';
import {
  parseUnboundDeltrelAiDecision,
  type DeltrelAiDecision,
  type DeltrelAiSearchBudget,
} from './decision';
import { DeltrelAiError, asDeltrelAiError, type DeltrelAiErrorCode } from './errors';
import type { LocalAiProgress, LocalAiReadyInfo } from './local-ai-status';
import {
  MAX_BROWSER_AI_MAX_CONSIDERED,
  MAX_BROWSER_AI_SIMULATIONS,
} from './manifest';
import {
  DELTREL_ACTION_LAYOUT_VERSION,
  DELTREL_AI_PROTOCOL_SCHEMA_ID,
  DELTREL_AI_PROTOCOL_VERSION,
  DELTREL_FEATURE_SCHEMA_HASH,
  DELTREL_FEATURE_SCHEMA_VERSION,
  semanticStateHash,
  type DeltrelAiRequest,
  type DeltrelAiSemanticState,
} from './protocol';

export type DeltrelAiWorkerCommand =
  | {
      type: 'choose';
      taskId: string;
      request: DeltrelAiRequest;
      search: DeltrelAiSearchBudget | null;
    }
  | { type: 'prepare'; taskId: string }
  | { type: 'cancel'; taskId: string };

export const DELTREL_AI_WORKER_PROTOCOL_VERSION = 4 as const;

/** Completed search work, never an elapsed-time heartbeat or estimated effort. */
export interface LocalAiSearchProgress {
  completedSimulations: number;
  totalSimulations: number;
}

export type DeltrelAiWorkerEvent =
  | { type: 'ready'; protocolVersion: typeof DELTREL_AI_WORKER_PROTOCOL_VERSION }
  | { type: 'progress'; taskId: string; progress: LocalAiProgress }
  | { type: 'search-progress'; taskId: string; progress: LocalAiSearchProgress }
  | { type: 'prepared'; taskId: string; info: LocalAiReadyInfo }
  | { type: 'result'; taskId: string; decision: DeltrelAiDecision }
  | {
      type: 'error';
      taskId: string;
      error: { code: DeltrelAiErrorCode; message: string; retryable: boolean };
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isTaskId(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 200;
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function isIntegerArray(value: unknown, minimum: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) => typeof item === 'number' && Number.isInteger(item) && item >= minimum,
    )
  );
}

export function parseBrowserSearchBudget(value: unknown): DeltrelAiSearchBudget {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['simulations', 'maxConsidered']) ||
    typeof value.simulations !== 'number' ||
    !Number.isSafeInteger(value.simulations) ||
    value.simulations <= 0 ||
    value.simulations > MAX_BROWSER_AI_SIMULATIONS ||
    typeof value.maxConsidered !== 'number' ||
    !Number.isSafeInteger(value.maxConsidered) ||
    value.maxConsidered <= 0 ||
    value.maxConsidered > MAX_BROWSER_AI_MAX_CONSIDERED
  ) {
    throw new DeltrelAiError(
      'protocol',
      `Browser AI search budget must use simulations in 1..${MAX_BROWSER_AI_SIMULATIONS} ` +
        `and max-considered in 1..${MAX_BROWSER_AI_MAX_CONSIDERED}.`,
    );
  }
  return {
    simulations: value.simulations,
    maxConsidered: value.maxConsidered,
  };
}

function parseNodeList(value: unknown, nodeCount: number): number[] {
  if (
    !isIntegerArray(value, 0) ||
    value.some((node) => node >= nodeCount) ||
    new Set(value).size !== value.length
  ) {
    throw new DeltrelAiError('protocol', 'Worker received an invalid placement history.');
  }
  return [...value];
}

function parseSemanticState(value: unknown): DeltrelAiSemanticState {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'rings',
      'stones',
      'toMove',
      'movesLeft',
      'opening',
      'terminal',
      'mode',
      'handicap',
      'pie',
      'swapAvailable',
      'swapped',
      'history',
    ]) ||
    !isSupportedRings(value.rings) ||
    !isIntegerArray(value.stones, -1) ||
    value.stones.some((stone) => stone > 1) ||
    value.stones.length !== getBoard(value.rings).n ||
    (value.toMove !== 0 && value.toMove !== 1) ||
    typeof value.movesLeft !== 'number' ||
    !Number.isInteger(value.movesLeft) ||
    value.movesLeft < 0 ||
    value.movesLeft > DELTREL_MAX_HANDICAP ||
    typeof value.opening !== 'boolean' ||
    typeof value.terminal !== 'boolean' ||
    (value.mode !== 'classic' && value.mode !== 'double') ||
    typeof value.handicap !== 'number' ||
    !Number.isInteger(value.handicap) ||
    value.handicap < 1 ||
    value.handicap > DELTREL_MAX_HANDICAP ||
    typeof value.pie !== 'boolean' ||
    (value.pie && value.handicap !== 1) ||
    typeof value.swapAvailable !== 'boolean' ||
    typeof value.swapped !== 'boolean' ||
    (value.swapAvailable && (!value.pie || value.toMove !== 1)) ||
    !isRecord(value.history) ||
    !hasExactKeys(value.history, [
      'currentTurn',
      'previousTurn',
      'ownPreviousTurn',
      'handicapStones',
    ])
  ) {
    throw new DeltrelAiError('protocol', 'Worker received an invalid semantic state.');
  }
  const nodeCount = value.stones.length;
  return {
    rings: value.rings,
    stones: [...value.stones],
    toMove: value.toMove,
    movesLeft: value.movesLeft,
    opening: value.opening,
    terminal: value.terminal,
    mode: value.mode,
    handicap: value.handicap,
    pie: value.pie,
    swapAvailable: value.swapAvailable,
    swapped: value.swapped,
    history: {
      currentTurn: parseNodeList(value.history.currentTurn, nodeCount),
      previousTurn: parseNodeList(value.history.previousTurn, nodeCount),
      ownPreviousTurn: parseNodeList(value.history.ownPreviousTurn, nodeCount),
      handicapStones: parseNodeList(value.history.handicapStones, nodeCount),
    },
  };
}

function parseRequest(value: unknown): DeltrelAiRequest {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'schema',
      'version',
      'requestId',
      'rulesSchema',
      'rulesHash',
      'featureSchema',
      'featureSchemaVersion',
      'featureSchemaHash',
      'actionLayout',
      'actionLayoutVersion',
      'stateHash',
      'state',
      'actionLog',
      'legalActions',
    ]) ||
    value.schema !== DELTREL_AI_PROTOCOL_SCHEMA_ID ||
    value.version !== DELTREL_AI_PROTOCOL_VERSION ||
    !isTaskId(value.requestId) ||
    value.rulesSchema !== DELTREL_RULES_SCHEMA_ID ||
    value.rulesHash !== DELTREL_RULES_HASH ||
    value.featureSchema !== DELTREL_FEATURE_SCHEMA_ID ||
    value.featureSchemaVersion !== DELTREL_FEATURE_SCHEMA_VERSION ||
    value.featureSchemaHash !== DELTREL_FEATURE_SCHEMA_HASH ||
    value.actionLayout !== DELTREL_ACTION_LAYOUT_SCHEMA_ID ||
    value.actionLayoutVersion !== DELTREL_ACTION_LAYOUT_VERSION ||
    typeof value.stateHash !== 'string' ||
    !isIntegerArray(value.actionLog, 0) ||
    !isIntegerArray(value.legalActions, 0)
  ) {
    throw new DeltrelAiError('protocol', 'Worker received an incompatible AI request.');
  }
  const state = parseSemanticState(value.state);
  if (semanticStateHash(state) !== value.stateHash) {
    throw new DeltrelAiError('protocol', 'Worker request state hash does not match its state.');
  }
  const nodeCount = state.stones.length;
  const actionLog = [...value.actionLog];
  const legalActions = [...value.legalActions];
  if (
    state.terminal ||
    legalActions.length === 0 ||
    // Placements are dense node ids; the pie swap is coded as the node count.
    actionLog.some((action) => action > nodeCount) ||
    legalActions.some(
      (action, index) =>
        action >= nodeCount ||
        state.stones[action] !== -1 ||
        (index > 0 && legalActions[index - 1] >= action),
    )
  ) {
    throw new DeltrelAiError('protocol', 'Worker received inconsistent AI actions.');
  }
  return {
    schema: DELTREL_AI_PROTOCOL_SCHEMA_ID,
    version: DELTREL_AI_PROTOCOL_VERSION,
    requestId: value.requestId,
    rulesSchema: DELTREL_RULES_SCHEMA_ID,
    rulesHash: DELTREL_RULES_HASH,
    featureSchema: DELTREL_FEATURE_SCHEMA_ID,
    featureSchemaVersion: DELTREL_FEATURE_SCHEMA_VERSION,
    featureSchemaHash: DELTREL_FEATURE_SCHEMA_HASH,
    actionLayout: DELTREL_ACTION_LAYOUT_SCHEMA_ID,
    actionLayoutVersion: DELTREL_ACTION_LAYOUT_VERSION,
    stateHash: value.stateHash,
    state,
    actionLog,
    legalActions,
  };
}

export function parseWorkerCommand(value: unknown): DeltrelAiWorkerCommand {
  if (!isRecord(value) || !isTaskId(value.taskId)) {
    throw new DeltrelAiError('protocol', 'Worker command is invalid.');
  }
  if (value.type === 'cancel') {
    if (!hasExactKeys(value, ['type', 'taskId'])) {
      throw new DeltrelAiError('protocol', 'Worker cancel command is invalid.');
    }
    return { type: 'cancel', taskId: value.taskId };
  }
  if (value.type === 'prepare' && hasExactKeys(value, ['type', 'taskId'])) {
    return { type: 'prepare', taskId: value.taskId };
  }
  if (value.type === 'choose') {
    if (!hasExactKeys(value, ['type', 'taskId', 'request', 'search'])) {
      throw new DeltrelAiError('protocol', 'Worker choose command is invalid.');
    }
    const request = parseRequest(value.request);
    if (request.requestId !== value.taskId) {
      throw new DeltrelAiError('stale', 'Worker task identity does not match its AI request.');
    }
    return {
      type: 'choose',
      taskId: value.taskId,
      request,
      search: value.search === null ? null : parseBrowserSearchBudget(value.search),
    };
  }
  throw new DeltrelAiError('protocol', 'Worker command type is invalid.');
}

export function parseWorkerEvent(value: unknown): DeltrelAiWorkerEvent {
  if (!isRecord(value)) {
    throw new DeltrelAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (
    value.type === 'ready' &&
    value.protocolVersion === DELTREL_AI_WORKER_PROTOCOL_VERSION &&
    hasExactKeys(value, ['type', 'protocolVersion'])
  ) {
    return { type: 'ready', protocolVersion: DELTREL_AI_WORKER_PROTOCOL_VERSION };
  }
  if (!isTaskId(value.taskId)) {
    throw new DeltrelAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (value.type === 'search-progress' && hasExactKeys(value, ['type', 'taskId', 'progress'])) {
    const progress = value.progress;
    if (isRecord(progress) && hasExactKeys(progress, ['completedSimulations', 'totalSimulations']) &&
        typeof progress.completedSimulations === 'number' && Number.isSafeInteger(progress.completedSimulations) &&
        typeof progress.totalSimulations === 'number' && Number.isSafeInteger(progress.totalSimulations) &&
        progress.completedSimulations > 0 && progress.completedSimulations <= progress.totalSimulations &&
        progress.totalSimulations <= MAX_BROWSER_AI_SIMULATIONS) {
      return { type: 'search-progress', taskId: value.taskId, progress: {
        completedSimulations: progress.completedSimulations,
        totalSimulations: progress.totalSimulations,
      } };
    }
  }
  if (value.type === 'progress' && hasExactKeys(value, ['type', 'taskId', 'progress'])) {
    const p = value.progress;
    if (isRecord(p) && hasExactKeys(p, ['phase', 'loadedBytes', 'totalBytes', 'modelVersion', 'cached']) &&
        ['checking', 'downloading', 'verifying', 'initializing'].includes(String(p.phase)) &&
        typeof p.loadedBytes === 'number' && Number.isSafeInteger(p.loadedBytes) && p.loadedBytes >= 0 &&
        (p.totalBytes === null || (typeof p.totalBytes === 'number' && Number.isSafeInteger(p.totalBytes) && p.totalBytes > 0 && p.loadedBytes <= p.totalBytes)) &&
        (p.modelVersion === null || isTaskId(p.modelVersion)) && typeof p.cached === 'boolean') {
      return { type: 'progress', taskId: value.taskId, progress: p as unknown as LocalAiProgress };
    }
  }
  if (value.type === 'prepared' && hasExactKeys(value, ['type', 'taskId', 'info'])) {
    const info = value.info;
    if (isRecord(info) && hasExactKeys(info, ['modelVersion', 'bytes', 'backend', 'cached']) &&
        isTaskId(info.modelVersion) && typeof info.bytes === 'number' && Number.isSafeInteger(info.bytes) && info.bytes > 0 &&
        (info.backend === 'webgpu' || info.backend === 'wasm') && typeof info.cached === 'boolean') {
      return { type: 'prepared', taskId: value.taskId, info: info as unknown as LocalAiReadyInfo };
    }
  }
  if (value.type === 'result') {
    if (!hasExactKeys(value, ['type', 'taskId', 'decision'])) {
      throw new DeltrelAiError('protocol', 'Local AI worker returned an invalid result.');
    }
    const decision = parseUnboundDeltrelAiDecision(value.decision);
    if (decision.response.requestId !== value.taskId) {
      throw new DeltrelAiError('stale', 'Local AI worker result identity is incompatible.');
    }
    return {
      type: 'result',
      taskId: value.taskId,
      decision,
    };
  }
  if (
    value.type === 'error' &&
    hasExactKeys(value, ['type', 'taskId', 'error']) &&
    isRecord(value.error) &&
    hasExactKeys(value.error, ['code', 'message', 'retryable'])
  ) {
    const code = value.error.code;
    if (
      typeof code === 'string' &&
      [
        'unavailable',
        'timeout',
        'network',
        'protocol',
        'stale',
        'illegal',
        'cancelled',
        'internal',
      ].includes(code) &&
      typeof value.error.message === 'string' &&
      typeof value.error.retryable === 'boolean'
    ) {
      return {
        type: 'error',
        taskId: value.taskId,
        error: {
          code: code as DeltrelAiErrorCode,
          message: value.error.message,
          retryable: value.error.retryable,
        },
      };
    }
  }
  throw new DeltrelAiError('protocol', 'Local AI worker returned an invalid event.');
}

export function workerErrorEvent(taskId: string, error: unknown): DeltrelAiWorkerEvent {
  const aiError = asDeltrelAiError(error);
  return {
    type: 'error',
    taskId,
    error: {
      code: aiError.code,
      message: aiError.message,
      retryable: aiError.retryable,
    },
  };
}
