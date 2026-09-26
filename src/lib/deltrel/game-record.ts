import type { AiSearchSettings, EarlyGameOutcome } from '../store';
import { isControllerType, type PlayerControllers } from './ai/controllers';
import type { DeltrelAiSearchBudget } from './ai/decision';
import { isSupportedRings } from './board';
import { scoreCompletionBounds } from './completion-bounds';
import {
  applyAction,
  configHandicap,
  initialState,
  type GameAction,
  type GameConfig,
  type GameState,
} from './game';
import { coordinateLabel } from './notation';
import { DELTREL_MAX_HANDICAP, DELTREL_RULES_SCHEMA_ID } from './rules';
import { validateTerminalWinner } from './scoring';
import { buildTimeline } from './timeline';

/** Deltrel Game Notation: a portable, replayable main line with explicit results. */
export const DGN_VERSION = 1;
export const MAX_GAME_RECORD_TEXT_LENGTH = 65_536;

export interface GameRecord {
  id: string;
  createdAt: string;
  updatedAt: string;
  config: GameConfig;
  controllers: PlayerControllers;
  aiSearchSettings: AiSearchSettings;
  log: GameAction[];
  earlyOutcome: EarlyGameOutcome | null;
}

export interface GameRecordResult {
  result: '*' | '1-0' | '0-1';
  termination: 'unfinished' | 'board-full' | 'clinch' | 'resignation';
  winner: 0 | 1 | null;
}

const HEADER_NAMES = [
  'DGN', 'Rules', 'Date', 'Updated', 'Player1', 'Player2',
  'Player1Type', 'Player2Type', 'Rings', 'Mode', 'PieRule', 'Handicap',
  'LocalAI', 'LocalSearch', 'CloudAI', 'CloudSearch', 'Result', 'Termination',
] as const;
type HeaderName = (typeof HEADER_NAMES)[number];

function fail(message: string): never {
  throw new Error(message);
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

function date(value: unknown, field: string): string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value) ||
      !Number.isFinite(Date.parse(value)) || new Date(value).toISOString() !== value) {
    fail(`${field} must be a valid UTC date, such as 2026-09-26T12:00:00.000Z.`);
  }
  return value;
}

function budget(value: unknown, field: string): DeltrelAiSearchBudget {
  // These describe historical search settings, not permission to run a search.
  // Runtime capabilities apply separately when a player starts an engine.
  if (!object(value) || !exactKeys(value, ['simulations', 'maxConsidered']) ||
      ![value.simulations, value.maxConsidered].every((number) =>
        typeof number === 'number' && Number.isSafeInteger(number) && number > 0 && number <= 0xffff_ffff)) {
    fail(`${field} must contain positive whole-number simulations and maxConsidered values.`);
  }
  return { simulations: value.simulations as number, maxConsidered: value.maxConsidered as number };
}

function effort(value: DeltrelAiSearchBudget): 'standard' | 'deep' | 'custom' {
  if (value.simulations === 544 && value.maxConsidered === 16) return 'standard';
  if (value.simulations === 4_096 && value.maxConsidered === 64) return 'deep';
  return 'custom';
}

function readConfig(value: unknown): GameConfig {
  if (!object(value) || !exactKeys(value, [
    'rings', 'mode', 'pieRule', 'playerNames', ...('handicap' in value ? ['handicap'] : []),
  ])) fail('Game settings are missing or invalid.');
  if (!isSupportedRings(value.rings)) fail('Rings must be 4, 6, 8, or 10.');
  if (value.mode !== 'classic' && value.mode !== 'double') fail('Mode must be classic or double.');
  if (typeof value.pieRule !== 'boolean') fail('PieRule must be true or false.');
  const handicap = value.handicap === undefined ? 1 : value.handicap;
  if (typeof handicap !== 'number' || !Number.isInteger(handicap) || handicap < 1 || handicap > DELTREL_MAX_HANDICAP) {
    fail(`Handicap must be a whole number from 1 to ${DELTREL_MAX_HANDICAP}.`);
  }
  if (value.pieRule && handicap !== 1) fail('Handicap openings cannot use the pie rule.');
  if (!Array.isArray(value.playerNames) || value.playerNames.length !== 2 ||
      ![value.playerNames[0], value.playerNames[1]].every((name) => typeof name === 'string')) {
    fail('Both player names must be text.');
  }
  return {
    rings: value.rings,
    mode: value.mode,
    pieRule: value.pieRule,
    handicap,
    playerNames: [value.playerNames[0], value.playerNames[1]],
  };
}

function outcome(value: unknown): EarlyGameOutcome | null {
  if (value === null) return null;
  if (!object(value) || (value.winner !== 0 && value.winner !== 1) ||
      (value.loser !== 0 && value.loser !== 1) || value.winner === value.loser) {
    fail('The game outcome must name different winning and losing players.');
  }
  if (value.reason === 'resignation' && exactKeys(value, ['reason', 'winner', 'loser'])) {
    return { reason: 'resignation', winner: value.winner, loser: value.loser };
  }
  if (value.reason === 'clinch' && exactKeys(value, ['reason', 'winner', 'loser', 'emptyNodes']) &&
      typeof value.emptyNodes === 'number' && Number.isSafeInteger(value.emptyNodes) && value.emptyNodes > 0) {
    return { reason: 'clinch', winner: value.winner, loser: value.loser, emptyNodes: value.emptyNodes };
  }
  return fail('The game outcome must be resignation or clinch.');
}

function resultFor(game: GameState, earlyOutcome: EarlyGameOutcome | null): GameRecordResult {
  if (game.over) {
    if (earlyOutcome) fail('A full board must use the board-full result.');
    const { winner } = validateTerminalWinner(game.board, game.stones);
    return { result: winner === 0 ? '1-0' : '0-1', termination: 'board-full', winner };
  }
  if (!earlyOutcome) return { result: '*', termination: 'unfinished', winner: null };
  if (earlyOutcome.reason === 'clinch') {
    const bounds = scoreCompletionBounds(game.board, game.stones);
    if (game.canSwap || bounds.guaranteedWinner !== earlyOutcome.winner || bounds.emptyNodes !== earlyOutcome.emptyNodes) {
      fail('The claimed clinch is not proven by this position.');
    }
  }
  return {
    result: earlyOutcome.winner === 0 ? '1-0' : '0-1',
    termination: earlyOutcome.reason,
    winner: earlyOutcome.winner,
  };
}

function readRecord(value: unknown): { record: GameRecord; result: GameRecordResult } {
  if (!object(value) || !exactKeys(value, [
    'id', 'createdAt', 'updatedAt', 'config', 'controllers', 'aiSearchSettings', 'log', 'earlyOutcome',
  ])) fail('The saved game has missing or unexpected fields.');
  if (typeof value.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value.id)) {
    fail('The saved game ID is invalid.');
  }
  const createdAt = date(value.createdAt, 'Date');
  const updatedAt = date(value.updatedAt, 'Updated');
  if (updatedAt < createdAt) fail('Updated cannot be earlier than Date.');
  const config = readConfig(value.config);
  if (!Array.isArray(value.controllers) || value.controllers.length !== 2 ||
      !isControllerType(value.controllers[0]) || !isControllerType(value.controllers[1])) {
    fail('Player types must be human, local, or server.');
  }
  if (!object(value.aiSearchSettings) || !exactKeys(value.aiSearchSettings, ['local', 'server'])) {
    fail('Both local and cloud AI search settings are required.');
  }
  const aiSearchSettings = {
    local: budget(value.aiSearchSettings.local, 'LocalSearch'),
    server: budget(value.aiSearchSettings.server, 'CloudSearch'),
  };
  let game = initialState(config);
  if (!Array.isArray(value.log) || value.log.length > game.board.n + 1) fail('The game contains too many moves.');
  const log: GameAction[] = [];
  for (const [index, action] of value.log.entries()) {
    if (!object(action)) fail(`Move ${index + 1} is invalid.`);
    let parsed: GameAction;
    if (action.type === 'swap' && exactKeys(action, ['type'])) parsed = { type: 'swap' };
    else if (action.type === 'place' && exactKeys(action, ['type', 'node']) &&
        typeof action.node === 'number' && Number.isSafeInteger(action.node)) parsed = { type: 'place', node: action.node };
    else fail(`Move ${index + 1} is invalid.`);
    try {
      game = applyAction(game, parsed);
    } catch {
      fail(`Move ${index + 1} is illegal in this position.`);
    }
    log.push(parsed);
  }
  const earlyOutcome = outcome(value.earlyOutcome);
  const result = resultFor(game, earlyOutcome);
  return {
    record: {
      id: value.id, createdAt, updatedAt, config,
      controllers: [value.controllers[0], value.controllers[1]],
      aiSearchSettings, log, earlyOutcome,
    },
    result,
  };
}

/** Strict, non-repairing boundary for browser archive data. Returns an owned copy. */
export function validateGameRecord(value: unknown): GameRecord | null {
  try {
    return readRecord(value).record;
  } catch {
    return null;
  }
}

/** A lead or a proven but unaccepted clinch still has an unfinished result. */
export function gameRecordResult(record: GameRecord): GameRecordResult {
  return readRecord(record).result;
}

export function serializeGameRecord(value: GameRecord): string {
  const { record, result } = readRecord(value);
  const { config, aiSearchSettings } = record;
  const searchText = (search: DeltrelAiSearchBudget) => `${search.simulations}/${search.maxConsidered}`;
  const headers: Record<HeaderName, string> = {
    DGN: String(DGN_VERSION), Rules: DELTREL_RULES_SCHEMA_ID,
    Date: record.createdAt, Updated: record.updatedAt,
    Player1: config.playerNames[0], Player2: config.playerNames[1],
    Player1Type: record.controllers[0], Player2Type: record.controllers[1],
    Rings: String(config.rings), Mode: config.mode, PieRule: String(config.pieRule),
    Handicap: String(configHandicap(config)),
    LocalAI: effort(aiSearchSettings.local), LocalSearch: searchText(aiSearchSettings.local),
    CloudAI: effort(aiSearchSettings.server), CloudSearch: searchText(aiSearchSettings.server),
    Result: result.result, Termination: result.termination,
  };
  const turns = buildTimeline(config, record.log).turns.map((turn) =>
    `${turn.turnNumber + 1}. ${turn.entries.map(({ action }) =>
      action.type === 'swap' ? 'swap' : coordinateLabel(action.node, config.rings)).join(',')}`,
  );
  // One line per pair of turns makes long games readable while remaining easy to paste.
  const moveLines = [];
  for (let index = 0; index < turns.length; index += 2) moveLines.push(turns.slice(index, index + 2).join(' '));
  moveLines.push(result.result);
  const text = `${HEADER_NAMES.map((name) => `[${name} ${JSON.stringify(headers[name])}]`).join('\n')}\n\n${moveLines.join('\n')}\n`;
  if (text.length > MAX_GAME_RECORD_TEXT_LENGTH) {
    fail('This game has unusually long player names and exceeds the 65,536-character sharing limit.');
  }
  return text;
}

function positiveInteger(text: string, field: string): number {
  if (!/^[1-9]\d{0,9}$/.test(text)) fail(`${field} must be a positive whole number.`);
  return Number(text);
}

function searchHeader(text: string, label: string, declaredEffort: string): DeltrelAiSearchBudget {
  const parts = text.split('/');
  if (parts.length !== 2) fail(`${label} must use simulations/maxConsidered, such as 544/16.`);
  const search = budget({
    simulations: positiveInteger(parts[0], label),
    maxConsidered: positiveInteger(parts[1], label),
  }, label);
  if (declaredEffort !== effort(search)) fail(`${label} does not match its standard, deep, or custom AI setting.`);
  return search;
}

/** Parse one bounded DGN main line; no moves or contradictory results are repaired. */
export function parseGameRecord(text: string): GameRecord {
  if (typeof text !== 'string' || text.length > MAX_GAME_RECORD_TEXT_LENGTH) {
    fail(`Game text must be at most ${MAX_GAME_RECORD_TEXT_LENGTH.toLocaleString('en-US')} characters.`);
  }
  const headers = new Map<HeaderName, string>();
  const moves: string[] = [];
  let inMoves = false;
  for (const raw of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (!inMoves && line.startsWith('[')) {
      const match = /^\[([A-Za-z][A-Za-z0-9]*)\s+("(?:[^"\\]|\\.)*")\]$/.exec(line);
      if (!match) fail('A header is malformed. Use [Name "value"].');
      const name = match[1] as HeaderName;
      if (!HEADER_NAMES.includes(name)) fail(`Unknown header: ${name}.`);
      if (headers.has(name)) fail(`Duplicate header: ${name}.`);
      let value: unknown;
      try { value = JSON.parse(match[2]); } catch { fail(`Invalid quoted text in ${name}.`); }
      if (typeof value !== 'string') fail(`${name} must contain quoted text.`);
      headers.set(name, value);
    } else {
      inMoves = true;
      moves.push(line);
    }
  }
  for (const name of HEADER_NAMES) if (!headers.has(name)) fail(`Missing header: ${name}.`);
  const header = (name: HeaderName): string => headers.get(name)!;
  if (header('DGN') !== String(DGN_VERSION)) fail(`Unsupported DGN version: ${header('DGN')}. This app reads DGN 1.`);
  if (header('Rules') !== DELTREL_RULES_SCHEMA_ID) fail('The game uses an unsupported rules version.');
  if (!['true', 'false'].includes(header('PieRule'))) fail('PieRule must be true or false.');
  const config = readConfig({
    rings: positiveInteger(header('Rings'), 'Rings'), mode: header('Mode'),
    pieRule: header('PieRule') === 'true', handicap: positiveInteger(header('Handicap'), 'Handicap'),
    playerNames: [header('Player1'), header('Player2')],
  });
  const aiSearchSettings = {
    local: searchHeader(header('LocalSearch'), 'LocalSearch', header('LocalAI')),
    server: searchHeader(header('CloudSearch'), 'CloudSearch', header('CloudAI')),
  };
  let game = initialState(config);
  const tokens = moves.join(' ').trim().split(/\s+/);
  if (tokens.length > (game.board.n + 1) * 2 + 1) fail('The game contains too many moves.');
  const finalResult = tokens.pop();
  if (finalResult !== '*' && finalResult !== '1-0' && finalResult !== '0-1') fail('End the moves with *, 1-0, or 0-1.');
  if (finalResult !== header('Result')) fail('The final result and Result header disagree.');
  if (tokens.length % 2 !== 0) fail('Each turn needs a number and its moves, such as 1. A10.');
  const log: GameAction[] = [];
  for (let index = 0; index < tokens.length; index += 2) {
    const turn = game.turnCount + 1;
    if (game.over) fail('Moves cannot follow a full board.');
    if (game.midTurn) fail(`Turn ${turn} is incomplete; join its placements with commas.`);
    if (tokens[index] !== `${turn}.`) fail(`Expected turn ${turn}., found ${tokens[index]}.`);
    const placements = tokens[index + 1].split(',');
    if (placements.length > DELTREL_MAX_HANDICAP) fail(`Turn ${turn} contains too many placements.`);
    for (const move of placements) {
      if (game.over || game.turnCount + 1 !== turn) fail(`Turn ${turn} contains too many moves.`);
      let action: GameAction;
      if (move === 'swap') {
        if (placements.length !== 1 || !game.canSwap) fail(`Turn ${turn}: swap is unavailable here.`);
        action = { type: 'swap' };
      } else {
        const node = game.board.labelToId.get(move);
        if (node === undefined) fail(`Turn ${turn}: ${move || '(empty)'} is not a coordinate on this board.`);
        if (game.stones[node] !== -1) fail(`Turn ${turn}: ${move} is already occupied.`);
        action = { type: 'place', node };
      }
      game = applyAction(game, action);
      log.push(action);
    }
  }
  const termination = header('Termination');
  if (!['unfinished', 'board-full', 'clinch', 'resignation'].includes(termination)) fail(`Unknown termination: ${termination}.`);
  const winner = finalResult === '*' ? null : finalResult === '1-0' ? 0 : 1;
  let earlyOutcome: EarlyGameOutcome | null = null;
  if (termination === 'clinch' || termination === 'resignation') {
    if (winner === null) fail(`${termination} needs a winning result, 1-0 or 0-1.`);
    const loser = (1 - winner) as 0 | 1;
    earlyOutcome = termination === 'clinch'
      ? { reason: 'clinch', winner, loser, emptyNodes: game.board.n - game.stonesPlaced }
      : { reason: 'resignation', winner, loser };
  }
  const result = resultFor(game, earlyOutcome);
  if (result.result !== finalResult || result.termination !== termination) {
    fail('The result or termination does not match the final position.');
  }
  return readRecord({
    id: crypto.randomUUID(), createdAt: header('Date'), updatedAt: header('Updated'), config,
    controllers: [header('Player1Type'), header('Player2Type')], aiSearchSettings, log, earlyOutcome,
  }).record;
}
