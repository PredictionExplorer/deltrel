import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { getBoard, SUPPORTED_RINGS } from '../board';
import { replay, type GameAction } from '../game';
import {
  gameRecordResult,
  MAX_GAME_RECORD_TEXT_LENGTH,
  parseGameRecord,
  serializeGameRecord,
  validateGameRecord,
  type GameRecord,
} from '../game-record';
import { coordinateLabel } from '../notation';
import { validateTerminalWinner } from '../scoring';

const timestamp = '2026-09-26T12:00:00.000Z';
const places = (count: number): GameAction[] => Array.from({ length: count }, (_, node) => ({ type: 'place', node }));

function record(overrides: Partial<GameRecord> = {}): GameRecord {
  return {
    id: 'game-1', createdAt: timestamp, updatedAt: timestamp,
    config: { rings: 4, mode: 'double', pieRule: false, handicap: 1, playerNames: ['Alice', 'Bob'] },
    controllers: ['human', 'human'],
    aiSearchSettings: { local: { simulations: 544, maxConsidered: 16 }, server: { simulations: 544, maxConsidered: 16 } },
    log: places(5), earlyOutcome: null, ...overrides,
  };
}

function header(text: string, name: string, value: string): string {
  return text.replace(new RegExp(`^\\[${name} .*\\]$`, 'm'), () => `[${name} ${JSON.stringify(value)}]`);
}

function body(text: string, moves: string): string {
  return `${text.slice(0, text.indexOf('\n\n'))}\n\n${moves}\n`;
}

function replaceResult(text: string, value: string): string {
  return header(text, 'Result', value).replace(/(?:\*|1-0|0-1)\n$/, `${value}\n`);
}

describe('Deltrel Game Notation', () => {
  it('exports readable turn groups and gives an import its own local identity', () => {
    const original = record();
    const text = serializeGameRecord(original);
    const labels = original.log.map((action) => coordinateLabel((action as { node: number }).node, 4));
    expect(text).toContain(`1. ${labels[0]} 2. ${labels[1]},${labels[2]}\n3. ${labels[3]},${labels[4]}\n*\n`);
    expect(text).toContain('[DGN "1"]');
    expect(text).toContain('[Player1Type "human"]');
    expect(text).toContain('[Termination "unfinished"]');
    expect(text).not.toContain(original.id);
    const imported = parseGameRecord(text);
    expect(imported.id).not.toBe(original.id);
    expect(imported).toEqual({ ...original, id: imported.id });
    expect(serializeGameRecord(imported)).toBe(text);
  });

  it('escapes header values reversibly without permitting header injection', () => {
    const original = record();
    original.config.playerNames = ['A "quote" \\ line\n[Result "0-1"]', 'Соня 🪷\t\r'];
    const text = serializeGameRecord(original);
    expect(text.match(/^\[Result /gm)).toHaveLength(1);
    expect(parseGameRecord(text).config.playerNames).toEqual(original.config.playerNames);
  });

  it('accepts CRLF, blank lines, a UTF-8 BOM, and arbitrary whitespace between turns', () => {
    const text = serializeGameRecord(record());
    expect(parseGameRecord(`\uFEFF\n${text.replace(/\n/g, '\r\n')}`).log).toEqual(record().log);
    expect(parseGameRecord(text.replace('1. ', '1.\t')).log).toEqual(record().log);
  });

  it.each(SUPPORTED_RINGS)('round trips all coordinates on the %i-ring board', (rings) => {
    const original = record();
    original.config.rings = rings;
    original.log = places(getBoard(rings).n);
    const imported = parseGameRecord(serializeGameRecord(original));
    expect(imported.log).toEqual(original.log);
    expect(gameRecordResult(imported).termination).toBe('board-full');
  });

  it('uses the exact three-symbol ring-ten addresses A00 through E09', () => {
    const original = record();
    original.config.rings = 10;
    original.log = [{ type: 'place', node: getBoard(10).idx(0, 10, 0) },
      { type: 'place', node: getBoard(10).idx(4, 10, 9) }];
    const text = serializeGameRecord(original);
    expect(text).toContain('1. A00 2. E09');
    expect(parseGameRecord(text).log).toEqual(original.log);
    expect(() => parseGameRecord(body(text, '1. A100 *'))).toThrow(/not a coordinate/);
  });

  it('round trips every opening variant, including partial handicap and Double turns', () => {
    for (const mode of ['classic', 'double'] as const) {
      for (const handicap of [1, 2, 9]) {
        for (const length of [0, 1, 2, 3, 8, 9, 10, 11, 50]) {
          const original = record();
          original.config = { ...original.config, mode, handicap };
          original.log = places(length);
          expect(parseGameRecord(serializeGameRecord(original)).log).toEqual(original.log);
        }
      }
    }
  });

  it.each(['classic', 'double'] as const)('round trips a pie swap in %s mode', (mode) => {
    const original = record();
    original.config = { ...original.config, mode, pieRule: true };
    original.log = [{ type: 'place', node: 0 }, { type: 'swap' }, ...places(4).slice(1)];
    const text = serializeGameRecord(original);
    expect(text).toContain('2. swap');
    expect(parseGameRecord(text).log).toEqual(original.log);
  });

  it('preserves controller types and standard, deep, and exact custom search settings', () => {
    const original = record({ controllers: ['local', 'server'] });
    original.aiSearchSettings = { local: { simulations: 4_096, maxConsidered: 64 }, server: { simulations: 1234, maxConsidered: 17 } };
    const text = serializeGameRecord(original);
    expect(text).toContain('[LocalAI "deep"]');
    expect(text).toContain('[CloudAI "custom"]');
    expect(text).toContain('[CloudSearch "1234/17"]');
    expect(parseGameRecord(text).aiSearchSettings).toEqual(original.aiSearchSettings);
    expect(parseGameRecord(text).controllers).toEqual(['local', 'server']);
    expect(serializeGameRecord(record())).toContain('[LocalAI "standard"]');
  });

  it.each([0, 1] as const)('represents player %i winning by resignation', (winner) => {
    const original = record({ log: places(2), earlyOutcome: { reason: 'resignation', winner, loser: (1 - winner) as 0 | 1 } });
    const text = serializeGameRecord(original);
    expect(gameRecordResult(parseGameRecord(text))).toEqual({ result: winner === 0 ? '1-0' : '0-1', termination: 'resignation', winner });
    expect(parseGameRecord(text).earlyOutcome).toEqual(original.earlyOutcome);
  });

  it.each([0, 1] as const)('verifies the proof before accepting player %i clinching', (winner) => {
    const log = winner === 1 ? places(48) : places(50).reverse().slice(0, 49);
    const original = record({ log, earlyOutcome: { reason: 'clinch', winner, loser: (1 - winner) as 0 | 1, emptyNodes: 50 - log.length } });
    const text = serializeGameRecord(original);
    expect(gameRecordResult(parseGameRecord(text))).toEqual({ result: winner === 0 ? '1-0' : '0-1', termination: 'clinch', winner });
  });

  it('keeps a proven lead unfinished until the players accept the result', () => {
    const original = record({ log: places(48) });
    expect(gameRecordResult(original)).toEqual({ result: '*', termination: 'unfinished', winner: null });
    expect(parseGameRecord(serializeGameRecord(original)).earlyOutcome).toBeNull();
  });

  it('derives the full-board winner and rejects a forged result or early ending', () => {
    const original = record({ log: places(50) });
    const game = replay(original.config, original.log);
    const winner = validateTerminalWinner(game.board, game.stones).winner;
    const text = serializeGameRecord(original);
    expect(gameRecordResult(original)).toEqual({ result: winner === 0 ? '1-0' : '0-1', termination: 'board-full', winner });
    expect(() => parseGameRecord(replaceResult(text, winner === 0 ? '0-1' : '1-0'))).toThrow(/does not match/);
    expect(() => parseGameRecord(header(text, 'Termination', 'resignation'))).toThrow(/full board/);
    expect(() => parseGameRecord(header(text, 'Termination', 'unfinished'))).toThrow(/does not match/);
  });

  it('rejects an unproven clinch and inconsistent outcome metadata', () => {
    const text = replaceResult(header(serializeGameRecord(record()), 'Termination', 'clinch'), '1-0');
    expect(() => parseGameRecord(text)).toThrow(/not proven/);
    expect(validateGameRecord(record({ earlyOutcome: { reason: 'clinch', winner: 0, loser: 1, emptyNodes: 45 } }))).toBeNull();
    expect(validateGameRecord(record({ log: places(48), earlyOutcome: { reason: 'clinch', winner: 1, loser: 0, emptyNodes: 1 } }))).toBeNull();
    expect(validateGameRecord(record({ earlyOutcome: { reason: 'resignation', winner: 0, loser: 0 } }))).toBeNull();
  });

  it('round trips varied legal main lines, names, and optional handicaps', () => {
    fc.assert(fc.property(
      fc.uniqueArray(fc.integer({ min: 0, max: 49 }), { maxLength: 50 }),
      fc.string({ maxLength: 200 }),
      fc.boolean(),
      (nodes, name, double) => {
        const original = record({ log: nodes.map((node) => ({ type: 'place', node })) });
        original.config.mode = double ? 'double' : 'classic';
        original.config.playerNames[0] = name;
        delete original.config.handicap;
        const imported = parseGameRecord(serializeGameRecord(original));
        expect(imported.log).toEqual(original.log);
        expect(imported.config).toEqual({ ...original.config, handicap: 1 });
      },
    ), { numRuns: 80 });
  });

  it.each([
    ['DGN', '2', /Unsupported DGN/], ['Rules', 'deltrel.rules.v0', /rules version/],
    ['Rings', '5', /Rings/], ['Rings', '04', /whole number/], ['Mode', 'hex', /Mode/],
    ['PieRule', 'yes', /PieRule/], ['Handicap', '10', /Handicap/],
    ['Player1Type', 'robot', /Player types/], ['Player2Type', 'robot', /Player types/],
    ['LocalAI', 'deep', /does not match/], ['CloudAI', 'fast', /does not match/],
    ['LocalSearch', '544', /simulations/], ['LocalSearch', '0/16', /positive/],
    ['CloudSearch', '4294967296/16', /positive/], ['CloudSearch', '1/1/1', /simulations/],
    ['Date', '2026-02-30T12:00:00.000Z', /UTC date/], ['Updated', '2025-09-26T12:00:00.000Z', /earlier/],
    ['Termination', 'draw', /Unknown termination/], ['Termination', 'resignation', /winning result/],
    ['Termination', 'board-full', /does not match/], ['Result', '0-1', /disagree/],
  ] as const)('rejects invalid %s=%s headers', (name, value, error) => {
    expect(() => parseGameRecord(header(serializeGameRecord(record()), name, value))).toThrow(error);
  });

  it('rejects conflicting pie and handicap settings', () => {
    const text = header(header(serializeGameRecord(record()), 'PieRule', 'true'), 'Handicap', '2');
    expect(() => parseGameRecord(text)).toThrow(/Handicap openings/);
  });

  it('rejects malformed, duplicate, missing, and unknown headers', () => {
    const text = serializeGameRecord(record());
    expect(() => parseGameRecord(`[DGN "1"]\n${text}`)).toThrow(/Duplicate/);
    expect(() => parseGameRecord(text.replace('[DGN "1"]\n', ''))).toThrow(/Missing header: DGN/);
    expect(() => parseGameRecord(`[Unknown "x"]\n${text}`)).toThrow(/Unknown header/);
    expect(() => parseGameRecord(text.replace('[DGN "1"]', '[DGN 1]'))).toThrow(/malformed/);
    expect(() => parseGameRecord(text.replace('[DGN "1"]', '[DGN "\\q"]'))).toThrow(/Invalid quoted text/);
    expect(() => parseGameRecord('')).toThrow(/Missing/);
  });

  it('rejects missing results, non-sequential turns, occupied nodes, bad grouping, and illegal swaps', () => {
    const text = serializeGameRecord(record());
    const a = coordinateLabel(0, 4);
    const b = coordinateLabel(1, 4);
    const c = coordinateLabel(2, 4);
    const d = coordinateLabel(3, 4);
    expect(() => parseGameRecord(body(text, `1. ${a}`))).toThrow(/End the moves/);
    expect(() => parseGameRecord(body(text, '1. *'))).toThrow(/number and its moves/);
    expect(() => parseGameRecord(body(text, `2. ${a} *`))).toThrow(/Expected turn 1/);
    expect(() => parseGameRecord(body(text, `1. ${a} 2. ${a} *`))).toThrow(/already occupied/);
    expect(() => parseGameRecord(body(text, '1. Z999 *'))).toThrow(/not a coordinate/);
    expect(() => parseGameRecord(body(text, `1. ${a},${b} *`))).toThrow(/too many moves/);
    expect(() => parseGameRecord(body(text, `1. ${a} 2. ${b} 3. ${c} *`))).toThrow(/incomplete/);
    expect(() => parseGameRecord(body(text, `1. ${a} 2. ${b},${c},${d} *`))).toThrow(/too many moves/);
    expect(() => parseGameRecord(body(text, '1. swap *'))).toThrow(/swap is unavailable/);
    expect(() => parseGameRecord(body(text, `1. ${a} 2. ${b}, *`))).toThrow(/not a coordinate/);
  });

  it('bounds input work before parsing oversized game data', () => {
    expect(() => parseGameRecord('x'.repeat(MAX_GAME_RECORD_TEXT_LENGTH + 1))).toThrow(/at most/);
    expect(() => parseGameRecord(body(serializeGameRecord(record()), `${'1. A10 '.repeat(400)}*`))).toThrow(/too many/);
    expect(validateGameRecord(record({ log: places(277) }))).toBeNull();
    const oversized = record();
    oversized.config.playerNames[0] = 'x'.repeat(MAX_GAME_RECORD_TEXT_LENGTH);
    expect(validateGameRecord(oversized)?.config.playerNames).toEqual(oversized.config.playerNames);
    expect(() => serializeGameRecord(oversized)).toThrow(/sharing limit/);
  });

  it('preserves legacy long player names in archives and shared text', () => {
    const original = record();
    original.config.playerNames = ['a'.repeat(201), 'b'.repeat(3_000)];
    expect(validateGameRecord(original)?.config.playerNames).toEqual(original.config.playerNames);
    expect(parseGameRecord(serializeGameRecord(original)).config.playerNames).toEqual(original.config.playerNames);
  });

  it('validates archive data strictly and returns owned copies', () => {
    const original = record();
    const validated = validateGameRecord(original)!;
    expect(validated).toEqual(original);
    expect(validated).not.toBe(original);
    expect(validated.log).not.toBe(original.log);
    expect(validated.log[0]).not.toBe(original.log[0]);
    expect(validated.config.playerNames).not.toBe(original.config.playerNames);
    expect(validated.aiSearchSettings.local).not.toBe(original.aiSearchSettings.local);
    for (const invalid of [null, [], {}, { ...original, extra: 1 }, { ...original, id: '' },
      { ...original, createdAt: 'today' }, { ...original, config: { ...original.config, playerNames: Array(2) } },
      { ...original, controllers: Array(2) }, { ...original, log: [{ type: 'place', node: 50 }] },
      { ...original, log: [{ type: 'place', node: 0, extra: true }] }, { ...original, log: [{ type: 'unknown' }] },
      { ...original, aiSearchSettings: { local: original.aiSearchSettings.local } },
      { ...original, earlyOutcome: { reason: 'aborted', winner: 0, loser: 1 } },
    ]) expect(validateGameRecord(invalid)).toBeNull();
    expect(() => serializeGameRecord(record({ log: [{ type: 'swap' }] }))).toThrow(/illegal/);
  });
});
