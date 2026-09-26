import { describe, expect, it } from 'vitest';
import {
  CONTROLLER_TYPES,
  canShowAiInsights,
  controllerLabel,
  isHumanVsAi,
  normalizeControllers,
  aiMatchLabel,
  playerNamesForControllers,
  supportsAiControllers,
  type PlayerControllers,
} from '../controllers';
import type { GameConfig } from '../../game';

const double: GameConfig = {
  rings: 6,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

describe('AI controller validation', () => {
  it.each([
    ['human', 'human', false],
    ['human', 'local', true],
    ['local', 'human', true],
    ['human', 'server', true],
    ['server', 'human', true],
    ['local', 'local', false],
    ['server', 'local', false],
    ['local', 'server', false],
    ['server', 'server', false],
  ] as const)('protects AI insights for %s versus %s', (first, second, hidden) => {
    const controllers: PlayerControllers = [first, second];
    expect(isHumanVsAi(controllers)).toBe(hidden);
    expect(canShowAiInsights(controllers, false)).toBe(!hidden);
    expect(canShowAiInsights(controllers, true)).toBe(false);
  });

  it('offers cloud and device AI while preserving saved controller choices', () => {
    expect(CONTROLLER_TYPES).toEqual(['human', 'local', 'server']);
    expect(controllerLabel('local')).toBe('AI');
    expect(controllerLabel('server')).toBe('Cloud AI');
    expect(normalizeControllers(double, ['server', 'human'])).toEqual(['server', 'human']);
  });

  it.each([
    ['human', 'human'], ['human', 'local'], ['local', 'local'], ['server', 'local'],
  ] as const)('migrates the old quick-start pair to neutral names for %s versus %s', (first, second) => {
    const names = ['You', 'Champion'] as const;
    expect(playerNamesForControllers(names, [first, second])).toEqual(['Player 1', 'Player 2']);
    expect(names).toEqual(['You', 'Champion']);
  });

  it('labels AI matches explicitly and reserves the quick-start You label for human players', () => {
    const names = ['You', 'Marina'] as const;
    expect(playerNamesForControllers(names, ['local', 'server'])).toEqual(['AI 1', 'Marina']);
    expect(playerNamesForControllers(names, ['human', 'local'])).toEqual(['You', 'Marina']);
    expect(playerNamesForControllers(['River', 'Shore'], ['local', 'local'])).toEqual(['River', 'Shore']);
    expect(names).toEqual(['You', 'Marina']);
    expect(aiMatchLabel(['local', 'local'])).toBe('AI versus AI');
    expect(aiMatchLabel(['server', 'local'])).toBe('AI versus AI');
    expect(aiMatchLabel(['human', 'local'])).toBe('Human versus AI');
    expect(aiMatchLabel(['human', 'human'])).toBeUndefined();
  });
  it('keeps independent valid controllers only for no-pie Double Deltrel', () => {
    const controllers: PlayerControllers = ['server', 'local'];
    expect(supportsAiControllers(double)).toBe(true);
    expect(normalizeControllers(double, controllers)).toEqual(['server', 'local']);
  });

  it('keeps AI controllers for every rules-v3 variant', () => {
    expect(normalizeControllers({ ...double, mode: 'classic' }, ['server', 'local'])).toEqual([
      'server',
      'local',
    ]);
    expect(normalizeControllers({ ...double, pieRule: true }, ['server', 'local'])).toEqual([
      'server',
      'local',
    ]);
    expect(
      normalizeControllers({ ...double, handicap: 9 }, ['server', 'local']),
    ).toEqual(['server', 'local']);
  });

  it('forces persisted AI controllers back to human outside the rules family', () => {
    expect(
      normalizeControllers({ ...double, handicap: 10 }, ['server', 'local']),
    ).toEqual(['human', 'human']);
    expect(
      normalizeControllers({ ...double, pieRule: true, handicap: 2 }, ['server', 'local']),
    ).toEqual(['human', 'human']);
  });

  it('sanitizes malformed persisted controller tuples per player', () => {
    expect(normalizeControllers(double, ['server', 'remote'])).toEqual(['server', 'human']);
    expect(normalizeControllers(double, ['local'])).toEqual(['human', 'human']);
    expect(normalizeControllers(double, null)).toEqual(['human', 'human']);
  });
});
