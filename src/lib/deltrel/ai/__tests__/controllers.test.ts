import { describe, expect, it } from 'vitest';
import {
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
    expect(normalizeControllers(double, controllers)).toEqual(controllers);
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
