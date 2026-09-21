import { configHandicap, type GameConfig } from '../game';
import { DELTREL_MAX_HANDICAP } from '../rules';

export const CONTROLLER_TYPES = ['human', 'server', 'local'] as const;

export type ControllerType = (typeof CONTROLLER_TYPES)[number];
export type PlayerControllers = [ControllerType, ControllerType];

export const HUMAN_CONTROLLERS: PlayerControllers = ['human', 'human'];

export function isControllerType(value: unknown): value is ControllerType {
  return typeof value === 'string' && CONTROLLER_TYPES.includes(value as ControllerType);
}

/**
 * The variant-capable network plays every rule variant: classic and double
 * turns, handicap openings, and pie games (including the swap decision). Only
 * configurations outside the rules-v3 family are refused.
 */
export function supportsAiControllers(
  config: Pick<GameConfig, 'mode' | 'pieRule' | 'handicap'>,
): boolean {
  const handicap = configHandicap(config);
  return (
    (config.mode === 'classic' || config.mode === 'double') &&
    Number.isInteger(handicap) &&
    handicap >= 1 &&
    handicap <= DELTREL_MAX_HANDICAP &&
    !(config.pieRule && handicap !== 1)
  );
}

/**
 * Persistence and setup both flow through this boundary. Invalid values, and
 * AI values attached to unsupported variants, become human controllers.
 */
export function normalizeControllers(
  config: Pick<GameConfig, 'mode' | 'pieRule' | 'handicap'>,
  value: unknown,
): PlayerControllers {
  if (!supportsAiControllers(config) || !Array.isArray(value) || value.length !== 2) {
    return [...HUMAN_CONTROLLERS];
  }

  return [
    isControllerType(value[0]) ? value[0] : 'human',
    isControllerType(value[1]) ? value[1] : 'human',
  ];
}

export function controllerLabel(controller: ControllerType): string {
  switch (controller) {
    case 'human':
      return 'Human';
    case 'server':
      return 'Server AI';
    case 'local':
      return 'Browser AI';
  }
}

/** Migrate the old quick-start names without replacing players' custom names. */
export function playerNamesForControllers(
  names: readonly [string, string],
  controllers: PlayerControllers,
): [string, string] {
  // Earlier quick starts wrote this pair to preferences. Neutral names remain
  // accurate when either seat changes between a human and a computer.
  if (names[0].trim() === 'You' && names[1].trim() === 'Champion') {
    return ['Player 1', 'Player 2'];
  }
  return names.map((name, player) =>
    controllers[player] !== 'human' && name.trim() === 'You' ? `AI ${player + 1}` : name,
  ) as [string, string];
}

export function aiMatchLabel(controllers: PlayerControllers): string | undefined {
  const computers = controllers.filter(controller => controller !== 'human').length;
  return computers === 2 ? 'AI versus AI' : computers === 1 ? 'Human versus AI' : undefined;
}
