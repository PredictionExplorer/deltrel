import { MAX_RINGS } from './board';
import { configHandicap, type GameConfig } from './game';

/** New matches use pie, except handicap matches on the largest board.
 * Apply only at a new-game boundary; historical games retain their rules.
 */
export function normalizeNewGameConfig(config: GameConfig): GameConfig {
  const handicap = config.rings === MAX_RINGS && !config.pieRule
    ? configHandicap(config)
    : 1;
  return {
    ...config,
    pieRule: handicap === 1,
    handicap,
    playerNames: [...config.playerNames],
  };
}
