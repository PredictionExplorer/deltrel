import { DeltrelAiError } from './errors';
import type { DeltrelAiRequest } from './protocol';

export interface FinalPlayerPrediction {
  player: 0 | 1;
  shores: number;
  networks: number;
  corners: number;
  cornerBonusProbability: number;
}

export interface FutureMovePrediction {
  player: 0 | 1;
  kind: 'place' | 'swap';
  node: number | null;
  probability: number;
}

export interface DeltrelAiPredictions {
  perspective: 0 | 1;
  finalBasis: 'official_end';
  /** Stable board-color order: player zero, then player one. */
  finalCounts: [FinalPlayerPrediction, FinalPlayerPrediction];
  opponentReply: FutureMovePrediction | null;
  secondStone: FutureMovePrediction | null;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exact(value: Record<string, unknown>, keys: string[]): boolean {
  return Object.keys(value).length === keys.length &&
    keys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
}

function bounded(value: unknown, maximum: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= maximum;
}

function invalid(): never {
  throw new DeltrelAiError('protocol', 'AI future predictions are invalid.');
}

function parseFuture(value: unknown): FutureMovePrediction | null {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['player', 'kind', 'node', 'probability']) ||
    (value.player !== 0 && value.player !== 1) || !bounded(value.probability, 1)) return invalid();
  if (value.kind === 'swap' && value.node === null) {
    return { player: value.player, kind: 'swap', node: null, probability: value.probability };
  }
  if (value.kind !== 'place' || !bounded(value.node, 274) || !Number.isSafeInteger(value.node)) return invalid();
  return { player: value.player, kind: 'place', node: value.node, probability: value.probability };
}

export function parsePredictions(value: unknown): DeltrelAiPredictions | null {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['perspective', 'finalBasis', 'finalCounts', 'opponentReply', 'secondStone']) ||
    (value.perspective !== 0 && value.perspective !== 1) || value.finalBasis !== 'official_end' ||
    !Array.isArray(value.finalCounts) || value.finalCounts.length !== 2) return invalid();
  const finalCounts = value.finalCounts.map((item, index): FinalPlayerPrediction => {
    if (!record(item) || !exact(item, ['player', 'shores', 'networks', 'corners', 'cornerBonusProbability']) ||
      item.player !== index || (item.player !== 0 && item.player !== 1) ||
      !bounded(item.shores, 50) || !bounded(item.networks, 25) || !bounded(item.corners, 5) ||
      !bounded(item.cornerBonusProbability, 1)) return invalid();
    return {
      player: item.player, shores: item.shores, networks: item.networks,
      corners: item.corners, cornerBonusProbability: item.cornerBonusProbability
    };
  }) as [FinalPlayerPrediction, FinalPlayerPrediction];
  const opponentReply = parseFuture(value.opponentReply);
  const secondStone = parseFuture(value.secondStone);
  if ((opponentReply && opponentReply.player !== 1 - value.perspective) ||
    (secondStone && (secondStone.player !== value.perspective || secondStone.kind !== 'place'))) return invalid();
  return { perspective: value.perspective, finalBasis: 'official_end', finalCounts, opponentReply, secondStone };
}

export function parseServerPredictions(value: unknown): DeltrelAiPredictions | null {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['perspective', 'final_basis', 'final_counts', 'opponent_reply', 'second_stone']) ||
    !Array.isArray(value.final_counts)) return invalid();
  return parsePredictions({
    perspective: value.perspective, finalBasis: value.final_basis,
    finalCounts: value.final_counts.map((item) => {
      if (!record(item) || !exact(item, ['player', 'shores', 'networks', 'corners', 'corner_bonus_probability'])) return invalid();
      return {
        player: item.player, shores: item.shores, networks: item.networks,
        corners: item.corners, cornerBonusProbability: item.corner_bonus_probability
      };
    }),
    opponentReply: value.opponent_reply, secondStone: value.second_stone,
  });
}

export function validatePredictionState(predictions: DeltrelAiPredictions, request: DeltrelAiRequest): void {
  const state = request.state;
  if (predictions.perspective !== state.toMove || predictions.finalCounts.some(
    (item) => item.shores > 5 * state.rings || item.networks > Math.floor(5 * state.rings / 2),
  )) return invalid();
  for (const move of [predictions.opponentReply, predictions.secondStone]) {
    if (move?.kind === 'place' && (move.node === null || state.stones[move.node] !== -1)) return invalid();
  }
  if (predictions.opponentReply?.kind === 'swap' && (!state.pie || !state.opening)) return invalid();
  if (predictions.secondStone && (state.mode !== 'double' || state.opening || state.movesLeft !== 2)) return invalid();
}
