import type { Board } from '@/lib/deltrel/board';
import type { DeltrelAiAnalysis } from '@/lib/deltrel/ai/decision';
import type { DeltrelNetworkHead, DeltrelNetworkOutput } from '@/lib/deltrel/ai/network-output';

export type NetworkHeadName = keyof DeltrelNetworkOutput['heads'];

export const NETWORK_HEADS: readonly { key: NetworkHeadName; label: string; description: string }[] = [
  { key: 'policy', label: 'Move policy', description: 'The network’s placement probabilities before search.' },
  { key: 'softPolicy', label: 'Soft move policy', description: 'The network’s secondary placement distribution.' },
  { key: 'outcome', label: 'Win and loss', description: 'The network’s probabilities for the game’s final outcome.' },
  { key: 'scoreMargin', label: 'Final score margin', description: 'The full distribution of final point differences, from the analyzed player’s perspective.' },
  { key: 'ownership', label: 'Ownership at every point', description: 'Predicted final ownership for every point on the board.' },
  { key: 'alive', label: 'Living networks at every point', description: 'The probability that each point’s final stone belongs to a living network; this is not the survival chance of its current stone.' },
  { key: 'opponentReply', label: 'Opponent’s next move', description: 'A forecast of the opponent’s next reply, including a pie swap when applicable.' },
  { key: 'secondStone', label: 'Second stone', description: 'A forecast of the analyzed player’s second placement in a double turn.' },
  { key: 'finalShores', label: 'Final shore counts', description: 'The full final shore-count distribution for both players.' },
  { key: 'finalNetworks', label: 'Final network counts', description: 'The full final living-network count distribution for both players.' },
  { key: 'finalCapes', label: 'Final cape counts', description: 'The full final cape-count distribution for both players.' },
];

export const AUXILIARY_HEADS = new Set<NetworkHeadName>([
  'opponentReply', 'secondStone', 'finalShores', 'finalNetworks', 'finalCapes',
]);

/** Keep the game’s fixed final total and map the evaluated perspective to colors. */
export function expectedFinalPoints(analysis: DeltrelAiAnalysis, board: Board): [number, number] {
  const total = board.shoreCount + 1;
  const firstPlayerMargin = analysis.perspective === 0 ? analysis.expectedMargin : -analysis.expectedMargin;
  return [(total + firstPlayerMargin) / 2, (total - firstPlayerMargin) / 2];
}

/** Independent auxiliary heads are shown separately from the score-margin head. */
export function countDerivedPoints(analysis: DeltrelAiAnalysis): [number, number] | null {
  if (!analysis.predictions) return null;
  const [first, second] = analysis.predictions.finalCounts;
  return [
    first.shores + first.cornerBonusProbability + 2 * (second.networks - first.networks),
    second.shores + second.cornerBonusProbability + 2 * (first.networks - second.networks),
  ];
}

export function winProbabilities(analysis: DeltrelAiAnalysis): [number, number] {
  return analysis.perspective === 0
    ? [analysis.outcome.win, analysis.outcome.loss]
    : [analysis.outcome.loss, analysis.outcome.win];
}

export interface HeadValue {
  probability: number;
  logit: number | null;
  masked: boolean;
}

export interface HeadRow {
  label: string;
  values: HeadValue[];
}

export interface HeadTable {
  rowLabel: string;
  valueLabels: string[];
  rows: HeadRow[];
  supportMinimum: number | null;
}

/** Convert flat model tensors into labeled rows without discarding masked bins. */
export function networkHeadTable(
  key: NetworkHeadName,
  head: DeltrelNetworkHead,
  output: DeltrelNetworkOutput,
  board: Board,
  playerNames: readonly [string, string],
): HeadTable {
  const value = (index: number): HeadValue => ({
    probability: head.probabilities[index],
    logit: head.logits[index],
    masked: !head.mask[index],
  });
  const playerChannels = output.perspective === 0 ? [0, 1] : [1, 0];

  if (key === 'ownership') {
    return {
      rowLabel: 'Point',
      valueLabels: [playerNames[0], playerNames[1], 'Unclaimed'],
      rows: board.labels.map((label, node) => ({
        label,
        values: [value(node * 3 + playerChannels[0]), value(node * 3 + playerChannels[1]), value(node * 3 + 2)],
      })),
      supportMinimum: null,
    };
  }

  if (key === 'finalShores' || key === 'finalNetworks' || key === 'finalCapes') {
    const bins = head.shape[1];
    return {
      rowLabel: 'Count',
      valueLabels: [...playerNames],
      rows: Array.from({ length: bins }, (_, count) => ({
        label: String(count),
        values: playerChannels.map((channel) => value(channel * bins + count)),
      })),
      supportMinimum: 0,
    };
  }

  if (key === 'outcome') {
    return {
      rowLabel: 'Result',
      valueLabels: ['Probability'],
      rows: [
        { label: `${playerNames[1 - output.perspective]} wins`, values: [value(0)] },
        { label: `${playerNames[output.perspective]} wins`, values: [value(1)] },
      ],
      supportMinimum: null,
    };
  }

  if (key === 'scoreMargin') {
    return {
      rowLabel: 'Point difference',
      valueLabels: ['Probability'],
      rows: head.probabilities.map((_, index) => ({ label: String(index - 151), values: [value(index)] })),
      supportMinimum: -151,
    };
  }

  return {
    rowLabel: 'Point',
    valueLabels: ['Probability'],
    rows: head.probabilities.map((_, node) => ({
      label: node === board.n ? 'Pie swap' : board.labels[node],
      values: [value(node)],
    })),
    supportMinimum: null,
  };
}
