import { describe, expect, it } from 'vitest';
import {
  NETWORK_HEADS,
  countDerivedPoints,
  expectedFinalPoints,
  networkHeadTable,
  winProbabilities,
} from '../engineEstimatePresentation';
import { analysisFixture, estimateBoard, networkFixture } from './engineEstimateFixtures';

describe('engine forecast presentation', () => {
  it('maps the evaluated perspective to stable players and preserves the fixed final total', () => {
    expect(expectedFinalPoints(analysisFixture(), estimateBoard)).toEqual([12, 9]);
    expect(expectedFinalPoints(analysisFixture({ perspective: 1 }), estimateBoard)).toEqual([9, 12]);
    expect(winProbabilities(analysisFixture())).toEqual([0.75, 0.25]);
    expect(winProbabilities(analysisFixture({ perspective: 1 }))).toEqual([0.25, 0.75]);
    const points = expectedFinalPoints(analysisFixture({ expectedMargin: -6.5 }), estimateBoard);
    expect(points[0] + points[1]).toBe(21);
    expect(points[0] - points[1]).toBe(-6.5);
  });

  it('retains out-of-board score predictions instead of silently clipping them', () => {
    expect(expectedFinalPoints(analysisFixture({ expectedMargin: 41 }), estimateBoard)).toEqual([31, -10]);
  });

  it('keeps count-derived forecasts separate from the score-margin estimate', () => {
    const analysis = analysisFixture({ predictions: {
      perspective: 0, finalBasis: 'official_end',
      finalCounts: [
        { player: 0, shores: 12.5, networks: 2.5, corners: 3.2, cornerBonusProbability: 0.8 },
        { player: 1, shores: 7.5, networks: 1.5, corners: 1.8, cornerBonusProbability: 0.2 },
      ], opponentReply: null, secondStone: null,
    } });
    expect(countDerivedPoints(analysis)).toEqual([11.3, 9.7]);
    expect(expectedFinalPoints(analysis, estimateBoard)).toEqual([12, 9]);
    expect(countDerivedPoints(analysisFixture())).toBeNull();
  });

  it('maps ownership classes and final-count rows from player-two perspective', () => {
    const output = networkFixture(1);
    const ownership = networkHeadTable('ownership', output.heads.ownership!, output, estimateBoard, ['Ada', 'Grace']);
    expect(ownership.valueLabels).toEqual(['Ada', 'Grace', 'Unclaimed']);
    expect(ownership.rows[0].values.map((value) => value.probability)).toEqual([0.3, 0.6, 0.1]);
    const counts = networkHeadTable('finalShores', output.heads.finalShores!, output, estimateBoard, ['Ada', 'Grace']);
    expect(counts.rows[7].values.map((value) => value.probability)).toEqual([0, 1]);
    expect(counts.rows[13].values.map((value) => value.probability)).toEqual([1, 0]);
  });

  it('includes every raw output exactly once, including masked bins and the swap slot', () => {
    const output = networkFixture();
    for (const { key } of NETWORK_HEADS) {
      const head = output.heads[key]!;
      const table = networkHeadTable(key, head, output, estimateBoard, ['Ada', 'Grace']);
      expect(table.rows.flatMap((row) => row.values)).toHaveLength(head.probabilities.length);
      expect(table.rows.flatMap((row) => row.values).map((value) => value.probability).sort()).toEqual([...head.probabilities].sort());
    }
    const margin = networkHeadTable('scoreMargin', output.heads.scoreMargin!, output, estimateBoard, ['Ada', 'Grace']);
    expect(margin.rows[0]).toMatchObject({ label: '-151', values: [{ probability: 0, logit: null, masked: true }] });
    expect(margin.rows.at(-1)?.label).toBe('151');
    const reply = networkHeadTable('opponentReply', output.heads.opponentReply!, output, estimateBoard, ['Ada', 'Grace']);
    expect(reply.rows.at(-1)?.label).toBe('Pie swap');
  });
});
