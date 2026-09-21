import type { DeltrelAiAnalysis } from '@/lib/deltrel/ai/decision';
import type { DeltrelNetworkHead, DeltrelNetworkOutput } from '@/lib/deltrel/ai/network-output';
import { getBoard } from '@/lib/deltrel/board';

export const estimateBoard = getBoard(4);

export function headFixture(shape: number[], probabilities: number[], activation: 'softmax' | 'sigmoid' = 'softmax', applicable = true): DeltrelNetworkHead {
  return {
    shape,
    activation,
    probabilities,
    logits: probabilities.map((probability) => probability === 0 ? null : activation === 'sigmoid' ? Math.log(probability / (1 - probability)) : Math.log(probability)),
    mask: probabilities.map((probability) => probability !== 0),
    applicable,
  };
}

const oneHot = (size: number, index: number) => Array.from({ length: size }, (_, bin) => bin === index ? 1 : 0);

export function networkFixture(perspective: 0 | 1 = 0): DeltrelNetworkOutput {
  const nodes = estimateBoard.n;
  return {
    schemaVersion: 1,
    perspective,
    nodeCount: nodes,
    auxiliaryStatus: 'ready',
    heads: {
      policy: headFixture([nodes], Array(nodes).fill(1 / nodes)),
      outcome: headFixture([2], [0.25, 0.75]),
      scoreMargin: headFixture([303], oneHot(303, 154)),
      ownership: headFixture([nodes, 3], Array.from({ length: nodes }, () => [0.6, 0.3, 0.1]).flat()),
      alive: headFixture([nodes], Array(nodes).fill(0.75), 'sigmoid'),
      softPolicy: headFixture([nodes], Array(nodes).fill(1 / nodes)),
      opponentReply: headFixture([nodes + 1], Array(nodes + 1).fill(1 / (nodes + 1))),
      secondStone: headFixture([nodes], Array(nodes).fill(1 / nodes), 'softmax', false),
      finalShores: headFixture([2, 51], [...oneHot(51, 7), ...oneHot(51, 13)]),
      finalNetworks: headFixture([2, 26], [...oneHot(26, 3), ...oneHot(26, 1)]),
      finalCapes: headFixture([2, 6], [...oneHot(6, 2), ...oneHot(6, 3)]),
    },
  };
}

export function analysisFixture(overrides: Partial<DeltrelAiAnalysis> = {}): DeltrelAiAnalysis {
  return {
    perspective: 0,
    stateHash: 'zobrist64:0123456789abcdef',
    outcome: { loss: 0.25, win: 0.75 },
    modelValue: 0.5,
    searchValue: -0.7,
    rootValue: 0.32,
    swapRecommended: false,
    expectedMargin: 3,
    rootActions: [0, 1, 2].map((node) => ({ type: 'place', node })),
    rootPolicy: [0.1, 0.6, 0.3],
    rootQ: [-0.8, 0.4, 0.1],
    rootVisits: [1, 9, 5],
    modelVersion: 'test-champion',
    modelStep: 478_534,
    modelIdentity: 'champion-test',
    simulations: 15,
    maxConsidered: 3,
    timingMs: { queue: 1, modelLoad: 2, inferenceSearch: 15, total: 18 },
    networkOutput: networkFixture(),
    ...overrides,
  };
}
