import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EngineEstimatePanel, type EngineEstimateContext } from '../EngineEstimatePanel';
import { NETWORK_HEADS } from '../engineEstimatePresentation';
import { analysisFixture, estimateBoard, headFixture, networkFixture } from './engineEstimateFixtures';

afterEach(cleanup);

const playerNames = ['Ada', 'Grace'] as const;
const liveContext: EngineEstimateContext = {
  analyzedPly: 4, displayedPly: 4, source: 'server', status: 'ready', isExact: true, phase: 'live',
};

describe('EngineEstimatePanel', () => {
  it('always shows both named win chances and coherent expected points without expanding anything', () => {
    render(<EngineEstimatePanel analysis={analysisFixture()} board={estimateBoard} playerNames={playerNames} showSearchDetails={false} context={liveContext} />);
    expect(screen.getByRole('region', { name: 'Engine estimate' })).toBeVisible();
    const ada = screen.getByRole('article', { name: 'Ada forecast' });
    const grace = screen.getByRole('article', { name: 'Grace forecast' });
    expect(within(ada).getByText('75.0%')).toBeVisible();
    expect(within(ada).getByText('12.0')).toBeVisible();
    expect(within(grace).getByText('25.0%')).toBeVisible();
    expect(within(grace).getByText('9.0')).toBeVisible();
    expect(screen.getByText(/21-point final total/)).toBeVisible();
    expect(screen.getByText(/After move 4/)).toBeVisible();
    expect(screen.queryByText('15.0%')).not.toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^All network outputs/ })).toBeVisible();
    for (const disclosure of screen.getAllByRole('button')) {
      const target = document.getElementById(disclosure.getAttribute('aria-controls')!);
      expect(target).toHaveAttribute('hidden');
    }
  });

  it('maps a player-two evaluation to the correct named forecasts', () => {
    render(<EngineEstimatePanel analysis={analysisFixture({ perspective: 1, networkOutput: networkFixture(1) })} board={estimateBoard} playerNames={playerNames} />);
    expect(within(screen.getByRole('article', { name: 'Ada forecast' })).getByText('25.0%')).toBeVisible();
    expect(within(screen.getByRole('article', { name: 'Ada forecast' })).getByText('9.0')).toBeVisible();
    expect(within(screen.getByRole('article', { name: 'Grace forecast' })).getByText('75.0%')).toBeVisible();
    expect(screen.getByText('Grace +3.0 expected points')).toBeVisible();
  });

  it('labels retained, hypothetical and historical positions instead of presenting them as current', () => {
    render(<EngineEstimatePanel analysis={analysisFixture()} board={estimateBoard} playerNames={playerNames} context={{
      ...liveContext, analyzedPly: 2, displayedPly: 5, isExact: false, status: 'thinking', proof: true,
      message: 'Evaluating move 5; showing the completed evaluation from move 2.',
    }} />);
    expect(screen.getByText('Earlier position · viewing move 5')).toBeVisible();
    expect(screen.getByText(/After move 2/)).toBeVisible();
    expect(screen.getByText(/proof stones are hypothetical/)).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('Evaluating move 5; showing the completed evaluation from move 2.');
    expect(screen.getByText('Updating')).toBeVisible();
  });

  it('shows pending and unavailable states without inventing win probabilities', async () => {
    const onAnalyze = vi.fn();
    const { rerender } = render(<EngineEstimatePanel analysis={null} board={estimateBoard} playerNames={playerNames} context={{ ...liveContext, analyzedPly: null, status: 'thinking' }} onAnalyze={onAnalyze} canAnalyze={false} />);
    expect(screen.getByText('Calculating a new evaluation…')).toBeVisible();
    expect(screen.queryByText('50.0%')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^All network outputs/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze position' })).toBeDisabled();
    rerender(<EngineEstimatePanel analysis={null} board={estimateBoard} playerNames={playerNames} context={{ ...liveContext, analyzedPly: null, status: 'unavailable', message: 'Connect an engine to analyze this position.' }} />);
    expect(screen.getByText('Unavailable')).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('Connect an engine');
    expect(screen.queryByRole('button', { name: 'Analyze position' })).not.toBeInTheDocument();
  });

  it('makes analyze and pause controls available without applying a suggested move', async () => {
    const user = userEvent.setup();
    const onAnalyze = vi.fn();
    const onPause = vi.fn();
    const { rerender } = render(<EngineEstimatePanel analysis={analysisFixture()} board={estimateBoard} playerNames={playerNames} context={{ ...liveContext, action: { type: 'place', node: 0 }, applied: false }} onAnalyze={onAnalyze} canAnalyze onPause={onPause} canPause />);
    expect(screen.getByText('Suggested move')).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Analyze position' }));
    await user.click(screen.getByRole('button', { name: 'Pause AI' }));
    expect(onAnalyze).toHaveBeenCalledOnce();
    expect(onPause).toHaveBeenCalledOnce();
    rerender(<EngineEstimatePanel analysis={analysisFixture({ expectedMargin: 0 })} board={estimateBoard} playerNames={playerNames} context={{ ...liveContext, analyzedPly: 0, source: 'local', action: { type: 'swap' }, applied: true }} onPause={onPause} canPause={false} />);
    expect(screen.getByText(/AI · Opening position/)).toBeVisible();
    expect(screen.getByText('Selected move')).toBeVisible();
    expect(screen.getByText('Pie swap')).toBeVisible();
    expect(screen.getByText(/Even · 0.0/)).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Pause AI' })).not.toBeInTheDocument();
  });

  it('preserves and identifies an out-of-board score forecast instead of clamping it', () => {
    render(<EngineEstimatePanel analysis={analysisFixture({ expectedMargin: 41 })} board={estimateBoard} playerNames={playerNames} />);
    expect(screen.getByText('31.0')).toBeVisible();
    expect(screen.getByText('-10.0')).toBeVisible();
    expect(screen.getByText(/original forecast is shown without clipping/)).toBeVisible();
  });

  it('allows every search candidate to be inspected, paged and located by coordinate', async () => {
    const user = userEvent.setup();
    const count = 20;
    const analysis = analysisFixture({
      rootActions: Array.from({ length: count }, (_, node) => ({ type: 'place', node })),
      rootPolicy: Array(count).fill(1 / count), rootQ: Array(count).fill(0.125), rootVisits: Array.from({ length: count }, (_, node) => node), swapRecommended: true,
    });
    render(<EngineEstimatePanel analysis={analysis} board={estimateBoard} playerNames={playerNames} />);
    const disclosure = screen.getByRole('button', { name: /^Search candidates/ });
    await user.click(disclosure);
    expect(disclosure).toHaveAttribute('aria-expanded', 'true');
    const table = screen.getByRole('table', { name: 'Search candidates' });
    expect(within(table).getAllByRole('row')).toHaveLength(13);
    expect(within(table).getAllByRole('rowheader')[0]).toHaveTextContent(estimateBoard.labels[19]);
    expect(screen.getByText(/not the raw network move policy/)).toBeVisible();
    expect(screen.getByText(/selected the pie swap/)).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Next page of Search candidates' }));
    expect(within(table).getAllByRole('row')).toHaveLength(9);
    expect(screen.getByText('13–20 of 20')).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Previous page of Search candidates' }));
    expect(screen.getByText('1–12 of 20')).toBeVisible();
    const search = screen.getByRole('searchbox', { name: 'Find a point or value' });
    await user.type(search, estimateBoard.labels[0]);
    expect(within(table).getAllByRole('row')).toHaveLength(2);
    expect(within(table).getByRole('rowheader')).toHaveTextContent(estimateBoard.labels[0]);
    await user.clear(search);
    await user.type(search, 'no such point');
    expect(screen.getByText('No matching values.')).toBeVisible();
    await user.click(disclosure);
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('exposes every network head, point ownership, masks, logits and all score bins', async () => {
    const user = userEvent.setup();
    render(<EngineEstimatePanel analysis={analysisFixture({ perspective: 1, networkOutput: networkFixture(1) })} board={estimateBoard} playerNames={playerNames} />);
    await user.click(screen.getByRole('button', { name: /^All network outputs/ }));
    const selector = screen.getByRole('combobox', { name: 'Network output' });
    expect(within(selector).getAllByRole('option')).toHaveLength(NETWORK_HEADS.length);
    for (const { key, label } of NETWORK_HEADS) {
      await user.selectOptions(selector, key);
      expect(screen.getByRole('table', { name: label })).toBeVisible();
    }
    await user.selectOptions(selector, 'ownership');
    const ownership = screen.getByRole('table', { name: 'Ownership at every point' });
    expect(within(ownership).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual(['Point', 'Ada', 'Grace', 'Unclaimed']);
    expect(within(within(ownership).getAllByRole('row')[1]).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['30.0%', '60.0%', '10.0%']);
    await user.click(screen.getByRole('checkbox', { name: 'Show raw logits' }));
    expect(within(ownership).getAllByText('logit -1.20397').length).toBeGreaterThan(0);
    await user.selectOptions(selector, 'scoreMargin');
    expect(screen.getByRole('img', { name: 'Final score margin probability distribution' })).toBeVisible();
    expect(screen.getByText('1–12 of 303')).toBeVisible();
    expect(screen.getAllByText('Masked').length).toBeGreaterThan(0);
    expect(screen.getAllByText('logit —').length).toBeGreaterThan(0);
    expect(screen.getByText(/Positive differences favor Grace/)).toBeVisible();
    await user.selectOptions(selector, 'opponentReply');
    await user.type(screen.getByRole('searchbox', { name: 'Find a point or value' }), 'swap');
    expect(screen.getByRole('rowheader', { name: 'Pie swap' })).toBeVisible();
    expect(screen.getByText(/swap is only legal after the opening/)).toBeVisible();
    await user.selectOptions(selector, 'secondStone');
    expect(screen.getByText(/does not apply to the analyzed position/)).toBeVisible();
  });

  it('keeps auxiliary count-derived points distinct from the headline score forecast', async () => {
    const user = userEvent.setup();
    const analysis = analysisFixture({ predictions: {
      perspective: 0, finalBasis: 'official_end',
      finalCounts: [
        { player: 0, shores: 12.5, networks: 2.5, corners: 3.2, cornerBonusProbability: 0.8 },
        { player: 1, shores: 7.5, networks: 1.5, corners: 1.8, cornerBonusProbability: 0.2 },
      ],
      secondStone: { player: 0, kind: 'place', node: 3, probability: 0.3 },
      opponentReply: { player: 1, kind: 'swap', node: null, probability: 0.2 },
    } });
    render(<EngineEstimatePanel analysis={analysis} board={estimateBoard} playerNames={playerNames} />);
    await user.click(screen.getByRole('button', { name: /^Final-count forecasts/ }));
    expect(screen.getByText('Ada 11.3 · Grace 9.7')).toBeVisible();
    expect(within(screen.getByRole('article', { name: 'Ada forecast' })).getByText('12.0')).toBeVisible();
    expect(screen.getByText(/independent heads can disagree/)).toBeVisible();
    expect(screen.getByText('Grace · next reply')).toBeVisible();
    expect(screen.getByText('Pie swap · 20.0%')).toBeVisible();
    expect(screen.getByText(`${estimateBoard.labels[3]} · 30.0%`)).toBeVisible();
  });

  it('keeps very small nonzero output probabilities distinguishable from zero', async () => {
    const user = userEvent.setup();
    const output = networkFixture();
    output.heads.policy = headFixture([estimateBoard.n], [0.0001, ...Array(estimateBoard.n - 1).fill(0.9999 / (estimateBoard.n - 1))]);
    render(<EngineEstimatePanel analysis={analysisFixture({ networkOutput: output })} board={estimateBoard} playerNames={playerNames} />);
    await user.click(screen.getByRole('button', { name: /^All network outputs/ }));
    expect(screen.getByText(/Ordered from highest to lowest probability/)).toBeVisible();
    await user.type(screen.getByRole('searchbox', { name: 'Find a point or value' }), estimateBoard.labels[0]);
    expect(screen.getByRole('table', { name: 'Move policy' })).toHaveTextContent('0.0100%');
  });

  it('labels missing and untrained outputs without turning them into forecasts', async () => {
    const user = userEvent.setup();
    const output = networkFixture();
    output.auxiliaryStatus = 'untrained';
    output.heads.alive = null;
    const { rerender } = render(<EngineEstimatePanel analysis={analysisFixture({ networkOutput: output })} board={estimateBoard} playerNames={playerNames} />);
    await user.click(screen.getByRole('button', { name: /^Final-count forecasts/ }));
    expect(screen.getByText(/auxiliary outputs are untrained/)).toBeVisible();
    expect(screen.queryByText('Points derived from these count forecasts')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /^All network outputs/ }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Network output' }), 'finalShores');
    expect(screen.getByText(/raw values are shown for inspection/)).toBeVisible();
    await user.selectOptions(screen.getByRole('combobox', { name: 'Network output' }), 'alive');
    expect(screen.getByText('This output was not returned by this model.')).toBeVisible();
    rerender(<EngineEstimatePanel analysis={analysisFixture({ networkOutput: null, predictions: null })} board={estimateBoard} playerNames={playerNames} />);
    expect(screen.getByText(/did not return its full network tensors/)).toBeVisible();
    expect(screen.getByText(/Final-count and future-move forecasts were not returned/)).toBeVisible();
  });

  it('includes all search diagnostics and exact full-precision outputs in the downloadable JSON', async () => {
    const user = userEvent.setup();
    const analysis = analysisFixture({ modelIdentity: null, modelStep: null, timingMs: { queue: 0, modelLoad: 0, inferenceSearch: 1_234, total: 1_234 } });
    const context: EngineEstimateContext = { ...liveContext, action: { type: 'place', node: 2 }, applied: true };
    const { container } = render(<EngineEstimatePanel analysis={analysis} board={estimateBoard} playerNames={playerNames} context={context} />);
    await user.click(screen.getByRole('button', { name: /^Search and model details/ }));
    expect(screen.getByText('Search input value')).toBeVisible();
    expect(screen.getByText('Searched root value')).toBeVisible();
    expect(screen.getByText('-0.700')).toBeVisible();
    expect(screen.getByText('+0.320')).toBeVisible();
    expect(screen.getAllByText('1.23 s').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Not reported')).toHaveLength(2);
    await user.click(screen.getByRole('button', { name: /^Raw engine output/ }));
    const raw = screen.getByRole('textbox', { name: 'Raw engine output' });
    expect(raw).toHaveAttribute('readonly');
    expect(JSON.parse((raw as HTMLTextAreaElement).value)).toEqual({ context, board: { rings: 4, labels: estimateBoard.labels }, analysis });
    const download = screen.getByRole('link', { name: 'Download full JSON' });
    expect(download).toHaveAttribute('download', 'deltrel-analysis-zobrist64-0123456789abcdef.json');
    const exported = JSON.parse(decodeURIComponent(download.getAttribute('href')!.split(',').slice(1).join(',')));
    expect(exported.analysis.networkOutput.heads.scoreMargin.probabilities).toHaveLength(303);
    expect(exported.context.action).toEqual({ type: 'place', node: 2 });
    expect((await axe(container)).violations).toEqual([]);
  });
});
