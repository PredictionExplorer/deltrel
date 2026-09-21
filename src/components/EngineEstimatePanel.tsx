import { useId, useMemo } from 'react';
import type { Board } from '@/lib/deltrel/board';
import type { DeltrelAiAnalysis } from '@/lib/deltrel/ai/decision';
import type { FutureMovePrediction } from '@/lib/deltrel/ai/predictions';

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatValue(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

function formatLatency(milliseconds: number): string {
  return milliseconds < 1_000
    ? `${milliseconds.toFixed(milliseconds < 10 ? 1 : 0)} ms`
    : `${(milliseconds / 1_000).toFixed(2)} s`;
}

function expectedMarginText(
  analysis: DeltrelAiAnalysis,
  playerNames: readonly [string, string],
): string {
  if (Math.abs(analysis.expectedMargin) < 0.05) return 'Even (0.0 points)';
  const leader =
    analysis.expectedMargin > 0
      ? analysis.perspective
      : ((1 - analysis.perspective) as 0 | 1);
  return `${playerNames[leader]} +${Math.abs(analysis.expectedMargin).toFixed(1)} points`;
}

export interface EngineEstimatePanelProps {
  analysis: DeltrelAiAnalysis;
  board: Board;
  playerNames: readonly [string, string];
  showSearchDetails?: boolean;
}

export function EngineEstimatePanel({
  analysis,
  board,
  playerNames,
  showSearchDetails = true,
}: EngineEstimatePanelProps) {
  const titleId = useId();
  const predictions = analysis.predictions;
  const moveText = (move: FutureMovePrediction) =>
    `${move.kind === 'swap' ? 'Pie swap' : board.labels[move.node!]} · ${formatPercent(move.probability)}`;
  const winEstimates: [number, number] =
    analysis.perspective === 0
      ? [analysis.outcome.win, analysis.outcome.loss]
      : [analysis.outcome.loss, analysis.outcome.win];
  const candidates = useMemo(
    () =>
      analysis.rootActions
        .map((action, index) => ({
          action,
          visits: analysis.rootVisits[index],
          value: analysis.rootQ[index],
          index,
        }))
        .sort((left, right) => right.visits - left.visits || left.index - right.index)
        .slice(0, 3),
    [analysis],
  );

  return (
    <section
      role="status"
      aria-live="polite"
      aria-labelledby={titleId}
      className="rounded-2xl border border-sand/30 bg-sand-faint"
    >
      <details className="group px-4 py-2">
        <summary className="flex min-h-11 cursor-pointer list-none items-center text-sm text-sand-strong marker:text-sand">
          <span id={titleId} className="font-medium">
            Engine estimate
          </span>
          <span className="ml-2 truncate text-xs text-muted">
            from {playerNames[analysis.perspective]}&apos;s turn
          </span>
        </summary>

        <div className="mb-2 mt-1 grid gap-2 text-xs sm:grid-cols-2">
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Win estimates
            </p>
            <p className="mt-1 text-ink">
              {playerNames[0]} {formatPercent(winEstimates[0])}
            </p>
            <p className="text-ink">
              {playerNames[1]} {formatPercent(winEstimates[1])}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Expected margin
            </p>
            <p className="mt-1 text-ink">
              {expectedMarginText(analysis, playerNames)}
            </p>
          </div>
          {showSearchDetails && (
            <>
              <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.12em] text-muted">
                  Search value
                </p>
                <p className="mt-1 font-mono text-ink">
                  {formatValue(analysis.searchValue)}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.12em] text-muted">
                  Root value
                </p>
                <p className="mt-1 font-mono text-ink">
                  {formatValue(analysis.rootValue)}
                  {analysis.swapRecommended ? ' · swapped sides' : ''}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.12em] text-muted">
                  Simulations
                </p>
                <p className="mt-1 text-ink">
                  {analysis.simulations.toLocaleString('en-US')}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.12em] text-muted">
                  Latency
                </p>
                <p className="mt-1 text-ink">
                  {formatLatency(analysis.timingMs.total)}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.12em] text-muted">
                  Model step
                </p>
                <p className="mt-1 text-ink">
                  {analysis.modelStep === null
                    ? 'Not reported'
                    : analysis.modelStep.toLocaleString('en-US')}
                </p>
              </div>
            </>
          )}
        </div>

        <div className="mt-3 rounded-lg border border-white/10 bg-black/10 px-3 py-3">
          <h3 className="text-xs uppercase tracking-[0.12em] text-muted">
            Predicted final result
          </h3>
          {predictions ? (
            <>
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-xs tabular-nums">
                  <caption className="sr-only">Expected final counts and corner bonus chance for each player</caption>
                  <thead className="text-muted">
                    <tr>
                      <th scope="col" className="py-1 pr-2 font-normal">Player</th>
                      <th scope="col" className="p-1 text-right font-normal">Shores</th>
                      <th scope="col" className="p-1 text-right font-normal">Networks</th>
                      <th scope="col" className="p-1 text-right font-normal">Corners</th>
                      <th scope="col" className="py-1 pl-2 text-right font-normal">Bonus chance</th>
                    </tr>
                  </thead>
                  <tbody className="text-ink">
                    {predictions.finalCounts.map((counts) => (
                      <tr key={counts.player}>
                        <th scope="row" className="py-1.5 pr-2 font-medium">{playerNames[counts.player]}</th>
                        <td className="p-1 text-right">{counts.shores.toFixed(1)}</td>
                        <td className="p-1 text-right">{counts.networks.toFixed(1)}</td>
                        <td className="p-1 text-right">{counts.corners.toFixed(1)}</td>
                        <td className="py-1.5 pl-2 text-right">{formatPercent(counts.cornerBonusProbability)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">
                Forecasts of the official final counts, including filling remaining cells with the losing
                side&apos;s stones when a win is clinched. Three controlled corners earn the one-point bonus.
              </p>
              <dl className="mt-3 space-y-2 text-xs">
                {predictions.secondStone && (
                  <div>
                    <dt className="text-muted">{playerNames[predictions.secondStone.player]}&apos;s second stone</dt>
                    <dd className="mt-0.5 font-mono text-ink">{moveText(predictions.secondStone)}</dd>
                  </div>
                )}
                {predictions.opponentReply && (
                  <div>
                    <dt className="text-muted">{playerNames[predictions.opponentReply.player]}&apos;s next reply</dt>
                    <dd className="mt-0.5 font-mono text-ink">{moveText(predictions.opponentReply)}</dd>
                  </div>
                )}
              </dl>
              <p className="mt-2 text-xs text-muted">Move forecasts come from this position; they are not a forced continuation.</p>
            </>
          ) : (
            <p className="mt-2 text-xs text-muted">Final-count and future-move predictions are unavailable for this model.</p>
          )}
        </div>

        {showSearchDetails && (
          <div className="mt-3">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Top candidate moves by visits
            </p>
            <ol className="mt-1.5 space-y-1">
              {candidates.map((candidate) => (
                <li
                  key={candidate.action.node}
                  className="grid grid-cols-[auto_1fr] items-baseline gap-2 rounded-lg bg-black/10 px-2.5 py-1.5 text-xs sm:grid-cols-[auto_1fr_auto]"
                >
                  <span className="font-mono text-sand-strong">
                    {board.labels[candidate.action.node]}
                  </span>
                  <span className="text-muted">
                    {candidate.visits.toLocaleString('en-US')} visits
                  </span>
                  <span className="col-span-2 font-mono text-ink sm:col-span-1">
                    value {formatValue(candidate.value)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}

        <p className="sr-only">
          Analysis for {analysis.stateHash}, player{' '}
          {analysis.perspective + 1} perspective.
        </p>
      </details>
    </section>
  );
}
