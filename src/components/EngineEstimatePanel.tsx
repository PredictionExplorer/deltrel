'use client';

import { useId, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import type { Board } from '@/lib/deltrel/board';
import type { DeltrelAiAnalysis } from '@/lib/deltrel/ai/decision';
import type { AtomicGameAction } from '@/lib/deltrel/ai/protocol';
import type { DeltrelNetworkOutput } from '@/lib/deltrel/ai/network-output';
import { PLAYER_COLORS } from './theme';
import {
  AUXILIARY_HEADS,
  NETWORK_HEADS,
  countDerivedPoints,
  expectedFinalPoints,
  networkHeadTable,
  winProbabilities,
  type NetworkHeadName,
} from './engineEstimatePresentation';
import styles from './EngineEstimatePanel.module.css';

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const outputPercent = (value: number) => value > 0 && value < 0.001
  ? `${(value * 100).toPrecision(3)}%`
  : percent(value);
const signed = (value: number, digits = 3) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
const latency = (milliseconds: number) => milliseconds < 1_000
  ? `${milliseconds.toFixed(milliseconds < 10 ? 1 : 0)} ms`
  : `${(milliseconds / 1_000).toFixed(2)} s`;

export interface EngineEstimateContext {
  analyzedPly: number | null;
  displayedPly: number;
  source: 'server' | 'local' | null;
  status: 'ready' | 'thinking' | 'unavailable' | 'error';
  message?: string;
  action?: AtomicGameAction;
  applied?: boolean;
  isExact?: boolean;
  proof?: boolean;
  phase?: 'live' | 'review' | 'ended';
}

export interface EngineEstimatePanelProps {
  analysis: DeltrelAiAnalysis | null;
  board: Board;
  playerNames: readonly [string, string];
  context?: EngineEstimateContext;
  onAnalyze?: () => void;
  canAnalyze?: boolean;
  onPause?: () => void;
  canPause?: boolean;
  /** Retained for older callers; all engine outputs are available during normal play. */
  showSearchDetails?: boolean;
}

function Disclosure({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={styles.disclosure}>
      <button type="button" className={styles.disclosureButton} aria-expanded={open} aria-controls={id} onClick={() => setOpen((value) => !value)}>
        <span><span>{title}</span>{hint && <small>{hint}</small>}</span>
        <svg aria-hidden viewBox="0 0 16 16" className={open ? styles.chevronOpen : styles.chevron}><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      <div id={id} className={styles.disclosureContent} hidden={!open}>{open && children}</div>
    </div>
  );
}

interface TableRow { id: string; label: string; cells: ReactNode[] }

function DataTable({ caption, columns, rows, searchable = false }: {
  caption: string;
  columns: string[];
  rows: TableRow[];
  searchable?: boolean;
}) {
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState('');
  const searchId = useId();
  const filtered = useMemo(() => rows.filter((row) => row.label.toLowerCase().includes(query.trim().toLowerCase())), [rows, query]);
  const pageSize = 12;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const offset = currentPage * pageSize;
  const visible = filtered.slice(offset, offset + pageSize);
  return (
    <div className={styles.dataTable}>
      {searchable && (
        <div className={styles.tableSearch}>
          <label htmlFor={searchId}>Find a point or value</label>
          <input id={searchId} type="search" value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="Filter rows" autoComplete="off" />
        </div>
      )}
      <div className={styles.tableScroll}>
        <table>
          <caption className="sr-only">{caption}</caption>
          <thead><tr>{columns.map((column, index) => <th key={index} scope="col">{column}</th>)}</tr></thead>
          <tbody>{visible.map((row) => <tr key={row.id}><th scope="row">{row.label}</th>{row.cells.map((cell, index) => <td key={index}>{cell}</td>)}</tr>)}</tbody>
        </table>
      </div>
      {filtered.length === 0 && <p className={styles.note}>No matching values.</p>}
      <div className={styles.pagination}>
        <span aria-live="polite">{filtered.length === 0 ? '0 results' : `${offset + 1}–${Math.min(offset + pageSize, filtered.length)} of ${filtered.length}`}</span>
        {pageCount > 1 && <div>
          <button type="button" aria-label={`Previous page of ${caption}`} disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button>
          <button type="button" aria-label={`Next page of ${caption}`} disabled={currentPage + 1 === pageCount} onClick={() => setPage(currentPage + 1)}>Next</button>
        </div>}
      </div>
    </div>
  );
}

function ProbabilityChart({ title, series, minimum }: {
  title: string;
  series: { label: string; color: string; values: number[] }[];
  minimum: number;
}) {
  const titleId = useId();
  const peak = Math.max(0.001, ...series.flatMap((item) => item.values));
  const bins = series[0]?.values.length ?? 0;
  if (bins === 0) return null;
  const x = (index: number) => 24 + index / Math.max(1, bins - 1) * 282;
  const baseline = 83;
  const maximum = minimum + bins - 1;
  const middleIndex = Math.floor((bins - 1) / 2);
  return (
    <svg viewBox="0 0 320 110" role="img" aria-labelledby={titleId} className={styles.chart}>
      <title id={titleId}>{title} probability distribution</title>
      <desc>Values from {minimum} to {maximum}. The largest probability is {percent(peak)}. Series: {series.map((item) => item.label).join(', ')}. Exact values are available in the table below.</desc>
      <path d="M24 12H306M24 83H306" stroke="currentColor" strokeOpacity=".14" strokeWidth=".6" />
      {series.map((item, seriesIndex) => {
        const line = item.values.map((value, index) => `${index === 0 ? 'M' : 'L'}${x(index).toFixed(2)} ${(baseline - value / peak * 66).toFixed(2)}`).join('');
        return <g key={seriesIndex}><path d={`${line}L306 ${baseline}L24 ${baseline}Z`} fill={item.color} fillOpacity=".14" /><path d={line} fill="none" stroke={item.color} strokeWidth="1.5" /></g>;
      })}
      <g fill="currentColor" fontSize="8" textAnchor="middle">
        <text x="24" y="101">{minimum}</text><text x={x(middleIndex)} y="101">{minimum + middleIndex}</text><text x="306" y="101">{maximum}</text>
        <text x="23" y="8" textAnchor="start">{percent(peak)}</text>
      </g>
    </svg>
  );
}

function NetworkExplorer({ output, board, playerNames }: {
  output: DeltrelNetworkOutput;
  board: Board;
  playerNames: readonly [string, string];
}) {
  const [selected, setSelected] = useState<NetworkHeadName>('policy');
  const [showLogits, setShowLogits] = useState(false);
  const selectId = useId();
  const descriptor = NETWORK_HEADS.find((item) => item.key === selected)!;
  const head = output.heads[selected];
  const table = useMemo(() => head ? networkHeadTable(selected, head, output, board, playerNames) : null, [selected, head, output, board, playerNames]);
  const untrained = AUXILIARY_HEADS.has(selected) && output.auxiliaryStatus === 'untrained';
  const probabilityOrder = selected === 'policy' || selected === 'softPolicy' || selected === 'opponentReply' || selected === 'secondStone';
  const rows = useMemo(() => {
    const ordered = table ? [...table.rows] : [];
    if (probabilityOrder) ordered.sort((left, right) => right.values[0].probability - left.values[0].probability);
    return ordered.map((row): TableRow => ({
    id: row.label,
    label: row.label,
    cells: row.values.map((value, index) => (
      <span key={index} className={styles.probabilityCell}>
        <span>{outputPercent(value.probability)}</span>
        {value.masked && <small>Masked</small>}
        {showLogits && <small>logit {value.logit === null ? '—' : signed(value.logit, 5)}</small>}
      </span>
    )),
    }));
  }, [table, showLogits, probabilityOrder]);

  return (
    <div>
      <div className={styles.explorerControls}>
        <label htmlFor={selectId}>Network output</label>
        <select id={selectId} value={selected} onChange={(event) => setSelected(event.target.value as NetworkHeadName)}>
          {NETWORK_HEADS.map((item) => <option key={item.key} value={item.key}>{item.label}{output.heads[item.key] ? '' : ' · unavailable'}</option>)}
        </select>
        <label className={styles.checkbox}><input type="checkbox" checked={showLogits} onChange={(event) => setShowLogits(event.target.checked)} /> Show raw logits</label>
      </div>
      <p className={styles.note}>{descriptor.description}{probabilityOrder ? ' Ordered from highest to lowest probability.' : ''}</p>
      {untrained && <p className={styles.notice}>This auxiliary output is untrained. Its raw values are shown for inspection, not as a trained forecast.</p>}
      {head && !head.applicable && <p className={styles.notice}>This output does not apply to the analyzed position. The complete values remain available below.</p>}
      {selected === 'opponentReply' && head && <p className={styles.note}>The model always emits a swap entry; a swap is only legal after the opening in a pie game. Future moves are unconditioned forecasts, not a forced continuation of the selected move.</p>}
      {selected === 'secondStone' && head && <p className={styles.note}>This forecast is not conditioned on the move selected by search.</p>}
      {selected === 'scoreMargin' && <p className={styles.note}>Positive differences favor {playerNames[output.perspective]}; negative differences favor {playerNames[1 - output.perspective]}.</p>}
      {head && table ? (
        <>
          {table.supportMinimum !== null && <ProbabilityChart title={descriptor.label} minimum={table.supportMinimum} series={table.valueLabels.map((label, index) => ({
            label,
            color: PLAYER_COLORS[table.valueLabels.length === 1 ? output.perspective : index as 0 | 1].base,
            values: table.rows.map((row) => row.values[index].probability),
          }))} />}
          <DataTable key={selected} caption={descriptor.label} columns={[table.rowLabel, ...table.valueLabels]} rows={rows} searchable={table.rows.length > 12} />
          <p className={styles.note}>All {head.probabilities.length} outputs are included. Masked entries are excluded by the model’s structural mask. Full-precision values are preserved in the JSON export.</p>
        </>
      ) : <p className={styles.emptyDetail}>This output was not returned by this model.</p>}
    </div>
  );
}

function SearchCandidates({ analysis, board }: { analysis: DeltrelAiAnalysis; board: Board }) {
  const rows = useMemo(() => analysis.rootActions.map((action, index) => ({ action, index, visits: analysis.rootVisits[index] }))
    .sort((left, right) => right.visits - left.visits || left.index - right.index)
    .map(({ action, index, visits }): TableRow => ({
      id: String(action.node), label: board.labels[action.node],
      cells: [percent(analysis.rootPolicy[index]), signed(analysis.rootQ[index]), visits.toLocaleString('en-US')],
    })), [analysis, board]);
  return <>
    <p className={styles.note}>All returned placement candidates, ordered by visits. Search policy is the search target distribution, not the raw network move policy. Values are search utility, not win percentages.</p>
    {analysis.swapRecommended && <p className={styles.notice}>The engine selected the pie swap. Swap is reported separately from placement candidates.</p>}
    <DataTable caption="Search candidates" columns={['Move', 'Search policy', 'Value', 'Visits']} rows={rows} searchable={rows.length > 12} />
  </>;
}

function FinalForecasts({ analysis, board, playerNames }: { analysis: DeltrelAiAnalysis; board: Board; playerNames: readonly [string, string] }) {
  const predictions = analysis.predictions;
  const untrained = analysis.networkOutput?.auxiliaryStatus === 'untrained';
  const countPoints = untrained ? null : countDerivedPoints(analysis);
  if (!predictions || untrained) return <p className={styles.emptyDetail}>{untrained ? 'The auxiliary outputs are untrained. Their raw distributions are available under All network outputs.' : 'Final-count and future-move forecasts were not returned by this model. Win chance and expected points above use the core outcome and score forecasts.'}</p>;
  return <>
    <div className={styles.tableScroll}>
      <table>
        <caption className="sr-only">Expected final counts and cape bonus chance for each player</caption>
        <thead><tr><th scope="col">Player</th><th scope="col">Shores</th><th scope="col">Networks</th><th scope="col">Capes</th><th scope="col">Bonus</th></tr></thead>
        <tbody>{predictions.finalCounts.map((counts) => <tr key={counts.player}><th scope="row">{playerNames[counts.player]}</th><td>{counts.shores.toFixed(1)}</td><td>{counts.networks.toFixed(1)}</td><td>{counts.corners.toFixed(1)}</td><td>{percent(counts.cornerBonusProbability)}</td></tr>)}</tbody>
      </table>
    </div>
    <p className={styles.note}>Forecasts of the official final counts. At a clinch, their reference completion gives the remaining points to the losing side. The bonus is the probability of controlling at least three capes.</p>
    {countPoints && <div className={styles.countScores}><span>Points derived from these count forecasts</span><strong>{playerNames[0]} {countPoints[0].toFixed(1)} · {playerNames[1]} {countPoints[1].toFixed(1)}</strong></div>}
    <p className={styles.note}>Count-derived points use shores + bonus chance + twice the difference in network counts. These independent heads can disagree with the score forecast above and with the board’s fixed final total.</p>
    <dl className={styles.metadata}>
      {[{ label: 'Second stone', forecast: predictions.secondStone }, { label: 'Next reply', forecast: predictions.opponentReply }].map(({ label, forecast }) => {
        if (!forecast) return null;
        return <div key={label}><dt>{playerNames[forecast.player]} · {label.toLowerCase()}</dt><dd>{forecast.kind === 'swap' ? 'Pie swap' : board.labels[forecast.node!]} · {percent(forecast.probability)}</dd></div>;
      })}
    </dl>
    <p className={styles.note}>Move forecasts refer to the analyzed position, not a forced continuation. Every value is available in All network outputs when the model returns full distributions.</p>
  </>;
}

function RawOutput({ analysis, board, context }: { analysis: DeltrelAiAnalysis; board: Board; context?: EngineEstimateContext }) {
  const json = useMemo(() => JSON.stringify({ context: context ?? null, board: { rings: board.rings, labels: board.labels }, analysis }, null, 2), [analysis, board, context]);
  const filename = `deltrel-analysis-${analysis.stateHash.replace(/[^a-zA-Z0-9_-]/g, '-')}.json`;
  return <>
    <a className={styles.download} href={`data:application/json;charset=utf-8,${encodeURIComponent(json)}`} download={filename}>Download full JSON</a>
    <label className={styles.rawLabel}>Raw engine output<textarea readOnly spellCheck={false} wrap="off" value={json} rows={12} /></label>
  </>;
}

export function EngineEstimatePanel({ analysis, board, playerNames, context, onAnalyze, canAnalyze, onPause, canPause }: EngineEstimatePanelProps) {
  const titleId = useId();
  const wins = analysis ? winProbabilities(analysis) : null;
  const points = analysis ? expectedFinalPoints(analysis, board) : null;
  const total = board.shoreCount + 1;
  const outsideBoardRange = points?.some((value) => value < -1e-6 || value > total + 1e-6) ?? false;
  const exact = context?.isExact ?? (context ? context.analyzedPly === context.displayedPly : false);
  const position = context?.analyzedPly === 0 ? 'Opening position' : context?.analyzedPly != null ? `After move ${context.analyzedPly}` : 'Evaluated position';
  const source = context?.source ? 'AI' : 'Engine';
  const marginLeader = analysis ? playerNames[analysis.expectedMargin >= 0 ? analysis.perspective : 1 - analysis.perspective] : '';
  const returnedHeads = analysis?.networkOutput ? Object.values(analysis.networkOutput.heads).filter(Boolean).length : 0;
  const staleLabel = context && !exact ? context.analyzedPly !== null && context.analyzedPly < context.displayedPly ? 'Earlier position' : 'Different position' : null;
  const statusLabel = context?.status === 'thinking' ? 'Updating' : context?.status === 'error' || context?.status === 'unavailable' ? analysis ? 'Last estimate' : 'Unavailable' : analysis ? 'Ready' : 'Awaiting analysis';

  return (
    <section aria-labelledby={titleId} className={styles.panel} data-engine-estimate>
      <header className={styles.header}>
        <div><p className={styles.eyebrow}>Reading the water</p><h2 id={titleId}>Engine estimate</h2></div>
        <span className={styles.status} data-status={context?.status}>{statusLabel}</span>
      </header>
      <div className={styles.summary}>
        <p className={styles.position}>{analysis ? `${source} · ${position} · ${playerNames[analysis.perspective]} to move` : 'Both players’ forecasts will appear here.'}</p>
        {staleLabel && analysis && <p className={styles.notice}>{staleLabel} · viewing move {context!.displayedPly}</p>}
        {context?.proof && <p className={styles.notice}>{analysis ? 'This estimate belongs to the played position. The displayed proof stones are hypothetical.' : 'Proof stones are hypothetical. Return to the played position to view engine estimates.'}</p>}
        <div className={styles.players}>
          {([0, 1] as const).map((player) => <article key={player} aria-label={`${playerNames[player]} forecast`} className={styles.player} style={{ '--player-color': PLAYER_COLORS[player].base } as CSSProperties}>
            <h3><span aria-hidden className={styles.playerDot} />{playerNames[player]}</h3>
            <span className={styles.metricLabel}>Win chance</span>
            <strong className={styles.winChance}>{wins ? percent(wins[player]) : '—'}</strong>
            <span className={styles.points}><strong>{points ? points[player].toFixed(1) : '—'}</strong><span>expected points</span></span>
          </article>)}
        </div>
        {wins && <div className={styles.winBar} aria-hidden><span style={{ width: `${wins[0] * 100}%`, background: PLAYER_COLORS[0].base }} /><span style={{ width: `${wins[1] * 100}%`, background: PLAYER_COLORS[1].base }} /></div>}
        {analysis && <p className={styles.margin}>{Math.abs(analysis.expectedMargin) < 0.05 ? 'Even · 0.0 expected point difference' : `${marginLeader} +${Math.abs(analysis.expectedMargin).toFixed(1)} expected points`}</p>}
        {analysis && <p className={styles.note}>Expected final points come from the score-margin forecast and this board’s {total}-point final total. Forecasts update after each completed search.</p>}
        {outsideBoardRange && <p className={styles.notice}>The model’s score forecast extends beyond this board’s possible final totals. The original forecast is shown without clipping.</p>}
        <div role="status" aria-live="polite" aria-atomic="true" className={styles.progress}>
          {context?.message ?? (context?.status === 'thinking' ? 'Calculating a new evaluation…' : !analysis ? 'No evaluation is available for this position yet.' : null)}
        </div>
        {context?.action && analysis && <p className={styles.selectedMove}>{context.applied ? 'Selected move' : 'Suggested move'} <strong>{context.action.type === 'swap' ? 'Pie swap' : board.labels[context.action.node]}</strong></p>}
        {(onAnalyze || (onPause && canPause)) && <div className={styles.actions}>
          {onAnalyze && <button type="button" onClick={onAnalyze} disabled={canAnalyze === false || context?.status === 'thinking'}>Analyze position</button>}
          {onPause && canPause && <button type="button" onClick={onPause}>Pause AI</button>}
        </div>}
      </div>
      {analysis && <div className={styles.details}>
        <Disclosure title="Search candidates" hint={`${analysis.rootActions.length} placements · policy, value, visits`}><SearchCandidates analysis={analysis} board={board} /></Disclosure>
        <Disclosure title="Final-count forecasts" hint="Shores, networks, capes and future moves"><FinalForecasts analysis={analysis} board={board} playerNames={playerNames} /></Disclosure>
        <Disclosure title="All network outputs" hint={analysis.networkOutput ? `${returnedHeads} outputs · every point and probability` : 'Full distributions unavailable'}>
          {analysis.networkOutput ? <NetworkExplorer output={analysis.networkOutput} board={board} playerNames={playerNames} /> : <p className={styles.emptyDetail}>This engine did not return its full network tensors. The available summary and search outputs are still shown here.</p>}
        </Disclosure>
        <Disclosure title="Search and model details" hint={`${analysis.simulations.toLocaleString('en-US')} simulations · ${latency(analysis.timingMs.total)}`}>
          <dl className={styles.metadata}>
            {([
              ['Model value', signed(analysis.modelValue)], ['Search input value', signed(analysis.searchValue)], ['Searched root value', signed(analysis.rootValue)],
              ['Expected margin', signed(analysis.expectedMargin)], ['Simulations', analysis.simulations.toLocaleString('en-US')], ['Maximum considered', analysis.maxConsidered.toLocaleString('en-US')],
              ['Queue time', latency(analysis.timingMs.queue)], ['Model load', latency(analysis.timingMs.modelLoad)], ['Inference and search', latency(analysis.timingMs.inferenceSearch)], ['Total time', latency(analysis.timingMs.total)],
              ['Model version', analysis.modelVersion], ['Model step', analysis.modelStep?.toLocaleString('en-US') ?? 'Not reported'], ['Model identity', analysis.modelIdentity ?? 'Not reported'], ['Position identity', analysis.stateHash], ['Perspective', playerNames[analysis.perspective]], ['Swap selected', analysis.swapRecommended ? 'Yes' : 'No'],
            ] as const).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
          </dl>
          <p className={styles.note}>Model value is win minus loss probability. Search input value is the evaluation before simulations and may equal model value. Searched root value combines the search results. These values are evaluation utilities, not win percentages.</p>
        </Disclosure>
        <Disclosure title="Raw engine output" hint="Full precision · view or download JSON"><RawOutput analysis={analysis} board={board} context={context} /></Disclosure>
      </div>}
    </section>
  );
}
