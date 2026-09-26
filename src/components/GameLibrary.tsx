'use client';

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { BookOpen, Check, Copy, Download, FolderOpen, Pause, Play, Share2, Trash2, Upload, X } from 'lucide-react';
import { gameRecordResult, MAX_GAME_RECORD_TEXT_LENGTH, parseGameRecord, serializeGameRecord, type GameRecord } from '@/lib/deltrel/game-record';
import { deleteGameRecord, retryGameLibrarySaves, saveGameRecord, useGameLibrary } from '@/lib/game-library';
import { controllerLabel } from '@/lib/deltrel/ai/controllers';
import { replay } from '@/lib/deltrel/game';
import { scorePosition } from '@/lib/deltrel/scoring';
import { buildTimeline, lastCompletedTurnMoves } from '@/lib/deltrel/timeline';
import { currentGameRecord, useAppStore } from '@/lib/store';
import { DeltrelBoard } from './DeltrelBoard';
import { ModalDialog } from './ModalDialog';
import { MovesPanel } from './MovesPanel';
import { ScorePanel } from './ScorePanel';
import { BOARD_PRESETS } from './theme';

const button = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors hover:border-sand/60 disabled:opacity-40';
const field = 'w-full rounded-xl border border-white/20 bg-black/20 p-3 text-sm text-ink placeholder:text-muted';

function resultLabel(record: GameRecord): string {
  const outcome = gameRecordResult(record);
  if (outcome.winner === null) return 'In progress';
  const reason = outcome.termination === 'clinch' ? 'cannot be caught'
    : outcome.termination === 'resignation' ? 'resignation' : 'board full';
  return `${record.config.playerNames[outcome.winner]} won · ${reason}`;
}

function description(record: GameRecord): string {
  const board = BOARD_PRESETS.find((preset) => preset.rings === record.config.rings)?.label;
  return `${record.config.mode === 'double' ? 'Double' : 'Classic'} · ${board} (${record.config.rings} rings) · ${record.log.length} ${record.log.length === 1 ? 'move' : 'moves'}`;
}

function download(record: GameRecord): string | null {
  try {
    const url = URL.createObjectURL(new Blob([serializeGameRecord(record)], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `deltrel-${record.createdAt.slice(0, 10)}-${record.id.slice(0, 8)}.dgn`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    return null;
  } catch (cause) {
    return cause instanceof Error ? cause.message : 'This game could not be downloaded. Copy its text instead.';
  }
}

export function ShareGameRecord({ record }: { record: GameRecord }) {
  const { text, error } = useMemo(() => {
    try { return { text: serializeGameRecord(record), error: null }; }
    catch (cause) { return { text: '', error: cause instanceof Error ? cause.message : 'This game could not be exported.' }; }
  }, [record]);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const [notice, setNotice] = useState('');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setNotice('Copied. Send this text to another player.');
    } catch {
      textarea.current?.focus();
      textarea.current?.select();
      setNotice('Text selected. Use your device’s Copy command.');
    }
  };
  return <section aria-label="Share game" className="space-y-3">
    <p className="text-sm leading-relaxed text-muted">Copy the complete game below. Another player can paste it into Import game to replay every move.</p>
    {error && <p role="alert" className="text-sm text-danger">{error}</p>}
    <label className="block text-xs font-semibold text-muted" htmlFor="game-record-text">Deltrel Game Notation (.dgn)</label>
    <textarea ref={textarea} id="game-record-text" aria-label="Game notation" readOnly value={text} rows={13} spellCheck={false} className={`${field} resize-y font-mono text-xs leading-relaxed`} />
    <div className="flex flex-wrap gap-2">
      <button type="button" className={button} disabled={!!error} onClick={() => void copy()}><Copy size={16} aria-hidden /> Copy game</button>
      <button type="button" className={button} disabled={!!error} onClick={() => setNotice(download(record) ?? 'Game downloaded.')}><Download size={16} aria-hidden /> Download .dgn</button>
    </div>
    <p role="status" className="min-h-5 text-xs text-sand">{notice}</p>
  </section>;
}

function RecordReview({ record, onBack }: { record: GameRecord; onBack: () => void }) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [ply, setPly] = useState(record.log.length);
  const [playing, setPlaying] = useState(false);
  const [share, setShare] = useState(false);
  const [influence, setInfluence] = useState(false);
  const game = useMemo(() => replay(record.config, record.log.slice(0, ply)), [record, ply]);
  const score = useMemo(() => scorePosition(game.board, game.stones), [game]);
  const timeline = useMemo(() => buildTimeline(record.config, record.log), [record]);
  const lastMoves = useMemo(() => lastCompletedTurnMoves(timeline, ply), [timeline, ply]);
  const seek = (next: number) => { setPlaying(false); setPly(Math.max(0, Math.min(record.log.length, next))); };
  useEffect(() => { headingRef.current?.focus(); }, []);
  useEffect(() => {
    if (!playing || ply >= record.log.length) return;
    const timer = window.setTimeout(() => setPly(ply + 1), 850);
    return () => window.clearTimeout(timer);
  }, [playing, ply, record.log.length]);
  const keyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.defaultPrevented || event.metaKey || event.altKey || event.ctrlKey || event.shiftKey) return;
    if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable="true"], svg')) return;
    const next = event.key === 'ArrowLeft' ? ply - 1 : event.key === 'ArrowRight' ? ply + 1 : event.key === 'Home' ? 0 : event.key === 'End' ? record.log.length : null;
    if (next === null) return;
    event.preventDefault();
    event.currentTarget.focus();
    seek(next);
  };
  const atEnd = ply === record.log.length;
  return <div onKeyDown={keyDown} tabIndex={-1}>
    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
      <button type="button" onClick={onBack} className={button}>← All games</button>
      <button type="button" aria-expanded={share} onClick={() => { setShare(!share); setPlaying(false); }} className={button}><Share2 size={16} aria-hidden /> Share this game</button>
    </div>
    <h3 ref={headingRef} tabIndex={-1} className="break-words font-display text-2xl text-sand-strong">{record.config.playerNames.join(' vs ')}</h3>
    <p className="mt-1 text-sm text-muted">{description(record)}</p>
    <p className="mt-1 text-xs text-muted">Settings at save: {record.controllers.map((controller) => controller === 'human' ? 'Human' : `${controllerLabel(controller)} · ${record.aiSearchSettings[controller].simulations === 544 && record.aiSearchSettings[controller].maxConsidered === 16 ? 'Standard' : record.aiSearchSettings[controller].simulations === 4096 && record.aiSearchSettings[controller].maxConsidered === 64 ? 'Deep' : 'Custom'}`).join(' vs ')}</p>
    <p className="mt-2 break-words text-sm font-medium text-sand" data-record-result>{resultLabel(record)}</p>
    {share ? <div className="mt-4"><ShareGameRecord record={record} /></div> : <div className="mt-3 grid items-start gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(18rem,1fr)]">
      <div className="min-w-0">
        <DeltrelBoard board={game.board} stones={game.stones} nodeOwner={score.nodeOwner} aliveStone={score.aliveStone} lastMove={game.lastMove} currentTurnMoves={game.currentTurnMoves} lastTurnMoves={lastMoves} currentTurnCapacity={game.currentTurnMoves.length + game.movesLeft} playerNames={record.config.playerNames} showTerritory={influence} interactive={false} className="block w-full" />
        <div className="flex items-center gap-3">
          <button type="button" className={button} disabled={record.log.length === 0} aria-label={playing && !atEnd ? 'Pause replay' : 'Play replay'} onClick={() => { if (atEnd) setPly(0); setPlaying(!playing || atEnd); }}>
            {playing && !atEnd ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />}
          </button>
          <input type="range" aria-label="Review move" min={0} max={record.log.length} value={ply} onChange={(event) => seek(Number(event.target.value))} className="min-w-0 flex-1 accent-[#e8c48b]" />
          <output className="font-mono text-xs text-muted" aria-live="polite">{ply} / {record.log.length}</output>
        </div>
        <p className="mt-2 text-xs text-muted">Read-only review · ← → step · Home / End jump</p>
        <label className="mt-2 flex min-h-11 items-center gap-2 text-sm text-muted"><input type="checkbox" checked={influence} onChange={(event) => setInfluence(event.target.checked)} className="accent-[#e8c48b]" /> Show influence</label>
      </div>
      <div className="min-w-0 space-y-4">
        <MovesPanel timeline={timeline} total={record.log.length} currentPly={ply} playerNames={record.config.playerNames} canRewind={false} onSeek={seek} onRewind={() => {}} finalLabel="Saved position" />
        <ScorePanel game={game} score={score} controllers={record.controllers} view={game.over ? { kind: 'live' } : atEnd && record.earlyOutcome ? { kind: 'ended' } : { kind: 'review', ply, total: record.log.length, readOnly: true }} />
      </div>
    </div>}
  </div>;
}

type LibraryTab = 'saved' | 'import' | 'share';

export function GameLibraryDialog({ onClose, initialTab = 'saved' }: { onClose: () => void; initialTab?: LibraryTab }) {
  const { records, error } = useGameLibrary();
  const [tab, setTab] = useState<LibraryTab>(initialTab);
  const [selected, setSelected] = useState<GameRecord | null>(null);
  const [query, setQuery] = useState('');
  const [importText, setImportText] = useState('');
  const [importError, setImportError] = useState('');
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const wasReviewing = useRef(false);
  useEffect(() => {
    if (wasReviewing.current && !selected) searchRef.current?.focus();
    wasReviewing.current = selected !== null;
  }, [selected]);
  const current = currentGameRecord(useAppStore.getState());
  const filtered = records.filter((record) => `${record.config.playerNames.join(' ')} ${description(record)} ${resultLabel(record)} ${record.controllers.map(controllerLabel).join(' ')}`.toLowerCase().includes(query.toLowerCase()));
  const importGame = () => {
    try {
      const record = parseGameRecord(importText);
      saveGameRecord(record);
      setImportError('');
      setSelected(record);
    } catch (cause) {
      setImportError(cause instanceof Error ? cause.message : 'This game could not be read.');
    }
  };
  return <ModalDialog open onClose={onClose} ariaLabel="Game library" className="max-w-5xl!">
    <div className="panel-surface max-h-[calc(100dvh-2rem)] overflow-y-auto rounded-3xl p-4 shadow-2xl sm:p-6">
      <header className="mb-4 flex items-center justify-between gap-3">
        <div><h2 className="font-display text-2xl font-semibold text-sand-strong">Your games</h2><p className="mt-1 text-xs text-muted">Saved automatically in this browser. Export to keep a backup or share.</p></div>
        <button type="button" aria-label="Close game library" onClick={onClose} className={`${button} shrink-0`}><X size={18} aria-hidden /></button>
      </header>
      {error && <div role="alert" className="mb-4 rounded-xl border border-danger/40 p-3 text-sm text-danger">{error}<button type="button" onClick={() => retryGameLibrarySaves()} className={`${button} mt-2`}>Retry saving</button></div>}
      {downloadError && <p role="alert" className="mb-3 text-sm text-danger">{downloadError}</p>}
      {selected ? <RecordReview key={selected.id} record={selected} onBack={() => { setSelected(null); setTab('saved'); }} /> : <>
        <div className="mb-5 flex flex-wrap gap-2" role="group" aria-label="Game library views">
          <button type="button" aria-pressed={tab === 'saved'} onClick={() => setTab('saved')} className={`${button} ${tab === 'saved' ? 'bg-sand/15' : ''}`}><BookOpen size={16} aria-hidden /> Saved games <span className="text-muted">{records.length}</span></button>
          <button type="button" aria-pressed={tab === 'import'} onClick={() => setTab('import')} className={`${button} ${tab === 'import' ? 'bg-sand/15' : ''}`}><Upload size={16} aria-hidden /> Import game</button>
          {current && <button type="button" aria-pressed={tab === 'share'} onClick={() => setTab('share')} className={`${button} ${tab === 'share' ? 'bg-sand/15' : ''}`}><Share2 size={16} aria-hidden /> Share current game</button>}
        </div>
        {tab === 'share' && current ? <ShareGameRecord record={current} /> : tab === 'import' ? <section className="space-y-3" aria-label="Import game">
          <p className="text-sm text-muted">Paste a Deltrel game record to save it and open a review board.</p>
          <label className="block text-xs font-semibold text-muted" htmlFor="import-record">Game notation</label>
          <textarea id="import-record" value={importText} onChange={(event) => { setImportText(event.target.value); setImportError(''); }} placeholder={'[DGN "1"]\n…'} rows={12} maxLength={MAX_GAME_RECORD_TEXT_LENGTH} spellCheck={false} className={`${field} font-mono text-xs`} />
          <label className="flex min-h-11 flex-wrap items-center gap-3 text-sm text-muted">Or open a .dgn file
            <input type="file" aria-label="Open game file" accept=".dgn,.txt,text/plain" className="max-w-full text-xs" onChange={async (event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              if (file.size > MAX_GAME_RECORD_TEXT_LENGTH * 4) { setImportError('Game files must be smaller than 256 KB.'); return; }
              try { setImportText(await file.text()); setImportError(''); } catch { setImportError('Could not read this file.'); }
            }} />
          </label>
          {importError && <p role="alert" className="text-sm text-danger">{importError}</p>}
          <button type="button" disabled={!importText.trim()} onClick={importGame} className={`${button} border-sand/50 bg-sand/10`}><Check size={16} aria-hidden /> Import and review</button>
        </section> : <section aria-label="Saved games" className="space-y-3">
          <input ref={searchRef} type="search" aria-label="Search saved games" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search players, board, or result…" className={field} />
          {filtered.length === 0 ? <p className="py-8 text-center text-sm text-muted">{records.length ? 'No games match your search.' : 'Your games will appear here as you play. You can also import a game from a friend.'}</p> : <ul className="space-y-2">
            {filtered.map((record) => <li key={record.id} className="rounded-2xl border border-white/10 bg-white/[0.025] p-3">
              <div className="flex items-start justify-between gap-3">
                <button type="button" className="min-w-0 flex-1 text-left" aria-label={`Review ${record.config.playerNames.join(' vs ')}`} onClick={() => setSelected(record)}>
                  <span className="block break-words text-sm font-semibold text-ink">{record.config.playerNames.join(' vs ')}</span>
                  <span className="mt-1 block text-xs text-muted">{record.controllers.map(controllerLabel).join(' vs ')} · {description(record)}</span>
                  <span className="mt-2 block break-words text-xs text-sand">{resultLabel(record)}</span>
                  <time dateTime={record.updatedAt} className="mt-1 block text-xs text-muted">{new Date(record.updatedAt).toLocaleString()}</time>
                </button>
                <div className="flex shrink-0 flex-col gap-1">
                  <button type="button" aria-label={`Download game from ${new Date(record.createdAt).toLocaleDateString()}`} className={button} onClick={() => setDownloadError(download(record))}><Download size={16} aria-hidden /></button>
                  {current?.id !== record.id && <button type="button" aria-label="Delete saved game" className={button} onClick={() => setDeleting(record.id)}><Trash2 size={16} aria-hidden /></button>}
                </div>
              </div>
              {deleting === record.id && <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/10 pt-3"><span className="text-xs text-muted">Remove this saved game?</span><button type="button" className={button} onClick={() => { deleteGameRecord(record.id); setDeleting(null); }}>Delete game</button><button type="button" className={button} onClick={() => setDeleting(null)}>Keep game</button></div>}
            </li>)}
          </ul>}
        </section>}
      </>}
    </div>
  </ModalDialog>;
}

export function GameLibraryButton({ onOpen, share = false }: { onOpen?: () => void; share?: boolean }) {
  const [open, setOpen] = useState(false);
  return <>
    <button type="button" aria-label={share ? 'Share game' : 'Game library'} onClick={() => { onOpen?.(); setOpen(true); }} className={button}>
      {share ? <Share2 size={16} aria-hidden /> : <FolderOpen size={16} aria-hidden />}
      <span className="hidden sm:inline">{share ? 'Share' : 'Games'}</span>
    </button>
    {open && <GameLibraryDialog initialTab={share ? 'share' : 'saved'} onClose={() => setOpen(false)} />}
  </>;
}

export function GameSaveStatus() {
  const { error } = useGameLibrary();
  return error
    ? <p role="alert" className="rounded-xl border border-danger/40 p-3 text-xs text-danger">{error} Open Games to copy a backup or retry saving.</p>
    : <p className="flex items-center gap-1.5 px-1 text-xs text-muted"><Check size={13} aria-hidden /> Saved in this browser · find past matches in Games</p>;
}
