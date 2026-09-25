'use client';

import { useEffect, useMemo, useState } from 'react';
import { ArrowUpRight, BookOpen, Users, Waves } from 'lucide-react';
import {
  getBoard,
  isSupportedRings,
  MAX_RINGS,
  MIN_RINGS,
} from '@/lib/deltrel/board';
import {
  INITIAL_AI_CAPABILITIES,
  capabilityForController,
  checkAiCapabilities,
  type AiCapabilities,
} from '@/lib/deltrel/ai/capabilities';
import {
  CONTROLLER_TYPES,
  controllerLabel,
  normalizeControllers,
  restrictSelfPlay,
  playerNamesForControllers,
  supportsAiControllers,
  type ControllerType,
  type PlayerControllers,
} from '@/lib/deltrel/ai/controllers';
import { EMPTY } from '@/lib/deltrel/scoring';
import { configHandicap, type GameConfig, type Mode } from '@/lib/deltrel/game';
import { normalizeNewGameConfig } from '@/lib/deltrel/new-game-policy';
import { DELTREL_MAX_HANDICAP } from '@/lib/deltrel/rules';
import { useAppStore, type AiRuntime } from '@/lib/store';
import { DeltrelBoard } from './DeltrelBoard';
import { BrowserAiPreparation } from './BrowserAiPreparation';
import { BrowserAiStrengthControl, budgetFitsCapability } from './BrowserAiStrengthControl';
import { useBrowserAiPreparation } from './useBrowserAiPreparation';
import { BOARD_PRESETS, PLAYER_COLORS } from './theme';
import { DeltrelMark } from './DeltrelMark';
import { RulesDialog } from './RulesDialog';
import styles from './SetupScreen.module.css';

export function SetupScreen() {
  const selfPlayAllowed = useAppStore((s) => s.selfPlayAllowed);
  const startGame = useAppStore((s) => s.startGame);
  const lastConfig = useAppStore((s) => s.config);
  const lastControllers = useAppStore((s) => s.controllers);
  const aiSearchSettings = useAppStore((s) => s.aiSearchSettings);
  const setAiSearchBudget = useAppStore((s) => s.setAiSearchBudget);
  const browserAi = useBrowserAiPreparation();
  const preparedModelVersion = browserAi.status.phase === 'ready' ? browserAi.status.info.modelVersion : null;

  const [showRules, setShowRules] = useState(false);
  const [mode, setMode] = useState<Mode>(lastConfig.mode);
  const [rings, setRings] = useState(lastConfig.rings);
  const [handicap, setHandicap] = useState(() =>
    configHandicap(normalizeNewGameConfig(lastConfig)),
  );
  const pieRule = handicap === 1;
  const [names, setNames] = useState<[string, string]>(() =>
    playerNamesForControllers(lastConfig.playerNames, lastControllers),
  );
  const [controllers, setControllers] = useState<PlayerControllers>(() =>
    restrictSelfPlay(normalizeControllers(lastConfig, lastControllers), selfPlayAllowed),
  );
  const [capabilities, setCapabilities] = useState<AiCapabilities>(
    INITIAL_AI_CAPABILITIES,
  );
  const [capabilityCheck, setCapabilityCheck] = useState(0);


  const board = useMemo(() => getBoard(rings), [rings]);
  const emptyStones = useMemo(() => new Int8Array(board.n).fill(EMPTY), [board]);
  const aiAllowed = supportsAiControllers({
    mode,
    pieRule,
    handicap: pieRule ? 1 : handicap,
  });
  const selectedCapabilities = controllers.map((controller) =>
    capabilityForController(capabilities, controller),
  );
  const selectedAiRuntimes = Array.from(
    new Set(
      controllers.filter(
        (controller): controller is AiRuntime => controller !== 'human',
      ),
    ),
  );
  const controllersReady = selectedCapabilities.every(
    (capability) => capability.status === 'available',
  ) && (!controllers.includes('local') || browserAi.authorized);
  const engineSettingsReady =
    selectedAiRuntimes.every((runtime) => {
      const capability = capabilities[runtime];
      return capability.status !== 'available' ||
        budgetFitsCapability(aiSearchSettings[runtime], capability.search);
    });

  useEffect(() => {
    const controller = new AbortController();
    void checkAiCapabilities(controller.signal).then((result) => {
      if (!controller.signal.aborted) setCapabilities(result);
    });
    return () => controller.abort();
  }, [capabilityCheck, preparedModelVersion]);

  const chooseMode = (nextMode: Mode) => {
    setMode(nextMode);
  };

  const chooseRings = (nextRings: number) => {
    setRings(nextRings);
    if (nextRings !== MAX_RINGS) setHandicap(1);
  };

  const chooseController = (player: number, controller: ControllerType) => {
    setControllers((previous) => {
      const next: PlayerControllers = [...previous];
      next[player] = controller;
      if (!selfPlayAllowed && next.every((value) => value !== 'human')) return previous;
      return next;
    });
  };

  const chooseBrowserChampion = () => {
    if (controllers.includes('local')) return;
    setControllers(['human', 'local']);
  };

  const checkCapabilitiesAgain = () => {
    setCapabilities(INITIAL_AI_CAPABILITIES);
    setCapabilityCheck((value) => value + 1);
  };

  const start = () => {
    const config: GameConfig = {
      rings,
      mode,
      pieRule,
      handicap: pieRule ? 1 : handicap,
      playerNames: [names[0].trim() || 'Player 1', names[1].trim() || 'Player 2'],
    };
    const validControllers = restrictSelfPlay(normalizeControllers(config, controllers), selfPlayAllowed);
    if (
      !engineSettingsReady ||
      (validControllers.includes('local') && !browserAi.authorized) ||
      validControllers.some(
        (controller) =>
          capabilityForController(capabilities, controller).status !== 'available',
      )
    ) {
      return;
    }
    startGame(config, validControllers);
  };

  const preset = BOARD_PRESETS.find((p) => p.rings === rings);
  const opening = pieRule ? 'pie' : 'handicap';
  const modeSummary = `${mode === 'classic' ? 'Classic' : 'Double'} · ${
    pieRule ? 'Even (pie)' : `${handicap}-stone handicap`
  } · ${rings} rings`;

  return (
    <main className={`screen-safe ${styles.screen}`}>
      <header className={`${styles.header}`}>
        <div className={styles.brand}>
          <DeltrelMark className={styles.brandMark} />
          <div>
            <h1 className={styles.wordmark}>Deltrel<span aria-hidden>.</span></h1>
            <p className={styles.brandCaption}>A game of connection</p>
          </div>
        </div>
        <div className={styles.headerAside}>
          <p>Two players.<br /><span>A world between you.</span></p>
          <button type="button" onClick={() => setShowRules(true)} className={styles.rulesButton}>
            <BookOpen size={15} aria-hidden /> How to play
          </button>
        </div>
      </header>

      <div className={styles.content}>
        <section className={`${styles.atlas}`} aria-label="The estuary">
          <div className={styles.atlasHeading}>
            <div>
              <p className={styles.eyebrow}>Where the river meets the sea</p>
              <h2>Connection runs deep.</h2>
            </div>
            <Waves className={styles.tideIcon} aria-hidden />
          </div>
          <div data-setup-preview className={styles.preview}>
            <div className={styles.chartLabel} aria-hidden>
              <span>THE ESTUARY</span><span>{String(rings).padStart(2, '0')} / RINGS</span>
            </div>
            <div className={styles.previewBoard}>
              <DeltrelBoard
                key={rings}
                board={board}
                stones={emptyStones}
                className="block h-full w-full"
              />
            </div>
            <div className={styles.chartLegend}>
              <span><b>{board.n}</b> places to begin</span>
              <span><b>{board.shoreCount}</b> shores</span>
              <span><b>5</b> capes</span>
            </div>
          </div>
          <div className={styles.atlasFooter}>
            <p>Reach the shore. Join your networks.<br />Make every connection count.</p>
            <span className={styles.matchTotal}>MATCH TOTAL <b>{board.shoreCount + 1}</b></span>
          </div>
        </section>

        <section className={`${styles.setup}`} aria-label="New game setup">
          <div className={styles.setupHeading}>
            <span className={styles.eyebrow}>Your next encounter</span>
            <h2>Set your course.</h2>
          </div>
          <div className={`${styles.setupScroll} thin-scroll`}>
          <BrowserAiPreparation
            status={browserAi.status}
            authorized={browserAi.authorized}
            capability={capabilities.local}
            selected={controllers.includes('local')}
            notice={browserAi.notice}
            onSelect={chooseBrowserChampion}
            onPrepare={() => { void browserAi.prepare(); }}
            onCancel={browserAi.cancel}
            onCheck={checkCapabilitiesAgain}
          />
          <BrowserAiStrengthControl
            budget={aiSearchSettings.local}
            capability={capabilities.local}
            onChange={(budget) => setAiSearchBudget('local', budget)}
          />

          {/* Mode */}
          <section>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.14em] text-muted">
              Variant
            </h2>
            <div className="grid grid-cols-2 gap-2.5">
              {(
                [
                  { id: 'classic', title: 'Classic Deltrel', sub: '1 stone per turn' },
                  { id: 'double', title: 'Double Deltrel', sub: '2 stones per turn · first turn 1' },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => chooseMode(m.id)}
                  aria-pressed={mode === m.id}
                  aria-label={`${m.title}, ${m.sub}`}
                  className={`min-h-16 rounded-2xl border px-4 py-3 text-left transition-[border-color,background-color,box-shadow,transform] duration-200 active:scale-[0.99] ${
                    mode === m.id
                      ? 'border-sand/70 bg-sand-faint shadow-[inset_0_0_0_1px_rgba(228,201,161,0.18)]'
                      : 'border-white/10 bg-white/[0.03] hover:border-sand/35'
                  }`}
                >
                  <span className="font-display block text-xl text-ink">{m.id === 'classic' ? 'Classic' : 'Double'}</span>
                  <span className="mt-1 block text-xs text-muted">{m.sub}</span>
                </button>
              ))}
            </div>
          </section>

          <section>
            <div>
              <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.14em] text-muted">Opening</h2>
              <div role="group" aria-label="Opening rule" className="grid grid-cols-2 gap-2">
                {([
                  ['pie', 'Even (pie)', 'Second player may swap'],
                  ['handicap', 'Handicap', 'Extra stones · 10 rings only'],
                ] as const).map(([id, label, description]) => (
                  <button
                    key={id}
                    type="button"
                    aria-label={`${label} opening`}
                    aria-pressed={opening === id}
                    disabled={id === 'handicap' && rings !== MAX_RINGS}
                    onClick={() => {
                      setHandicap(id === 'handicap' ? Math.max(2, handicap) : 1);
                    }}
                    className={`min-h-16 rounded-xl border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                      opening === id ? 'border-sand/70 bg-sand-faint' : 'border-white/10 bg-white/[0.03] hover:border-sand/35'
                    }`}
                  >
                    <span className="block text-sm text-ink">{label}</span>
                    <span className="mt-1 block text-[10px] leading-relaxed text-muted">{description}</span>
                  </button>
                ))}
              </div>
              {opening === 'pie' && <p className="mt-2 text-xs leading-relaxed text-muted">
                After the opening stone, {names[1].trim() || 'Player 2'} may swap sides.
              </p>}
            </div>
            {opening === 'handicap' && <div className="control-surface mt-3 rounded-xl px-4 py-2.5">
              <div className="flex items-center justify-between gap-4">
                <span>
                  <span className="block text-sm text-ink">Handicap</span>
                  <span className="block text-xs text-muted">
                    {names[0].trim() || 'Player 1'} opens with {handicap} stones
                  </span>
                </span>
                <span className="text-xs uppercase tracking-[0.12em] text-muted">
                  {handicap} stones
                </span>
              </div>
              <div
                role="radiogroup"
                aria-label="Handicap stones"
                className="mt-2 grid grid-cols-8 gap-1"
              >
                {Array.from({ length: DELTREL_MAX_HANDICAP - 1 }, (_, index) => index + 2).map(
                  (stones) => (
                    <button
                      key={stones}
                      type="button"
                      role="radio"
                      aria-checked={handicap === stones}
                      aria-label={`${stones} handicap stones`}
                      tabIndex={handicap === stones ? 0 : -1}
                      data-handicap={stones}
                      onClick={() => setHandicap(stones)}
                      onKeyDown={(event) => {
                        const next = event.key === 'Home' ? 2
                          : event.key === 'End' ? DELTREL_MAX_HANDICAP
                            : event.key === 'ArrowRight' || event.key === 'ArrowDown'
                              ? stones === DELTREL_MAX_HANDICAP ? 2 : stones + 1
                              : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
                                ? stones === 2 ? DELTREL_MAX_HANDICAP : stones - 1
                                : null;
                        if (next === null) return;
                        event.preventDefault();
                        setHandicap(next);
                        event.currentTarget.parentElement?.querySelector<HTMLButtonElement>(`[data-handicap="${next}"]`)?.focus();
                      }}
                      className={`min-h-10 rounded-lg border text-xs transition-[border-color,background-color] duration-200 disabled:cursor-not-allowed disabled:opacity-40 ${
                        handicap === stones
                          ? 'border-sand/70 bg-sand-faint text-ink'
                          : 'border-white/10 bg-white/[0.03] text-muted hover:border-sand/35'
                      }`}
                    >
                      {stones}
                    </button>
                  ),
                )}
              </div>
            </div>}
          </section>

          {/* Board size */}
          <section>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.14em] text-muted">
              Board
            </h2>
            <div className="grid grid-cols-4 gap-2">
              {BOARD_PRESETS.map((p) => (
                <button
                  key={p.rings}
                  type="button"
                  onClick={() => chooseRings(p.rings)}
                  aria-pressed={rings === p.rings}
                  aria-label={`${p.label}, ${p.rings} rings`}
                  className={`min-h-12 rounded-xl border px-2 py-2 text-center transition-[border-color,background-color,transform] duration-200 active:scale-[0.98] ${
                    rings === p.rings
                      ? 'border-sand/70 bg-sand-faint'
                      : 'border-white/10 bg-white/[0.03] hover:border-sand/35'
                  }`}
                >
                  <span className="block text-sm text-ink">{p.label}</span>
                  <span className="block text-[11px] text-muted">{p.rings} rings</span>
                </button>
              ))}
            </div>
            <div className="control-surface mt-3 flex items-center gap-3 rounded-xl px-4 py-2">
              <label htmlFor="rings" className="shrink-0 text-xs text-muted">
                Custom
              </label>
              <input
                id="rings"
                type="range"
                min={MIN_RINGS}
                max={MAX_RINGS}
                step={2}
                value={rings}
                onChange={(event) => {
                  const nextRings = Number(event.target.value);
                  if (isSupportedRings(nextRings)) chooseRings(nextRings);
                }}
                className="w-full accent-[#e4c9a1]"
              />
              <span className="w-20 shrink-0 text-right text-sm text-ink">
                {rings} rings{preset ? '' : ' ·'}
                {preset ? '' : <span className="text-muted"> {board.n}n</span>}
              </span>
            </div>
          </section>

          {/* Players */}
          <section>
            <h2 className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-muted">
              <Users className="h-3.5 w-3.5" aria-hidden /> Players
            </h2>
            <div className="grid grid-cols-1 gap-2.5 min-[480px]:grid-cols-2">
              {[0, 1].map((i) => (
                <div
                  key={i}
                  className="control-surface flex min-w-0 items-center gap-3 rounded-xl px-3 py-2.5"
                >
                  <span
                    aria-hidden
                    className="h-5 w-5 shrink-0 rounded-full shadow-inner"
                    style={{
                      background: `radial-gradient(circle at 35% 30%, ${PLAYER_COLORS[i].bright}, ${PLAYER_COLORS[i].base} 55%, ${PLAYER_COLORS[i].deep})`,
                    }}
                  />
                  <div className="min-w-0 flex-1">
                    <input
                      value={names[i]}
                      maxLength={18}
                      aria-label={`Player ${i + 1} name`}
                      onChange={(e) =>
                        setNames((prev) => {
                          const next: [string, string] = [...prev];
                          next[i] = e.target.value;
                          return next;
                        })
                      }
                      className="w-full bg-transparent text-sm text-ink outline-none placeholder:text-muted"
                      placeholder={`Player ${i + 1}`}
                    />
                    {aiAllowed && (
                      <select
                        value={controllers[i]}
                        aria-label={`Player ${i + 1} controller`}
                        onChange={(event) =>
                          chooseController(i, event.target.value as ControllerType)
                        }
                        className="mt-1 w-full bg-transparent text-xs text-muted outline-none"
                      >
                        {CONTROLLER_TYPES.filter((controller) => selfPlayAllowed || controller === 'human' || controllers[1 - i] === 'human').map((controller) => {
                          const capability = capabilityForController(
                            capabilities,
                            controller,
                          );
                          const suffix =
                            capability.status === 'checking'
                              ? ' (checking…)'
                              : capability.status === 'unavailable'
                                ? ' (unavailable)'
                                : '';
                          return (
                            <option
                              key={controller}
                              value={controller}
                              disabled={capability.status !== 'available'}
                              className="bg-[#103e40]"
                            >
                              {controllerLabel(controller)}
                              {suffix}
                            </option>
                          );
                        })}
                      </select>
                    )}
                  </div>
                  <span className="text-xs uppercase tracking-[0.12em] text-muted">
                    {i === 0 ? 'first' : 'second'}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-2 min-h-6 px-1 text-xs" aria-live="polite">
              {!aiAllowed && (
                <p className="text-muted">
                  AI controllers support classic and Double Deltrel, handicaps up to
                  nine stones, and the pie rule (without a handicap).
                </p>
              )}
              {aiAllowed &&
                selectedCapabilities.map(
                  (capability, player) =>
                    capability.status === 'unavailable' &&
                    controllers[player] !== 'human' && (
                      <p
                        key={`${player}-${capability.code}`}
                        role="alert"
                        className="text-danger"
                      >
                        {controllerLabel(controllers[player])}
                        : {capability.reason}
                      </p>
                    ),
                )}
              {aiAllowed && selectedAiRuntimes.map((runtime) => {
                const capability = capabilities[runtime];
                if (capability.status !== 'available' || !capability.search ||
                  budgetFitsCapability(aiSearchSettings[runtime], capability.search)) return null;
                return <p key={runtime} role="alert" className="mb-2 text-danger">
                  The saved thinking-time setting is not available.{' '}
                  <button type="button" className="underline" onClick={() =>
                    setAiSearchBudget(runtime, { ...capability.search!.default })
                  }>Use recommended thinking time</button>
                </p>;
              })}
              {controllers.includes('local') && !browserAi.authorized && (
                <p className="mb-2 text-sand-strong">The AI model must finish preparing before the game can begin.</p>
              )}
              {aiAllowed && (
                  <button
                    type="button"
                    onClick={checkCapabilitiesAgain}
                    className="mt-1 min-h-8 text-left text-sand-strong underline decoration-sand/40 underline-offset-2"
                  >
                    Check AI availability again
                  </button>
                )}
            </div>
          </section>



          </div>
          <div className={styles.setupFooter}>
          <p className={styles.modeSummary} aria-live="polite" data-selected-game-mode>
            {modeSummary}
          </p>
          <button
            type="button"
            onClick={start}
            disabled={!controllersReady || !engineSettingsReady}
            className={styles.beginButton}
          >
            <span>Begin the game</span>
            <ArrowUpRight size={21} aria-hidden />
          </button>
          </div>
        </section>
      </div>
      <footer className={styles.footer}>
        <span>Deltrel · A meeting of minds</span>
        <span>Rules by Ea Ea · Made for the moment</span>
      </footer>
      <RulesDialog open={showRules} onClose={() => setShowRules(false)} />
    </main>
  );
}
