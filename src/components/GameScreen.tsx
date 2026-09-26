'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpenText,
  CircleAlert,
  Eye,
  Flag,
  Redo2,
  Replace,
  Settings2,
  ShieldCheck,
  Trophy,
  Undo2,
} from 'lucide-react';
import { aiMatchLabel, canShowAiInsights, controllerLabel, normalizeControllers, playerNamesForControllers, type ControllerType } from '@/lib/deltrel/ai/controllers';
import { normalizeBrowserStrengthBudget } from '@/lib/deltrel/ai/browser-strength';
import { useAiCapabilities } from './useAiCapabilities';
import { CloudAiControl } from './CloudAiControl';
import {
  DeltrelAiError,
  asDeltrelAiError,
  type DeltrelAiErrorCode,
} from '@/lib/deltrel/ai/errors';
import type {
  DeltrelAiSearchBudget,
  DeltrelAiSearchProgress,
} from '@/lib/deltrel/ai/decision';
import {
  analysisConfigKey,
  recordEngineAnalysis,
  selectEngineAnalysis,
  type StoredEngineAnalysis,
} from '@/lib/deltrel/ai/analysis-history';
import {
  acceptAiResponse,
  buildAiRequest,
  semanticStateFromGame,
  semanticStateHash,
  type DeltrelAiRequest,
} from '@/lib/deltrel/ai/protocol';
import { scoreCompletionBounds } from '@/lib/deltrel/completion-bounds';
import { configHandicap, replay, type GameAction } from '@/lib/deltrel/game';
import {
  EMPTY,
  scorePosition,
  validateTerminalWinner,
} from '@/lib/deltrel/scoring';
import { buildTimeline, lastCompletedTurnMoves } from '@/lib/deltrel/timeline';
import { useAppStore } from '@/lib/store';
import { BoardStage } from './BoardStage';
import { DeltrelMark } from './DeltrelMark';
import { EngineEstimatePanel } from './EngineEstimatePanel';
import { BrowserAiPreparation, browserAiIsPreparing } from './BrowserAiPreparation';
import { BrowserAiStrengthControl, budgetFitsCapability } from './BrowserAiStrengthControl';
import { useBrowserAiPreparation } from './useBrowserAiPreparation';
import {
  ClinchDialog,
  EndGameConfirmDialog,
  ResignDialog,
} from './EndGameDialogs';
import { GameOverOverlay, type GameResult } from './GameOverOverlay';
import { GameStatus, type GameStatusState } from './GameStatus';
import { MovesPanel } from './MovesPanel';
import { GameLibraryButton, GameSaveStatus } from './GameLibrary';
import { RulesDialog } from './RulesDialog';
import { ScorePanel } from './ScorePanel';
import styles from './GameScreen.module.css';
import { PLAYER_COLORS } from './theme';

type AiStatus =
  | { kind: 'idle' }
  | {
      kind: 'thinking';
      controller: Exclude<ControllerType, 'human'>;
      positionKey: string;
      searchProgress: DeltrelAiSearchProgress;
    }
  | {
      kind: 'error';
      controller: Exclude<ControllerType, 'human'>;
      code: DeltrelAiErrorCode;
      message: string;
      retryable: boolean;
    };

interface AiFlight {
  key: string;
  request: DeltrelAiRequest;
  controller: Exclude<ControllerType, 'human'>;
  abortController: AbortController;
  cancelScheduled: boolean;
  cancelled: boolean;
  settled: boolean;
}

function reducedSearchBudget(
  budget: DeltrelAiSearchBudget,
): DeltrelAiSearchBudget | null {
  if (budget.simulations <= 544) return null;
  return { simulations: 544, maxConsidered: 16 };
}

export function GameScreen() {
  const storedConfig = useAppStore((state) => state.config);
  const storedControllers = useAppStore((state) => state.controllers);
  const aiInsightsVisible = useAppStore((state) => canShowAiInsights(state.controllers, state.aiInsightsHidden));
  const controllers = useMemo(() => normalizeControllers(storedConfig, storedControllers), [storedConfig, storedControllers]);
  const config = useMemo(() => {
    const playerNames = playerNamesForControllers(storedConfig.playerNames, controllers);
    return playerNames.every((name, player) => name === storedConfig.playerNames[player])
      ? storedConfig : { ...storedConfig, playerNames };
  }, [storedConfig, controllers]);
  const aiSearchSettings = useAppStore((state) => state.aiSearchSettings);
  const aiPaused = useAppStore((state) => state.aiPaused);
  const log = useAppStore((state) => state.log);
  const redoStack = useAppStore((state) => state.redoStack);
  const reviewing = useAppStore((state) => state.reviewing);
  const earlyOutcome = useAppStore((state) => state.earlyOutcome);
  const clinchAcknowledgement = useAppStore(
    (state) => state.clinchAcknowledgement,
  );
  const undo = useAppStore((state) => state.undo);
  const redo = useAppStore((state) => state.redo);
  const rewindTo = useAppStore((state) => state.rewindTo);
  const rematch = useAppStore((state) => state.rematch);
  const toSetup = useAppStore((state) => state.toSetup);
  const resumeAi = useAppStore((state) => state.resumeAi);
  const pauseAi = useAppStore((state) => state.pauseAi);
  const setPlayerController = useAppStore((state) => state.setPlayerController);
  const setAiSearchBudget = useAppStore((state) => state.setAiSearchBudget);
  const setReviewing = useAppStore((state) => state.setReviewing);
  const acknowledgeClinch = useAppStore((state) => state.acknowledgeClinch);
  const endClinchedGame = useAppStore((state) => state.endClinchedGame);
  const resign = useAppStore((state) => state.resign);
  const { status: browserStatus, authorized: browserAuthorized, notice: browserNotice, prepare: prepareBrowserAi, cancel: cancelBrowserAi } = useBrowserAiPreparation(controllers.includes('local') || (aiInsightsVisible && !controllers.includes('server')));
  const preparedModelVersion = browserStatus.phase === 'ready' ? browserStatus.info.modelVersion : null;
  const { capabilities: runtimeCapabilities, checkAgain: checkCapabilitiesAgain, markCloudUnavailable } = useAiCapabilities(preparedModelVersion);
  // Old saved custom and Quick budgets use the supported public presets.
  const browserSearch = useMemo(() => normalizeBrowserStrengthBudget(aiSearchSettings.local), [aiSearchSettings.local]);

  const [rulesOpen, setRulesOpen] = useState(false);
  const [showInfluence, setShowInfluence] = useState(false);
  const [proofMode, setProofMode] = useState(false);
  /** Number of log actions on the board while reviewing; null = live. */
  const [viewPly, setViewPly] = useState<number | null>(null);
  const [endConfirmOpen, setEndConfirmOpen] = useState(false);
  const [resignConfirmOpen, setResignConfirmOpen] = useState(false);
  const [aiStatus, setAiStatus] = useState<AiStatus>({ kind: 'idle' });
  const [analysisHistory, setAnalysisHistory] = useState<StoredEngineAnalysis[]>([]);
  const [inspectionStatus, setInspectionStatus] = useState<
    { kind: 'idle' } |
    { kind: 'thinking'; stateHash: string } |
    { kind: 'error'; stateHash: string; message: string }
  >({ kind: 'idle' });
  const inspectionRef = useRef<AbortController | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const flightRef = useRef<AiFlight | null>(null);
  const failedRequestKey = useRef<string | null>(null);
  const boardStageRef = useRef<HTMLElement>(null);

  const cancelInspection = useCallback(() => {
    inspectionRef.current?.abort();
    inspectionRef.current = null;
    setInspectionStatus({ kind: 'idle' });
  }, []);

  useEffect(() => () => { inspectionRef.current?.abort(); }, []);

  useEffect(() => useAppStore.subscribe((current, previous) => {
    if (!canShowAiInsights(current.controllers, current.aiInsightsHidden) &&
        canShowAiInsights(previous.controllers, previous.aiInsightsHidden)) {
      cancelInspection();
      setAnalysisHistory([]);
    }
  }), [cancelInspection]);

  const game = useMemo(() => {
    try {
      return replay(config, log);
    } catch {
      return null; // corrupted persisted log
    }
  }, [config, log]);

  const score = useMemo(
    () => (game ? scorePosition(game.board, game.stones) : null),
    [game],
  );

  // Non-destructive move review: `historyView` selects a prefix of the log to
  // put on the board; the live game (and any AI opponent) continues
  // underneath, untouched. Every action that can shrink the log (undo, redo,
  // rewind, rematch, leave) resets `viewPly`, so the render-time clamp only
  // covers transient states.
  const historyView = viewPly !== null && viewPly < log.length ? viewPly : null;
  const viewingHistory = historyView !== null;

  const seek = useCallback((ply: number | null) => {
    cancelInspection();
    const total = useAppStore.getState().log.length;
    setViewPly(ply === null || ply >= total ? null : Math.max(0, ply));
    setProofMode(false);
  }, [cancelInspection]);

  const shownGame = useMemo(() => {
    if (!game || historyView === null) return game;
    return replay(config, log.slice(0, historyView));
  }, [config, game, historyView, log]);

  const shownScore = useMemo(() => {
    if (!shownGame) return null;
    if (shownGame === game) return score;
    return scorePosition(shownGame.board, shownGame.stones);
  }, [game, score, shownGame]);

  const timeline = useMemo(
    () => (game ? buildTimeline(config, log) : null),
    [config, game, log],
  );
  const currentPly = historyView ?? log.length;
  const lastTurnMoves = useMemo(
    () => (timeline ? lastCompletedTurnMoves(timeline, currentPly) : []),
    [currentPly, timeline],
  );
  const completionBounds = useMemo(
    () =>
      game && !game.canSwap
        ? scoreCompletionBounds(game.board, game.stones)
        : null,
    [game],
  );
  const guaranteedWinner = completionBounds?.guaranteedWinner ?? null;
  const clinchAcknowledged =
    guaranteedWinner !== null &&
    clinchAcknowledgement?.winner === guaranteedWinner &&
    log.length >= clinchAcknowledgement.atLogLength;
  const clinchPending =
    Boolean(game) &&
    !game?.over &&
    earlyOutcome === null &&
    guaranteedWinner !== null &&
    !clinchAcknowledged;
  const effectiveOver = Boolean(game?.over || earlyOutcome);
  const proofEligible =
    Boolean(game) &&
    !game?.over &&
    guaranteedWinner !== null &&
    earlyOutcome?.reason !== 'resignation';
  const validProofMode = proofMode && proofEligible;
  const validEndConfirmOpen =
    endConfirmOpen &&
    guaranteedWinner !== null &&
    !effectiveOver &&
    !clinchPending;
  const validResignConfirmOpen = resignConfirmOpen && !effectiveOver;
  const uiBlocksPlay =
    effectiveOver ||
    clinchPending ||
    validProofMode ||
    validEndConfirmOpen ||
    validResignConfirmOpen;

  useEffect(() => {
    if (game === null) toSetup();
  }, [game, toSetup]);

  const aiPositionKey = useMemo(() => {
    if (!game || game.over || uiBlocksPlay) return null;
    const controller = controllers[game.toMove];
    if (controller === 'human') return null;
    if (typeof BigInt !== 'function') return `${controller}:bigint-unavailable`;
    return `${controller}:${semanticStateHash(semanticStateFromGame(game))}`;
  }, [controllers, game, uiBlocksPlay]);

  const cancelActiveAi = useCallback(() => {
    cancelInspection();
    failedRequestKey.current = null;
    setAiStatus({ kind: 'idle' });
    const flight = flightRef.current;
    if (!flight) return;
    flight.cancelScheduled = false;
    flight.cancelled = true;
    flightRef.current = null;
    flight.abortController.abort();
  }, [cancelInspection]);

  const turnController = game ? controllers[game.toMove] : 'human';
  const autoplaySearch = turnController === 'server' ? aiSearchSettings.server : browserSearch;
  const turnCapability = turnController === 'human' ? null : runtimeCapabilities[turnController];
  const autoplayReady = turnCapability?.status === 'available' &&
    budgetFitsCapability(autoplaySearch, turnCapability.search) &&
    (turnController === 'server' || browserAuthorized);
  const autoplaySimulations = autoplaySearch.simulations;
  const autoplayMaxConsidered = autoplaySearch.maxConsidered;

  useEffect(() => {
    if (!game || game.over || aiPaused || uiBlocksPlay) return;
    const controller = controllers[game.toMove];
    if (controller === 'human' || !aiPositionKey) return;
    if (!autoplayReady) return;

    const selectedSearch = { simulations: autoplaySimulations, maxConsidered: autoplayMaxConsidered };
    // Strength changes apply to the next request. Reattach to an in-flight
    // search for this same position instead of discarding its completed work.
    const key = `${aiPositionKey}:${retryNonce}`;
    if (failedRequestKey.current === key) return;
    const scheduleCancellation = (flight: AiFlight) => {
      if (flight.settled) return;
      flight.cancelScheduled = true;
      queueMicrotask(() => {
        if (flightRef.current !== flight || !flight.cancelScheduled) return;
        flight.cancelled = true;
        flightRef.current = null;
        flight.abortController.abort();
        setAiStatus({ kind: 'idle' });
      });
    };

    const existing = flightRef.current;
    if (existing?.key === key) {
      // React Strict Mode replays effects. Reattach to the same logical
      // request before the cleanup microtask can cancel it.
      existing.cancelScheduled = false;
      return () => scheduleCancellation(existing);
    }
    if (existing) {
      existing.cancelled = true;
      existing.abortController.abort();
      flightRef.current = null;
    }

    let request: DeltrelAiRequest;
    try {
      request = buildAiRequest(config, log);
    } catch (error) {
      const aiError = asDeltrelAiError(error);
      queueMicrotask(() => {
        setAiStatus({
          kind: 'error',
          controller,
          code: aiError.code,
          message: aiError.message,
          retryable: aiError.retryable,
        });
      });
      return;
    }

    const flight: AiFlight = {
      key,
      request,
      controller,
      abortController: new AbortController(),
      cancelScheduled: false,
      cancelled: false,
      settled: false,
    };
    flightRef.current = flight;
    const positionConfig = useAppStore.getState().config;
    queueMicrotask(() => {
      if (flightRef.current === flight && !flight.cancelled) {
        setAiStatus({
          kind: 'thinking', controller, positionKey: aiPositionKey,
          searchProgress: { completedSimulations: 0, totalSimulations: selectedSearch.simulations },
        });
      }
    });

    const options = {
      signal: flight.abortController.signal,
      search: selectedSearch,
      onSearchProgress: (progress: DeltrelAiSearchProgress) => {
        if (flight.cancelled || flight.settled || flightRef.current !== flight) return;
        const current = useAppStore.getState();
        if (current.phase !== 'playing' || current.aiPaused || current.earlyOutcome ||
            current.config !== positionConfig || current.log !== log ||
            normalizeControllers(current.config, current.controllers)[request.state.toMove] !== controller) return;
        setAiStatus({
          kind: 'thinking', controller, positionKey: aiPositionKey,
          searchProgress: { ...progress },
        });
      },
    };
    const response = controller === 'server'
      ? import('@/lib/deltrel/ai/server-client').then(({ requestServerAiDecision }) => requestServerAiDecision(request, options))
      : import('@/lib/deltrel/ai/local-client').then(({ requestLocalAiDecision }) => requestLocalAiDecision(request, options));

    void response
      .then((decision) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        const current = useAppStore.getState();
        if (current.phase !== 'playing' || current.aiPaused) {
          flight.settled = true;
          setAiStatus({ kind: 'idle' });
          return;
        }
        const accepted = acceptAiResponse(
          request,
          decision.response,
          current.config,
          current.log,
        );
        if (!accepted.ok) {
          throw new DeltrelAiError(accepted.code, accepted.message, accepted.code === 'stale');
        }
        const currentGame = replay(current.config, current.log);
        if (
          currentGame.over ||
          current.earlyOutcome ||
          normalizeControllers(current.config, current.controllers)[currentGame.toMove] !== flight.controller
        ) {
          flight.settled = true;
          return;
        }
        flight.settled = true;
        setAiStatus({ kind: 'idle' });
        current.act(accepted.action);
        if (canShowAiInsights(current.controllers, current.aiInsightsHidden)) {
          setAnalysisHistory((history) => recordEngineAnalysis(history, {
            analysis: decision.analysis,
            source: controller,
            ply: log.length,
            configKey: analysisConfigKey(config),
            prefix: log,
            action: accepted.action,
            applied: true,
          }));
        }
      })
      .catch((error) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        flight.settled = true;
        const aiError = asDeltrelAiError(error);
        failedRequestKey.current = key;
        if (flight.controller === 'server' && ['network', 'timeout', 'unavailable'].includes(aiError.code)) markCloudUnavailable();
        setAiStatus({
          kind: 'error',
          controller: flight.controller,
          code: aiError.code,
          message: aiError.message,
          // Deliberate cancellations already returned above. An interrupted
          // active request must offer recovery instead of waiting forever.
          retryable: aiError.retryable || aiError.code === 'cancelled' || aiError.code === 'stale',
        });
      })
      .finally(() => {
        if (flightRef.current === flight) flightRef.current = null;
      });

    return () => scheduleCancellation(flight);
  }, [
    aiPaused,
    aiPositionKey,
    autoplayReady,
    autoplaySimulations,
    autoplayMaxConsidered,
    config,
    controllers,
    game,
    log,
    retryNonce,
    markCloudUnavailable,
    uiBlocksPlay,
  ]);

  const disposeLocalAi = useCallback(() => {
    void import('@/lib/deltrel/ai/local-client').then(({ disposeLocalAiClient }) => {
      disposeLocalAiClient();
    });
  }, []);

  const leaveGame = useCallback(() => {
    cancelActiveAi();
    setAnalysisHistory([]);
    setProofMode(false);
    setViewPly(null);
    setEndConfirmOpen(false);
    setResignConfirmOpen(false);
    if (controllers.includes('local')) disposeLocalAi();
    toSetup();
  }, [cancelActiveAi, controllers, disposeLocalAi, toSetup]);

  const undoAction = useCallback(() => {
    cancelActiveAi();
    setViewPly(null);
    undo();
  }, [cancelActiveAi, undo]);

  const redoAction = useCallback(() => {
    cancelActiveAi();
    setViewPly(null);
    redo();
  }, [cancelActiveAi, redo]);

  const rewindToViewed = useCallback(() => {
    if (historyView === null) return;
    cancelActiveAi();
    rewindTo(historyView);
    setViewPly(null);
  }, [cancelActiveAi, historyView, rewindTo]);

  const rematchAction = useCallback(() => {
    cancelActiveAi();
    setAnalysisHistory([]);
    setRulesOpen(false);
    setProofMode(false);
    setViewPly(null);
    setEndConfirmOpen(false);
    setResignConfirmOpen(false);
    rematch();
  }, [cancelActiveAi, rematch]);

  const takeOverAsHuman = useCallback(
    (player: 0 | 1, controller: ControllerType) => {
      cancelActiveAi();
      if (controller === 'local') disposeLocalAi();
      setAiStatus({ kind: 'idle' });
      setPlayerController(player, 'human');
    },
    [cancelActiveAi, disposeLocalAi, setPlayerController],
  );

  const resumeAiAction = useCallback(() => {
    cancelInspection();
    setAiStatus({ kind: 'idle' });
    setRetryNonce((value) => value + 1);
    checkCapabilitiesAgain();
    resumeAi();
  }, [cancelInspection, checkCapabilitiesAgain, resumeAi]);

  const pauseAiAction = useCallback(() => {
    cancelActiveAi();
    setAiStatus({ kind: 'idle' });
    pauseAi();
  }, [cancelActiveAi, pauseAi]);

  const retryWithLessEffort = useCallback(
    (controller: Exclude<ControllerType, 'human'>) => {
      const reduced = reducedSearchBudget(
        useAppStore.getState().aiSearchSettings[controller],
      );
      if (!reduced) return;
      setAiStatus({ kind: 'idle' });
      setAiSearchBudget(controller, reduced);
      setRetryNonce((value) => value + 1);
      if (controller === 'server') checkCapabilitiesAgain();
    },
    [checkCapabilitiesAgain, setAiSearchBudget],
  );

  const currentController = game ? controllers[game.toMove] : 'human';
  const thinking =
    Boolean(game) &&
    currentController !== 'human' &&
    aiStatus.kind === 'thinking' &&
    aiStatus.controller === currentController &&
    aiStatus.positionKey === aiPositionKey;
  const activeAiError =
    currentController !== 'human' &&
    aiStatus.kind === 'error' &&
    aiStatus.controller === currentController
      ? aiStatus
      : null;
  const lowerBudget =
    currentController === 'human'
      ? null
      : reducedSearchBudget(aiSearchSettings[currentController]);
  const canUseLessEffort =
    activeAiError?.retryable === true &&
    activeAiError.code === 'timeout' &&
    lowerBudget !== null;
  const humanCanAct =
    Boolean(game) &&
    currentController === 'human' &&
    !thinking &&
    !uiBlocksPlay &&
    !viewingHistory;
  const applyHumanAction = useCallback(
    (action: GameAction) => {
      if (!humanCanAct) return;
      const current = useAppStore.getState();
      if (current.phase !== 'playing' || current.earlyOutcome || current.config !== storedConfig) return;
      // Several clicks can arrive before React renders the next turn. Always
      // check the latest position so a human cannot place the computer's stone.
      const latestGame = replay(current.config, current.log);
      if (latestGame.over || current.controllers[latestGame.toMove] !== 'human') return;
      cancelInspection();
      current.act(action);
    },
    [cancelInspection, humanCanAct, storedConfig],
  );
  const placeStone = useCallback((node: number) => applyHumanAction({ type: 'place', node }), [applyHumanAction]);

  const shownPositionHash = useMemo(() => {
    if (!shownGame || typeof BigInt !== 'function') return null;
    return semanticStateHash(semanticStateFromGame(shownGame));
  }, [shownGame]);
  const inspectedAnalysis = useMemo(() => selectEngineAnalysis(analysisHistory, {
    config, log, positionHash: shownPositionHash, ply: currentPly,
    allowPrevious: !viewingHistory,
  }), [analysisHistory, config, currentPly, log, shownPositionHash, viewingHistory]);
  const inspectionRuntime = controllers.includes('server') && !controllers.includes('local') ? 'server' : 'local';
  const inspectionRuntimeReady = runtimeCapabilities[inspectionRuntime].status === 'available' && (inspectionRuntime === 'server' || browserAuthorized);
  const inspectionSearch = inspectionRuntime === 'server' ? aiSearchSettings.server : browserSearch;
  // Cancelling a request resets the shared browser worker, so inspection
  // cannot overlap an automatic move.
  const localInspectionNeedsPause = currentController !== 'human' && !aiPaused && !uiBlocksPlay;

  const analyzePosition = useCallback(() => {
    const current = useAppStore.getState();
    if (!canShowAiInsights(current.controllers, current.aiInsightsHidden)) return;
    if (!shownGame || shownGame.over || validProofMode || localInspectionNeedsPause || !inspectionRuntimeReady ||
        typeof BigInt !== 'function') return;
    // A separate, read-only request never applies the returned suggested move.
    if (!viewingHistory && currentController !== 'human' && !aiPaused && !effectiveOver) return;
    cancelInspection();
    const prefix = log.slice(0, currentPly);
    let request: DeltrelAiRequest;
    try { request = buildAiRequest(config, prefix); }
    catch (error) {
      setInspectionStatus({ kind: 'error', stateHash: shownPositionHash ?? '', message: asDeltrelAiError(error).message });
      return;
    }
    const abortController = new AbortController();
    inspectionRef.current = abortController;
    setInspectionStatus({ kind: 'thinking', stateHash: request.stateHash });
    const options = {
      signal: abortController.signal,
      search: inspectionSearch,
    };
    const pending = inspectionRuntime === 'server'
      ? import('@/lib/deltrel/ai/server-client').then(({ requestServerAiDecision }) => requestServerAiDecision(request, options))
      : import('@/lib/deltrel/ai/local-client').then(({ requestLocalAiDecision }) => requestLocalAiDecision(request, options));
    void pending.then((decision) => {
      if (inspectionRef.current !== abortController || abortController.signal.aborted) return;
      const current = useAppStore.getState();
      if (!canShowAiInsights(current.controllers, current.aiInsightsHidden)) return;
      const accepted = acceptAiResponse(request, decision.response, config, prefix);
      if (!accepted.ok || decision.analysis.stateHash !== request.stateHash ||
          decision.analysis.perspective !== request.state.toMove) {
        throw new DeltrelAiError('protocol', 'Engine analysis does not match the requested position.');
      }
      const entry: StoredEngineAnalysis = {
        analysis: decision.analysis, source: inspectionRuntime, ply: prefix.length,
        configKey: analysisConfigKey(config), prefix, action: accepted.action,
        applied: false,
      };
      if (current.phase === 'playing' && selectEngineAnalysis([entry], {
        config: current.config, log: current.log, positionHash: request.stateHash,
        ply: prefix.length, allowPrevious: false,
      })) setAnalysisHistory((history) => recordEngineAnalysis(history, entry));
      setInspectionStatus({ kind: 'idle' });
    }).catch((error) => {
      if (inspectionRef.current !== abortController || abortController.signal.aborted) return;
      setInspectionStatus({ kind: 'error', stateHash: request.stateHash, message: asDeltrelAiError(error).message });
    }).finally(() => {
      if (inspectionRef.current === abortController) inspectionRef.current = null;
    });
  }, [aiPaused, inspectionSearch, inspectionRuntime, cancelInspection, config, currentController, currentPly,
    effectiveOver, inspectionRuntimeReady, localInspectionNeedsPause, log, shownGame, shownPositionHash, validProofMode, viewingHistory]);

  // Arrow keys step through the move history whenever no dialog needs them.
  // Board-focused arrow presses call preventDefault first and are skipped.
  const historyNavEnabled =
    !rulesOpen &&
    !clinchPending &&
    !validProofMode &&
    !validEndConfirmOpen &&
    !validResignConfirmOpen &&
    !(effectiveOver && !reviewing);

  useEffect(() => {
    if (!historyNavEnabled) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) {
        return;
      }
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest('dialog, input, textarea, select, [contenteditable="true"]')
      ) {
        return;
      }
      if (event.key === 'ArrowLeft') {
        if (currentPly > 0) {
          event.preventDefault();
          seek(currentPly - 1);
        }
      } else if (viewingHistory) {
        event.preventDefault();
        seek(currentPly + 1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [currentPly, historyNavEnabled, seek, viewingHistory]);

  const proofWinner =
    earlyOutcome?.reason === 'resignation'
      ? null
      : earlyOutcome?.reason === 'clinch'
        ? earlyOutcome.winner
        : guaranteedWinner;
  const proofLoser =
    proofWinner === null ? null : ((1 - proofWinner) as 0 | 1);
  const proofScenario =
    proofLoser === null ? null : (completionBounds?.scenarios[proofLoser] ?? null);
  const proofActive = validProofMode && proofScenario !== null;
  const proofMask = useMemo(() => {
    if (!game || !proofActive) return null;
    return Uint8Array.from(
      { length: game.board.n },
      (_, node) => (game.stones[node] === EMPTY ? 1 : 0),
    );
  }, [game, proofActive]);
  const humanPlayers = ([0, 1] as const).filter(
    (player) => controllers[player] === 'human',
  );
  const resigningPlayer: 0 | 1 | null =
    humanPlayers.length === 1
      ? humanPlayers[0]
      : humanPlayers.length === 2 && game
        ? game.toMove
        : null;

  const showProofBoard = useCallback(() => {
    if (proofScenario === null) return;
    cancelActiveAi();
    setRulesOpen(false);
    setEndConfirmOpen(false);
    setViewPly(null);
    setProofMode(true);
  }, [cancelActiveAi, proofScenario]);

  const continueAfterClinch = useCallback(() => {
    if (proofWinner === null) return;
    cancelActiveAi();
    acknowledgeClinch(proofWinner);
    setRulesOpen(false);
    setProofMode(false);
    setEndConfirmOpen(false);
  }, [acknowledgeClinch, cancelActiveAi, proofWinner]);

  const endClinchNow = useCallback(() => {
    if (proofWinner === null) return;
    cancelActiveAi();
    setRulesOpen(false);
    setProofMode(false);
    setEndConfirmOpen(false);
    endClinchedGame(proofWinner);
  }, [cancelActiveAi, endClinchedGame, proofWinner]);

  const confirmResignation = useCallback(() => {
    if (resigningPlayer === null) return;
    cancelActiveAi();
    setResignConfirmOpen(false);
    resign(resigningPlayer);
  }, [cancelActiveAi, resign, resigningPlayer]);

  if (!game || !score || !shownGame || !shownScore || !timeline) return null;

  const { board } = game;
  const displayedScore = proofActive ? proofScenario.score : shownScore;
  const showTerritory = shownGame.over || showInfluence || proofActive;
  const gameResult: GameResult | null = game.over
    ? {
        reason: 'full-board',
        winner: validateTerminalWinner(game.board, game.stones).winner,
        score,
      }
    : earlyOutcome;
  const statusPlayer = viewingHistory
    ? shownGame.toMove
    : (gameResult?.winner ??
      ((clinchPending || proofActive) && proofWinner !== null
        ? proofWinner
        : game.toMove));
  const activeColor = PLAYER_COLORS[statusPlayer];
  const gameStatusState: GameStatusState = viewingHistory
    ? 'review'
    : effectiveOver
      ? 'over'
      : proofActive
        ? 'proof'
        : clinchPending
          ? 'clinch'
          : validEndConfirmOpen || validResignConfirmOpen
            ? 'confirming'
            : currentController === 'human'
              ? 'human'
              : aiPaused
                ? 'paused'
                : activeAiError
                  ? 'error'
                  : !autoplayReady
                    ? 'waiting'
                  : 'thinking';
  const aiWaitingReason = currentController === 'server'
    ? runtimeCapabilities.server.status === 'unavailable' ? 'Cloud AI is unavailable. Your game is saved.'
      : runtimeCapabilities.server.status === 'available' ? 'Choose a supported cloud search setting.' : 'Checking cloud availability.'
    : runtimeCapabilities.local.status === 'checking'
    ? 'Checking AI availability.'
    : runtimeCapabilities.local.status === 'unavailable'
      ? runtimeCapabilities.local.reason
      : browserAiIsPreparing(browserStatus)
        ? 'Preparing AI before play can begin.'
        : 'Prepare AI to begin this turn.';
  const shownTurnCapacity = shownGame.over
    ? Math.max(shownGame.currentTurnMoves.length, 1)
    : shownGame.currentTurnMoves.length + shownGame.movesLeft;
  const turnProgress = {
    placed: shownGame.currentTurnMoves.length,
    total: shownTurnCapacity,
  };
  const canRewind =
    earlyOutcome === null && !proofActive && !clinchPending;
  const currentControllerName =
    currentController === 'human'
      ? 'Human'
      : controllerLabel(currentController);
  const proofDescription =
    proofActive && proofScenario && proofWinner !== null && proofLoser !== null
      ? `Proof scenario—not actual moves. Every remaining open node is hypothetically assigned to ${config.playerNames[proofLoser]}; ${config.playerNames[proofWinner]} still wins ${proofScenario.score.players[proofWinner].total} to ${proofScenario.score.players[proofLoser].total}.`
      : null;
  const scoreView = proofActive && proofLoser !== null
    ? ({ kind: 'proof', fillPlayer: proofLoser } as const)
    : viewingHistory
      ? ({ kind: 'review', ply: currentPly, total: log.length } as const)
      : earlyOutcome
        ? ({ kind: 'ended' } as const)
        : ({ kind: 'live' } as const);

  const analysisSelection = !aiInsightsVisible || proofActive ? null : inspectedAnalysis;
  const inspectionThinking = inspectionStatus.kind === 'thinking' &&
    inspectionStatus.stateHash === shownPositionHash;
  const inspectionError = inspectionStatus.kind === 'error' &&
    inspectionStatus.stateHash === shownPositionHash ? inspectionStatus.message : null;
  const estimateThinking = inspectionThinking || (!viewingHistory && thinking && !aiPaused && !uiBlocksPlay);
  const canAnalyze = aiInsightsVisible && !shownGame.over && !proofActive && !inspectionThinking && !localInspectionNeedsPause && inspectionRuntimeReady &&
    typeof BigInt === 'function' &&
    (viewingHistory || currentController === 'human' || aiPaused || effectiveOver);
  const estimateMessage = proofActive
    ? 'This proof board is hypothetical. Return to the game to inspect engine forecasts.'
    : !browserAuthorized
      ? 'Prepare AI to analyze this position on your device.'
    : localInspectionNeedsPause && viewingHistory
      ? 'Pause AI to analyze this position with the browser engine.'
    : inspectionThinking
      ? 'Analyzing this position without playing a move. Results appear after the search completes.'
      : estimateThinking
        ? analysisSelection
          ? 'The next position is being searched. The last completed forecast stays visible.'
          : 'The engine is searching. Predictions appear when this search completes.'
        : inspectionError ?? (!viewingHistory ? activeAiError?.message : null)
      ?? (analysisSelection?.isExact
        ? undefined
        : viewingHistory
          ? 'No engine estimate is recorded for this position. Analyze it without changing the game.'
          : analysisSelection
            ? 'Showing the last completed estimate, from before the indicated move.'
            : 'Analyze this position, or wait for the first AI search to finish.');

  return (
    <main
      className={`${styles.screen} relative z-10 mx-auto flex w-full max-w-[100rem] flex-col`}
    >
      <h1 className="sr-only">Deltrel game</h1>
      <header className="mb-3 flex shrink-0 items-center justify-between gap-3">
        <button
          type="button"
          onClick={leaveGame}
          aria-label="Return to setup"
          className="group flex min-h-11 min-w-0 items-center gap-3 text-left"
        >
          <DeltrelMark className="h-9 w-9 shrink-0 text-sand" />
          <span className="hidden font-display text-3xl font-semibold leading-none tracking-tight text-sand-strong sm:inline">
            Deltrel
          </span>
          <span className="hidden truncate text-xs text-muted sm:block">
            {config.mode === 'double' ? 'Double Deltrel' : 'Classic'} ·{' '}
            {config.pieRule ? 'Pie' : configHandicap(config) > 1 ? `${configHandicap(config)}-stone handicap` : 'Standard'} ·{' '}
            {config.rings} rings
          </span>
        </button>
        <nav className="flex shrink-0 items-center gap-1.5 sm:gap-2" aria-label="Game">
          <GameLibraryButton onOpen={() => { pauseAi(); cancelActiveAi(); }} />
          <GameLibraryButton share onOpen={() => { pauseAi(); cancelActiveAi(); }} />
          {effectiveOver && reviewing && (
            <button
              type="button"
              onClick={() => setReviewing(false)}
              aria-label="Result"
              className={`min-h-11 items-center gap-2 rounded-xl border border-sand/60 bg-sand-faint px-3 text-sm text-sand-strong transition-colors hover:bg-sand/25 ${
                earlyOutcome?.reason === 'clinch'
                  ? 'hidden lg:flex'
                  : 'flex'
              }`}
            >
              <Trophy className="h-4 w-4" aria-hidden />
              <span className="hidden sm:inline">Result</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setRulesOpen(true)}
            aria-label="Rules"
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-3 text-sm text-ink transition-colors hover:border-sand/50"
          >
            <BookOpenText className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">Rules</span>
          </button>
          <button
            type="button"
            onClick={leaveGame}
            aria-label="New game"
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-3 text-sm text-ink transition-colors hover:border-sand/50"
          >
            <Settings2 className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">New game</span>
          </button>
        </nav>
      </header>

      <div className={`${styles.workspace} w-full`}>
        <BoardStage
          className={styles.boardArea}
          board={board}
          stones={proofActive ? proofScenario.stones : shownGame.stones}
          nodeOwner={displayedScore.nodeOwner}
          aliveStone={displayedScore.aliveStone}
          provablyDeadStone={
            proofActive || viewingHistory
              ? null
              : completionBounds?.provablyDeadStone
          }
          syntheticStone={proofMask}
          showTerritory={showTerritory}
          lastMove={proofActive ? -1 : shownGame.lastMove}
          currentTurnMoves={proofActive ? [] : shownGame.currentTurnMoves}
          lastTurnMoves={proofActive ? [] : lastTurnMoves}
          currentTurnCapacity={shownTurnCapacity}
          toMove={shownGame.toMove}
          interactive={!effectiveOver && humanCanAct}
          playerNames={config.playerNames}
          onPlace={placeStone}
          filledCount={proofActive ? board.n : shownGame.stonesPlaced}
          review={
            viewingHistory
              ? {
                  ply: currentPly,
                  total: log.length,
                  onExit: () => seek(null),
                }
              : null
          }
          focusRef={boardStageRef}
          proof={
            proofDescription && proofScenario && proofLoser !== null
              ? {
                  label: 'Clinch proof',
                  detail: `${completionBounds?.emptyNodes ?? 0} hypothetical ${
                    config.playerNames[proofLoser]
                  } stone${
                    (completionBounds?.emptyNodes ?? 0) === 1 ? '' : 's'
                  }`,
                  description: proofDescription,
                }
              : null
          }
        />

        {/* Side panel */}
        <div className={styles.sidePanel}>
          <GameStatus
            className={styles.status}
            state={gameStatusState}
            playerName={config.playerNames[statusPlayer]}
            controllerName={currentControllerName}
            matchLabel={aiMatchLabel(controllers)}
            waitingReason={aiWaitingReason}
            searchProgress={thinking && aiStatus.kind === 'thinking'
              ? aiStatus.searchProgress
              : { completedSimulations: 0, totalSimulations: browserSearch.simulations }}
            mode={config.mode}
            movesLeft={shownGame.movesLeft}
            turnProgress={turnProgress}
            review={
              viewingHistory ? { ply: currentPly, total: log.length } : null
            }
            color={activeColor}
          />
          <div className={`${styles.rail} thin-scroll flex min-w-0 flex-col gap-3`}>
            {!effectiveOver &&
              currentController !== 'human' &&
              (aiPaused || activeAiError || (currentController === 'server' && runtimeCapabilities.server.status === 'unavailable')) && (
                <section
                  aria-live="polite"
                  className="rounded-2xl border border-danger/35 bg-danger/[0.06] px-4 py-3 text-xs text-muted"
                >
                  <div className="flex items-start gap-3">
                    <CircleAlert
                      className="mt-0.5 h-4 w-4 shrink-0 text-danger"
                      aria-hidden
                    />
                    <div className="min-w-0 flex-1">
                      {aiPaused ? (
                        <p>{aiInsightsVisible
                          ? 'AI play is paused. Inspect the forecasts, then resume when ready.'
                          : 'AI play is paused. Resume when ready.'}</p>
                      ) : activeAiError ? (
                        <p role="alert">
                          {activeAiError.message}{' '}
                          <span className="font-mono text-xs opacity-70">
                            ({activeAiError.code})
                          </span>
                        </p>
                      ) : currentController === 'server' ? <p role="alert">Cloud AI is unavailable. Your game is saved. Retry later or switch to AI on this device.</p> : null}
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2 pl-7">
                    {aiPaused && (
                      <button
                        type="button"
                        onClick={resumeAiAction}
                        className="min-h-9 rounded-lg border border-sand/50 px-3 py-1 text-sand-strong transition-colors hover:bg-sand/15"
                      >
                        Resume AI
                      </button>
                    )}
                    {(activeAiError?.retryable || (currentController === 'server' && runtimeCapabilities.server.status === 'unavailable')) && (
                      <button
                        type="button"
                        onClick={() => {
                          setAiStatus({ kind: 'idle' });
                          setRetryNonce((value) => value + 1);
                          if (currentController === 'server') checkCapabilitiesAgain();
                        }}
                        className="min-h-9 rounded-lg border border-sand/50 px-3 py-1 text-sand-strong transition-colors hover:bg-sand/15"
                      >
                        Retry
                      </button>
                    )}
                    {currentController === 'server' && <button
                      type="button"
                      disabled={runtimeCapabilities.local.status !== 'available'}
                      title={runtimeCapabilities.local.status === 'unavailable' ? runtimeCapabilities.local.reason : undefined}
                      onClick={() => {
                        cancelActiveAi();
                        setPlayerController(game.toMove, 'local');
                        resumeAi();
                      }}
                      className="min-h-9 rounded-lg border border-sand/50 px-3 py-1 text-sand-strong transition-colors hover:bg-sand/15"
                    >Switch to AI on this device</button>}
                    {canUseLessEffort && (
                      <button
                        type="button"
                        onClick={() => retryWithLessEffort(currentController)}
                        className="min-h-9 rounded-lg border border-sand/50 px-3 py-1 text-sand-strong transition-colors hover:bg-sand/15"
                      >
                        Use Standard
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => takeOverAsHuman(game.toMove, currentController)}
                      className="min-h-9 rounded-lg border border-white/20 px-3 py-1 text-ink transition-colors hover:border-sand/40"
                    >
                      Take over as human
                    </button>
                  </div>
                </section>
              )}

          {/* Pie rule offer */}
          {game.canSwap && (
            <div className="rounded-2xl border border-dashed border-sand/50 bg-sand-faint px-4 py-3 text-sm">
              <p className="text-ink">
                Pie rule — {config.playerNames[1]} may steal the opening stone.
              </p>
              <button
                type="button"
                disabled={!humanCanAct}
                onClick={() => applyHumanAction({ type: 'swap' })}
                className="mt-2 flex min-h-10 items-center gap-2 rounded-lg border border-sand/60 px-3 py-1.5 text-xs font-medium text-sand-strong transition-colors hover:bg-sand/20"
              >
                <Replace className="h-3.5 w-3.5" aria-hidden /> Steal it (swap sides)
              </button>
            </div>
          )}

          {proofScenario &&
            proofWinner !== null &&
            proofLoser !== null &&
            !game.over &&
            earlyOutcome?.reason !== 'resignation' && (
              <section
                aria-labelledby="clinch-status-heading"
                data-clinch-banner
                className="rounded-2xl border px-4 py-3.5"
                style={{
                  borderColor: `${PLAYER_COLORS[proofWinner].base}66`,
                  background: `linear-gradient(145deg, ${PLAYER_COLORS[proofWinner].soft}, rgba(16,21,42,0.9))`,
                }}
              >
                <div className="flex items-start gap-3">
                  <span
                    aria-hidden
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border"
                    style={{
                      borderColor: `${PLAYER_COLORS[proofWinner].base}66`,
                      color: PLAYER_COLORS[proofWinner].bright,
                    }}
                  >
                    <ShieldCheck className="h-4.5 w-4.5" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h2
                        id="clinch-status-heading"
                        className="text-sm font-semibold"
                        style={{ color: PLAYER_COLORS[proofWinner].bright }}
                      >
                        {config.playerNames[proofWinner]} has clinched
                      </h2>
                      <span className="rounded-full border border-sand/30 bg-sand-faint px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-sand">
                        Result locked
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      Give all {completionBounds?.emptyNodes ?? 0} open node
                      {(completionBounds?.emptyNodes ?? 0) === 1 ? '' : 's'} to{' '}
                      {config.playerNames[proofLoser]} and{' '}
                      {config.playerNames[proofWinner]} still wins.
                    </p>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/15 px-3 py-2 text-xs">
                  <span className="flex min-w-0 items-center gap-2 text-muted">
                    <span
                      aria-hidden
                      className="h-4 w-4 shrink-0 rounded-full border border-dashed border-white/80 opacity-70"
                      style={{
                        background: `repeating-linear-gradient(45deg, ${PLAYER_COLORS[proofLoser].base}99 0 2px, transparent 2px 4px)`,
                      }}
                    />
                    <span className="truncate">
                      All open → {config.playerNames[proofLoser]}
                    </span>
                  </span>
                  <span className="shrink-0 font-mono tabular-nums text-ink">
                    {proofScenario.score.players[proofWinner].total}–{proofScenario.score.players[proofLoser].total}
                  </span>
                </div>
                {proofActive && (
                  <p className="mt-2 text-center text-xs font-medium text-sand">
                    Proof board active · striped stones are hypothetical
                  </p>
                )}
              </section>
            )}

            {(controllers.includes('local') || (aiInsightsVisible && !controllers.includes('server'))) && (!browserAuthorized || runtimeCapabilities.local.status !== 'available' || browserAiIsPreparing(browserStatus) || browserStatus.phase === 'error') && (
              <BrowserAiPreparation
                status={browserStatus}
                authorized={browserAuthorized}
                capability={runtimeCapabilities.local}
                notice={browserNotice}
                selected={controllers.includes('local')}
                inGame
                onPrepare={() => {
                  void prepareBrowserAi().then((prepared) => {
                    if (prepared && browserAuthorized && currentController === 'local') {
                      setAiStatus({ kind: 'idle' });
                      setRetryNonce((value) => value + 1);
                    }
                  });
                }}
                onCancel={() => {
                  pauseAi();
                  cancelActiveAi();
                  cancelBrowserAi();
                }}
                onCheck={checkCapabilitiesAgain}
              />
            )}
            {controllers.includes('server') && <>
              <CloudAiControl capability={runtimeCapabilities.server} selected onCheck={checkCapabilitiesAgain} />
              <BrowserAiStrengthControl
                budget={aiSearchSettings.server}
                capability={runtimeCapabilities.server}
                onChange={(budget) => setAiSearchBudget('server', budget)}
                runtime="server"
                inGame
              />
              {runtimeCapabilities.server.status === 'available' && runtimeCapabilities.server.search && !budgetFitsCapability(aiSearchSettings.server, runtimeCapabilities.server.search) && <button type="button" onClick={() => { if (runtimeCapabilities.server.status === 'available' && runtimeCapabilities.server.search) setAiSearchBudget('server', { ...runtimeCapabilities.server.search.default }); }}>Use recommended cloud search</button>}
            </>}
            {(!controllers.includes('server') || controllers.includes('local')) && <BrowserAiStrengthControl
              budget={aiSearchSettings.local}
              capability={runtimeCapabilities.local}
              onChange={(budget) => setAiSearchBudget('local', budget)}
              inGame
            />}
            {!aiInsightsVisible && currentController !== 'human' && !aiPaused && !uiBlocksPlay && (
              <button
                type="button"
                onClick={pauseAiAction}
                className="min-h-10 rounded-xl border border-sand/40 px-3 py-2 text-sm text-sand-strong transition-colors hover:bg-sand/10"
              >Pause AI</button>
            )}
            {aiInsightsVisible && <EngineEstimatePanel
              analysis={analysisSelection?.entry.analysis ?? null}
              board={board}
              playerNames={config.playerNames}
              context={{
                analyzedPly: analysisSelection?.entry.ply ?? null,
                displayedPly: currentPly,
                source: analysisSelection?.entry.source ?? 'local',
                status: estimateThinking ? 'thinking'
                  : inspectionError || (!viewingHistory && activeAiError) ? 'error'
                  : analysisSelection || canAnalyze ? 'ready' : 'unavailable',
                message: estimateMessage,
                action: analysisSelection?.entry.action,
                applied: analysisSelection?.entry.applied,
                isExact: analysisSelection?.isExact ?? false,
                proof: proofActive,
                phase: viewingHistory ? 'review' : effectiveOver ? 'ended' : 'live',
              }}
              onAnalyze={analyzePosition}
              canAnalyze={canAnalyze}
              onPause={pauseAiAction}
              canPause={currentController !== 'human' && !aiPaused && !uiBlocksPlay}
            />}

          <ScorePanel
            game={shownGame}
            controllers={controllers}
            score={displayedScore}
            completionBounds={completionBounds}
            view={scoreView}
          />

            <MovesPanel
              timeline={timeline}
              total={log.length}
              currentPly={currentPly}
              playerNames={config.playerNames}
              canRewind={canRewind}
              onSeek={seek}
              onRewind={rewindToViewed}
            />
            <GameSaveStatus />

            <details className="rounded-xl border border-white/10 bg-white/[0.025] px-3">
              <summary className="flex min-h-11 cursor-pointer items-center text-xs font-medium text-muted">
                About the scoring display
              </summary>
              <p className="pb-3 text-xs leading-relaxed text-muted">
                Influence shows the current scoring projection and dims groups that are not yet
                networks. Crosses mark only groups that cannot become a network even if they received
                every open node. A striped proof stone is hypothetical and never changes the
                actual move history.
              </p>
            </details>
          </div>

          {/* Persistent actions */}
          <div
            className={`${styles.actions} panel-surface grid grid-cols-2 gap-2 rounded-2xl p-2`}
            data-action-dock
          >
            <button
              type="button"
              disabled={
                log.length === 0 ||
                earlyOutcome !== null ||
                proofActive ||
                clinchPending
              }
              onClick={undoAction}
              className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors enabled:hover:border-sand/50 disabled:opacity-35"
            >
              <Undo2 className="h-4 w-4" aria-hidden /> Undo
            </button>
            <button
              type="button"
              disabled={
                redoStack.length === 0 ||
                earlyOutcome !== null ||
                proofActive ||
                clinchPending
              }
              onClick={redoAction}
              className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors enabled:hover:border-sand/50 disabled:opacity-35"
            >
              <Redo2 className="h-4 w-4" aria-hidden /> Redo
            </button>

            <label className="col-span-2 flex min-h-11 cursor-pointer items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
              <span className="flex items-center gap-2 text-sm text-ink">
                <Eye className="h-4 w-4 text-muted" aria-hidden /> Show influence
              </span>
              <input
                type="checkbox"
                checked={showTerritory}
                disabled={shownGame.over || proofActive}
                onChange={(e) => setShowInfluence(e.target.checked)}
                className="h-4 w-4 accent-[#e8c48b]"
              />
            </label>

            {proofScenario &&
              proofWinner !== null &&
              proofLoser !== null &&
              !game.over &&
              earlyOutcome?.reason !== 'resignation' && (
                <button
                  type="button"
                  aria-pressed={proofActive}
                  onClick={() => {
                    if (proofActive) setProofMode(false);
                    else showProofBoard();
                  }}
                  className={`col-span-2 min-h-11 items-center justify-between gap-3 rounded-xl border px-3 py-2 text-left text-sm transition-colors ${
                    proofActive
                      ? 'border-sand/60 bg-sand-faint text-sand-strong'
                      : 'border-white/15 bg-white/[0.03] text-ink hover:border-sand/50'
                  } hidden lg:flex`}
                >
                  <span className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4" aria-hidden />
                    {proofActive ? 'Return to live board' : 'Show proof board'}
                  </span>
                  <span className="text-xs text-muted">
                    all open → {config.playerNames[proofLoser]}
                  </span>
                </button>
              )}

            {clinchPending && proofActive ? (
              <>
                <button
                  type="button"
                  onClick={continueAfterClinch}
                  className="hidden min-h-11 items-center justify-center rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors hover:border-sand/50 lg:flex"
                >
                  Continue playing
                </button>
                <button
                  type="button"
                  onClick={endClinchNow}
                  className="hidden min-h-11 items-center justify-center gap-2 rounded-xl border border-sand/60 bg-sand-faint px-3 py-2 text-sm font-medium text-sand-strong transition-colors hover:bg-sand/25 lg:flex"
                >
                  <Trophy className="h-4 w-4" aria-hidden /> End now
                </button>
              </>
            ) : guaranteedWinner !== null && earlyOutcome === null ? (
              <button
                type="button"
                onClick={() => {
                  cancelActiveAi();
                  setEndConfirmOpen(true);
                }}
                className="col-span-2 hidden min-h-11 items-center justify-center gap-2 rounded-xl border border-sand/60 bg-sand-faint px-3 py-2 text-sm font-medium text-sand-strong transition-colors hover:bg-sand/25 lg:flex"
              >
                <Trophy className="h-4 w-4" aria-hidden /> End game
              </button>
            ) : (
              resigningPlayer !== null &&
              !effectiveOver && (
                <button
                  type="button"
                  onClick={() => {
                    cancelActiveAi();
                    setResignConfirmOpen(true);
                  }}
                  className="col-span-2 flex min-h-11 items-center justify-center gap-2 rounded-xl border border-danger/35 px-3 py-2 text-sm text-danger transition-colors hover:border-danger/60 hover:bg-danger/[0.07]"
                >
                  <Flag className="h-4 w-4" aria-hidden /> Resign{' '}
                  {config.playerNames[resigningPlayer]}
                </button>
              )
            )}
          </div>
        </div>
      </div>

      {proofScenario &&
        proofWinner !== null &&
        proofLoser !== null &&
        !game.over &&
        earlyOutcome?.reason !== 'resignation' &&
        (!gameResult || reviewing) &&
        (!clinchPending || proofActive) && (
          <div
            className={`panel-surface fixed z-30 grid gap-2 rounded-2xl p-2 shadow-[0_18px_70px_rgba(0,0,0,0.55)] lg:hidden ${
              clinchPending && proofActive ? 'grid-cols-3' : 'grid-cols-2'
            }`}
            style={{
              left: 'max(0.5rem, env(safe-area-inset-left))',
              right: 'max(0.5rem, env(safe-area-inset-right))',
              bottom: 'max(0.5rem, env(safe-area-inset-bottom))',
            }}
            data-mobile-clinch-controls
            role="group"
            aria-label="Clinched game controls"
          >
            <button
              type="button"
              aria-pressed={proofActive}
              onClick={() => {
                if (proofActive) setProofMode(false);
                else showProofBoard();
              }}
              className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-white/15 px-2 py-2 text-xs text-ink transition-colors hover:border-sand/50"
            >
              <ShieldCheck className="h-4 w-4" aria-hidden />
              {proofActive ? 'Live board' : 'Show proof'}
            </button>
            {clinchPending && proofActive ? (
              <>
                <button
                  type="button"
                  onClick={continueAfterClinch}
                  className="min-h-11 rounded-xl border border-white/15 px-2 py-2 text-xs text-ink transition-colors hover:border-sand/50"
                >
                  Continue
                </button>
                <button
                  type="button"
                  onClick={endClinchNow}
                  className="min-h-11 rounded-xl border border-sand/60 bg-sand-faint px-2 py-2 text-xs font-medium text-sand-strong"
                >
                  End now
                </button>
              </>
            ) : earlyOutcome?.reason === 'clinch' ? (
              <button
                type="button"
                onClick={() => setReviewing(false)}
                className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-sand/60 bg-sand-faint px-2 py-2 text-xs font-medium text-sand-strong"
              >
                <Trophy className="h-4 w-4" aria-hidden /> Result
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  cancelActiveAi();
                  setEndConfirmOpen(true);
                }}
                className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-sand/60 bg-sand-faint px-2 py-2 text-xs font-medium text-sand-strong"
              >
                <Trophy className="h-4 w-4" aria-hidden /> End game
              </button>
            )}
          </div>
        )}

      <RulesDialog
        open={
          rulesOpen &&
          !clinchPending &&
          !validEndConfirmOpen &&
          !validResignConfirmOpen &&
          (!gameResult || reviewing)
        }
        onClose={() => setRulesOpen(false)}
      />
      {proofScenario && proofWinner !== null && proofLoser !== null && (
        <ClinchDialog
          open={clinchPending && !proofActive}
          winner={proofWinner}
          winnerName={config.playerNames[proofWinner]}
          loserName={config.playerNames[proofLoser]}
          emptyNodes={completionBounds?.emptyNodes ?? 0}
          returnFocusRef={boardStageRef}
          proofScores={[
            proofScenario.score.players[0].total,
            proofScenario.score.players[1].total,
          ]}
          onContinue={continueAfterClinch}
          onProof={showProofBoard}
          onEnd={endClinchNow}
        />
      )}
      {proofWinner !== null && (
        <EndGameConfirmDialog
          open={validEndConfirmOpen}
          winnerName={config.playerNames[proofWinner]}
          emptyNodes={completionBounds?.emptyNodes ?? 0}
          onCancel={() => setEndConfirmOpen(false)}
          onConfirm={endClinchNow}
        />
      )}
      {resigningPlayer !== null && (
        <ResignDialog
          open={validResignConfirmOpen}
          loserName={config.playerNames[resigningPlayer]}
          winnerName={config.playerNames[1 - resigningPlayer]}
          onCancel={() => setResignConfirmOpen(false)}
          onConfirm={confirmResignation}
        />
      )}
      {gameResult && (
        <GameOverOverlay
          open={!reviewing}
          game={game}
          result={gameResult}
          returnFocusRef={boardStageRef}
          onReview={() => {
            setRulesOpen(false);
            setReviewing(true);
            if (gameResult.reason === 'clinch') showProofBoard();
            else setProofMode(false);
          }}
          onRematch={rematchAction}
          onSetup={leaveGame}
        />
      )}
    </main>
  );
}
