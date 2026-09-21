'use client';

import {
  memo,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
} from 'react';
import type { Board } from '@/lib/deltrel/board';
import { EMPTY } from '@/lib/deltrel/scoring';
import { PLAYER_COLORS } from './theme';
import {
  BOARD_SCALE as S,
  BOARD_VIEWBOX_HALF,
  BOARD_VIEWBOX_SIZE,
  BEACH_OUTLINE,
  LAGOON_OUTLINE,
  buildChannelRoutes,
  channelPath,
} from './boardGeometry';
import styles from './DeltrelBoard.module.css';

interface InspectionGesture {
  pointerId: number;
  node: number;
  x: number;
  y: number;
  held: boolean;
  cancelled: boolean;
  allowPlacement: boolean;
}

interface GroupPresentation {
  groupOf: Int32Array;
  groupSize: Int32Array;
  connectionPaths: [string, string];
  networkConnectionPaths: [string, string];
  syntheticConnectionPaths: [string, string];
  groupPaths: string[];
}

function buildGroupPresentation(
  board: Board,
  stones: ArrayLike<number>,
  aliveStone?: ArrayLike<number> | null,
  syntheticStone?: ArrayLike<number> | null,
): GroupPresentation {
  const parent = new Int32Array(board.n).fill(-1);
  for (let node = 0; node < board.n; node++) {
    if (stones[node] !== EMPTY) parent[node] = node;
  }

  const find = (node: number): number => {
    let root = node;
    while (parent[root] !== root) {
      parent[root] = parent[parent[root]];
      root = parent[root];
    }
    return root;
  };

  for (let node = 0; node < board.n; node++) {
    const color = stones[node];
    if (color === EMPTY) continue;
    for (let edge = board.adjOff[node]; edge < board.adjOff[node + 1]; edge++) {
      const neighbor = board.adj[edge];
      if (neighbor <= node || stones[neighbor] !== color) continue;
      const nodeRoot = find(node);
      const neighborRoot = find(neighbor);
      if (nodeRoot !== neighborRoot) parent[neighborRoot] = nodeRoot;
    }
  }

  const groupOf = new Int32Array(board.n).fill(-1);
  const groupSize = new Int32Array(board.n);
  for (let node = 0; node < board.n; node++) {
    if (stones[node] === EMPTY) continue;
    const root = find(node);
    groupOf[node] = root;
    groupSize[root]++;
  }

  const connections: [string[], string[]] = [[], []];
  const networkConnections: [string[], string[]] = [[], []];
  const syntheticConnections: [string[], string[]] = [[], []];
  const groupSegments = Array.from({ length: board.n }, () => [] as string[]);
  for (let node = 0; node < board.n; node++) {
    const color = stones[node];
    if (color !== 0 && color !== 1) continue;
    for (let edge = board.adjOff[node]; edge < board.adjOff[node + 1]; edge++) {
      const neighbor = board.adj[edge];
      if (neighbor <= node || stones[neighbor] !== color) continue;
      const segment = channelPath(board, node, neighbor);
      const synthetic =
        syntheticStone?.[node] === 1 || syntheticStone?.[neighbor] === 1;
      if (synthetic) syntheticConnections[color].push(segment);
      else connections[color].push(segment);
      if (
        !synthetic &&
        aliveStone?.[node] === 1 &&
        aliveStone[neighbor] === 1
      ) {
        networkConnections[color].push(segment);
      }
      groupSegments[groupOf[node]].push(segment);
    }
  }

  return {
    groupOf,
    groupSize,
    connectionPaths: [connections[0].join(''), connections[1].join('')],
    networkConnectionPaths: [networkConnections[0].join(''), networkConnections[1].join('')],
    syntheticConnectionPaths: [
      syntheticConnections[0].join(''),
      syntheticConnections[1].join(''),
    ],
    groupPaths: groupSegments.map((segments) => segments.join('')),
  };
}

export interface DeltrelBoardProps {
  board: Board;
  stones: ArrayLike<number>;
  /** Node controller from the scorer, for the territory overlay. */
  nodeOwner?: ArrayLike<number> | null;
  /** 1 = stone currently belongs to a living network. */
  aliveStone?: ArrayLike<number> | null;
  /** 1 = existing stone cannot form a living network in any completion. */
  provablyDeadStone?: ArrayLike<number> | null;
  /** 1 = stone is hypothetical and was added only for a completion proof. */
  syntheticStone?: ArrayLike<number> | null;
  /** Accessible summary for a synthetic proof board. */
  proofDescription?: string;
  showTerritory?: boolean;
  lastMove?: number;
  currentTurnMoves?: number[];
  /** Nodes of the most recent completed turn, in placement order. */
  lastTurnMoves?: number[];
  /** Total placements the in-progress turn allows, for numbering badges. */
  currentTurnCapacity?: number;
  toMove?: 0 | 1;
  interactive?: boolean;
  onPlace?: (node: number) => void;
  onHover?: (node: number) => void;
  playerNames?: readonly [string, string];
  className?: string;
}

type DirectionKey = 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight';

/**
 * Choose the nearest node that lies substantially in the requested visual
 * direction. This makes the board behave like a spatial control rather than
 * exposing implementation-order navigation.
 */
function nodeInDirection(board: Board, from: number, key: DirectionKey): number {
  const direction =
    key === 'ArrowUp'
      ? [0, -1]
      : key === 'ArrowDown'
        ? [0, 1]
        : key === 'ArrowLeft'
          ? [-1, 0]
          : [1, 0];
  let best = from;
  let bestCost = Number.POSITIVE_INFINITY;

  for (let candidate = 0; candidate < board.n; candidate++) {
    if (candidate === from) continue;
    const dx = board.xs[candidate] - board.xs[from];
    const dy = board.ys[candidate] - board.ys[from];
    const distance = Math.hypot(dx, dy);
    const alignment = (dx * direction[0] + dy * direction[1]) / distance;
    if (alignment < 0.35) continue;
    const cost = distance / alignment;
    if (cost < bestCost) {
      best = candidate;
      bestCost = cost;
    }
  }

  return best;
}

export const DeltrelBoard = memo(function DeltrelBoard({
  board,
  stones,
  nodeOwner,
  aliveStone,
  provablyDeadStone,
  syntheticStone,
  proofDescription,
  showTerritory = false,
  lastMove = -1,
  currentTurnMoves = [],
  lastTurnMoves = [],
  currentTurnCapacity = 1,
  toMove = 0,
  interactive = false,
  onPlace,
  onHover,
  playerNames,
  className,
}: DeltrelBoardProps) {
  const [hoverSelection, setHoverSelection] = useState<{ board: Board; node: number } | null>(null);
  const hovered = hoverSelection?.board === board ? hoverSelection.node : -1;
  const [activeNode, setActiveNode] = useState(0);
  const [focusedNode, setFocusedNode] = useState(-1);
  const [coordinateDismissed, setCoordinateDismissed] = useState(false);
  const [touchInspection, setTouchInspection] = useState(false);
  const gesture = useRef<InspectionGesture | null>(null);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blockClickUntil = useRef(0);
  const touchInput = useRef(false);
  const keyboardInput = useRef(true);
  const svgRef = useRef<SVGSVGElement>(null);
  const nodeRefs = useRef(new Map<number, SVGCircleElement>());
  const instructionsId = useId();
  const svgId = `lagoon-${instructionsId.replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const proofPatternId = `${svgId}-proof-hatch`;
  const paint = (name: string) => `url(#${svgId}-${name})`;
  const boardInteractive = interactive && syntheticStone == null;

  useEffect(() => () => {
    if (holdTimer.current !== null) clearTimeout(holdTimer.current);
    holdTimer.current = null;
    gesture.current = null;
  }, [board]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const syncVisibility = () => {
      svg.style.setProperty('--water-motion-state', document.hidden ? 'paused' : 'running');
    };
    const useKeyboard = () => {
      touchInput.current = false;
      keyboardInput.current = true;
    };
    syncVisibility();
    document.addEventListener('visibilitychange', syncVisibility);
    document.addEventListener('keydown', useKeyboard, true);
    return () => {
      document.removeEventListener('visibilitychange', syncVisibility);
      document.removeEventListener('keydown', useKeyboard, true);
    };
  }, []);

  const stoneR = Math.min(board.minEdge * 0.46 * S, 9.5);
  const selectedCandidate = hovered >= 0 ? hovered : boardInteractive ? focusedNode : -1;
  const selectedNode = !coordinateDismissed && selectedCandidate >= 0 && selectedCandidate < board.n ? selectedCandidate : -1;
  const groups = useMemo(
    () => buildGroupPresentation(board, stones, aliveStone, syntheticStone),
    [aliveStone, board, stones, syntheticStone],
  );

  const { meshPath, crossingRoutes, shorePath } = useMemo(() => {
    const routes = buildChannelRoutes(board);
    return {
      meshPath: routes.filter((route) => !route.crossing).map((route) => route.path).join(''),
      crossingRoutes: routes.filter((route) => route.crossing),
      shorePath: routes
        .filter((route) => board.isShore[route.from] && board.isShore[route.to])
        .map((route) => route.path)
        .join(''),
    };
  }, [board]);

  const hover = useCallback((u: number) => {
    if (u >= 0) setCoordinateDismissed(false);
    setHoverSelection(u < 0 ? null : { board, node: u });
    onHover?.(u);
  }, [board, onHover]);

  const nodeAtPointer = useCallback(
    (clientX: number, clientY: number): number => {
      const svg = svgRef.current;
      if (!svg) return -1;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return -1;

      const scale = Math.min(rect.width / BOARD_VIEWBOX_SIZE, rect.height / BOARD_VIEWBOX_SIZE);
      const renderedWidth = BOARD_VIEWBOX_SIZE * scale;
      const renderedHeight = BOARD_VIEWBOX_SIZE * scale;
      const originX = rect.left + (rect.width - renderedWidth) / 2;
      const originY = rect.top + (rect.height - renderedHeight) / 2;
      const x = (clientX - originX) / scale - BOARD_VIEWBOX_HALF;
      const y = (clientY - originY) / scale - BOARD_VIEWBOX_HALF;
      const maxDistance = Math.max(board.minEdge * S * 0.56, stoneR * 1.25);
      let nearest = -1;
      let nearestDistance = maxDistance;

      for (let node = 0; node < board.n; node++) {
        const distance = Math.hypot(board.xs[node] * S - x, board.ys[node] * S - y);
        if (distance <= nearestDistance) {
          nearest = node;
          nearestDistance = distance;
        }
      }
      return nearest;
    },
    [board, stoneR],
  );

  const clearHoldTimer = useCallback(() => {
    if (holdTimer.current !== null) clearTimeout(holdTimer.current);
    holdTimer.current = null;
  }, []);

  const cancelInspection = useCallback(() => {
    clearHoldTimer();
    if (gesture.current) {
      gesture.current.cancelled = true;
      blockClickUntil.current = Date.now() + 800;
    }
    hover(-1);
    setTouchInspection(false);
  }, [clearHoldTimer, hover]);

  const handlePointerDown = useCallback((event: PointerEvent<SVGSVGElement>) => {
    const isTouch = event.pointerType === 'touch' || event.pointerType === 'pen';
    touchInput.current = isTouch;
    keyboardInput.current = false;
    setFocusedNode(-1);
    if (!isTouch) {
      // A fresh mouse press is intentional; touch compatibility clicks have no
      // matching mouse pointerdown and remain suppressed by the capture handler.
      blockClickUntil.current = 0;
      return;
    }
    if (event.isPrimary === false) {
      cancelInspection();
      return;
    }
    clearHoldTimer();
    blockClickUntil.current = 0;
    hover(-1);
    setTouchInspection(false);
    const node = nodeAtPointer(event.clientX, event.clientY);
    const next: InspectionGesture = {
      pointerId: event.pointerId,
      node,
      x: event.clientX,
      y: event.clientY,
      held: false,
      cancelled: false,
      allowPlacement: boardInteractive,
    };
    gesture.current = next;
    // Capture keeps release/cancel handling reliable outside the small node target.
    if (event.nativeEvent.isTrusted) event.currentTarget.setPointerCapture?.(event.pointerId);
    holdTimer.current = setTimeout(() => {
      holdTimer.current = null;
      if (gesture.current !== next || next.cancelled) return;
      next.held = true;
      setTouchInspection(next.node >= 0);
      hover(next.node);
    }, 350);
  }, [boardInteractive, cancelInspection, clearHoldTimer, hover, nodeAtPointer]);

  const handlePointerMove = useCallback(
    (event: PointerEvent<SVGSVGElement>) => {
      if (event.pointerType === 'touch' || event.pointerType === 'pen') {
        const current = gesture.current;
        if (current?.pointerId === event.pointerId &&
            Math.hypot(event.clientX - current.x, event.clientY - current.y) > 10) {
          cancelInspection();
        }
        return;
      }
      touchInput.current = false;
      keyboardInput.current = false;
      setTouchInspection(false);
      setFocusedNode(-1);
      const node = nodeAtPointer(event.clientX, event.clientY);
      if (node !== hovered) hover(node);
    },
    [cancelInspection, hover, hovered, nodeAtPointer],
  );

  const handlePointerUp = useCallback((event: PointerEvent<SVGSVGElement>) => {
    const current = gesture.current;
    if (!current || current.pointerId !== event.pointerId) return;
    touchInput.current = true;
    clearHoldTimer();
    if (current.held || current.cancelled || current.node < 0 || !current.allowPlacement) {
      blockClickUntil.current = Date.now() + 800;
    }
    gesture.current = null;
    hover(-1);
    setTouchInspection(false);
  }, [clearHoldTimer, hover]);

  const handlePointerCancel = useCallback((event: PointerEvent<SVGSVGElement>) => {
    if (gesture.current?.pointerId !== event.pointerId) return;
    touchInput.current = true;
    cancelInspection();
    gesture.current = null;
  }, [cancelInspection]);

  const handleBoardClick = useCallback(
    (event: MouseEvent<SVGSVGElement>) => {
      if (!boardInteractive) return;
      const node = nodeAtPointer(event.clientX, event.clientY);
      if (node >= 0 && stones[node] === EMPTY) onPlace?.(node);
    },
    [boardInteractive, nodeAtPointer, onPlace, stones],
  );

  const territory = showTerritory && nodeOwner ? nodeOwner : null;
  const occupiedCount = Array.from({ length: board.n }, (_, node) => stones[node]).filter(
    (stone) => stone !== EMPTY,
  ).length;
  const currentPlayerName = playerNames?.[toMove] || PLAYER_COLORS[toMove].name;
  const highlightedGroup =
    hovered >= 0 && stones[hovered] !== EMPTY ? groups.groupOf[hovered] : -1;
  const highlightedColor =
    highlightedGroup >= 0 ? (stones[hovered] as 0 | 1) : null;

  const focusNode = (node: number) => {
    setActiveNode(node);
    setFocusedNode(node);
    nodeRefs.current.get(node)?.focus();
    hover(node);
  };

  const handleNodeKeyDown = (
    event: KeyboardEvent<SVGCircleElement>,
    node: number,
    isEmpty: boolean,
  ) => {
    if (event.key === 'Escape') {
      cancelInspection();
      setFocusedNode(node);
      setCoordinateDismissed(true);
      event.preventDefault();
      return;
    }
    touchInput.current = false;
    blockClickUntil.current = 0;
    setCoordinateDismissed(false);
    setFocusedNode(node);
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (isEmpty) onPlace?.(node);
      return;
    }

    let next = node;
    if (
      event.key === 'ArrowUp' ||
      event.key === 'ArrowDown' ||
      event.key === 'ArrowLeft' ||
      event.key === 'ArrowRight'
    ) {
      next = nodeInDirection(board, node, event.key);
    } else if (event.key === 'Home') {
      next = 0;
    } else if (event.key === 'End') {
      next = board.n - 1;
    } else {
      return;
    }

    event.preventDefault();
    focusNode(next);
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`${-BOARD_VIEWBOX_HALF} ${-BOARD_VIEWBOX_HALF} ${BOARD_VIEWBOX_SIZE} ${BOARD_VIEWBOX_SIZE}`}
      className={`${styles.board} ${className ?? ''}`}
      role={boardInteractive ? 'group' : 'img'}
      aria-label={
        proofDescription
          ? 'Clinch proof board'
          : `Deltrel board with ${board.rings} rings, ${occupiedCount} of ${board.n} nodes occupied`
      }
      aria-describedby={instructionsId}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onPointerLeave={(event) => {
        if (event.pointerType === 'touch' || event.pointerType === 'pen') cancelInspection();
        else hover(-1);
      }}
      onMouseLeave={() => { if (!touchInput.current) hover(-1); }}
      onClickCapture={(event) => {
        if (Date.now() < blockClickUntil.current) {
          event.preventDefault();
          event.stopPropagation();
          blockClickUntil.current = 0;
        }
      }}
      onContextMenu={(event) => { if (touchInput.current) event.preventDefault(); }}
      onClick={handleBoardClick}
    >
      <desc id={instructionsId}>
        {proofDescription
          ? `${proofDescription} This proof board is read-only.`
          : boardInteractive
          ? 'Hover or focus a point to see its coordinate. On a touchscreen, press and hold to inspect without placing; a quick tap places a stone. Use arrow keys to move between nodes, Enter or Space to place, and Escape to hide the coordinate. Curved channels connect nodes; crossings are not playable junctions.'
          : 'A read-only game board. Hover or press and hold a point to inspect its coordinate.'}
      </desc>
      <defs>
        <radialGradient id={`${svgId}-water`} cx="32%" cy="20%" r="96%">
          <stop offset="0%" stopColor="#339d98" />
          <stop offset="42%" stopColor="#1b737b" />
          <stop offset="100%" stopColor="#0b414f" />
        </radialGradient>
        <linearGradient id={`${svgId}-sand`} x1="0%" y1="0%" x2="85%" y2="100%">
          <stop offset="0%" stopColor="#fff0d3" />
          <stop offset="32%" stopColor="#f0d6a9" />
          <stop offset="65%" stopColor="#ddbc88" />
          <stop offset="100%" stopColor="#f2dcb2" />
        </linearGradient>
        <linearGradient id={`${svgId}-duneShade`} x1="0%" y1="0%" x2="100%" y2="85%">
          <stop offset="0%" stopColor="#fff6de" stopOpacity=".68" />
          <stop offset="48%" stopColor="#b8945c" stopOpacity=".18" />
          <stop offset="100%" stopColor="#9d8053" stopOpacity=".34" />
        </linearGradient>
        <linearGradient id={`${svgId}-wetSand`} x1="12%" y1="0%" x2="80%" y2="100%">
          <stop offset="0%" stopColor="#ceba8e" />
          <stop offset="50%" stopColor="#b39e73" />
          <stop offset="100%" stopColor="#d1bf98" />
        </linearGradient>
        <pattern id={`${svgId}-sandGrain`} patternUnits="userSpaceOnUse" width="8.3" height="7.1" patternTransform="rotate(13)">
          <path
            d="M.6 1.2h.16M2.8 5.9h.12M4.7.8h.18M7.3 4.4h.13M1.3 4.1h.11M5.5 3.4h.16M3.8 2.8h.12M7.8 6.7h.17M2.1.3h.1M6.3 5.8h.12"
            stroke="#806746"
            strokeOpacity=".28"
            strokeWidth=".14"
            strokeLinecap="round"
          />
          <path d="M1.8 2.4h.3M4.3 6.2h.24M6.8 1.8h.3" stroke="#fff9e6" strokeOpacity=".6" strokeWidth=".18" strokeLinecap="round" />
        </pattern>
        <filter id={`${svgId}-duneSoftness`} x="-15%" y="-15%" width="130%" height="130%">
          <feGaussianBlur stdDeviation="1.35" />
        </filter>
        <radialGradient id={`${svgId}-shallows`} cx="26%" cy="18%" r="88%">
          <stop offset="0%" stopColor="#b4efcd" stopOpacity="0.62" />
          <stop offset="60%" stopColor="#80d5c2" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#37968e" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${svgId}-reflection`} x1="0%" y1="0%" x2="70%" y2="100%">
          <stop offset="0%" stopColor="#c8f8e4" stopOpacity="0.2" />
          <stop offset="55%" stopColor="#c8f8e4" stopOpacity="0" />
        </linearGradient>
        {PLAYER_COLORS.map((color, player) => (
          <radialGradient key={player} id={`${svgId}-stone${player}`} cx="32%" cy="22%" r="90%">
            <stop offset="0%" stopColor={color.bright} />
            <stop offset="42%" stopColor={color.base} />
            <stop offset="82%" stopColor={color.deep} />
            <stop offset="100%" stopColor={color.base} />
          </radialGradient>
        ))}
        <pattern id={proofPatternId} patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(35)">
          <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(255,255,255,0.9)" strokeWidth="0.7" />
        </pattern>
        <filter id={`${svgId}-basinShadow`} x="-15%" y="-15%" width="130%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="5" floodColor="#001b22" floodOpacity="0.48" />
        </filter>
        <filter id={`${svgId}-stoneShadow`} x="-60%" y="-60%" width="220%" height="240%">
          <feDropShadow dx="0" dy="1.3" stdDeviation="1.1" floodColor="#001c25" floodOpacity="0.7" />
        </filter>
        <filter id={`${svgId}-connectionGlow`} x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="0.65" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <clipPath id={`${svgId}-beachClip`}><path d={BEACH_OUTLINE} /></clipPath>
        <clipPath id={`${svgId}-lagoonClip`}><path d={LAGOON_OUTLINE} transform="scale(.968)" /></clipPath>
      </defs>

      {/* Broad sunlit dunes give way to damp sand, foam and turquoise shallows. */}
      <g aria-hidden pointerEvents="none" data-water-decoration>
        <path data-coast-layer="dry-sand" d={BEACH_OUTLINE} fill={paint('sand')} filter={paint('basinShadow')} />
        <g clipPath={paint('beachClip')}>
          <g filter={paint('duneSoftness')}>
            <path
              d="M-124-28C-121-79-80-91-38-102C-65-81-94-68-103-34C-111-8-107 11-104 30C-120 13-128-6-124-28Z"
              fill={paint('duneShade')}
            />
            <path
              d="M-77 98C-39 119-9 102 28 106C60 109 84 86 101 57C97 85 81 107 59 113C19 122-15 109-44 117Z"
              fill={paint('duneShade')}
            />
            <path d="M-7-124C24-123 43-94 73-94C98-94 118-72 119-50" fill="none" stroke="#fff5dc" strokeOpacity=".58" strokeWidth="5" />
            <path d="M-112 24C-97 49-103 73-80 98" fill="none" stroke="#fff4d8" strokeOpacity=".5" strokeWidth="4" />
          </g>
          <g fill="none" strokeLinecap="round">
            <path d="M-113-56C-106-77-85-86-66-90M-110-61C-101-78-86-82-74-85" stroke="#fff7e1" strokeOpacity=".58" strokeWidth=".45" />
            <path d="M36 110C60 110 83 92 92 77M43 112C67 108 81 96 88 84" stroke="#b29465" strokeOpacity=".28" strokeWidth=".4" />
            <path d="M-30-116C-19-130-3-129 11-122" stroke="#b2956d" strokeOpacity=".2" strokeWidth=".35" />
          </g>
          <path data-coast-layer="grain" d={BEACH_OUTLINE} fill={paint('sandGrain')} />
        </g>
        <path data-coast-layer="wet-sand" d={LAGOON_OUTLINE} transform="scale(1.035)" fill={paint('wetSand')} />
        <path d={LAGOON_OUTLINE} transform="scale(1.011)" fill="#cae0c3" stroke="#f5f4dc" strokeOpacity=".7" strokeWidth=".55" />
        <path d={LAGOON_OUTLINE} transform="scale(.997)" fill="#80c6b6" />
        <path data-coast-layer="foam" d={LAGOON_OUTLINE} transform="scale(.987)" fill="none" stroke="#effced" strokeOpacity=".72" strokeWidth="1.4" strokeLinecap="round" strokeDasharray="17 1.2 3 .8 8 1.6" />
        <path d={LAGOON_OUTLINE} transform="scale(.977)" fill="#6dbcae" />
        <path data-coast-layer="lagoon" d={LAGOON_OUTLINE} transform="scale(.965)" fill={paint('water')} />
        <g clipPath={paint('lagoonClip')}>
          <path d={LAGOON_OUTLINE} transform="scale(.959)" fill={paint('shallows')} />
          <g className={styles.depthContours} fill="none" stroke="#bce8d1" strokeWidth=".22">
            {[0.936, 0.902, 0.852, 0.787].map((scale) => (
              <path key={scale} d={LAGOON_OUTLINE} transform={`scale(${scale})`} strokeOpacity={scale > 0.9 ? 0.45 : 0.16} />
            ))}
          </g>
          <g className={styles.surfaceLight} data-water-light>
            <path d="M-118-81C-42-101-20-44 60-62S122-43 134-11L138-85-90-137Z" fill={paint('reflection')} />
            <path d="M-113-30C-58-52-20-1 40-15S92-28 121-1" fill="none" stroke="#b0efd7" strokeWidth=".24" strokeOpacity=".3" />
            <path d="M-100-27C-56-44-15 6 43-11S94-23 117 3" fill="none" stroke="#b0efd7" strokeWidth=".2" strokeOpacity=".18" />
          </g>
        </g>
      </g>

      {/* Every waterway corresponds to one edge of the unchanged game graph. */}
      <g aria-hidden pointerEvents="none" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path d={meshPath} stroke="#063e49" strokeOpacity=".34" strokeWidth="1.55" />
        <path data-channel-layer="mesh" d={meshPath} stroke="#b7e0cf" strokeOpacity=".5" strokeWidth=".42" />
        <path d={shorePath} stroke="#79bfb1" strokeOpacity=".18" strokeWidth="4" />
        <path data-channel-layer="shore" d={shorePath} stroke="#e7d5ae" strokeOpacity=".72" strokeWidth=".58" />
        {crossingRoutes.map((route) => (
          <g key={`${route.from}-${route.to}`} data-channel-crossing={`${route.from}-${route.to}`}>
            <path d={route.path} stroke="#104754" strokeWidth="2.2" />
            <path data-channel-layer="confluence" d={route.path} stroke="#b0e3d1" strokeOpacity=".76" strokeWidth=".55" />
          </g>
        ))}
      </g>

      {/* Same-color graph connections. The brighter pass marks living networks. */}
      <g aria-hidden pointerEvents="none">
        {PLAYER_COLORS.map((color, player) => (
          <path
            key={`connections-${player}`}
            data-connection-layer="group"
            data-player={player}
            d={groups.connectionPaths[player as 0 | 1]}
            stroke={color.deep}
            strokeWidth={Math.max(stoneR * 0.48, 1.2)}
            strokeOpacity="0.72"
            fill="none"
            strokeLinecap="round"
          />
        ))}
        {PLAYER_COLORS.map((color, player) => (
          <path
            key={`deltrel-connections-${player}`}
            data-connection-layer="network"
            data-player={player}
            d={groups.networkConnectionPaths[player as 0 | 1]}
            stroke={color.base}
            strokeWidth={Math.max(stoneR * 0.24, 0.7)}
            strokeOpacity="0.82"
            fill="none"
            strokeLinecap="round"
            filter={paint('connectionGlow')}
          />
        ))}
        {PLAYER_COLORS.map((color, player) => (
          <path
            key={`proof-connections-${player}`}
            data-connection-layer="proof"
            data-player={player}
            d={groups.syntheticConnectionPaths[player as 0 | 1]}
            stroke={color.bright}
            strokeWidth={Math.max(stoneR * 0.24, 0.7)}
            strokeOpacity="0.5"
            strokeDasharray="1.4 1.4"
            fill="none"
            strokeLinecap="round"
          />
        ))}
        {highlightedGroup >= 0 && highlightedColor !== null && (
          <path
            data-connection-layer="highlight"
            d={groups.groupPaths[highlightedGroup]}
            stroke={PLAYER_COLORS[highlightedColor].bright}
            strokeWidth={Math.max(stoneR * 0.68, 1.6)}
            strokeOpacity="0.95"
            fill="none"
            strokeLinecap="round"
            filter={paint('connectionGlow')}
          />
        )}
      </g>

      {/* Node layers */}
      {Array.from({ length: board.n }, (_, u) => {
        const x = board.xs[u] * S;
        const y = board.ys[u] * S;
        const stone = stones[u];
        const isEmpty = stone === EMPTY;
        const owner = territory ? territory[u] : -1;
        const cape = board.isCape[u] === 1;
        const shore = board.isShore[u] === 1;
        const notInNetwork = !isEmpty && aliveStone ? aliveStone[u] === 0 : false;
        const provablyDead = !isEmpty && provablyDeadStone?.[u] === 1;
        const synthetic = !isEmpty && syntheticStone?.[u] === 1;
        const dimmed = provablyDead || (showTerritory && notInNetwork);
        const inHighlightedGroup =
          !isEmpty && highlightedGroup >= 0 && groups.groupOf[u] === highlightedGroup;
        const nodeKind = cape ? 'cape shore' : shore ? 'shore' : 'interior node';
        const currentTurnIdx = isEmpty ? -1 : currentTurnMoves.indexOf(u);
        const lastTurnIdx = isEmpty ? -1 : lastTurnMoves.indexOf(u);
        const moveMarker =
          currentTurnIdx >= 0
            ? `${u === lastMove ? ', last move' : ''}, placed this turn${
                currentTurnCapacity > 1
                  ? ` (stone ${currentTurnIdx + 1} of ${currentTurnCapacity})`
                  : ''
              }`
            : lastTurnIdx >= 0
              ? `${u === lastMove ? ', last move' : ''}, placed last turn${
                  lastTurnMoves.length > 1
                    ? ` (stone ${lastTurnIdx + 1} of ${lastTurnMoves.length})`
                    : ''
                }`
              : u === lastMove
                ? ', last move'
                : '';
        const nodeState = isEmpty
          ? `empty ${nodeKind}; ${currentPlayerName} may place here`
          : `${playerNames?.[stone as 0 | 1] || PLAYER_COLORS[stone as 0 | 1].name} stone on ${nodeKind}${moveMarker}${
              provablyDead
                ? ', provably dead; cannot form a living network in any completion'
                : notInNetwork
                  ? `, connected group of ${groups.groupSize[groups.groupOf[u]]} stone${
                      groups.groupSize[groups.groupOf[u]] === 1 ? '' : 's'
                    }, not currently part of a living network`
                  : `, part of a living network with ${groups.groupSize[groups.groupOf[u]]} stones`
            }`;

        return (
          <g key={u}>
            {/* shore / cape markers */}
            {shore && (
              <g aria-hidden pointerEvents="none">
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR * (cape ? 1.24 : 1.16)}
                  fill="none"
                  stroke={cape ? 'rgba(245,224,189,0.56)' : 'rgba(204,219,188,0.38)'}
                  strokeWidth={cape ? 0.4 : 0.3}
                  strokeDasharray={cape ? '2.4 2.6' : '0.8 1.8'}
                />
                {cape && (
                  <circle
                    cx={x}
                    cy={y}
                    r={stoneR * 1.02}
                    fill="rgba(228,201,161,0.08)"
                    stroke="rgba(245,224,189,0.72)"
                    strokeWidth="0.45"
                  />
                )}
              </g>
            )}

            {/* territory tint */}
            {territory && owner !== -1 && (isEmpty || notInNetwork) && (
              <circle
                cx={x}
                cy={y}
                r={stoneR * 0.55}
                fill={PLAYER_COLORS[owner as 0 | 1].glow}
                opacity={shore ? 0.95 : 0.5}
              />
            )}

            {/* Unoccupied anchor wells remain readable above the water. */}
            {isEmpty && (
              <g aria-hidden pointerEvents="none">
                <circle cx={x} cy={y} r={stoneR * 0.37} fill="#0c3b48" stroke="#89c4b7" strokeOpacity=".42" strokeWidth=".25" />
                <circle cx={x} cy={y} r={stoneR * 0.12} fill="#dae8cb" fillOpacity=".8" />
              </g>
            )}

            {/* hover ghost */}
            {isEmpty && hovered === u && boardInteractive && (
              <circle
                cx={x}
                cy={y}
                r={stoneR}
                fill={paint(`stone${toMove}`)}
                opacity={0.45}
              />
            )}

            {/* stone */}
            {!isEmpty && (
              <g
                className={synthetic ? 'proof-stone' : 'stone-pop'}
                opacity={
                  synthetic
                    ? dimmed
                      ? 0.32
                      : 0.68
                    : provablyDead
                      ? 0.22
                      : dimmed
                        ? 0.35
                        : 1
                }
                data-group-root={groups.groupOf[u]}
                data-stone-node={u}
                data-proof-stone={synthetic ? u : undefined}
              >
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR}
                  fill={paint(`stone${stone}`)}
                  stroke={PLAYER_COLORS[stone as 0 | 1].deep}
                  strokeWidth="0.5"
                  filter={dimmed ? undefined : paint('stoneShadow')}
                />
                {synthetic ? (
                  <>
                    <circle
                      cx={x}
                      cy={y}
                      r={stoneR * 0.86}
                      fill={`url(#${proofPatternId})`}
                      opacity="0.72"
                    />
                    <circle
                      cx={x}
                      cy={y}
                      r={stoneR * 1.08}
                      fill="none"
                      stroke="rgba(255,255,255,0.9)"
                      strokeWidth="0.75"
                      strokeDasharray="1.5 1.35"
                    />
                  </>
                ) : (
                  <>
                  <circle cx={x} cy={y} r={stoneR * .81} fill="none" stroke={PLAYER_COLORS[stone as 0 | 1].bright} strokeOpacity=".32" strokeWidth=".35" />
                  <path d={`M${x - stoneR * .55} ${y + stoneR * .52}Q${x} ${y + stoneR * .87} ${x + stoneR * .53} ${y + stoneR * .45}`} fill="none" stroke={PLAYER_COLORS[stone as 0 | 1].bright} strokeOpacity=".4" strokeWidth=".5" strokeLinecap="round" />
                  <ellipse
                    cx={x - stoneR * 0.3}
                    cy={y - stoneR * 0.38}
                    rx={stoneR * 0.34}
                    ry={stoneR * 0.22}
                    fill="rgba(255,255,255,0.55)"
                    transform={`rotate(-24 ${x - stoneR * 0.3} ${y - stoneR * 0.38})`}
                  />
                  </>
                )}
              </g>
            )}

            {inHighlightedGroup && (
              <circle
                aria-hidden
                data-group-highlight={u}
                cx={x}
                cy={y}
                r={stoneR * 1.18}
                fill="none"
                stroke={PLAYER_COLORS[stone as 0 | 1].bright}
                strokeWidth="0.8"
                strokeOpacity="0.9"
                pointerEvents="none"
              />
            )}

            {provablyDead && (
              <g
                aria-hidden
                data-provably-dead-stone={u}
                pointerEvents="none"
                stroke="rgba(255,125,104,0.95)"
                strokeWidth={Math.max(stoneR * 0.14, 0.8)}
                strokeLinecap="round"
              >
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR * 0.92}
                  fill="none"
                  strokeDasharray="1.6 1.6"
                  strokeWidth={Math.max(stoneR * 0.1, 0.65)}
                />
                <path
                  d={`M${x - stoneR * 0.42} ${y - stoneR * 0.42}L${x + stoneR * 0.42} ${
                    y + stoneR * 0.42
                  }M${x + stoneR * 0.42} ${y - stoneR * 0.42}L${
                    x - stoneR * 0.42
                  } ${y + stoneR * 0.42}`}
                  fill="none"
                />
              </g>
            )}

            {/* recent-move markers: last completed turn (white), in-progress
                turn (sand), pulse on the newest placement */}
            {lastTurnIdx >= 0 && u !== lastMove && (
              <circle
                aria-hidden
                data-last-turn-move={u}
                pointerEvents="none"
                cx={x}
                cy={y}
                r={stoneR * 1.18}
                fill="none"
                stroke="rgba(255,255,255,0.85)"
                strokeWidth="0.7"
              />
            )}
            {currentTurnIdx >= 0 && u !== lastMove && (
              <circle
                aria-hidden
                data-current-turn-move={u}
                pointerEvents="none"
                cx={x}
                cy={y}
                r={stoneR * 1.18}
                fill="none"
                stroke="rgba(247,220,166,0.95)"
                strokeWidth="0.8"
              />
            )}
            {u === lastMove && (
              <g aria-hidden pointerEvents="none">
                <circle
                  data-last-move={u}
                  cx={x}
                  cy={y}
                  r={stoneR * 1.18}
                  fill="none"
                  stroke={
                    currentTurnIdx >= 0
                      ? 'rgba(247,220,166,0.98)'
                      : 'rgba(255,255,255,0.9)'
                  }
                  strokeWidth="0.9"
                />
                <circle
                  className={styles.moveRipple}
                  data-move-ripple={u}
                  cx={x}
                  cy={y}
                  r={stoneR * 1.18}
                  fill="none"
                  stroke={currentTurnIdx >= 0 ? '#e4c9a1' : '#dbf3e6'}
                  strokeWidth=".65"
                />
              </g>
            )}
            {((currentTurnIdx >= 0 && currentTurnCapacity > 1) ||
              (currentTurnIdx < 0 &&
                lastTurnIdx >= 0 &&
                lastTurnMoves.length > 1)) && (
              <g
                aria-hidden
                pointerEvents="none"
                data-move-badge={u}
                data-move-order={
                  (currentTurnIdx >= 0 ? currentTurnIdx : lastTurnIdx) + 1
                }
              >
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR * 0.5}
                  fill="rgba(6,8,18,0.78)"
                  stroke={
                    currentTurnIdx >= 0
                      ? 'rgba(247,220,166,0.65)'
                      : 'rgba(255,255,255,0.4)'
                  }
                  strokeWidth="0.3"
                />
                <text
                  x={x}
                  y={y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={stoneR * 0.72}
                  fontWeight={700}
                  fill={
                    currentTurnIdx >= 0
                      ? 'rgba(247,220,166,0.98)'
                      : 'rgba(255,255,255,0.95)'
                  }
                >
                  {(currentTurnIdx >= 0 ? currentTurnIdx : lastTurnIdx) + 1}
                </text>
              </g>
            )}

            {boardInteractive && focusedNode === u && (
              <circle
                aria-hidden
                data-keyboard-focus={board.labels[u]}
                cx={x}
                cy={y}
                r={Math.max(stoneR * 1.45, 5.5)}
                fill="none"
                stroke="rgba(255,255,255,0.95)"
                strokeWidth="1.1"
                pointerEvents="none"
              />
            )}

            {/* hit target */}
            {boardInteractive && (
              <circle
                ref={(element) => {
                  if (element) nodeRefs.current.set(u, element);
                  else nodeRefs.current.delete(u);
                }}
                cx={x}
                cy={y}
                r={Math.max(stoneR, 4)}
                fill="transparent"
                role="button"
                tabIndex={activeNode === u ? 0 : -1}
                aria-label={`Node ${board.labels[u]}, ${nodeState}`}
                aria-disabled={!isEmpty}
                style={{ cursor: isEmpty ? 'pointer' : 'default', outline: 'none' }}
                onMouseEnter={() => { if (!touchInput.current) hover(u); }}
                onFocus={(event) => {
                  setActiveNode(u);
                  if (!touchInput.current && (keyboardInput.current || event.currentTarget.matches(':focus-visible')) && !gesture.current) {
                    setFocusedNode(u);
                    if (!touchInput.current) hover(u);
                  }
                }}
                onBlur={() => {
                  setFocusedNode((current) => (current === u ? -1 : current));
                  hover(-1);
                }}
                onKeyDown={(event) => handleNodeKeyDown(event, u, isEmpty)}
                onClick={(event) => {
                  event.stopPropagation();
                  if (isEmpty) onPlace?.(u);
                }}
              />
            )}
          </g>
        );
      })}
      {selectedNode >= 0 && (
        <g
          aria-hidden
          pointerEvents="none"
          data-coordinate-tooltip={board.labels[selectedNode]}
          transform={`translate(${board.xs[selectedNode] * S} ${Math.max(-BOARD_VIEWBOX_HALF + 8, board.ys[selectedNode] * S - (touchInspection ? 30 : stoneR + 7.5))})`}
        >
          <rect x="-11.5" y="-5" width="23" height="10" rx="3.3" fill="#092f38" fillOpacity=".97" stroke="#e4c9a1" strokeOpacity=".7" strokeWidth=".4" />
          <text className={styles.coordinateTooltip} textAnchor="middle" dominantBaseline="central" fill="#fff2d8">
            {board.labels[selectedNode]}
          </text>
        </g>
      )}
    </svg>
  );
});
