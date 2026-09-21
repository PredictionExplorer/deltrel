import type { Board } from '@/lib/deltrel/board';

export const BOARD_SCALE = 100;
export const BOARD_VIEWBOX_HALF = 132;
export const BOARD_VIEWBOX_SIZE = BOARD_VIEWBOX_HALF * 2;

/** A broad beach with independent dune contours, rather than a scaled frame. */
export const BEACH_OUTLINE =
  'M-14-126C9-131 24-119 41-108C61-97 87-101 108-81' +
  'C125-65 124-45 117-26C109-4 110 20 103 44' +
  'C98 68 86 92 63 104C40 119 13 112-11 116' +
  'C-37 123-68 119-83 98C-96 83-101 61-107 40' +
  'C-113 17-126-8-124-33C-126-59-112-81-91-92' +
  'C-68-103-44-100-29-117C-24-123-21-126-14-126Z';

/** An organic lagoon contour, independent of the playable graph. */
export const LAGOON_OUTLINE =
  'M-5-111C18-115 28-100 46-91C66-84 99-80 108-56' +
  'C116-36 104-15 96 5C88 27 94 56 79 78' +
  'C64 97 31 94 7 97C-20 105-53 106-70 87' +
  'C-90 66-89 35-99 10C-110-14-117-47-98-68' +
  'C-78-89-48-84-29-99C-19-108-15-110-5-111Z';

export interface ChannelRoute {
  from: number;
  to: number;
  path: string;
  crossing: boolean;
}

const CENTRAL_LANES: Record<string, readonly [number, number]> = {
  '0:2': [-0.46, -0.12],
  '0:3': [-0.2, 0.36],
  '1:3': [0.35, -0.08],
  '1:4': [-0.2, -0.36],
  '2:4': [0.08, 0.12],
};

function point(x: number, y: number): string {
  return `${x.toFixed(2)} ${y.toFixed(2)}`;
}

export function isCrossingChannel(board: Board, from: number, to: number): boolean {
  return (
    board.ringOf[from] === 1 &&
    board.ringOf[to] === 1 &&
    (board.sectorOf[from] + 1) % 5 !== board.sectorOf[to] &&
    (board.sectorOf[to] + 1) % 5 !== board.sectorOf[from]
  );
}

/**
 * Use this route for both the waterway and every ownership/proof overlay.
 * Endpoint order is canonical, so an undirected edge has exactly one route.
 * Curves change only presentation; the engine's coordinates and graph stay intact.
 */
export function channelPath(board: Board, from: number, to: number): string {
  const a = Math.min(from, to);
  const b = Math.max(from, to);
  const ax = board.xs[a] * BOARD_SCALE;
  const ay = board.ys[a] * BOARD_SCALE;
  const bx = board.xs[b] * BOARD_SCALE;
  const by = board.ys[b] * BOARD_SCALE;

  if (isCrossingChannel(board, a, b)) {
    // Asymmetric lanes make the central confluence read as braided channels.
    // Their recessed casings identify crossings as overpasses, never new nodes.
    const [laneX, laneY] = CENTRAL_LANES[`${board.sectorOf[a]}:${board.sectorOf[b]}`];
    const radius = BOARD_SCALE / board.rings;
    const cx = laneX * radius;
    const cy = laneY * radius;
    return `M${point(ax, ay)}C${point(ax * 0.12 + cx, ay * 0.12 + cy)} ${point(bx * 0.12 + cx, by * 0.12 + cy)} ${point(bx, by)}`;
  }

  const dx = bx - ax;
  const dy = by - ay;
  const bend = board.ringOf[a] === board.ringOf[b] ? 0.07 : 0.035;
  const direction = (a + b) % 2 === 0 ? 1 : -1;
  const cx = (ax + bx) / 2 - dy * bend * direction;
  const cy = (ay + by) / 2 + dx * bend * direction;
  return `M${point(ax, ay)}Q${point(cx, cy)} ${point(bx, by)}`;
}

/** Enumerates every undirected edge once, including all ten central links. */
export function buildChannelRoutes(board: Board): ChannelRoute[] {
  const routes: ChannelRoute[] = [];
  for (let from = 0; from < board.n; from++) {
    for (let edge = board.adjOff[from]; edge < board.adjOff[from + 1]; edge++) {
      const to = board.adj[edge];
      if (to <= from) continue;
      routes.push({
        from,
        to,
        path: channelPath(board, from, to),
        crossing: isCrossingChannel(board, from, to),
      });
    }
  }
  return routes;
}
