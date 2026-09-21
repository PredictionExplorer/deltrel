import { describe, expect, it } from 'vitest';
import { getBoard, SUPPORTED_RINGS } from '@/lib/deltrel/board';
import { BOARD_SCALE, buildChannelRoutes, channelPath } from '../boardGeometry';

describe('waterway geometry', () => {
  it.each(SUPPORTED_RINGS)('preserves every graph edge and both endpoints on a %i-ring board', (rings) => {
    const board = getBoard(rings);
    const routes = buildChannelRoutes(board);
    expect(routes).toHaveLength(board.adj.length / 2);
    expect(new Set(routes.map(({ from, to }) => `${from}:${to}`)).size).toBe(routes.length);

    for (const route of routes) {
      const { from, to, path } = route;
      expect(Array.from(board.adj.slice(board.adjOff[from], board.adjOff[from + 1]))).toContain(to);
      const values = path.match(/-?\d+(?:\.\d+)?/g)!.map(Number);
      expect(values[0]).toBeCloseTo(board.xs[from] * BOARD_SCALE, 2);
      expect(values[1]).toBeCloseTo(board.ys[from] * BOARD_SCALE, 2);
      expect(values.at(-2)).toBeCloseTo(board.xs[to] * BOARD_SCALE, 2);
      expect(values.at(-1)).toBeCloseTo(board.ys[to] * BOARD_SCALE, 2);
      expect(channelPath(board, to, from)).toBe(path);
      expect(path).not.toMatch(/NaN|Infinity/);
    }
  });

  it.each(SUPPORTED_RINGS)('routes all ten central links with five distinct curved crossings at %i rings', (rings) => {
    const board = getBoard(rings);
    const central = buildChannelRoutes(board).filter(({ from, to }) => board.ringOf[from] === 1 && board.ringOf[to] === 1);
    expect(central).toHaveLength(10);
    const crossings = central.filter((route) => route.crossing);
    expect(crossings).toHaveLength(5);
    expect(new Set(crossings.map((route) => route.path)).size).toBe(5);
    for (const route of crossings) {
      expect(route.path).toContain('C');
      expect(route.path).not.toContain('L');
      expect(route.path).not.toContain('Z');
    }
  });
});
