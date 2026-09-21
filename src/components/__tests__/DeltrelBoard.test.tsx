import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getBoard, SUPPORTED_RINGS } from '@/lib/deltrel/board';
import { scoreCompletionBounds } from '@/lib/deltrel/completion-bounds';
import { EMPTY, scorePosition } from '@/lib/deltrel/scoring';
import { DeltrelBoard } from '../DeltrelBoard';
import { BOARD_VIEWBOX_HALF, BOARD_VIEWBOX_SIZE } from '../boardGeometry';

const board = getBoard(4);

function emptyBoard(): Int8Array {
  return new Int8Array(board.n).fill(EMPTY);
}

function nodeButton(node: number) {
  return screen.getByRole('button', { name: new RegExp(`^Node ${board.labels[node]},`) });
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('DeltrelBoard', () => {
  it('uses one tab stop and spatial arrow-key navigation', async () => {
    const user = userEvent.setup();
    const onPlace = vi.fn();
    render(
      <DeltrelBoard
        board={board}
        stones={emptyBoard()}
        interactive
        playerNames={['Ada', 'Grace']}
        onPlace={onPlace}
      />,
    );

    const nodes = screen.getAllByRole('button');
    const initialNode = nodeButton(0);
    expect(initialNode).toHaveAccessibleName(`Node ${board.labels[0]}, empty interior node; Ada may place here`);

    expect(nodes).toHaveLength(board.n);
    expect(nodes.filter((node) => node.tabIndex === 0)).toEqual([initialNode]);

    initialNode.focus();
    await user.keyboard('{ArrowRight}');

    const nextNode = document.activeElement as HTMLElement;
    expect(nextNode).toBeInstanceOf(SVGElement);
    expect(nextNode).not.toBe(initialNode);
    expect(nextNode).toHaveAttribute('tabindex', '0');
    expect(initialNode).toHaveAttribute('tabindex', '-1');

    const nextNodeIndex = nodes.indexOf(nextNode);
    await user.keyboard('{Enter}');
    fireEvent.keyDown(nextNode, { key: ' ', code: 'Space' });
    expect(onPlace).toHaveBeenNthCalledWith(1, nextNodeIndex);
    expect(onPlace).toHaveBeenNthCalledWith(2, nextNodeIndex);

    await user.keyboard('{End}');
    expect(document.activeElement).toBe(nodes.at(-1));
    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(nodes[0]);
  });

  it('preserves hover and mouse placement behavior', async () => {
    const user = userEvent.setup();
    const onHover = vi.fn();
    const onPlace = vi.fn();
    render(
      <DeltrelBoard
        board={board}
        stones={emptyBoard()}
        interactive
        onHover={onHover}
        onPlace={onPlace}
      />,
    );

    const firstNode = nodeButton(0);
    fireEvent.mouseEnter(firstNode);
    expect(onHover).toHaveBeenLastCalledWith(0);

    await user.click(firstNode);
    expect(onPlace).toHaveBeenCalledOnce();
    expect(onPlace).toHaveBeenCalledWith(0);

    const svg = screen.getByRole('group', { name: /deltrel board with 4 rings/i });
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: BOARD_VIEWBOX_SIZE,
      bottom: BOARD_VIEWBOX_SIZE,
      width: BOARD_VIEWBOX_SIZE,
      height: BOARD_VIEWBOX_SIZE,
      toJSON: () => ({}),
    });
    fireEvent.pointerMove(svg, {
      clientX: board.xs[1] * 100 + BOARD_VIEWBOX_HALF,
      clientY: board.ys[1] * 100 + BOARD_VIEWBOX_HALF,
    });
    expect(onHover).toHaveBeenLastCalledWith(1);
    fireEvent.click(svg, {
      clientX: board.xs[1] * 100 + BOARD_VIEWBOX_HALF,
      clientY: board.ys[1] * 100 + BOARD_VIEWBOX_HALF,
    });
    expect(onPlace).toHaveBeenNthCalledWith(2, 1);

    fireEvent.mouseLeave(svg);
    expect(onHover).toHaveBeenLastCalledWith(-1);
  });

  it('announces occupied-node state and does not reactivate it', async () => {
    const user = userEvent.setup();
    const stones = emptyBoard();
    stones[0] = 0;
    const onPlace = vi.fn();
    render(
      <DeltrelBoard
        board={board}
        stones={stones}
        interactive
        lastMove={0}
        playerNames={['Ada', 'Grace']}
        onPlace={onPlace}
      />,
    );

    const occupiedNode = nodeButton(0);
    expect(occupiedNode).toHaveAccessibleName(/Ada stone on interior node, last move/);
    expect(occupiedNode).toHaveAttribute('aria-disabled', 'true');

    occupiedNode.focus();
    await user.keyboard('{Enter}');
    await user.click(occupiedNode);
    expect(onPlace).not.toHaveBeenCalled();
  });

  it('projects pointer placement through the wider beach frame with horizontal letterboxing', () => {
    const onPlace = vi.fn();
    render(<DeltrelBoard board={board} stones={emptyBoard()} interactive onPlace={onPlace} />);
    const svg = screen.getByRole('group');
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      x: 20, y: 30, top: 30, left: 20,
      right: 548, bottom: 426, width: 528, height: 396,
      toJSON: () => ({}),
    });
    // The square drawing occupies 396px, centered in a 528px-wide SVG.
    const scale = 396 / BOARD_VIEWBOX_SIZE;
    const node = board.n - 1;
    fireEvent.click(svg, {
      clientX: 20 + (528 - 396) / 2 + (board.xs[node] * 100 + BOARD_VIEWBOX_HALF) * scale,
      clientY: 30 + (board.ys[node] * 100 + BOARD_VIEWBOX_HALF) * scale,
    });
    expect(onPlace).toHaveBeenCalledExactlyOnceWith(node);
    fireEvent.click(svg, { clientX: 25, clientY: 35 });
    expect(onPlace).toHaveBeenCalledOnce();
  });

  it('draws same-color connections and highlights a whole group', () => {
    const stones = emptyBoard();
    const first = board.idx(0, 4, 0);
    const second = board.idx(0, 4, 1);
    stones[first] = 0;
    stones[second] = 0;
    const score = scorePosition(board, stones);
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={stones}
        aliveStone={score.aliveStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    const groupPath = container.querySelector(
      'path[data-connection-layer="group"][data-player="0"]',
    );
    const networkPath = container.querySelector(
      'path[data-connection-layer="network"][data-player="0"]',
    );
    const groupPathData = groupPath?.getAttribute('d');
    expect(groupPathData).toContain('M');
    expect(networkPath?.getAttribute('d')).toContain('M');

    fireEvent.mouseEnter(
      nodeButton(first),
    );
    expect(
      container
        .querySelector('path[data-connection-layer="highlight"]')
        ?.getAttribute('d'),
    ).toBe(groupPathData);
    expect(container.querySelectorAll('[data-group-highlight]')).toHaveLength(2);
  });

  it('does not cross out rescuable stones in projected opponent territory', () => {
    const stones = emptyBoard();
    const amber = [board.idx(1, 1, 0), board.idx(4, 1, 0)];
    for (const node of amber) stones[node] = 0;
    stones[board.idx(0, 4, 0)] = 1;
    stones[board.idx(0, 4, 1)] = 1;
    const score = scorePosition(board, stones);
    const bounds = scoreCompletionBounds(board, stones);
    const { container, rerender } = render(
      <DeltrelBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    for (const node of amber) {
      expect(score.nodeOwner[node]).toBe(1);
      expect(bounds.provablyDeadStone[node]).toBe(0);
      expect(
        container.querySelector(`[data-stone-node="${node}"]`),
      ).toHaveAttribute('opacity', '1');
    }
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);

    rerender(
      <DeltrelBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        showTerritory
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );
    for (const node of amber) {
      expect(
        container.querySelector(`[data-stone-node="${node}"]`),
      ).toHaveAttribute('opacity', '0.35');
    }
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);
  });

  it('marks a provably dead stone without influence enabled', () => {
    const stones = emptyBoard();
    for (const node of [board.idx(0, 4, 3), board.idx(2, 4, 2), board.idx(2, 4, 3)]) {
      stones[node] = 0;
    }
    for (const node of [board.idx(0, 4, 2), board.idx(0, 3, 2), board.idx(1, 3, 0), board.idx(1, 4, 0)]) {
      stones[node] = 1;
    }
    const score = scorePosition(board, stones);
    const bounds = scoreCompletionBounds(board, stones);
    const captured = board.idx(0, 4, 3);
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect(score.nodeOwner[captured]).toBe(1);
    expect(bounds.provablyDeadStone[captured]).toBe(1);
    expect(
      container.querySelector(`[data-provably-dead-stone="${captured}"]`),
    ).toBeInTheDocument();
    expect(
      nodeButton(captured),
    ).toHaveAccessibleName(/Ada stone on shore, provably dead; cannot form a living network in any completion/);
  });

  it('crosses every stone in a supplied provably dead group', () => {
    const stones = emptyBoard();
    const first = board.idx(1, 2, 0);
    const second = board.idx(1, 3, 0);
    stones[first] = 0;
    stones[second] = 0;
    const dead = new Uint8Array(board.n);
    dead[first] = 1;
    dead[second] = 1;
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={stones}
        provablyDeadStone={dead}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(2);
  });

  it('renders completion stones as a read-only, explicitly hypothetical proof', async () => {
    const live = emptyBoard();
    live[0] = 0;
    const scenario = scoreCompletionBounds(board, live).scenarios[1];
    const synthetic = Uint8Array.from(
      { length: board.n },
      (_, node) => (live[node] === EMPTY ? 1 : 0),
    );
    const description =
      'Proof scenario—not actual moves. Every open node is hypothetically assigned to Grace.';
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={scenario.stones}
        nodeOwner={scenario.score.nodeOwner}
        aliveStone={scenario.score.aliveStone}
        syntheticStone={synthetic}
        proofDescription={description}
        showTerritory
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    const proofBoard = screen.getByRole('img', { name: 'Clinch proof board' });
    expect(proofBoard).toHaveAccessibleDescription(/proof scenario—not actual moves/i);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(container.querySelectorAll('[data-proof-stone]')).toHaveLength(
      board.n - 1,
    );
    expect(
      container.querySelector('[data-stone-node="0"]'),
    ).not.toHaveAttribute('data-proof-stone');
    expect(
      container.querySelector('[data-connection-layer="proof"]'),
    ).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('marks the whole previous turn and numbers double-turn stones', () => {
    const stones = emptyBoard();
    stones[0] = 0; // opening
    stones[1] = 1; // Grace's completed pair
    stones[2] = 1;
    stones[3] = 0; // Ada's in-progress first stone
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={stones}
        interactive
        lastMove={3}
        currentTurnMoves={[3]}
        lastTurnMoves={[1, 2]}
        currentTurnCapacity={2}
        playerNames={['Ada', 'Grace']}
      />,
    );

    // Both stones of Grace's finished pair stay ringed and numbered.
    expect(container.querySelector('[data-last-turn-move="1"]')).toBeInTheDocument();
    expect(container.querySelector('[data-last-turn-move="2"]')).toBeInTheDocument();
    expect(
      container.querySelector('[data-move-badge="1"][data-move-order="1"]'),
    ).toHaveTextContent('1');
    expect(
      container.querySelector('[data-move-badge="2"][data-move-order="2"]'),
    ).toHaveTextContent('2');

    // Ada's mid-turn stone pulses as the last move and is numbered 1 of 2.
    expect(container.querySelector('[data-last-move="3"]')).toBeInTheDocument();
    expect(
      container.querySelector('[data-move-badge="3"][data-move-order="1"]'),
    ).toHaveTextContent('1');

    expect(
      nodeButton(1),
    ).toHaveAccessibleName(/Grace stone on interior node, placed last turn \(stone 1 of 2\)/);
    expect(
      nodeButton(3),
    ).toHaveAccessibleName(/Ada stone on interior node, last move, placed this turn \(stone 1 of 2\)/);
  });

  it('has no detectable accessibility violations', async () => {
    const { container } = render(
      <DeltrelBoard
        board={board}
        stones={emptyBoard()}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect((await axe(container)).violations).toEqual([]);
  });

  it('keeps paint resources isolated when live and proof boards share a page', () => {
    const stones = emptyBoard();
    stones[0] = 0;
    const { container } = render(
      <>
        <DeltrelBoard board={board} stones={stones} interactive />
        <DeltrelBoard board={board} stones={stones} proofDescription="A hypothetical completion." />
      </>,
    );
    const ids = Array.from(container.querySelectorAll('[id]'), (element) => element.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const svg of container.querySelectorAll('svg')) {
      const localIds = new Set(Array.from(svg.querySelectorAll('[id]'), (element) => element.id));
      for (const element of svg.querySelectorAll('[fill], [stroke], [filter], [clip-path]')) {
        for (const attribute of ['fill', 'stroke', 'filter', 'clip-path']) {
          const reference = element.getAttribute(attribute)?.match(/^url\(#(.+)\)$/)?.[1];
          if (reference) expect(localIds.has(reference)).toBe(true);
        }
      }
    }
  });

  it('renders recessed confluence crossings without persistent coordinate labels or guides', () => {
    const { container } = render(<DeltrelBoard board={board} stones={emptyBoard()} interactive />);
    expect(container.querySelector('polygon')).not.toBeInTheDocument();
    expect(container.querySelectorAll('[data-channel-crossing]')).toHaveLength(5);
    for (const crossing of container.querySelectorAll('[data-channel-crossing]')) {
      expect(crossing.querySelectorAll('path')).toHaveLength(2);
      expect(crossing.querySelector('[data-channel-layer="confluence"]')).toHaveAttribute('d', expect.stringContaining('C'));
    }
    expect(container.querySelector('[data-coordinate-axes], [data-coordinate-axis], [data-coordinate-guides], [data-coordinate-band], [data-coordinate-cell]')).not.toBeInTheDocument();
    expect(container.querySelector('text')).not.toBeInTheDocument();
    expect(screen.getByRole('group')).toHaveAccessibleDescription(/crossings are not playable junctions/i);
  });

  it.each(SUPPORTED_RINGS)('shows only the hovered point coordinate on a %i-ring board', (rings) => {
    const atlas = getBoard(rings);
    const { container } = render(<DeltrelBoard board={atlas} stones={new Int8Array(atlas.n).fill(EMPTY)} interactive />);
    const nodes = screen.getAllByRole('button');
    expect(nodes).toHaveLength(atlas.n);
    expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
    for (let node = 0; node < atlas.n; node++) {
      expect(nodes[node]).toHaveAccessibleName(new RegExp(`^Node ${atlas.labels[node]}, empty`));
      fireEvent.mouseEnter(nodes[node]);
      expect(container.querySelectorAll('[data-coordinate-tooltip]')).toHaveLength(1);
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveAttribute('data-coordinate-tooltip', atlas.labels[node]);
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(atlas.labels[node]);
      expect(container.querySelectorAll('text')).toHaveLength(1);
    }
    fireEvent.mouseLeave(screen.getByRole('group'));
    expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
    expect(container.querySelector('text')).not.toBeInTheDocument();
  });

  it('reveals the focused point coordinate through keyboard navigation and keeps proof boards read-only', async () => {
    const user = userEvent.setup();
    const { container, rerender } = render(<DeltrelBoard board={board} stones={emptyBoard()} interactive />);
    nodeButton(0).focus();
    await user.keyboard('{End}');
    const last = board.n - 1;
    expect(document.activeElement).toBe(nodeButton(last));
    expect(container.querySelector('[data-coordinate-tooltip]')).toHaveAttribute('data-coordinate-tooltip', board.labels[last]);
    expect(container.querySelector('[data-coordinate-axes], [data-coordinate-band]')).not.toBeInTheDocument();
    await user.keyboard('{Home}');
    expect(container.querySelector('[data-coordinate-tooltip]')).toHaveAttribute('data-coordinate-tooltip', board.labels[0]);
    await user.keyboard('{Escape}');
    expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
    expect(document.activeElement).toBe(nodeButton(0));
    expect(container.querySelector('[data-keyboard-focus]')).toHaveAttribute('data-keyboard-focus', board.labels[0]);
    await user.keyboard('{ArrowRight}');
    const focusedCoordinate = document.activeElement?.getAttribute('aria-label')?.match(/^Node ([A-Z]+\d+),/)?.[1];
    expect(container.querySelector('[data-coordinate-tooltip]')).toHaveAttribute('data-coordinate-tooltip', focusedCoordinate);
    await user.keyboard('{Home}');
    fireEvent.blur(nodeButton(0));
    expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();

    rerender(<DeltrelBoard board={board} stones={emptyBoard()} interactive syntheticStone={new Uint8Array(board.n)} proofDescription="Hypothetical completion." />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(container.querySelector('[data-coordinate-axes], [data-coordinate-guides]')).not.toBeInTheDocument();
    expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
  });

  describe('touch coordinate inspection', () => {
    beforeEach(() => vi.useFakeTimers());

    function touchBoard(stones = emptyBoard()) {
      const onPlace = vi.fn();
      const onHover = vi.fn();
      const rendered = render(<DeltrelBoard board={board} stones={stones} interactive onPlace={onPlace} onHover={onHover} />);
      const svg = screen.getByRole('group');
      vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
        x: 0, y: 0, top: 0, left: 0,
        right: BOARD_VIEWBOX_SIZE, bottom: BOARD_VIEWBOX_SIZE,
        width: BOARD_VIEWBOX_SIZE, height: BOARD_VIEWBOX_SIZE,
        toJSON: () => ({}),
      });
      return { ...rendered, svg, onPlace, onHover };
    }

    function point(node: number, pointerType = 'touch') {
      return {
        pointerId: 7,
        pointerType,
        isPrimary: true,
        button: 0,
        buttons: 1,
        clientX: board.xs[node] * 100 + BOARD_VIEWBOX_HALF,
        clientY: board.ys[node] * 100 + BOARD_VIEWBOX_HALF,
      };
    }

    it('places exactly once on a short tap and cancels its pending inspection timer', () => {
      const { container, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      // A real touch can focus its target and emit compatibility mouse events.
      act(() => target.focus());
      fireEvent.mouseEnter(target);
      act(() => vi.advanceTimersByTime(349));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      fireEvent.pointerUp(target, { ...pointer, buttons: 0 });
      fireEvent.click(target, { ...pointer, detail: 1 });
      fireEvent.mouseEnter(target);
      fireEvent.focus(target);
      expect(onPlace).toHaveBeenCalledExactlyOnceWith(0);
      act(() => vi.advanceTimersByTime(1_000));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      expect(onPlace).toHaveBeenCalledOnce();
    });

    it.each(['touch', 'pen'])('shows a coordinate after a %s hold without placing on release', (pointerType) => {
      const { container, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0, pointerType);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(350));
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveAttribute('data-coordinate-tooltip', board.labels[0]);
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(board.labels[0]);
      expect(onPlace).not.toHaveBeenCalled();
      fireEvent.pointerUp(target, { ...pointer, buttons: 0 });
      // Browsers may synthesize a click after releasing a press-and-hold.
      fireEvent.click(target, { ...pointer, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
      expect(container.querySelector('[data-coordinate-axis], [data-coordinate-band]')).not.toBeInTheDocument();

      const next = nodeButton(1);
      const nextPointer = point(1, pointerType);
      fireEvent.pointerDown(next, nextPointer);
      act(() => vi.advanceTimersByTime(100));
      fireEvent.pointerUp(next, { ...nextPointer, buttons: 0 });
      fireEvent.click(next, { ...nextPointer, detail: 1 });
      expect(onPlace).toHaveBeenCalledExactlyOnceWith(1);
    });

    it('cancels a hold when a finger moves to scroll and suppresses any following click', () => {
      const { container, svg, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(100));
      const moved = { ...pointer, clientY: pointer.clientY + 11 };
      fireEvent.pointerMove(svg, moved);
      act(() => vi.advanceTimersByTime(500));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      fireEvent.pointerUp(target, { ...moved, buttons: 0 });
      fireEvent.click(target, { ...moved, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
    });

    it('does not turn a hold that began on empty coastline into a placement', () => {
      const { container, svg, onPlace } = touchBoard();
      const outside = { ...point(0), clientX: 1, clientY: 1 };
      fireEvent.pointerDown(svg, outside);
      act(() => vi.advanceTimersByTime(350));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      fireEvent.pointerUp(svg, { ...point(0), buttons: 0 });
      fireEvent.click(nodeButton(0), { ...point(0), detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
    });

    it('accepts a fresh mouse click immediately after touch inspection', () => {
      const { onPlace } = touchBoard();
      const target = nodeButton(0);
      fireEvent.pointerDown(target, point(0));
      act(() => vi.advanceTimersByTime(350));
      fireEvent.pointerUp(target, { ...point(0), buttons: 0 });
      const next = nodeButton(1);
      fireEvent.pointerDown(next, point(1, 'mouse'));
      fireEvent.pointerUp(next, { ...point(1, 'mouse'), buttons: 0 });
      fireEvent.click(next, { ...point(1, 'mouse'), detail: 1 });
      expect(onPlace).toHaveBeenCalledExactlyOnceWith(1);
    });

    it.each([100, 350])('clears a cancelled gesture after %ims without a move or delayed tooltip', (elapsed) => {
      const { container, svg, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(elapsed));
      fireEvent.pointerCancel(svg, pointer);
      fireEvent.click(target, { ...pointer, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
      act(() => vi.advanceTimersByTime(1_000));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      expect(onPlace).not.toHaveBeenCalled();
    });

    it('can inspect an occupied point without placing another stone', () => {
      const stones = emptyBoard();
      stones[0] = 1;
      const { container, onPlace } = touchBoard(stones);
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(350));
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(board.labels[0]);
      fireEvent.pointerUp(target, { ...pointer, buttons: 0 });
      fireEvent.click(target, { ...pointer, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
    });

    it('clears the pending hold timer when the board unmounts', () => {
      const { unmount, onPlace, onHover } = touchBoard();
      fireEvent.pointerDown(nodeButton(0), point(0));
      expect(vi.getTimerCount()).toBeGreaterThan(0);
      unmount();
      expect(vi.getTimerCount()).toBe(0);
      const hoverCalls = onHover.mock.calls.length;
      act(() => vi.advanceTimersByTime(1_000));
      expect(onPlace).not.toHaveBeenCalled();
      expect(onHover).toHaveBeenCalledTimes(hoverCalls);
    });

    it('does not resurrect a held coordinate after Escape cancels the pending gesture', () => {
      const { container, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(100));
      fireEvent.keyDown(target, { key: 'Escape' });
      act(() => vi.advanceTimersByTime(500));
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();
      fireEvent.pointerUp(target, { ...pointer, buttons: 0 });
      fireEvent.click(target, { ...pointer, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
    });

    it('restores keyboard inspection when tabbing into the board after using touch', async () => {
      render(<button type="button">Before the board</button>);
      const { container, onPlace } = touchBoard();
      const target = nodeButton(0);
      const pointer = point(0);
      fireEvent.pointerDown(target, pointer);
      act(() => vi.advanceTimersByTime(350));
      fireEvent.pointerUp(target, { ...pointer, buttons: 0 });
      fireEvent.click(target, { ...pointer, detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();

      // The timed gesture has finished; exercise real keyboard focus traversal.
      vi.useRealTimers();
      const user = userEvent.setup();
      screen.getByRole('button', { name: 'Before the board' }).focus();
      await user.keyboard('{Tab}');
      expect(document.activeElement).toBe(target);
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(board.labels[0]);
    });

    it.each(['preview', 'proof'])('allows mouse and long-press inspection on a read-only %s without any move', (kind) => {
      const onPlace = vi.fn();
      const { container } = render(
        <DeltrelBoard
          board={board}
          stones={emptyBoard()}
          interactive={kind === 'proof'}
          syntheticStone={kind === 'proof' ? new Uint8Array(board.n) : undefined}
          proofDescription={kind === 'proof' ? 'Hypothetical completion.' : undefined}
          onPlace={onPlace}
        />,
      );
      const svg = screen.getByRole('img');
      vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
        x: 0, y: 0, top: 0, left: 0,
        right: BOARD_VIEWBOX_SIZE, bottom: BOARD_VIEWBOX_SIZE,
        width: BOARD_VIEWBOX_SIZE, height: BOARD_VIEWBOX_SIZE,
        toJSON: () => ({}),
      });
      fireEvent.pointerMove(svg, point(0, 'mouse'));
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(board.labels[0]);
      fireEvent.click(svg, { ...point(0, 'mouse'), detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
      fireEvent.mouseLeave(svg);
      expect(container.querySelector('[data-coordinate-tooltip]')).not.toBeInTheDocument();

      fireEvent.pointerDown(svg, point(1));
      act(() => vi.advanceTimersByTime(350));
      expect(container.querySelector('[data-coordinate-tooltip]')).toHaveTextContent(board.labels[1]);
      fireEvent.pointerUp(svg, { ...point(1), buttons: 0 });
      fireEvent.click(svg, { ...point(1), detail: 1 });
      expect(onPlace).not.toHaveBeenCalled();
      expect(screen.queryByRole('button')).not.toBeInTheDocument();
      expect(container.querySelector('[data-coordinate-axis], [data-coordinate-band]')).not.toBeInTheDocument();
    });
  });

  it('keeps connected-group overlays on the same curved confluence route', () => {
    const stones = emptyBoard();
    stones[0] = 0;
    stones[2] = 0;
    const { container } = render(<DeltrelBoard board={board} stones={stones} interactive />);
    const waterway = container.querySelector('[data-channel-crossing="0-2"] [data-channel-layer="confluence"]');
    const group = container.querySelector('[data-connection-layer="group"][data-player="0"]');
    expect(group?.getAttribute('d')).toBe(waterway?.getAttribute('d'));
    fireEvent.mouseEnter(nodeButton(0));
    expect(container.querySelector('[data-connection-layer="highlight"]')?.getAttribute('d')).toBe(waterway?.getAttribute('d'));
  });

  it('uses CSS motion so reduced-motion preferences can suppress every moving layer', () => {
    const stones = emptyBoard();
    stones[0] = 0;
    const { container } = render(<DeltrelBoard board={board} stones={stones} lastMove={0} />);
    expect(container.querySelector('animate')).not.toBeInTheDocument();
    expect(container.querySelector('[data-last-move="0"]')).toBeInTheDocument();
  });

  it('pauses water light in a hidden document and removes its visibility listener', () => {
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    const addListener = vi.spyOn(document, 'addEventListener');
    const removeListener = vi.spyOn(document, 'removeEventListener');
    const { container, unmount } = render(<DeltrelBoard board={board} stones={emptyBoard()} />);
    const svg = container.querySelector('svg')!;
    expect(svg.style.getPropertyValue('--water-motion-state')).toBe('paused');

    hidden.mockReturnValue(false);
    fireEvent(document, new Event('visibilitychange'));
    expect(svg.style.getPropertyValue('--water-motion-state')).toBe('running');
    hidden.mockReturnValue(true);
    fireEvent(document, new Event('visibilitychange'));
    expect(svg.style.getPropertyValue('--water-motion-state')).toBe('paused');

    const listener = addListener.mock.calls.find(([event]) => event === 'visibilitychange')![1];
    unmount();
    expect(removeListener).toHaveBeenCalledWith('visibilitychange', listener);
  });
});
