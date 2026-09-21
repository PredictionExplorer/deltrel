import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getBoard, parseLabel } from '@/lib/deltrel/board';
import { scoreCompletionBounds } from '@/lib/deltrel/completion-bounds';
import { EMPTY, scorePosition } from '@/lib/deltrel/scoring';
import { DeltrelBoard } from '../DeltrelBoard';
import { BOARD_VIEWBOX_HALF, BOARD_VIEWBOX_SIZE } from '../boardGeometry';

const board = getBoard(4);

function emptyBoard(): Int8Array {
  return new Int8Array(board.n).fill(EMPTY);
}

afterEach(cleanup);

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
    const initialNode = screen.getByRole('button', {
      name: /node a, empty interior node; ada may place here/i,
    });

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

    const firstNode = screen.getByRole('button', { name: /node a, empty/i });
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

    const occupiedNode = screen.getByRole('button', {
      name: /node a, ada stone on interior node, last move/i,
    });
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
    const first = parseLabel(board, 'AE');
    const second = parseLabel(board, 'AF');
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
      screen.getByRole('button', { name: /node ae, ada stone/i }),
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
    const amber = [parseLabel(board, 'B'), parseLabel(board, 'E')];
    for (const node of amber) stones[node] = 0;
    stones[parseLabel(board, 'AE')] = 1;
    stones[parseLabel(board, 'AF')] = 1;
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
    for (const label of ['AH', 'AO', 'AP']) {
      stones[parseLabel(board, label)] = 0;
    }
    for (const label of ['AG', 'R', 'S', 'AI']) {
      stones[parseLabel(board, label)] = 1;
    }
    const score = scorePosition(board, stones);
    const bounds = scoreCompletionBounds(board, stones);
    const captured = parseLabel(board, 'AH');
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
      screen.getByRole('button', {
        name: /node ah, ada stone on shore, provably dead; cannot form a living network in any completion/i,
      }),
    ).toBeInTheDocument();
  });

  it('crosses every stone in a supplied provably dead group', () => {
    const stones = emptyBoard();
    const first = parseLabel(board, 'H');
    const second = parseLabel(board, 'S');
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
      screen.getByRole('button', {
        name: /node b, grace stone on interior node, placed last turn \(stone 1 of 2\)/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: /node d, ada stone on interior node, last move, placed this turn \(stone 1 of 2\)/i,
      }),
    ).toBeInTheDocument();
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

  it('renders recessed confluence crossings and alphabetic shoreline coordinates', () => {
    const { container } = render(<DeltrelBoard board={board} stones={emptyBoard()} interactive />);
    expect(container.querySelector('polygon')).not.toBeInTheDocument();
    expect(container.querySelectorAll('[data-channel-crossing]')).toHaveLength(5);
    for (const crossing of container.querySelectorAll('[data-channel-crossing]')) {
      expect(crossing.querySelectorAll('path')).toHaveLength(2);
      expect(crossing.querySelector('[data-channel-layer="confluence"]')).toHaveAttribute('d', expect.stringContaining('C'));
    }
    const coordinates = Array.from(container.querySelectorAll('text'), (label) => label.textContent);
    expect(coordinates).toHaveLength(board.shoreCount);
    expect(coordinates.every((label) => /^[A-Z]+$/.test(label ?? ''))).toBe(true);
    expect(screen.getByRole('group')).toHaveAccessibleDescription(/crossings are not playable junctions/i);
  });

  it('keeps connected-group overlays on the same curved confluence route', () => {
    const stones = emptyBoard();
    stones[0] = 0;
    stones[2] = 0;
    const { container } = render(<DeltrelBoard board={board} stones={stones} interactive />);
    const waterway = container.querySelector('[data-channel-crossing="0-2"] [data-channel-layer="confluence"]');
    const group = container.querySelector('[data-connection-layer="group"][data-player="0"]');
    expect(group?.getAttribute('d')).toBe(waterway?.getAttribute('d'));
    fireEvent.mouseEnter(screen.getByRole('button', { name: /^Node A, / }));
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
