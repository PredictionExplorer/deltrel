import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { GameStatus } from '../GameStatus';
import { PLAYER_COLORS } from '../theme';

afterEach(cleanup);

describe('GameStatus turn instructions', () => {
  it('exposes reported search progress accessibly and hides it outside an active search', () => {
    const props = {
      playerName: 'Player 1', controllerName: 'Browser AI', mode: 'classic' as const,
      movesLeft: 1, color: PLAYER_COLORS[0],
      searchProgress: { completedSimulations: 256, totalSimulations: 1024 },
    };
    const { rerender } = render(<GameStatus {...props} state="thinking" />);
    const progress = screen.getByRole('progressbar', { name: 'Browser AI search progress' });
    expect(progress).toHaveAttribute('max', '1024');
    expect(progress).toHaveAttribute('value', '256');
    expect(screen.getByText('25% · 256 of 1,024 simulations')).toBeVisible();
    rerender(<GameStatus {...props} state="paused" />);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    rerender(<GameStatus {...props} state="review" />);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });

  it.each(['classic', 'double'] as const)('describes a nine-stone %s handicap accurately at each placement', (mode) => {
    const { rerender } = render(<GameStatus state="human" playerName="Player 1" controllerName="Human" mode={mode} movesLeft={9} turnProgress={{ placed: 0, total: 9 }} color={PLAYER_COLORS[0]} />);
    for (let placed = 0; placed < 9; placed += 1) {
      const remaining = 9 - placed;
      rerender(<GameStatus state="human" playerName="Player 1" controllerName="Human" mode={mode} movesLeft={remaining} turnProgress={{ placed, total: 9 }} color={PLAYER_COLORS[0]} />);
      expect(screen.getByText(`9-stone opening — ${remaining} stone${remaining === 1 ? '' : 's'} left to place.`)).toBeVisible();
      expect(screen.getByRole('img', { name: `${placed} of 9 stones placed this turn` })).toBeVisible();
      expect(screen.queryByText('Two stones this turn — place the first.')).not.toBeInTheDocument();
    }
  });

  it('retains ordinary double-turn instructions', () => {
    const { rerender } = render(<GameStatus state="human" playerName="Player 2" controllerName="Human" mode="double" movesLeft={2} turnProgress={{ placed: 0, total: 2 }} color={PLAYER_COLORS[1]} />);
    expect(screen.getByText('Two stones this turn — place the first.')).toBeVisible();
    rerender(<GameStatus state="human" playerName="Player 2" controllerName="Human" mode="double" movesLeft={1} turnProgress={{ placed: 1, total: 2 }} color={PLAYER_COLORS[1]} />);
    expect(screen.getByText('One more stone finishes the turn.')).toBeVisible();
  });
});
