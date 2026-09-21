'use client';

import { useId, useRef } from 'react';
import { X } from 'lucide-react';
import { ModalDialog } from './ModalDialog';

interface RulesDialogProps {
  open: boolean;
  onClose: () => void;
}

export function RulesDialog({ open, onClose }: RulesDialogProps) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const titleId = useId();

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      ariaLabel="How to play Deltrel"
      initialFocusRef={closeButton}
      className="max-w-2xl"
    >
      <article
        className="thin-scroll panel-surface relative max-h-[calc(100dvh-2rem)] w-full overflow-y-auto rounded-3xl p-5 shadow-2xl sm:p-8"
      >
        <button
          ref={closeButton}
          type="button"
          onClick={onClose}
          aria-label="Close rules"
          className="absolute right-4 top-4 rounded-full border border-white/10 p-2 text-muted transition-colors hover:border-sand/40 hover:text-ink"
        >
          <X className="h-4 w-4" />
        </button>

        <h2 id={titleId} className="font-display pr-12 text-3xl text-sand-strong">
          How to play
        </h2>

        <div className="mt-4 space-y-4 text-sm leading-relaxed text-ink/90">
          <p className="font-display text-xl text-sand-strong">
            Shape the waterways. Claim the shoreline.
          </p>
          <p>
            Two players take turns placing stones on empty nodes — one stone per turn in{' '}
            <em>Deltrel</em>, two per turn in <em>Double Deltrel</em> (the first player places just
            one stone on the game&apos;s very first turn). A placement is mandatory while an
            empty node remains, and stones never move. Boards use 4, 6, 8, or 10 rings. The
            shared <strong className="text-sand">confluence</strong> in the center cannot be
            played, but its channels connect every pair of the five innermost nodes for
            <em> both</em> players. Crossing channels do not add a playable node.
          </p>
          <p>
            Even games use the <strong className="text-sand">pie rule</strong> in both
            variants: after the opening stone, the second player may swap sides instead of
            placing a stone. This choice is available only once and ends when they place
            their first stone. Handicap games are available only on the Full (10-ring)
            board: the first player places two to nine opening stones, and there is no swap.
          </p>
          <p>
            Every node on the boundary is a <strong className="text-sand">shore point</strong>,
            worth one point. Five special shore points are marked as{' '}
            <strong className="text-sand">capes</strong>. Coordinates run A–Z, then AA, AB,
            and onward; they identify positions without changing their connections.
          </p>
          <p>
            A connected group of your stones that directly occupies at least two shore points is a{' '}
            <strong className="text-sand">living network</strong>. The shore points can be anywhere
            on the boundary. A living network owns the shores it occupies and the shores in
            territory bordered only by your living networks. Other groups are ignored when
            calculating territory and scores.
            Stones are never physically removed. A whole group is crossed out early only when it
            cannot reach two shores even if every remaining open node were assigned to its color.
          </p>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted">
              Final score
            </h3>
            <ul className="space-y-1.5">
              <li>+1 for each shore you own</li>
              <li>
                +1 <em>cape bonus</em> if you own three or more of the five capes
              </li>
              <li>
                ±2 × the difference in network counts — the player with <em>fewer</em> networks is
                rewarded, the player with more is penalized
              </li>
            </ul>
          </div>
          <p>
            So two stones grabbing two shores look like two points, but as a separate network they
            cancel out — unless they split the opponent or claim a decisive cape. Connect
            everything; waste nothing. Live totals may be tied, but on a full board the two
            totals sum to the number of shores plus one. That sum is odd on every supported
            board, so the final margin is nonzero and someone always wins.
          </p>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted">
              Completion bounds
            </h3>
            <p>
              The score panel also fills every open node with one color, then the other, to show
              the two extreme final scores. These are mathematical bounds, not possible turn
              sequences. On a full board, changing an opponent stone to your color can only merge
              your groups or split theirs, so it cannot make your final score worse. If your
              opponent still wins after every open node is assigned to you, they have clinched
              the game. You may inspect that proof on the board, end the game immediately, or
              keep playing.
            </p>
          </div>
          <p>
            The live score and territory colors are only a current projection and are not
            monotone. Creating a new separate network can lower its player&apos;s current network award,
            and projected territory can change owners, even though the completion bounds and
            crossed-out groups remain valid.
          </p>
          <p>
            A game normally ends when the board is full. It may also end early when the players
            accept a clinched result or when a human player resigns. Early endings record the
            winner without claiming a final score for the unfinished board.
          </p>
        </div>
      </article>
    </ModalDialog>
  );
}
