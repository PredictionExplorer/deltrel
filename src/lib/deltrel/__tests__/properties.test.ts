import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { getBoard, SUPPORTED_RINGS } from '../board';
import {
  applyAction,
  initialState,
  isLegalAction,
  replay,
  type GameAction,
} from '../game';
import { EMPTY, scorePosition, validateTerminalWinner } from '../scoring';

const configFor = (rings: number) => ({
  rings,
  mode: 'double' as const,
  pieRule: false,
  playerNames: ['Zero', 'One'] as [string, string],
});

describe('game properties', () => {
  it('matches an independent turn schedule through complete randomized games in every variant', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.constantFrom('classic' as const, 'double' as const),
        fc.integer({ min: 1, max: 9 }),
        fc.boolean(),
        fc.boolean(),
        fc.array(fc.nat(), { minLength: 275, maxLength: 275 }),
        (rings, mode, handicap, pie, takeSwap, selectors) => {
          const config = { ...configFor(rings), mode, handicap, pieRule: handicap === 1 && pie };
          const turnSize = mode === 'classic' ? 1 : 2;
          const swapped = config.pieRule && takeSwap;
          let state = initialState(config);
          const log: GameAction[] = [];
          const remaining = Array.from({ length: state.board.n }, (_, node) => node);
          const expectedStones = new Int8Array(state.board.n).fill(EMPTY);

          for (let placed = 0; placed < state.board.n; placed++) {
            if (placed === 1 && swapped) {
              state = applyAction(state, { type: 'swap' });
              log.push({ type: 'swap' });
              expectedStones[(log[0] as { type: 'place'; node: number }).node] = 1;
            }
            const turn = placed < handicap ? 0 : 1 + Math.floor((placed - handicap) / turnSize);
            const actor = placed < handicap ? 0 : (turn + Number(swapped)) % 2;
            const movesLeft = placed < handicap ? handicap - placed : turnSize - ((placed - handicap) % turnSize);
            expect(state.toMove).toBe(actor);
            expect(state.movesLeft).toBe(movesLeft);
            expect(state.turnCount).toBe(turn + Number(swapped && placed > 0));
            const [node] = remaining.splice(selectors[placed] % remaining.length, 1);
            const action: GameAction = { type: 'place', node };
            const previousStones = state.stones.slice();
            const previous = state;
            state = applyAction(state, action);
            log.push(action);
            expectedStones[node] = actor;
            expect(previous.stones).toEqual(previousStones);
            expect(state.stones).toEqual(expectedStones);
            expect(state.stonesPlaced).toBe(placed + 1);
            expect(state.over).toBe(remaining.length === 0);
            expect(state.canSwap).toBe(config.pieRule && placed === 0);
          }
          expect(state.over).toBe(true);
          expect(replay(config, log)).toEqual(state);
          expect(isLegalAction(state, { type: 'swap' })).toBe(false);
          expect(validateTerminalWinner(state.board, state.stones).winner).toBeOneOf([0, 1]);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('incremental placement and replay agree for arbitrary legal traces', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.nat(), { maxLength: 275 }),
        (rings, selectors) => {
          const config = configFor(rings);
          let state = initialState(config);
          const log: GameAction[] = [];
          for (const selector of selectors) {
            if (state.over) break;
            const empty = Array.from(
              { length: state.board.n },
              (_, node) => node,
            ).filter((node) => state.stones[node] === EMPTY);
            const action: GameAction = {
              type: 'place',
              node: empty[selector % empty.length],
            };
            expect(isLegalAction(state, action)).toBe(true);
            state = applyAction(state, action);
            log.push(action);

            expect(state.stonesPlaced).toBe(log.length);
            expect(state.over).toBe(log.length === state.board.n);

            const rebuilt = replay(config, log);
            expect(Array.from(rebuilt.stones)).toEqual(Array.from(state.stones));
            expect(rebuilt).toMatchObject({
              toMove: state.toMove,
              movesLeft: state.movesLeft,
              over: state.over,
              turnCount: state.turnCount,
            });
          }
        },
      ),
      { numRuns: 120 },
    );
  });

  it('the two placements of a completed Double Deltrel turn commute', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.uniqueArray(fc.integer({ min: 0, max: 49 }), {
          minLength: 3,
          maxLength: 3,
        }),
        (rings, [opening, first, second]) => {
          const config = configFor(rings);
          const start = applyAction(initialState(config), {
            type: 'place',
            node: opening,
          });
          const left = applyAction(
            applyAction(start, { type: 'place', node: first }),
            { type: 'place', node: second },
          );
          const right = applyAction(
            applyAction(start, { type: 'place', node: second }),
            { type: 'place', node: first },
          );
          expect(Array.from(left.stones)).toEqual(Array.from(right.stones));
          expect(left).toMatchObject({
            toMove: right.toMove,
            movesLeft: right.movesLeft,
            over: right.over,
          });
        },
      ),
      { numRuns: 100 },
    );
  });

  it('swapping colors swaps every generic scoring result', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.integer({ min: -1, max: 1 }), {
          minLength: 275,
          maxLength: 275,
        }),
        (rings, raw) => {
          const board = getBoard(rings);
          const stones = Int8Array.from(raw.slice(0, board.n));
          const swapped = Int8Array.from(stones, (stone) =>
            stone === EMPTY ? EMPTY : 1 - stone,
          );
          const original = scorePosition(board, stones);
          const mirrored = scorePosition(board, swapped);
          expect(mirrored.players).toEqual([
            original.players[1],
            original.players[0],
          ]);
          expect(mirrored.contestedShores).toBe(original.contestedShores);
          expect(mirrored.leader).toBe(
            original.leader === -1 ? -1 : 1 - original.leader,
          );
          expect(Array.from(mirrored.nodeOwner)).toEqual(
            Array.from(original.nodeOwner, (owner) =>
              owner === -1 ? -1 : 1 - owner,
            ),
          );
          expect(Array.from(mirrored.aliveStone)).toEqual(
            Array.from(original.aliveStone),
          );
        },
      ),
      { numRuns: 80 },
    );
  });
});

describe('full-board winner properties', () => {
  for (const rings of SUPPORTED_RINGS) {
    it(`shrinks any no-draw invariant failure on ${rings} rings`, () => {
      const board = getBoard(rings);
      fc.assert(
        fc.property(
          fc.array(fc.integer({ min: 0, max: 1 }), {
            minLength: board.n,
            maxLength: board.n,
          }),
          (raw) => {
            const terminal = validateTerminalWinner(
              board,
              Int8Array.from(raw),
            );
            expect(terminal.winner === 0 || terminal.winner === 1).toBe(true);
            expect(terminal.score.contestedShores).toBe(0);
            expect(
              terminal.score.players[0].total +
                terminal.score.players[1].total,
            ).toBe(5 * rings + 1);
            expect(terminal.margin).not.toBe(0);
            expect(Math.abs(terminal.margin) % 2).toBe(1);
          },
        ),
        { numRuns: 100 },
      );
    });
  }
});
