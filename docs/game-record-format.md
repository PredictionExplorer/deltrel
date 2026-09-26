# Deltrel Game Notation (DGN), version 1

DGN is a UTF-8 plain-text game record for copying, pasting, and reviewing a game.
It contains its complete main line, player names, settings, dates, and result.
The file extension is `.dgn`. It borrows PGN's quoted headers and result markers;
it is a separate format, not chess PGN.

```text
[DGN "1"]
[Rules "deltrel.rules.v3"]
[Date "2026-09-26T12:00:00.000Z"]
[Updated "2026-09-26T12:05:00.000Z"]
[Player1 "Alice"]
[Player2 "Cloud AI"]
[Player1Type "human"]
[Player2Type "server"]
[Rings "6"]
[Mode "double"]
[PieRule "true"]
[Handicap "1"]
[LocalAI "standard"]
[LocalSearch "544/16"]
[CloudAI "standard"]
[CloudSearch "544/16"]
[Result "*"]
[Termination "unfinished"]

1. A10 2. swap
3. B10,C10 4. D10,E10
*
```

## Moves

Coordinates follow the [original published notation](https://gamepuzzles.com/starbook-final.pdf#page=20),
with sector names A–B–C–D–E. Each address has exactly three symbols: sector,
ring, and offset. `A10` means sector A, ring 1, offset 0. The middle digit is
the ring, with `0` representing ring 10; the last digit is the offset.
`A00` is sector A, ring 10, offset 0, and `A09` is ring 10, offset 9.
Ring 1 is innermost. Coordinates identify the same location across board sizes.

Each player's turn has its own consecutive number, starting at `1.`. A comma
joins placements made during the same turn. The opening turn contains the
handicap number of placements (one by default); subsequent turns contain one
placement in `classic` mode or two in `double` mode. `swap` occupies an entire
turn when the pie rule allows it. A partial turn is allowed only as the last
turn of the record, including a game saved after the first Double placement.

Whitespace may separate turn numbers, move groups, and the final result. Spaces
inside a comma-separated move group are not permitted. Export puts two turns
on each line. Import accepts LF or CRLF line endings and an optional UTF-8 BOM.
Version 1 contains one game and one main line; comments and variations are not
part of its grammar.

## Headers

Every header shown above is required exactly once. Header order is flexible.
Names and fixed values are case-sensitive. Values are JSON strings, so quotes,
backslashes, and control characters use their JSON escapes. For example,
`[Player1 "Alice \"Ace\""]` preserves a nickname containing quotes.

| Header | Meaning |
| --- | --- |
| `DGN` | Format version, currently `1`. |
| `Rules` | Exact rules schema, currently `deltrel.rules.v3`. |
| `Date`, `Updated` | Original game creation and last update, as UTC ISO timestamps with milliseconds. Import preserves both. |
| `Player1`, `Player2` | Player names, including longer names from earlier saved games. Player 1 is the opening player; identities do not exchange during a pie swap. |
| `Player1Type`, `Player2Type` | `human`, `local` (browser AI), or `server` (cloud AI). |
| `Rings` | `4`, `6`, `8`, or `10`. |
| `Mode` | `classic` or `double`. |
| `PieRule` | `true` or `false`; requires handicap 1. |
| `Handicap` | Number of consecutive opening placements, from 1 through 9. |
| `LocalAI`, `CloudAI` | `standard`, `deep`, or `custom`. |
| `LocalSearch`, `CloudSearch` | Exact `simulations/maxConsidered` search budgets. Standard is `544/16`; Deep is `4096/64`; every other budget is Custom. Both fields are retained even in human-only games. |
| `Result` | `*` (unfinished), `1-0` (Player 1 won), or `0-1` (Player 2 won). |
| `Termination` | `unfinished`, `board-full`, `clinch`, or `resignation`. |

Controller types and AI settings describe the settings **at save time**. Version
1 does not record controller takeovers or search-setting changes at individual
moves. Import preserves exact budgets for historical review; starting an engine
is separately subject to that engine's current limits.

The local archive ID is deliberately absent from shared text. Each import gets
a fresh local ID, so a shared game cannot overwrite an existing archived game.

## Results

The final token repeats the `Result` header and must match it:

- `unfinished` requires `*` and a board with empty locations. A player leading
  on the current score does not imply that the game has ended. Even a proven
  clinch remains unfinished until the result is accepted.
- `board-full` requires a full board. Import computes the final winner and
  requires the declared result to agree.
- `clinch` requires empty locations and a winning result. Import verifies that
  the same player wins even if every remaining empty location belongs to the
  opponent. A pending pie swap prevents a clinch claim.
- `resignation` requires empty locations and a winning result. The other player
  is the player who resigned. A resignation can occur during a partial turn.

Deltrel's full-board tiebreak produces a winner; the format has no draw marker.
Move text and the result are separate: a resignation is not a board action.

## Validation and limits

The parser replays every action through the game rules and rejects occupied or
nonexistent coordinates, illegal swaps, incorrect turn numbers, incomplete
earlier turns, excess placements, incompatible settings, and contradictory
results. Unknown or duplicate headers and unsupported versions are rejected.
Import never silently repairs moves or changes the declared winner.

Input is limited to 65,536 UTF-16 code units, matching JavaScript string length.
Export also enforces this limit; a legacy record with unusually large names
remains available locally even if it cannot fit in a shared record.
A game has at most one placement
per board node plus one pie swap. Historical budget values must be positive
unsigned 32-bit integers. Dates must be valid UTC timestamps, and `Updated`
cannot precede `Date`.

Local archive records store stable numeric node IDs rather than display
coordinates. `validateGameRecord` also checks these records by replay, verifies
their outcomes, and returns independent owned data. An invalid archive record
is rejected rather than normalized into a different game.
