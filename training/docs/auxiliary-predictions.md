# Future moves and final scoring predictions

This extension teaches the existing network to predict the opponent's next
reply, its own second stone in a regular Double turn, and each player's final
peries, stars, and quarks. It adds small output heads without replacing the
trained network or changing its search utility.

## Official game ending and labels

When the native engine proves a winner even if the losing side receives every
remaining empty cell, self-play ends. The losing side's stones fill those cells
and the native scorer computes the official final result. Final component
targets use exactly that result, including the star-count adjustment and the
quark-peri bonus. They do not continue self-play after the proof.

Existing identifiers such as `clinch_auxiliary_targets: synthetic` and
`final=clinch-loser-fill` remain for checkpoint/replay compatibility; they name
this same official clinch-completion convention. Other terminal paths retain
their existing final-board scoring semantics.

For each player, `score = peries + (quarks >= 3) + 2*(opponent_stars-own_stars)`.
A star is a connected same-color group directly occupying at least two peries.
Corner control includes territory, not only corner-stone occupancy.

## Prediction contract

`model.auxiliary_predictions` defaults to `false` for older profiles. Enabling
it adds 65,065 parameters to a width-384 model. Existing tensor names and shapes
are unchanged.

| Head | Target |
| --- | --- |
| Opponent reply | The next recorded opponent decision's search policy; an actual pie swap has its own final action slot |
| Second stone | The later search policy for the second placement of the same regular Double turn |
| Final peries | Categorical count, 0–50, for each player |
| Final stars | Categorical count, 0–25, for each player |
| Final quarks | Categorical count, 0–5, for each player |

Player pairs are ordered player-to-move, then opponent. Peries and stars mask
counts impossible for the current board size. The probability of receiving the
quark-peri bonus is the sum of quark-count probabilities for counts 3–5.

Future move labels stay within a game. They account for the remaining own
placements, exclude handicap opening placements from the second-stone task,
and represent an actual pie swap explicitly. A game ending or interruption
before the relevant decision leaves that target unavailable. No continuation
is invented after a clinch.

Schema-5 shards use a versioned optional auxiliary capability. Older shards
remain readable: final counts are derived from their existing final ownership,
living-stone and corner labels; missing future moves remain masked. Derived
components are cached outside the training batch's graph traversal. Progressive
publication can enrich existing positions without adding fresh-position credit.

The preparation script's initial loss weights are opponent reply 0.1, second
stone 0.1, final peries 0.05, final stars 0.05 and final quarks 0.025. Other loss
weights, optimizer settings, sampling, search budgets and publication cadence
are retained. These are initial settings, not an established optimum.

## Continuation and recovery

The explicitly enabled one-way checkpoint upgrade retains all existing model
weights, optimizer moments, scheduler state, EMA averages and update counters,
gradient-clipping history, learner step, and consumed-example count. Only new
heads and their optimizer state start fresh. Existing checkpoints are never
overwritten by this upgrade.

The new learner can evaluate and self-play against older champions using their
verified original architectures. Architecture differences other than the
optional new heads remain rejected. Existing arena evidence retains the same
rules, openings and search contract.

Auxiliary-enabled rollback retains the extended architecture and every learned
update, while disabling the new losses. It uses the compatible reader and the
latest checkpoint rather than restoring older weights. Source profiles, replay,
checkpoints and original evaluation evidence are preserved before deployment.

## Inference and game display

Search leaves omit auxiliary heads. Detailed root analysis computes and caches
them, without adding them to search utility or changing the ordinary policy.
The browser's six-output ONNX contract is unchanged.

The full-model API uses opt-in `include_predictions` on schema-3 analysis
requests. Old clients retain their original response shape. New clients display
expected final peries, stars, controlled corners and corner-bonus probability
for both players, plus applicable future-move predictions. These are estimates
for the analyzed position, separate from the current scoring projection.

The estimate panel is visible in ordinary network games. Technical search
details remain behind the developer setting. Old models and newly initialized
heads report the additional predictions as unavailable; an old champion is
never silently replaced by an untested candidate to populate the panel.
Migrated checkpoints retain the first observed supervised update for each
head. Their display remains unavailable until all five heads have received
labels, so progress on older count-only replay cannot expose untrained future
move predictions. This observation reuses the normal metrics synchronization.

## Qualification

Check strict replay validation, D5 transformations, pie/turn semantics,
publication revision accounting, loss masks, primary-output parity, checkpoint
and optimizer preservation, legacy serving, and frontend accessibility. On the
stopped H100 host, require a bounded real-checkpoint CUDA/BF16/compiled/native
search smoke before activating the staged immutable release.

Deployment establishes continuation and correctness. Elo per wall-clock hour
still requires measured learning curves, including self-play and evaluation
cost, with single- and double-stone results reported separately.
