# Conditional keep values in pie search

Search identity `gumbel-completed-q-v3-conditional-keep` separates two value
questions without changing the model, game rules, features, or replay labels.
The network predicts the actual game's result at a state where the responder
can swap. Placement search estimates the result conditional on keeping.

Previously, an opening simulation could back up the responder's swap-inclusive
network value, then combine it with keep-only values from deeper simulations.
For example, when every keep move has value -0.8 and swapping has value +0.8,
the first opening simulation backed up -0.8 for the opener and the second +0.8.
Applying `-abs` to their mean falsely made the opening look balanced. The
responder's unvisited-action completion had a second mismatch: it combined a
positive swap-inclusive root value with negative keep action values, raising
unvisited placements solely because they had not been searched.

At a swap-available node, search now uses the network's placement policy but
never backs up its value. The first inference expands policy only; the same
forced root simulation continues through a placement before consuming its
simulation budget. Its root edge remains reserved. Wrong-edge continuation and
stale response tokens fail without changing statistics. Cancellation can drop
pending inference while preserving already expanded policy and completed work.

At those nodes, unvisited Q values use the prior-weighted estimate of visited
keep Q values. Before any keep action has been evaluated they use zero, which
adds the same constant to all policy logits. If the visited priors underflow to
zero, the fallback is the visit-weighted keep estimate, never the network's
swap-inclusive value. Ordinary nodes retain the existing mixed-value formula.

The opening transform remains `-abs(mean keep Q)`. It is not applied separately
to each leaf return; that would change the nonlinear objective. The game driver
continues to decide whether to swap using the selected keep move's searched
value and the existing dead zone. Actual tree states and inference keys retain
`swap_available`, and replay continues to describe actual states and outcomes.
No synthetic keep state or counterfactual training label is generated.

Serial search, first-visit prefetch, native batches, retained subtrees and browser
WASM implement the same completion protocol. A policy-only response is not a
completed simulation. Prefetch can change inference order, but it must preserve
the request multiset, backup order, exact budget, selected action and targets.
An opening search needs one additional policy inference for each newly visited
responder node; unchanged simulation counts are not a claim of unchanged cost.

All new arena/evaluation contracts and browser cache URLs use the new search
identity. Historical results remain historical; they must not be joined into
the corrected search's evidence namespace. The correction does not establish
an Elo improvement, and the separate asymmetric-compute swap approximation and
search-strength calibration remain outside this change.

Validation covers classic and double pie opening/responder oracle values,
nonlinear transform order, neutral completion before visits, root refresh and
subtree reuse, cancellation and token ownership, serial/prefetch parity, native
budget counters, true state preservation and browser progress reporting.
