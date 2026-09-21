# Root network outputs

The optional `include_network_output: true` request flag on `/v2/move` and
`/v2/analyze` adds `network_output` to the existing schema-v3 response. Health
advertises `network_output_schema_version: 1`. Existing callers omit the flag
and receive the existing response. The browser requests diagnostics explicitly;
it retries without the flag only if an older service rejects exactly that field
as unknown before inference. Cancellation and the original deadline span both
attempts.

The response describes the evaluated root position, from `perspective` (the
player who was to move). Enclosing request id and analysis state hash bind it to
that position. It does not describe subsequent search leaves or a later board.
The model identity, rules fingerprint, feature fingerprint, and weights are
unchanged by diagnostic inspection.

Every head contains row-major `shape`, `logits`, `probabilities`, `mask`, an
`activation`, and an `applicable` flag. A false structural mask has zero
probability and a null serialized logit. Active logits are finite. Softmax acts
on the last dimension; the alive head uses independent sigmoid probabilities.

| Head | Shape | Meaning |
| --- | --- | --- |
| `policy` | N | Direct placement policy before search |
| `outcome` | 2 | Loss, win for the root player |
| `score_margin` | 303 | Root-player score minus opponent score, support −151…151 |
| `ownership` | N × 3 | Final controller: root player, opponent, unclaimed |
| `alive` | N | Whether the final stone at each point belongs to a living network |
| `soft_policy` | N | Separately trained soft-policy distribution; no extra temperature is applied |
| `opponent_reply` | N + 1 | Unconditioned next-opponent-placement forecast; final slot is swap |
| `second_stone` | N | Unconditioned second-placement forecast for a Double turn |
| `final_shores` | 2 × 51 | Shore count, current player then opponent |
| `final_networks` | 2 × 26 | Living-network count, current player then opponent |
| `final_capes` | 2 × 6 | Cape count, current player then opponent |

N is the selected board's node count. Node positions use stable numeric ids;
the browser supplies the current display coordinates. Count classes start at
zero. Shore/network classes exceeding the selected board's capacity are masked.

The model always emits the opponent-reply swap slot, including outside pie
games. Its raw probability remains visible; actual swapping is relevant only
after an opening in a pie game. Future-placement forecasts are not conditioned
on the move subsequently selected by search and may include that same point.
`second_stone` is inapplicable outside a non-opening Double turn with two
placements remaining, or after a recommended swap. `opponent_reply` is
inapplicable when the current turn fills the board. Inapplicable heads retain
their real outputs, with `applicable: false`.

`auxiliary_status` is `ready`, `untrained`, or `absent`. Absent models return null
for the five auxiliary heads. Present but insufficiently supervised heads keep
their raw outputs and are marked untrained; those values should not be presented
as trained score estimates.

With trained count heads, a player's expected points are:

`E(shores) + P(capes ≥ 3) + 2 × (E(opponent networks) − E(own networks))`.

This follows the scoring rule by linearity of expectation. The separate margin
head is another learned estimate, so its expectation need not equal the
difference between the two component-derived totals. Win probabilities map to
both stable board colors using the root perspective, including after a pie swap.

Inspection adds one bounded forward pass for the root and transfers all heads
together. It uses the existing inference/device locks, never enlarges leaf
caches, and never changes search visits, backed-up values, or selected moves.
All shapes, masks, activations, and normalization are validated. Full-board live
responses are approximately 164 KB, below the browser/proxy 1 MiB response cap.

## Browser-local inference

The published champion ONNX artifact includes all eleven heads. The browser uses
the same root output schema and perspectives described above;
`training.auxiliary_predictions_ready` in the verified manifest records readiness.
Legacy six-head browser artifacts remain supported and explicitly mark the five
auxiliary heads absent. Root inspection reads every available head, while search
leaf evaluation continues to consume only the required policy/value outputs.
