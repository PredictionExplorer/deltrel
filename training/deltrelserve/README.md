# Server inference resources

Clients can request live search progress from `POST /v2/move` or `POST /v2/analyze`
with `Accept: application/x-ndjson`. Each newline-delimited JSON event is either
`{"type":"progress","completed_simulations":12,"total_simulations":128}`,
`{"type":"result","result":...}` containing the usual validated analysis, or
`{"type":"error","error":...}`. Progress counts completed native simulations,
including exact terminal continuations and cached inference; it is not inferred
from elapsed time. A single search produces progress and its final result.

The stream retains authentication, request-size, concurrency, cancellation, and
deadline limits. Slow readers receive coalesced progress rather than an unbounded
event backlog. The existing JSON response remains the default for clients that
do not request streaming. The web proxy forwards the stream as it arrives, and
the game discards progress from cancelled or superseded searches.

Admission happens before sending response headers, including for streaming
clients. `limits.max_concurrency` bounds active native searches, while
`limits.max_queued_requests` bounds additional waiting requests (zero disables
waiting). A full queue or expired queue deadline returns HTTP 503 with
`error.code: "service_busy"` and `Retry-After: 1`; callers should offer an explicit
retry without making a move. The request deadline includes queue time. Waiting
requests leave the queue immediately on disconnect. Running requests signal
cooperative cancellation and retain their slot until native work finishes.

Health includes `capacity` with active, queued, and configured request counts.
When a bearer token is configured, `/v2/health` and `/v2/openapi.json` require it,
just like analysis. The unauthenticated `/healthz` exposes only readiness status.
A full queue does not make the model unready: the service still reports its
loaded champion while rejecting excess work. Shutdown rejects new admissions,
cancels outstanding work, and drains native workers before releasing the model.

`configs/deltrelserve-cloud.yaml` is a starting configuration for one H100 with
eight concurrent games, sixteen waiting requests, and a 512 MiB prediction cache.
Run one service process per GPU so requests share the same batching worker.
Standard uses 544 simulations and 16 candidates; Deep uses 4,096 simulations
and 64 candidates, matching the local controls. Benchmark the largest board and
Deep under concurrent load before increasing capacity. The cloud configuration
binds to loopback and requires `DELTRELSERVE_BEARER_TOKEN`; the website's server
must reach it over a private or TLS-protected transport. Stop the service or
machine whenever needed; the website treats it as temporarily unavailable.

Cloud inference enables the existing CUDA graph cache to reduce repeated launch
overhead without changing precision or search budgets. Each capture is checked
against eager inference before use; unsupported captures fall back to eager.
Retained graphs are bounded by `cuda_graph_max_entries` (32) and
`cuda_graph_max_bytes` (8 GiB), in addition to prediction-cache limits. Install
CUDA runtime bindings (`cuda.bindings.runtime`) on the GPU host. This option
requires shared batching so one thread owns graph capture and replay. Other
server configurations leave `cuda_graphs` disabled by default.

The optional `inference` section in the server YAML controls prediction reuse and
cross-request batching. Defaults retain at most 4,096 exact-input predictions,
charged against a 64 MiB per-model budget, and allow at most sixteen pending
inference jobs and sixteen rows in a neural batch. The effective row limit is
also capped by the server's request concurrency limit.

One worker owns neural execution and prediction storage for each immutable model.
Compatible requests can share a forward pass; different board sizes never mix.
Cache keys include the model identity and every model input, including retained
history and playout-doubling advantage. Cached raw predictions are converted into
the caller's response without reusing request tokens or caller-specific utility.

`max_wait_seconds` defaults to 0.001 and applies only while multiple searches hold
model leases. A lone search, including the single-request Mac configuration,
never waits for another row. Completed cached requests do not invoke the model.
Set both cache limits to zero to disable prediction storage, and set
`shared_batching: false` to disable the inference broker. These settings do not
change model precision, simulation budgets, or tree selection rules.

Model replacement happens between active searches. The retired model's worker
and prediction storage are closed after its final lease; application shutdown
drains admitted searches before releasing their model. Cancellation remains
cooperative at the neural-call boundary, so a cancelled request cannot release
model resources while its inference is still running.

For pie decisions, `swap_recommended` compares the selected placement's searched
value with the swap dead zone. `root_value` remains the diagnostic average of
root visits; exploration of other placements does not determine whether to swap.
The native extension must provide `selected_action_values`; older extensions
are rejected rather than silently using the aggregate value.

Cross-request batching may change floating-point rounding because the physical
batch shape changes. Server latency and playing-strength comparisons should use
the target device; unit tests establish cache identity, response routing, and
resource lifetime, not a hardware speedup.

## Optional learned forecasts

Schema-v3 callers can set `include_predictions: true` to receive the optional
`predictions` field. Existing callers keep the same response shape. The field is
`null` for legacy models and migrated models until every auxiliary head has
received labeled supervision. Historical replay alone cannot establish that
the future-move heads have learned. Model health reports
`auxiliary_predictions_ready`.

`final_counts` names players 0 and 1 explicitly, independent of whose turn it is.
It reports expected final shores, networks, controlled corners, and the probability
of earning the cape-bonus bonus (at least three corners). `final_basis` is
`official_end`: clinched games are scored after filling remaining cells with the
losing side's stones, exactly as in training. These are learned expectations;
the current board score is displayed separately.

`opponent_reply` and `second_stone` contain an action and its model probability.
The second-stone forecast applies only at the beginning of a normal Double turn.
An opponent's future pie swap can be forecast during the opening. Predictions
describe the root position, rather than a forced continuation of the chosen move.

Auxiliary heads run with detailed root inference and are cached with a separate
root key. Normal search leaves skip their computation. None of these predictions
changes search utility or the selected move.
