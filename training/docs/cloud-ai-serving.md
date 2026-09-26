# Cloud AI serving

The browser calls the website's same-origin `/v2/move` endpoint. Next.js forwards
the request over HTTPS to `deltrelserve`, adding a private bearer token. The
browser receives newline-delimited JSON progress followed by one validated JSON
result. Ordinary API callers can request `application/json` instead. There is
one search per request; streaming does not start a second search.

Each request contains the board, rule variant, side to move, remaining stones in
the turn, pie/swap state, retained placement history, deterministic seed, and
simulation/candidate budgets. Each response identifies its request and model,
and contains the chosen move, outcome probabilities, score distribution, root
search statistics, and optional learned forecasts. Both ends validate the
protocol. The browser rejects stale results after undo, pause, a new game, or a
controller change. Human-versus-AI games keep their existing hidden-insight UI.

The service does not store user games or require session affinity. A stopped
server loses disposable inference caches, not browser game state. No database,
WebSocket session, or per-game GPU model is needed.

## Concurrency and limits

One process owns the model and GPU. Up to eight searches share compatible neural
inference batches, with a bounded exact-input prediction cache. Batch keys
include model identity, rules, history, and advantage inputs; incompatible board
sizes never mix. Per-game tree search still uses one first-visit row and no
subtree reuse, matching the browser champion's search contract. Physical batching
can change floating-point rounding; identical cross-device move choices are not
guaranteed.

The cloud profile allows eight active searches and sixteen queued requests, with
a five-second queue timeout and 180-second total request deadline. The longer
deadline lets concurrent Full-board Deep searches finish without reducing their
requested effort. Excess work
receives retryable HTTP 503 with `Retry-After`, before a progress stream is opened.
Requests are limited to 64 KiB. The proxy deadline is 190 seconds, the client 195,
and the Next.js route 200. Standard uses 544 simulations/16 candidates; Deep uses
4,096/64. Custom cloud searches cannot exceed those server maximums.

Client disconnects cancel queued and active work. Active capacity is released
only after native search actually stops; an aborted browser connection cannot
free a slot while its GPU work continues. Shutdown drains admitted work before
closing the shared model. Configure only one service process per GPU: multiple
independent Uvicorn workers would duplicate weights, caches, and admission limits.

## Install and run

The initial host is `ubuntu@209.20.158.77`, with one H100 PCIe 80 GB GPU. Runtime
files use this layout:

```
/opt/deltrel-cloud/releases/<release>/training/  immutable application and venv
/opt/deltrel-cloud/current                     active release symlink
/opt/deltrel-cloud/models/champion-566428/      verified model snapshot
/etc/deltrelserve/production.yaml              serving configuration
/etc/deltrelserve/service.env                  private bearer token (root, 0600)
/var/cache/deltrelserve/                        disposable runtime cache
```

Use the repository's locked Python dependencies (`uv sync --frozen --python 3.11
--extra serve`), Rust 1.93, and the native extension from the same checkout.
Build that extension in release mode with `Cargo.lock`. Install Python under
`/opt/deltrel-cloud/python` using `UV_PYTHON_INSTALL_DIR` so the service does not
depend on an operator's home directory. The service runs as the dedicated
unprivileged `deltrelserve` user. Code, configuration, and models are read-only
to that user; model directories need group read/traverse permission. Install
the checked-in [service unit](../deploy/deltrelserve.service).

Start from [the cloud configuration](../configs/deltrelserve-cloud.yaml) and set
absolute `experiment_config` and `model_manifest` paths to a verified snapshot.
Use the snapshot's actual model/game configuration, with FP32 inference and
`train.compile: false`. A generic training profile may have a different model
architecture and must not be substituted. Startup validates checkpoint hash,
size, schema, step, lineage, and EMA weights. Candidate pointers are rejected.

The initial cloud model is the same confirmed step 566,428 used by the browser.
Its losslessly migrated checkpoint SHA-256 is
`47a8e20edb462330bd7d2877e6a2fc3de15f3c0d9d88c7e4c6406c854521d42b`.
See [the release evidence](../../docs/browser-champion-566428-release.md).

Generate a cryptographically random token and save it as
`DELTRELSERVE_BEARER_TOKEN` in the root-readable environment file. Do not put the
token in Git, command arguments, URLs, browser storage, or public build variables.
The service listens on `127.0.0.1:8080`. Keep that port private.

Point `ai.deltrel.com` at the GPU host, issue a publicly trusted TLS certificate,
and install [the nginx configuration](../deploy/deltrelserve.nginx.conf). It
exposes only the three application endpoints, disables buffering, and never
automatically retries an inference request. Enable automatic certificate renewal
with a successful-renewal nginx reload. Do not reuse the host's Jupyter tunnel.

In the website deployment, set these server-only variables and redeploy:

```
DELTREL_AI_SERVER_URL=https://ai.deltrel.com
DELTREL_AI_BEARER_TOKEN=<same private token as DELTRELSERVE_BEARER_TOKEN>
```

The backend requires the token for search, detailed health, and OpenAPI. Public
loopback `/healthz` reports only readiness. The website exposes only sanitized
capability health and safe errors. Cross-origin browser POSTs are rejected by
the website proxy. This is drive-by abuse mitigation, not user authentication
or a distributed per-user rate limit. Configure website-platform rate limits
when opening service to substantial public traffic; the bounded backend queue
continues to protect GPU admission independently.

The website needs a function runtime supporting the 200-second route limit.
On Vercel Hobby, enable Fluid compute: the legacy runtime is capped at 60
seconds. [Vercel duration limits](https://vercel.com/docs/functions/configuring-functions/duration)
document the applicable plan and runtime limits.

## Validation and operation

Test authenticated health, both JSON and progress responses, all board/rule
variants, concurrent distinct positions, cancellation, overload, and server-off
recovery before public cutover. The reproducible target-host workload is:

```sh
cd /opt/deltrel-cloud/current/training
# Supply DELTRELSERVE_BEARER_TOKEN through the protected operator environment.
.venv/bin/python scripts/benchmark_cloud_serving.py --concurrency 4
.venv/bin/python scripts/benchmark_cloud_serving.py --concurrency 8
.venv/bin/python scripts/benchmark_cloud_serving.py --concurrency 1 --simulations 4096
```

The benchmark uses distinct legal positions with retained history, validates
progress through the full requested budget, and checks response identity and
move legality. A small benchmark establishes a tested configuration, not a
guaranteed number of production games. Slow clients and long searches occupy
capacity only for their current request; idle games use no search slot.

```sh
sudo systemctl status deltrelserve
sudo journalctl -u deltrelserve --since '10 minutes ago'
sudo systemctl stop deltrelserve       # intentionally take Cloud AI offline
sudo systemctl start deltrelserve      # restore service
sudo systemctl disable deltrelserve   # also prevent startup after reboot
sudo systemctl enable deltrelserve    # restore boot startup
```

Stopping the service releases GPU work but does not stop cloud-instance billing.
On Lambda, billing ends only when the instance is terminated; an operating-system
shutdown does not suspend billing. Preserve the model snapshot and configuration
outside the instance before termination, then recreate from the release and
snapshot when needed. See [Lambda billing](https://docs.lambda.ai/public-cloud/billing/)
and [instance operations](https://docs.lambda.ai/public-cloud/console/).
The enabled service starts automatically after an ordinary reboot. Verify
whether a recreated instance has a new IP and update DNS if needed. Offline browsers retain
their game, recheck availability every 30 seconds while visible, and offer
explicit Retry or Switch to AI on this device after a failed move.

Deploy future changes into a new release directory, validate them on a separate
loopback port, then stop the service, switch `current`, and start it. Rollback
uses the previous symlink target. Keep the previous snapshot until validation
passes. Publish only confirmed champions; browser and cloud publications are
separate deliberate operations, so check their step/identity after each release.
