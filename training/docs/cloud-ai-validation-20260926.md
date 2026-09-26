# Cloud AI validation — September 26, 2026

The service is installed on `209.20.158.77` and runs as `deltrelserve.service`,
using a dedicated unprivileged user and loopback port 8080. The verified model is
champion 566,428, with unchanged EMA weights and FP32 inference. Dependencies are
locked to PyTorch 2.13.0+cu130, Python 3.11.16, and the repository's native engine
built with Rust 1.93.0. The GPU is an H100 PCIe with 80 GB memory.

## Champion verification before production publication

At 06:10–06:13 UTC on September 26, the training host still identified step
566,428 as its champion, with a completed, conclusive `promote` decision. Newer
candidate checkpoints had not replaced this approved champion.

The live website's 72,474,137-byte browser download matched the checked-in FP32
ONNX artifact exactly, with SHA-256
`20f52f268869396de096ce23419ca071c69a43d0f5002e4ee2e03d984255a8bb`.
The cloud service reported ready with champion step 566,428, and its on-disk
213,745,031-byte checkpoint matched SHA-256
`47a8e20edb462330bd7d2877e6a2fc3de15f3c0d9d88c7e4c6406c854521d42b`.

A fresh CPU-only comparison against the original approved training checkpoint
(`876b5a7bf3efe843d0c3ae26b5ea14a358980759d73a1f9f7ed2c490c03478be`)
confirmed exact equality of values, shapes, and dtypes for all 267 model tensors
and all 267 EMA tensors: 17,467,840 parameters in each set. The comparison mapped
only the three documented final-count head renames during identity migration.
This verifies the cloud's actual weights; the browser serves the exact export
whose numerical parity is documented in the original browser release evidence.

The reviewed implementation was pushed to `main` as `f8d9079`. Public cloud
connectivity remains a separate requirement, described below; model identity
verification does not imply that Vercel can reach the GPU service.

## Real GPU workload

`scripts/benchmark_cloud_serving.py` generated distinct legal Double/pie positions
with 12–23 placements and retained history on the Full (10-ring) board. Every
successful request completed its full simulation budget, emitted monotonic
progress, returned the expected champion step, and selected a legal action.

| Configuration | Simultaneous searches | Simulations each | Wall time for all searches |
| --- | ---: | ---: | ---: |
| FP32, shared inference, graph cache | 1 | 544 | 5.154 s |
| FP32, shared inference, graph cache | 8 | 544 | 21.401 s |
| FP32, shared inference, graph cache | 1 | 4,096 | 28.984 s |
| FP32, shared inference, graph cache | 8 | 4,096 | 141.366 s |

These are small workload samples with varying cache warmth, not latency
percentiles or guarantees. The eight-Deep test supports the 180-second service
deadline; the prior 60-second setting failed under concurrent Deep load. The
active limit is eight searches, with sixteen waiting slots and a five-second
queue limit. Requests beyond capacity receive an explicit busy response.

The graph cache is bounded to 32 entries / 8 GiB. It uses the existing inference
implementation, which compares its first replay with eager output and falls back
if graph validation or capture fails. Search simulation budgets, tree selection,
per-game first-visit row count, and model precision are unchanged.

## Protocol and browser checks

- Eighteen real JSON searches covered all four board sizes, both game modes,
  available pie swaps, already-swapped positions, and nine-stone handicap
  positions. Predictions and full network outputs passed response validation.
- Detailed health rejected an unauthenticated request. Minimal `/healthz`
  returned readiness without model or filesystem diagnostics.
- The production Next.js build played a real Full-board cloud game through an
  encrypted SSH tunnel. It displayed simulation progress and applied the move.
- Stopping the actual service during that game produced an unavailable state;
  the board and swap history remained intact. Restoring health did not retry
  the failed move. Explicit Retry resumed the same game.
- Pausing a live request cancelled it and released backend capacity. Resume
  completed a valid move. Human-game model insights stayed hidden.
- Browser tests simulated a device with no Worker support and verified that
  cloud play did not download the ONNX model or browser inference runtime.

## Automated checks

- `npm run check` passed; after the final proxy fix, coverage, lint, typecheck,
  and the production build passed again: **729 tests in 47 files**.
- Backend serving/admission/inference/graph suite: **179 passed, 4 skipped**.
  Skips were GPU-only or host-specific local tests; the real GPU checks above
  exercised the deployed target separately.
- Cloud/recovery, private self-play, and engine-inspection browser scenarios:
  **8 Chromium + 8 WebKit tests passed**.
- Responsive layout suite: **10 passed**. Phone and laptop setup screenshots
  were updated and visually inspected; existing game/dialog baselines remained.
- Firefox could not launch within a bounded test timeout on the workstation;
  the cloud scenarios did not execute in that browser.

## Public deployment status

The backend is ready over SSH; it is **not connected to the public website yet**.
The provider firewall blocks inbound HTTP/HTTPS, although nginx serves the ACME
challenge correctly on the host itself. Vercel's deployment settings also require
account access. No service token has been exposed to browsers or committed.

To finish public deployment:

1. Allow inbound TCP 80 and 443 on this instance's Lambda firewall.
2. Issue a trusted certificate for `ai.deltrel.com` after pointing DNS to the
   instance, or use an IP certificate for `209.20.158.77`. Certbot 5.8 is installed.
   IP certificates require the `shortlived` profile, automatic renewal, and a
   renewal hook to reload nginx; see [Certbot IP certificate guidance](https://letsencrypt.org/2026/03/11/shorter-certs-certbot).
3. Install the prepared TLS nginx configuration with the selected hostname/IP
   and certificate paths. Keep Python port 8080 private.
4. Set the two server-only variables from the
   [serving runbook](cloud-ai-serving.md) in Vercel, confirm a runtime supporting
   the 200-second route duration, and deploy the feature branch.
5. Verify cloud readiness and a real search on `deltrel.com`, including service
   stop/start recovery, before calling the public cutover complete.
