# Native search memory and response submission — September 12, 2026

These local changes preserve search budgets, arithmetic, response ordering within
each search, and output contracts. They have not been deployed to the training
server. Neither a whole-machine throughput gain nor an Elo/hour gain is established.

## Changes

Search edges encode their optional child index with a checked index-plus-one
`NonZeroUsize`. An absent child uses the zero niche. On the tested 64-bit host,
`Edge` shrinks from 40 to 32 bytes: **20% less edge-record storage**. This is not a
20% reduction in total search memory. Node states, logits, allocator overhead and
other structures are separate. Root reuse remaps indices through the same checked
representation; index zero and the full practically allocatable index range remain
representable. There is no lazy expansion or search-algorithm change.

The first-visit response path still matches every token and validates every
session before any session changes. Validation and deterministic error selection
remain in the original pending order. Once all sessions validate, sufficiently
large batches submit independent sessions in parallel, retaining each session's
response order. A workload threshold of 8,192 policy logits avoids parallel
scheduling for small batches; one-session and one-thread execution stay serial.

An initial experiment parallelizing both validation and submission regressed on
small batches. It was rejected. The retained implementation uses one parallel
submission pass only after serial validation. The threshold is a local CPU
engineering choice and requires target-host comparison before a performance claim
for the training fleet.

## Local measurements

On an Apple M4 Max, release builds made with Rust 1.93.1 were compared in two
alternating AB/BA pairs, with two measured repeats per arm and case after warmup.
The table shows baseline time divided by final time for `next_requests` plus
`submit`, using four Rayon threads, ring 10, and first-visit width 8. The focused
53-candidate check caps candidates at the simulation budget.

| Roots | Simulations | Native throughput ratio, up to 53 candidates |
| --- | ---: | ---: |
| 32 | 27 | 1.067× |
| 32 | 640 | 1.198× |
| 128 | 27 | 1.122× |
| 128 | 640 | 1.332× |

The wider initial matrix used eight candidates and covered rings 4/10, roots
1/8/32/128, and budgets 27/640. Its larger ring-10 cases improved 1.129–1.304×;
two tiny cases were approximately 1.4% slower. Separately comparing compact edges
alone against compact edges plus parallel submission retained 1.136–1.246× gains
at 32/128 roots. This supports keeping both changes for multithreaded actors.

Single-thread controls ranged from 0.967× to 1.082×, with several small cases about
3% slower. Legacy first-visit-width-1 controls ranged from 0.997× to 1.085×.
These results do not establish a universal speedup. All **336 measured searches**
matched their baseline's token-independent request/result trace exactly, including
the one-thread fallback and legacy path. Two paired runs are not a confidence
interval or a production-weighted estimate.

The [evidence ledger](native-search-memory-and-submit-evidence-20260912.json)
retains every timing sample, binary/source hashes, workload settings, controls,
and the rejected unconditional-parallel screening result. The earlier incomplete
large sweep was stopped to remove repeated CSR exports from the untimed Python
fixture; no partial result from that sweep is claimed as a qualified comparison.

## Verification

The rebuilt candidate passed all 100 Rust workspace tests and strict Clippy.
Thirty-six native integration tests passed with four Rayon threads. The added
tests exercise 40 ring-10 roots with mixed budgets, sparse global caps, reordered
CSR response rows, invalid first/last response values and policy lengths, retries,
and cancellation after a rejected batch. Final search outputs match unmodified
execution. Existing native tests cover root reuse and transposition remapping.
The child-index test verifies zero and large index round trips and the 32-byte
64-bit edge layout. Ruff and the benchmark's Pyright check passed.

## Reproducing the comparison

Build baseline and candidate separately with:

```text
maturin build --release --locked --manifest-path crates/star-py/Cargo.toml
```

Extract each wheel's `star_native.abi3.so` into its own directory, then run:

```text
python scripts/benchmark_native_submit.py \
  --baseline /absolute/baseline/star_native.abi3.so \
  --candidate /absolute/candidate/star_native.abi3.so \
  --roots 1 8 32 128 --budgets 27 640 --pairs 2 --repeats 2 \
  --output /absolute/native-comparison.json
```

For the focused production-scaled candidate check, add `--candidates 53` and use
`--rings 10 --roots 32 128`. For the recorded controls use
`--roots 1 32 --budgets 27 64 --pairs 2 --repeats 2`, adding `--threads 1`
or `--width 1` in separate runs for the serial-fallback and legacy checks.
The default candidate count is eight; both control runs use that value.

The controller alternates baseline/candidate and candidate/baseline order across
fresh worker processes. Each case warms up before measurement. All samples must
match the same token-independent semantic request/result trace digest. Opaque
tokens are intentionally omitted because independent roots allocate them
concurrently. State identity is represented by engine state hashes, not complete
serialized state/history or model-input bytes. Raw samples and exact binary
hashes are retained.

Only `next_requests` and `submit` time are timed. The synthetic evaluator,
root setup, semantic trace extraction, and result export are excluded. These
measurements isolate native search CPU work and must not be presented as trained
network, H100, full actor wall time, or Elo measurements. Run comparisons without
concurrent builds or CPU-heavy tests.
