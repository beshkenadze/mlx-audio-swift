# Spec: Fork patches for mlx-audio-swift — Parakeet throughput

**Status:** proposed  
**Owner:** TBD  
**Priority:** P2 (ANE remains default; MLX is secondary path)  
**Created:** 2026-04-21  
**Last updated:** 2026-04-21

## Context

MLX Parakeet on M1 Max currently runs at ~24× realtime (1.23 s/chunk,
wall-clock 55.5 s for 26 min audio). ANE/CoreML via FluidAudio runs at
~100–170× realtime. We have exhausted the obvious optimizations outside
the fork:

- `_MLXSingletonStore` singleton + async-semaphore (no OOM)
- `Memory.cacheLimit = 0` + per-chunk `Memory.clearCache()`
- Semaphore cap tuned to 2 (measured: more lanes do not help)
- JIT warm-up in `initialize()`
- Per-task MLX stream A/B did not yield a usable path

The remaining wins require changes inside the fork:
`github.com/beshkenadze/mlx-audio-swift` branch `upgrade/swift-lm-v3`.

## Goal

Reduce MLX Parakeet latency enough to make it a credible privacy-first /
non-ANE backend on the target benchmark environment.

This spec is intentionally conservative. It prioritizes the highest-
confidence speedup first, keeps protocol churn out of the first milestone,
and treats `compile(shapeless: true)` as a targeted experiment rather than
as the foundation of the plan.

## Non-goals

- CoreML / ANE work
- Changes to `mlx-swift` core
- Immediate changes to shared STT protocols unless a later milestone proves
  they are necessary
- Full vectorization of the TDT/RNNT decode loop in milestone 1
- Batched long-form transcription that preserves the full current
  chunk/overlap-merge behavior of `generate(audio:)`

## What the current code already tells us

The live Parakeet path in `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
already contains the first half of the desired batching strategy:

- the decode methods accept batch-shaped features
- the encoder can run on the batch in one pass
- the current implementation then drops into per-row decode loops with
  scalar `.item()` reads

That means the highest-confidence first seam is not “vectorize the entire
decoder.” It is:

1. extract mel per audio
2. pad/stack mel to `[B, T, F]`
3. run one encoder pass
4. decode each encoded row with existing logic

This is consistent with the nearby Python `mlx-audio` repo, where Parakeet
batches encoder input but still decodes each item in Python.

## Benchmark contract

All milestone decisions and throughput claims in this spec use one
benchmark contract.

### Required benchmark dimensions

- target hardware: M1 Max, FP16
- target checkpoint: `mlx-community/parakeet-tdt-0.6b-v3`
- workload manifest: fixed corpus file list with hashes
- chunking policy: fixed chunk duration and overlap policy for the baseline
- warm-up policy: explicit warm-up run count before measured runs
- measured runs: explicit run count and aggregation rule
- batch policy: report both item-count batching (`B=2/4/8`) and the
  corresponding per-batch max mel/frame lengths
- memory method: explicit RSS capture method outside `STTOutput`

### Required reporting rule

Every reported speedup in this spec must name:

1. benchmark contract version
2. checkpoint
3. batch size
4. warm-up count
5. measured run count
6. aggregate statistic used for comparison

If these fields are missing, the number is informational only and does not
count toward milestone acceptance.

---

## Milestone 0 — parity harness and profiling seam  [required first]

**Purpose:** remove ambiguity before optimization work starts.

### Deliverables

1. A small parity/perf harness in the fork that can run:
   - single-item `generate(audio:)`
   - candidate batched path on the same audio set
2. Measurement capture for:
   - wall-clock
   - peak RSS
   - transcript parity under greedy settings
3. A profiling note that estimates where current wall time is spent:
   - mel extraction
   - encoder
   - TDT/RNNT decode loop
4. A go / no-go recommendation for milestone 1 based on the benchmark
   contract and profiling output.

### Why first

The current spec overstates confidence in the final speedup. Before writing
new APIs, we need a stable way to prove whether encoder batching is enough
to move the number materially, or whether decode remains dominant.

### Acceptance criteria

1. Harness runs on a fixed held-out corpus and fixed chunk sizes.
2. Greedy transcript output is deterministic for repeat runs.
3. Baseline numbers are recorded for `B=1` and the current serial chunk
   path.
4. Instrumentation records separate timing spans for mel extraction,
   encoder, and decode.
5. Milestone 1 proceeds only if milestone 0 shows that the proposed batch
   seam is likely to produce a meaningful improvement on the benchmark
   contract.

---

## Milestone 1 — additive `generateBatch` on `ParakeetModel`  [primary]

**Expected outcome:** the highest-confidence throughput win, with the first
version limited to batched encoder execution and per-row decode.

**Target improvement:**
- target: **≥1.5×–2×** wall-clock improvement on concurrent chunk workloads
- upside beyond that is possible, but not promised until profiling proves
  decode is not the dominant cost

### Public API

Add an additive, model-specific API:

```swift
public func generateBatch(
    audios: [MLXArray],
    generationParameters: STTGenerateParameters
) -> [STTOutput]
```

This API lives on `MLXAudioSTT.ParakeetModel` only.

The shared `STTGenerationModel` protocol remains unchanged in this
milestone. That avoids unnecessary churn while we validate the design.

### Scope contract for version 1

Milestone 1 `generateBatch(...)` accepts independent, already chunk-sized
audio inputs only. It does not replace the full long-form behavior of
`generate(audio:)`, which currently owns chunking, overlap, token merge,
and final output assembly.

Long-form batched transcription is a separate problem and is out of scope
for this milestone.

### Variant scope for version 1

Milestone 1 is required only for the target checkpoint and variant used in
the benchmark contract:

- `mlx-community/parakeet-tdt-0.6b-v3`
- TDT path

Support for `.tdtCtc`, `.rnnt`, and `.ctc` is follow-up work unless later
explicitly added to this spec.

### First implementation boundary

Version 1 should do the following:

1. Accept `[MLXArray]` input.
2. Convert each input to mono and compute mel features independently.
3. Pad/stack features to a common `[B, T, F]` tensor.
4. Run the encoder once on the batch.
5. Decode each encoded row with the existing per-row TDT/RNNT/CTC logic.
6. Preserve input order in output order.

### Downstream adoption seam

Milestone 1 downstream adoption happens through explicit `ParakeetModel`
specialization at the call site or provider boundary. The shared
`STTGenerationModel` protocol remains unchanged.

This means the spec allows a concrete-type branch, capability probe, or
adapter at the integration boundary, but does not require shared protocol
changes in milestone 1.

### Explicit non-requirements for version 1

Version 1 does **not** require:

- full vectorization of the TDT loop across batch rows
- a new shared batch protocol
- provider-level API changes outside the fork
- fancy state compaction or gather/scatter kernels

Those may come later if profiling shows they matter.

### Why this shape

This matches both the current Swift code boundary and the nearest Python
analog:

- current Swift Parakeet already batches encoder input internally
- Python `mlx-audio` Parakeet also batches encoder input while looping per
  item in decode
- the stronger reusable public API pattern comes from `mlx-audio`
  `cohere_asr`, which keeps batching explicit and output order stable

### Acceptance criteria

1. `generateBatch([a, b, c])` returns three outputs in the same order.
2. Under greedy settings, `output[i].text` matches the result of
   `generate(audio: audios[i])` for the same checkpoint and parameters.
3. Existing `generate(audio:)` text output remains unchanged for current
   callers on the target checkpoint.
4. On M1 Max, FP16, fixed chunk corpus:
    - `B=2`, `B=4`, and `B=8` are benchmarked
    - at least one batch size reaches **≥1.5×** wall-clock improvement over
      the serial baseline
5. Peak RSS is reported for each tested batch size using the benchmark
   contract’s external RSS measurement method.
6. Each benchmark result reports the associated max mel/frame length inside
   the batch.
7. A configurable `batchCap` exists in the benchmark harness or Parakeet-
   specific generation settings.
8. If memory exceeds the configured cap, the implementation must fail fast
   or fall back according to documented harness behavior; it must not crash
   unpredictably.

### Risks

- decode may still dominate wall time after encoder batching
- mel padding may increase peak memory more than expected on long chunks
- parity bugs may show up around lengths, masking, or segment timing
- mixed-length batches may erase gains if frame-length bucketing is poor

---

## Milestone 2 — targeted compile experiments  [secondary]

**Purpose:** pursue incremental gains only after milestone 1 is stable.

### Scope

Investigate `compile(shapeless: true)` only on pure, shape-stable subgraphs.
Likely candidates:

- encoder path
- joint projection if it can be isolated cleanly
- simple tensor kernels that avoid control-flow-heavy decode logic
- explicit sync-cleanup only where profiling proves avoidable readbacks or
  eval barriers exist outside the dynamic decode loop

### Non-goal

Do not make decoder-loop compile success a dependency for shipping
milestone 1.

### Why secondary

Official MLX guidance supports `compile(shapeless: true)`, but also warns
about shape-dependent logic and control flow. Our current repo uses compile
only on simple kernels, not on recurrent decode loops. The Parakeet hot
path is therefore a lower-confidence compile target than batched encoder
execution.

### Acceptance criteria

1. Compile stays optional and correctness-neutral.
2. No transcript regression versus the non-compiled path.
3. Any claimed gain is measured separately from milestone 1 using the same
   benchmark contract.
4. Any claimed steady-state win names the exact compiled subgraph.
5. If compile does not provide a measurable steady-state win on the
   benchmark contract, it does not block the milestone.

### Risks

- negligible gains if decode control flow dominates
- compile instability on shape-sensitive or loop-heavy code
- time lost chasing compile issues before the batch seam is validated

---

## Optional follow-up — decode-loop vectorization

Only pursue this if milestone 1 proves that encoder batching helps but
decode remains the next dominant bottleneck.

Possible areas:

- remove or reduce scalar `.item()` reads
- keep more per-row state on-device
- batch portions of joint/duration/token updates
- add state compaction or gather/scatter only if profiling proves value

This is a separate optimization phase, not part of the initial spec.

---

## Workflow

1. Branch from `upgrade/swift-lm-v3`.
2. Add milestone 0 parity/perf harness.
3. Implement milestone 1 as an additive `ParakeetModel.generateBatch(...)`.
4. Benchmark fixed corpus at `B=2/4/8`; record wall-clock and RSS.
5. Repoint downstream consumer to the fork branch only after milestone 1 is
   stable and the concrete adoption seam is implemented.
6. Run the end-to-end corpus benchmark and compare against the current
   55.5 s baseline.
7. Attempt milestone 2 only if milestone 1 is correct and clearly useful.

## Definition of done

- fork branch contains a stable additive `generateBatch(...)` on
  `ParakeetModel`
- existing single-audio callers remain unchanged
- fixed-corpus benchmark shows a real throughput improvement versus the
  current serial path
- transcript parity is preserved under greedy settings
- benchmark notes include wall-clock and peak RSS by batch size
- benchmark notes include corpus identity, checkpoint, warm-up policy,
  measured run count, and per-batch max frame lengths
- downstream integration can adopt the new API without requiring shared STT
  protocol churn

## When to pick this up

Only pick this up if at least one of the following is true:

- users complain about MLX Parakeet speed specifically
- we decide to ship a privacy-first “no ANE” path and need MLX to be
  competitive
- the ANE/CoreML path regresses or becomes unavailable

Until then, ANE remains the default path, MLX Parakeet remains the stable
fallback path, and this spec stays parked.
