# Spec: Parakeet Hybrid Batch-Aware TDT Decoder

**Status:** proposed  
**Owner:** TBD  
**Priority:** P1 for the MLX batch path  
**Created:** 2026-04-22  
**Last updated:** 2026-04-22

## Context

`generateBatch(...)` now exists on `ParakeetModel`, but the current TDT
decoder still leaves most decode cost on the table.

Current state on `feat/parakeet-batch-generate`:

- batch mel assembly is implemented in `ParakeetModel.makeBatchFeatures(...)`
- encoder execution is already batched through `encoder(features, lengths:)`
- the rel-pos batch crash was fixed by broadcasting positional projections
  across batch in `ParakeetRelPositionMultiHeadAttention`
- `decodeTDT(...)` still falls back to row-serial decode with:
  - `for b in 0..<batchSize`
  - per-row `while t < maxLength`
  - per-row `.item()` reads for token and duration

That means throughput still scales too close to linearly with batch size in
real runs. Encoder batching alone is not enough because TDT decode remains a
set of `B` independent greedy decoders.

## Goal

Replace the current row-serial TDT decode loop with a **hybrid batch-aware
decoder** that batches the expensive predictor and joint work across active
rows while keeping control flow on the CPU.

The aim is to capture most of the available decode-side win without jumping
immediately to a full device-side masked state machine.

## Current baseline

This spec starts **after** the existing batch encoder seam.

Already true in the current branch:

- `generateBatch(...)` exists
- `makeBatchFeatures(...)` already produces batched mel input
- `encoder(features, lengths:)` already runs once on the full batch
- the rel-pos batch crash was fixed and should now be treated only as a
  regression requirement

This spec changes only the part of `decodeTDT(...)` that starts after:

```swift
let encoded = encoder(features, lengths: lengths)
```

and currently falls into:

```swift
for b in 0..<batchSize { ... }
```

It does **not** redesign mel batching, long-form chunking, or shared STT
interfaces.

## Non-goals

- No shared `STTGenerationModel` protocol changes
- No label-looping v2 in the first implementation
- No device-side output buffer/scatter rewrite in the first implementation
- No attempt to fully tensorize `t`, `newSymbols`, and output assembly on day 1
- No changes to RNNT or CTC decode in this spec
- No attempt to re-solve the rel-pos batch issue beyond keeping a regression
  guard for the already-landed fix

## Codebase facts this design depends on

### Existing good seams

- `ParakeetModel.generateBatch(...)` already produces a real batched mel path
- `ParakeetConformer` already consumes `[B, T, F]` and produces `[B, T_sub, dModel]`
- `ParakeetStackedLSTM` already stores state as `[layers, batch, hidden]`
- `ParakeetJointNetwork` already accepts batched encoder/predictor tensors
- local `mlx-swift` supports:
  - `compile { [MLXArray] -> [MLXArray] }`
  - `takeAlong`
  - `putAlong`
  - `MLX.where`

### Current blockers

- `decodeTDT(...)` is row-serial
- `ParakeetPredictNetwork.callAsFunction(_ token: MLXArray?, state: ...)`
  still uses a `nil` token path for blank input
- current tests do not yet protect a true TDT decoder parity fixture

### Validation gap that must be closed first

Current `ParakeetBatchParityTests.swift` exercises batch API behavior and the
rel-pos batch regression, but does **not** yet provide an end-to-end TDT
decoder parity oracle. Existing TDT coverage in `MLXAudioSTTTests.swift` is
limited to config parsing and `ParakeetDecodingLogic.tdtStep(...)` unit tests.

That means a `decodeTDT(...)` rewrite is currently under-tested unless this
spec explicitly adds a true TDT fixture and old/new decoder comparison.

## Diagnosis

The next practical win is **not** a full tensorized decoder. The next
practical win is to remove the outer batch-row loop from `decodeTDT(...)`
while keeping the state machine logic readable and verifiable.

That means:

- keep `time`, `newSymbols`, `done`, and token assembly on the CPU
- batch only the expensive predictor + joint evaluation over active rows
- reduce `.item()` calls from `O(B)` per step to one batched host read of
  token/duration vectors per step

This is the minimum viable decoder vectorization step for this codebase.

---

## Proposed design: Hybrid v1

## Preconditions

Hybrid v1 must not start until both of these exist:

1. a tiny local **TDT** fixture checkpoint with decoder, joint, and durations
2. a trace-level parity harness for old vs new `decodeTDT(...)`

### High-level shape

For each outer decode iteration:

1. Build `activeRows` on the CPU.
2. Gather one encoder frame for each active row.
3. Build one batched token tensor `[A, 1]` for active rows.
4. Gather batched decoder state `[L, A, H]` from the full state.
5. Run predictor once for all active rows.
6. Run joint once for all active rows.
7. Compute token argmax and duration argmax for `[A]` rows in one shot.
8. Move only the resulting `[A]` token/duration vectors to the CPU.
9. Update `time`, `newSymbols`, `lastToken`, and output tokens in a small
   Swift loop.
10. Commit LSTM state only for rows that emitted non-blank tokens.

### Why this is the right first step

- it removes the main model-compute serialization point
- it does not require a full device-side output state machine
- it keeps `ParakeetDecodingLogic.tdtStep(...)` reusable as-is
- it is testable against the current sequential decoder

---

## Detailed data flow

### CPU-owned state

Keep the following on the host for v1:

- `timeByRow: [Int]`
- `newSymbolsByRow: [Int]`
- `lastTokenByRow: [Int]`
- `doneByRow: [Bool]`
- `hypothesisByRow: [[ParakeetAlignedToken]]`

These stay on CPU in v1 by design so the existing `ParakeetDecodingLogic.tdtStep(...)`
semantics remain visible and comparable during the rewrite.

### Device-owned state

Keep the following on device for v1:

- `hidden: [L, B, H]`
- `cell: [L, B, H]`
- encoder output `batchFeatures: [B, T_sub, dModel]`

### Predictor input contract

Add a new batch-safe predictor overload that always accepts a token tensor:

```swift
func callAsFunction(
    _ tokenIds: MLXArray,            // [B, 1], int32
    state: ParakeetLSTMState,
    blankToken: Int32
) -> (MLXArray, ParakeetLSTMState)
```

Blank handling must be mask-based, not `nil`-based.

This is required because the current `nil` path infers batch size from state
and is not safe for the first batched active-row step.

### Gather strategy

The first implementation may gather active rows with Swift loops and
`MLX.concatenated(...)`. That is acceptable in v1 because the model compute
after gather is what dominates.

### State merge strategy

The first implementation may merge updated state back with split/replace/
concatenate rather than `putAlong`. This is simpler and easier to verify.

---

## API changes

### Required

#### `ParakeetPredictNetwork`

Add a new overload for batch-safe blank masking. Keep the old overload during
the transition if RNNT still needs it.

#### `ParakeetModel.decodeTDT(...)`

Replace row-serial model compute with active-row batched predictor+joint
evaluation.

### Explicitly unchanged

- `STTGenerationModel`
- `generate(audio:)`
- `generateStream(audio:)`
- `decodeRNNT(...)`
- `decodeCTC(...)`

---

## Acceptance criteria

### Functional

1. `decodeTDT(...)` no longer performs predictor+joint work inside
   `for b in 0..<batchSize`.
2. A batch of chunk-sized TDT inputs preserves output order.
3. Under greedy settings, batched TDT output text matches sequential output
   text for the same inputs.
4. Blank-token handling is batch-safe and no longer depends on `token == nil`.

### Correctness oracle

5. A trace-level oracle compares old and new `decodeTDT(...)` on the same TDT
   fixture features and verifies, per row and per step:
   - row id
   - `time`
   - `newSymbols`
   - predicted token id
   - predicted duration index / jump
   - whether decoder state was committed

6. Final text parity alone is not sufficient. Acceptance must also verify:
   - final row ordering
   - sentence/segment count
   - segment start/end values, or a documented tolerance if float drift exists

### Testing

7. A true TDT fixture exists in tests; CTC-only fixtures are not sufficient.
8. Mixed-length TDT batch parity is covered.
9. Early-finish, blank-heavy, zero-duration, and `maxSymbols` fallback cases
   are explicitly covered.
10. The rel-pos batch regression remains covered.
11. Existing `ParakeetSTTTests` remain green.

### Performance

12. Real runs show a measurable wall-clock improvement over the current
    batch-encoder + row-serial-TDT baseline.
13. `.item()` usage inside the TDT hot path is reduced from per-row scalar
    reads to batched row-vector extraction.
14. Benchmark reporting separates:
    - mel time
    - encoder time
    - decoder total time
    - active-row predictor+joint time if instrumented

---

## Verification strategy

### Required new tests

Add or extend tests in `Tests/ParakeetBatchParityTests.swift` for:

- TDT fixture parity vs sequential decode
- mixed-length TDT batch with at least 8 rows
- blank-token first-step behavior in batch mode
- rel-pos attention batch safety
- trace-level decoder comparison between old and new `decodeTDT(...)`

### Required TDT fixture

The new fixture must be a tiny local TDT checkpoint, not a CTC checkpoint.
It must contain enough weights/config to execute:

- decoder
- joint
- duration decisions
- greedy TDT decode path

The fixture does not need to be realistic, but it must be deterministic and
must exercise `decodeTDT(...)` end to end.

### Required existing tests

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetSTTTests' CODE_SIGNING_ALLOWED=NO
```

### Benchmark confirmation

Benchmark on the same workload used to expose the no-speedup finding and
report:

- batch size
- wall-clock
- per-chunk wall-clock
- max frame length in batch
- padding ratio
- active-row occupancy by decode iteration if available

### Rollout safety

The old row-serial `decodeTDT(...)` path must remain available behind a local
switch or test seam until parity and benchmark gates pass.

Do not flip the new decoder path to default without:

1. TDT fixture trace parity
2. real-checkpoint corpus parity
3. decoder-isolated benchmark improvement

---

## Risks

### Major

- hidden/cell merge-back logic can be subtly wrong for rows that do not emit
- blank-token masking can accidentally perturb non-blank rows if batch shapes
  are wrong
- mixed-length rows can still reduce net gain if active-row gather/merge is
  too expensive
- old/new decoder divergence can hide behind identical final text if trace
  parity is not enforced

### Moderate

- Swift-side gather/merge may be simpler but not yet optimal
- host-side output assembly still limits the absolute upside
- per-step host synchronization remains if active rows are checked too often

### Deferred by design

- device-side output buffers
- fully tensorized `t / done / outLen`
- label-looping v2

---

## GAPs

These are known gaps that remain after this spec and must be acknowledged.

1. **No TDT fixture yet**  
   Current batch parity tests mostly rely on CTC-friendly fixture behavior.

2. **No trace-level decoder oracle yet**  
   We do not yet compare per-step old/new `decodeTDT(...)` behavior.

3. **No measured benchmark result for hybrid v1 yet**  
   We know why the current decoder is slow, but we have not yet measured the
   hybrid replacement.

4. **No RNNT parity decision yet**  
   This spec intentionally excludes RNNT until TDT is proven.

5. **No device-side output buffer design yet**  
   This is deferred intentionally, not solved.

6. **No label-looping v2 design contract yet**  
   It is explicitly out of scope for the first patch.

7. **No rollout toggle defined yet**  
   The exact mechanism for selecting old vs new `decodeTDT(...)` is not yet chosen.

---

## Recommended implementation order

1. Add TDT-specific fixture and trace-level regression tests.
2. Add batch-safe predictor overload.
3. Introduce a temporary rollout seam for old vs new `decodeTDT(...)`.
4. Replace row-serial predictor+joint work in `decodeTDT(...)` with active-row
   batched evaluation.
5. Re-run trace parity and final-output parity tests.
6. Benchmark.
7. Decide whether further tensorization is warranted.

## Out of scope for this document

- full device-side decoder state machine
- label-looping TDT
- batch-aware RNNT rewrite
- protocol/API redesign above `ParakeetModel`
