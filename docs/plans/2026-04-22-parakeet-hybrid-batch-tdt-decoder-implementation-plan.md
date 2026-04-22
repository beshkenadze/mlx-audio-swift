# Parakeet Hybrid Batch-Aware TDT Decoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace row-serial TDT decode in Parakeet with a hybrid active-row batched predictor+joint path while keeping CPU-owned control flow and preserving correctness against the current decoder.

**Architecture:** Keep encoder batching exactly as it is today. Change only the `decodeTDT(...)` phase after `encoder(features, lengths:)`, introducing a batch-safe predictor overload, active-row frame/state gathers, and old/new parity tests using a true TDT fixture. Do not touch `STTGenerationModel`, RNNT, or CTC in this plan.

**Tech Stack:** Swift Package Manager, MLX / MLXNN, Swift Testing in `Tests/`, `xcodebuild test`, `swift run mlx-audio-swift-stt`

---

## Before You Start

- Read: `docs/2026-04-22-parakeet-hybrid-batch-tdt-decoder-spec.md`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetRNNTLayers.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetAttention.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetDecodingLogic.swift`
- Read: `Tests/MLXAudioSTTTests.swift`
- Read: `Tests/ParakeetBatchParityTests.swift`
- Read: `Sources/Tools/mlx-audio-swift-stt/App.swift`

## Ground Rules

- Do **not** change `STTGenerationModel`.
- Do **not** change `generate(audio:)` or `generateStream(audio:)` public signatures.
- Do **not** change `decodeRNNT(...)` or `decodeCTC(...)` in this plan.
- Do **not** implement label-looping v2 in this plan.
- Do **not** make device-side output buffers or scatter writes a prerequisite for v1.
- Keep the old row-serial `decodeTDT(...)` reachable behind a local seam until parity and benchmark gates pass.

---

### Task 1: Add a real tiny TDT fixture and trace oracle scaffold

**Files:**
- Modify: `Tests/ParakeetBatchParityTests.swift`
- Read for reference: `Tests/MLXAudioSTTTests.swift`
- Optional new fixture path: `Tests/media/parakeet-tdt-fixture/`

**Step 1: Write the failing TDT fixture test skeleton**

Add a new serialized test suite section that explicitly targets TDT, not CTC:

```swift
@Test("TDT fixture can decode batched inputs")
func tdtFixtureCanDecodeBatchedInputs() throws {
    #expect(Bool(false), "Implement TDT fixture")
}
```

**Step 2: Run it to verify failure**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests/tdtFixtureCanDecodeBatchedInputs()' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because the TDT fixture does not exist yet.

**Step 3: Add a tiny deterministic TDT fixture builder**

Extend `Tests/ParakeetBatchParityTests.swift` with a new helper, e.g.:

```swift
private func makeTDTFixtureModel() throws -> ParakeetModel
```

Requirements for the fixture:
- target must resolve to `.tdt`
- include `decoder`
- include `joint`
- include `decoding.model_type = "tdt"`
- include deterministic `durations`
- include enough weights for:
  - pre-encode
  - decoder embed/LSTM
  - joint projections

If the fixture becomes too unwieldy inline, move the generated files into:

```text
Tests/media/parakeet-tdt-fixture/
```

**Step 4: Add trace capture scaffolding**

Add a local struct in the test file, for example:

```swift
struct TDTTraceStep: Equatable {
    let row: Int
    let time: Int
    let newSymbols: Int
    let token: Int
    let decisionIndex: Int
    let committedState: Bool
}
```

and prepare test-only hooks so the old and new `decodeTDT(...)` paths can emit comparable traces.

**Step 5: Re-run the focused test**

Run the same `xcodebuild test` command.

Expected: FAIL later, now because trace hooks or the new batched path are still missing.

**Step 6: Commit**

```bash
git add Tests/ParakeetBatchParityTests.swift Tests/media/parakeet-tdt-fixture
git commit -m "test: add Parakeet TDT fixture scaffold"
```

---

### Task 2: Add a batch-safe predictor overload

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetRNNTLayers.swift`
- Test: `Tests/ParakeetBatchParityTests.swift`

**Step 1: Write a failing predictor test**

Add a focused test that proves the predictor can process:
- batched token ids `[B, 1]`
- batched hidden/cell `[L, B, H]`
- blank-token masking without `token == nil`

Skeleton:

```swift
@Test("Parakeet predictor accepts batched blank-masked token input")
func predictorAcceptsBatchedTokenInput() throws {
    #expect(Bool(false), "Implement predictor overload")
}
```

**Step 2: Run to verify failure**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests/predictorAcceptsBatchedTokenInput()' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because the overload does not exist yet.

**Step 3: Add the overload**

In `ParakeetRNNTLayers.swift`, add a new overload next to the current one:

```swift
func callAsFunction(
    _ tokenIds: MLXArray,
    state: ParakeetLSTMState,
    blankToken: Int32
) -> (MLXArray, ParakeetLSTMState)
```

Implementation requirements:
- embed batched token ids
- compute `blankMask = (tokenIds .== MLXArray(blankToken)).expandedDimensions(axis: 2)`
- zero out blank embeddings with `MLX.where`
- pass the masked embedding into `prediction.decRnn`

Do **not** remove the old optional-token overload in this task.

**Step 4: Run focused tests**

Run the same `xcodebuild test` command.

Expected: PASS for the predictor-specific test.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetRNNTLayers.swift Tests/ParakeetBatchParityTests.swift
git commit -m "feat: add batch-safe Parakeet predictor overload"
```

---

### Task 3: Add a rollout seam for old vs new `decodeTDT(...)`

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Test: `Tests/ParakeetBatchParityTests.swift`

**Step 1: Write a failing trace-parity test**

Add a test that runs the same TDT fixture through:
- old row-serial decoder
- new hybrid decoder

and compares trace steps.

Skeleton:

```swift
@Test("old and new decodeTDT traces match on TDT fixture")
func oldAndNewDecodeTDTTracesMatch() throws {
    #expect(Bool(false), "Add rollout seam and trace comparison")
}
```

**Step 2: Run to verify failure**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests/oldAndNewDecodeTDTTracesMatch()' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because no rollout seam exists yet.

**Step 3: Introduce a temporary local rollout seam**

In `ParakeetModel.swift`, split today’s `decodeTDT(...)` into:

```swift
private func decodeTDTSerial(...)
private func decodeTDTHybrid(...)
```

and make `decodeTDT(...)` route through one of them using a local test seam or switch.

Requirements:
- default path remains current serial path until parity is proven
- tests can force old/new execution deterministically
- no public API change required

**Step 4: Re-run trace-parity test**

Run the same `xcodebuild test` command.

Expected: FAIL later because hybrid implementation is not complete yet, but routing now exists.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "refactor: add rollout seam for Parakeet TDT decoder"
```

---

### Task 4: Add active-row gather helpers in `ParakeetModel`

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetRNNTLayers.swift`

**Step 1: Write a failing helper-level test**

Use the TDT fixture test file to assert that gathered active rows produce stable batched shapes for:
- frames `[A, 1, dModel]`
- hidden `[L, A, H]`
- cell `[L, A, H]`

**Step 2: Run to verify failure**

Run `ParakeetBatchParityTests`.

Expected: FAIL because helper functions do not exist yet.

**Step 3: Add private helpers**

In `ParakeetModel.swift`, add private helpers near the existing helper cluster:

```swift
private func gatherActiveFrames(...)
private func gatherActiveState(...)
private func mergeUpdatedState(...)
```

Implementation rules:
- v1 may use Swift loops + `MLX.concatenated(...)`
- do not introduce `putAlong` yet unless the simple path is obviously broken
- keep shapes explicit in comments/assertions

**Step 4: Re-run the helper test**

Run the same focused test command.

Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "feat: add active-row gather helpers for Parakeet TDT"
```

---

### Task 5: Replace row-serial predictor+joint work in `decodeTDT(...)`

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Test: `Tests/ParakeetBatchParityTests.swift`

**Step 1: Write the failing parity test for mixed-length TDT batch**

Add a test that runs a mixed-length TDT batch and compares:
- row ordering
- final text
- segment count/timing if applicable
- trace equality against the old serial path

**Step 2: Run to verify failure**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL until `decodeTDTHybrid(...)` is implemented.

**Step 3: Implement the hybrid loop**

In `decodeTDTHybrid(...)`:
- keep `timeByRow`, `newSymbolsByRow`, `lastTokenByRow`, `doneByRow`, `hypothesisByRow` on CPU
- gather active rows each outer iteration
- gather frames `[A, 1, dModel]`
- gather active hidden/cell `[L, A, H]`
- call the new batch-safe predictor once
- call joint once
- compute token argmax and duration argmax for `[A]`
- move only those `[A]` vectors to CPU
- update `ParakeetDecodingLogic.tdtStep(...)` row by row on CPU
- commit hidden/cell only for rows that emitted non-blank tokens

Do **not** tensorize output buffers in this task.

**Step 4: Re-run parity tests**

Run the same `xcodebuild test ... -only-testing:'MLXAudioTests/ParakeetBatchParityTests'` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "feat: batch Parakeet TDT predictor and joint evaluation"
```

---

### Task 6: Re-run existing Parakeet suite and keep regressions visible

**Files:**
- Modify only if needed: `Tests/MLXAudioSTTTests.swift`
- Verify: `Tests/ParakeetBatchParityTests.swift`

**Step 1: Run the existing Parakeet suite**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetSTTTests' CODE_SIGNING_ALLOWED=NO
```

Expected: PASS.

**Step 2: If needed, add or tighten low-level tests**

Only if regressions surface, extend `Tests/MLXAudioSTTTests.swift` around:
- `deterministicRNNTAndTDTControlFlow()`
- config parsing
- tokenizer/alignment behavior

Do not expand scope casually.

**Step 3: Commit only if extra tests were needed**

```bash
git add Tests/MLXAudioSTTTests.swift Tests/ParakeetBatchParityTests.swift
git commit -m "test: strengthen Parakeet TDT regression coverage"
```

---

### Task 7: Benchmark the hybrid decoder against the current baseline

**Files:**
- Modify: `Tests/ParakeetBatchParityTests.swift` or a dedicated local benchmark helper in the same file
- Read: `docs/2026-04-22-parakeet-hybrid-batch-tdt-decoder-spec.md`

**Step 1: Add a benchmark reporting test/harness step**

Ensure the benchmark output includes:
- batch size
- wall-clock
- per-chunk wall-clock
- max frame length in batch
- padding ratio
- active-row occupancy if available

**Step 2: Run the benchmark harness**

Use the same `ParakeetBatchParityTests` command or a targeted test name.

Expected: benchmark metadata prints/records cleanly.

**Step 3: Compare against the current batch-encoder + row-serial-TDT baseline**

Do not compare against pre-batch history. Compare against the current branch’s existing decode behavior.

**Step 4: Apply the gate**

If the hybrid decoder does not produce a meaningful improvement, stop and record that result before any device-side output buffer work.

**Step 5: Commit benchmark result plumbing if it changed code**

```bash
git add Tests/ParakeetBatchParityTests.swift docs/2026-04-22-parakeet-hybrid-batch-tdt-decoder-spec.md
git commit -m "test: record Parakeet hybrid TDT benchmark contract"
```

---

## Verification Checklist

Run these before calling the plan implemented:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetSTTTests' CODE_SIGNING_ALLOWED=NO
```

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

Optional CLI sanity check (only if a rollout seam touches the CLI later):

```bash
swift run mlx-audio-swift-stt --audio /path/to/audio.wav --output-path /tmp/transcript
```

## Notes for the Implementer

- Do not spend time on device-side output buffers until hybrid v1 proves a real gain.
- The current batch parity tests are useful, but they are not enough until a true TDT fixture exists.
- The old serial `decodeTDT(...)` path is not dead code during this work; it is the oracle.
- The rel-pos attention batch fix is already landed. Preserve it with tests; do not redesign it here.

---

Plan complete and saved to `docs/plans/2026-04-22-parakeet-hybrid-batch-tdt-decoder-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
