# Parakeet Batch Generate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Parakeet-only `generateBatch(...)` API plus a parity/perf harness that proves whether batched encoder execution improves throughput on the target benchmark contract.

**Architecture:** Keep the shared `STTGenerationModel` protocol unchanged. Implement batching only on `ParakeetModel` by reusing the existing internal seam that already accepts batch-shaped mel tensors in `decode(mel:)`, while keeping version 1 scoped to independent chunk-sized audios. Add a dedicated parity/perf harness in tests, then optionally add a concrete-type adoption seam in the STT CLI after the batch path is stable.

**Tech Stack:** Swift Package Manager, MLX / MLXNN, XCTest-style Swift Testing suites in `Tests/`, `xcodebuild test`, `swift run mlx-audio-swift-stt`

---

## Before You Start

- Read: `docs/2026-04-21-mlx-audio-swift-fork-task.md`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetAudio.swift`
- Read: `Sources/MLXAudioSTT/Generation.swift`
- Read: `Sources/MLXAudioSTT/Models/GLMASR/STTOutput.swift`
- Read: `Tests/MLXAudioSTTTests.swift`
- Read: `Sources/Tools/mlx-audio-swift-stt/App.swift`
- Read: `Sources/Tools/mlx-audio-swift-stt/README.md`

## Ground Rules

- Do **not** change `STTGenerationModel` in this plan.
- Do **not** attempt long-form batched transcription in v1.
- Do **not** begin with `compile(shapeless: true)` work.
- Keep v1 limited to independent, already chunk-sized audio inputs.
- Measure wall-clock and RSS outside `STTOutput`; current Parakeet output is not a source of truth for those metrics.

---

### Task 1: Add a dedicated batch parity/perf harness file

**Files:**
- Create: `Tests/ParakeetBatchParityTests.swift`
- Read for reference: `Tests/MLXAudioSTTTests.swift`
- Read for fixture patterns: `Tests/media/`

**Step 1: Write the failing test skeleton**

Create a new serialized suite with explicit placeholders for:
- single-item baseline
- batched candidate path
- text parity assertions
- benchmark metadata logging

```swift
import Testing
@testable import MLXAudioSTT
@testable import MLXAudioCore
import Foundation
import MLX

@Suite("Parakeet Batch Parity Tests", .serialized)
struct ParakeetBatchParityTests {
    @Test("generateBatch preserves order and text parity for chunk-sized audio")
    func generateBatchPreservesOrderAndTextParity() async throws {
        #expect(Bool(false), "Implement parity harness")
    }
}
```

**Step 2: Run the new test to verify it fails**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests/generateBatchPreservesOrderAndTextParity()' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because the harness is incomplete.

**Step 3: Add benchmark contract helpers before real assertions**

Add small local helpers in the new file for:
- loading the target fixture list
- truncating/rejecting non-chunk-sized audio
- measuring wall-clock around a closure
- recording batch size and max mel/frame length
- capturing RSS via a small local process-info helper

Keep these helpers inside the test file for v1.

**Step 4: Re-run the single test**

Run the same `xcodebuild test` command.

Expected: FAIL later, now because `generateBatch(...)` does not exist yet.

**Step 5: Commit**

```bash
git add Tests/ParakeetBatchParityTests.swift
git commit -m "test: add Parakeet batch parity harness skeleton"
```

---

### Task 2: Extract the reusable batch seam in `ParakeetModel`

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Read: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetAudio.swift`

**Step 1: Write a failing test for the model-specific API surface**

In `Tests/ParakeetBatchParityTests.swift`, add a test that tries to call:

```swift
let outputs = try model.generateBatch(
    audios: [audioA, audioB],
    generationParameters: STTGenerateParameters()
)

#expect(outputs.count == 2)
```

**Step 2: Run the test to verify it fails**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL with missing `generateBatch` symbol.

**Step 3: Add the public API and internal helper seam**

In `ParakeetModel.swift`, add:

```swift
public func generateBatch(
    audios: [MLXArray],
    generationParameters: STTGenerateParameters = STTGenerateParameters()
) throws -> [STTOutput] {
    // validate non-empty input
    // preprocess each audio to mel independently
    // pad/stack mel to [B, T, F]
    // call shared batched decode path
    // map outputs back in input order
}
```

Add private helpers as needed, for example:

```swift
private func makeBatchFeatures(_ audios: [MLXArray]) throws -> (features: MLXArray, lengths: [Int])
private func decodeBatchFeatures(_ features: MLXArray) throws -> [STTOutput]
```

Do **not** change the semantics of `generate(audio:)`.

**Step 4: Keep version 1 behavior aligned with the spec**

Inside `generateBatch(...)`:
- reject empty input with a clear error
- assume each audio is already chunk-sized
- do mono conversion and mel extraction per input
- pad to a common time dimension
- call existing `decode(mel:)`
- return one `STTOutput` per input in the same order

Do **not** add long-form chunking/overlap-merge here.

**Step 5: Run the focused test suite**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

Expected: parity test still fails if text/order logic is incomplete, but API now compiles.

**Step 6: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "feat: add Parakeet batch generate API"
```

---

### Task 3: Reuse current decode paths without changing shared protocol boundaries

**Files:**
- Modify: `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`
- Read: `Sources/MLXAudioSTT/Generation.swift`

**Step 1: Write a failing regression test for single-audio behavior**

Extend `Tests/ParakeetBatchParityTests.swift` or `Tests/MLXAudioSTTTests.swift` with a test that confirms:

```swift
let single = try model.generate(audio: audioA)
let batched = try model.generateBatch(audios: [audioA])

#expect(single.text == batched[0].text)
```

**Step 2: Run the test to verify current failure**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL until output mapping and shared decode path are correct.

**Step 3: Refactor to share logic, not protocol**

In `ParakeetModel.swift`:
- leave `STTGenerationModel` unchanged
- leave `generate(audio:)` public signature unchanged
- reuse a new lower-level helper so both single-item and batched paths can reach the same decode code cleanly where appropriate

The key rule: no change to `Sources/MLXAudioSTT/Generation.swift` in v1.

**Step 4: Run the Parakeet-focused existing suite**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:MLXAudioTests/ParakeetSTTTests CODE_SIGNING_ALLOWED=NO
```

Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "refactor: share Parakeet decode seam for batch generation"
```

---

### Task 4: Define the benchmark harness output and metrics contract

**Files:**
- Modify: `Tests/ParakeetBatchParityTests.swift`
- Read: `Sources/MLXAudioSTT/Models/GLMASR/STTOutput.swift`

**Step 1: Add a failing test around benchmark metadata completeness**

Add assertions that the harness prints or records:
- checkpoint
- batch size
- warm-up count
- measured run count
- aggregate statistic
- max mel/frame length per batch
- RSS measurement

**Step 2: Run it to verify failure**

Run the same `ParakeetBatchParityTests` command.

Expected: FAIL until the harness reports everything.

**Step 3: Implement simple local benchmark structs in the test file**

Example:

```swift
struct BatchBenchmarkResult {
    let checkpoint: String
    let batchSize: Int
    let warmupRuns: Int
    let measuredRuns: Int
    let medianWallClock: TimeInterval
    let peakRSSBytes: UInt64
    let maxFrameLength: Int
}
```

Do not overload `STTOutput` for this. Keep benchmark reporting external.

**Step 4: Re-run tests**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

Expected: PASS for benchmark metadata assertions.

**Step 5: Commit**

```bash
git add Tests/ParakeetBatchParityTests.swift
git commit -m "test: record Parakeet batch benchmark contract metadata"
```

---

### Task 5: Add mixed-length and edge-case coverage

**Files:**
- Modify: `Tests/ParakeetBatchParityTests.swift`
- Read: `Tests/MLXAudioSTTTests.swift`

**Step 1: Add failing tests for edge cases**

Add tests for:
- empty `audios: []`
- single-item batch
- mixed short and longer chunk-sized inputs
- multichannel input collapsed to mono before mel extraction

Example skeleton:

```swift
@Test("generateBatch rejects empty input")
func generateBatchRejectsEmptyInput() async throws {
    #expect(throws: Error.self) {
        _ = try model.generateBatch(audios: [])
    }
}
```

**Step 2: Run the suite and confirm failures**

Run the `ParakeetBatchParityTests` command.

Expected: FAIL until edge handling is implemented.

**Step 3: Implement the minimal behavior in `ParakeetModel.swift`**

Implement only the behavior the tests require:
- explicit empty-input rejection
- preserve one-item batch parity
- normalize per-input audio before mel extraction

**Step 4: Re-run tests**

Run the same `xcodebuild test` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift Tests/ParakeetBatchParityTests.swift
git commit -m "test: cover Parakeet batch edge cases"
```

---

### Task 6: Add an optional CLI adoption seam without protocol churn

**Files:**
- Modify: `Sources/Tools/mlx-audio-swift-stt/App.swift`
- Read: `Sources/MLXAudioSTT/Generation.swift`
- Read: `Sources/Tools/mlx-audio-swift-stt/README.md`

**Step 1: Write the failing integration shape as comments or TODO-backed branch**

Plan for a concrete-type adoption seam such as:

```swift
if let parakeet = model as? ParakeetModel {
    // explicit batch path here
} else {
    // existing single-audio STTGenerationModel path
}
```

If no CLI change is needed in this milestone, document that and skip code edits.

**Step 2: If implementing now, add the smallest non-breaking branch**

Only do this if milestone 1 is already stable. Do not change the shared protocol.

**Step 3: Verify the CLI still builds/tests**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

Expected: PASS.

**Step 4: Optionally smoke the CLI command shape**

Run:

```bash
swift run mlx-audio-swift-stt --audio /path/to/audio.wav --output-path /tmp/transcript
```

Expected: command still parses and uses the existing STT path without protocol regressions.

**Step 5: Commit**

```bash
git add Sources/Tools/mlx-audio-swift-stt/App.swift Sources/Tools/mlx-audio-swift-stt/README.md
git commit -m "feat: add Parakeet-specific batch adoption seam"
```

---

### Task 7: Record milestone-0 decision and stop if the data says stop

**Files:**
- Modify: `docs/2026-04-21-mlx-audio-swift-fork-task.md` only if outcomes require clarifying the spec
- Modify: `Tests/ParakeetBatchParityTests.swift`

**Step 1: Run the full Parakeet-focused verification set**

Run:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:MLXAudioTests/ParakeetSTTTests -only-testing:MLXAudioTests/ParakeetBatchParityTests CODE_SIGNING_ALLOWED=NO
```

Expected: PASS.

**Step 2: Capture milestone-0 benchmark results**

Use the new harness to record:
- `B=1`
- `B=2`
- `B=4`
- `B=8`

Capture:
- wall-clock
- peak RSS
- max frame length per batch
- text parity status

**Step 3: Apply the decision gate from the spec**

If batching does **not** produce a meaningful improvement on the benchmark contract, stop and open a decode-focused follow-up instead of pushing forward into compile work.

**Step 4: Only if milestone 1 is clearly successful, queue milestone 2**

Milestone 2 is limited to:
- pure, shape-stable subgraphs
- measured compile experiments
- no dependency on decoder-loop compile success

**Step 5: Commit**

```bash
git add Tests/ParakeetBatchParityTests.swift docs/2026-04-21-mlx-audio-swift-fork-task.md
git commit -m "docs: record Parakeet batch milestone results"
```

---

## Verification Checklist

Run these before calling the plan implemented:

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:MLXAudioTests/ParakeetSTTTests CODE_SIGNING_ALLOWED=NO
```

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:'MLXAudioTests/ParakeetBatchParityTests' CODE_SIGNING_ALLOWED=NO
```

```bash
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

Optional CLI smoke command:

```bash
swift run mlx-audio-swift-stt --audio /path/to/audio.wav --output-path /tmp/transcript
```

## Notes for the Implementer

- The nearest internal implementation seam is **not** the public protocol; it is the existing batched mel decode path inside `ParakeetModel.decode(mel:)` and its variant-specific decode methods.
- The nearest external API shape reference is `mlx-audio` Python `cohere_asr`, but you should copy only the design idea (explicit batching and stable order), not its whole structure.
- Do not let milestone 2 leak into milestone 1. If the batch seam does not move the number enough, record that result honestly.

---

Plan complete and saved to `docs/plans/2026-04-21-parakeet-batch-generate-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
