# Fast Model Loading Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Whisper model loading time from ~12s to ~4-5s through quantized model support and parallel loading optimizations.

**Architecture:** Load pre-quantized 4-bit models from HuggingFace (400MB vs 1.6GB), parallelize encoder/decoder parameter updates, and support background eval for improved UX.

**Tech Stack:** MLX-Swift, HuggingFace Hub, Swift Concurrency

---

## Expert Panel Review Applied

This design was reviewed by spec experts (Wiegers, Fowler, Nygard, Crispin) and updated:

| Issue | Fix Applied |
|-------|-------------|
| Background eval failure would hang forever | Added do-catch in Task.detached, state transitions to `.failed(error)` |
| `readyStream` referenced but not declared | Added proper stream/continuation initialization in init |
| Accuracy test used exact match (would fail) | Changed to WER threshold (<2%) comparison |
| Silent fallback to float16 | Made explicit via `LoadResult.didFallback` |
| No timeout for `waitUntilReady()` | Added configurable timeout with racing task pattern |
| Missing hardware requirements | Added hardware matrix in Success Criteria |

## Codex CLI Review Applied

Additional issues found by Codex (gpt-5.2) review:

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| API inconsistency: `fromPretrained` hides `didFallback` | High | Expose `actualQuantization` on `WhisperSession` |
| `ManagedAtomic<LoadingState>` won't compile (Error not AtomicValue) | High | Use `OSAllocatedUnfairLock` for state management |
| No cancellation if session dropped during background eval | Medium | Add `backgroundTask` handle + `deinit` cancellation |
| Timeout race: may throw even if ready | Medium | Recheck state after timeout, return `.ready` if true |
| Thread safety of parallel updates unverified | Medium | Document MLX-Swift assumption + add serial fallback option |
| No tests for timeout/multi-waiter/fallback | Medium | Added comprehensive test cases |
| Integration tests require network | Medium | Added `MockModelLoader` protocol for CI |

---

## Success Criteria

- Load `large-v3-turbo` in <5s on M1 Max (currently ~12s)
- Maintain API compatibility (no breaking changes)
- Optional quantization (users can choose precision)
- Quantized transcription WER within 2% of float16 baseline

### Hardware Requirements

| Hardware | Min Load Time | Recommended |
|----------|---------------|-------------|
| M1       | ~6s (int4)    | ✓           |
| M1 Pro   | ~5s (int4)    | ✓           |
| M1 Max   | ~4s (int4)    | ✓ (target)  |
| M2/M3+   | ~3-4s (int4)  | ✓           |
| Intel    | N/A           | ✗ (no MLX)  |

## Scope

**In Scope:**
1. Add quantized model variants (4-bit, 8-bit) to `WhisperModel` enum
2. Parallelize encoder/decoder parameter updates
3. Add `ModelLoadingOptions` for user control
4. Background-ready loading pattern

**Out of Scope (future work):**
- Memory-mapped loading (requires upstream MLX changes)
- Custom quantization at runtime
- Core ML / ANE encoder offload

---

## API Design

### New Types

```swift
/// Quantization level for model weights
public enum WhisperQuantization: String, CaseIterable, Sendable {
    case float16      // Default, highest quality
    case int8         // 8-bit, 2x smaller
    case int4         // 4-bit, 4x smaller, recommended
}

/// Options for model loading behavior
public struct ModelLoadingOptions: Sendable {
    public var quantization: WhisperQuantization
    public var loadInBackground: Bool  // Return immediately, eval in background
    public var fallbackToFloat16: Bool  // If quantized unavailable, use float16

    public init(
        quantization: WhisperQuantization = .float16,
        loadInBackground: Bool = false,
        fallbackToFloat16: Bool = true
    ) {
        self.quantization = quantization
        self.loadInBackground = loadInBackground
        self.fallbackToFloat16 = fallbackToFloat16
    }

    // MARK: - Presets

    /// Default: float16, blocking, with fallback
    public static let `default` = ModelLoadingOptions(
        quantization: .float16,
        loadInBackground: false,
        fallbackToFloat16: true
    )

    /// Fast: int4, background loading, with fallback
    public static let fast = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: true,
        fallbackToFloat16: true
    )

    /// Fast but blocking: int4, wait for eval, with fallback
    public static let fastBlocking = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: false,
        fallbackToFloat16: true
    )

    /// Strict: int4, fail if unavailable (no fallback)
    public static let strict = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: false,
        fallbackToFloat16: false  // Throws if int4 unavailable
    )
}
```

### Updated WhisperSession API

```swift
// Existing (unchanged)
let session = try await WhisperSession.fromPretrained(.largeTurbo)

// New: with options
let session = try await WhisperSession.fromPretrained(
    .largeTurbo,
    options: .fast  // 4-bit + background loading
)

// New: check readiness for background loading
if session.isReady {
    let result = try await session.transcribe(audio)
} else {
    await session.waitUntilReady()
}
```

**Backward Compatibility:**
- Default behavior unchanged (float16, blocking load)
- New `options` parameter is optional with default value

---

## Quantized Model Repository Mapping

### HuggingFace Repo IDs

| Model | float16 | 8-bit | 4-bit |
|-------|---------|-------|-------|
| tiny | `whisper-tiny-mlx` | `whisper-tiny-mlx-8bit` | `whisper-tiny-mlx-4bit` |
| base | `whisper-base-mlx` | `whisper-base-mlx-8bit` | `whisper-base-mlx-4bit` |
| small | `whisper-small-mlx` | `whisper-small-mlx-8bit` | `whisper-small-mlx-4bit` |
| medium | `whisper-medium-mlx` | `whisper-medium-mlx-8bit` | `whisper-medium-mlx-4bit` |
| largeV3 | `whisper-large-v3-mlx` | `whisper-large-v3-mlx-8bit` | `whisper-large-v3-mlx-4bit` |
| **largeTurbo** | `whisper-large-v3-turbo` | `whisper-large-v3-turbo-8bit` | `whisper-large-v3-turbo-4bit` |

### Implementation

```swift
public static func repoId(
    for model: WhisperModel,
    quantization: WhisperQuantization = .float16
) -> String {
    let base: String
    let suffix: String

    switch model {
    case .tiny:       base = "mlx-community/whisper-tiny-mlx"
    case .base:       base = "mlx-community/whisper-base-mlx"
    case .small:      base = "mlx-community/whisper-small-mlx"
    case .medium:     base = "mlx-community/whisper-medium-mlx"
    case .largeV3:    base = "mlx-community/whisper-large-v3-mlx"
    case .largeTurbo: base = "mlx-community/whisper-large-v3-turbo"  // No -mlx suffix
    }

    switch quantization {
    case .float16: suffix = ""
    case .int8:    suffix = "-8bit"
    case .int4:    suffix = "-4bit"
    }

    return base + suffix
}
```

### File Size Comparison (large-v3-turbo)

| Quantization | File Size | Load Time (est.) |
|--------------|-----------|------------------|
| float16      | 1.6 GB    | ~12s             |
| int8         | 800 MB    | ~6s              |
| int4         | 400 MB    | ~4s              |

---

## Parallel Loading Implementation

### Current Sequential Flow (slow)

```swift
// Lines 257-262 in WhisperModelLoader.swift
let encoderParams = ModuleParameters.unflattened(encoderWeights)
try encoder.update(parameters: encoderParams, verify: [.noUnusedKeys])

let decoderParams = ModuleParameters.unflattened(decoderWeights)
try decoder.update(parameters: decoderParams, verify: [.all])

eval(encoder, decoder)  // Blocks until GPU ready
```

### New Parallel Flow

```swift
private static func loadWeightsParallel(
    encoderWeights: [String: MLXArray],
    decoderWeights: [String: MLXArray],
    encoder: AudioEncoder,
    decoder: TextDecoder
) async throws {
    // Run encoder and decoder updates concurrently
    try await withThrowingTaskGroup(of: Void.self) { group in
        group.addTask {
            let params = ModuleParameters.unflattened(encoderWeights)
            try encoder.update(parameters: params, verify: [.noUnusedKeys])
        }

        group.addTask {
            let params = ModuleParameters.unflattened(decoderWeights)
            try decoder.update(parameters: params, verify: [.all])
        }

        try await group.waitForAll()
    }
}
```

**Expected Savings:** ~1-2s (parallel vs sequential param updates)

### Thread Safety Considerations

**Assumption:** MLX-Swift's `Module.update(parameters:)` is thread-safe when called on independent modules (encoder vs decoder). This is based on:
- Encoder and decoder are separate `Module` instances with no shared mutable state
- Each update operates on distinct parameter trees
- MLX operations are dispatched to GPU which handles synchronization

**If assumption is wrong:** Add serial fallback:

```swift
public struct ModelLoadingOptions: Sendable {
    // ... existing properties
    public var parallelWeightLoading: Bool  // Default true, set false if issues arise
}

private static func loadWeights(..., parallel: Bool) async throws {
    if parallel {
        // Parallel path (default)
        try await loadWeightsParallel(...)
    } else {
        // Serial fallback
        let encoderParams = ModuleParameters.unflattened(encoderWeights)
        try encoder.update(parameters: encoderParams, verify: [.noUnusedKeys])

        let decoderParams = ModuleParameters.unflattened(decoderWeights)
        try decoder.update(parameters: decoderParams, verify: [.all])
    }
}
```

**Verification plan:**
1. Run parallel loading 100x in test, check for crashes/corruption
2. Compare transcription output between parallel and serial loading
3. Monitor with Thread Sanitizer enabled

---

## Background Loading Pattern

### Loading State

```swift
import os  // For OSAllocatedUnfairLock

public final class WhisperSession: @unchecked Sendable {

    public enum LoadingState: Sendable {
        case loading
        case ready
        case failed(Error)
    }

    // MARK: - Thread-Safe State (OSAllocatedUnfairLock instead of ManagedAtomic)
    // Note: ManagedAtomic<LoadingState> won't compile because Error isn't AtomicValue
    private let _state = OSAllocatedUnfairLock(initialState: LoadingState.loading)

    /// Stream and continuation for signaling background load completion
    private let readyStream: AsyncStream<Void>
    private let readyContinuation: AsyncStream<Void>.Continuation

    /// Handle to background eval task for cancellation on deinit
    private var backgroundTask: Task<Void, Never>?

    /// The quantization that was actually loaded (may differ from requested if fallback occurred)
    public let actualQuantization: WhisperQuantization

    /// True if a fallback occurred during loading
    public let didFallback: Bool

    public var state: LoadingState { _state.withLock { $0 } }
    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    internal init(loadResult: LoadResult) {
        // Store fallback information for caller visibility
        self.actualQuantization = loadResult.actualQuantization
        self.didFallback = loadResult.didFallback

        // Create stream/continuation pair for background loading coordination
        var continuation: AsyncStream<Void>.Continuation!
        self.readyStream = AsyncStream { continuation = $0 }
        self.readyContinuation = continuation

        // ... store loaded model components from loadResult.model
    }

    deinit {
        // Cancel background eval if session is dropped before completion
        backgroundTask?.cancel()
    }

    /// Wait until model is fully loaded and evaluated
    /// - Parameter timeout: Maximum time to wait (default 30s)
    /// - Returns: true if ready, false if timed out (but may become ready later)
    /// - Throws: WhisperError.loadingFailed if background eval fails
    public func waitUntilReady(timeout: Duration = .seconds(30)) async throws -> Bool {
        // Fast path: already terminal state
        switch state {
        case .ready: return true
        case .failed(let error): throw error
        case .loading: break
        }

        // Wait for background eval with timeout
        let didComplete = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await _ in self.readyStream { break }
                return true  // Stream completed
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return false  // Timeout
            }
            // First task to complete wins
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }

        // IMPORTANT: Recheck state after wait (fixes timeout race condition)
        // Even if timeout won, state may have transitioned to .ready
        switch state {
        case .ready: return true
        case .failed(let error): throw error
        case .loading:
            // Only return false (timeout) if still loading
            return didComplete
        }
    }
}
```

**Key Changes from Codex Review:**
- `OSAllocatedUnfairLock` instead of `ManagedAtomic` (Error isn't AtomicValue-conformant)
- `actualQuantization` and `didFallback` exposed on session (API consistency)
- `backgroundTask` stored for cancellation in `deinit`
- `waitUntilReady` returns `Bool` instead of throwing on timeout (cleaner semantics)
- State recheck after timeout (fixes race condition)

### Updated Load Flow

```swift
public static func fromPretrained(
    _ model: WhisperModel,
    options: ModelLoadingOptions = .default
) async throws -> WhisperSession {

    let loadResult = try await WhisperModelLoader.load(
        model: model,
        quantization: options.quantization,
        fallbackToFloat16: options.fallbackToFloat16  // New option
    )

    let session = WhisperSession(loadResult: loadResult)

    if options.loadInBackground {
        // Store task handle for cancellation in deinit
        session.backgroundTask = Task.detached { [weak session] in
            guard let session = session else { return }  // Session dropped, skip eval

            do {
                // Check for cancellation before expensive eval
                try Task.checkCancellation()
                eval(loadResult.model.encoder, loadResult.model.decoder)
                session._state.withLock { $0 = .ready }
            } catch is CancellationError {
                // Session was dropped, don't update state
                return
            } catch {
                session._state.withLock { $0 = .failed(error) }
            }
            // Always signal completion (success or failure)
            session.readyContinuation.finish()
        }
    } else {
        // Blocking eval (current behavior)
        eval(loadResult.model.encoder, loadResult.model.decoder)
        session._state.withLock { $0 = .ready }
    }

    return session
}
```

**Key Changes:**
- Uses `LoadResult` to pass fallback info to session
- Stores `backgroundTask` for cancellation
- Uses `[weak session]` to avoid retain cycle
- Checks `Task.checkCancellation()` before expensive eval
- Uses `withLock` instead of atomic store

### Usage

```swift
// Fast startup - show UI while loading
let session = try await WhisperSession.fromPretrained(.largeTurbo, options: .fast)
showRecordingUI()

// When user taps record
try await session.waitUntilReady()
let result = try await session.transcribe(audio)
```

---

## Error Handling

### New Error Cases

```swift
public enum WhisperError: Error, LocalizedError {
    // Existing errors...
    case invalidModelFormat(String)
    case tokenizationFailed(String)

    // New errors for quantization and background loading
    case quantizedModelNotAvailable(WhisperModel, WhisperQuantization)
    case modelNotReady
    case loadingFailed(underlying: Error)
    case loadingTimeout
    case quantizationFallback(requested: WhisperQuantization, actual: WhisperQuantization)

    public var errorDescription: String? {
        switch self {
        case .quantizedModelNotAvailable(let model, let quant):
            return "Quantized model \(model) with \(quant) not available on HuggingFace"
        case .modelNotReady:
            return "Model not ready. Call waitUntilReady() first"
        case .loadingFailed(let error):
            return "Model loading failed: \(error.localizedDescription)"
        case .loadingTimeout:
            return "Model loading timed out. Background eval may still be running."
        case .quantizationFallback(let requested, let actual):
            return "Requested \(requested) unavailable, loaded \(actual) instead"
        default:
            // existing cases...
        }
    }
}
```

### Explicit Fallback (No Silent Behavior)

```swift
/// Result of loading a model, including any fallback that occurred
public struct LoadResult {
    public let model: LoadedModel
    public let requestedQuantization: WhisperQuantization
    public let actualQuantization: WhisperQuantization

    /// True if we fell back to a different quantization than requested
    public var didFallback: Bool { requestedQuantization != actualQuantization }
}

public static func load(
    model: WhisperModel,
    quantization: WhisperQuantization,
    fallbackToFloat16: Bool = true
) async throws -> LoadResult {

    let repoId = repoId(for: model, quantization: quantization)

    do {
        let loaded = try await loadFromRepo(repoId)
        return LoadResult(
            model: loaded,
            requestedQuantization: quantization,
            actualQuantization: quantization
        )
    } catch {
        // If quantized model not found, fallback to float16
        if fallbackToFloat16 && quantization != .float16 {
            let fallbackRepo = repoId(for: model, quantization: .float16)
            let loaded = try await loadFromRepo(fallbackRepo)
            return LoadResult(
                model: loaded,
                requestedQuantization: quantization,
                actualQuantization: .float16  // Explicit: caller knows what happened
            )
        }
        throw WhisperError.quantizedModelNotAvailable(model, quantization)
    }
}
```

**Usage - Caller Can Handle Fallback:**

```swift
let result = try await WhisperModelLoader.load(model: .largeTurbo, quantization: .int4)

if result.didFallback {
    print("Note: Using \(result.actualQuantization) instead of \(result.requestedQuantization)")
    // Optionally: throw WhisperError.quantizationFallback(...) if strict mode required
}
```

### Transcribe Guard

```swift
public func transcribe(_ audio: MLXArray) async throws -> TranscriptionResult {
    guard isReady else {
        throw WhisperError.modelNotReady
    }
    // ... existing transcription logic
}
```

---

## Testing Strategy

### Mock Loader Protocol (CI-Safe)

```swift
/// Protocol for model loading - allows mocking in CI without network
public protocol ModelLoaderProtocol: Sendable {
    func load(
        model: WhisperModel,
        quantization: WhisperQuantization,
        fallbackToFloat16: Bool
    ) async throws -> LoadResult
}

/// Real implementation using HuggingFace
public struct HuggingFaceModelLoader: ModelLoaderProtocol {
    public func load(...) async throws -> LoadResult {
        // Real HF download logic
    }
}

/// Mock for unit tests (no network required)
public struct MockModelLoader: ModelLoaderProtocol {
    public var shouldFail: Bool = false
    public var simulatedDelay: Duration = .zero
    public var forceFallback: Bool = false

    public func load(
        model: WhisperModel,
        quantization: WhisperQuantization,
        fallbackToFloat16: Bool
    ) async throws -> LoadResult {
        if simulatedDelay > .zero {
            try await Task.sleep(for: simulatedDelay)
        }

        if shouldFail {
            throw WhisperError.quantizedModelNotAvailable(model, quantization)
        }

        let actualQuant = forceFallback ? .float16 : quantization
        return LoadResult(
            model: MockLoadedModel(),
            requestedQuantization: quantization,
            actualQuantization: actualQuant
        )
    }
}

// Inject loader via environment or init
public static func fromPretrained(
    _ model: WhisperModel,
    options: ModelLoadingOptions = .default,
    loader: ModelLoaderProtocol = HuggingFaceModelLoader()
) async throws -> WhisperSession
```

### Unit Tests

```swift
// Tests/MLXAudioSTTTests/WhisperQuantizationTests.swift

final class WhisperQuantizationTests: XCTestCase {

    // MARK: - Repo ID Mapping

    func testRepoIdFloat16() {
        XCTAssertEqual(
            WhisperModelLoader.repoId(for: .largeTurbo, quantization: .float16),
            "mlx-community/whisper-large-v3-turbo"
        )
    }

    func testRepoId4Bit() {
        XCTAssertEqual(
            WhisperModelLoader.repoId(for: .largeTurbo, quantization: .int4),
            "mlx-community/whisper-large-v3-turbo-4bit"
        )
    }

    func testRepoId8Bit() {
        XCTAssertEqual(
            WhisperModelLoader.repoId(for: .tiny, quantization: .int8),
            "mlx-community/whisper-tiny-mlx-8bit"
        )
    }

    // MARK: - Loading Options

    func testDefaultOptions() {
        let opts = ModelLoadingOptions.default
        XCTAssertEqual(opts.quantization, .float16)
        XCTAssertFalse(opts.loadInBackground)
        XCTAssertTrue(opts.fallbackToFloat16)
    }

    func testFastOptions() {
        let opts = ModelLoadingOptions.fast
        XCTAssertEqual(opts.quantization, .int4)
        XCTAssertTrue(opts.loadInBackground)
    }

    func testStrictOptions() {
        let opts = ModelLoadingOptions.strict
        XCTAssertEqual(opts.quantization, .int4)
        XCTAssertFalse(opts.fallbackToFloat16)  // Will throw if unavailable
    }
}
```

### Integration Tests

```swift
final class WhisperLoadingIntegrationTests: XCTestCase {

    func testLoadQuantizedModel() async throws {
        let session = try await WhisperSession.fromPretrained(
            .tiny,  // Use tiny for fast CI
            options: ModelLoadingOptions(quantization: .int4, loadInBackground: false)
        )
        XCTAssertTrue(session.isReady)
    }

    func testBackgroundLoading() async throws {
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: ModelLoadingOptions(quantization: .int4, loadInBackground: true)
        )
        // Should return before ready
        try await session.waitUntilReady()
        XCTAssertTrue(session.isReady)
    }

    func testTranscriptionAccuracy() async throws {
        // Compare float16 vs int4 on reference audio
        let audio = try loadTestAudio("hello_world.wav")

        let fp16 = try await WhisperSession.fromPretrained(.tiny, options: .default)
        let int4 = try await WhisperSession.fromPretrained(.tiny, options: .fast)

        let result16 = try await fp16.transcribe(audio)
        let result4 = try await int4.transcribe(audio)

        // WER (Word Error Rate) should be within acceptable threshold
        // Quantization may cause minor differences, but should be <2%
        let wer = calculateWER(reference: result16.text, hypothesis: result4.text)
        XCTAssertLessThan(wer, 0.02, "Quantized model WER should be <2% vs float16")
    }

    func testBackgroundLoadingFailure() async throws {
        let mockLoader = MockModelLoader(shouldFail: true)
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: .fast,
            loader: mockLoader
        )

        // waitUntilReady should throw the underlying error
        do {
            _ = try await session.waitUntilReady(timeout: .seconds(5))
            XCTFail("Should have thrown")
        } catch WhisperError.quantizedModelNotAvailable {
            // Expected
        }
    }

    // MARK: - Timeout Tests

    func testWaitUntilReadyTimeout() async throws {
        let mockLoader = MockModelLoader(simulatedDelay: .seconds(10))
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: .fast,
            loader: mockLoader
        )

        let ready = try await session.waitUntilReady(timeout: .milliseconds(100))
        XCTAssertFalse(ready, "Should timeout and return false")
        XCTAssertFalse(session.isReady, "Should still be loading")
    }

    func testTimeoutThenEventualSuccess() async throws {
        let mockLoader = MockModelLoader(simulatedDelay: .milliseconds(200))
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: .fast,
            loader: mockLoader
        )

        // First wait times out
        let ready1 = try await session.waitUntilReady(timeout: .milliseconds(50))
        XCTAssertFalse(ready1)

        // Wait longer, should eventually succeed
        let ready2 = try await session.waitUntilReady(timeout: .seconds(1))
        XCTAssertTrue(ready2)
        XCTAssertTrue(session.isReady)
    }

    // MARK: - Multi-Waiter Tests

    func testMultipleWaiters() async throws {
        let mockLoader = MockModelLoader(simulatedDelay: .milliseconds(100))
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: .fast,
            loader: mockLoader
        )

        // Multiple concurrent waiters
        async let ready1 = session.waitUntilReady(timeout: .seconds(5))
        async let ready2 = session.waitUntilReady(timeout: .seconds(5))
        async let ready3 = session.waitUntilReady(timeout: .seconds(5))

        let results = try await [ready1, ready2, ready3]
        XCTAssertTrue(results.allSatisfy { $0 }, "All waiters should see ready")
    }

    // MARK: - Fallback Tests

    func testFallbackExposedOnSession() async throws {
        let mockLoader = MockModelLoader(forceFallback: true)
        let session = try await WhisperSession.fromPretrained(
            .tiny,
            options: ModelLoadingOptions(quantization: .int4, loadInBackground: false),
            loader: mockLoader
        )

        XCTAssertTrue(session.didFallback)
        XCTAssertEqual(session.actualQuantization, .float16)
    }

    func testStrictModeNoFallback() async throws {
        let mockLoader = MockModelLoader(forceFallback: true)

        do {
            _ = try await WhisperSession.fromPretrained(
                .tiny,
                options: .strict,  // fallbackToFloat16 = false
                loader: mockLoader
            )
            XCTFail("Should throw when quantized unavailable and strict mode")
        } catch WhisperError.quantizedModelNotAvailable {
            // Expected
        }
    }

    // MARK: - Cancellation Tests

    func testSessionDropCancelsBackgroundEval() async throws {
        let mockLoader = MockModelLoader(simulatedDelay: .seconds(10))

        var session: WhisperSession? = try await WhisperSession.fromPretrained(
            .tiny,
            options: .fast,
            loader: mockLoader
        )

        // Drop session while background eval is running
        session = nil

        // Give cancellation time to propagate
        try await Task.sleep(for: .milliseconds(50))

        // No crash, no leak - test passes if we get here
    }

    // MARK: - Helper

    private func calculateWER(reference: String, hypothesis: String) -> Double {
        let refWords = reference.lowercased().split(separator: " ")
        let hypWords = hypothesis.lowercased().split(separator: " ")

        // Levenshtein distance at word level
        let distance = levenshteinDistance(Array(refWords), Array(hypWords))
        return Double(distance) / max(Double(refWords.count), 1)
    }
}
```

---

## Files to Modify

### Changed Files

| File | Changes |
|------|---------|
| `WhisperModelLoader.swift` | Add `quantization` param to `repoId()`, parallel weight loading |
| `WhisperSession.swift` | Add `LoadingState`, `isReady`, `waitUntilReady()`, update `fromPretrained()` |
| `WhisperConfiguration.swift` | Add `WhisperQuantization` enum |
| `WhisperError.swift` | Add new error cases |

### New Files

| File | Purpose |
|------|---------|
| `ModelLoadingOptions.swift` | Options struct for loading behavior |
| `WhisperQuantizationTests.swift` | Unit tests for repo mapping |
| `WhisperLoadingIntegrationTests.swift` | Integration tests |

### File Tree

```
mlx_audio_swift/stt/MLXAudioSTT/
├── Models/Whisper/
│   ├── WhisperModelLoader.swift    ← modify
│   ├── WhisperSession.swift        ← modify
│   ├── WhisperConfiguration.swift  ← modify (add enum)
│   ├── WhisperError.swift          ← modify
│   └── ModelLoadingOptions.swift   ← new
└── ...

Tests/MLXAudioSTTTests/
├── WhisperQuantizationTests.swift          ← new
└── WhisperLoadingIntegrationTests.swift    ← new
```

---

## Expected Results

### Before Optimization

```
Load time:     ~12s (large-v3-turbo, float16)
File size:     1.6 GB
First transcribe: Blocked until eval complete
```

### After Optimization

```
Load time:     ~4-5s (large-v3-turbo, int4)
File size:     400 MB (75% smaller)
First transcribe: Can start immediately with background eval
```

### Performance Gains

| Optimization | Time Saved | Cumulative |
|--------------|------------|------------|
| 4-bit quantization | ~7-8s | 12s → 4-5s |
| Parallel param updates | ~1-2s | 4-5s → 3-4s |
| Background eval | 0s (UX only) | Perceived instant |

### API Compatibility

- Existing code works unchanged (defaults to float16, blocking)
- New `options` parameter is optional
- All models support quantization variants
