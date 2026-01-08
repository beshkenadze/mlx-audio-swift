# Fast Model Loading Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Whisper model loading time from ~12s to ~4-5s through quantized model support and parallel loading optimizations.

**Architecture:** Load pre-quantized 4-bit models from HuggingFace (400MB vs 1.6GB), parallelize encoder/decoder parameter updates, and support background eval for improved UX.

**Tech Stack:** MLX-Swift, HuggingFace Hub, Swift Concurrency

---

## Success Criteria

- Load `large-v3-turbo` in <5s on M1 Max (currently ~12s)
- Maintain API compatibility (no breaking changes)
- Optional quantization (users can choose precision)

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

    public static let `default` = ModelLoadingOptions(
        quantization: .float16,
        loadInBackground: false
    )

    public static let fast = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: true
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

---

## Background Loading Pattern

### Loading State

```swift
public final class WhisperSession: @unchecked Sendable {

    public enum LoadingState: Sendable {
        case loading
        case ready
        case failed(Error)
    }

    private let _state = ManagedAtomic<LoadingState>(.loading)
    private let readyContinuation: AsyncStream<Void>.Continuation?

    public var state: LoadingState { _state.load(ordering: .acquiring) }
    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    /// Wait until model is fully loaded and evaluated
    public func waitUntilReady() async throws {
        switch state {
        case .ready: return
        case .failed(let error): throw error
        case .loading:
            // Wait for background eval to complete
            for await _ in readyStream { break }
        }
    }
}
```

### Updated Load Flow

```swift
public static func fromPretrained(
    _ model: WhisperModel,
    options: ModelLoadingOptions = .default
) async throws -> WhisperSession {

    let loaded = try await WhisperModelLoader.load(
        model: model,
        quantization: options.quantization
    )

    let session = WhisperSession(loaded: loaded)

    if options.loadInBackground {
        // Return immediately, eval in background
        Task.detached {
            eval(loaded.encoder, loaded.decoder)
            session._state.store(.ready, ordering: .releasing)
            session.readyContinuation?.finish()
        }
    } else {
        // Blocking eval (current behavior)
        eval(loaded.encoder, loaded.decoder)
        session._state.store(.ready, ordering: .releasing)
    }

    return session
}
```

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

    // New errors for quantization
    case quantizedModelNotAvailable(WhisperModel, WhisperQuantization)
    case modelNotReady
    case loadingFailed(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .quantizedModelNotAvailable(let model, let quant):
            return "Quantized model \(model) with \(quant) not available on HuggingFace"
        case .modelNotReady:
            return "Model not ready. Call waitUntilReady() first"
        case .loadingFailed(let error):
            return "Model loading failed: \(error.localizedDescription)"
        default:
            // existing cases...
        }
    }
}
```

### Graceful Fallback

```swift
public static func load(
    model: WhisperModel,
    quantization: WhisperQuantization,
    fallbackToFloat16: Bool = true
) async throws -> LoadedModel {

    let repoId = repoId(for: model, quantization: quantization)

    do {
        return try await loadFromRepo(repoId)
    } catch {
        // If quantized model not found, fallback to float16
        if fallbackToFloat16 && quantization != .float16 {
            print("⚠️ Quantized model unavailable, falling back to float16")
            let fallbackRepo = repoId(for: model, quantization: .float16)
            return try await loadFromRepo(fallbackRepo)
        }
        throw WhisperError.quantizedModelNotAvailable(model, quantization)
    }
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
    }

    func testFastOptions() {
        let opts = ModelLoadingOptions.fast
        XCTAssertEqual(opts.quantization, .int4)
        XCTAssertTrue(opts.loadInBackground)
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

        // Should produce same text (or very close)
        XCTAssertEqual(result16.text.lowercased(), result4.text.lowercased())
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
