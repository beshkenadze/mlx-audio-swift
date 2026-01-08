# CoreML Backend Design for mlx-audio-swift

> Design Date: 2026-01-08

## Executive Summary

Add optional CoreML/ANE acceleration to mlx-audio-swift while keeping MLX as the primary framework. CoreML provides 3-18x speedup for encoders and 30-50% for vocoders on Apple Neural Engine.

## Context

- **Project**: mlx-audio-swift (SDK library for STT/TTS)
- **Primary Framework**: MLX
- **Goal**: Leverage CoreML/ANE when available for performance gains
- **Status**: Initial release (no backwards compatibility concerns)

## Research Findings

### Components Suitable for CoreML

| Component | CoreML Viable | Expected Speedup | Rationale |
|-----------|---------------|------------------|-----------|
| Whisper Encoder | ✅ Yes | 3-18x | Fixed shapes, transformer, proven (WhisperKit, Lightning-SimulWhisper) |
| Whisper Decoder | ❌ No | N/A | Dynamic shapes, KV-cache incompatible |
| Kokoro HAR Vocoder | ✅ Yes | 30-50% | Conv-based, proven (MeloTTS-CoreML) |
| Kokoro Duration Model | ❌ No | N/A | LSTM, dynamic sequence lengths |
| Orpheus SNAC | ⚠️ Potentially | Unknown | Conv-based but needs validation |

### Reference Implementations

- **WhisperKit**: CoreML encoder + MLX-like decoder
- **Lightning-SimulWhisper**: Hybrid CoreML + MLX
- **MeloTTS-CoreML**: CoreML vocoder for TTS

## Architecture Design

### Protocol Abstractions

Two protocols covering 90% of real performance gains:

```swift
// MARK: - AudioEncoder Protocol (STT)

public protocol AudioEncoder {
    /// Encodes mel spectrogram into encoder output
    /// - Parameter mel: Mel spectrogram [batch, n_mels, n_frames]
    /// - Returns: Encoder output [batch, seq_len, d_model]
    /// - Throws: EncoderError on conversion failure or inference error
    func encode(_ mel: MLXArray) throws -> MLXArray
}

public class MLXAudioEncoder: AudioEncoder {
    private let encoder: WhisperEncoder

    public func encode(_ mel: MLXArray) throws -> MLXArray {
        encoder(mel)
    }
}

public class CoreMLAudioEncoder: AudioEncoder {
    private let model: MLModel

    public func encode(_ mel: MLXArray) throws -> MLXArray {
        // Convert MLXArray → MLMultiArray (throws on shape mismatch)
        // Run CoreML inference (throws on ANE failure)
        // Convert back → MLXArray
    }
}
```

```swift
// MARK: - AudioVocoder Protocol (TTS)

public protocol AudioVocoder {
    /// Decodes acoustic features into audio waveform
    /// - Parameter features: Acoustic features from TTS model
    /// - Returns: Audio waveform samples
    /// - Throws: VocoderError on conversion failure or inference error
    func decode(_ features: MLXArray) throws -> MLXArray
}

public class MLXVocoder: AudioVocoder {
    public func decode(_ features: MLXArray) throws -> MLXArray {
        // MLX implementation
    }
}

public class CoreMLVocoder: AudioVocoder {
    private let model: MLModel

    public func decode(_ features: MLXArray) throws -> MLXArray {
        // CoreML implementation with MLX interop
    }
}
```

### Error Types

```swift
public enum EncoderError: Error {
    case shapeConversionFailed(expected: [Int], got: [Int])
    case coreMLInferenceFailed(underlying: Error)
    case aneNotAvailable
    case modelNotLoaded
}

public enum VocoderError: Error {
    case shapeConversionFailed(expected: [Int], got: [Int])
    case coreMLInferenceFailed(underlying: Error)
    case unsupportedSampleRate(Int)
}
```

### Naming Convention

Following Apple's standard library pattern (like `Sequence` → `Array`, `Set`):

- Protocol gets the clean, general name
- Implementations get prefixed names indicating their backend

### API Design

#### Extended ModelLoadingOptions

```swift
public struct ModelLoadingOptions: Sendable {
    // Existing properties
    public var quantization: WhisperQuantization
    public var loadInBackground: Bool
    public var fallbackToFloat16: Bool

    // New CoreML properties
    public var backend: BackendPreference = .auto
    public var backendOverrides: [Component: BackendPreference] = [:]

    public enum BackendPreference: Sendable {
        case auto      // System decides best backend
        case coreML    // Prefer CoreML/ANE
        case mlx       // Force MLX only
    }

    public enum Component: Sendable, Hashable {
        case encoder
        case vocoder
    }

    // MARK: - Presets

    /// Default: auto backend selection
    public static let `default` = ModelLoadingOptions(
        quantization: .float16,
        loadInBackground: false,
        fallbackToFloat16: true,
        backend: .auto
    )

    /// Maximum performance: prefer CoreML where available
    public static let maxPerformance = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: true,
        fallbackToFloat16: true,
        backend: .coreML
    )

    /// MLX only: skip CoreML entirely
    public static let mlxOnly = ModelLoadingOptions(
        quantization: .float16,
        loadInBackground: false,
        fallbackToFloat16: true,
        backend: .mlx
    )
}
```

#### Extended WhisperProgress

```swift
public enum WhisperProgress {
    case downloading(Float)
    case downloadingCoreML(Float)   // CoreML model download
    case compilingANE               // First-run ANE compilation (~4 min)
    case loading(Float)

    // Error/fallback states
    case coreMLUnavailable(reason: String)  // Falling back to MLX
    case compilationFailed(Error)           // ANE compilation failed
    case downloadFailed(Error)              // CoreML download failed
}
```

#### Session Observability

```swift
/// Information about the loaded session for debugging and logging
public struct SessionInfo: Sendable {
    /// Actual backend used for encoder (may differ from requested)
    public let encoderBackend: BackendPreference

    /// Actual backend used for vocoder (may differ from requested)
    public let vocoderBackend: BackendPreference

    /// Whether ANE compilation completed successfully
    public let aneCompiled: Bool

    /// Whether fallback occurred (requested != actual)
    public let didFallback: Bool

    /// Time spent loading the model
    public let loadDuration: Duration
}

// Usage:
let session = try await WhisperSession.fromPretrained(...)
print("Encoder backend: \(session.info.encoderBackend)")  // .coreML
print("Did fallback: \(session.info.didFallback)")        // false
```

### Backend Selection Logic

#### Auto Selection Criteria

Auto selection SHALL prefer CoreML when ALL conditions met:
1. CoreML model is available for the component
2. Device supports ANE (A12+ chip / M1+ chip)
3. Model has been pre-compiled OR user accepts compilation delay

Auto selection SHALL prefer MLX when ANY condition met:
1. CoreML model unavailable for the component
2. Device does not support ANE
3. User explicitly set `backend: .mlx`

```swift
func selectBackend(
    for component: Component,
    options: ModelLoadingOptions,
    model: WhisperModel
) -> BackendPreference {
    // Check for explicit override first
    if let override = options.backendOverrides[component] {
        return override
    }

    switch options.backend {
    case .mlx:
        return .mlx

    case .coreML:
        // Prefer CoreML, fallback to MLX if unavailable
        guard hasCoreMLModel(for: model, component: component) else {
            return .mlx  // Fallback
        }
        guard deviceSupportsANE() else {
            return .mlx  // ANE required for meaningful speedup
        }
        return .coreML

    case .auto:
        // Auto: use CoreML only if available AND beneficial
        guard hasCoreMLModel(for: model, component: component) else {
            return .mlx
        }
        guard deviceSupportsANE() else {
            return .mlx
        }
        return .coreML
    }
}

func deviceSupportsANE() -> Bool {
    // Check for A12+ (iPhone XS) or M1+ (Mac)
    // Implementation uses ProcessInfo or Metal device capabilities
}
```

### Model Distribution

- **Download on demand** from HuggingFace
- CoreML models stored in separate repos (e.g., `mlx-community/whisper-large-v3-turbo-coreml`)
- Fallback to MLX if CoreML model unavailable

### ANE Compilation Handling

First-run ANE compilation takes ~4 minutes. SDK provides:

1. **Progress callbacks** via `WhisperProgress.compilingANE`
2. **Programmatic warmup** for app developers to control timing
3. **Cancellation support** for long-running operations
4. **Timeout handling** with configurable limits

```swift
// App developer can trigger warmup at appropriate time
let session = try await WhisperSession.fromPretrained(
    model: .largeTurbo,
    options: .maxPerformance,
    progressHandler: { progress in
        switch progress {
        case .compilingANE:
            showANECompilationUI()  // App's responsibility
        case .compilationFailed(let error):
            showFallbackNotice(error)
        case .coreMLUnavailable(let reason):
            logFallback(reason)
        default:
            updateProgressBar(progress)
        }
    }
)

// Cancellation support
let warmupTask = Task {
    try await session.warmup()
}

// User navigates away - cancel warmup
warmupTask.cancel()
```

### Failure Modes

| Failure | Trigger | Recovery |
|---------|---------|----------|
| ANE compilation timeout | >10 min compilation | Cancel, fallback to MLX, emit `.compilationFailed(.timeout)` |
| CoreML download failure | Network error, 404, disk full | Emit `.downloadFailed(error)`, continue with MLX |
| MLMultiArray conversion | Unsupported dtype/shape | Throw `EncoderError.shapeConversionFailed` |
| ANE not available | Unsupported device | Auto-select MLX, emit `.coreMLUnavailable("ANE not supported")` |

## Test Scenarios (Given/When/Then)

### Backend Selection

```gherkin
Scenario: Auto selects CoreML when available
  Given: Whisper large-v3-turbo model
  And: CoreML model exists at mlx-community/whisper-large-v3-turbo-coreml
  And: Device is M1 Mac (ANE supported)
  When: Session created with backend: .auto
  Then: Encoder uses CoreMLAudioEncoder
  And: session.info.encoderBackend == .coreML
  And: session.info.didFallback == false

Scenario: Auto falls back to MLX when CoreML unavailable
  Given: Whisper tiny model
  And: No CoreML model exists for tiny variant
  When: Session created with backend: .auto
  Then: Encoder uses MLXAudioEncoder
  And: session.info.encoderBackend == .mlx
  And: Progress callback receives .coreMLUnavailable("Model not available")

Scenario: User forces MLX despite CoreML availability
  Given: Whisper large-v3-turbo with CoreML available
  When: Session created with backend: .mlx
  Then: Encoder uses MLXAudioEncoder
  And: No CoreML download attempted
  And: session.info.encoderBackend == .mlx

Scenario: Auto falls back on unsupported device
  Given: Whisper large-v3-turbo with CoreML available
  And: Device is A11 (iPhone X, no ANE)
  When: Session created with backend: .auto
  Then: Encoder uses MLXAudioEncoder
  And: Progress callback receives .coreMLUnavailable("ANE not supported")
```

### Error Handling

```gherkin
Scenario: ANE compilation timeout
  Given: Large CoreML model requiring >10 min compilation
  And: Default timeout of 10 minutes
  When: Warmup initiated
  Then: After 10 minutes, throws ANECompilationError.timeout
  And: Progress callback receives .compilationFailed(.timeout)
  And: Session falls back to MLX encoder

Scenario: Download failure with fallback enabled
  Given: Network unavailable
  And: backend: .auto with fallbackToFloat16: true
  When: Session creation attempts CoreML download
  Then: Progress callback receives .downloadFailed(NetworkError)
  And: Session continues with MLX backend
  And: session.info.didFallback == true

Scenario: Shape conversion failure
  Given: CoreML encoder loaded
  And: Input mel has unexpected shape [1, 80, 5000] (exceeds 3000 frames)
  When: encode() called
  Then: Throws EncoderError.shapeConversionFailed(expected: [1, 80, 3000], got: [1, 80, 5000])

Scenario: Warmup cancellation
  Given: CoreML encoder with pending ANE compilation
  When: warmupTask.cancel() called
  Then: Compilation stops within 1 second
  And: No error thrown (cancellation is clean)
  And: Session remains usable with MLX fallback
```

## Implementation Plan

### Phase 1: Protocol Foundation
- [ ] Define `AudioEncoder` protocol
- [ ] Refactor existing encoder to `MLXAudioEncoder`
- [ ] Extend `ModelLoadingOptions` with backend properties
- [ ] Extend `WhisperProgress` with CoreML states

### Phase 2: CoreML Encoder
- [ ] Implement `CoreMLAudioEncoder`
- [ ] Add MLXArray ↔ MLMultiArray conversion utilities
- [ ] Implement backend selection logic
- [ ] Add CoreML model download support

### Phase 3: Vocoder Support
- [ ] Define `AudioVocoder` protocol
- [ ] Implement `MLXVocoder`
- [ ] Implement `CoreMLVocoder` (when models available)

### Phase 4: Testing & Optimization
- [ ] Benchmark CoreML vs MLX performance
- [ ] Test ANE compilation flow
- [ ] Validate fallback behavior

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Abstraction Level | 2 protocols (Encoder, Vocoder) | Covers 90% of gains, maintains simplicity |
| Backend Selection | Auto with options override | Users don't know which models support CoreML |
| Options Structure | Extend existing `ModelLoadingOptions` | Consistent with current API |
| ANE Compilation | Progress callbacks + programmatic warmup | SDK can't decide UI, developer controls timing |
| Model Distribution | Download on demand from HuggingFace | Smaller initial download, models may not be needed |
| Protocol Visibility | All public | Initial release, follow Apple standard library pattern |
| Naming Convention | Protocol gets clean name, implementations prefixed | Apple standard library pattern |

## File Structure

```
mlx_audio_swift/
├── stt/
│   └── MLXAudioSTT/
│       └── Models/
│           └── Whisper/
│               ├── Encoders/
│               │   ├── AudioEncoder.swift          # Protocol
│               │   ├── MLXAudioEncoder.swift       # MLX implementation
│               │   └── CoreMLAudioEncoder.swift    # CoreML implementation
│               ├── ModelLoadingOptions.swift       # Extended
│               └── WhisperProgress.swift           # Extended (or in WhisperSession.swift)
└── tts/
    └── MLXAudioTTS/
        └── Vocoders/
            ├── AudioVocoder.swift                  # Protocol
            ├── MLXVocoder.swift                    # MLX implementation
            └── CoreMLVocoder.swift                 # CoreML implementation
```

## References

- [WhisperKit](https://github.com/argmaxinc/WhisperKit) - CoreML + Swift implementation
- [Lightning-SimulWhisper](https://github.com/example) - Hybrid CoreML + MLX
- [MeloTTS-CoreML](https://github.com/example) - CoreML vocoder
- [GPU_OPTIMIZATION_RESEARCH.md](../GPU_OPTIMIZATION_RESEARCH.md) - MLX optimization notes
