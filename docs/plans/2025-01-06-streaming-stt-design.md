# Native Swift STT with AlignAtt Streaming — Design Document

> **Date**: 2025-01-06
> **Branch**: feat/streaming-stt
> **Status**: Approved for implementation

---

## 1. Overview

### Goal
Native Swift Speech-to-Text (STT) implementation with AlignAtt streaming for Apple Silicon. Key advantage: **~1s latency** to first token (vs ~2-3s with VAD-based approaches like WhisperKit).

### Key Decisions Summary

| Aspect | Decision |
|--------|----------|
| Model | Whisper + AlignAtt streaming |
| Implementation | Full MLX Swift port |
| Input | `MLXArray` + `sampleRate: Int` |
| Output | `AsyncThrowingStream<StreamingResult, Error>` |
| Default model | `large-v3-turbo` (base for tests) |
| API design | `STTSession` protocol + `WhisperSession` implementation |
| Config | `StreamingConfig` at session creation |

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MLXAudio Package                        │
├─────────────────────────────────────────────────────────────┤
│  TTS/                          │  STT/                       │
│  ├── Kokoro/                   │  ├── STTSession.swift       │
│  ├── Orpheus/                  │  └── Whisper/               │
│  └── Marvis/                   │      ├── WhisperSession     │
│                                │      ├── WhisperModel       │
│                                │      └── Streaming/         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Public API

### STTSession Protocol

```swift
public protocol STTSession: Sendable {
    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) -> AsyncThrowingStream<StreamingResult, Error>

    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) async throws -> String
}
```

### StreamingResult

```swift
public struct StreamingResult: Sendable {
    public let text: String
    public let isFinal: Bool
    public let timestamp: ClosedRange<TimeInterval>
}
```

### WhisperSession

```swift
public final class WhisperSession: STTSession, @unchecked Sendable {

    // MARK: - Factory Methods

    public static func fromPretrained(
        model: WhisperModel = .largeTurbo,
        streaming: StreamingConfig = .default,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> WhisperSession

    // MARK: - Transcription

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = 16000
    ) -> AsyncThrowingStream<StreamingResult, Error>

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = 16000
    ) async throws -> String

    // MARK: - Lifecycle

    public func cancel()
    public func cleanupMemory()
}
```

### Configuration

```swift
public struct StreamingConfig: Sendable {
    public var frameThreshold: Int
    public var minChunkDuration: TimeInterval
    public var emitPartial: Bool

    public static let `default` = StreamingConfig(
        frameThreshold: 25,
        minChunkDuration: 0.5,
        emitPartial: true
    )
}

public enum WhisperModel: String, CaseIterable, Sendable {
    case tiny = "mlx-community/whisper-tiny-mlx"
    case base = "mlx-community/whisper-base-mlx"
    case small = "mlx-community/whisper-small-mlx"
    case medium = "mlx-community/whisper-medium-mlx"
    case largeV3 = "mlx-community/whisper-large-v3-mlx"
    case largeTurbo = "mlx-community/whisper-large-v3-turbo"
}
```

### Usage Example

```swift
// Create session
let session = try await WhisperSession.fromPretrained(
    model: .largeTurbo,
    streaming: .default
) { progress in
    print("Loading: \(progress)")
}

// Streaming transcription
for try await result in session.transcribe(audioBuffer, sampleRate: 16000) {
    if result.isFinal {
        print("Final: \(result.text)")
    } else {
        print("Partial: \(result.text)")
    }
}
```

---

## 3. Module Structure

```
MLXAudio/
├── STT/
│   ├── STTSession.swift                 # Protocol
│   ├── StreamingResult.swift            # Output type
│   └── Whisper/
│       ├── WhisperSession.swift         # Main entry point
│       ├── WhisperModel.swift           # Model enum + config
│       ├── WhisperError.swift           # Error types
│       │
│       ├── Audio/
│       │   └── AudioProcessor.swift     # Mel spectrogram
│       │
│       ├── Model/
│       │   ├── AudioEncoder.swift       # Conv + Transformer encoder
│       │   ├── TextDecoder.swift        # Transformer decoder
│       │   └── MultiHeadAttention.swift # With cross-attention capture
│       │
│       ├── Decoding/
│       │   ├── DecodingOptions.swift    # Temperature, language, etc.
│       │   ├── GreedyDecoder.swift      # Token sampling
│       │   └── KVCache.swift            # Key-value cache
│       │
│       ├── Streaming/
│       │   ├── StreamingConfig.swift    # AlignAtt parameters
│       │   └── StreamingDecoder.swift   # AlignAtt logic
│       │
│       └── Tokenizer/
│           └── WhisperTokenizer.swift   # tiktoken wrapper
│
└── TTS/
    ├── Kokoro/
    ├── Orpheus/
    └── Marvis/
```

---

## 4. AlignAtt Streaming Implementation

### Core Algorithm

AlignAtt emits tokens **during decoding** by monitoring cross-attention weights, unlike VAD-based approaches that wait for chunk completion:

```
VAD Chunking:     [───chunk───] → decode → emit all
AlignAtt:         [audio stream] → decode → emit token → decode → emit token
```

### Key Functions

```swift
// StreamingDecoder.swift

public func getMostAttendedFrame(crossAttentionWeights: [MLXArray]) -> Int {
    // 1. Collect attention weights from alignment heads only
    let weights = alignmentHeads.map { (layer, head) in
        crossAttentionWeights[layer][0, head, -1, ...]  // Last token
    }

    // 2. Average across heads
    let avgAttention = MLX.stack(weights).mean(axis: 0)

    // 3. Find max — this is "current position" in audio
    return Int(MLX.argmax(avgAttention).item(Int.self))
}

public func shouldEmit(
    mostAttendedFrame: Int,
    totalContentFrames: Int
) -> Bool {
    // Emit when model has "moved past" threshold distance from audio end
    return (totalContentFrames - mostAttendedFrame) >= config.frameThreshold
}
```

### Alignment Heads by Model

```swift
public enum WhisperAlignmentHeads {
    public static func heads(for model: WhisperModel) -> [(layer: Int, head: Int)] {
        switch model {
        case .tiny:
            return [(2,2), (3,0), (3,2), (3,3), (3,4), (3,5)]
        case .base:
            return [(3,1), (4,2), (4,3), (4,7), (5,1), (5,2), (5,4), (5,6)]
        case .small:
            return [(5,3), (5,9), (8,0), (8,4), (8,7), (8,8), (9,0), (9,7), (9,9), (10,5)]
        case .medium:
            return [(13,15), (15,4), (15,15), (16,1), (20,0), (23,4)]
        case .largeV3:
            return [(10,12), (13,17), (16,11), ...] // 23 heads
        case .largeTurbo:
            // 4 decoder layers — needs empirical testing
            return [(0,0), (1,0), (2,0), (3,0)] // TODO: optimize
        }
    }
}
```

### Reference Implementation

Port from `mlx_audio/stt/models/whisper/streaming.py` (282 LoC):
- `get_most_attended_frame()`
- `should_emit()`
- `StreamingDecoder.decode_chunk()`

---

## 5. Error Handling

### WhisperError

```swift
public enum WhisperError: LocalizedError, Sendable {
    // Model Loading
    case modelNotFound(String)
    case modelDownloadFailed(URL, underlying: Error)
    case invalidModelFormat(String)

    // Audio Processing
    case invalidAudioFormat(expected: String, got: String)
    case audioTooShort(minSeconds: Double)

    // Transcription
    case encodingFailed(String)
    case decodingFailed(String)
    case cancelled

    // Tokenizer
    case tokenizerLoadFailed(String)
}
```

### Progress Reporting

```swift
public enum WhisperProgress: Sendable {
    case downloading(Float)
    case loading(Float)
    case encoding
    case decoding(Float)
}
```

### Cancellation

```swift
public func transcribe(...) -> AsyncThrowingStream<StreamingResult, Error> {
    AsyncThrowingStream { continuation in
        currentTask = Task {
            do {
                for try await token in decodeLoop() {
                    try Task.checkCancellation()
                    continuation.yield(token)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: WhisperError.cancelled)
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
```

---

## 6. Implementation Phases

### Phase 1: Core Foundation (5-7 days)

- [ ] `AudioProcessor.swift` — mel spectrogram (FFT + filterbank)
- [ ] `WhisperTokenizer.swift` — swift-tiktoken integration
- [ ] Model loading — HuggingFace download + weight loading

**Milestone**: Model loads + mel spectrogram from audio

### Phase 2: Whisper Model (5-7 days)

- [ ] `AudioEncoder.swift` — Conv1D + Transformer encoder
- [ ] `TextDecoder.swift` — Transformer decoder + cross-attention capture
- [ ] `MultiHeadAttention.swift` — self/cross attention + KV cache

**Milestone**: Batch transcription works (no streaming)

### Phase 3: AlignAtt Streaming (3-4 days)

- [ ] `StreamingDecoder.swift` — AlignAtt logic port
- [ ] `WhisperSession.swift` — AsyncThrowingStream API
- [ ] Integration — wire up all components

**Milestone**: Streaming transcription works

### Phase 4: Polish & Optimize (3-4 days)

- [ ] Performance — KV cache optimization, memory management
- [ ] Testing — unit tests, integration tests, benchmarks
- [ ] Documentation — API docs, examples, README

**Milestone**: Production-ready release

### Timeline

| Phase | Days | Cumulative |
|-------|------|------------|
| 1. Core | 5-7 | Week 1 |
| 2. Model | 5-7 | Week 2 |
| 3. Streaming | 3-4 | Week 2-3 |
| 4. Polish | 3-4 | Week 3 |
| **Total** | **16-22** | **~3 weeks** |

---

## 7. Testing Strategy

### Unit Tests

```swift
// AudioProcessorTests
func testMelSpectrogram_matchesPythonReference()

// WhisperTokenizerTests
func testEncode_englishText()
func testSOTSequence_multilingual()

// StreamingDecoderTests
func testShouldEmit_nearEnd()
func testShouldEmit_farFromEnd()
```

### Integration Tests

```swift
func testTranscribe_shortAudio()
func testTranscribeStreaming_emitsPartialResults()
func testCancel_stopsTranscription()
```

### Benchmark Tests

```swift
func testLatencyToFirstToken()  // Target: < 1.5s
func testRealTimeFactor()       // Target: RTF < 0.1
```

### Test Data

```
Tests/STT/Resources/
├── audio/
│   ├── hello_world_3s.wav
│   ├── speech_30s.wav
│   └── multilingual/
└── references/
    ├── mel_reference.npy
    └── expected_transcripts.json
```

---

## 8. Dependencies

| Dependency | Purpose | Status |
|------------|---------|--------|
| `MLX` | Tensor operations | Already in project |
| `MLXNN` | Neural network layers | Already in project |
| `MLXLMCommon` | Model loading | Already in project |
| `swift-tiktoken` | Whisper tokenizer | **New dependency** |
| `Accelerate` | vDSP for mel spectrogram | System framework |

---

## 9. References

- [AlignAtt Paper](https://arxiv.org/abs/2211.00895) — SimulMT with AlignAtt
- [SimulStreaming](https://github.com/ufal/SimulStreaming) — UFAL reference implementation
- [Lightning-SimulWhisper](https://github.com/altalt-org/Lightning-SimulWhisper) — MLX/CoreML implementation
- [Whisper Alignment Heads](https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a) — Head values by model
- [mlx-audio Python](https://github.com/Blaizzy/mlx-audio) — Our Python reference

---

## 10. Future Enhancements (Phase 2+)

| Feature | Benefit |
|---------|---------|
| CoreML Encoder | Battery efficiency (18x faster, lower power) |
| VAD Integration | Skip silence, reduce compute |
| CIF Model | Precise word boundaries |
| Beam Search | Higher accuracy |
| Multi-language | Auto language detection |

---

*Document approved for implementation.*
