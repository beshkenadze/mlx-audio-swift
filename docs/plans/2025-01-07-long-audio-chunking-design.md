# Long Audio Chunking Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable transcription of audio longer than 30 seconds using configurable chunking strategies

**Architecture:** Three pluggable chunking strategies (Sequential, VAD-based, Sliding Window) with unified LongAudioProcessor API. Two VAD providers (Energy-based, Silero MLX) for voice activity detection.

**Tech Stack:** MLX Swift, async/await, AsyncThrowingStream

---

## Table of Contents

1. [Overview](#overview)
2. [Performance Requirements](#performance-requirements)
3. [Error Handling & Failure Modes](#error-handling--failure-modes)
4. [VAD Provider Protocol](#vad-provider-protocol)
5. [VAD Implementations](#vad-implementations)
6. [ChunkingStrategy Protocol](#chunkingstrategy-protocol)
7. [Strategy Implementations](#strategy-implementations)
8. [LongAudioProcessor API](#longaudioprocessor-api)
9. [Telemetry & Observability](#telemetry--observability)
10. [Testing Strategy](#testing-strategy)
11. [Usage Examples](#usage-examples)
12. [Implementation Tasks](#implementation-tasks)

---

## Overview

### The 30-Second Limitation

Whisper has a **fixed 30-second context window**:
- Input: 3000 mel frames (30 sec × 100 fps)
- After Conv2 stride-2: 1500 frames (`nAudioCtx`)
- Positional embeddings: Fixed size array [1500, nAudioState]

### Three Chunking Strategies

| Aspect | Sequential | VAD-Based | Sliding Window |
|--------|-----------|-----------|----------------|
| **Accuracy** | Best | High | Good |
| **Speed** | 1x | 4-12x (parallel) | 2-4x (parallel) |
| **Latency** | High | Medium | Low (predictable) |
| **Dependencies** | None | VAD model | None |
| **Context** | Yes (prev text) | No | No (parallel-safe) |
| **Best for** | Quality-critical | Noisy/long audio | Real-time |

### Two VAD Providers

| Aspect | EnergyVAD | SileroMLXVAD |
|--------|-----------|--------------|
| **Dependencies** | Zero | MLX (already have) |
| **Accuracy** | Medium (noise-sensitive) | High (voice-specific) |
| **Model size** | None | ~2MB weights |
| **Startup** | Instant | Model load time |
| **Best for** | Clean audio, quick tests | Production, noisy audio |

---

## Performance Requirements

### Latency Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Time to first result** | < 2.0s | For streaming use cases |
| **Sequential RTF** | < 0.5 | Real-time factor (processing time / audio duration) |
| **Parallel RTF** | < 0.15 | With 4 concurrent chunks |
| **VAD overhead** | < 100ms | Per minute of audio |

### Memory Budget

| Component | Budget | Notes |
|-----------|--------|-------|
| **Single chunk** | ~500MB | Encoder + decoder + KV cache |
| **Parallel (4x)** | ~1.5GB | Shared encoder, separate KV caches |
| **VAD model** | ~50MB | Silero MLX weights + state |
| **Audio buffer** | ~10MB/min | 16-bit mono @ 16kHz |

### Resource Limits Configuration

```swift
/// Resource governance for long audio processing
public struct ProcessingLimits: Sendable {
    /// Maximum concurrent chunk transcriptions
    public var maxConcurrentChunks: Int = 4
    /// Timeout for single chunk transcription
    public var chunkTimeout: TimeInterval = 60
    /// Total processing timeout (nil = unlimited)
    public var totalTimeout: TimeInterval? = nil
    /// Maximum memory budget in MB (nil = unlimited)
    public var maxMemoryMB: Int? = nil
    /// Whether to abort on first chunk failure
    public var abortOnFirstFailure: Bool = false

    public static let `default` = ProcessingLimits()

    public static let conservative = ProcessingLimits(
        maxConcurrentChunks: 2,
        chunkTimeout: 30,
        maxMemoryMB: 1024
    )

    public static let aggressive = ProcessingLimits(
        maxConcurrentChunks: 8,
        chunkTimeout: 120,
        maxMemoryMB: 4096
    )
}
```

---

## Error Handling & Failure Modes

### Error Types

```swift
/// Errors from chunking and long audio processing
public enum ChunkingError: Error, Sendable {
    // VAD errors
    case vadFailed(underlying: Error)
    case vadModelLoadFailed(String)

    // Chunk processing errors
    case chunkTranscriptionFailed(chunkIndex: Int, timeRange: ClosedRange<TimeInterval>, underlying: Error)
    case chunkTimeout(chunkIndex: Int, timeRange: ClosedRange<TimeInterval>)

    // Resource errors
    case resourceExhausted(ResourceType)
    case totalTimeoutExceeded(processedDuration: TimeInterval, totalDuration: TimeInterval)

    // Input validation
    case audioTooShort(minimum: TimeInterval, actual: TimeInterval)
    case invalidSampleRate(expected: Int, got: Int)

    // Cancellation
    case cancelled(partialResult: PartialTranscriptionResult?)

    public enum ResourceType: Sendable {
        case memory(requestedMB: Int, availableMB: Int)
        case concurrency(requested: Int, limit: Int)
    }
}

/// Partial result available when cancelled or failed mid-stream
public struct PartialTranscriptionResult: Sendable {
    public let text: String
    public let processedDuration: TimeInterval
    public let totalDuration: TimeInterval
    public let completedChunks: Int
    public let totalChunks: Int
}
```

### Failure Mode Analysis

| Failure | Detection | Recovery | User Impact |
|---------|-----------|----------|-------------|
| VAD model load fails | `throws` on init | Fallback to `EnergyVADProvider` | Degraded accuracy |
| Chunk transcription timeout | Watchdog timer | Skip chunk, emit warning, continue | Gap in transcript |
| Chunk transcription error | `throws` from transcriber | Retry once, then skip with warning | Gap in transcript |
| Memory pressure | MLX allocation fails | Reduce concurrency, retry | Slower processing |
| Corrupted audio segment | NaN in mel spectrogram | Skip segment, emit warning | Missing content |
| Total timeout exceeded | Timer check | Emit partial result, finish | Incomplete transcript |
| Cancellation requested | `Task.checkCancellation()` | Emit partial result, clean up | Incomplete transcript |

### Cancellation Semantics

```swift
/// Cancellation behavior configuration
public struct CancellationPolicy: Sendable {
    /// Whether to emit partial results on cancellation
    public var emitPartialOnCancel: Bool = true
    /// Grace period before force-cancelling in-flight chunks
    public var gracePeriod: TimeInterval = 1.0
    /// Whether to wait for current chunk to complete
    public var waitForCurrentChunk: Bool = true

    public static let `default` = CancellationPolicy()
}
```

**Cancellation flow:**
1. `Task.cancel()` called on transcription stream
2. Set cancellation flag, stop starting new chunks
3. If `waitForCurrentChunk`: wait up to `gracePeriod` for in-flight chunks
4. If `emitPartialOnCancel`: yield final result with `isFinal: true` and partial text
5. Clean up resources (KV caches, audio buffers)
6. Throw `ChunkingError.cancelled(partialResult:)`

---

## VAD Provider Protocol

### Core Types

```swift
/// A detected speech segment with timing and confidence
public struct SpeechSegment: Sendable, Equatable {
    public let start: TimeInterval
    public let end: TimeInterval
    public let confidence: Float

    public var duration: TimeInterval { end - start }

    public init(start: TimeInterval, end: TimeInterval, confidence: Float) {
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

/// Configuration for segmenting speech probabilities
public struct VADSegmentConfig: Sendable {
    /// Minimum duration to consider as speech
    public var minSpeechDuration: TimeInterval = 0.25
    /// Minimum silence duration to split segments
    public var minSilenceDuration: TimeInterval = 0.3
    /// Maximum segment duration (for chunking)
    public var maxSegmentDuration: TimeInterval = 30.0
    /// Padding to add around speech segments
    public var speechPadding: TimeInterval = 0.1

    public static let `default` = VADSegmentConfig()
}

/// Errors from VAD processing
public enum VADError: Error, Sendable {
    case sampleRateMismatch(expected: Int, got: Int)
    case modelLoadFailed(String)
    case modelOutputMissing
    case audioTooShort(minimum: TimeInterval)
}
```

### Protocol Definition

```swift
/// Protocol for Voice Activity Detection providers
public protocol VADProvider: Sendable {
    /// Detect speech segments in audio
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray
    ///   - sampleRate: Sample rate (typically 16000)
    /// - Returns: Array of detected speech segments
    func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment]

    /// Get frame-level speech probabilities
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray
    ///   - sampleRate: Sample rate (typically 16000)
    /// - Returns: Array of (time, probability) tuples
    func speechProbabilities(in audio: MLXArray, sampleRate: Int) async throws -> [(time: TimeInterval, probability: Float)]

    /// Reset internal state (for stateful models like Silero LSTM)
    func reset() async
}
```

---

## VAD Implementations

### 1. EnergyVADProvider

Zero dependencies, pure signal processing (~50 lines):

```swift
/// Energy-based Voice Activity Detection
/// Uses RMS (Root Mean Square) energy to detect speech segments
public final class EnergyVADProvider: VADProvider, @unchecked Sendable {
    private let config: EnergyVADConfig

    public struct EnergyVADConfig: Sendable {
        /// RMS threshold for speech detection (0.0-1.0)
        public var speechThreshold: Float = 0.02
        /// Window size in seconds for RMS calculation
        public var windowDuration: TimeInterval = 0.025  // 25ms
        /// Hop size in seconds between windows
        public var hopDuration: TimeInterval = 0.010     // 10ms
        /// Smoothing factor for energy (0=none, 1=max)
        public var smoothingFactor: Float = 0.9

        public static let `default` = EnergyVADConfig()
    }

    public init(config: EnergyVADConfig = .default) {
        self.config = config
    }

    public func detectSpeech(
        in audio: MLXArray,
        sampleRate: Int
    ) async throws -> [SpeechSegment] {
        let probabilities = try await speechProbabilities(in: audio, sampleRate: sampleRate)
        return segmentFromProbabilities(probabilities, threshold: config.speechThreshold)
    }

    public func speechProbabilities(
        in audio: MLXArray,
        sampleRate: Int
    ) async throws -> [(time: TimeInterval, probability: Float)] {
        let windowSamples = Int(config.windowDuration * Double(sampleRate))
        let hopSamples = Int(config.hopDuration * Double(sampleRate))
        let totalSamples = audio.shape[0]

        var results: [(time: TimeInterval, probability: Float)] = []
        var smoothedEnergy: Float = 0

        var position = 0
        while position + windowSamples <= totalSamples {
            // Extract window and compute RMS
            let window = audio[position..<(position + windowSamples)]
            let squared = window * window
            let rms = sqrt(Float(MLX.mean(squared).item(Float.self)))

            // Apply smoothing
            smoothedEnergy = config.smoothingFactor * smoothedEnergy + (1 - config.smoothingFactor) * rms

            // Normalize to probability (sigmoid-like mapping)
            let probability = min(1.0, smoothedEnergy / config.speechThreshold)

            let time = Double(position) / Double(sampleRate)
            results.append((time: time, probability: probability))

            position += hopSamples
        }

        return results
    }

    public func reset() async {
        // No state to reset for energy-based VAD
    }
}
```

### 2. SileroMLXVADProvider

MLX-native Silero VAD port (~100 lines wrapper + model):

```swift
import MLX
import MLXNN

/// Silero VAD ported to MLX
/// Architecture: Conv1D stack + 2-layer LSTM (64 units)
public final class SileroMLXVADProvider: VADProvider, @unchecked Sendable {
    private let model: SileroVADModel
    private var hiddenState: MLXArray?
    private var cellState: MLXArray?
    private let stateLock = NSLock()
    private let config: SileroVADConfig

    public struct SileroVADConfig: Sendable {
        public var speechThreshold: Float = 0.5
        public var chunkDuration: TimeInterval = 0.064  // 64ms
        public var sampleRate: Int = 16000
        public static let `default` = SileroVADConfig()
    }

    public init(config: SileroVADConfig = .default) async throws {
        self.config = config
        self.model = try await SileroVADModel.fromPretrained()
        resetStates()
    }

    private func resetStates() {
        // LSTM: 2 layers, 64 hidden units
        hiddenState = MLXArray.zeros([2, 1, 64])
        cellState = MLXArray.zeros([2, 1, 64])
    }

    public func speechProbabilities(
        in audio: MLXArray,
        sampleRate: Int
    ) async throws -> [(time: TimeInterval, probability: Float)] {
        guard sampleRate == config.sampleRate else {
            throw VADError.sampleRateMismatch(expected: config.sampleRate, got: sampleRate)
        }

        let chunkSamples = Int(config.chunkDuration * Double(sampleRate))
        let totalSamples = audio.shape[0]
        var results: [(time: TimeInterval, probability: Float)] = []
        var position = 0

        while position + chunkSamples <= totalSamples {
            let chunk = audio[position..<(position + chunkSamples)]
            let batchedChunk = chunk.reshaped([1, chunkSamples])

            // Get current states
            let (h, c) = stateLock.withLock { (hiddenState!, cellState!) }

            // Forward pass
            let (prob, newH, newC) = model(batchedChunk, hiddenState: h, cellState: c)

            // Update states
            stateLock.withLock {
                hiddenState = newH
                cellState = newC
            }

            let time = Double(position) / Double(sampleRate)
            results.append((time: time, probability: Float(prob.item(Float.self))))

            position += chunkSamples
        }

        return results
    }

    public func detectSpeech(
        in audio: MLXArray,
        sampleRate: Int
    ) async throws -> [SpeechSegment] {
        let probabilities = try await speechProbabilities(in: audio, sampleRate: sampleRate)
        return segmentFromProbabilities(probabilities, threshold: config.speechThreshold)
    }

    public func reset() async {
        stateLock.withLock { resetStates() }
    }
}
```

### SileroVADModel (MLX Architecture)

```swift
/// Silero VAD v5 architecture in MLX
/// Based on: https://github.com/snakers4/silero-vad
public class SileroVADModel: Module {
    // Encoder: Conv1D stack
    @ModuleInfo var conv1: Conv1d  // 1 -> 64
    @ModuleInfo var conv2: Conv1d  // 64 -> 64
    @ModuleInfo var conv3: Conv1d  // 64 -> 128
    @ModuleInfo var conv4: Conv1d  // 128 -> 64

    // LSTM: 2 layers, 64 hidden
    @ModuleInfo var lstm: LSTM

    // Output projection
    @ModuleInfo var fc: Linear  // 64 -> 1

    public init() {
        _conv1.wrappedValue = Conv1d(inputChannels: 1, outputChannels: 64, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1)
        _conv3.wrappedValue = Conv1d(inputChannels: 64, outputChannels: 128, kernelSize: 3, padding: 1)
        _conv4.wrappedValue = Conv1d(inputChannels: 128, outputChannels: 64, kernelSize: 3, padding: 1)
        _lstm.wrappedValue = LSTM(inputSize: 64, hiddenSize: 64, numLayers: 2, batchFirst: true)
        _fc.wrappedValue = Linear(64, 1)
    }

    public func callAsFunction(
        _ input: MLXArray,
        hiddenState: MLXArray,
        cellState: MLXArray
    ) -> (probability: MLXArray, hiddenState: MLXArray, cellState: MLXArray) {
        // input: [batch, samples]
        var x = input.expandedDimensions(axis: -1)  // [batch, samples, 1]

        // Conv encoder
        x = relu(conv1(x))
        x = relu(conv2(x))
        x = relu(conv3(x))
        x = relu(conv4(x))

        // Global average pooling over time
        x = x.mean(axis: 1, keepDims: true)  // [batch, 1, 64]

        // LSTM with state
        let (output, newH, newC) = lstm(x, hiddenState: hiddenState, cellState: cellState)

        // Output projection + sigmoid
        let logits = fc(output.squeezed(axis: 1))  // [batch, 1]
        let prob = sigmoid(logits)

        return (prob.squeezed(), newH, newC)
    }

    public static func fromPretrained() async throws -> SileroVADModel {
        // Load from HuggingFace or convert weights
        let model = SileroVADModel()
        // TODO: Weight loading similar to WhisperModelLoader
        return model
    }
}
```

### Shared Segmentation Utilities

```swift
/// Convert frame-level probabilities to speech segments
func segmentFromProbabilities(
    _ probabilities: [(time: TimeInterval, probability: Float)],
    threshold: Float,
    config: VADSegmentConfig = .default
) -> [SpeechSegment] {
    var segments: [SpeechSegment] = []
    var segmentStart: TimeInterval?
    var maxConfidence: Float = 0

    for (time, prob) in probabilities {
        if prob >= threshold {
            if segmentStart == nil {
                segmentStart = max(0, time - config.speechPadding)
            }
            maxConfidence = max(maxConfidence, prob)
        } else if let start = segmentStart {
            let end = time + config.speechPadding
            let duration = end - start

            if duration >= config.minSpeechDuration {
                segments.append(SpeechSegment(
                    start: start,
                    end: end,
                    confidence: maxConfidence
                ))
            }
            segmentStart = nil
            maxConfidence = 0
        }
    }

    // Handle segment at end of audio
    if let start = segmentStart, let lastTime = probabilities.last?.time {
        let duration = lastTime - start
        if duration >= config.minSpeechDuration {
            segments.append(SpeechSegment(
                start: start,
                end: lastTime,
                confidence: maxConfidence
            ))
        }
    }

    return mergeCloseSegments(segments, minSilence: config.minSilenceDuration)
}

/// Merge segments that are closer than minSilence
func mergeCloseSegments(
    _ segments: [SpeechSegment],
    minSilence: TimeInterval
) -> [SpeechSegment] {
    guard !segments.isEmpty else { return [] }

    var merged: [SpeechSegment] = []
    var current = segments[0]

    for next in segments.dropFirst() {
        if next.start - current.end < minSilence {
            // Merge
            current = SpeechSegment(
                start: current.start,
                end: next.end,
                confidence: max(current.confidence, next.confidence)
            )
        } else {
            merged.append(current)
            current = next
        }
    }
    merged.append(current)

    return merged
}
```

---

## ChunkingStrategy Protocol

### Core Types

```swift
/// Result from processing a single chunk
public struct ChunkResult: Sendable {
    /// Transcribed text for this chunk
    public let text: String
    /// Token IDs
    public let tokens: [Int]
    /// Time range relative to original audio
    public let timeRange: ClosedRange<TimeInterval>
    /// Confidence score (0.0-1.0)
    public let confidence: Float
    /// Word-level timestamps (if available)
    public let words: [WordTimestamp]?
}

public struct WordTimestamp: Sendable, Equatable {
    public let word: String
    public let start: TimeInterval
    public let end: TimeInterval
    public let confidence: Float
}
```

### Protocol Definition

```swift
/// Protocol for long audio chunking strategies
public protocol ChunkingStrategy: Sendable {
    /// Process long audio and yield results as chunks complete
    func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error>

    /// Strategy identifier for logging/debugging
    var name: String { get }

    /// Transcription mode (affects context handling)
    var transcriptionMode: TranscriptionMode { get }
}

/// How chunks relate to each other for context
public enum TranscriptionMode: Sendable {
    /// Chunks are independent, can be parallelized
    case independent
    /// Chunks use previous tokens as context, must be sequential
    case sequential
}

/// Abstraction for transcribing a single ≤30s chunk
/// Public protocol to enable alternative backends and testing
public protocol ChunkTranscriber: Sendable {
    func transcribe(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?
    ) async throws -> ChunkResult
}
```

---

## Strategy Implementations

### 1. SequentialChunkingStrategy

OpenAI-style seek-based decoding with timestamp token conditioning:

```swift
/// Sequential decoding with timestamp-based seeking
/// Best accuracy, no parallelization
public final class SequentialChunkingStrategy: ChunkingStrategy {
    public let name = "sequential"
    public let transcriptionMode = TranscriptionMode.sequential
    private let config: SequentialConfig

    public struct SequentialConfig: Sendable {
        /// Use previous transcription as decoder context
        public var conditionOnPreviousText: Bool = true
        /// Max tokens to use as context from previous chunk
        public var maxPreviousTokens: Int = 224
        /// Temperature for context reset on repetition
        public var contextResetTemperature: Float = 0.5
        /// Compression ratio threshold (skip if exceeded)
        public var compressionRatioThreshold: Float = 2.4
        /// Log probability threshold (skip if below)
        public var logprobThreshold: Float = -1.0
        /// No speech probability threshold
        public var noSpeechThreshold: Float = 0.6
        /// Optional initial prompt for first chunk
        public var initialPrompt: String? = nil

        public static let `default` = SequentialConfig()
    }

    public init(config: SequentialConfig = .default) {
        self.config = config
    }

    public func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let startTime = Date()
                var chunkIndex = 0

                do {
                    let totalSamples = audio.shape[0]
                    let maxSamples = 30 * sampleRate
                    let totalDuration = Double(totalSamples) / Double(sampleRate)

                    telemetry?.strategyStarted(name, audioDuration: totalDuration)

                    var seekSample = 0
                    var previousTokens: [Int]? = nil

                    while seekSample < totalSamples {
                        try Task.checkCancellation()

                        // Check total timeout
                        if let timeout = limits.totalTimeout {
                            let elapsed = Date().timeIntervalSince(startTime)
                            if elapsed > timeout {
                                throw ChunkingError.totalTimeoutExceeded(
                                    processedDuration: Double(seekSample) / Double(sampleRate),
                                    totalDuration: totalDuration
                                )
                            }
                        }

                        let endSample = min(seekSample + maxSamples, totalSamples)
                        let chunk = audio[seekSample..<endSample]
                        let timeRange = (Double(seekSample) / Double(sampleRate))...(Double(endSample) / Double(sampleRate))

                        telemetry?.chunkStarted(index: chunkIndex, timeRange: timeRange)

                        let context = config.conditionOnPreviousText ? previousTokens : nil

                        // Transcribe with timeout
                        let result = try await withTimeout(limits.chunkTimeout) {
                            try await transcriber.transcribe(
                                audio: chunk,
                                sampleRate: sampleRate,
                                previousTokens: context
                            )
                        }

                        if shouldSkipResult(result) {
                            seekSample += sampleRate
                            continue
                        }

                        let absoluteStart = Double(seekSample) / Double(sampleRate)
                        let adjustedResult = adjustTimestamps(result, offset: absoluteStart)

                        telemetry?.chunkCompleted(
                            index: chunkIndex,
                            duration: Date().timeIntervalSince(startTime),
                            text: adjustedResult.text
                        )

                        continuation.yield(adjustedResult)

                        let lastTimestamp = result.timeRange.upperBound
                        seekSample += Int(lastTimestamp * Double(sampleRate))

                        if config.conditionOnPreviousText {
                            let tokenCount = min(result.tokens.count, config.maxPreviousTokens)
                            previousTokens = Array(result.tokens.suffix(tokenCount))
                        }

                        chunkIndex += 1
                    }

                    telemetry?.strategyCompleted(totalChunks: chunkIndex, totalDuration: Date().timeIntervalSince(startTime))
                    continuation.finish()
                } catch {
                    telemetry?.error(error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func shouldSkipResult(_ result: ChunkResult) -> Bool {
        result.confidence < 0.1 || result.text.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func adjustTimestamps(_ result: ChunkResult, offset: TimeInterval) -> ChunkResult {
        ChunkResult(
            text: result.text,
            tokens: result.tokens,
            timeRange: (offset + result.timeRange.lowerBound)...(offset + result.timeRange.upperBound),
            confidence: result.confidence,
            words: result.words?.map { word in
                WordTimestamp(
                    word: word.word,
                    start: offset + word.start,
                    end: offset + word.end,
                    confidence: word.confidence
                )
            }
        )
    }
}
```

### 2. VADChunkingStrategy

WhisperX-style VAD segmentation with parallel processing:

```swift
/// VAD-based chunking with parallel transcription
/// Best for noisy audio, fastest with batching
public final class VADChunkingStrategy: ChunkingStrategy {
    public let name = "vad"
    public let transcriptionMode = TranscriptionMode.independent
    private let config: VADConfig
    private let vadProvider: VADProvider

    public struct VADConfig: Sendable {
        /// Target chunk duration (will merge short segments)
        public var targetChunkDuration: TimeInterval = 30.0
        /// Maximum chunk duration (hard limit)
        public var maxChunkDuration: TimeInterval = 30.0
        /// Minimum speech duration to keep
        public var minSpeechDuration: TimeInterval = 0.5
        /// Padding around speech segments
        public var speechPadding: TimeInterval = 0.2
        /// Enable parallel transcription
        public var parallelProcessing: Bool = true

        public static let `default` = VADConfig()
    }

    public init(
        vadProvider: VADProvider,
        config: VADConfig = .default
    ) {
        self.vadProvider = vadProvider
        self.config = config
    }

    public func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let startTime = Date()

                do {
                    let totalDuration = Double(audio.shape[0]) / Double(sampleRate)
                    telemetry?.strategyStarted(name, audioDuration: totalDuration)

                    // 1. Run VAD to get speech segments
                    let segments = try await vadProvider.detectSpeech(in: audio, sampleRate: sampleRate)
                    telemetry?.vadSegmentsDetected(
                        count: segments.count,
                        totalSpeechDuration: segments.reduce(0) { $0 + $1.duration }
                    )

                    // 2. Merge/split segments to target duration
                    let chunks = prepareChunks(segments: segments, audioDuration: totalDuration)

                    // 3. Process chunks
                    if config.parallelProcessing {
                        try await processParallel(
                            chunks: chunks,
                            audio: audio,
                            sampleRate: sampleRate,
                            transcriber: transcriber,
                            limits: limits,
                            telemetry: telemetry,
                            continuation: continuation
                        )
                    } else {
                        try await processSequential(
                            chunks: chunks,
                            audio: audio,
                            sampleRate: sampleRate,
                            transcriber: transcriber,
                            limits: limits,
                            telemetry: telemetry,
                            continuation: continuation
                        )
                    }

                    telemetry?.strategyCompleted(
                        totalChunks: chunks.count,
                        totalDuration: Date().timeIntervalSince(startTime)
                    )
                    continuation.finish()
                } catch {
                    telemetry?.error(error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // ... (chunk preparation and parallel processing methods)
}
```

### 3. SlidingWindowChunkingStrategy

Fixed window with overlap and merge:

```swift
/// Sliding window with fixed overlap
/// Predictable latency, good for real-time
public final class SlidingWindowChunkingStrategy: ChunkingStrategy {
    public let name = "slidingWindow"
    public let transcriptionMode = TranscriptionMode.independent
    private let config: SlidingWindowConfig

    public struct SlidingWindowConfig: Sendable {
        /// Window duration (max 30s for Whisper)
        public var windowDuration: TimeInterval = 30.0
        /// Overlap between windows
        public var overlapDuration: TimeInterval = 5.0
        /// Computed hop size
        public var hopDuration: TimeInterval { windowDuration - overlapDuration }
        /// Strategy for merging overlap regions
        public var mergeStrategy: MergeStrategy = .timestampAlignment

        public enum MergeStrategy: Sendable {
            case simple           // Just concatenate, dedupe obvious repeats
            case timestampAlignment  // Align using word timestamps
            case lcs              // Longest common subsequence matching
        }

        public static let `default` = SlidingWindowConfig()
    }

    public init(config: SlidingWindowConfig = .default) {
        self.config = config
    }

    public func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let startTime = Date()
                var chunkIndex = 0

                do {
                    let totalDuration = Double(audio.shape[0]) / Double(sampleRate)
                    var position: TimeInterval = 0
                    var previousResult: ChunkResult?
                    var accumulatedText = ""
                    var accumulatedWords: [WordTimestamp] = []

                    telemetry?.strategyStarted(name, audioDuration: totalDuration)

                    while position < totalDuration {
                        try Task.checkCancellation()

                        // Check total timeout
                        if let timeout = limits.totalTimeout {
                            let elapsed = Date().timeIntervalSince(startTime)
                            if elapsed > timeout {
                                throw ChunkingError.totalTimeoutExceeded(
                                    processedDuration: position,
                                    totalDuration: totalDuration
                                )
                            }
                        }

                        let windowEnd = min(position + config.windowDuration, totalDuration)
                        let startSample = Int(position * Double(sampleRate))
                        let endSample = Int(windowEnd * Double(sampleRate))
                        let chunk = audio[startSample..<endSample]
                        let timeRange = position...windowEnd

                        telemetry?.chunkStarted(index: chunkIndex, timeRange: timeRange)

                        let result = try await withTimeout(limits.chunkTimeout) {
                            try await transcriber.transcribe(
                                audio: chunk,
                                sampleRate: sampleRate,
                                previousTokens: nil
                            )
                        }

                        let adjustedResult = adjustTimestamps(result, offset: position)

                        if let prev = previousResult {
                            let merged = mergeResults(
                                previous: prev,
                                current: adjustedResult,
                                overlapStart: position,
                                overlapEnd: prev.timeRange.upperBound
                            )
                            accumulatedText = merged.text
                            accumulatedWords = merged.words ?? []
                        } else {
                            accumulatedText = adjustedResult.text
                            accumulatedWords = adjustedResult.words ?? []
                        }

                        let progressResult = ChunkResult(
                            text: accumulatedText,
                            tokens: [],
                            timeRange: 0...windowEnd,
                            confidence: adjustedResult.confidence,
                            words: accumulatedWords
                        )

                        telemetry?.chunkCompleted(
                            index: chunkIndex,
                            duration: Date().timeIntervalSince(startTime),
                            text: progressResult.text
                        )

                        continuation.yield(progressResult)

                        previousResult = adjustedResult
                        position += config.hopDuration
                        chunkIndex += 1
                    }

                    telemetry?.strategyCompleted(
                        totalChunks: chunkIndex,
                        totalDuration: Date().timeIntervalSince(startTime)
                    )
                    continuation.finish()
                } catch {
                    telemetry?.error(error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // ... (merge strategy implementations)
}
```

---

## LongAudioProcessor API

### Unified Interface

```swift
/// Unified API for transcribing audio of any length
public final class LongAudioProcessor: @unchecked Sendable {
    private let session: WhisperSession
    private let strategy: ChunkingStrategy
    private let mergeConfig: MergeConfig
    private let limits: ProcessingLimits
    private let telemetry: ChunkingTelemetry?

    public struct MergeConfig: Sendable {
        public var deduplicateOverlap: Bool = true
        public var minWordConfidence: Float = 0.3
        public var normalizeText: Bool = true
        public static let `default` = MergeConfig()
    }

    // MARK: - Strategy Types

    public enum StrategyType: Sendable {
        case auto
        case sequential(SequentialChunkingStrategy.SequentialConfig = .default)
        case vad(VADChunkingStrategy.VADConfig = .default, vadProvider: VADProviderType = .energy)
        case slidingWindow(SlidingWindowChunkingStrategy.SlidingWindowConfig = .default)
    }

    public enum VADProviderType: Sendable {
        case energy
        case sileroMLX
    }

    // MARK: - Factory

    public static func create(
        model: WhisperModel = .largeTurbo,
        strategy: StrategyType = .auto,
        limits: ProcessingLimits = .default,
        telemetry: ChunkingTelemetry? = nil,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> LongAudioProcessor

    // MARK: - Transcription

    /// Streaming transcription with progress
    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) -> AsyncThrowingStream<TranscriptionProgress, Error>

    /// Blocking transcription returning final result
    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult

    /// Cancel any in-progress transcription
    public func cancel()
}
```

### Result Types

```swift
public struct TranscriptionProgress: Sendable {
    public let text: String
    public let words: [WordTimestamp]?
    public let isFinal: Bool
    public let processedDuration: TimeInterval
    public let audioDuration: TimeInterval
    public let chunkIndex: Int
    public let totalChunks: Int

    public var progress: Float {
        Float(processedDuration / audioDuration)
    }
}

public struct TranscriptionResult: Sendable {
    public let text: String
    public let words: [WordTimestamp]?
    public let duration: TimeInterval
    public let language: String?
}
```

---

## Telemetry & Observability

### Telemetry Protocol

```swift
/// Protocol for observability and debugging of chunking operations
public protocol ChunkingTelemetry: Sendable {
    /// Called when a chunking strategy begins processing
    func strategyStarted(_ strategyName: String, audioDuration: TimeInterval)

    /// Called when a chunk begins processing
    func chunkStarted(index: Int, timeRange: ClosedRange<TimeInterval>)

    /// Called when a chunk completes successfully
    func chunkCompleted(index: Int, duration: TimeInterval, text: String)

    /// Called when a chunk fails
    func chunkFailed(index: Int, error: Error)

    /// Called when VAD completes segment detection
    func vadSegmentsDetected(count: Int, totalSpeechDuration: TimeInterval)

    /// Called when strategy completes all chunks
    func strategyCompleted(totalChunks: Int, totalDuration: TimeInterval)

    /// Called on any error
    func error(_ error: Error)
}

/// Default implementation that logs to console
public final class ConsoleTelemetry: ChunkingTelemetry, @unchecked Sendable {
    public init() {}

    public func strategyStarted(_ strategyName: String, audioDuration: TimeInterval) {
        print("[Chunking] Strategy '\(strategyName)' started for \(String(format: "%.1f", audioDuration))s audio")
    }

    public func chunkStarted(index: Int, timeRange: ClosedRange<TimeInterval>) {
        print("[Chunking] Chunk \(index) started: \(String(format: "%.1f", timeRange.lowerBound))-\(String(format: "%.1f", timeRange.upperBound))s")
    }

    public func chunkCompleted(index: Int, duration: TimeInterval, text: String) {
        let preview = text.prefix(50) + (text.count > 50 ? "..." : "")
        print("[Chunking] Chunk \(index) completed in \(String(format: "%.2f", duration))s: \"\(preview)\"")
    }

    public func chunkFailed(index: Int, error: Error) {
        print("[Chunking] Chunk \(index) failed: \(error)")
    }

    public func vadSegmentsDetected(count: Int, totalSpeechDuration: TimeInterval) {
        print("[Chunking] VAD detected \(count) segments, \(String(format: "%.1f", totalSpeechDuration))s of speech")
    }

    public func strategyCompleted(totalChunks: Int, totalDuration: TimeInterval) {
        print("[Chunking] Completed \(totalChunks) chunks in \(String(format: "%.2f", totalDuration))s")
    }

    public func error(_ error: Error) {
        print("[Chunking] Error: \(error)")
    }
}
```

---

## Testing Strategy

### Test Doubles

```swift
/// Mock transcriber for strategy testing without loading models
public final class MockChunkTranscriber: ChunkTranscriber, @unchecked Sendable {
    /// Fixed result to return for all chunks
    public var fixedResult: ChunkResult?
    /// Results to return in sequence (cycles if more chunks than results)
    public var sequentialResults: [ChunkResult] = []
    /// Artificial delay before returning result
    public var transcribeDelay: TimeInterval = 0
    /// Whether to throw an error
    public var shouldFail: Bool = false
    /// Error to throw when shouldFail is true
    public var errorToThrow: Error = MockError.intentional
    /// Track all transcribe calls for verification
    public private(set) var transcribeCalls: [(audio: MLXArray, sampleRate: Int, previousTokens: [Int]?)] = []

    private var callIndex = 0
    private let lock = NSLock()

    public enum MockError: Error {
        case intentional
    }

    public init() {}

    public func transcribe(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?
    ) async throws -> ChunkResult {
        lock.withLock {
            transcribeCalls.append((audio, sampleRate, previousTokens))
        }

        if shouldFail {
            throw errorToThrow
        }

        if transcribeDelay > 0 {
            try await Task.sleep(for: .seconds(transcribeDelay))
        }

        if let fixed = fixedResult {
            return fixed
        }

        if !sequentialResults.isEmpty {
            let index = lock.withLock {
                let i = callIndex
                callIndex = (callIndex + 1) % sequentialResults.count
                return i
            }
            return sequentialResults[index]
        }

        // Default mock result
        let duration = Double(audio.shape[0]) / Double(sampleRate)
        return ChunkResult(
            text: "Mock transcription for chunk",
            tokens: [1, 2, 3],
            timeRange: 0...duration,
            confidence: 0.95,
            words: nil
        )
    }

    public func reset() {
        lock.withLock {
            transcribeCalls = []
            callIndex = 0
        }
    }
}

/// Mock VAD provider for testing
public final class MockVADProvider: VADProvider, @unchecked Sendable {
    public var segments: [SpeechSegment] = []
    public var probabilities: [(time: TimeInterval, probability: Float)] = []
    public var shouldFail: Bool = false

    public init() {}

    public func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment] {
        if shouldFail { throw VADError.modelLoadFailed("Mock failure") }
        return segments
    }

    public func speechProbabilities(in audio: MLXArray, sampleRate: Int) async throws -> [(time: TimeInterval, probability: Float)] {
        if shouldFail { throw VADError.modelLoadFailed("Mock failure") }
        return probabilities
    }

    public func reset() async {}
}
```

### Test Scenarios

#### VAD Provider Tests

```swift
// Test: EnergyVADProvider detects speech segments accurately
// Given: 60 seconds of audio with 3 speech segments (5s, 20s, 10s) separated by 5s silence
// When: EnergyVADProvider.detectSpeech() is called with threshold 0.02
// Then: Returns 3 SpeechSegments with ±0.5s accuracy on boundaries

// Test: EnergyVADProvider handles silence-only audio
// Given: 30 seconds of silence (all zeros)
// When: EnergyVADProvider.detectSpeech() is called
// Then: Returns empty array

// Test: SileroMLXVADProvider maintains LSTM state across chunks
// Given: Audio split into 64ms chunks
// When: Processing sequentially
// Then: State is preserved between chunks, results match single-pass processing
```

#### Sliding Window Tests

```swift
// Test: SlidingWindowChunkingStrategy merges overlapping text correctly
// Given: Two chunks with 5s overlap containing "hello world" repeated at boundary
// When: timestampAlignment merge is applied
// Then: Output contains "hello world" exactly once

// Test: SlidingWindowChunkingStrategy handles audio shorter than window
// Given: 10 seconds of audio (less than 30s window)
// When: process() is called
// Then: Single chunk is processed, no overlap handling needed

// Test: SlidingWindowChunkingStrategy respects cancellation
// Given: 5-minute audio file being processed
// When: Task is cancelled after 2 chunks
// Then: Processing stops, partial result is emitted, resources cleaned up
```

#### VADChunkingStrategy Tests

```swift
// Test: VADChunkingStrategy processes chunks in parallel
// Given: Audio with 4 non-overlapping speech segments
// When: parallelProcessing=true, maxConcurrency=4
// Then: All 4 chunks start processing within 100ms of each other

// Test: VADChunkingStrategy falls back on VAD failure
// Given: SileroMLX VAD that throws on initialization
// When: LongAudioProcessor.create() with vad strategy
// Then: Falls back to EnergyVADProvider with warning

// Test: VADChunkingStrategy handles no speech detected
// Given: Audio where VAD returns empty segments
// When: process() is called
// Then: Empty result stream, no errors
```

#### LongAudioProcessor E2E Tests

```swift
// Test: LongAudioProcessor transcribes 5-minute audio
// Given: 5-minute audio file with known transcript
// When: LongAudioProcessor.transcribe() completes with auto strategy
// Then: WER < 15% compared to reference

// Test: LongAudioProcessor respects timeout limits
// Given: 10-minute audio with totalTimeout=30s
// When: transcribe() is called
// Then: Throws totalTimeoutExceeded after ~30s with partial result

// Test: LongAudioProcessor emits progress updates
// Given: 2-minute audio
// When: Iterating transcribe() stream
// Then: At least 4 progress updates with increasing processedDuration
```

### Edge Cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| Audio shorter than overlap (< 5s) | Process as single chunk, no overlap handling |
| Audio with no speech (all silence) | Return empty transcription, no error |
| Audio with continuous speech (no breaks) | VAD returns single segment, chunked at max duration |
| Single word utterances (< 1s) | Included in segments if above minSpeechDuration |
| Audio at exactly 30s boundary | Single chunk, no overlap needed |
| Very long audio (> 1 hour) | Process normally with progress updates |
| Audio with NaN values | Skip affected segment with warning |
| Chunk transcription returns empty | Skip and continue to next chunk |

---

## Usage Examples

```swift
// Simple usage - auto strategy
let processor = try await LongAudioProcessor.create(model: .largeTurbo)
let result = try await processor.transcribe(longAudio)
print(result.text)

// Streaming with progress
for try await progress in processor.transcribe(longAudio) {
    print("[\(Int(progress.progress * 100))%] \(progress.text)")
}

// Custom strategy - VAD with Silero MLX
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .vad(
        .init(parallelProcessing: true),
        vadProvider: .sileroMLX
    )
)

// Sequential for best accuracy
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .sequential(.init(conditionOnPreviousText: true))
)

// Sliding window for predictable latency
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .slidingWindow(.init(windowDuration: 30, overlapDuration: 5))
)

// With telemetry for debugging
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .auto,
    telemetry: ConsoleTelemetry()
)

// With resource limits
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .vad(),
    limits: .init(
        maxConcurrentChunks: 2,
        chunkTimeout: 30,
        totalTimeout: 300
    )
)
```

---

## Implementation Tasks

### Task 1: Core Types and Error Handling
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingTypes.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingError.swift`

### Task 2: VAD Core Types and Protocol
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/VADTypes.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/VADProvider.swift`

### Task 3: EnergyVADProvider
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/EnergyVADProvider.swift`
- Create: `mlx_audio_swift/stt/Tests/EnergyVADProviderTests.swift`

### Task 4: Test Doubles (MockChunkTranscriber, MockVADProvider)
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Testing/MockChunkTranscriber.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Testing/MockVADProvider.swift`

### Task 5: Telemetry Protocol
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingTelemetry.swift`

### Task 6: ChunkingStrategy Protocol
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingStrategy.swift`

### Task 7: SlidingWindowChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/SlidingWindowChunkingStrategy.swift`
- Create: `mlx_audio_swift/stt/Tests/SlidingWindowChunkingStrategyTests.swift`

### Task 8: SequentialChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/SequentialChunkingStrategy.swift`
- Create: `mlx_audio_swift/stt/Tests/SequentialChunkingStrategyTests.swift`

### Task 9: VADChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/VADChunkingStrategy.swift`
- Create: `mlx_audio_swift/stt/Tests/VADChunkingStrategyTests.swift`

### Task 10: LongAudioProcessor
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift`
- Create: `mlx_audio_swift/stt/Tests/LongAudioProcessorTests.swift`

### Task 11: SileroMLXVADProvider (Phase 2)
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/SileroVADModel.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/SileroMLXVADProvider.swift`

### Task 12: Integration and Demo
**Files:**
- Modify: `mlx_audio_swift/stt/stt-demo/main.swift` (add long audio support)

---

## References

### Papers
- [Whisper: Robust Speech Recognition](https://cdn.openai.com/papers/whisper.pdf) - Radford et al., 2022
- [WhisperX: Time-Accurate Speech Transcription](https://arxiv.org/abs/2303.00747) - Bain et al., 2023
- [ChunkFormer: Masked Chunking Conformer](https://arxiv.org/abs/2502.14673) - Le et al., 2025

### Implementations
- [OpenAI Whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Silero VAD](https://github.com/snakers4/silero-vad)

---

*Design created: 2025-01-07*
*Last updated: 2025-01-07 (spec-panel recommendations)*
*Branch: feat/streaming-stt*
