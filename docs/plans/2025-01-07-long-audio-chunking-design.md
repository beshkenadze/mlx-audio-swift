# Long Audio Chunking Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable transcription of audio longer than 30 seconds using configurable chunking strategies

**Architecture:** Three pluggable chunking strategies (Sequential, VAD-based, Sliding Window) with unified LongAudioProcessor API. Two VAD providers (Energy-based, Silero MLX) for voice activity detection.

**Tech Stack:** MLX Swift, async/await, AsyncThrowingStream

---

## Table of Contents

1. [Overview](#overview)
2. [VAD Provider Protocol](#vad-provider-protocol)
3. [VAD Implementations](#vad-implementations)
4. [ChunkingStrategy Protocol](#chunkingstrategy-protocol)
5. [Strategy Implementations](#strategy-implementations)
6. [LongAudioProcessor API](#longaudioprocessor-api)
7. [Usage Examples](#usage-examples)
8. [Implementation Tasks](#implementation-tasks)

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

public struct WordTimestamp: Sendable {
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
        transcriber: ChunkTranscriber
    ) -> AsyncThrowingStream<ChunkResult, Error>

    /// Strategy identifier for logging/debugging
    var name: String { get }
}

/// Abstraction for transcribing a single ≤30s chunk
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
        transcriber: ChunkTranscriber
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let totalSamples = audio.shape[0]
                    let maxSamples = 30 * sampleRate  // 30 seconds

                    var seekSample = 0
                    var previousTokens: [Int]? = nil

                    while seekSample < totalSamples {
                        try Task.checkCancellation()

                        // Extract up to 30s chunk starting at seek position
                        let endSample = min(seekSample + maxSamples, totalSamples)
                        let chunk = audio[seekSample..<endSample]

                        // Transcribe with optional previous context
                        let context = config.conditionOnPreviousText ? previousTokens : nil
                        let result = try await transcriber.transcribe(
                            audio: chunk,
                            sampleRate: sampleRate,
                            previousTokens: context
                        )

                        // Validate result quality
                        if shouldSkipResult(result) {
                            seekSample += sampleRate  // Skip 1 second
                            continue
                        }

                        // Adjust time range to absolute position
                        let absoluteStart = Double(seekSample) / Double(sampleRate)
                        let adjustedResult = ChunkResult(
                            text: result.text,
                            tokens: result.tokens,
                            timeRange: (absoluteStart + result.timeRange.lowerBound)...(absoluteStart + result.timeRange.upperBound),
                            confidence: result.confidence,
                            words: result.words?.map { word in
                                WordTimestamp(
                                    word: word.word,
                                    start: absoluteStart + word.start,
                                    end: absoluteStart + word.end,
                                    confidence: word.confidence
                                )
                            }
                        )

                        continuation.yield(adjustedResult)

                        // Advance seek based on last timestamp
                        let lastTimestamp = result.timeRange.upperBound
                        seekSample += Int(lastTimestamp * Double(sampleRate))

                        // Update context for next chunk
                        if config.conditionOnPreviousText {
                            let tokenCount = min(result.tokens.count, config.maxPreviousTokens)
                            previousTokens = Array(result.tokens.suffix(tokenCount))
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func shouldSkipResult(_ result: ChunkResult) -> Bool {
        result.confidence < 0.1 || result.text.trimmingCharacters(in: .whitespaces).isEmpty
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
        /// Max concurrent transcriptions
        public var maxConcurrency: Int = 4

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
        transcriber: ChunkTranscriber
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // 1. Run VAD to get speech segments
                    let segments = try await vadProvider.detectSpeech(in: audio, sampleRate: sampleRate)

                    // 2. Merge/split segments to target duration
                    let chunks = prepareChunks(
                        segments: segments,
                        audioDuration: Double(audio.shape[0]) / Double(sampleRate)
                    )

                    // 3. Process chunks (parallel or sequential)
                    if config.parallelProcessing {
                        try await processParallel(
                            chunks: chunks,
                            audio: audio,
                            sampleRate: sampleRate,
                            transcriber: transcriber,
                            continuation: continuation
                        )
                    } else {
                        for timeRange in chunks {
                            try Task.checkCancellation()
                            let result = try await transcribeChunk(
                                timeRange: timeRange,
                                audio: audio,
                                sampleRate: sampleRate,
                                transcriber: transcriber
                            )
                            continuation.yield(result)
                        }
                    }

                    continuation.finish()
                } catch {
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
        transcriber: ChunkTranscriber
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let totalDuration = Double(audio.shape[0]) / Double(sampleRate)
                    var position: TimeInterval = 0
                    var previousResult: ChunkResult?
                    var accumulatedText = ""
                    var accumulatedWords: [WordTimestamp] = []

                    while position < totalDuration {
                        try Task.checkCancellation()

                        let windowEnd = min(position + config.windowDuration, totalDuration)
                        let startSample = Int(position * Double(sampleRate))
                        let endSample = Int(windowEnd * Double(sampleRate))
                        let chunk = audio[startSample..<endSample]

                        let result = try await transcriber.transcribe(
                            audio: chunk,
                            sampleRate: sampleRate,
                            previousTokens: nil
                        )

                        // Adjust timestamps and merge with previous
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

                        continuation.yield(ChunkResult(
                            text: accumulatedText,
                            tokens: [],
                            timeRange: 0...windowEnd,
                            confidence: adjustedResult.confidence,
                            words: accumulatedWords
                        ))

                        previousResult = adjustedResult
                        position += config.hopDuration
                    }

                    continuation.finish()
                } catch {
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
        case vad(VADChunkingStrategy.VADConfig = .default, vadProvider: VADProviderType = .energy())
        case slidingWindow(SlidingWindowChunkingStrategy.SlidingWindowConfig = .default)
    }

    public enum VADProviderType: Sendable {
        case energy(threshold: Float = 0.02)
        case sileroMLX(threshold: Float = 0.5)
    }

    // MARK: - Factory

    public static func create(
        model: WhisperModel = .largeTurbo,
        strategy: StrategyType = .auto,
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
        .init(parallelProcessing: true, maxConcurrency: 4),
        vadProvider: .sileroMLX()
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
```

---

## Implementation Tasks

### Task 1: VAD Core Types and Protocol
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/VADTypes.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/VADProvider.swift`

### Task 2: EnergyVADProvider
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/EnergyVADProvider.swift`
- Create: `mlx_audio_swift/stt/Tests/EnergyVADProviderTests.swift`

### Task 3: SileroMLXVADProvider (Optional - Phase 2)
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/SileroVADModel.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/VAD/SileroMLXVADProvider.swift`

### Task 4: ChunkingStrategy Protocol
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingTypes.swift`
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/ChunkingStrategy.swift`

### Task 5: SlidingWindowChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/SlidingWindowChunkingStrategy.swift`
- Create: `mlx_audio_swift/stt/Tests/SlidingWindowChunkingStrategyTests.swift`

### Task 6: SequentialChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/SequentialChunkingStrategy.swift`

### Task 7: VADChunkingStrategy
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/VADChunkingStrategy.swift`

### Task 8: LongAudioProcessor
**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift`
- Create: `mlx_audio_swift/stt/Tests/LongAudioProcessorTests.swift`

### Task 9: Integration and Demo
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
*Branch: feat/streaming-stt*
