# Streaming STT Phase 2: Model Loading & Transcription Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the STT implementation by adding model weight loading, fixing the FFT issue, wiring up the full transcription pipeline, and integrating the tokenizer.

**Architecture:** Use MLXLMCommon's ModelConfiguration and Hub utilities for weight downloading. Fix vDSP FFT by using power-of-two size (512) with proper padding. Wire encoder→decoder→streaming decoder pipeline with KV cache.

**Tech Stack:** MLX Swift, MLXLMCommon (Hub), swift-tiktoken, Accelerate (vDSP)

**Prerequisite:** Phase 1 complete (feat/streaming-stt branch with core architecture)

---

## Phase 1: Fix FFT Issue

### Task 1.1: Fix vDSP FFT to Use Power-of-Two Size

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Audio/AudioConstants.swift`
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Audio/MelSpectrogram.swift`

**Problem:** vDSP_fft_zrip requires power-of-two FFT sizes. Current nFFT=400 is not a power of two, which may cause undefined behavior or incorrect results.

**Step 1: Update AudioConstants to use FFT size 512**

```swift
// In AudioConstants.swift, change nFFT:
/// FFT window size for mel spectrogram (must be power of 2 for vDSP)
/// Whisper uses 400 samples, but we pad to 512 for vDSP compatibility
public static let nFFT: Int = 512

/// Original Whisper window size (used for Hann window)
public static let whisperWindowSize: Int = 400
```

**Step 2: Update MelSpectrogram to use correct window**

```swift
// In MelSpectrogram.swift, update hannWindow call in compute():
// Create Hann window (use Whisper's 400-sample window, zero-pad to 512)
let window = paddedHannWindow(windowSize: AudioConstants.whisperWindowSize, fftSize: AudioConstants.nFFT)

// Add new helper function:
private static func paddedHannWindow(windowSize: Int, fftSize: Int) -> [Float] {
    var window = [Float](repeating: 0, count: windowSize)
    vDSP_hann_window(&window, vDSP_Length(windowSize), Int32(vDSP_HANN_NORM))

    // Zero-pad to fftSize
    if fftSize > windowSize {
        window.append(contentsOf: [Float](repeating: 0, count: fftSize - windowSize))
    }
    return window
}
```

**Step 3: Update frame windowing logic**

```swift
// In compute(), update the frame extraction:
for frame in 0..<nFrames {
    let start = frame * AudioConstants.hopLength
    let end = min(start + AudioConstants.whisperWindowSize, nSamples)

    // Extract and window frame with Whisper's 400-sample window
    var windowedFrame = [Float](repeating: 0, count: AudioConstants.nFFT)
    let frameLength = end - start
    for i in 0..<frameLength {
        windowedFrame[i] = samples[start + i] * window[i]
    }
    // Remaining samples stay zero (zero-padded)

    magnitudes[frame] = fftMagnitude(frame: windowedFrame, setup: fftSetup, log2n: log2n)
}
```

**Step 4: Run tests to verify**

Run: `swift test --filter MelSpectrogramTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Audio/
git commit -m "fix(stt): use power-of-two FFT size for vDSP compatibility"
```

---

## Phase 2: Model Weight Loading

### Task 2.1: Create WhisperModelLoader

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperModelLoader.swift`

**Step 1: Write the loader**

```swift
// mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperModelLoader.swift
import Foundation
import Hub
import MLX
import MLXNN

public enum WhisperModelLoader {

    /// Download and load Whisper model weights from HuggingFace
    /// - Parameters:
    ///   - model: Which Whisper model variant to load
    ///   - progressHandler: Optional callback for download progress
    /// - Returns: Tuple of (encoder, decoder, config)
    public static func load(
        model: WhisperModel,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws -> (encoder: AudioEncoder, decoder: TextDecoder, config: WhisperConfiguration) {
        let hubApi = HubApi()

        // Download model files
        let repo = Hub.Repo(id: model.rawValue)

        progressHandler?(0.1)

        let modelURL = try await hubApi.snapshot(from: repo, matching: ["*.safetensors", "config.json"])

        progressHandler?(0.5)

        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(WhisperConfiguration.self, from: configData)

        progressHandler?(0.6)

        // Create model components
        let encoder = AudioEncoder(
            nMels: config.nMels,
            nCtx: config.nAudioCtx,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer
        )

        let decoder = TextDecoder(
            nVocab: config.nVocab,
            nCtx: config.nTextCtx,
            nState: config.nTextState,
            nHead: config.nTextHead,
            nLayer: config.nTextLayer
        )

        progressHandler?(0.7)

        // Load weights
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        let weights = try loadWeights(from: weightsURL)

        progressHandler?(0.9)

        // Apply weights to encoder
        try applyEncoderWeights(encoder, weights: weights)

        // Apply weights to decoder
        try applyDecoderWeights(decoder, weights: weights)

        progressHandler?(1.0)

        return (encoder, decoder, config)
    }

    // MARK: - Private Helpers

    private static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        // MLX Swift safetensors loading
        return try MLX.loadArrays(url: url)
    }

    private static func applyEncoderWeights(
        _ encoder: AudioEncoder,
        weights: [String: MLXArray]
    ) throws {
        // Map weight names from HuggingFace format to our model structure
        // encoder.conv1.weight -> model.encoder.conv1.weight
        // This will be expanded based on actual weight keys

        let prefix = "model.encoder."

        if let w = weights[prefix + "conv1.weight"] {
            encoder.conv1.weight = w
        }
        if let b = weights[prefix + "conv1.bias"] {
            encoder.conv1.bias = b
        }
        if let w = weights[prefix + "conv2.weight"] {
            encoder.conv2.weight = w
        }
        if let b = weights[prefix + "conv2.bias"] {
            encoder.conv2.bias = b
        }

        // Load transformer blocks
        for (i, block) in encoder.blocks.enumerated() {
            let blockPrefix = prefix + "blocks.\(i)."
            try applyBlockWeights(block, weights: weights, prefix: blockPrefix)
        }

        // Layer norm
        if let w = weights[prefix + "ln_post.weight"] {
            encoder.lnPost.weight = w
        }
        if let b = weights[prefix + "ln_post.bias"] {
            encoder.lnPost.bias = b
        }
    }

    private static func applyDecoderWeights(
        _ decoder: TextDecoder,
        weights: [String: MLXArray]
    ) throws {
        let prefix = "model.decoder."

        // Token embedding
        if let w = weights[prefix + "token_embedding.weight"] {
            decoder.tokenEmbedding.weight = w
        }

        // Positional embedding
        if let pe = weights[prefix + "positional_embedding"] {
            decoder.positionalEmbedding = pe
        }

        // Load transformer blocks (with cross-attention)
        for (i, block) in decoder.blocks.enumerated() {
            let blockPrefix = prefix + "blocks.\(i)."
            try applyBlockWeights(block, weights: weights, prefix: blockPrefix)
        }

        // Layer norm
        if let w = weights[prefix + "ln.weight"] {
            decoder.ln.weight = w
        }
        if let b = weights[prefix + "ln.bias"] {
            decoder.ln.bias = b
        }
    }

    private static func applyBlockWeights(
        _ block: ResidualAttentionBlock,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        // Self-attention
        try applyAttentionWeights(block.attn, weights: weights, prefix: prefix + "attn.")

        if let w = weights[prefix + "attn_ln.weight"] {
            block.attnLn.weight = w
        }
        if let b = weights[prefix + "attn_ln.bias"] {
            block.attnLn.bias = b
        }

        // Cross-attention (if present)
        if let crossAttn = block.crossAttn, let crossAttnLn = block.crossAttnLn {
            try applyAttentionWeights(crossAttn, weights: weights, prefix: prefix + "cross_attn.")

            if let w = weights[prefix + "cross_attn_ln.weight"] {
                crossAttnLn.weight = w
            }
            if let b = weights[prefix + "cross_attn_ln.bias"] {
                crossAttnLn.bias = b
            }
        }

        // MLP
        if let w = weights[prefix + "mlp.0.weight"] {
            block.mlp1.weight = w
        }
        if let b = weights[prefix + "mlp.0.bias"] {
            block.mlp1.bias = b
        }
        if let w = weights[prefix + "mlp.2.weight"] {
            block.mlp2.weight = w
        }
        if let b = weights[prefix + "mlp.2.bias"] {
            block.mlp2.bias = b
        }

        if let w = weights[prefix + "mlp_ln.weight"] {
            block.mlpLn.weight = w
        }
        if let b = weights[prefix + "mlp_ln.bias"] {
            block.mlpLn.bias = b
        }
    }

    private static func applyAttentionWeights(
        _ attn: WhisperMultiHeadAttention,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        if let w = weights[prefix + "query.weight"] {
            attn.query.weight = w
        }
        if let b = weights[prefix + "query.bias"] {
            attn.query.bias = b
        }
        if let w = weights[prefix + "key.weight"] {
            attn.key.weight = w
        }
        if let w = weights[prefix + "value.weight"] {
            attn.value.weight = w
        }
        if let b = weights[prefix + "value.bias"] {
            attn.value.bias = b
        }
        if let w = weights[prefix + "out.weight"] {
            attn.out.weight = w
        }
        if let b = weights[prefix + "out.bias"] {
            attn.out.bias = b
        }
    }
}
```

**Step 2: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperModelLoader.swift
git commit -m "feat(stt): add WhisperModelLoader for HuggingFace weights"
```

---

### Task 2.2: Add WhisperTokenizer Wrapper

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperTokenizer.swift`

**Step 1: Write the tokenizer wrapper**

```swift
// mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperTokenizer.swift
import Foundation
import SwiftTiktoken

public final class WhisperTokenizer: @unchecked Sendable {
    private let encoding: Encoding

    // Special token IDs (from Whisper tokenizer)
    public static let eotToken: Int = 50257
    public static let sotToken: Int = 50258
    public static let translateToken: Int = 50358
    public static let transcribeToken: Int = 50359
    public static let noTimestampsToken: Int = 50363
    public static let timestampBegin: Int = 50364

    /// Language token range
    public static let languageTokenStart: Int = 50259
    public static let languageTokenEnd: Int = 50358

    public init() throws {
        // Whisper uses a GPT-2 style BPE encoding with multilingual extensions
        self.encoding = try Encoding.p50kBase
    }

    public func encode(_ text: String) -> [Int] {
        encoding.encode(text: text)
    }

    public func decode(_ tokens: [Int]) -> String {
        // Filter out special tokens before decoding
        let textTokens = tokens.filter { $0 < Self.eotToken }
        return encoding.decode(tokens: textTokens)
    }

    public func decodeWithTimestamps(_ tokens: [Int]) -> [(text: String, start: Double, end: Double)] {
        var segments: [(text: String, start: Double, end: Double)] = []
        var currentTokens: [Int] = []
        var currentStart: Double = 0
        var currentEnd: Double = 0

        for token in tokens {
            if token >= Self.timestampBegin {
                // Timestamp token: convert to seconds
                let time = Double(token - Self.timestampBegin) * 0.02  // 20ms per timestamp

                if currentTokens.isEmpty {
                    currentStart = time
                } else {
                    currentEnd = time
                    let text = decode(currentTokens)
                    if !text.isEmpty {
                        segments.append((text: text, start: currentStart, end: currentEnd))
                    }
                    currentTokens = []
                    currentStart = time
                }
            } else if token == Self.eotToken {
                // End of text - finalize current segment
                if !currentTokens.isEmpty {
                    let text = decode(currentTokens)
                    if !text.isEmpty {
                        segments.append((text: text, start: currentStart, end: currentEnd))
                    }
                }
                break
            } else if token < Self.eotToken {
                // Regular text token
                currentTokens.append(token)
            }
            // Skip other special tokens
        }

        return segments
    }

    /// Get language token for ISO language code
    public func languageToken(for language: String) -> Int? {
        // Map of ISO codes to token offsets
        let languageOffsets: [String: Int] = [
            "en": 0, "zh": 1, "de": 2, "es": 3, "ru": 4, "ko": 5, "fr": 6,
            "ja": 7, "pt": 8, "tr": 9, "pl": 10, "ca": 11, "nl": 12, "ar": 13,
            "sv": 14, "it": 15, "id": 16, "hi": 17, "fi": 18, "vi": 19, "he": 20,
            "uk": 21, "el": 22, "ms": 23, "cs": 24, "ro": 25, "da": 26, "hu": 27,
            "ta": 28, "no": 29, "th": 30, "ur": 31, "hr": 32, "bg": 33, "lt": 34,
            "la": 35, "mi": 36, "ml": 37, "cy": 38, "sk": 39, "te": 40, "fa": 41,
            "lv": 42, "bn": 43, "sr": 44, "az": 45, "sl": 46, "kn": 47, "et": 48,
            "mk": 49, "br": 50, "eu": 51, "is": 52, "hy": 53, "ne": 54, "mn": 55,
            "bs": 56, "kk": 57, "sq": 58, "sw": 59, "gl": 60, "mr": 61, "pa": 62,
            "si": 63, "km": 64, "sn": 65, "yo": 66, "so": 67, "af": 68, "oc": 69,
            "ka": 70, "be": 71, "tg": 72, "sd": 73, "gu": 74, "am": 75, "yi": 76,
            "lo": 77, "uz": 78, "fo": 79, "ht": 80, "ps": 81, "tk": 82, "nn": 83,
            "mt": 84, "sa": 85, "lb": 86, "my": 87, "bo": 88, "tl": 89, "mg": 90,
            "as": 91, "tt": 92, "haw": 93, "ln": 94, "ha": 95, "ba": 96, "jw": 97,
            "su": 98
        ]

        guard let offset = languageOffsets[language.lowercased()] else {
            return nil
        }

        return Self.languageTokenStart + offset
    }

    /// Create initial decoder tokens for transcription
    public func initialTokens(language: String?, task: TranscriptionOptions.TranscriptionTask) -> [Int] {
        var tokens = [Self.sotToken]

        // Add language token
        if let lang = language, let langToken = languageToken(for: lang) {
            tokens.append(langToken)
        }

        // Add task token
        switch task {
        case .transcribe:
            tokens.append(Self.transcribeToken)
        case .translate:
            tokens.append(Self.translateToken)
        }

        // No timestamps for streaming
        tokens.append(Self.noTimestampsToken)

        return tokens
    }
}
```

**Step 2: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperTokenizer.swift
git commit -m "feat(stt): add WhisperTokenizer wrapper for swift-tiktoken"
```

---

## Phase 3: Wire Up Transcription Pipeline

### Task 3.1: Update WhisperSession with Model Components

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperSession.swift`

**Step 1: Add model component storage**

```swift
// At top of WhisperSession class, replace placeholders with actual components:
private let modelType: WhisperModel
private let streamingConfig: StreamingConfig
private var currentTask: Task<Void, Never>?
private let taskLock = NSLock()

// Model components
private let encoder: AudioEncoder
private let decoder: TextDecoder
private let tokenizer: WhisperTokenizer
private let config: WhisperConfiguration
private let alignmentHeads: [(layer: Int, head: Int)]

private init(
    modelType: WhisperModel,
    streamingConfig: StreamingConfig,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    tokenizer: WhisperTokenizer,
    config: WhisperConfiguration
) {
    self.modelType = modelType
    self.streamingConfig = streamingConfig
    self.encoder = encoder
    self.decoder = decoder
    self.tokenizer = tokenizer
    self.config = config
    self.alignmentHeads = WhisperAlignmentHeads.heads(for: modelType)
}
```

**Step 2: Update factory method**

```swift
public static func fromPretrained(
    model: WhisperModel = .largeTurbo,
    streaming: StreamingConfig = .default,
    progressHandler: ((WhisperProgress) -> Void)? = nil
) async throws -> WhisperSession {
    progressHandler?(.downloading(0))

    let (encoder, decoder, config) = try await WhisperModelLoader.load(
        model: model,
        progressHandler: { progress in
            progressHandler?(.downloading(progress * 0.8))
        }
    )

    progressHandler?(.loading(0.9))

    let tokenizer = try WhisperTokenizer()

    progressHandler?(.loading(1.0))

    return WhisperSession(
        modelType: model,
        streamingConfig: streaming,
        encoder: encoder,
        decoder: decoder,
        tokenizer: tokenizer,
        config: config
    )
}
```

**Step 3: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperSession.swift
git commit -m "feat(stt): add model components to WhisperSession"
```

---

### Task 3.2: Implement Transcription Loop

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperSession.swift`

**Step 1: Implement the streaming transcription**

```swift
// Replace the placeholder transcribe method with actual implementation:
public func transcribe(
    _ audio: MLXArray,
    sampleRate: Int = AudioConstants.sampleRate,
    options: TranscriptionOptions = .default
) -> AsyncThrowingStream<StreamingResult, Error> {
    AsyncThrowingStream { continuation in
        self.taskLock.withLock {
            self.currentTask = Task {
                do {
                    // Validate sample rate
                    guard sampleRate == AudioConstants.sampleRate else {
                        throw WhisperError.sampleRateMismatch(
                            expected: AudioConstants.sampleRate,
                            got: sampleRate
                        )
                    }

                    try Task.checkCancellation()

                    // 1. Preprocess audio: pad/trim to 30s
                    let paddedAudio = AudioUtils.padOrTrim(audio, length: AudioConstants.nSamples)

                    // 2. Compute mel spectrogram
                    let mel = try MelSpectrogram.compute(audio: paddedAudio)
                    // Add batch dimension: [nMels, nFrames] -> [1, nFrames, nMels]
                    let melBatched = mel.T.expandedDimensions(axis: 0)

                    try Task.checkCancellation()

                    // 3. Encode audio
                    let encoderOutput = self.encoder(melBatched)
                    let totalFrames = encoderOutput.shape[1]

                    // 4. Initialize decoder state
                    var tokens = self.tokenizer.initialTokens(
                        language: options.language,
                        task: options.task
                    )
                    var kvCache: TextDecoder.KVCache? = nil
                    var emittedText = ""
                    var lastEmittedIndex = tokens.count

                    // 5. Decoding loop
                    let maxTokens = self.config.nTextCtx - tokens.count

                    for step in 0..<maxTokens {
                        try Task.checkCancellation()

                        // Decode one step
                        let tokenArray = MLXArray(tokens.suffix(1)).expandedDimensions(axis: 0)
                        let (logits, newCache, crossQK) = self.decoder(
                            step == 0 ? MLXArray(tokens).expandedDimensions(axis: 0) : tokenArray,
                            xa: encoderOutput,
                            kvCache: kvCache
                        )
                        kvCache = newCache

                        // Sample next token (greedy for now)
                        let nextToken = Int(MLX.argMax(logits[0, -1]).item(Int.self))

                        // Check for end of transcription
                        if nextToken == WhisperTokenizer.eotToken {
                            // Emit final result
                            let finalText = self.tokenizer.decode(Array(tokens.dropFirst(lastEmittedIndex)))
                            if !finalText.isEmpty {
                                continuation.yield(StreamingResult(
                                    text: emittedText + finalText,
                                    isFinal: true,
                                    timestamp: 0...Double(audio.shape[0]) / Double(sampleRate)
                                ))
                            }
                            break
                        }

                        tokens.append(nextToken)

                        // Check if we should emit (AlignAtt logic)
                        let mostAttendedFrame = StreamingDecoder.getMostAttendedFrame(
                            crossQK: crossQK,
                            alignmentHeads: self.alignmentHeads
                        )

                        let shouldEmit = StreamingDecoder.shouldEmit(
                            mostAttendedFrame: mostAttendedFrame,
                            totalContentFrames: totalFrames,
                            threshold: self.streamingConfig.frameThreshold
                        )

                        if shouldEmit && self.streamingConfig.emitPartial {
                            let newTokens = Array(tokens[lastEmittedIndex...])
                            let newText = self.tokenizer.decode(newTokens)

                            if !newText.isEmpty {
                                emittedText += newText
                                lastEmittedIndex = tokens.count

                                // Calculate approximate timestamp
                                let frameTime = Double(mostAttendedFrame) / Double(AudioConstants.framesPerSecond)

                                continuation.yield(StreamingResult(
                                    text: emittedText,
                                    isFinal: false,
                                    timestamp: 0...frameTime
                                ))
                            }
                        }
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
}
```

**Step 2: Add framesPerSecond constant to AudioConstants**

```swift
// In AudioConstants.swift, add:
/// Frames per second (sampleRate / hopLength)
public static let framesPerSecond: Int = sampleRate / hopLength  // 100
```

**Step 3: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/
git commit -m "feat(stt): implement streaming transcription loop with AlignAtt"
```

---

## Phase 4: Integration Tests

### Task 4.1: Add End-to-End Test

**Files:**
- Modify: `mlx_audio_swift/stt/Tests/WhisperSessionTests.swift`

**Step 1: Add model loading test (requires network)**

```swift
@Test(.disabled("Requires network and model download"))
func transcribe_silentAudio_returnsEmptyOrMinimal() async throws {
    // Given: A session with tiny model
    let session = try await WhisperSession.fromPretrained(model: .tiny)

    // Silent audio (1 second)
    let audio = MLXArray.zeros([AudioConstants.sampleRate])

    // When: Transcribing
    var results: [StreamingResult] = []
    for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
        results.append(result)
    }

    // Then: Should complete (may have minimal output for silence)
    #expect(!results.isEmpty)
    #expect(results.last?.isFinal == true)
}
```

**Step 2: Run tests**

Run: `swift test --filter WhisperSessionTests`
Expected: Existing tests pass, new test skipped

**Step 3: Commit**

```bash
git add mlx_audio_swift/stt/Tests/
git commit -m "test(stt): add end-to-end transcription test"
```

---

### Task 4.2: Update MultiHeadAttention to WhisperMultiHeadAttention

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/MultiHeadAttention.swift`

**Step 1: Rename class to avoid collision with MLXNN.MultiHeadAttention**

If there's a naming collision with MLXNN's MultiHeadAttention, rename our class:

```swift
// Rename class from MultiHeadAttention to WhisperMultiHeadAttention
public class WhisperMultiHeadAttention: Module {
    // ... same implementation
}
```

**Step 2: Update references in ResidualAttentionBlock.swift**

```swift
// Update type references
let attn: WhisperMultiHeadAttention
let crossAttn: WhisperMultiHeadAttention?
```

**Step 3: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/
git commit -m "refactor(stt): rename MultiHeadAttention to avoid MLXNN collision"
```

---

## Summary

This plan completes the STT implementation in 4 phases:

1. **Phase 1** (Task 1.1): Fix vDSP FFT to use power-of-two size
2. **Phase 2** (Tasks 2.1-2.2): Model weight loading and tokenizer
3. **Phase 3** (Tasks 3.1-3.2): Wire up full transcription pipeline
4. **Phase 4** (Tasks 4.1-4.2): Integration tests and cleanup

**After this plan:**
- Full end-to-end STT works
- AlignAtt streaming emits tokens progressively
- Model weights load from HuggingFace

**Future work (not in this plan):**
- Performance optimization (batching, quantization)
- Real-time audio input integration
- Voice Activity Detection (VAD) pre-filtering
- Benchmark tests for latency/RTF metrics
