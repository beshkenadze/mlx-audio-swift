# Streaming STT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement native Swift STT with AlignAtt streaming for Apple Silicon, achieving ~1s latency to first token.

**Architecture:** Port Python mlx-audio STT to Swift using MLX Swift framework. The implementation follows the existing TTS pattern (WhisperSession mirrors MarvisSession). Core components: AudioProcessor (mel spectrogram), Whisper model (encoder/decoder), StreamingDecoder (AlignAtt logic).

**Tech Stack:** MLX Swift, MLXNN, MLXLMCommon, swift-tiktoken, Accelerate (vDSP)

**Design Document:** [2025-01-06-streaming-stt-design.md](./2025-01-06-streaming-stt-design.md)

---

## Phase 1: Project Setup & Audio Processing

### Task 1.1: Add swift-tiktoken Dependency

**Files:**
- Modify: `Package.swift`

**Step 1: Add the dependency**

```swift
// In dependencies array, add:
.package(url: "https://github.com/aespinilla/swift-tiktoken", from: "1.0.0"),

// In MLXAudio target dependencies, add:
.product(name: "SwiftTiktoken", package: "swift-tiktoken"),
```

**Step 2: Verify package resolves**

Run: `swift package resolve`
Expected: Package downloads successfully

**Step 3: Commit**

```bash
git add Package.swift Package.resolved
git commit -m "feat(stt): add swift-tiktoken dependency"
```

---

### Task 1.2: Create STT Directory Structure

**Files:**
- Create: `MLXAudio/STT/STTSession.swift`
- Create: `MLXAudio/STT/StreamingResult.swift`

**Step 1: Create STTSession protocol**

```swift
// MLXAudio/STT/STTSession.swift
import Foundation
import MLX

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

**Step 2: Create StreamingResult type**

```swift
// MLXAudio/STT/StreamingResult.swift
import Foundation

public struct StreamingResult: Sendable {
    public let text: String
    public let isFinal: Bool
    public let timestamp: ClosedRange<TimeInterval>

    public init(text: String, isFinal: Bool, timestamp: ClosedRange<TimeInterval>) {
        self.text = text
        self.isFinal = isFinal
        self.timestamp = timestamp
    }
}
```

**Step 3: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add MLXAudio/STT/
git commit -m "feat(stt): add STTSession protocol and StreamingResult"
```

---

### Task 1.3: Create Audio Constants

**Files:**
- Create: `MLXAudio/STT/Whisper/Audio/AudioConstants.swift`

**Step 1: Write the constants file**

```swift
// MLXAudio/STT/Whisper/Audio/AudioConstants.swift
import Foundation

public enum AudioConstants {
    public static let sampleRate: Int = 16000
    public static let nFFT: Int = 400
    public static let hopLength: Int = 160
    public static let nMels: Int = 80
    public static let chunkLength: Int = 30  // seconds
    public static let nSamples: Int = chunkLength * sampleRate  // 480000
    public static let nFrames: Int = nSamples / hopLength  // 3000
    public static let framesPerSecond: Int = sampleRate / hopLength  // 100
}
```

**Step 2: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add MLXAudio/STT/Whisper/Audio/AudioConstants.swift
git commit -m "feat(stt): add audio constants matching Python reference"
```

---

### Task 1.4: Implement Mel Spectrogram - Test First

**Files:**
- Create: `Tests/STTTests/AudioProcessorTests.swift`
- Create: `MLXAudio/STT/Whisper/Audio/AudioProcessor.swift`

**Step 1: Write failing test**

```swift
// Tests/STTTests/AudioProcessorTests.swift
import Testing
import MLX
@testable import MLXAudio

struct AudioProcessorTests {
    @Test func melSpectrogram_outputShape() async throws {
        // Given: 1 second of audio at 16kHz
        let audio = MLXArray.zeros([16000])

        // When: Computing mel spectrogram
        let mel = AudioProcessor.logMelSpectrogram(audio)

        // Then: Shape is (n_frames, n_mels) = (100, 80)
        #expect(mel.shape == [100, 80])
    }

    @Test func melSpectrogram_30sAudio_fullFrames() async throws {
        // Given: 30 seconds of audio (max chunk)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        // When: Computing mel spectrogram
        let mel = AudioProcessor.logMelSpectrogram(audio)

        // Then: Shape is (3000, 80)
        #expect(mel.shape == [AudioConstants.nFrames, AudioConstants.nMels])
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter AudioProcessorTests`
Expected: FAIL with "AudioProcessor not found"

**Step 3: Implement AudioProcessor**

```swift
// MLXAudio/STT/Whisper/Audio/AudioProcessor.swift
import Accelerate
import Foundation
import MLX

public enum AudioProcessor {
    /// Compute log-mel spectrogram from audio waveform
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray, shape [N]
    ///   - nMels: Number of mel filterbank channels (default 80)
    ///   - padding: Zero-padding to add
    /// - Returns: Log-mel spectrogram, shape [frames, nMels]
    public static func logMelSpectrogram(
        _ audio: MLXArray,
        nMels: Int = AudioConstants.nMels,
        padding: Int = 0
    ) -> MLXArray {
        // Pad audio if requested
        var paddedAudio = audio
        if padding > 0 {
            paddedAudio = MLX.concatenate([
                MLXArray.zeros([padding]),
                audio,
                MLXArray.zeros([padding])
            ])
        }

        // Convert to Float array for vDSP
        let audioData = paddedAudio.asArray(Float.self)

        // Compute STFT using vDSP
        let stftResult = computeSTFT(
            audioData,
            nFFT: AudioConstants.nFFT,
            hopLength: AudioConstants.hopLength
        )

        // Get magnitude squared
        let magnitudes = stftResult.map { $0 * $0 }

        // Apply mel filterbank
        let melFilters = melFilterbank(
            sampleRate: AudioConstants.sampleRate,
            nFFT: AudioConstants.nFFT,
            nMels: nMels
        )

        let melSpec = applyMelFilters(magnitudes, filters: melFilters, nMels: nMels)

        // Log scale with clamping
        let logSpec = melSpec.map { max(log10(max($0, 1e-10)), log10(1e-10)) }

        // Normalize: (log_spec + 4.0) / 4.0
        let normalized = logSpec.map { ($0 + 4.0) / 4.0 }

        // Reshape to [frames, nMels]
        let nFrames = normalized.count / nMels
        return MLXArray(normalized).reshaped([nFrames, nMels])
    }

    // MARK: - Private Helpers

    private static func computeSTFT(
        _ audio: [Float],
        nFFT: Int,
        hopLength: Int
    ) -> [Float] {
        let window = vDSP.window(
            ofType: Float.self,
            usingSequence: .hanningNormalized,
            count: nFFT,
            isHalfWindow: false
        )

        let nFrames = (audio.count - nFFT) / hopLength + 1
        var magnitudes = [Float](repeating: 0, count: nFrames * (nFFT / 2 + 1))

        // Setup FFT
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return magnitudes
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)

        for frame in 0..<nFrames {
            let start = frame * hopLength
            var windowedFrame = [Float](repeating: 0, count: nFFT)

            // Apply window
            for i in 0..<nFFT where start + i < audio.count {
                windowedFrame[i] = audio[start + i] * window[i]
            }

            // FFT
            windowedFrame.withUnsafeMutableBufferPointer { framePtr in
                realPart.withUnsafeMutableBufferPointer { realPtr in
                    imagPart.withUnsafeMutableBufferPointer { imagPtr in
                        var splitComplex = DSPSplitComplex(
                            realp: realPtr.baseAddress!,
                            imagp: imagPtr.baseAddress!
                        )
                        framePtr.baseAddress!.withMemoryRebound(
                            to: DSPComplex.self,
                            capacity: nFFT / 2
                        ) { complexPtr in
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))
                        }
                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
                    }
                }
            }

            // Compute magnitude
            for i in 0..<(nFFT / 2 + 1) {
                let r = i < nFFT / 2 ? realPart[i] : 0
                let im = i < nFFT / 2 ? imagPart[i] : 0
                magnitudes[frame * (nFFT / 2 + 1) + i] = sqrt(r * r + im * im)
            }
        }

        return magnitudes
    }

    private static func melFilterbank(
        sampleRate: Int,
        nFFT: Int,
        nMels: Int
    ) -> [[Float]] {
        let fMin: Float = 0
        let fMax = Float(sampleRate) / 2

        // Mel scale conversion
        func hzToMel(_ hz: Float) -> Float {
            return 2595 * log10(1 + hz / 700)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700 * (pow(10, mel / 2595) - 1)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0...nMels + 1).map { i in
            melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
        }

        let fftBins = melPoints.map { hz -> Int in
            Int((hz * Float(nFFT) / Float(sampleRate)).rounded())
        }

        var filters = [[Float]](repeating: [Float](repeating: 0, count: nFFT / 2 + 1), count: nMels)

        for i in 0..<nMels {
            let left = fftBins[i]
            let center = fftBins[i + 1]
            let right = fftBins[i + 2]

            for j in left..<center where j < nFFT / 2 + 1 {
                filters[i][j] = Float(j - left) / Float(center - left)
            }
            for j in center..<right where j < nFFT / 2 + 1 {
                filters[i][j] = Float(right - j) / Float(right - center)
            }
        }

        return filters
    }

    private static func applyMelFilters(
        _ magnitudes: [Float],
        filters: [[Float]],
        nMels: Int
    ) -> [Float] {
        let nFFTBins = filters[0].count
        let nFrames = magnitudes.count / nFFTBins
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)

        for frame in 0..<nFrames {
            for mel in 0..<nMels {
                var sum: Float = 0
                for bin in 0..<nFFTBins {
                    sum += magnitudes[frame * nFFTBins + bin] * filters[mel][bin]
                }
                melSpec[frame * nMels + mel] = sum
            }
        }

        return melSpec
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter AudioProcessorTests`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Audio/AudioProcessor.swift Tests/STTTests/
git commit -m "feat(stt): implement mel spectrogram with vDSP"
```

---

### Task 1.5: Implement Pad or Trim Utility

**Files:**
- Modify: `MLXAudio/STT/Whisper/Audio/AudioProcessor.swift`
- Modify: `Tests/STTTests/AudioProcessorTests.swift`

**Step 1: Write failing test**

```swift
// Add to AudioProcessorTests.swift
@Test func padOrTrim_shortAudio_pads() async throws {
    // Given: 1 second audio
    let audio = MLXArray.ones([16000])

    // When: Padding to 30 seconds
    let result = AudioProcessor.padOrTrim(audio, length: AudioConstants.nSamples)

    // Then: Padded with zeros
    #expect(result.shape == [AudioConstants.nSamples])
}

@Test func padOrTrim_longAudio_trims() async throws {
    // Given: 60 seconds audio
    let audio = MLXArray.ones([32000 * 60])

    // When: Trimming to 30 seconds
    let result = AudioProcessor.padOrTrim(audio, length: AudioConstants.nSamples)

    // Then: Trimmed
    #expect(result.shape == [AudioConstants.nSamples])
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter padOrTrim`
Expected: FAIL with "padOrTrim not found"

**Step 3: Implement padOrTrim**

```swift
// Add to AudioProcessor.swift
public static func padOrTrim(_ audio: MLXArray, length: Int) -> MLXArray {
    if audio.shape[0] > length {
        return audio[0..<length]
    } else if audio.shape[0] < length {
        let padding = MLXArray.zeros([length - audio.shape[0]])
        return MLX.concatenate([audio, padding])
    }
    return audio
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter padOrTrim`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Audio/AudioProcessor.swift Tests/STTTests/
git commit -m "feat(stt): add padOrTrim utility for audio preprocessing"
```

---

## Phase 2: Whisper Model Architecture

### Task 2.1: Create Model Configuration

**Files:**
- Create: `MLXAudio/STT/Whisper/Model/ModelDimensions.swift`
- Create: `MLXAudio/STT/Whisper/WhisperModel.swift`

**Step 1: Create ModelDimensions**

```swift
// MLXAudio/STT/Whisper/Model/ModelDimensions.swift
import Foundation

public struct ModelDimensions: Codable, Sendable {
    public let nMels: Int
    public let nAudioCtx: Int
    public let nAudioState: Int
    public let nAudioHead: Int
    public let nAudioLayer: Int
    public let nVocab: Int
    public let nTextCtx: Int
    public let nTextState: Int
    public let nTextHead: Int
    public let nTextLayer: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
    }
}
```

**Step 2: Create WhisperModel enum**

```swift
// MLXAudio/STT/Whisper/WhisperModel.swift
import Foundation

public enum WhisperModel: String, CaseIterable, Sendable {
    case tiny = "mlx-community/whisper-tiny-mlx"
    case base = "mlx-community/whisper-base-mlx"
    case small = "mlx-community/whisper-small-mlx"
    case medium = "mlx-community/whisper-medium-mlx"
    case largeV3 = "mlx-community/whisper-large-v3-mlx"
    case largeTurbo = "mlx-community/whisper-large-v3-turbo"

    public var estimatedMemoryMB: Int {
        switch self {
        case .tiny: return 150
        case .base: return 290
        case .small: return 970
        case .medium: return 3100
        case .largeV3: return 6200
        case .largeTurbo: return 1600
        }
    }
}
```

**Step 3: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add MLXAudio/STT/Whisper/Model/ MLXAudio/STT/Whisper/WhisperModel.swift
git commit -m "feat(stt): add ModelDimensions and WhisperModel enum"
```

---

### Task 2.2: Implement MultiHeadAttention

**Files:**
- Create: `MLXAudio/STT/Whisper/Model/MultiHeadAttention.swift`
- Create: `Tests/STTTests/MultiHeadAttentionTests.swift`

**Step 1: Write failing test**

```swift
// Tests/STTTests/MultiHeadAttentionTests.swift
import Testing
import MLX
import MLXNN
@testable import MLXAudio

struct MultiHeadAttentionTests {
    @Test func selfAttention_outputShape() async throws {
        // Given: MultiHeadAttention with 512 state, 8 heads
        let mha = MultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 10, 512])  // batch=1, seq=10, state=512

        // When: Self-attention (no xa)
        let (output, _, _) = mha(x)

        // Then: Same shape as input
        #expect(output.shape == [1, 10, 512])
    }

    @Test func crossAttention_returnsQK() async throws {
        // Given: MultiHeadAttention
        let mha = MultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 5, 512])   // decoder input
        let xa = MLXArray.zeros([1, 20, 512]) // encoder output

        // When: Cross-attention
        let (_, _, qk) = mha(x, xa: xa)

        // Then: QK weights returned for AlignAtt
        #expect(qk != nil)
        #expect(qk!.shape == [1, 8, 5, 20])  // batch, heads, q_seq, k_seq
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter MultiHeadAttentionTests`
Expected: FAIL with "MultiHeadAttention not found"

**Step 3: Implement MultiHeadAttention**

```swift
// MLXAudio/STT/Whisper/Model/MultiHeadAttention.swift
import Foundation
import MLX
import MLXNN

public class MultiHeadAttention: Module {
    let nHead: Int
    let query: Linear
    let key: Linear
    let value: Linear
    let out: Linear

    public init(nState: Int, nHead: Int) {
        self.nHead = nHead
        self.query = Linear(nState, nState)
        self.key = Linear(nState, nState, bias: false)
        self.value = Linear(nState, nState)
        self.out = Linear(nState, nState)
    }

    public func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), MLXArray?) {
        let q = query(x)

        var k: MLXArray
        var v: MLXArray

        if let xa = xa {
            // Cross-attention
            if let cache = kvCache {
                k = cache.0
                v = cache.1
            } else {
                k = key(xa)
                v = value(xa)
            }
        } else {
            // Self-attention
            k = key(x)
            v = value(x)
            if let cache = kvCache {
                k = MLX.concatenate([cache.0, k], axis: 1)
                v = MLX.concatenate([cache.1, v], axis: 1)
            }
        }

        let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask)
        return (out(wv), (k, v), qk)
    }

    private func qkvAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (nBatch, nCtx, nState) = (q.shape[0], q.shape[1], q.shape[2])
        let scale = Float(pow(Double(nState / nHead), -0.25))

        // Reshape and transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let qReshaped = (q.reshaped([nBatch, nCtx, nHead, -1]).transposed(0, 2, 1, 3)) * scale
        let kReshaped = (k.reshaped([nBatch, k.shape[1], nHead, -1]).transposed(0, 2, 3, 1)) * scale
        let vReshaped = v.reshaped([nBatch, v.shape[1], nHead, -1]).transposed(0, 2, 1, 3)

        // Attention scores
        var qk = MLX.matmul(qReshaped, kReshaped)
        if let mask = mask {
            qk = qk + mask[0..<nCtx, 0..<nCtx]
        }

        // Softmax and weighted sum
        let w = MLX.softmax(qk, axis: -1)
        var output = MLX.matmul(w, vReshaped)

        // Transpose back and reshape
        output = output.transposed(0, 2, 1, 3).reshaped([nBatch, nCtx, nState])

        return (output, qk)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter MultiHeadAttentionTests`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Model/MultiHeadAttention.swift Tests/STTTests/
git commit -m "feat(stt): implement MultiHeadAttention with cross-attention QK capture"
```

---

### Task 2.3: Implement ResidualAttentionBlock

**Files:**
- Create: `MLXAudio/STT/Whisper/Model/ResidualAttentionBlock.swift`

**Step 1: Write failing test**

```swift
// Add to Tests/STTTests/ModelTests.swift
@Test func residualBlock_withCrossAttention_returnsCrossQK() async throws {
    // Given: Block with cross-attention
    let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: true)
    let x = MLXArray.zeros([1, 5, 512])
    let xa = MLXArray.zeros([1, 20, 512])

    // When: Forward pass
    let (output, _, crossQK) = block(x, xa: xa)

    // Then: Cross-attention QK returned
    #expect(output.shape == [1, 5, 512])
    #expect(crossQK != nil)
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter residualBlock`
Expected: FAIL with "ResidualAttentionBlock not found"

**Step 3: Implement ResidualAttentionBlock**

```swift
// MLXAudio/STT/Whisper/Model/ResidualAttentionBlock.swift
import Foundation
import MLX
import MLXNN

public class ResidualAttentionBlock: Module {
    let attn: MultiHeadAttention
    let attnLn: LayerNorm
    let crossAttn: MultiHeadAttention?
    let crossAttnLn: LayerNorm?
    let mlp1: Linear
    let mlp2: Linear
    let mlpLn: LayerNorm

    public init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self.attn = MultiHeadAttention(nState: nState, nHead: nHead)
        self.attnLn = LayerNorm(dimensions: nState)

        if crossAttention {
            self.crossAttn = MultiHeadAttention(nState: nState, nHead: nHead)
            self.crossAttnLn = LayerNorm(dimensions: nState)
        } else {
            self.crossAttn = nil
            self.crossAttnLn = nil
        }

        let nMLP = nState * 4
        self.mlp1 = Linear(nState, nMLP)
        self.mlp2 = Linear(nMLP, nState)
        self.mlpLn = LayerNorm(dimensions: nState)
    }

    public typealias KVCache = ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)

    public func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: KVCache? = nil
    ) -> (MLXArray, KVCache, MLXArray?) {
        let (kv, crossKV) = kvCache ?? (nil, nil)

        // Self-attention
        let (y1, newKV, _) = attn(attnLn(x), mask: mask, kvCache: kv)
        var output = x + y1

        // Cross-attention (if configured)
        var crossQK: MLXArray? = nil
        var newCrossKV: (MLXArray, MLXArray)? = crossKV

        if let crossAttn = crossAttn, let crossAttnLn = crossAttnLn, let xa = xa {
            let (y2, updatedCrossKV, qk) = crossAttn(crossAttnLn(output), xa: xa, kvCache: crossKV)
            output = output + y2
            crossQK = qk
            newCrossKV = updatedCrossKV
        }

        // MLP
        output = output + mlp2(MLXNN.gelu(mlp1(mlpLn(output))))

        return (output, (newKV, newCrossKV), crossQK)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter residualBlock`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Model/ResidualAttentionBlock.swift Tests/STTTests/
git commit -m "feat(stt): implement ResidualAttentionBlock"
```

---

### Task 2.4: Implement AudioEncoder

**Files:**
- Create: `MLXAudio/STT/Whisper/Model/AudioEncoder.swift`

**Step 1: Write failing test**

```swift
// Add to Tests/STTTests/ModelTests.swift
@Test func audioEncoder_outputShape() async throws {
    // Given: AudioEncoder for base model dimensions
    let encoder = AudioEncoder(
        nMels: 80,
        nCtx: 1500,
        nState: 512,
        nHead: 8,
        nLayer: 6
    )
    let mel = MLXArray.zeros([1, 3000, 80])  // batch, frames, mels

    // When: Encoding
    let output = encoder(mel)

    // Then: Output shape is (batch, n_ctx, n_state)
    #expect(output.shape == [1, 1500, 512])
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter audioEncoder`
Expected: FAIL with "AudioEncoder not found"

**Step 3: Implement AudioEncoder**

```swift
// MLXAudio/STT/Whisper/Model/AudioEncoder.swift
import Foundation
import MLX
import MLXNN

public class AudioEncoder: Module {
    let conv1: Conv1d
    let conv2: Conv1d
    let positionalEmbedding: MLXArray
    let blocks: [ResidualAttentionBlock]
    let lnPost: LayerNorm

    public init(
        nMels: Int,
        nCtx: Int,
        nState: Int,
        nHead: Int,
        nLayer: Int,
        dtype: DType = .float16
    ) {
        self.conv1 = Conv1d(inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
        self.conv2 = Conv1d(inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        self.positionalEmbedding = Self.sinusoids(length: nCtx, channels: nState).asType(dtype)
        self.blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead) }
        self.lnPost = LayerNorm(dimensions: nState)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [batch, frames, mels]
        var output = MLXNN.gelu(conv1(x))
        output = MLXNN.gelu(conv2(output))

        // Add positional embedding
        output = output + positionalEmbedding

        // Transformer blocks
        for block in blocks {
            (output, _, _) = block(output)
        }

        return lnPost(output)
    }

    private static func sinusoids(length: Int, channels: Int, maxTimescale: Float = 10000) -> MLXArray {
        precondition(channels % 2 == 0, "channels must be even")

        let logTimescaleIncrement = log(maxTimescale) / Float(channels / 2 - 1)
        let invTimescales = MLX.exp(-logTimescaleIncrement * MLXArray(0..<(channels / 2)))
        let scaledTime = MLXArray(0..<length).expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)

        return MLX.concatenate([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter audioEncoder`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Model/AudioEncoder.swift Tests/STTTests/
git commit -m "feat(stt): implement AudioEncoder with Conv1d and Transformer"
```

---

### Task 2.5: Implement TextDecoder

**Files:**
- Create: `MLXAudio/STT/Whisper/Model/TextDecoder.swift`

**Step 1: Write failing test**

```swift
// Add to Tests/STTTests/ModelTests.swift
@Test func textDecoder_returnsCrossAttentionQK() async throws {
    // Given: TextDecoder
    let decoder = TextDecoder(
        nVocab: 51865,
        nCtx: 448,
        nState: 512,
        nHead: 8,
        nLayer: 6
    )
    let tokens = MLXArray([50258, 50259, 50360])  // SOT tokens
    let xa = MLXArray.zeros([1, 1500, 512])  // Encoder output

    // When: Decoding
    let (logits, _, crossQK) = decoder(tokens.expandedDimensions(axis: 0), xa: xa)

    // Then: Logits and cross-attention QK returned
    #expect(logits.shape[2] == 51865)  // vocab size
    #expect(crossQK.count == 6)  // one per layer
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter textDecoder`
Expected: FAIL with "TextDecoder not found"

**Step 3: Implement TextDecoder**

```swift
// MLXAudio/STT/Whisper/Model/TextDecoder.swift
import Foundation
import MLX
import MLXNN

public class TextDecoder: Module {
    let tokenEmbedding: Embedding
    var positionalEmbedding: MLXArray
    let blocks: [ResidualAttentionBlock]
    let ln: LayerNorm
    let mask: MLXArray

    public init(
        nVocab: Int,
        nCtx: Int,
        nState: Int,
        nHead: Int,
        nLayer: Int,
        dtype: DType = .float16
    ) {
        self.tokenEmbedding = Embedding(embeddingCount: nVocab, dimensions: nState)
        self.positionalEmbedding = MLXArray.zeros([nCtx, nState])
        self.blocks = (0..<nLayer).map { _ in
            ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)
        }
        self.ln = LayerNorm(dimensions: nState)
        self.mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(nCtx).asType(dtype)
    }

    public typealias KVCache = [ResidualAttentionBlock.KVCache?]

    public func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray,
        kvCache: KVCache? = nil
    ) -> (MLXArray, KVCache, [MLXArray?]) {
        let offset = kvCache?.first??.0?.shape[1] ?? 0

        // Token + positional embedding
        var output = tokenEmbedding(x) + positionalEmbedding[offset..<(offset + x.shape[1])]

        var newKVCache: KVCache = kvCache ?? Array(repeating: nil, count: blocks.count)
        var crossQK: [MLXArray?] = Array(repeating: nil, count: blocks.count)

        for (i, block) in blocks.enumerated() {
            let (blockOutput, blockKV, blockCrossQK) = block(
                output,
                xa: xa,
                mask: mask,
                kvCache: newKVCache[i]
            )
            output = blockOutput
            newKVCache[i] = blockKV
            crossQK[i] = blockCrossQK
        }

        output = ln(output)

        // Project to vocabulary using embedding weights
        let logits = MLX.matmul(output, tokenEmbedding.weight.T)

        return (logits, newKVCache, crossQK)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter textDecoder`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Model/TextDecoder.swift Tests/STTTests/
git commit -m "feat(stt): implement TextDecoder with cross-attention QK capture"
```

---

## Phase 3: AlignAtt Streaming

### Task 3.1: Implement StreamingConfig

**Files:**
- Create: `MLXAudio/STT/Whisper/Streaming/StreamingConfig.swift`

**Step 1: Create StreamingConfig**

```swift
// MLXAudio/STT/Whisper/Streaming/StreamingConfig.swift
import Foundation

public struct StreamingConfig: Sendable {
    public var frameThreshold: Int
    public var minChunkDuration: TimeInterval
    public var emitPartial: Bool

    public init(
        frameThreshold: Int = 25,
        minChunkDuration: TimeInterval = 0.5,
        emitPartial: Bool = true
    ) {
        self.frameThreshold = frameThreshold
        self.minChunkDuration = minChunkDuration
        self.emitPartial = emitPartial
    }

    public static let `default` = StreamingConfig()
}
```

**Step 2: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add MLXAudio/STT/Whisper/Streaming/StreamingConfig.swift
git commit -m "feat(stt): add StreamingConfig"
```

---

### Task 3.2: Implement Alignment Heads

**Files:**
- Create: `MLXAudio/STT/Whisper/Streaming/AlignmentHeads.swift`

**Step 1: Create AlignmentHeads**

```swift
// MLXAudio/STT/Whisper/Streaming/AlignmentHeads.swift
import Foundation

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
            return [
                (10,12), (13,17), (16,11), (17,3), (18,11), (19,9),
                (20,1), (20,8), (21,0), (21,4), (21,8), (22,3),
                (22,5), (22,7), (22,10), (22,12), (22,16), (23,0),
                (23,2), (23,4), (23,8), (23,10), (23,13)
            ]
        case .largeTurbo:
            // Provisional values for 4-layer decoder
            return [(1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
        }
    }
}
```

**Step 2: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add MLXAudio/STT/Whisper/Streaming/AlignmentHeads.swift
git commit -m "feat(stt): add alignment heads for all Whisper models"
```

---

### Task 3.3: Implement StreamingDecoder Core

**Files:**
- Create: `MLXAudio/STT/Whisper/Streaming/StreamingDecoder.swift`
- Create: `Tests/STTTests/StreamingDecoderTests.swift`

**Step 1: Write failing test**

```swift
// Tests/STTTests/StreamingDecoderTests.swift
import Testing
import MLX
@testable import MLXAudio

struct StreamingDecoderTests {
    @Test func getMostAttendedFrame_findsMaxAttention() async throws {
        // Given: Cross-attention weights with known peak
        // Shape: [batch, heads, tokens, frames]
        var weights = MLXArray.zeros([1, 8, 1, 100])
        // Set peak at frame 42
        weights[0, 0, 0, 42] = MLXArray([1.0])

        let crossQK = Array(repeating: weights, count: 6)
        let alignmentHeads = [(0, 0)]  // Use first head of first layer

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Returns frame 42
        #expect(frame == 42)
    }

    @Test func shouldEmit_nearEnd_returnsFalse() async throws {
        // Given: Most attended frame near end of audio
        let mostAttendedFrame = 95
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let shouldEmit = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should NOT emit (only 5 frames from end < threshold)
        #expect(!shouldEmit)
    }

    @Test func shouldEmit_farFromEnd_returnsTrue() async throws {
        // Given: Most attended frame far from end
        let mostAttendedFrame = 50
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let shouldEmit = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should emit (50 frames from end >= threshold)
        #expect(shouldEmit)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter StreamingDecoderTests`
Expected: FAIL with "StreamingDecoder not found"

**Step 3: Implement StreamingDecoder**

```swift
// MLXAudio/STT/Whisper/Streaming/StreamingDecoder.swift
import Foundation
import MLX

public enum StreamingDecoder {
    /// Find the audio frame that received the most attention from the last decoded token
    /// - Parameters:
    ///   - crossQK: Cross-attention weights from each decoder layer, shape [batch, heads, tokens, frames]
    ///   - alignmentHeads: List of (layer, head) tuples to use for alignment
    /// - Returns: Frame index with highest average attention
    public static func getMostAttendedFrame(
        crossQK: [MLXArray?],
        alignmentHeads: [(layer: Int, head: Int)]
    ) -> Int {
        // Collect attention weights from alignment heads
        var weights: [MLXArray] = []

        for (layer, head) in alignmentHeads {
            guard layer < crossQK.count, let layerQK = crossQK[layer] else { continue }
            // Extract last token's attention: [batch, head, -1, frames] -> [frames]
            let attention = layerQK[0, head, -1, 0...]
            weights.append(attention)
        }

        guard !weights.isEmpty else { return 0 }

        // Average across heads
        let stacked = MLX.stack(weights, axis: 0)
        let avgAttention = stacked.mean(axis: 0)

        // Find max
        return Int(MLX.argMax(avgAttention).item(Int.self))
    }

    /// Determine if the current token should be emitted based on attention stability
    /// - Parameters:
    ///   - mostAttendedFrame: Frame index with highest attention
    ///   - totalContentFrames: Total number of audio content frames
    ///   - threshold: Minimum distance from end before emitting
    /// - Returns: True if token should be emitted
    public static func shouldEmit(
        mostAttendedFrame: Int,
        totalContentFrames: Int,
        threshold: Int
    ) -> Bool {
        let distanceToEnd = totalContentFrames - mostAttendedFrame
        return distanceToEnd >= threshold
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter StreamingDecoderTests`
Expected: PASS

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/Streaming/StreamingDecoder.swift Tests/STTTests/
git commit -m "feat(stt): implement StreamingDecoder with AlignAtt core logic"
```

---

### Task 3.4: Implement WhisperSession

**Files:**
- Create: `MLXAudio/STT/Whisper/WhisperSession.swift`
- Create: `MLXAudio/STT/Whisper/WhisperError.swift`
- Create: `MLXAudio/STT/Whisper/TranscriptionOptions.swift`

**Step 1: Create WhisperError**

```swift
// MLXAudio/STT/Whisper/WhisperError.swift
import Foundation

public enum WhisperError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelDownloadFailed(URL, underlying: Error)
    case invalidModelFormat(String)
    case invalidAudioFormat(expected: String, got: String)
    case audioTooShort(minSeconds: Double)
    case sampleRateMismatch(expected: Int, got: Int)
    case encodingFailed(String)
    case decodingFailed(String)
    case cancelled
    case timeout(TimeInterval)
    case tokenizerLoadFailed(String)
    case insufficientMemory(required: Int, available: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .modelDownloadFailed(let url, let error):
            return "Failed to download model from \(url): \(error.localizedDescription)"
        case .invalidModelFormat(let reason):
            return "Invalid model format: \(reason)"
        case .invalidAudioFormat(let expected, let got):
            return "Invalid audio format. Expected \(expected), got \(got)"
        case .audioTooShort(let min):
            return "Audio too short. Minimum duration: \(min) seconds"
        case .sampleRateMismatch(let expected, let got):
            return "Sample rate mismatch. Expected \(expected) Hz, got \(got) Hz"
        case .encodingFailed(let reason):
            return "Audio encoding failed: \(reason)"
        case .decodingFailed(let reason):
            return "Decoding failed: \(reason)"
        case .cancelled:
            return "Transcription was cancelled"
        case .timeout(let duration):
            return "Transcription timed out after \(duration) seconds"
        case .tokenizerLoadFailed(let reason):
            return "Failed to load tokenizer: \(reason)"
        case .insufficientMemory(let required, let available):
            return "Insufficient GPU memory. Required: \(required) MB, Available: \(available) MB"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound:
            return "Check network connection and try again"
        case .modelDownloadFailed:
            return "Verify URL and retry, or use a local model path"
        case .sampleRateMismatch(let expected, _):
            return "Resample audio to \(expected) Hz before transcription"
        case .audioTooShort(let min):
            return "Provide audio at least \(min) seconds long"
        case .timeout:
            return "Try smaller audio chunks or increase timeout"
        case .insufficientMemory:
            return "Use a smaller model (tiny/base) or free GPU memory"
        default:
            return nil
        }
    }
}
```

**Step 2: Create TranscriptionOptions**

```swift
// MLXAudio/STT/Whisper/TranscriptionOptions.swift
import Foundation

public struct TranscriptionOptions: Sendable {
    public var language: String?
    public var task: Task
    public var timeout: TimeInterval
    public var temperature: Float

    public enum Task: String, Sendable {
        case transcribe
        case translate
    }

    public init(
        language: String? = nil,
        task: Task = .transcribe,
        timeout: TimeInterval = 30.0,
        temperature: Float = 0.0
    ) {
        self.language = language
        self.task = task
        self.timeout = timeout
        self.temperature = temperature
    }

    public static let `default` = TranscriptionOptions()
}
```

**Step 3: Create WhisperSession skeleton**

```swift
// MLXAudio/STT/Whisper/WhisperSession.swift
import Foundation
import MLX
import MLXLMCommon

public final class WhisperSession: STTSession, @unchecked Sendable {
    private let model: WhisperModel
    private let streamingConfig: StreamingConfig
    private var currentTask: Task<Void, Never>?

    // TODO: Add actual model components
    // private var encoder: AudioEncoder?
    // private var decoder: TextDecoder?
    // private var tokenizer: WhisperTokenizer?

    private init(model: WhisperModel, streamingConfig: StreamingConfig) {
        self.model = model
        self.streamingConfig = streamingConfig
    }

    // MARK: - Factory

    public static func fromPretrained(
        model: WhisperModel = .largeTurbo,
        streaming: StreamingConfig = .default,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> WhisperSession {
        progressHandler?(.downloading(0))

        // TODO: Implement model downloading and loading

        progressHandler?(.loading(1.0))

        return WhisperSession(model: model, streamingConfig: streaming)
    }

    // MARK: - STTSession

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) -> AsyncThrowingStream<StreamingResult, Error> {
        transcribe(audio, sampleRate: sampleRate, options: .default)
    }

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) async throws -> String {
        try await transcribe(audio, sampleRate: sampleRate, options: .default)
    }

    // MARK: - Extended API

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = 16000,
        options: TranscriptionOptions = .default
    ) -> AsyncThrowingStream<StreamingResult, Error> {
        AsyncThrowingStream { continuation in
            currentTask = Task {
                do {
                    // Validate sample rate
                    guard sampleRate == AudioConstants.sampleRate else {
                        throw WhisperError.sampleRateMismatch(
                            expected: AudioConstants.sampleRate,
                            got: sampleRate
                        )
                    }

                    // TODO: Implement actual transcription loop
                    // This is a placeholder that emits a single result

                    try Task.checkCancellation()

                    continuation.yield(StreamingResult(
                        text: "Transcription not yet implemented",
                        isFinal: true,
                        timestamp: 0...0
                    ))

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: WhisperError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = 16000,
        options: TranscriptionOptions = .default
    ) async throws -> String {
        var result = ""
        for try await chunk in transcribe(audio, sampleRate: sampleRate, options: options) {
            if chunk.isFinal {
                result = chunk.text
            }
        }
        return result
    }

    // MARK: - Lifecycle

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
    }

    public func cleanupMemory() {
        // TODO: Release model weights
    }

    public var estimatedMemoryUsage: Int {
        model.estimatedMemoryMB * 1024 * 1024
    }
}

// MARK: - Progress

public enum WhisperProgress: Sendable {
    case downloading(Float)
    case loading(Float)
    case encoding
    case decoding(Float)
}
```

**Step 4: Verify compilation**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 5: Commit**

```bash
git add MLXAudio/STT/Whisper/
git commit -m "feat(stt): add WhisperSession skeleton with error types"
```

---

## Phase 4: Integration & Testing

### Task 4.1: Write Integration Test

**Files:**
- Create: `Tests/STTTests/WhisperSessionTests.swift`

**Step 1: Write integration test**

```swift
// Tests/STTTests/WhisperSessionTests.swift
import Testing
import MLX
@testable import MLXAudio

struct WhisperSessionTests {
    @Test func transcribe_invalidSampleRate_throws() async throws {
        // Given: A session
        let session = try await WhisperSession.fromPretrained(model: .tiny)
        let audio = MLXArray.zeros([44100])  // 1 second at wrong sample rate

        // When/Then: Throws sample rate mismatch
        await #expect(throws: WhisperError.self) {
            _ = try await session.transcribe(audio, sampleRate: 44100)
        }
    }

    @Test func cancel_stopsTranscription() async throws {
        // Given: An ongoing transcription
        let session = try await WhisperSession.fromPretrained(model: .tiny)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        let stream = session.transcribe(audio, sampleRate: 16000)

        // When: Cancelled immediately
        session.cancel()

        // Then: Stream completes with cancellation
        var wasCancelled = false
        do {
            for try await _ in stream { }
        } catch WhisperError.cancelled {
            wasCancelled = true
        }

        #expect(wasCancelled)
    }
}
```

**Step 2: Run tests**

Run: `swift test --filter WhisperSessionTests`
Expected: Tests pass (session returns placeholder, cancel works)

**Step 3: Commit**

```bash
git add Tests/STTTests/WhisperSessionTests.swift
git commit -m "test(stt): add WhisperSession integration tests"
```

---

### Task 4.2: Update Package.swift with STT Files

**Files:**
- Modify: `Package.swift`

**Step 1: Update Package.swift to include STT path**

The STT files should already be included since they're in the MLXAudio target path. Verify by building.

**Step 2: Verify all tests pass**

Run: `swift test`
Expected: All tests pass

**Step 3: Commit if changes needed**

```bash
git add Package.swift
git commit -m "chore: update Package.swift for STT module"
```

---

## Summary

This plan implements the core STT infrastructure in 4 phases:

1. **Phase 1** (Tasks 1.1-1.5): Project setup, dependencies, audio processing
2. **Phase 2** (Tasks 2.1-2.5): Whisper model architecture (Attention, Encoder, Decoder)
3. **Phase 3** (Tasks 3.1-3.4): AlignAtt streaming logic
4. **Phase 4** (Task 4.1-4.2): Integration and testing

**Not included in this plan (future work):**
- Actual model weight loading from HuggingFace
- Tokenizer integration with swift-tiktoken
- Full decoding loop with KV cache
- Benchmark tests for latency/RTF

These will be addressed in a follow-up implementation plan once the core architecture is validated.
