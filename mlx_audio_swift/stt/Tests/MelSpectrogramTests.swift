import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct MelSpectrogramTests {
    @Test func outputShapeIsCorrect() async throws {
        // 1 second of audio at 16kHz = 16000 samples
        let audio = MLXArray.zeros([16000])
        let mel = try MelSpectrogram.compute(audio: audio)

        // Expected: (nMels, nFrames) = (80, 100) for 1 second
        // 16000 samples / 160 hop = 100 frames
        #expect(mel.shape == [80, 100])
    }

    @Test func fullChunkOutputShape() async throws {
        // Full 30-second chunk = 480000 samples
        let audio = MLXArray.zeros([480000])
        let mel = try MelSpectrogram.compute(audio: audio)

        // Expected: (80, 3000) for full chunk
        #expect(mel.shape == [80, 3000])
    }

    @Test func outputIsNormalized() async throws {
        // Create simple sine wave
        let sampleCount = 16000
        let frequency: Double = 440.0
        let sampleRate: Double = 16000.0
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            let phase = Double(i) * 2.0 * Double.pi * frequency / sampleRate
            samples[i] = Float(Foundation.sin(phase))
        }
        let audio = MLXArray(samples)
        let mel = try MelSpectrogram.compute(audio: audio)

        // Mel spectrogram should be log-scaled, typically in range [-4, 4] after normalization
        let maxVal = MLX.max(mel).item(Float.self)
        let minVal = MLX.min(mel).item(Float.self)
        #expect(maxVal <= 4.0)
        #expect(minVal >= -4.0)
    }

    @Test func handlesShortAudio() async throws {
        // Very short audio: 0.1 second = 1600 samples
        let audio = MLXArray.zeros([1600])
        let mel = try MelSpectrogram.compute(audio: audio)

        // Expected: 1600 / 160 = 10 frames
        #expect(mel.shape == [80, 10])
    }

    @Test func outputDTypeIsFloat32() async throws {
        let audio = MLXArray.zeros([16000])
        let mel = try MelSpectrogram.compute(audio: audio)

        #expect(mel.dtype == .float32)
    }
}
