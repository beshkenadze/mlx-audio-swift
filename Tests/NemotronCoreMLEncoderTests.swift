#if canImport(CoreML)
import Foundation
import Testing

@testable import MLXAudioSTT

@Suite("Nemotron CoreML Encoder Tests")
struct NemotronCoreMLEncoderTests {
    /// The offline Nemotron encoder reuses the generic fixed-shape Conformer CoreML encoder
    /// (`ConformerCoreMLEncoder`). A missing/invalid model must surface as a thrown error — the
    /// model then falls back to the MLX encoder — never a crash.
    @Test func conformerEncoderThrowsOnMissingModel() {
        let bogus = URL(fileURLWithPath: "/nonexistent/nemotron_enc.mlpackage")
        #expect(throws: (any Error).self) {
            _ = try ConformerCoreMLEncoder(
                modelURL: bogus, featIn: 128, fixedFrames: 1000, subsamplingFactor: 8)
        }
    }

    /// Subsampled-length math (8× dw-striding: `floor((L-1)/2)+1`, log2(8)=3 times). Nemotron's
    /// FastConformer uses the same subsampling as Parakeet, so the shared encoder applies.
    @Test func subsampledLengthMatchesDwStriding() {
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1000, subsamplingFactor: 8) == 125)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 112, subsamplingFactor: 8) == 14)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1, subsamplingFactor: 8) == 1)
    }

    /// The cache-aware **streaming** encoder (4-in/4-out functional model with manual cache
    /// threading) must also surface a missing model as a thrown error, never a crash — the
    /// streaming path then stays on MLX.
    @Test func streamingEncoderThrowsOnMissingModel() {
        let bogus = URL(fileURLWithPath: "/nonexistent/nemotron_stream_func.mlpackage")
        #expect(throws: (any Error).self) {
            _ = try NemotronCoreMLStreamingEncoder(
                modelURL: bogus, featIn: 128, dModel: 1024, subsamplingFactor: 8,
                preFrames: 9, newFrames: 112, layers: 24, attnCache: 70, convCache: 8)
        }
    }

    // The on-ANE run + cache-threading + reset checks need MLX/Metal, which `swift test` can't load
    // (no metallib in the test bundle). They live in the `nemotron-stream-probe` executable instead:
    //   swift run nemotron-stream-probe tools/coreml-ane/out/nemotron_stream_func.mlpackage
}
#endif
