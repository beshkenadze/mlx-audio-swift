//
//  MLXAudioTests.swift
//  MLXAudioTests
//
//  Created by Ben Harraway on 14/04/2025.
//


import Testing
import MLX
import Foundation
import MLX
import Foundation

@testable import MLXAudioTTS
@testable import MLXAudioCodecs


// Run ALL tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/MLXAudioCodecsTests
//  2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


// MARK: - SNAC Tests
// Run SNAC tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/SNACTests
//  2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct SNACTests {

    @Test func testSNACEncodeDecodeCycle() async throws {
        // 1. Load audio from file
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

        // 2. Load SNAC model from HuggingFace (24kHz model)
        print("\u{001B}[33mLoading SNAC model...\u{001B}[0m")
        let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        print("\u{001B}[32mSNAC model loaded!\u{001B}[0m")

        // 3. Reshape audio for SNAC: [batch, channels, samples]
        let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
        print("Audio input shape: \(audioInput.shape)")

        // 4. Encode audio to codes
        print("\u{001B}[33mEncoding audio...\u{001B}[0m")
        let codes = snac.encode(audioInput)
        print("Encoded to \(codes.count) codebook levels:")
        for (i, code) in codes.enumerated() {
            print("  Level \(i): \(code.shape)")
        }

        // 5. Decode codes back to audio
        print("\u{001B}[33mDecoding audio...\u{001B}[0m")
        let reconstructed = snac.decode(codes)
        print("Reconstructed audio shape: \(reconstructed.shape)")

        // 6. Save reconstructed audio to the same media folder as input
        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_snac_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()  // Remove batch/channel dims
        try saveAudioArray(outputAudio, sampleRate: Double(snac.samplingRate), to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        // Basic check: output should have samples
        #expect(reconstructed.shape.last! > 0)
    }
}


// MARK: - Mimi Tests
// Run Mimi tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/MimiTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct MimiTests {

    @Test func testMimiEncodeDecodeCycle() async throws {
        // 1. Load audio from file
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

        // 2. Load Mimi model from HuggingFace
        print("\u{001B}[33mLoading Mimi model...\u{001B}[0m")
        let mimi = try await Mimi.fromPretrained(
            repoId: "kyutai/moshiko-pytorch-bf16",
            filename: "tokenizer-e351c8d8-checkpoint125.safetensors"
        ) { progress in
            print("Download progress: \(progress.fractionCompleted * 100)%")
        }
        print("\u{001B}[32mMimi model loaded!\u{001B}[0m")

        // 3. Reshape audio for Mimi: [batch, channels, samples]
        let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
        print("Audio input shape: \(audioInput.shape)")

        // 4. Encode audio to codes
        print("\u{001B}[33mEncoding audio...\u{001B}[0m")
        let codes = mimi.encode(audioInput)
        print("Encoded to codes shape: \(codes.shape)")

        // 5. Decode codes back to audio
        print("\u{001B}[33mDecoding audio...\u{001B}[0m")
        let reconstructed = mimi.decode(codes)
        GPU.clearCache()
        print("Reconstructed audio shape: \(reconstructed.shape)")

        // 6. Save reconstructed audio
        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_mimi_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()  // Remove batch/channel dims
        try saveAudioArray(outputAudio, sampleRate: mimi.sampleRate, to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        // Basic check: output should have samples
        #expect(reconstructed.shape.last! > 0)
    }
}


// MARK: - Vocos Tests
// Run Vocos tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/VocosTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct VocosTests {

    @Test func testConvNeXtBlock() throws {
        // Test basic ConvNeXtBlock forward pass
        let dim = 64
        let intermediateDim = 192
        let block = ConvNeXtBlock(
            dim: dim,
            intermediateDim: intermediateDim,
            layerScaleInitValue: 0.125,
            dwKernelSize: 7
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim])
        let output = block(input)

        // Output should have same shape as input (residual connection)
        #expect(output.shape == input.shape)
        print("ConvNeXtBlock output shape: \(output.shape)")
    }

    @Test func testConvNeXtBlockWithAdaNorm() throws {
        // Test ConvNeXtBlock with adaptive normalization
        let dim = 64
        let intermediateDim = 192
        let numEmbeddings = 4

        let block = ConvNeXtBlock(
            dim: dim,
            intermediateDim: intermediateDim,
            layerScaleInitValue: 0.125,
            adanormNumEmbeddings: numEmbeddings,
            dwKernelSize: 7
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim])
        let condEmbedding = MLXRandom.normal([1, numEmbeddings])

        let output = block(input, condEmbeddingId: condEmbedding)

        // Output should have same shape as input
        #expect(output.shape == input.shape)
        print("ConvNeXtBlock with AdaNorm output shape: \(output.shape)")
    }

    @Test func testVocosBackbone() throws {
        // Test VocosBackbone forward pass
        let inputChannels = 100
        let dim = 512
        let intermediateDim = 1536
        let numLayers = 8

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers
        )

        // Input shape: (batch, length, input_channels)
        let input = MLXRandom.normal([1, 50, inputChannels])
        let output = backbone(input)

        // Output should have shape (batch, length, dim)
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 50)
        #expect(output.shape[2] == dim)
        print("VocosBackbone output shape: \(output.shape)")
    }

    @Test func testVocosBackboneWithAdaNorm() throws {
        // Test VocosBackbone with adaptive normalization
        let inputChannels = 100
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let numEmbeddings = 4

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers,
            adanormNumEmbeddings: numEmbeddings
        )

        // Input shape: (batch, length, input_channels)
        let input = MLXRandom.normal([1, 50, inputChannels])
        let bandwidthId = MLXRandom.normal([1, numEmbeddings])

        let output = backbone(input, bandwidthId: bandwidthId)

        // Output should have shape (batch, length, dim)
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 50)
        #expect(output.shape[2] == dim)
        print("VocosBackbone with AdaNorm output shape: \(output.shape)")
    }

    @Test func testISTFTHead() throws {
        // Test ISTFTHead forward pass
        let dim = 512
        let nFft = 1024
        let hopLength = 256

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        // Input shape: (batch, length, dim)
        let numFrames = 100
        let input = MLXRandom.normal([1, numFrames, dim])

        let output = head(input)

        // Output should be audio waveform
        // Expected length: approximately (numFrames - 1) * hopLength after trimming
        #expect(output.ndim == 1 || output.ndim == 2)
        print("ISTFTHead output shape: \(output.shape)")
    }

    @Test func testAdaLayerNorm() throws {
        // Test AdaLayerNorm
        let numEmbeddings = 4
        let embeddingDim = 256

        let adaNorm = AdaLayerNorm(
            numEmbeddings: numEmbeddings,
            embeddingDim: embeddingDim
        )

        // Input shape: (batch, length, dim)
        let input = MLXRandom.normal([2, 50, embeddingDim])
        let condEmbedding = MLXRandom.normal([2, numEmbeddings])

        let output = adaNorm(input, condEmbedding: condEmbedding)

        // Output should have same shape as input
        #expect(output.shape == input.shape)
        print("AdaLayerNorm output shape: \(output.shape)")
    }

    @Test func testVocosModel() throws {
        // Test full Vocos model
        let inputChannels = 100
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let nFft = 1024
        let hopLength = 256

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers
        )

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        let vocos = Vocos(backbone: backbone, head: head)

        // Input shape: (batch, length, input_channels)
        let numFrames = 50
        let input = MLXRandom.normal([1, numFrames, inputChannels])

        let output = vocos(input)

        // Output should be audio waveform
        print("Vocos output shape: \(output.shape)")
        #expect(output.shape.count >= 1)
    }

    @Test func testVocosDecodeWithBandwidthId() throws {
        // Test Vocos decode with bandwidth conditioning
        let inputChannels = 128
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let numEmbeddings = 4
        let nFft = 1024
        let hopLength = 256

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers,
            adanormNumEmbeddings: numEmbeddings
        )

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        let vocos = Vocos(backbone: backbone, head: head)

        // Input shape: (batch, length, input_channels)
        let numFrames = 50
        let input = MLXRandom.normal([1, numFrames, inputChannels])
        let bandwidthId = MLXRandom.normal([1, numEmbeddings])

        let output = vocos.decode(input, bandwidthId: bandwidthId)

        // Output should be audio waveform
        print("Vocos decode with bandwidthId output shape: \(output.shape)")
        #expect(output.shape.count >= 1)
    }
}
