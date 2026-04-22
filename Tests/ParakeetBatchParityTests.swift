import Foundation
import MLX
import Testing
import Darwin

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite("Parakeet Batch Parity Tests", .serialized)
struct ParakeetBatchParityTests {
    private func makeFixtureModel() throws -> ParakeetModel {
        let fixtureDir = try makeFixtureDirectory()
        return try ParakeetModel.fromDirectory(fixtureDir)
    }

    @Test("generateBatch preserves order and text parity for chunk-sized audio")
    func generateBatchPreservesOrderAndTextParity() throws {
        let model = try makeFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 3_200, frequency: 180),
            makeChunkAudio(sampleCount: 9_600, frequency: 260),
            makeChunkAudio(sampleCount: 16_000, frequency: 340),
        ]

        let singleOutputs = audios.map { model.generate(audio: $0) }
        let singleSignatures = singleOutputs.map(outputSignature)
        #expect(Set(singleSignatures).count == singleSignatures.count)

        let batchOutputs = try model.generateBatch(audios: audios)

        #expect(batchOutputs.count == audios.count)
        #expect(batchOutputs.map(outputSignature) == singleSignatures)
    }

    @Test("generateBatch rejects empty input")
    func generateBatchRejectsEmptyInput() throws {
        let model = try makeFixtureModel()

        #expect(throws: STTError.self) {
            _ = try model.generateBatch(audios: [])
        }
    }

    @Test("generateBatch matches single-item generate for singleton input")
    func generateBatchMatchesSingleItemGenerate() throws {
        let model = try makeFixtureModel()
        let audio = makeChunkAudio(sampleCount: 8_000, frequency: 220)

        let single = model.generate(audio: audio)
        let batch = try model.generateBatch(audios: [audio])

        #expect(batch.count == 1)
        #expect(outputSignature(batch[0]) == outputSignature(single))
    }

    @Test("generateBatch normalizes multichannel audio to mono before mel extraction")
    func generateBatchNormalizesMultichannelAudioToMono() throws {
        let model = try makeFixtureModel()
        let mono = makeChunkAudio(sampleCount: 12_000, frequency: 300)
        let stereo = makeStereoAudio(from: mono)

        let monoOutput = try model.generateBatch(audios: [mono])
        let stereoOutput = try model.generateBatch(audios: [stereo])

        #expect(monoOutput.count == 1)
        #expect(stereoOutput.count == 1)
        #expect(outputSignature(monoOutput[0]) == outputSignature(stereoOutput[0]))
    }

    @Test("benchmark metadata captures the batch contract")
    func benchmarkMetadataCapturesBatchContract() throws {
        let timed = measureWallClock {
            Thread.sleep(forTimeInterval: 0.001)
            return 42
        }

        #expect(timed.wallClock >= 0)
        #expect(timed.value == 42)

        let result = BatchBenchmarkResult(
            checkpoint: "mlx-community/parakeet-tdt-0.6b-v3",
            batchSize: 4,
            warmupRuns: 1,
            measuredRuns: 3,
            medianWallClock: median([0.9, 0.4, 0.7]),
            peakRSSBytes: captureCurrentRSSBytes(),
            maxFrameLength: 123
        )

        #expect(result.checkpoint == "mlx-community/parakeet-tdt-0.6b-v3")
        #expect(result.batchSize == 4)
        #expect(result.warmupRuns == 1)
        #expect(result.measuredRuns == 3)
        #expect(result.medianWallClock == 0.7)
        #expect(result.peakRSSBytes > 0)
        #expect(result.maxFrameLength == 123)
        #expect(result.summary.contains("parakeet-tdt-0.6b-v3"))
        #expect(result.summary.contains("batch_size=4"))
        #expect(result.summary.contains("warmup_runs=1"))
        #expect(result.summary.contains("measured_runs=3"))
        #expect(result.summary.contains("median_wall_clock=0.700000"))
        #expect(result.summary.contains("max_frame_length=123"))
    }

    @Test("benchmark harness measures single and batched Parakeet paths")
    func benchmarkHarnessMeasuresSingleAndBatchedPaths() throws {
        let model = try makeFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 4_000, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 240),
        ]

        let singleTimed = measureWallClock {
            audios.map { model.generate(audio: $0) }
        }
        let batchTimed = try measureWallClock {
            try model.generateBatch(audios: audios)
        }

        #expect(singleTimed.value.map(outputSignature) == batchTimed.value.map(outputSignature))

        let maxFrameLength = audios
            .map { ParakeetAudio.logMelSpectrogram($0, config: model.preprocessConfig).shape[1] }
            .max() ?? 0

        let benchmark = BatchBenchmarkResult(
            checkpoint: "fixture/parakeet-ctc-batch-smoke",
            batchSize: audios.count,
            warmupRuns: 0,
            measuredRuns: 1,
            medianWallClock: batchTimed.wallClock,
            peakRSSBytes: captureCurrentRSSBytes(),
            maxFrameLength: maxFrameLength
        )

        #expect(singleTimed.wallClock >= 0)
        #expect(batchTimed.wallClock >= 0)
        #expect(benchmark.batchSize == 2)
        #expect(benchmark.maxFrameLength > 0)
        #expect(benchmark.peakRSSBytes > 0)
    }
}

private struct BatchBenchmarkResult {
    let checkpoint: String
    let batchSize: Int
    let warmupRuns: Int
    let measuredRuns: Int
    let medianWallClock: TimeInterval
    let peakRSSBytes: UInt64
    let maxFrameLength: Int

    var summary: String {
        let formattedMedian = String(format: "%.6f", medianWallClock)
        return "checkpoint=\(checkpoint) batch_size=\(batchSize) warmup_runs=\(warmupRuns) measured_runs=\(measuredRuns) median_wall_clock=\(formattedMedian) peak_rss_bytes=\(peakRSSBytes) max_frame_length=\(maxFrameLength)"
    }
}

private func makeFixtureDirectory() throws -> URL {
    let fixtureDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("parakeet-batch-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

    let configJSON = """
    {
      "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
      "preprocessor": {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.02,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "dither": 0.0
      },
      "encoder": {
        "feat_in": 80,
        "n_layers": 0,
        "d_model": 16,
        "n_heads": 4,
        "ff_expansion_factor": 2,
        "subsampling_factor": 2,
        "self_attention_model": "abs_pos",
        "subsampling": "dw_striding",
        "conv_kernel_size": 15,
        "subsampling_conv_channels": 16,
        "pos_emb_max_len": 128
      },
      "decoder": {
        "feat_in": 16,
        "num_classes": 4,
        "vocabulary": ["▁", "a", "b", "."]
      },
      "decoding": {"greedy": {"max_symbols": 8}}
    }
    """
    try configJSON.write(
        to: fixtureDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let weights: [String: MLXArray] = [
        "encoder.pre_encode.conv0.weight": MLXArray.zeros([16, 3, 3, 1], type: Float.self),
        "encoder.pre_encode.conv0.bias": MLXArray.zeros([16], type: Float.self),
        "encoder.pre_encode.out.weight": MLXArray.zeros([16, 640], type: Float.self),
        "encoder.pre_encode.out.bias": MLXArray.zeros([16], type: Float.self),
        "decoder.decoder_layers.0.weight": MLXArray.zeros([5, 1, 16], type: Float.self),
        "decoder.decoder_layers.0.bias": MLXArray([Float(0.3), 0.2, 0.1, -0.1, -0.5]),
    ]
    try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

    return fixtureDir
}

private func makeChunkAudio(sampleCount: Int, frequency: Float) -> MLXArray {
    let sampleRate = 16_000.0
    let values = (0..<sampleCount).map { index in
        let phase = (2.0 * Double.pi * Double(frequency) * Double(index)) / sampleRate
        return Float(Darwin.sin(phase)) * Float(0.25)
    }
    return MLXArray(values)
}

private func makeStereoAudio(from mono: MLXArray) -> MLXArray {
    let left = mono.expandedDimensions(axis: 1)
    let right = mono.expandedDimensions(axis: 1)
    return MLX.concatenated([left, right], axis: 1)
}

private func outputSignature(_ output: STTOutput) -> String {
    let segments = (output.segments ?? []).map { segment -> String in
        let text = segment["text"] as? String ?? ""
        let start = segment["start"] as? Double ?? -1
        let end = segment["end"] as? Double ?? -1
        let formattedStart = String(format: "%.5f", start)
        let formattedEnd = String(format: "%.5f", end)
        return "\(text)@\(formattedStart)-\(formattedEnd)"
    }
    return "\(output.text)|\(segments.joined(separator: ","))"
}

private func measureWallClock<T>(_ body: () throws -> T) rethrows -> (value: T, wallClock: TimeInterval) {
    let start = CFAbsoluteTimeGetCurrent()
    let value = try body()
    return (value, CFAbsoluteTimeGetCurrent() - start)
}

private func median(_ values: [TimeInterval]) -> TimeInterval {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let midpoint = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[midpoint - 1] + sorted[midpoint]) / 2
    }
    return sorted[midpoint]
}

private func captureCurrentRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
        }
    }

    guard result == KERN_SUCCESS else {
        return 0
    }
    return UInt64(info.resident_size)
}
