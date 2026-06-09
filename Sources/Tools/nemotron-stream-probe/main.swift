import Foundation
import MLX
import MLXAudioSTT

// On-ANE smoke probe for the cache-aware streaming CoreML encoder. `swift test` can't load
// MLX's metallib, so this runs the same checks as an executable (which does run Metal):
// loads the .mlpackage, feeds uniform-121 windows, and verifies finite stride-read frames of
// the right shape plus real cache threading (same window twice → different output; reset →
// initial-state output).
//
//   swift run nemotron-stream-probe tools/coreml-ane/out/nemotron_stream_func.mlpackage

#if canImport(CoreML)
let path = CommandLine.arguments.dropFirst().first
    ?? ProcessInfo.processInfo.environment["NEMOTRON_STREAM_MLPACKAGE"] ?? ""
guard FileManager.default.fileExists(atPath: path) else {
    print("usage: nemotron-stream-probe <stream.mlpackage>  (or set NEMOTRON_STREAM_MLPACKAGE)")
    exit(2)
}

let enc = try NemotronCoreMLStreamingEncoder(
    modelURL: URL(fileURLWithPath: path),
    featIn: 128, dModel: 1024, subsamplingFactor: 8,
    preFrames: 9, newFrames: 112, layers: 24, attnCache: 70, convCache: 8)
print("fixedFrames=\(enc.fixedFrames)  (expect 121)")

let vals = (0..<(121 * 128)).map { Float($0 % 97) * 0.01 - 0.5 }
let window = MLXArray(vals, [1, 121, 128])

let o1 = try enc.step(window)
let f1 = o1.asArray(Float.self)
let finite = f1.allSatisfy { $0.isFinite }
print("chunk1: shape=\(o1.shape) finite=\(finite) first3=\(Array(f1.prefix(3)))")

let f2 = try enc.step(window).asArray(Float.self)
let threaded = (f1 != f2)
print("chunk2 (same input): differs=\(threaded)  → caches are threaded")

enc.reset()
let f3 = try enc.step(window).asArray(Float.self)
let maxDiff = zip(f1, f3).map { abs($0 - $1) }.max() ?? 1
print("after reset(): maxDiff vs chunk1=\(maxDiff)  (≈0 expected)")

let pass = finite && threaded && maxDiff < 1e-3
    && o1.shape == [1, 1024, o1.shape[2]] && o1.shape[2] >= 1 && o1.shape[2] <= 16
print(pass ? "PROBE PASS ✅" : "PROBE FAIL ❌")
exit(pass ? 0 : 1)
#else
print("CoreML unavailable")
exit(2)
#endif
