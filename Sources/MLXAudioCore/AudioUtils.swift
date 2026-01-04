import AVFoundation
import Foundation
import MLX

class AudioUtils {
  enum AudioUtilsErrors: Error {
    case cannotCreateAVAudioFormat
  }

  private init() {}

  // Debug method to write output to .wav file for checking the speech generation
  static func writeWavFile(samples: [Float], sampleRate: Double, fileURL: URL) throws {
    let frameCount = AVAudioFrameCount(samples.count)

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
          let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
    else {
      throw AudioUtilsErrors.cannotCreateAVAudioFormat
    }

    buffer.frameLength = frameCount
    let channelData = buffer.floatChannelData![0]
    for i in 0 ..< Int(frameCount) {
      channelData[i] = samples[i]
    }

    let audioFile = try AVAudioFile(
      forWriting: fileURL,
      settings: format.settings,
      commonFormat: format.commonFormat,
      interleaved: format.isInterleaved
    )

    try audioFile.write(from: buffer)
  }
}




/// Load audio from a file and return the sample rate and audio data.
func loadAudioArray(from url: URL) throws -> (Int, MLXArray) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw NSError(domain: "TestHelpers", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    try audioFile.read(into: buffer)

    guard let floatChannelData = buffer.floatChannelData else {
        throw NSError(domain: "TestHelpers", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
    }

    let sampleRate = Int(format.sampleRate)
    let samples = Array(UnsafeBufferPointer(start: floatChannelData[0], count: Int(buffer.frameLength)))
    let audioData = MLXArray(samples)

    return (sampleRate, audioData)
}

/// Save audio data to a WAV file.
func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)

    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw NSError(domain: "TestHelpers", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    buffer.frameLength = frameCount

    if let channelData = buffer.floatChannelData {
        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }
    }

    try audioFile.write(from: buffer)
}
