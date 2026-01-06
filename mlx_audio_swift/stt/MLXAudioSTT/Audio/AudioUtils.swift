import Foundation
import MLX

/// Audio utility functions for Whisper STT
public enum AudioUtils {

    /// Pad or trim audio to a specific length
    /// - Parameters:
    ///   - audio: Input audio as MLXArray, shape [nSamples]
    ///   - length: Target length in samples (default: 480000 for 30s at 16kHz)
    /// - Returns: Audio padded with zeros or trimmed to exact length
    public static func padOrTrim(_ audio: MLXArray, length: Int = AudioConstants.nSamples) -> MLXArray {
        let currentLength = audio.shape[0]

        if currentLength == length {
            return audio
        } else if currentLength > length {
            return audio[0..<length]
        } else {
            let padding = MLXArray.zeros([length - currentLength])
            return MLX.concatenated([audio, padding], axis: 0)
        }
    }
}
