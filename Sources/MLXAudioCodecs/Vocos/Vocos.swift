//
//  Vocos.swift
//  MLXAudioCodecs
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - AdaLayerNorm

/// Adaptive Layer Normalization for conditional generation.
///
/// Learns scale and shift parameters conditioned on an embedding ID.
public class AdaLayerNorm: Module {
    let eps: Float
    let dim: Int
    @ModuleInfo(key: "scale") var scale: Linear
    @ModuleInfo(key: "shift") var shift: Linear

    public init(numEmbeddings: Int, embeddingDim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.dim = embeddingDim

        self._scale.wrappedValue = Linear(numEmbeddings, embeddingDim)
        self._shift.wrappedValue = Linear(numEmbeddings, embeddingDim)
    }

    public func callAsFunction(_ x: MLXArray, condEmbedding: MLXArray) -> MLXArray {
        let scaleVal = scale(condEmbedding)
        let shiftVal = shift(condEmbedding)

        // Manual layer norm without learnable parameters
        // Compute mean and variance along last axis
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)

        // Apply adaptive scale and shift: x * scale[:, None, :] + shift[:, None, :]
        let scaleBroadcast = scaleVal.expandedDimensions(axis: 1)
        let shiftBroadcast = shiftVal.expandedDimensions(axis: 1)

        return normalized * scaleBroadcast + shiftBroadcast
    }
}

// MARK: - ISTFTHead

/// ISTFT Head for converting decoder output to audio waveforms.
///
/// Predicts magnitude and phase from backbone output, then uses ISTFT to reconstruct audio.
public class ISTFTHead: Module {
    let nFft: Int
    let hopLength: Int
    @ModuleInfo(key: "out") var out: Linear

    public init(dim: Int, nFft: Int, hopLength: Int, padding: String = "center") {
        self.nFft = nFft
        self.hopLength = hopLength
        // Output n_fft + 2 for magnitude and phase (n_fft/2 + 1 each)
        self._out.wrappedValue = Linear(dim, nFft + 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Project to STFT coefficients: (B, L, C) -> (B, L, n_fft+2)
        var h = out(x)

        // Transpose: (B, L, n_fft+2) -> (B, n_fft+2, L)
        h = h.swappedAxes(1, 2)

        // Split into magnitude and phase
        let halfSize = (nFft + 2) / 2
        let mag = exp(h[0..., 0..<halfSize, 0...])
        let clippedMag = clip(mag, max: MLXArray(Float(1e2)))
        let phase = h[0..., halfSize..., 0...]

        // Construct complex STFT: S = mag * e^(i * phase)
        let cosPhase = cos(phase)
        let sinPhase = sin(phase)

        // For ISTFT, create complex representation
        let stftReal = clippedMag * cosPhase
        let stftImag = clippedMag * sinPhase

        // Perform ISTFT
        let audio = performISTFT(real: stftReal, imag: stftImag)

        return audio
    }

    /// Perform inverse STFT using overlap-add synthesis.
    private func performISTFT(real: MLXArray, imag: MLXArray) -> MLXArray {
        // real/imag shape: (B, n_fft/2+1, L)
        let batchSize = real.shape[0]
        let numFrames = real.shape[2]

        // Create window (matches Python hanning function)
        let window = hanningWindow(length: nFft)

        // Output length: t = (num_frames - 1) * hop_length + win_length
        let outputLength = (numFrames - 1) * hopLength + nFft

        var outputs: [MLXArray] = []

        for b in 0..<batchSize {
            // Get single batch: (n_fft/2+1, L)
            let realB = real[b]
            let imagB = imag[b]

            // Create complex STFT
            let complexSpec = realB + MLXArray(real: Float(0), imaginary: Float(1)) * imagB

            // Perform IRFFT along axis 0 (frequency axis)
            let framesFreq = MLXFFT.irfft(complexSpec, axis: 0)

            // Transpose to get (num_frames, n_fft)
            let framesTime = framesFreq.transposed(1, 0)

            // Apply window
            let windowedFrames = framesTime * window

            // Overlap-add synthesis
            var audioSamples = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            let windowArray = window.asArray(Float.self)

            for i in 0..<numFrames {
                let start = i * hopLength
                let frameData = windowedFrames[i].asArray(Float.self)

                for j in 0..<min(nFft, frameData.count) {
                    if start + j < outputLength {
                        audioSamples[start + j] += frameData[j]
                        windowSum[start + j] += windowArray[j]
                    }
                }
            }

            // Normalize by window sum
            for i in 0..<outputLength {
                if windowSum[i] != 0 {
                    audioSamples[i] /= windowSum[i]
                }
            }

            // Trim center padding
            let trimStart = nFft / 2
            let trimEnd = outputLength - nFft / 2
            let trimmedAudio: [Float]
            if trimEnd > trimStart {
                trimmedAudio = Array(audioSamples[trimStart..<trimEnd])
            } else {
                trimmedAudio = audioSamples
            }

            outputs.append(MLXArray(trimmedAudio))
        }

        // Stack outputs
        if outputs.count == 1 {
            return outputs[0]
        }
        return MLX.stacked(outputs, axis: 0)
    }

    /// Generate Hanning window
    private func hanningWindow(length: Int) -> MLXArray {
        if length == 1 {
            return MLXArray([Float(1.0)])
        }
        let n = Array(stride(from: 0, to: length, by: 1)).map { Float($0) }
        let factor = Float.pi / Float(length - 1)
        let window = n.map { 0.5 - 0.5 * cos(2.0 * factor * $0) }
        return MLXArray(window)
    }
}

// MARK: - Feature Extractor Protocol

/// Protocol for feature extractors used by Vocos.
public protocol FeatureExtractor {
    func callAsFunction(_ audio: MLXArray, bandwidthId: Int?) -> MLXArray
}

// MARK: - Vocos

/// Vocos vocoder model for high-quality audio synthesis.
///
/// Combines a feature extractor, backbone, and ISTFT head for audio reconstruction.
public class Vocos: Module {
    @ModuleInfo(key: "backbone") var backbone: VocosBackbone
    @ModuleInfo(key: "head") var head: ISTFTHead

    public init(
        backbone: VocosBackbone,
        head: ISTFTHead
    ) {
        self._backbone.wrappedValue = backbone
        self._head.wrappedValue = head
    }

    /// Decode features to audio waveform.
    public func decode(_ features: MLXArray, bandwidthId: MLXArray? = nil) -> MLXArray {
        let x = backbone(features, bandwidthId: bandwidthId)
        let audioOutput = head(x)
        return audioOutput
    }

    /// Forward pass: extract features and decode to audio.
    public func callAsFunction(_ features: MLXArray, bandwidthId: MLXArray? = nil) -> MLXArray {
        return decode(features, bandwidthId: bandwidthId)
    }
}
