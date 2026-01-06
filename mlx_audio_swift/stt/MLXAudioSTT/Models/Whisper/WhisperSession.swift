import Foundation
import MLX

public final class WhisperSession: @unchecked Sendable {
    private let modelType: WhisperModel
    private let streamingConfig: StreamingConfig
    private var currentTask: Task<Void, Never>?
    private let taskLock = NSLock()

    private init(modelType: WhisperModel, streamingConfig: StreamingConfig) {
        self.modelType = modelType
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

        return WhisperSession(modelType: model, streamingConfig: streaming)
    }

    // MARK: - Transcription (Streaming)

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) -> AsyncThrowingStream<StreamingResult, Error> {
        AsyncThrowingStream { continuation in
            self.taskLock.withLock {
                self.currentTask = Task {
                    do {
                        guard sampleRate == AudioConstants.sampleRate else {
                            throw WhisperError.sampleRateMismatch(
                                expected: AudioConstants.sampleRate,
                                got: sampleRate
                            )
                        }

                        // TODO: Implement actual streaming transcription loop
                        // 1. Compute mel spectrogram
                        // 2. Encode audio
                        // 3. Decode with AlignAtt streaming
                        // 4. Yield StreamingResult for each stable token

                        try Task.checkCancellation()

                        continuation.yield(StreamingResult(
                            text: "[Transcription not yet implemented]",
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
    }

    // MARK: - Transcription (Blocking)

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) async throws -> String {
        var result = ""
        for try await partial in transcribe(audio, sampleRate: sampleRate, options: options) {
            if partial.isFinal {
                result = partial.text
            }
        }
        return result
    }

    // MARK: - Control

    public func cancel() {
        taskLock.withLock {
            currentTask?.cancel()
            currentTask = nil
        }
    }
}

// MARK: - STTSession Conformance

extension WhisperSession: STTSession {
    public func transcribe(_ audio: MLXArray, sampleRate: Int) -> AsyncThrowingStream<StreamingResult, Error> {
        transcribe(audio, sampleRate: sampleRate, options: .default)
    }

    public func transcribe(_ audio: MLXArray, sampleRate: Int) async throws -> String {
        try await transcribe(audio, sampleRate: sampleRate, options: .default)
    }
}
