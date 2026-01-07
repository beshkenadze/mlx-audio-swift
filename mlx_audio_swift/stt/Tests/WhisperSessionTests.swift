import Testing
import MLX
@testable import MLXAudioSTT

struct WhisperSessionTests {
    @Test func transcribe_invalidSampleRate_throws() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([44100])

        await #expect(throws: WhisperError.self) {
            _ = try await session.transcribe(audio, sampleRate: 44100)
        }
    }

    @Test func cancel_stopsTranscription() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        // Use type annotation to get streaming version
        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(audio, sampleRate: AudioConstants.sampleRate)

        session.cancel()

        var wasCancelled = false
        do {
            for try await _ in stream { }
        } catch let error as WhisperError {
            if case .cancelled = error {
                wasCancelled = true
            }
        }

        #expect(wasCancelled)
    }

    @Test func fromPretrained_createsSession() async throws {
        let session = try await WhisperSession.fromPretrained(
            model: .largeTurbo,
            streaming: .default
        )

        // Session created successfully (type is non-optional)
        _ = session
    }

    @Test func transcribe_validSampleRate_streams() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        var results: [StreamingResult] = []
        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(audio, sampleRate: AudioConstants.sampleRate)
        for try await result in stream {
            results.append(result)
        }

        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)
    }

    @Test(.disabled("Requires network and model download"))
    func transcribe_silentAudio_returnsEmptyOrMinimal() async throws {
        // Given: A session with tiny model
        let session = try await WhisperSession.fromPretrained(model: .tiny)

        // Silent audio (1 second)
        let audio = MLXArray.zeros([AudioConstants.sampleRate])

        // When: Transcribing
        var results: [StreamingResult] = []
        for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
            results.append(result)
        }

        // Then: Should complete (may have minimal output for silence)
        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)
    }
}
