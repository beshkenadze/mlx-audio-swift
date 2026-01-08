# STT Benchmarking Guide

Research compiled for MLX Audio Swift STT benchmarking and evaluation.

## Benchmark Datasets

| Dataset | Description | Size | Best For |
|---------|-------------|------|----------|
| **LibriSpeech** | Read English speech, 16kHz | ~1000h train, ~10h test | Standard accuracy benchmark |
| **Common Voice** | Crowdsourced recordings | 3,209h English, 86k+ voices | Diverse accents, real-world |
| **CHiME-5** | Noisy/reverberant recordings | ~50h | Challenging acoustic conditions |

### LibriSpeech (Recommended)

Industry standard - MLPerf uses it for Whisper benchmarks.

- **test-clean**: ~5h, clear audio, easier
- **test-other**: ~5h, harder acoustic conditions

```bash
# Download test-clean (346MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz

# Download test-other (328MB)
wget https://www.openslr.org/resources/12/test-other.tar.gz
tar -xzf test-other.tar.gz
```

## Key Metrics

### Word Error Rate (WER)

Primary accuracy metric: `WER = (S + D + I) / N`

- **S**: Substitutions
- **D**: Deletions
- **I**: Insertions
- **N**: Total words in reference

**Target WER** (Whisper large-v3 on LibriSpeech):
- test-clean: ~2-3%
- test-other: ~5-6%

### Real-Time Factor (RTF)

`RTF = processing_time / audio_duration`

- RTF < 1.0 = faster than real-time
- RTF ~0.20 = 5x faster than real-time (current MLX Audio Swift)

### Other Metrics

- **Latency**: Time to first token (streaming)
- **Memory**: Peak GPU memory usage
- **Throughput**: Samples per second

## Apple Silicon Whisper Implementations

| Implementation | Speed | Language | Notes |
|----------------|-------|----------|-------|
| [WhisperKit](https://github.com/argmaxinc/WhisperKit) | Reference | Swift | Core ML, ANE, production-ready |
| [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx) | 10x vs cpp | Python | Fastest MLX implementation |
| [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | 30-40% > cpp | Python | Official MLX example |
| whisper.cpp | Baseline | C++ | ANE support, widely used |

## Benchmarking Tools

### 1. mac-whisper-speedtest

Compares 8 Whisper implementations on Apple Silicon.

```bash
git clone https://github.com/anvanvan/mac-whisper-speedtest
cd mac-whisper-speedtest
# Follow setup instructions
```

### 2. Picovoice STT Benchmark

Framework for systematic STT evaluation.

```bash
git clone https://github.com/Picovoice/speech-to-text-benchmark
```

### 3. WhisperKit Benchmarks

Reference benchmarks on real Apple devices:
https://huggingface.co/argmaxinc/whisperkit-coreml

## WER Calculation

### Using jiwer (Python)

```python
from jiwer import wer

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumped over the lazy dog"

error_rate = wer(reference, hypothesis)
print(f"WER: {error_rate:.2%}")
```

### Text Normalization

Important for fair comparison:
- Lowercase
- Remove punctuation
- Normalize numbers ("5" → "five")
- Handle contractions

Whisper's official normalizer: `whisper.normalizers.EnglishTextNormalizer`

## Benchmarking Script Template

```swift
// Pseudocode for MLX Audio Swift benchmarking
import Foundation
import MLXAudioSTT

struct BenchmarkResult {
    let audioFile: String
    let audioDuration: Double
    let processingTime: Double
    let transcription: String
    let rtf: Double
}

func benchmark(session: WhisperSession, audioFiles: [URL]) async throws -> [BenchmarkResult] {
    var results: [BenchmarkResult] = []

    for file in audioFiles {
        let audio = try loadAudio(from: file)
        let duration = Double(audio.shape[0]) / 16000.0

        let start = CFAbsoluteTimeGetCurrent()
        let transcription = try await session.transcribe(audio, sampleRate: 16000)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        results.append(BenchmarkResult(
            audioFile: file.lastPathComponent,
            audioDuration: duration,
            processingTime: elapsed,
            transcription: transcription,
            rtf: elapsed / duration
        ))
    }

    return results
}
```

## Current MLX Audio Swift Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model | large-v3-turbo | 809M params |
| RTF | ~0.20 | 11.3s audio → 2.2s |
| Device | Apple Silicon | M-series |

## References

- [LibriSpeech ASR Corpus](https://www.openslr.org/12)
- [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/open-asr-leaderboard/leaderboard)
- [MLPerf Whisper Benchmark](https://mlcommons.org/2025/09/whisper-inferencev5-1/)
- [WhisperKit GitHub](https://github.com/argmaxinc/WhisperKit)
- [Ionio Edge STT Benchmark 2025](https://www.ionio.ai/blog/2025-edge-speech-to-text-model-benchmark-whisper-vs-competitors)
- [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)
