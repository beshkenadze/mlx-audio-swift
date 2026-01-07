# Research Report: Long Audio Chunking Strategies for Whisper STT

> **Date**: 2025-01-07
> **Topic**: Implementation strategies for handling audio >30 seconds in Whisper-based STT
> **Confidence**: High (multiple peer-reviewed sources, production implementations)

## Executive Summary

This research covers three main approaches for handling long-form audio transcription with Whisper models:

1. **Sequential (OpenAI-style)** - Seek-based window advancement with context conditioning
2. **VAD-based (WhisperX-style)** - Voice activity detection with parallel batched inference
3. **Sliding Window (whisper.cpp-style)** - Fixed overlap with chunk merging

Each approach has distinct trade-offs between accuracy, latency, and complexity.

---

## Strategy 1: Sequential Long-Form Decoding

### Key Papers & Sources

| Source | Description |
|--------|-------------|
| [Whisper Paper (arXiv:2212.04356)](https://cdn.openai.com/papers/whisper.pdf) | Original OpenAI paper, Section 4.5 covers long-form transcription |
| [HuggingFace Whisper Docs](https://huggingface.co/docs/transformers/model_doc/whisper) | Sequential vs chunked algorithm comparison |
| [Distil-Whisper](https://github.com/huggingface/distil-whisper) | Optimized for sequential algorithm |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 implementation with sequential support |

### Algorithm Details

```
┌─────────────────────────────────────────────────────────────────┐
│                 SEQUENTIAL DECODING ALGORITHM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  seek = 0                                                       │
│  while seek < content_frames:                                   │
│      segment = mel[seek : seek + N_FRAMES]                      │
│      result = decode(segment, previous_tokens)                  │
│                                                                 │
│      # Advance based on timestamp tokens                        │
│      seek = last_timestamp_position                             │
│                                                                 │
│      # Condition next window on previous text                   │
│      if condition_on_previous_text:                             │
│          previous_tokens = result.tokens[-max_context:]         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `condition_on_previous_text` | `True` | Use previous transcription as context |
| `temperature` | `0.0` | Sampling temperature (tuple for fallback) |
| `compression_ratio_threshold` | `2.4` | Skip if compression ratio exceeds |
| `logprob_threshold` | `-1.0` | Skip if avg logprob below threshold |
| `no_speech_threshold` | `0.6` | Skip if no_speech probability above |

### Performance Characteristics

- **Accuracy**: Up to 0.5% WER better than chunked
- **Speed**: Sequential, cannot parallelize
- **Latency**: Must wait for each segment
- **Memory**: Low (single segment at a time)

### Swift Implementation Considerations

```swift
struct SequentialConfig: Sendable {
    var conditionOnPreviousText: Bool = true
    var contextResetTemperature: Float = 0.5
    var compressionRatioThreshold: Float = 2.4
    var logprobThreshold: Float = -1.0
    var noSpeechThreshold: Float = 0.6
    var initialPrompt: String? = nil
}
```

---

## Strategy 2: VAD-Based Chunking (WhisperX)

### Key Papers & Sources

| Source | Description |
|--------|-------------|
| [WhisperX Paper (arXiv:2303.00747)](https://arxiv.org/abs/2303.00747) | Original WhisperX paper |
| [WhisperX PDF](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf) | Full paper with benchmarks |
| [WhisperX GitHub](https://github.com/m-bain/whisperX) | Reference implementation |
| [pyannote.audio (arXiv:1911.01255)](https://arxiv.org/abs/1911.01255) | VAD model architecture |

### Algorithm Details

```
┌─────────────────────────────────────────────────────────────────┐
│                   VAD-BASED CHUNKING (WhisperX)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. VAD SEGMENTATION                                            │
│     speech_segments = VAD(audio)  # e.g., pyannote, Silero      │
│                                                                 │
│  2. CUT & MERGE                                                 │
│     chunks = []                                                 │
│     for segment in speech_segments:                             │
│         if segment.duration > 30s:                              │
│             # Min-cut at lowest energy points                   │
│             sub_segments = min_cut(segment, max_duration=30s)   │
│             chunks.extend(sub_segments)                         │
│         elif can_merge_with_previous(segment, chunks):          │
│             chunks[-1].merge(segment)  # Keep ~30s chunks       │
│         else:                                                   │
│             chunks.append(segment)                              │
│                                                                 │
│  3. PARALLEL TRANSCRIPTION (no previous text conditioning)      │
│     results = parallel_map(transcribe, chunks)                  │
│                                                                 │
│  4. FORCED ALIGNMENT (wav2vec2)                                 │
│     word_timestamps = forced_align(audio, results)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### VAD Options for Swift/iOS

| Library | Model | Size | Accuracy | Speed | License |
|---------|-------|------|----------|-------|---------|
| [FluidAudio](https://github.com/FluidInference/FluidAudio) | Silero VAD (CoreML) | ~2MB | High | ANE-optimized | MIT |
| [ios-vad](https://github.com/baochuquan/ios-vad) | WebRTC/Silero/Yamnet | 158KB-2MB | Medium-High | <1ms/chunk | Apache 2.0 |
| [RealTimeCutVADLibrary](https://github.com/helloooideeeeea/RealTimeCutVADLibrary) | Silero (ONNX) | ~2MB | High | Real-time | MIT |

### Silero VAD Technical Specs

- **Architecture**: CNN + LSTM (2 layers, 64 units each)
- **Input**: 16-bit mono PCM, 8-16kHz
- **Chunk size**: 30ms minimum (256ms recommended for batching)
- **Output**: Speech probability (0.0-1.0)
- **Threshold**: v4=0.5, v5=0.5 (with different onset/offset heuristics)

### Performance Characteristics

- **Accuracy**: Comparable to sequential (reduced hallucinations)
- **Speed**: 12x faster with batching
- **Latency**: Depends on VAD + chunk processing
- **Memory**: Higher (parallel chunks in memory)

### Swift Implementation Considerations

```swift
struct VADConfig: Sendable {
    var minSpeechDuration: TimeInterval = 0.5
    var minSilenceDuration: TimeInterval = 0.3
    var speechThreshold: Float = 0.5
    var targetChunkDuration: TimeInterval = 30.0
    var parallelProcessing: Bool = true
    var maxConcurrency: Int = 4

    enum VADModel: Sendable {
        case silero      // Recommended: High accuracy
        case webrtc      // Fallback: Lightweight
        case energy      // Simplest: RMS-based
    }
    var model: VADModel = .silero
}
```

---

## Strategy 3: Sliding Window with Overlap

### Key Papers & Sources

| Source | Description |
|--------|-------------|
| [ChunkFormer (arXiv:2502.14673)](https://arxiv.org/abs/2502.14673) | ICASSP 2025, masked chunking for long-form |
| [whisper.cpp stream](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp) | Reference C++ implementation |
| [HuggingFace ASR Chunking](https://huggingface.co/blog/asr-chunking) | Wav2Vec2 chunking with stride |
| [whisper.cpp Discussion #206](https://github.com/ggml-org/whisper.cpp/discussions/206) | Chunking implementation details |

### Algorithm Details

```
┌─────────────────────────────────────────────────────────────────┐
│                   SLIDING WINDOW WITH OVERLAP                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  window_duration = 30s                                          │
│  overlap_duration = 5s                                          │
│  hop_duration = window_duration - overlap_duration  # 25s       │
│                                                                 │
│  position = 0                                                   │
│  results = []                                                   │
│                                                                 │
│  while position < audio_duration:                               │
│      chunk = audio[position : position + window_duration]       │
│      result = transcribe(chunk)                                 │
│      results.append(result)                                     │
│      position += hop_duration                                   │
│                                                                 │
│  # Merge overlapping results                                    │
│  final = merge_with_deduplication(results, overlap_duration)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Window duration | 30s | Whisper training max |
| Overlap/Stride | chunk_length / 6 ≈ 5s | HuggingFace empirical |
| Merge strategy | Word-level timestamp alignment | whisper.cpp |

### Merge Strategies

1. **Simple deduplication**: Remove repeated words in overlap region
2. **Timestamp-based**: Use word timestamps to align overlap
3. **Confidence-weighted**: Prefer higher-confidence transcriptions
4. **LCS (Longest Common Subsequence)**: Find matching sequences

### Performance Characteristics

- **Accuracy**: Slightly lower than sequential (boundary effects)
- **Speed**: Can parallelize chunks
- **Latency**: Predictable (fixed window size)
- **Memory**: Moderate (overlap buffer)

### Swift Implementation Considerations

```swift
struct SlidingWindowConfig: Sendable {
    var windowDuration: TimeInterval = 30.0
    var overlapDuration: TimeInterval = 5.0
    var hopDuration: TimeInterval { windowDuration - overlapDuration }

    enum MergeStrategy: Sendable {
        case deduplication      // Simple word dedup
        case timestampAlignment // Use word timestamps
        case confidenceWeighted // Prefer higher confidence
        case lcs                // Longest common subsequence
    }
    var mergeStrategy: MergeStrategy = .timestampAlignment
}
```

---

## Comparison Matrix

| Aspect | Sequential | VAD-Based | Sliding Window |
|--------|-----------|-----------|----------------|
| **WER** | Best (baseline) | ~Same (-hallucinations) | Slightly worse |
| **Speed** | 1x | 12x (batched) | 2-4x (parallel) |
| **Latency to first** | High | Medium | Low (predictable) |
| **Complexity** | Low | High (VAD model) | Medium |
| **Memory** | Low | High | Medium |
| **Real-time friendly** | No | Yes (with VAD) | Yes |
| **Context continuity** | Yes | No | Partial (overlap) |

---

## Implementation Recommendations for Swift

### Phase 1: Sliding Window (Simplest)
Start with fixed sliding window - no additional dependencies, predictable behavior.

### Phase 2: Sequential (Most Accurate)
Add OpenAI-style sequential for accuracy-critical use cases.

### Phase 3: VAD-Based (Most Flexible)
Integrate FluidAudio or ios-vad for production-quality VAD chunking.

### Recommended Swift Dependencies

```swift
// Package.swift
dependencies: [
    // Core ML inference
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),

    // VAD (choose one)
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.7.9"),
    // OR
    .package(url: "https://github.com/baochuquan/ios-vad.git", from: "1.0.0"),
]
```

---

## Sources

### Papers
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) - Radford et al., 2022
- [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://arxiv.org/abs/2303.00747) - Bain et al., 2023
- [ChunkFormer: Masked Chunking Conformer](https://arxiv.org/abs/2502.14673) - Le et al., 2025
- [pyannote.audio: Neural Building Blocks for Speaker Diarization](https://arxiv.org/abs/1911.01255) - Bredin et al., 2019

### Implementations
- [OpenAI Whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [FluidAudio](https://github.com/FluidInference/FluidAudio)
- [ios-vad](https://github.com/baochuquan/ios-vad)

### Documentation
- [HuggingFace Whisper Documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- [HuggingFace ASR Chunking Blog](https://huggingface.co/blog/asr-chunking)
- [Apple Speech Framework](https://developer.apple.com/documentation/speech)
