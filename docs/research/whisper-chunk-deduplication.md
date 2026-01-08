# Whisper STT Chunk Text Deduplication Research

> Research Date: 2025-01-08

## Executive Summary

There are **3 main approaches** used in production Whisper implementations for handling chunk overlap deduplication:

| Approach | Used By | Accuracy | Complexity | Best For |
|----------|---------|----------|------------|----------|
| **Longest Common Sequence (LCS)** | HuggingFace Transformers | High | Medium | Batched processing |
| **Levenshtein Distance** | Whispy | High | Medium | Real-time streaming |
| **Word Timestamp Alignment** | WhisperX, our SDK | Highest | Low | When timestamps available |

---

## 1. Longest Common Sequence (HuggingFace)

The HuggingFace Transformers pipeline uses token-level LCS matching.

**Algorithm:**
- Compare end of previous chunk with beginning of new chunk
- Find longest matching token sequence
- Extend combined output only from non-overlapping portion

**Known Limitations:**
- Fails with silence in overlap (no overlapping tokens)
- Fails with repeated words (overlap may be too large)
- Case sensitivity issues: "sir" vs "Sir" are different tokens

**Optimal Settings:** `stride = chunk_length_s / 6` works well

**Source:** [HuggingFace Transformers ASR Pipeline](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py)

---

## 2. Levenshtein Distance (Whispy)

Whispy uses edit distance for real-time streaming.

**Algorithm:**
1. Generate token sequence from new chunk
2. Working backwards from last token, compute edit distance between all subsequences and previous chunk
3. Select subsequence with **minimum edit distance**
4. Use that as the overlap boundary

**Advantages:**
- ~1ms computation time
- Handles mid-word/mid-sentence boundaries
- Context-aware output

**Source:** [Whispy: Adapting STT to Real-Time (arXiv)](https://arxiv.org/html/2405.03484v1)

---

## 3. Timestamp-Based Alignment (WhisperX, Our SDK)

WhisperX and similar implementations use word timestamps.

**Algorithm:**
- Filter words where `word.start >= overlapEnd`
- Or where `word.start >= overlapStart && word.end > overlapEnd`

**Our current implementation** (in `LongAudioProcessor`):
```swift
// Simple word-based deduplication
let words = text.split(separator: " ")
// Match suffix of previous chunk to prefix of current
// Remove matching prefix from current chunk
```

**Source:** [WhisperX Paper (Oxford)](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf)

---

## 4. VAD-Based Segmentation (Faster-Whisper)

Faster-Whisper avoids deduplication entirely by using **VAD to create non-overlapping segments**:

- SileroVAD detects speech boundaries
- Chunks are cut at natural speech pauses
- No overlap = no deduplication needed
- Adds 100ms padding around voiced regions

**Benefit:** Reduces repetition errors by avoiding timestamp token reliance

**Source:** [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)

---

## Recommendations for MLX Audio Swift

**Current approach** (word-based matching) is valid but could be enhanced:

1. **Add fallback to LCS** when word timestamps are unavailable:
   ```swift
   if let words = result.words, !words.isEmpty {
       // Use timestamp-based filtering (current approach)
   } else {
       // Fall back to LCS token matching
   }
   ```

2. **Consider Levenshtein** for streaming mode where latency matters (~1ms)

3. **VAD integration** would eliminate deduplication need entirely (already have SileroVAD)

---

## Implementation Notes

### Current MLX Audio Swift Implementation

Located in `LongAudioProcessor.swift`:

```swift
private func deduplicateOverlapText(_ text: String, previousEndWords: [String]) -> String {
    let words = text.split(separator: " ").map(String.init)
    guard !words.isEmpty else { return text }

    var matchLength = 0
    for len in 1...min(previousEndWords.count, words.count) {
        let prevSuffix = previousEndWords.suffix(len)
        let currPrefix = words.prefix(len)

        if prevSuffix.elementsEqual(currPrefix, by: { $0.lowercased() == $1.lowercased() }) {
            matchLength = len
        }
    }

    if matchLength > 0 {
        return words.dropFirst(matchLength).joined(separator: " ")
    }
    return text
}
```

### Key Config

- `MergeConfig.deduplicateOverlap` (default: `true`) - enables/disables deduplication
- `SlidingWindowConfig.overlapDuration` (default: 5.0s) - overlap between chunks

---

## Sources

- [Whisper Long-Form Transcription (Medium)](https://medium.com/@yoad/whisper-long-form-transcription-1924c94a9b86)
- [Whispy: Adapting STT to Real-Time (arXiv)](https://arxiv.org/html/2405.03484v1)
- [HuggingFace Transformers ASR Pipeline](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py)
- [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX Paper (Oxford)](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf)
- [whisper_streaming (UFAL)](https://github.com/ufal/whisper_streaming)
- [HuggingFace Whisper Chunking PR #20104](https://github.com/huggingface/transformers/pull/20104)
