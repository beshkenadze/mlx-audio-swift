# Whisper Chunk Deduplication Techniques

## Summary

Three main approaches for handling chunk text overlap in Whisper STT:

### 1. Longest Common Sequence (LCS)
- Used by: HuggingFace Transformers
- Token-level matching between chunk boundaries
- Optimal stride: `chunk_length_s / 6`
- Limitations: fails with silence, repeated words, case sensitivity

### 2. Levenshtein Distance
- Used by: Whispy (real-time streaming)
- ~1ms computation time
- Works backwards from last token, finds minimum edit distance
- Good for mid-word/mid-sentence boundaries

### 3. Word Timestamp Alignment
- Used by: WhisperX, MLX Audio Swift
- Filter words by `word.start >= overlapEnd`
- Most accurate when timestamps available
- Current implementation in `LongAudioProcessor.deduplicateOverlapText()`

### 4. VAD-Based (Alternative)
- Used by: Faster-Whisper
- SileroVAD creates non-overlapping segments
- No deduplication needed
- 100ms padding around voiced regions

## MLX Audio Swift Implementation

Config: `MergeConfig.deduplicateOverlap` (default: true)

```swift
// In LongAudioProcessor.swift
private func deduplicateOverlapText(_ text: String, previousEndWords: [String]) -> String
```

Algorithm: Match suffix of previous chunk words to prefix of current chunk words (case-insensitive), remove matching prefix.

## Future Improvements

1. Add LCS fallback when word timestamps unavailable
2. Consider Levenshtein for streaming latency
3. VAD integration eliminates deduplication need (SileroVAD already available)
