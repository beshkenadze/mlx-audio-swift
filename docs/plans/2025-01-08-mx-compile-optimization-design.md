# mx.compile() Optimization Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `mx.compile()` to encoder and decoder with warmup at init time for 2-5x inference speedup.

**Architecture:** Compile both encoder and decoder at WhisperSession initialization, preallocate KVCache with fixed shapes to avoid recompilation, warmup with dummy inputs before first transcription.

**Tech Stack:** MLX-Swift compile API, preallocated tensors

---

## Background

From GPU_OPTIMIZATION_RESEARCH.md, `mx.compile()` fuses operations and caches execution plans:
- **Impact:** 2-5x speedup on repeated calls (decoder runs 100-400x per transcription)
- **Requirement:** Input shapes must stay fixed to avoid recompilation

### Current Performance Gap

| Implementation | 5s Audio (total) | Transcription Only | RTF |
|----------------|------------------|-------------------|-----|
| whisper.cpp | 1.94s | 1.58s | 0.32 |
| Our MLX (before) | 15s | ~3s | 0.60 |
| Our MLX (after int4) | ~5s | ~3s | 0.60 |

This optimization targets the **transcription time** (decoder loop), not loading.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to compile | Both encoder and decoder | Encoder has large graph, decoder runs 100-400x |
| When to compile | Init time with warmup | Zero latency on first transcription |
| Input shapes | Fixed (pad to 30s, max 448 tokens) | Avoid recompilation |
| KVCache strategy | Preallocated + offset | Concat changes shapes → recompilation |
| Cache in compile | External management (Option A) | Simpler, main gain is attention/MLP fusion |

---

## Architecture

```
WhisperSession.fromPretrained()
  ├── Load model weights (existing)
  ├── Create compiled functions (NEW)
  │     ├── compiledEncode: (MLXArray) -> MLXArray
  │     └── compiledDecode: (MLXArray, MLXArray) -> (MLXArray, [MLXArray])
  ├── Warmup with dummy inputs (NEW)
  │     ├── compiledEncode(zeros[1, 80, 3000])
  │     └── compiledDecode(zeros[1, 4], dummyEncoderOut)
  │     └── compiledDecode(zeros[1, 1], dummyEncoderOut)  // Single-token hot path
  └── eval() to force compilation

WhisperSession.transcribe()
  ├── compiledEncode(mel)           ← Uses cached compiled graph
  └── decoder loop (100-400x)
        └── compiledDecode(...)      ← Uses cached compiled graph
```

---

## Implementation Details

### 1. Preallocated KVCache

**Problem:** Current KVCache grows via `concatenated()`, changing shapes each step → forces recompilation every token.

**Solution:** Preallocate fixed-size cache, use offset-based slicing.

```swift
/// Preallocated Key-Value cache for compile-friendly incremental decoding
public class KVCache {
    private var keys: MLXArray      // [batch, maxSeq, dim] - preallocated
    private var values: MLXArray    // [batch, maxSeq, dim] - preallocated
    private var offset: Int = 0     // Current position in cache

    public let maxSequenceLength: Int

    public init(batchSize: Int = 1, maxSequenceLength: Int = 448, dim: Int) {
        self.maxSequenceLength = maxSequenceLength
        // Preallocate to fixed shape - no shape changes during decoding
        self.keys = MLXArray.zeros([batchSize, maxSequenceLength, dim])
        self.values = MLXArray.zeros([batchSize, maxSequenceLength, dim])
    }

    /// Update cache using slice assignment (fixed shape, no concat)
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let newSeqLen = newKeys.shape[1]
        let end = offset + newSeqLen

        // Slice assignment - shape stays fixed
        keys[0..., offset..<end, 0...] = newKeys
        values[0..., offset..<end, 0...] = newValues
        offset = end

        // Return only the valid portion
        return (keys[0..., 0..<offset, 0...], values[0..., 0..<offset, 0...])
    }

    public func reset() {
        offset = 0
        // No need to zero - will be overwritten
    }
}
```

**Why 448?** Whisper's max token length is 448 tokens (nTextCtx). This is a known fixed upper bound.

### 2. Compiled Functions in WhisperSession

```swift
public final class WhisperSession: @unchecked Sendable {
    // Existing
    private let encoder: AudioEncoder
    private let decoder: TextDecoder
    private let config: WhisperConfiguration

    // NEW: Compiled closures (created once at init)
    private let compiledEncode: @Sendable ([MLXArray]) -> [MLXArray]
    private let compiledDecode: @Sendable ([MLXArray]) -> [MLXArray]

    private init(...) {
        // ... existing init code ...

        // Compile encoder: [mel] -> [encoderOutput]
        self.compiledEncode = compile { inputs in
            let mel = inputs[0]
            return [self.encoder(mel)]
        }

        // Compile decoder: [tokens, encoderOutput] -> [logits, crossQK...]
        // Cache managed externally (Option A - simpler)
        self.compiledDecode = compile { inputs in
            let tokens = inputs[0]
            let encoderOutput = inputs[1]
            let (logits, crossQK) = self.decoder(
                tokens: tokens,
                encoderOutput: encoderOutput,
                kvCache: self.kvCaches
            )
            return [logits] + crossQK
        }

        // Warmup to populate compilation cache
        warmup()
    }
}
```

### 3. Warmup Function

```swift
private func warmup() {
    // Warmup encoder with standard 30s mel shape
    let dummyMel = MLXArray.zeros([1, config.nMels, 3000])
    let encoderOut = compiledEncode([dummyMel])[0]

    // Warmup decoder with typical input shapes
    let dummyTokens = MLXArray.zeros([1, 4], dtype: .int32)  // Initial prompt tokens
    _ = compiledDecode([dummyTokens, encoderOut])

    // Also warmup single-token decode (the hot path in loop)
    let singleToken = MLXArray.zeros([1, 1], dtype: .int32)
    _ = compiledDecode([singleToken, encoderOut])

    eval()  // Force compilation graphs to execute
}
```

### 4. Updated Transcribe Flow

```swift
public func transcribe(...) -> AsyncThrowingStream<StreamingResult, Error> {
    AsyncThrowingStream { continuation in
        Task {
            // 1. Encode audio (COMPILED)
            let mel = MelSpectrogram.compute(audio: paddedAudio, nMels: config.nMels)
            let melBatched = mel.expandedDimensions(axis: 0)
            let encoderOutput = compiledEncode([melBatched])[0]

            // 2. Reset caches (preallocated, fixed shape)
            for cache in kvCaches {
                cache.reset()
            }

            // 3. Decode loop
            var tokens = tokenizer.initialTokens(...)

            for step in 0..<maxTokens {
                let tokenArray: MLXArray
                if step == 0 {
                    tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
                } else {
                    tokenArray = MLXArray([tokens.last!]).expandedDimensions(axis: 0)
                }

                // COMPILED decode
                let outputs = compiledDecode([tokenArray, encoderOutput])
                let logits = outputs[0]
                let crossQK = Array(outputs.dropFirst())

                let nextToken = argmax(logits[0, -1])
                tokens.append(nextToken)

                // ... streaming logic ...
            }
        }
    }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `MultiHeadAttention.swift` | Update `KVCache` to preallocated + offset pattern |
| `WhisperSession.swift` | Add `compiledEncode`, `compiledDecode`, `warmup()`, `kvCaches` property |
| `TextDecoder.swift` | No changes needed |
| `AudioEncoder.swift` | No changes needed |

### New Test File

| File | Purpose |
|------|---------|
| `WhisperCompileTests.swift` | Verify compilation works, no accuracy regression |

---

## Expected Results

### Performance Gains

| Component | Calls per Transcription | Expected Speedup |
|-----------|------------------------|------------------|
| Encoder | 1 | Modest (graph fusion) |
| Decoder | 100-400 | **2-5x** (main benefit) |

### Before vs After

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Decoder per-token | ~7-10ms | ~2-4ms |
| Total transcription (5s audio) | ~3s | ~1-1.5s |
| RTF | 0.60 | ~0.25-0.30 |

### Warmup Cost

- Added to init: ~1-2s (one-time)
- Hidden by background loading if using `ModelLoadingOptions.fast`

---

## Testing Strategy

### Unit Tests

```swift
func testCompiledEncoderMatchesUncompiled() async throws {
    let session = try await WhisperSession.fromPretrained(.tiny)
    let audio = loadTestAudio("hello.wav")

    // Results should be identical
    let compiled = session.transcribe(audio)  // Uses compiled path
    // Compare with known reference output
}

func testKVCachePreallocatedNoShapeChange() {
    let cache = KVCache(maxSequenceLength: 448, dim: 512)

    // Verify shape stays fixed across updates
    let initialShape = cache.keys.shape
    cache.update(keys: randomKeys, values: randomValues)
    XCTAssertEqual(cache.keys.shape, initialShape)
}

func testWarmupCompletesWithoutError() async throws {
    // Warmup should not throw
    let session = try await WhisperSession.fromPretrained(.tiny)
    XCTAssertTrue(session.isReady)
}
```

### Integration Tests

```swift
func testTranscriptionAccuracyWithCompile() async throws {
    let session = try await WhisperSession.fromPretrained(.tiny)
    let result = try await session.transcribe(testAudio)

    // WER should be same as before compile optimization
    let wer = calculateWER(reference: expectedText, hypothesis: result)
    XCTAssertLessThan(wer, 0.05)
}
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compile breaks with MLX updates | Pin MLX-Swift version, test on updates |
| Accuracy regression | WER tests compare compiled vs uncompiled |
| Memory increase from preallocation | 448 tokens × 512 dim × 2 (k,v) × layers ≈ small |
| Warmup slows cold start | Hidden by background loading |

---

## Future Optimizations (Out of Scope)

After this optimization, remaining opportunities from GPU_OPTIMIZATION_RESEARCH.md:

1. **`mx.fast.scaled_dot_product_attention`** - Fused attention kernel
2. **Batched decoding** - Decode multiple tokens in parallel (10x from Lightning Whisper)
3. **`mx.async_eval()`** - Pipeline graph construction with GPU execution
4. **Core ML / ANE encoder offload** - 3x encoder speedup

---

## Success Criteria

- [ ] Decoder loop 2x+ faster (measured with profiling)
- [ ] No accuracy regression (WER within 1% of baseline)
- [ ] Warmup completes in <2s
- [ ] All existing tests pass
