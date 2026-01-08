# GPU Optimization Research for MLX Whisper

> Research Date: 2025-01-08 (Updated with mx.compile deep dive)

## Executive Summary

Our MLX Swift implementation is **~2x slower** than whisper.cpp for transcription. The main bottleneck is **model loading (~12s)** which dominates short audio processing.

### Benchmark Results (Apple M1 Max, large-v3-turbo)

| Implementation | 5s Audio (total) | Transcription Only | RTF | CPU/GPU |
|----------------|------------------|-------------------|-----|---------|
| **whisper.cpp** | 1.94s | 1.58s | **0.32** | 56% (Metal GPU) |
| **Our Swift (MLX)** | 15.15s | ~3s | **0.60** | 96% (mostly CPU) |

**Key insight:** Model loading (~12s) is the main issue, not transcription speed.

---

## Key Optimization Opportunities

### 1. Use `mx.compile()` for Graph Compilation

Currently, every forward pass builds a new compute graph. Compiling fuses kernels and caches the execution plan.

```swift
// Wrap encoder/decoder calls in compiled functions
let compiledEncoder = compile { mel in encoder(mel) }
let compiledDecoder = compile { tokens, encoderOut, cache in decoder(tokens, encoderOut, cache) }
```

**Impact:** 2-5x speedup for repeated calls (GELU benchmark: 15.5ms → 3.1ms)

**Source:** [MLX Compilation Docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html)

#### When to Use `compile()`

| Scenario | Use Compile? | Why |
|----------|-------------|-----|
| Decoder loop (many calls) | ✅ Yes | Compilation overhead amortized |
| Encoder (once per audio) | ⚠️ Maybe | Only 1 call, but large graph may benefit |
| Functions with fusible ops | ✅ Yes | Element-wise ops fuse dramatically |
| Single-use lambdas in loops | ❌ No | Recompiles every iteration! |
| Debugging (need to print) | ❌ No | Can't inspect placeholders |

#### MLX Swift API

```swift
public func compile(
    inputs: [any Updatable] = [],
    outputs: [any Updatable] = [],
    shapeless: Bool = false,
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> @Sendable ([MLXArray]) -> [MLXArray]

// Global toggle for debugging
compile(enable: false)  // Disable all compilation
```

#### Recompilation Triggers (Avoid These!)

Recompilation happens when:
- Input **shapes** change (default behavior)
- Input **types** change
- Number of inputs changes

```swift
// ❌ BAD: Different shapes each call → recompiles
for chunk in variableLengthChunks {
    compiledFn(chunk)  // Recompiles if chunk.shape differs!
}

// ✅ GOOD: Pad to consistent shape
let paddedChunk = padToFixedLength(chunk, maxLength: 3000)
compiledFn(paddedChunk)  // Cached
```

#### `shapeless: true` - Use Carefully

Prevents recompilation on shape changes, but breaks with shape-dependent ops:

```swift
// ❌ BREAKS with shapeless
func bad(x: MLXArray) -> MLXArray {
    return x.reshape([x.shape[0] * x.shape[1], -1])  // Hardcoded shape!
}

// ✅ WORKS with shapeless
func good(x: MLXArray) -> MLXArray {
    return x.flatten(start: 0, end: 1)  // Dynamic
}
```

#### State Management for Neural Networks

When compiling functions that read/modify model state:

```swift
// Capture state as inputs AND outputs
let state: [any Updatable] = [model.state, optimizer.state]

let compiledStep = compile(inputs: state, outputs: state) { arrays in
    // Training step that modifies model weights
}
```

**Critical for Dropout/Random:** Include random state if using stochastic layers:
```python
state = [model.state, optimizer.state, mx.random.state]
```

#### Closure Pitfall - Don't Capture Arrays

```swift
// ❌ BAD: Captured array bakes entire graph
let weights = loadWeights()
let compiled = compile { x in
    return x + weights  // weights frozen at trace time!
}

// ✅ GOOD: Pass as explicit input
let compiled = compile(inputs: [weights]) { x in
    return x + weights  // Reflects weight updates
}
```

#### Warmup Strategy (Pre-compile at App Startup)

Compilation cache is **in-memory only** - doesn't persist to disk. Use warmup:

```swift
func warmupCompilation() {
    let dummyMel = MLXArray.zeros([1, 80, 3000])
    let dummyTokens = MLXArray.zeros([1, 1])

    _ = compiledEncoder(dummyMel)
    _ = compiledDecoder(dummyTokens, dummyEncoderOut, dummyCache)
    eval()  // Force compilation before user requests
}
```

#### Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Printing inside compiled fn | Crash (placeholder has no data) | Use `compile(enable: false)` |
| Lambda in loop | Slow (recompiles each iteration) | Hoist compile outside loop |
| Variable input shapes | Slow (recompiles per shape) | Pad to fixed size or `shapeless: true` |
| Captured mutable state | Stale values | Use `inputs`/`outputs` parameters |
| Forgot random state | Dropout gives same values | Include random state |

### 2. Batched Decoding

Lightning Whisper MLX uses batched decoding with batch_size=12 for **10x speedup** over vanilla MLX Whisper.

```swift
// Decode multiple tokens in parallel instead of one at a time
let batchedTokens = decoder(tokenBatch, encoderOutput, kvCache)
```

**Source:** [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)

#### Recommended: User-Configurable Batch Size

| Model | Recommended batch_size | Memory Trade-off |
|-------|----------------------|------------------|
| tiny/base | 16-24 | Low memory usage |
| small/medium | 12 | Balanced |
| large/large-v3 | 6-8 | High memory usage |

```swift
public struct TranscriptionOptions {
    /// Batch size for parallel decoding (1 = sequential, higher = faster but more memory)
    public var batchSize: Int?

    public static let lowMemory = TranscriptionOptions(batchSize: 1)
    public static let balanced = TranscriptionOptions(batchSize: 12)
    public static let maxSpeed = TranscriptionOptions(batchSize: 24)
}
```

### 3. Use `mx.fast.*` Operations

MLX provides optimized kernels:
- `mx.fast.scaled_dot_product_attention` - fused attention
- `mx.fast.rms_norm` / `mx.fast.layer_norm` - fused normalization
- `mx.fast.rope` - rotary position embedding

**Impact:** 30-50% speedup on attention-heavy models

**Source:** [Writing Fast MLX](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)

### 4. Model Quantization

whisper.cpp uses GGML quantized weights (4-bit/8-bit). MLX supports similar:

```swift
// Load 4-bit quantized model
let model = try quantize(model, groupSize: 64, bits: 4)
```

**Impact:** 4x memory reduction, faster inference

**Source:** [MLX Whisper PyPI](https://pypi.org/project/mlx-whisper/)

### 5. Strategic `eval()` Placement

Don't call `eval()` too frequently. Batch operations:

```swift
// BAD: eval after each token
for step in 0..<maxTokens {
    let logits = decoder(tokens)
    eval(logits)  // Too frequent!
}

// GOOD: eval at chunk boundaries
for chunk in chunks {
    let result = processChunk(chunk)
    eval(result)  // Once per 30s chunk
}
```

### 6. Async Evaluation for Streaming

Use `mx.async_eval()` to pipeline graph construction with GPU execution:

```swift
mx.async_eval(nextTokenLogits)  // Returns immediately
// Do other work while GPU processes
```

---

## Comparison: Why whisper.cpp is Faster

| Aspect | whisper.cpp | Our MLX Implementation |
|--------|-------------|----------------------|
| GPU Backend | Metal (direct) | MLX (abstraction layer) |
| Weights | GGML 4-bit quantized | Float16/32 |
| Attention | Custom Metal kernels | Standard MLX ops |
| Encoder | Core ML / ANE option | MLX GPU only |
| Graph | Pre-compiled | Dynamic per-call |

---

## Recommended Action Plan

### Quick Wins (1-2 days)
- [ ] Add `mx.compile()` to encoder/decoder forward passes
- [ ] Replace attention with `mx.fast.scaled_dot_product_attention`
- [ ] Profile with `mactop` to verify GPU utilization

### Medium Effort (1 week)
- [ ] Implement batched decoding (batch_size=8-12)
- [ ] Add 4-bit quantization option for models
- [ ] Use `mx.async_eval()` for streaming latency

### Advanced (2+ weeks)
- [ ] Port encoder to Core ML for ANE acceleration (3x speedup)
- [ ] Custom Metal kernels for mel spectrogram computation
- [ ] Implement speculative decoding (WhisperKit approach)

---

## Additional Resources

### Core Documentation
- [MLX Compilation Docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [Writing Fast MLX (Awni Hannun)](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)
- [MLX Swift Transforms+Compile.swift](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms+Compile.swift)

### Reference Implementations
- [MLX Examples - Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [Whisper Turbo MLX](https://github.com/JosefAlbers/whisper-turbo-mlx)
- [whisper.cpp GitHub](https://github.com/ggml-org/whisper.cpp)

### Apple Resources
- [MLX-Swift Blog](https://www.swift.org/blog/mlx-swift/)
- [Apple MLX Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [WWDC 2025 - Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)

---

## Notes on MLX Best Practices

From [Writing Fast MLX](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50):

1. **Profile first** - Check GPU utilization before micro-optimizing
2. **Lazy-load with type casting** - Cast weights to lower precision *before* evaluation
3. **Release temporaries early** - Delete intermediate variables before `eval()` calls
4. **Matrix operation ordering** - Prefer `x @ W.T` over `x @ W` for vector-matrix multiplication
5. **Avoid closure traps** - Pass arrays as explicit inputs to compiled functions
