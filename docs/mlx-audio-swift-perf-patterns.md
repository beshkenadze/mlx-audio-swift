# MLX-Swift Perf Patterns — extracted from Parakeet optimization

Distilled playbook for applying Parakeet-derived optimizations to other models
in this repo. Based on end-to-end investigation 2026-04-22/23.

Parakeet final result: **17.34s wall on 26-min batched corpus, 91× realtime,
+19% faster than ANE** on same M1 Max + harness. Baseline was 29s fp32 serial.

## Five transferable tricks

### 1. End-to-end bf16 cast (biggest win — −8% wall)
fp32 weights on disk + fp32 activations → MLX dispatches `_float32` Metal
kernels and emits `vn_copyfloat16float32` cast ops (observed ~18% GPU time in
one trace). Casting weights AND input activations AND state tensors to
`bfloat16` shifts kernel dispatch to `_bfloat16` variants, eliminating the
cast op entirely.

**Gotcha**: partial cast does NOT work. MLX promotes to fp32 if ANY input is
fp32. Must be end-to-end on the hot path.

**Implementation shape** (see `ParakeetModel.swift`):
```swift
public var computeDType: DType = .bfloat16  // runtime preference on model

// Factory accepts override:
fromDirectory(_ dir: URL, computeDType: DType = .bfloat16)

// Cast pass at load (quantization-aware):
for (key, value) in model.parameters().flattened() {
    guard value.dtype.isFloatingPoint, value.dtype != computeDType else { continue }
    // cast ...
}

// Cast input features and state tensors at call sites:
features = features.asType(computeDType)
```

### 2. `@ParameterInfo` audit for plain `var MLXArray`
Plain `var MLXArray` fields are NOT enumerated by `model.parameters()` walk,
so weight-cast passes and quantization walkers silently skip them. They stay
in their init-time dtype (usually fp32), causing dtype promotion at use-site.

**Example** (Parakeet): `posBiasU/V` in `ParakeetRelPositionMultiHeadAttention`
were plain vars → missed cast pass → bf16 attention matmul + fp32 bias →
promoted back to `_float32` SDPA kernel dispatch.

**Fix**: convert to `@ParameterInfo var posBiasU: MLXArray`. Check
checkpoint-key sanitize step for any needed remap (Parakeet already had
`pos_bias_u/v` remap).

**Anti-gotcha**: **inference state** (LSTM h/c, attention kv-cache) is NOT
parameters. Do not convert those — they need post-init reassignment which
`@ParameterInfo` blocks after `model.update(parameters:)`.

### 3. Quantization-aware cast filter
A blanket cast pass will corrupt `uint32`-packed quantized weights. Always
filter:

```swift
guard value.dtype.isFloatingPoint, value.dtype != targetDType else {
    return (key, value)  // skip: uint32 quantized, already matching, or integer
}
```

### 4. Fused readback via `MLX.stacked`
Any hot loop doing two `.item(T.self)` or `.asArray(T.self)` calls in sequence
pays two GPU→CPU syncs per step. Fuse into one:

```swift
// Before: 2 syncs per step
let token = tokenArgMax.item(Int32.self)
let duration = durationArgMax.item(Int32.self)

// After: 1 sync per step
let decisions = MLX.stacked([tokenArgMax, durationArgMax], axis: 0)
let [token, duration] = decisions.asArray(Int32.self)  // unpack on CPU
```

Each sync is tens of microseconds; on 1000-step decode loop this is 40-80ms
pure wall savings.

### 5. `zeros_like` inside compiled steps
`MLX.compile(shapeless: true)` needs stable-shape traced bodies. Dynamic-shape
init like `MLXArray.zeros(shape, type: Float.self).asType(targetDType)` inside
a compiled closure defeats graph reuse.

**Fix**: `MLXArray.zeros(like: referenceTensor)` — shape + dtype inferred from
traced input, stable graph.

## Other models — ROI ranking

Grep survey shows 20+ files with plain `var MLXArray` across the repo.
Candidates for applying the five patterns above:

### 🟢 Priority 1 — Large-encoder ASR (Parakeet-class architecture)

Same regime: Conformer/Transformer encoder + autoregressive decoder + fp32
weights on disk. Expected −5-10% wall each.

- `Sources/MLXAudioSTT/Models/Qwen3ASR/`
- `Sources/MLXAudioSTT/Models/FireRedASR2/` — already has `@ParameterInfo` in 2 files; audit for completeness
- `Sources/MLXAudioSTT/Models/CohereTranscribe/` — 3 files with plain MLXArray vars
- `Sources/MLXAudioSTT/Models/GraniteSpeech/`
- `Sources/MLXAudioSTT/Models/GLMASR/`
- `Sources/MLXAudioSTT/Models/VoxtralRealtime/`

### 🟢 Priority 1 — TTS autoregressive paths

Same fp32-on-disk, per-token `.item()` readback anti-pattern.

- `Sources/MLXAudioTTS/Models/Chatterbox/S3Gen/ConformerEncoder.swift` —
  direct Parakeet-shape match
- `Sources/MLXAudioTTS/Models/Chatterbox/T3/Perceiver.swift` — Perceiver attention
- `Sources/MLXAudioTTS/Models/Qwen3TTS/` — already has one `compile` seam, partial

### 🟡 Priority 2 — STS / speech enhancement

- `Sources/MLXAudioSTS/Models/LFMAudio/Conformer.swift` — direct architecture match
- `Sources/MLXAudioSTS/Models/MossFormer2SE/`
- `Sources/MLXAudioSTS/Models/SAMAudio/` — multi-file transformer

### 🟡 Priority 2 — VAD/diarization (latency-sensitive)

- `Sources/MLXAudioVAD/Models/Sortformer/` — 4 `@ParameterInfo` hits, transformer attention

### ⚪ Skip (low ROI or different compute profile)

- Codecs (no attention, different kernel regime)
- G2P (pure Swift logic, no MLX hot path)

## Anti-patterns confirmed dead on Apple Silicon / MLX-Swift

Do NOT re-investigate these without new evidence:

- **Whole-model INT4 quantization**: slower than fp32 on Parakeet (prior
  testing). Small sequential matmul regime in TDT decode doesn't amortize
  dequant overhead.
- **Encoder-only INT8 (W8A16, group_size=64)**: +34% wall regression on
  Parakeet encoder. Same dequant-overhead failure mode as INT4. Skip for
  Conformer/Transformer ASR encoders unless MLX ships a new fast INT8 path.
- **`MLX.compile(shapeless: true)` around `AddMM`/matmul-fused primitives**:
  deterministic crash at shape inference ("AddMM cannot infer output
  shapes"). shapeless compile OK for pointwise/reduce/reshape/argmax/stack,
  breaks on fused matmul. Narrow compile region around argmax-only if tried.
- **Padded fixed-batch compile for hybrid decode**: theoretical win, but
  padding overhead at B≤8 likely exceeds compile gain. Not attempted;
  consensus was negative EV.

## Verification checklist per model

Before claiming a bf16 win:

1. **Kernel trace in Instruments** — confirm `_float32` kernels replaced by
   `_bfloat16` variants; `vn_copyfloat16float32` dropped to ~0.
2. **WER / parity spot-check** — word drift should be <1% on multilingual
   samples.
3. **Same-machine same-corpus A/B** — compare against competitor (ANE,
   previous version) on identical harness, warm runs, exclude cold-start.
4. **`cum_enc` and `cum_dec` split** — judge encoder and decoder savings
   independently; bf16 typically helps encoder 5-10% and decoder 10-20%.
5. **Variance sanity** — ≥3 warm runs, spread should be <1% for clean signal.

## Discovered gotchas (learn-once)

- `MLXArray(scalar, type: dtype)` initializer **does not exist** in current
  MLX-Swift. Use `MLXArray(scalar).asType(dtype)` instead.
- `ParakeetRelPositionalEncoding` (sin/cos lookup) was converted to `Module`
  with `@ParameterInfo var pe` — caused runtime crash because `calculatePE()`
  reassigns `pe` post-init. Lesson: deterministic lookup tables should stay
  plain-class + use-site `asType()` casts, not be registered as params.
- `MLX.compile(shapeless: true)` API does exist in MLX-Swift (prior art:
  `Qwen3TTSCodePredictor.swift`), not a phantom.

## Files from this investigation

- `docs/parakeet-swift-vs-py-perf.md` — initial Python-vs-Swift gap analysis
- `docs/parakeet-bf16-cheap-check.md` — pre-flight verification
- `docs/parakeet-bf16-experiment.md` — first experimental patch
- `docs/parakeet-bf16-hybrid-leaks.md` — hybrid path audit
- `docs/parakeet-hybrid-perf-fixes.md` — hybrid fused-readback fix
- `docs/parakeet-ship-checklist.md` — final ship summary (API shape)
