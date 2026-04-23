# Parakeet bf16 — Ship Summary

## Config API

`ParakeetModel.computeDType: DType` — mutable instance property, defaults to `.bfloat16`.
Applied at three runtime sites:

1. `decodeTDT` entry — casts input features to `computeDType`.
2. `decodeTDTHybrid` — LSTM state init dtype.
3. `makeInitialDecoderState` (serial TDT) — caller passes `computeDType`.

On load, `fromDirectory` casts all floating-point parameters to `computeDType`
unconditionally (skips params already matching; leaves non-float weights, e.g.
uint32-packed quantized tensors, untouched).

## Factory signatures

```swift
public static func fromDirectory(
    _ modelDir: URL,
    computeDType: DType = .bfloat16
) throws -> ParakeetModel

public static func fromPretrained(
    _ modelPath: String,
    computeDType: DType = .bfloat16,
    cache: HubCache = .default
) async throws -> ParakeetModel
```

To fall back to fp32, pass `computeDType: .float32` to the factory.

## Measured effect (26-min batched corpus, B=8)

- Wall clock: 18.90s → 17.34s (−8.3%)
- cum_dec: −17.2%, cum_enc: −7.5%
- Word count drift: 3478 → 3485 (+0.2%)

## KEEP — structural changes in Attention/Conformer

These are correct changes independent of the bf16 experiment. They stay.

- `ParakeetAttention.swift` — `posBiasU/posBiasV` converted to `@ParameterInfo`
  (so they reach `model.parameters()` walk). This is a real structural fix.
- `ParakeetAttention.swift` — `MLXArray(scale).asType(matrixBD.dtype)`
  replacing non-existent `MLXArray(scale, type:)` sig. Compile fix.
- `ParakeetConformer.swift` — same `MLXArray(0.5).asType(...)` pattern.
  Compile fix.
- `ParakeetRelPositionalEncoding` — left as plain class (not `Module`).
  Earlier conversion to Module caused runtime crash on `pe` reassignment;
  `.asType(x.dtype)` at use-site is sufficient.

## Open

- **Parity tests**: with +0.2% word drift, existing fp32-fixture tests may need
  tolerance adjustment (edit distance ≤ 3), new bf16 fixtures, or explicit
  `computeDType: .float32` in the test harness. Not yet addressed.
