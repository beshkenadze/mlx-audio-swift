# Parakeet bf16 — Upstream PR Ship Checklist

This doc lists everything that must be removed or converted before submitting
the bf16 work as an upstream PR. All affected sites are marked in code with
`TEMPORARY (feat-branch only — REMOVE before upstream PR)`.

## Context

Branch `feat/parakeet-integration` uses env-var gates (`PARAKEET_BF16=1`,
`PARAKEET_PROFILE=1`) to ship bf16 and profiling as opt-in experiments.
This keeps the feat-branch safe for A/B testing but is **not a ship-ready
pattern for upstream**.

Measured bf16 effect on 26-min batched corpus (B=8): **−8.3% wall**
(18.90s → 17.34s), cum_dec −17.2%, cum_enc −7.5%, word drift +0.2% (3478→3485).

## Decision: config flag via `ParakeetPreprocessConfig.computeDType`

Not default-on. Expose a configurable dtype with `.bfloat16` as default value.
Rationale: ASR WER on multilingual (especially Russian — see PR #108) is
sensitive to precision; users must be able to fall back to fp32 without a
rebuild. Config flag also plays well with existing parity tests (run in fp32),
matches PyTorch/MLX-LM ecosystem patterns, and keeps API surface minimal by
reusing `ParakeetPreprocessConfig` (no new type).

Proposed field:

```swift
public struct ParakeetPreprocessConfig {
    // ... existing fields ...
    public var computeDType: DType = .bfloat16
}
```

For sites that don't have `preprocessConfig` in scope (`makeInitialDecoderState`,
hybrid state init inside `decodeTDTHybrid`), thread the dtype through method
parameters or capture it on the model instance at init time.

## Delete list

### 1. `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`

Four `PARAKEET_BF16` env gates to replace with `preprocessConfig.computeDType`:

| Site | Line (approx.) | Action |
|---|---|---|
| `decodeTDT` entry feature cast | ~362 | `features = features.asType(preprocessConfig.computeDType)` |
| `decodeTDTHybrid` state init dtype | ~504 | Use `preprocessConfig.computeDType` (or thread via method param) |
| `makeInitialDecoderState` target dtype | ~880 | Add `dtype` parameter to method signature; callers pass `preprocessConfig.computeDType` |
| `fromDirectory` bf16 cast pass | ~1052 | Cast to `preprocessConfig.computeDType` at load; skip if already matching |

Two `PARAKEET_PROFILE` sites to delete entirely:

| Site | Line (approx.) | Action |
|---|---|---|
| `decodeTDT` timing wrap | ~369-378 | Delete the 6 gated lines |
| `ParakeetTDTProfile` class | ~1224-end | Delete the class entirely |

### 2. Swift files with structural/compile-fix changes — KEEP

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

## Remaining decisions

1. **Parity tests**: existing tests may fail with +0.2% word drift. Either
   update fixtures to bf16 outputs, or add tolerance-based comparison
   (edit distance ≤ 3 on typical sample). Alternative: keep parity tests
   running under `computeDType = .float32` explicitly.
2. **Documentation**: changelog entry noting bf16 default + how to opt out
   (`config.computeDType = .float32`) + measured −8.3% wall speedup.

## Verification before submitting PR

After resolving the gates:

- [ ] All `TEMPORARY` markers removed from source
- [ ] `grep -r "PARAKEET_BF16\|PARAKEET_PROFILE" Sources/` returns empty
- [ ] `swift build` clean
- [ ] `swift test` passes (with parity adjusted if needed)
- [ ] Re-run 26-min bench to confirm −8.3% still holds after cleanup
- [ ] This file (`docs/parakeet-ship-checklist.md`) deleted
