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

## Delete list

### 1. `Sources/MLXAudioSTT/Models/Parakeet/ParakeetModel.swift`

Four `PARAKEET_BF16` env gates to resolve (not just delete — upstream must
pick default-on OR a config flag):

| Site | Line (approx.) | Action |
|---|---|---|
| `decodeTDT` entry feature cast | ~362 | Unconditional `features = features.asType(.bfloat16)` OR via config |
| `decodeTDTHybrid` state init dtype | ~504 | Use `batchFeatures.dtype` unconditionally (already bf16 if feature cast is on) |
| `makeInitialDecoderState` target dtype | ~880 | Use caller-provided `dtype` unconditionally |
| `fromDirectory` bf16 cast pass | ~1052 | Make default-on OR expose via config |

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

## Upstream PR decision points

Upstream maintainer must decide (out of scope for feat-branch):

1. **bf16 default policy**: always-on, config flag, or runtime parameter?
2. **Parity tests**: existing tests may fail with +0.2% word drift. Either
   update fixtures or add tolerance-based comparison.
3. **fp32 fallback**: is there a use case that needs fp32? (parity vs
   reference, old hardware, debugging).

## Verification before submitting PR

After resolving the gates:

- [ ] All `TEMPORARY` markers removed from source
- [ ] `grep -r "PARAKEET_BF16\|PARAKEET_PROFILE" Sources/` returns empty
- [ ] `swift build` clean
- [ ] `swift test` passes (with parity adjusted if needed)
- [ ] Re-run 26-min bench to confirm −8.3% still holds after cleanup
- [ ] This file (`docs/parakeet-ship-checklist.md`) deleted
