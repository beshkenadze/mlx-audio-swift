# G2P Consolidation + PR Cleanup Plan

## Problem

Two G2P modules with zero actual code sharing:

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `MLXAudioG2P` | 19 source + 5 tests | 1982 + ~500 | Generic pipeline — CMUDict, TextTokenizer, TextNormalizer, G2PPipeline |
| `MLXAudioTTS/G2P/` | 17 source | ~2200 | Misaki port — EnglishG2P(500L), Lexicon(595L), BART fallback(8 files) |

**Critical finding**: MLXAudioG2P's ONLY consumer is `NeuralPhonemizer.swift`, which imports exactly 3 types:
- `Phonemizing` protocol (3 lines)
- `PhonemeUnit` struct (7 lines)
- `G2PError` enum (7 lines)

Nobody in MLXAudioTTS/G2P imports MLXAudioG2P. The two stacks are completely independent.

**Cost**: 1982 lines of source + ~500 lines of tests + an entire Swift module — all for 17 lines of actual API surface.

## Solution: Eliminate MLXAudioG2P

Move the 3 used types (17 lines) into `MLXAudioNeuralG2P`. Delete everything else.

### What gets deleted (24 files, ~2500 lines)

**Source** (19 files):
- `MLXAudioG2P.swift` — module entry point
- `README.md`
- `Pipeline/` — G2PPipeline.swift, G2PError.swift, G2PInput.swift, G2POutput.swift
- `Phonemes/` — PhonemeUnit.swift, PhonemeSequence.swift
- `Lexicon/` — InMemoryLexicon.swift, LexiconEntry.swift, LexiconProviding.swift, CMUDict/ (CMUDictLoader, CMUDictParser, ARPAbetMapper)
- `Fallback/` — FallbackPhonemizer.swift
- `Languages/English/` — EnglishLanguagePack.swift
- `TextNormalization/` — TextNormalizer.swift
- `Tokenization/` — TextTokenizer.swift
- `Alignment/` — HeuristicTokenAligner.swift, TokenAligning.swift

**Tests** (5 files):
- MLXAudioG2PAlignmentTests.swift
- MLXAudioG2PCMUDictTests.swift
- MLXAudioG2PLexiconTests.swift
- MLXAudioG2PSmokeTests.swift (renamed to G2PPipelineTests)
- MLXAudioG2PTextNormalizerTests.swift

### What moves to MLXAudioNeuralG2P (17 lines, 1 new file)

Create `Sources/MLXAudioNeuralG2P/G2PTypes.swift`:
```swift
public struct PhonemeUnit: Sendable, Hashable {
    public let symbol: String
    public init(symbol: String) { self.symbol = symbol }
}

public protocol Phonemizing: Sendable {
    func phonemize(_ grapheme: String) throws -> [PhonemeUnit]
}

public enum G2PError: Error, Sendable, Equatable {
    case emptyInput
    case unsupportedLocale(String)
    case phonemizationFailed(token: String, reason: String)
    case alignmentFailed(reason: String)
    case resourceLoadFailed(name: String, reason: String)
}
```

### Changes to NeuralPhonemizer.swift

```diff
-import MLXAudioG2P
+// Phonemizing, PhonemeUnit, G2PError now local to this module
```

### Package.swift changes

- Remove `MLXAudioG2P` product
- Remove `MLXAudioG2P` target
- Remove from umbrella library
- Remove from test target deps
- Remove `MLXAudioG2P` from `MLXAudioNeuralG2P` dependencies
- Remove `"MLXAudioG2P"` from test target

## Updated PR Structure (5 → 4 PRs, 1 squashed commit each)

### Current (5 PRs, 21 total commits)
```
PR #115 (6 commits) → #116 (3) → #117 (3) → #118 (4) → #119 (5)
```

### New (4 PRs, 1 clean commit each)

**PR A**: `feat/neural-g2p` from `upstream/main`
- MLXAudioNeuralG2P (13 source + G2PTypes.swift = 14 files)
- `Tests/MLXAudioNeuralG2PTests.swift`
- Package.swift: add MLXAudioNeuralG2P product + target (deps: MLX, MLXFast, MLXNN, MLXRandom — NO MLXAudioG2P)
- tests.yaml: fix (add G2P test suite to matrix per Lucas's request)
- 1 commit: `feat: add MLXAudioNeuralG2P multilingual ByT5 G2P`
- **~700 reviewable lines**

**PR B**: `feat/styletts2-infra` from `feat/neural-g2p` (PR A)
- `Models/StyleTTS2/Blocks/` (8 NN block files)
- `Models/StyleTTS2/Albert.swift` + `SharedConfigs.swift`
- `G2P/` (17 Misaki files: EnglishG2P, Lexicon, BART, MisakiTextProcessor, etc.)
- `TextProcessor.swift`
- Package.swift: no new module deps (files are inside MLXAudioTTS)
- 1 commit: `feat: add StyleTTS2 shared blocks, ALBERT, and English G2P`
- **~3500 reviewable lines**

**PR C**: `feat/kitten-tts` from `feat/styletts2-infra` (PR B)
- `Models/StyleTTS2/KittenTTS/` (5 model files)
- `TTSModel.swift` (kitten_tts case only)
- `App.swift` (CLI, upstream version)
- Tests (KittenTTSTests)
- 1 commit: `feat: add KittenTTS text-to-speech model`
- **~1500 reviewable lines**

**PR D**: `feat/kokoro-tts` from `feat/kitten-tts` (PR C)
- `Models/StyleTTS2/Kokoro/` (6 model files + README)
- `KokoroMultilingualProcessor.swift`
- `TTSModel.swift` (+kokoro case)
- `App.swift` (+--language/--raw-ipa flags)
- Tests (KokoroTTSTests + KokoroMultilingualProcessorTests)
- Package.swift: add `MLXAudioNeuralG2P` to MLXAudioTTS deps
- 1 commit: `feat: add Kokoro TTS with multilingual support`
- **~2200 reviewable lines**

### Dependency chain
```
PR A (NeuralG2P) → PR B (StyleTTS2 infra) → PR C (KittenTTS) → PR D (Kokoro)
```

## Impact Summary

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| PRs | 5 | 4 | -1 |
| Total commits | 21 | 4 | -17 |
| Source modules | MLXAudioG2P + MLXAudioNeuralG2P | MLXAudioNeuralG2P | -1 module |
| G2P source files | 19 + 13 + 17 = 49 | 14 + 17 = 31 | -18 files |
| G2P test files | 5 + 1 = 6 | 1 | -5 files |
| G2P source lines | ~1982 + ~660 + ~2200 = ~4842 | ~700 + ~2200 = ~2900 | -1942 lines |
| Package.swift targets | +2 (G2P + NeuralG2P) | +1 (NeuralG2P) | -1 target |

## Execution Plan

### Step 0: Close existing PRs #115–119

### Step 1: Create PR A branch
```bash
git checkout upstream/main
git checkout -b feat/neural-g2p-v2
# Copy from feat/kokoro-tts:
# - Sources/MLXAudioNeuralG2P/ (13 files)
# - Create Sources/MLXAudioNeuralG2P/G2PTypes.swift (17 lines)
# - Remove `import MLXAudioG2P` from NeuralPhonemizer.swift
# - Tests/MLXAudioNeuralG2PTests.swift (remove import MLXAudioG2P, use local types)
# - Edit Package.swift: add MLXAudioNeuralG2P, NO MLXAudioG2P
# - Edit tests.yaml: add G2P struct name to matrix
# Build verify: swift build --target MLXAudioNeuralG2P + swift build --build-tests
```

### Step 2: Create PR B branch
```bash
git checkout feat/neural-g2p-v2
git checkout -b feat/styletts2-infra
# Copy from feat/kokoro-tts:
# - Sources/MLXAudioTTS/Models/StyleTTS2/Blocks/ (8 files)
# - Sources/MLXAudioTTS/Models/StyleTTS2/Albert.swift
# - Sources/MLXAudioTTS/Models/StyleTTS2/SharedConfigs.swift
# - Sources/MLXAudioTTS/G2P/ (17 files)
# - Sources/MLXAudioTTS/TextProcessor.swift
# Build verify: swift build --target MLXAudioTTS
```

### Step 3: Create PR C branch
```bash
git checkout feat/styletts2-infra
git checkout -b feat/kitten-tts-v3
# Copy from feat/kokoro-tts:
# - Sources/MLXAudioTTS/Models/StyleTTS2/KittenTTS/ (5 files)
# - Sources/MLXAudioTTS/TTSModel.swift (kitten_tts case)
# - Sources/Tools/mlx-audio-swift-tts/App.swift (upstream version)
# - Tests/MLXAudioTTSTests.swift (KittenTTSTests section)
# Build verify
```

### Step 4: Create PR D branch
```bash
git checkout feat/kitten-tts-v3
git checkout -b feat/kokoro-tts-v2
# Copy from feat/kokoro-tts:
# - Sources/MLXAudioTTS/Models/StyleTTS2/Kokoro/ (6+README files)
# - Sources/MLXAudioTTS/G2P/KokoroMultilingualProcessor.swift (or Models/StyleTTS2/Kokoro/)
# - Update TTSModel.swift (+kokoro case)
# - Update App.swift (+--language/--raw-ipa)
# - Tests (Kokoro sections)
# - Package.swift: add MLXAudioNeuralG2P to MLXAudioTTS deps
# Build verify
```

### Step 5: Create all 4 PRs, close #115–119

## FAQ

**Q: What happens to CMUDict?**
A: Already on HuggingFace (`beshkenadze/cmudict-ipa`). CMUDictLoader, CMUDictParser, ARPAbetMapper are deleted. If needed in the future, they can be reimplemented or restored from git history.

**Q: What about the MLXAudioG2P tests (TextNormalizer, Alignment, etc.)?**
A: Deleted. They tested infrastructure that nobody uses. The MLXAudioNeuralG2PTests.swift remains and covers the ByT5 pipeline.

**Q: Can MLXAudioG2P be restored later?**
A: Yes, it's preserved in git history on `feat/audio-modules-g2p` branch and `feat/kitten-tts-v2` branch. If a future model needs generic G2P pipeline, it can be extracted.

**Q: Where does KokoroMultilingualProcessor live?**
A: In `Sources/MLXAudioTTS/G2P/KokoroMultilingualProcessor.swift` alongside MisakiTextProcessor. Both are TextProcessor implementations — G2P is the right place.
