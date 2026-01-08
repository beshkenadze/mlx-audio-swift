# Whisper Tokenizer Sources and Loading

## Key Finding
MLX-community repos do NOT contain tokenizer files. Must load from OpenAI repos.

## Tokenizer File Locations

| Source | Weights | Tokenizer Files |
|--------|---------|-----------------|
| mlx-community/whisper-* | ✅ .safetensors, config.json | ❌ No tokenizer |
| openai/whisper-* | ✅ model.safetensors | ✅ tokenizer.json, vocab.json, merges.txt |

## Required Tokenizer Files (from OpenAI repos)
- `tokenizer.json` (2.7 MB) - main tokenizer file
- `tokenizer_config.json` - configuration
- `vocab.json` (1.0 MB) - vocabulary
- `merges.txt` (494 KB) - BPE merges

## OpenAI Repo Mapping
```
tiny     -> openai/whisper-tiny
base     -> openai/whisper-base
small    -> openai/whisper-small
medium   -> openai/whisper-medium
largeV3  -> openai/whisper-large-v3
largeTurbo -> openai/whisper-large-v3-turbo
```

## Swift Implementation
Use `swift-transformers` library:
```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
// or
let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v3-turbo")
```

## Vocab Size Info
- Multilingual models: 51,865 tokens
- English-only models: smaller vocab
- Check: `is_multilingual = vocab_size >= 51865`

## Sources
- https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
- https://huggingface.co/openai/whisper-large-v3-turbo
- https://github.com/huggingface/swift-transformers
- https://github.com/argmaxinc/WhisperKit
