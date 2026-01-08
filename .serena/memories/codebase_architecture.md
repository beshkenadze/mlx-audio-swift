# Codebase Architecture

## Module Overview

### MLXAudio (Library)
Main library providing TTS functionality. Published as SPM package.

**Key Components:**
- `TTSProvider.swift` - Enum defining TTS providers (Kokoro, Orpheus, Marvis)
- `ContentView.swift` - Main SwiftUI view

### Kokoro Module (`MLXAudio/Kokoro/`)
Kokoro TTS implementation using eSpeak NG for phoneme generation.

**Architecture:**
```
Kokoro/
├── Albert/           # ALBERT model for text encoding
├── Decoder/          # Audio decoder (STFT, HiFi-GAN style)
├── BuildingBlocks/   # Reusable neural network layers
├── TTSEngine/        # Core TTS engine
├── TextProcessing/   # Tokenization, phonemization
├── Utils/            # Audio utilities
└── KokoroTTSModel.swift  # Main entry point
```

### Orpheus Module (`MLXAudio/Orpheus/`)
3B parameter conversational TTS model.

**Architecture:**
```
Orpheus/
├── BuildingBlocks/   # Transformer blocks, RoPE
├── TTSEngine/        # TTS generation engine
├── SNAC/             # SNAC audio codec decoder
├── TextProcessing/   # Tokenizer
└── OrpheusTTSModel.swift  # Main entry point
```

### Marvis Module (`MLXAudio/Marvis/`)
Streaming conversational TTS with Mimi vocoder.

**Architecture:**
```
Marvis/
├── Mimi/             # Mimi vocoder (Transformer, Conv, Quantization)
├── Models/           # MarvisModel, LlamaModel
├── Audio/            # MLXArray audio extensions
├── MarvisSession.swift   # Main session manager (streaming capable)
└── AudioPlayback.swift   # Audio playback handling
```

**Key Class: MarvisSession**
- Factory methods: `make(voice:)`, `fromPretrained()`
- Generation: `generate()`, `stream()`, `generateAsync()`
- Supports quality levels and custom voices

### Views (`MLXAudio/Views/`)
SwiftUI views organized by function:
- `Sidebar/` - Navigation sidebar
- `Inspector/` - Settings panels (voice picker, model picker, etc.)
- `MainContent/` - Main TTS interface

### Utils (`MLXAudio/Utils/`)
- `AudioPlayerManager.swift` - Audio playback management
- `AudioSessionManager.swift` - Audio session configuration

## Data Flow (TTS)
```
Text Input → Tokenization → Model Inference (MLX) → Audio Decode → AVAudioEngine → Speaker
```

## Key Patterns
1. **Protocol-oriented**: TTS providers share common interface
2. **Observable**: Models use `ObservableObject` for SwiftUI binding
3. **Streaming**: Marvis supports real-time audio streaming
4. **Factory pattern**: Session creation via `make()` methods
