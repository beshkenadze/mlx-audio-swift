# MLX Audio Swift - Project Overview

## Purpose
macOS and iOS Text-to-Speech (TTS) application using Apple's MLX framework for local AI inference on Apple Silicon.

## Platform Requirements
- macOS 14+ (Sonoma)
- iOS 17+
- Apple Silicon (M1 chip or better required for Kokoro)

## Tech Stack
- **Language**: Swift 5.9
- **UI Framework**: SwiftUI
- **ML Framework**: MLX-Swift (Apple's ML framework for Apple Silicon)
- **Audio**: AVFoundation, AVAudioEngine
- **Dependencies**:
  - `mlx-swift-lm` (main branch) - MLX language model support
  - `swift-transformers` (1.1.x) - Hugging Face Transformers for Swift
  - `ESpeakNG` - Binary framework for phoneme generation (Kokoro)

## TTS Models Supported

### 1. Kokoro
- Fast, high-quality TTS
- Uses eSpeak NG for phoneme generation
- Multiple voices available (voice JSON files in Resources)
- Based on kokoro-ios implementation

### 2. Orpheus
- 3B parameter model
- Currently runs slow (~0.1x speed on M1) due to MLX-Swift caching limitations
- Supports expressions: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, etc.
- Voices: tara, leah, jess, leo, dan, mia, zac, zoe

### 3. Marvis
- Advanced conversational TTS
- Streaming audio generation
- Two voices: conversational_a, conversational_b
- 24kHz sample rate
- Auto-downloads model weights from Hugging Face

## Project Structure
```
mlx-audio-swift/
├── Package.swift           # SPM package definition
├── MLXAudio/               # Main library source (SPM target)
│   ├── Kokoro/             # Kokoro TTS implementation
│   ├── Orpheus/            # Orpheus TTS implementation
│   ├── Marvis/             # Marvis TTS implementation
│   ├── Views/              # SwiftUI views
│   └── Utils/              # Audio utilities
├── Swift-TTS/              # macOS demo application
├── MLXAudio-iOS/           # iOS demo application
├── Tests/                  # Unit tests
└── *.xcodeproj             # Xcode projects
```
