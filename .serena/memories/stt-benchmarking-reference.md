# STT Benchmarking Reference

## Quick Reference

### Datasets
- **LibriSpeech test-clean/test-other** - Industry standard, ~5h each
- **Common Voice** - Diverse accents
- Download: https://www.openslr.org/12

### Metrics
- **WER**: Word Error Rate (primary accuracy metric)
- **RTF**: Real-Time Factor = processing_time / audio_duration
- **Latency**: Time to first token

### Current MLX Audio Swift Performance
- Model: large-v3-turbo
- RTF: ~0.20 (5x real-time)
- Test: 11.3s audio → 2.2s processing

### Comparable Implementations
1. WhisperKit (Swift/CoreML) - https://github.com/argmaxinc/WhisperKit
2. Lightning Whisper MLX (Python) - 10x faster than whisper.cpp
3. whisper.cpp (C++) - Baseline reference

### Benchmarking Tools
- mac-whisper-speedtest: https://github.com/anvanvan/mac-whisper-speedtest
- WhisperKit benchmarks: https://huggingface.co/argmaxinc/whisperkit-coreml

### WER Calculation
```python
from jiwer import wer
error_rate = wer(reference, hypothesis)
```

### Full Guide Location
docs/STT_BENCHMARKING_GUIDE.md
