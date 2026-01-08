# STT Streaming Research Findings

## Reference Implementations

### 1. mlx-audio Python (наш проект)
- **Файл**: `mlx_audio/stt/models/whisper/streaming.py` (282 LoC)
- **Алгоритм**: AlignAtt с cross-attention monitoring
- **Ключевые функции**:
  - `get_most_attended_frame()` — находит фрейм с max attention
  - `should_emit()` — решает когда эмитить токены
  - `StreamingDecoder.decode_chunk()` — основной streaming loop

### 2. Lightning-SimulWhisper (MLX/CoreML)
- **Repo**: https://github.com/altalt-org/Lightning-SimulWhisper
- **Архитектура**: CoreML encoder (18x faster) + MLX decoder (15x faster)
- **frame_threshold**: 4 для последнего чанка
- **Результат**: real-time large-v3-turbo на M2

### 3. SimulStreaming (UFAL)
- **Repo**: https://github.com/ufal/SimulStreaming
- **Статус**: Топ IWSLT 2025 Simultaneous Shared Task
- **Фичи**:
  - CIF model для word boundary detection
  - VAD (Silero) для фильтрации тишины
  - Context preservation across 30s windows

## Alignment Heads по моделям

```swift
// Из OpenAI checkpoints
tiny:     [[2,2], [3,0], [3,2], [3,3], [3,4], [3,5]]
base:     [[3,1], [4,2], [4,3], [4,7], [5,1], [5,2], [5,4], [5,6]]
small:    [[5,3], [5,9], [8,0], [8,4], [8,7], [8,8], [9,0], [9,7], [9,9], [10,5]]
medium:   [[13,15], [15,4], [15,15], [16,1], [20,0], [23,4]]
large-v2: 23 heads (layers 10-27)
large-v3-turbo: needs empirical testing (only 4 decoder layers)
```

## Рекомендации для Swift реализации

| Компонент | Фаза 1 (MVP) | Фаза 2 (Optimization) |
|-----------|--------------|----------------------|
| Encoder | MLX | CoreML (battery saving) |
| Decoder | MLX + cross-attention capture | Beam search |
| AlignAtt | Порт из streaming.py | Tune frame_threshold |
| VAD | Нет | Silero или встроенный |
| CIF | Нет | Для word boundaries |

## Ключевой алгоритм AlignAtt

```
1. Decode token
2. Capture cross-attention weights
3. Get most_attended_frame from alignment heads
4. If (content_frames - most_attended_frame) <= threshold:
   → Emit token (stable)
5. Repeat until EOT
```

## Ссылки
- AlignAtt Paper: https://arxiv.org/abs/2211.00895
- Whisper Alignment Heads Gist: https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
- WhisperLive iOS (WebSocket): https://github.com/collabora/WhisperLive
