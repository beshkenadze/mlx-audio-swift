# Positional Embeddings in STT Models

## Key Finding
Different STT models handle positional embeddings differently. This affects how weights are loaded.

## Whisper Specifics
- **Encoder**: Uses **computed sinusoidal** positional embeddings (NOT loaded from weights)
- **Decoder**: Uses **learned** positional embeddings (loaded from weights)

### Evidence
1. OpenAI Whisper `model.py`: `self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))`
   - `register_buffer` = not a trainable parameter, not saved in model weights
2. Official confirmation (Discussion #697): "We used sinusoidal encoding for the encoder and learned positional encoding for the decoder"

## Other STT Models Comparison

| Model | Encoder Pos Emb | Decoder Pos Emb | Type |
|-------|-----------------|-----------------|------|
| Whisper | Sinusoidal (computed) | Learned (weights) | Enc-Dec |
| Wav2Vec 2.0 | Convolutional (learned) | — | Encoder-only |
| HuBERT | Convolutional (learned) | — | Encoder-only |
| Conformer | Relative (RoPE-like) | — | Encoder-only |
| Speech-T5 | Learned (weights) | Learned (weights) | Enc-Dec |
| Seamless M4T | Learned (weights) | Learned (weights) | Enc-Dec |

## Implementation Notes

### For Whisper AudioEncoder
- Do NOT use `@ParameterInfo` for positionalEmbedding — it's computed, not loaded
- Use plain property and compute sinusoidal embeddings in init()

### For Whisper TextDecoder
- DO use `@ParameterInfo(key: "positional_embedding")` — it IS loaded from weights

### Future Multi-Model Support
When adding non-Whisper models, consider:
1. Protocol abstraction: `PositionalEncoding` protocol
2. Enum-based config: `PositionalEmbeddingType`
3. Keep it simple until second model is actually needed (YAGNI)

## Sources
- https://github.com/openai/whisper/blob/main/whisper/model.py
- https://github.com/openai/whisper/discussions/697
- https://github.com/JosefAlbers/whisper-turbo-mlx/blob/main/whisper_turbo.py
