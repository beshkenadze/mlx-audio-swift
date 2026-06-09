#!/usr/bin/env python3
"""Probe the 3.5 multilingual streaming model: does this NeMo load it, what att_context /
streaming_cfg does its encoder use? Informs the streaming CoreML conversion params."""
import nemo
print("nemo", nemo.__version__)
try:
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt  # noqa: F401
    print("EncDecRNNTBPEModelWithPrompt: OK")
except Exception as e:  # noqa: BLE001
    print("EncDecRNNTBPEModelWithPrompt: MISSING", type(e).__name__, e)

import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu").train(False)
enc = m.encoder
print("model class:", type(m).__name__)
print("att_context_size_all:", getattr(enc, "att_context_size_all", None))
print("default att_context:", getattr(enc, "att_context_size", None))
enc.setup_streaming_params()
cfg = enc.streaming_cfg
print("streaming_cfg chunk_size:", cfg.chunk_size,
      "pre_encode:", cfg.pre_encode_cache_size,
      "valid_out:", cfg.valid_out_len,
      "last_channel_cache:", getattr(cfg, "last_channel_cache_size", None))
ch, t, clen = enc.get_initial_cache_state(batch_size=1)
print("cache_last_channel:", tuple(ch.shape), "cache_last_time:", tuple(t.shape))
print("feat_in:", m.cfg.encoder.feat_in, "d_model:", m.cfg.encoder.d_model)
