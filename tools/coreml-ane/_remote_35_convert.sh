#!/usr/bin/env bash
# Runs ON alex-mac in tmux (venv35g = NeMo 2.8 + coremltools). Converts the 3.5 MULTILINGUAL
# streaming encoder at att_context [56,13] (left 56, right 13 -> chunk 112, F=121), matching
# the MLX 3.5 model's streaming feeding.
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
source .venv35g/bin/activate

echo "=== convert 3.5 stream (att 56 13) ==="
PYTHONUNBUFFERED=1 python -u convert_encoder_coreml_stream.py \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b --att-context 56 13 \
  --out out/nemotron_35_stream_func.mlpackage
echo "=== manifest ==="
cat out/nemotron_35_stream_func_manifest.json
echo "=== DONE ==="
