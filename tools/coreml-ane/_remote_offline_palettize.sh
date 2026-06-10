#!/usr/bin/env bash
# Runs ON alex-mac in tmux (venv35g). Converts the OFFLINE encoders at 8-bit palettization:
# Nemotron 3.5 and Parakeet v3 (same shared converter, no Swift change needed).
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
source .venv35g/bin/activate

echo "=== offline Nemotron 3.5 @ 8-bit ==="
PYTHONUNBUFFERED=1 python -u convert_encoder.py \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b --frames 1000 --palettize 8 \
  --out out/nemotron_35_enc_p8.mlpackage 2>&1 | grep -vE "compression pass|ops/s|Frontend ==>|MIL "
du -sh out/nemotron_35_enc_p8.mlpackage

echo "=== Parakeet v3 @ 8-bit ==="
PYTHONUNBUFFERED=1 python -u convert_encoder.py \
  --model nvidia/parakeet-tdt-0.6b-v3 --frames 1000 --palettize 8 \
  --out out/parakeet_v3_enc_p8.mlpackage 2>&1 | grep -vE "compression pass|ops/s|Frontend ==>|MIL "
du -sh out/parakeet_v3_enc_p8.mlpackage
echo "=== DONE ==="
