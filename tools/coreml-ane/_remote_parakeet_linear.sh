#!/usr/bin/env bash
# Runs ON alex-mac in tmux (venv35g). Parakeet v3 encoder @ per-channel linear int8, then
# parity-check the encoder output vs the torch fp32 reference (uniform palettize gave cosine 0.21).
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
source .venv35g/bin/activate

echo "=== Parakeet v3 @ linear int8 ==="
PYTHONUNBUFFERED=1 python -u convert_encoder.py \
  --model nvidia/parakeet-tdt-0.6b-v3 --frames 1000 --palettize -1 \
  --out out/parakeet_v3_enc_q8.mlpackage 2>&1 | grep -vE "compression pass|ops/s|Frontend ==>|MIL "
du -sh out/parakeet_v3_enc_q8.mlpackage

echo "=== parity vs torch fp32 ==="
python -u - <<'PY' 2>&1 | grep -vE "scikit|Torch ver"
import numpy as np, coremltools as ct
ref = np.load("out/parakeet_v3_enc_q8_ref_io.npz")
feats = ref["features"].astype(np.float32); tref = ref["encoded_torch_fp32"].astype(np.float64).ravel()
m = ct.models.MLModel("out/parakeet_v3_enc_q8.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
enc = np.asarray(m.predict({"features": feats})["encoded"], np.float64)
a = enc.ravel(); n = min(a.size, tref.size)
cos = float(a[:n]@tref[:n]/(np.linalg.norm(a[:n])*np.linalg.norm(tref[:n])))
print(f"linear-int8: cosine(vs torch fp32) = {cos:.4f}  range[{enc.min():.2f},{enc.max():.2f}]")
PY
echo "=== DONE ==="
