#!/usr/bin/env bash
# Runs ON alex-mac in tmux. NeMo main/2.8 (for the 3.5 multilingual
# EncDecRNNTBPEModelWithPrompt class) + coremltools in a dedicated venv, then probes the
# 3.5 streaming model. Stable PyPI nemo (2.7.3) lacks the prompt class.
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"

echo "=== venv35g ==="
[ -d .venv35g ] || uv venv --python 3.11 .venv35g
source .venv35g/bin/activate

if ! python -c "from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt" 2>/dev/null; then
  echo "=== install coremltools+librosa ==="
  uv pip install -q coremltools librosa numpy
  echo "=== try nemo 2.8 prerelease wheel ==="
  if uv pip install -q --prerelease=allow "nemo-toolkit[asr]>=2.8.0rc0,<2.10" 2>err.txt; then
    echo "prerelease wheel installed"
  else
    echo "prerelease wheel failed -> git main"; tail -3 err.txt
    uv pip install -q "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main" 2>&1 | tail -6
  fi
fi

echo "=== nemo version + class check ==="
python -c "import nemo; print('nemo', nemo.__version__)"
python -c "from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt; print('WithPrompt OK')"

echo "=== probe 3.5 ==="
PYTHONUNBUFFERED=1 python -u probe_35.py
echo "=== DONE ==="
