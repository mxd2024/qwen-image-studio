#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/models/hf-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONIOENCODING=utf-8
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3
fi
echo 'Local Diffusers Studio: http://127.0.0.1:8321'
exec "$python_bin" -m uvicorn app:app --host 127.0.0.1 --port 8321
