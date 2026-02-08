#!/usr/bin/env bash
set -euo pipefail

# 用法：
# ./predict_one.sh -PDB 1URN -CHAIN A -Mut D92A --model_json ./model.json

PDB=""
CHAIN=""
MUT=""
MODEL_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -PDB) PDB="${2:-}"; shift 2;;
    -CHAIN) CHAIN="${2:-}"; shift 2;;
    -Mut) MUT="${2:-}"; shift 2;;
    --model_json) MODEL_JSON="${2:-}"; shift 2;;
    *) echo "[ERROR] Unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -n "$PDB" && -n "$CHAIN" && -n "$MUT" && -n "$MODEL_JSON" ]] || {
  echo "Usage: $0 -PDB 1URN -CHAIN A -Mut D92A --model_json ./model.json" >&2
  exit 1
}

PY_ENV1="/home/lenovo/env1/bin/python3"   # 按你的实际路径修改
BASH="bash"

"$PY_ENV1" predict_prismrna_xgb.py \
  -PDB "$PDB" -CHAIN "$CHAIN" -Mut "$MUT" \
  --model_json "$MODEL_JSON" \
  --py_struct "$PY_ENV1" \
  --py_network "$PY_ENV1" \
  --py_esm "$PY_ENV1" \
  --bash "$BASH"