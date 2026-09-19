#!/usr/bin/env bash
set -euo pipefail

PDB=""
CHAIN=""
MUT=""
MODEL_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -PDB)
      PDB="${2:-}"
      shift 2
      ;;
    -CHAIN)
      CHAIN="${2:-}"
      shift 2
      ;;
    -Mut)
      MUT="${2:-}"
      shift 2
      ;;
    --model_json)
      MODEL_JSON="${2:-}"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PDB" || -z "$CHAIN" || -z "$MUT" || -z "$MODEL_JSON" ]]; then
  echo "Usage: $0 -PDB 1URN -CHAIN A -Mut D92A --model_json model.json" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Do not write __pycache__/*.pyc for the Pred-MutPRI wrapper modules.
export PYTHONDONTWRITEBYTECODE=1

# Keep your current environment path as the default.
# It can still be overridden without editing the script:
#   PRED_MUTPRI_PYTHON=/path/to/python ./predict.sh ...
PY_ENV1="${PRED_MUTPRI_PYTHON:-/home/lenovo/env1/bin/python3}"

[[ -x "${PY_ENV1}" ]] || {
  echo "[ERROR] Python not executable: ${PY_ENV1}" >&2
  exit 1
}

cd "${SCRIPT_DIR}"

exec "${PY_ENV1}" -B "${SCRIPT_DIR}/predict_prismrna_xgb.py" \
  -PDB "${PDB}" \
  -CHAIN "${CHAIN}" \
  -Mut "${MUT}" \
  --model_json "${MODEL_JSON}"
