#!/usr/bin/env bash
set -euo pipefail

# Internal AF3 stage.


PDB=""
CHAIN=""
MUT=""

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
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PDB" || -z "$CHAIN" || -z "$MUT" ]]; then
  echo "[ERROR] Missing required arguments." >&2
  exit 1
fi

if [[ -z "${PRED_MUTPRI_RESULT_FD:-}" ]]; then
  echo "[ERROR] Internal result channel unavailable. Use ./predict.sh." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
LABEL="${PDB}_${CHAIN}_${MUT}"

PDB_FILE="${SCRIPT_DIR}/${PDB}.pdb"
AF3_DIR="${SCRIPT_DIR}/alphafold3"

[[ -f "${PDB_FILE}" ]] || {
  echo "[ERROR] PDB not found: ${PDB_FILE}" >&2
  exit 1
}

[[ -d "${AF3_DIR}" ]] || {
  echo "[ERROR] alphafold3 folder not found: ${AF3_DIR}" >&2
  exit 1
}

[[ -f "${AF3_DIR}/run_alphafold.py" ]] || {
  echo "[ERROR] run_alphafold.py not found in ${AF3_DIR}" >&2
  exit 1
}

[[ -f "${SCRIPT_DIR}/run_af3_mutation.py" ]] || {
  echo "[ERROR] run_af3_mutation.py not found." >&2
  exit 1
}

[[ -f "${SCRIPT_DIR}/calculating_es.py" ]] || {
  echo "[ERROR] calculating_es.py not found." >&2
  exit 1
}

TMP_BASE="${PRED_MUTPRI_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "${TMP_BASE}"

WORK_TMP="$(mktemp -d "${TMP_BASE%/}/predmutpri_af3.XXXXXX")"
chmod 700 "${WORK_TMP}"

cleanup() {
  rm -rf "${WORK_TMP}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM HUP

JSON_DIR="${WORK_TMP}/af3_json"
AF3_ROOT="${WORK_TMP}/af3_outputs"
JSON_PATH="${JSON_DIR}/${LABEL}.json"
OUT_DIR="${AF3_ROOT}/${LABEL}"
AF3_LOG="${WORK_TMP}/af3.log"

mkdir -p "${JSON_DIR}" "${AF3_ROOT}"

if declare -F deactivate >/dev/null 2>&1; then
  deactivate || true
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate af3
AF3_PY="${CONDA_PREFIX}/bin/python"

"${AF3_PY}" -B "${SCRIPT_DIR}/run_af3_mutation.py" \
  -PDB "${PDB}" \
  -CHAIN "${CHAIN}" \
  -Mut "${MUT}" \
  --json_path "${JSON_PATH}"

[[ -f "${JSON_PATH}" ]] || {
  echo "[ERROR] AF3 JSON was not generated." >&2
  exit 1
}

pushd "${AF3_DIR}" >/dev/null

set +e
"${AF3_PY}" -B run_alphafold.py \
  --json_path="${JSON_PATH}" \
  --output_dir="${OUT_DIR}" \
  >"${AF3_LOG}" 2>&1
AF3_RC=$?
set -e

popd >/dev/null

if [[ ${AF3_RC} -ne 0 ]]; then
  echo "[ERROR] AlphaFold3 failed." >&2
  if [[ "${PRED_MUTPRI_DEBUG:-}" == "1" ]]; then
    echo "[DEBUG] Last AF3 log lines:" >&2
    tail -n 120 "${AF3_LOG}" >&2 || true
  else
    echo "[ERROR] Set PRED_MUTPRI_DEBUG=1 for AF3 diagnostics." >&2
  fi
  exit "${AF3_RC}"
fi

"${AF3_PY}" -B "${SCRIPT_DIR}/calculating_es.py" \
  -PDB "${PDB}" \
  -CHAIN "${CHAIN}" \
  -Mut "${MUT}" \
  --work-dir "${SCRIPT_DIR}" \
  --af3-root "${AF3_ROOT}"
