#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./run_af3_all.sh -PDB 1URN -CHAIN A -Mut D92A
#
# 功能：
# 1) 切换到 conda env: af3
# 2) 调用 run_af3_mutation.py 生成 JSON（写到 ./af3_json/<label>.json）
# 3) 进入 ./alphafold3 运行 run_alphafold.py 生成预测（写到 ./af3_outputs/<label>/）
# 4) 回到工作目录调用 calculating_es.py，只打印平均 ES（一个数字），并清理 af3_json/logs/af3_outputs

PDB=""
CHAIN=""
MUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -PDB)   PDB="${2:-}"; shift 2;;
    -CHAIN) CHAIN="${2:-}"; shift 2;;
    -Mut)   MUT="${2:-}"; shift 2;;
    *) echo "[ERROR] Unknown argument: $1" >&2; echo "Usage: $0 -PDB 1URN -CHAIN A -Mut D92A" >&2; exit 1;;
  esac
done

if [[ -z "${PDB}" || -z "${CHAIN}" || -z "${MUT}" ]]; then
  echo "[ERROR] Missing required arguments." >&2
  echo "Usage: $0 -PDB 1URN -CHAIN A -Mut D92A" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${PDB}_${CHAIN}_${MUT}"

PDB_FILE="${SCRIPT_DIR}/${PDB}.pdb"
JSON_PATH="${SCRIPT_DIR}/af3_json/${LABEL}.json"
OUT_DIR="${SCRIPT_DIR}/af3_outputs/${LABEL}"
AF3_DIR="${SCRIPT_DIR}/alphafold3"

# sanity checks
[[ -f "${PDB_FILE}" ]] || { echo "[ERROR] PDB not found: ${PDB_FILE}" >&2; exit 1; }
[[ -d "${AF3_DIR}" ]] || { echo "[ERROR] alphafold3 folder not found: ${AF3_DIR}" >&2; exit 1; }
[[ -f "${AF3_DIR}/run_alphafold.py" ]] || { echo "[ERROR] run_alphafold.py not found in ${AF3_DIR}" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/run_af3_mutation.py" ]] || { echo "[ERROR] run_af3_mutation.py not found in ${SCRIPT_DIR}" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/calculating_es.py" ]] || { echo "[ERROR] calculating_es.py not found in ${SCRIPT_DIR}" >&2; exit 1; }

mkdir -p "${SCRIPT_DIR}/af3_json" "${SCRIPT_DIR}/af3_outputs"

# try to leave venv if present (optional)
if declare -F deactivate >/dev/null 2>&1; then
  deactivate || true
fi

# conda activate af3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate af3
AF3_PY="${CONDA_PREFIX}/bin/python"

# 1) generate JSON
"${AF3_PY}" "${SCRIPT_DIR}/run_af3_mutation.py" -PDB "${PDB}" -CHAIN "${CHAIN}" -Mut "${MUT}"
[[ -f "${JSON_PATH}" ]] || { echo "[ERROR] JSON not generated: ${JSON_PATH}" >&2; exit 1; }

# 2) run AF3 prediction
pushd "${AF3_DIR}" >/dev/null
"${AF3_PY}" run_alphafold.py --json_path="${JSON_PATH}" --output_dir="${OUT_DIR}"
popd >/dev/null

# 3) compute Avg_ES (prints only one number) and cleanup (default behavior)
#    If you want to debug without deleting, add --no_cleanup here.
"${AF3_PY}" "${SCRIPT_DIR}/calculating_es.py" -PDB "${PDB}" -CHAIN "${CHAIN}" -Mut "${MUT}"