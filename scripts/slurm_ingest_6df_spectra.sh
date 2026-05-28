#!/bin/bash
# Slurm script: ingest 6dFGS spectra (VR extension only) from a file list.
#
# Required env:
#   DATA_LAKE_CONFIG
#   LAKE_INGEST_TOKEN
#   FILE_LIST            # one FITS path per line
#
# Optional env:
#   DATA_LAKE_REPO       # default: repo root inferred from scripts/
#   SURVEY               # default: SIXDF_DR3
#   NORDER               # default: 5
#   SOURCE_ID_COL        # catalog column used for _spectrum_index patch (default: targetname)
#
# Submit:
#   mkdir -p logs
#   export DATA_LAKE_CONFIG=/path/to/lake_config.toml
#   export LAKE_INGEST_TOKEN='secret'
#   export FILE_LIST=/path/to/6df_files.txt
#   sbatch scripts/slurm_ingest_6df_spectra.sh
#
# Re-submit to resume: checkpoint tracks completed files.

#SBATCH --job-name=dl-6df-spectra
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/dl-6df-spectra-%j.out
#SBATCH --error=logs/dl-6df-spectra-%j.err

set -euo pipefail

: "${DATA_LAKE_CONFIG:?Set DATA_LAKE_CONFIG}"
: "${LAKE_INGEST_TOKEN:?Set LAKE_INGEST_TOKEN}"
: "${FILE_LIST:?Set FILE_LIST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${DATA_LAKE_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SURVEY="${SURVEY:-SIXDF_DR3}"
NORDER="${NORDER:-5}"
SOURCE_ID_COL="${SOURCE_ID_COL:-targetname}"

CONFIG_DIR="$(dirname "${DATA_LAKE_CONFIG}")"
STATE_DIR="${CONFIG_DIR}/ingest_state/6df"
mkdir -p "${REPO}/logs" "${STATE_DIR}"

CHECKPOINT="${STATE_DIR}/checkpoint.json"
FAILURES_LOG="${STATE_DIR}/failures.jsonl"

source "${REPO}/.venv/bin/activate"

echo "6dF ingest start: $(date --iso-8601=seconds)"
echo "survey=${SURVEY} norder=${NORDER}"
echo "file_list=${FILE_LIST}"
echo "checkpoint=${CHECKPOINT}"
echo "failures=${FAILURES_LOG}"

dl-ingest-spectra-from-list \
  "${FILE_LIST}" \
  --config "${DATA_LAKE_CONFIG}" \
  --survey "${SURVEY}" \
  --fmt 6df \
  --source-id-col "${SOURCE_ID_COL}" \
  --norder "${NORDER}" \
  --wavelength-mode shared \
  --on-duplicate skip \
  --on-length-mismatch pad \
  --checkpoint "${CHECKPOINT}" \
  --failures-log "${FAILURES_LOG}" \
  --update-catalog \
  --verbose

EXIT_CODE=$?
echo "6dF ingest end: $(date --iso-8601=seconds), exit=${EXIT_CODE}"

if [[ -f "${FAILURES_LOG}" ]]; then
  echo "failures_logged=$(wc -l < "${FAILURES_LOG}")"
fi

# Keep catalog metadata current for downstream queries.
dl-finalize-catalog \
  --config "${DATA_LAKE_CONFIG}" \
  --survey "${SURVEY}" \
  --ra-col ra \
  --dec-col dec || true

exit "${EXIT_CODE}"

