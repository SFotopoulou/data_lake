#!/bin/bash
# Slurm script: ingest 2dFGRS 1-D spectra (single long job with checkpoint/failure log).
#
# Required (set as environment variables before sbatch, or via --export):
#   DATA_LAKE_CONFIG  — path to your lake_config.toml
#   LAKE_INGEST_TOKEN — ingest secret
#   FILE_LIST         — text file with one absolute .fits path per line (300 k rows)
#
# Optional:
#   DATA_LAKE_REPO    — repo root (default: parent of this scripts/ directory)
#   SURVEY            — survey name written into the lake (default: 2DFGRS_DR3)
#   NORDER            — HEALPix order (default: 5)
#   WAVELENGTH_MODE   — shared or per_source (default: shared; 2dF has a fixed grid)
#
# How to submit:
#   mkdir -p logs
#   export DATA_LAKE_CONFIG=/path/to/lake_config.toml
#   export LAKE_INGEST_TOKEN='your-secret'
#   export FILE_LIST=/path/to/2df_files.txt
#   sbatch scripts/slurm_ingest_2df_spectra.sh
#
# Restarting after preemption or timeout:
#   Re-submit the same sbatch command — the checkpoint file keeps track of
#   already-processed files; they are skipped automatically.
#
# Reviewing failures:
#   cat $CHECKPOINT_DIR/failures.jsonl | python -m json.tool | less
#   # Extract just the failed paths for a retry list:
#   python -c "
#   import sys, json
#   for line in open('$CHECKPOINT_DIR/failures.jsonl'):
#       obj = json.loads(line)
#       print(obj.get('path', obj))
#   " > retry_list.txt
#   # Re-submit with FILE_LIST=retry_list.txt

#SBATCH --job-name=dl-2df-spectra
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/dl-2df-spectra-%j.out
#SBATCH --error=logs/dl-2df-spectra-%j.err
# Uncomment to receive email on finish/fail:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.address

set -euo pipefail

# ---- Validate required env vars -------------------------------------------
: "${DATA_LAKE_CONFIG:?Set DATA_LAKE_CONFIG to your lake_config.toml}"
: "${LAKE_INGEST_TOKEN:?Set LAKE_INGEST_TOKEN to your ingest secret}"
: "${FILE_LIST:?Set FILE_LIST to a file containing one .fits path per line}"

# ---- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${DATA_LAKE_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SURVEY="${SURVEY:-2DFGRS_DR3}"
NORDER="${NORDER:-5}"
WAVELENGTH_MODE="${WAVELENGTH_MODE:-shared}"

# Checkpoint and failure log live next to the lake config so they persist
# across job submissions.
CONFIG_DIR="$(dirname "${DATA_LAKE_CONFIG}")"
CHECKPOINT_DIR="${CONFIG_DIR}/ingest_state/2df"
mkdir -p "${CHECKPOINT_DIR}" "${REPO}/logs"

CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.json"
FAILURES_LOG="${CHECKPOINT_DIR}/failures.jsonl"

# ---- Activate virtualenv ---------------------------------------------------
source "${REPO}/.venv/bin/activate"

echo "========================================================="
echo "2dFGRS spectra ingest — $(date --iso-8601=seconds)"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Survey         : ${SURVEY}"
echo "File list      : ${FILE_LIST}"
echo "Norder         : ${NORDER}"
echo "Wavelength mode: ${WAVELENGTH_MODE}"
echo "Checkpoint     : ${CHECKPOINT}"
echo "Failures log   : ${FAILURES_LOG}"
echo "========================================================="

# ---- Count total files to give a progress baseline ------------------------
TOTAL=$(wc -l < "${FILE_LIST}" || echo "?")
echo "Total files in list: ${TOTAL}"

# ---- Main ingest -----------------------------------------------------------
dl-ingest-spectra-from-list \
    "${FILE_LIST}" \
    --config "${DATA_LAKE_CONFIG}" \
    --survey "${SURVEY}" \
    --fmt 2df \
    --norder "${NORDER}" \
    --wavelength-mode "${WAVELENGTH_MODE}" \
    --on-duplicate skip \
    --on-length-mismatch pad \
    --checkpoint "${CHECKPOINT}" \
    --failures-log "${FAILURES_LOG}" \
    --update-catalog \
    --verbose

INGEST_EXIT=$?

echo "========================================================="
echo "Ingest finished — $(date --iso-8601=seconds)"
echo "Exit code      : ${INGEST_EXIT}"

# ---- Post-ingest: count failures ------------------------------------------
if [[ -f "${FAILURES_LOG}" ]]; then
    N_FAIL=$(wc -l < "${FAILURES_LOG}")
    echo "Failures logged: ${N_FAIL} (see ${FAILURES_LOG})"
else
    echo "Failures logged: 0"
fi

# ---- Post-ingest: finalize catalog metadata --------------------------------
# Refreshes _metadata, catalog_info.json total_rows, and schema_manifest.json
# for the survey. Run even when individual files failed; the tile data written
# so far remains valid.
echo ""
echo "Running dl-finalize-catalog …"
dl-finalize-catalog \
    --config "${DATA_LAKE_CONFIG}" \
    --survey "${SURVEY}" \
    --ra-col ra \
    --dec-col dec

echo "dl-finalize-catalog done."
echo "========================================================="

# ---- Post-ingest: quick QA ------------------------------------------------
echo ""
echo "Running dl-validate-spectra-ingest (spot-check) …"
dl-validate-spectra-ingest \
    --config "${DATA_LAKE_CONFIG}" \
    --survey "${SURVEY}" \
    --norder "${NORDER}" \
|| echo "WARNING: validation reported issues — check the log above."

echo "All done — $(date --iso-8601=seconds)"
exit "${INGEST_EXIT}"
