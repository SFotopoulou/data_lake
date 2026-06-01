#!/bin/bash
# Example Slurm wrapper for dl-ingest-spectra-batch-desi-coadds.
#
# Required (export before sbatch, or pass via sbatch --export):
#   DATA_LAKE_CONFIG  — path to lake_config.toml
#   COADD_LIST        — one coadd/FITS path per line
#
# Optional:
#   DATA_LAKE_REPO    — repo root (default: parent of this scripts/ directory)
#   SURVEY            — survey name (default: DESI_DR1)
#   N_WORKERS         — parallel workers (default: 16; keep <= cpus-per-task)
#
# Submit from the repo root so job logs land in ./logs/:
#   mkdir -p logs
#   export DATA_LAKE_CONFIG=/path/to/lake_config.toml
#   export COADD_LIST=/path/to/coadds.txt
#   sbatch scripts/slurm_batch_ingest_spectra.py
#
#SBATCH --job-name=dl-desi-spectra
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/dl-desi-spectra-%j.out
#SBATCH --error=logs/dl-desi-spectra-%j.err
# Optional: avoid sharing the node with other jobs if the cluster allows it
# #SBATCH --exclusive

set -euo pipefail

: "${DATA_LAKE_CONFIG:?Set DATA_LAKE_CONFIG to your lake_config.toml}"
: "${COADD_LIST:?Set COADD_LIST to a file list (one path per line)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${DATA_LAKE_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SURVEY="${SURVEY:-DESI_DR1}"
N_WORKERS="${N_WORKERS:-16}"

mkdir -p "${REPO}/logs"
source "${REPO}/.venv/bin/activate"

# n_workers <= cpus-per-task (leave headroom for writer + OS)
exec dl-ingest-spectra-batch-desi-coadds \
  --config "$DATA_LAKE_CONFIG" \
  --survey "$SURVEY" \
  --file-list "$COADD_LIST" \
  --n-workers "$N_WORKERS" \
  --max-in-flight $((N_WORKERS + 2)) \
  --max-open-tiles 64 \
  --norder 5 \
  --on-duplicate skip \
  --update-catalog
