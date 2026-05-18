#!/bin/bash
#SBATCH --job-name=dl-desi-spectra
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/caspian/logs/dl-desi-spectra-%j.out
#SBATCH --error=/shared/caspian/logs/dl-desi-spectra-%j.err
# Optional: avoid sharing the node with other jobs if the cluster allows it
# #SBATCH --exclusive
set -euo pipefail
REPO=/home/sotiria/cursor/data_lake
export DATA_LAKE_CONFIG=/shared/caspian/lake_config.toml
COADD_LIST=/shared/caspian/desi_coadds.txt
N_WORKERS=16
mkdir -p /shared/caspian/logs
source "${REPO}/.venv/bin/activate"
# n_workers <= cpus-per-task (leave headroom for writer + OS)
exec dl-ingest-spectra-batch \
  --config "$DATA_LAKE_CONFIG" \
  --survey DESI_DR1 \
  --file-list "$COADD_LIST" \
  --n-workers "$N_WORKERS" \
  --max-in-flight $((N_WORKERS + 2)) \
  --max-open-tiles 64 \
  --norder 5 \
  --on-duplicate skip \
  --update-catalog