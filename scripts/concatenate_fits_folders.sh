#!/usr/bin/env bash
# Concatenate catalog FITS shards in each subdirectory of an input root.
#
# For every immediate child directory under INPUT_ROOT, runs
# scripts/concatenate_fits.py and writes:
#   OUTPUT_DIR/{SURVEY}_{FOLDER}_merged.fits
#
# Usage:
#   scripts/concatenate_fits_folders.sh SURVEY INPUT_ROOT OUTPUT_DIR [options...]
#
# Examples:
#   scripts/concatenate_fits_folders.sh LEGACY_DR10 /data/legacy/bricks /data/legacy/merged
#   scripts/concatenate_fits_folders.sh GAIA_DR3 /data/gaia/runs /data/gaia/merged \
#     --pattern 'GaiaSource_*.fits' --recursive --overwrite
#   scripts/concatenate_fits_folders.sh MY_SURVEY /data/shards /data/out --dry-run
#
# Extra options after OUTPUT_DIR are forwarded to concatenate_fits.py
# (e.g. --pattern, --recursive, --hdu, --join, --fits-memmap, --overwrite,
# --dry-run, --no-progress).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: concatenate_fits_folders.sh SURVEY INPUT_ROOT OUTPUT_DIR [options...]

  SURVEY       Survey name used in output filenames
  INPUT_ROOT   Parent directory whose immediate subfolders each hold FITS shards
  OUTPUT_DIR   Directory for merged files: {SURVEY}_{FOLDER}_merged.fits

Options after OUTPUT_DIR are passed through to scripts/concatenate_fits.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 1
fi

SURVEY="$1"
INPUT_ROOT="$2"
OUTPUT_DIR="$3"
shift 3
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONCAT="${SCRIPT_DIR}/concatenate_fits.py"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "ERROR: input root is not a directory: ${INPUT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CONCAT}" ]]; then
  echo "ERROR: concatenate script not found: ${CONCAT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Prefer project venv when present (same as other scripts/).
if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
fi

shopt -s nullglob
folders=("${INPUT_ROOT}"/*/)
shopt -u nullglob

if [[ ${#folders[@]} -eq 0 ]]; then
  echo "ERROR: no subdirectories under ${INPUT_ROOT}" >&2
  exit 1
fi

echo "survey=${SURVEY}"
echo "input_root=${INPUT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "folders=${#folders[@]}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "extra_args=${EXTRA_ARGS[*]}"
fi

failed=0
succeeded=0
skipped=0

for folder_path in "${folders[@]}"; do
  folder_name="$(basename "${folder_path}")"
  output_path="${OUTPUT_DIR}/${SURVEY}_${folder_name}_merged.fits"

  echo "----"
  echo "folder=${folder_name}"
  echo "output=${output_path}"

  if ! python "${CONCAT}" "${folder_path}" -o "${output_path}" "${EXTRA_ARGS[@]}"; then
    echo "FAILED: ${folder_name}" >&2
    failed=$((failed + 1))
    continue
  fi

  # --dry-run exits 0 without writing; count as skipped for the summary.
  if [[ " ${EXTRA_ARGS[*]} " == *" --dry-run "* ]]; then
    skipped=$((skipped + 1))
  else
    succeeded=$((succeeded + 1))
  fi
done

echo "----"
echo "done: succeeded=${succeeded} dry_run=${skipped} failed=${failed}"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
