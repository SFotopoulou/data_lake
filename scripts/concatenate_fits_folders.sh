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
#   scripts/concatenate_fits_folders.sh MY_SURVEY /data/shards /data/out \
#     --dry-run --summary
#
# Wrapper-only options:
#   --summary   After all folders, print output paths and sizes.
#               With --dry-run: planned paths + estimated size (sum of input shards).
#               Without --dry-run: actual written file sizes.
#               Existing outputs without --overwrite are skipped (not failed).
#
# Other options after OUTPUT_DIR are forwarded to concatenate_fits.py
# (e.g. --pattern, --recursive, --hdu, --join, --fits-memmap, --overwrite,
# --dry-run, --no-progress).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: concatenate_fits_folders.sh SURVEY INPUT_ROOT OUTPUT_DIR [options...]

  SURVEY       Survey name used in output filenames
  INPUT_ROOT   Parent directory whose immediate subfolders each hold FITS shards
  OUTPUT_DIR   Directory for merged files: {SURVEY}_{FOLDER}_merged.fits

Wrapper options:
  --summary    Report output files and sizes at the end (with --dry-run: planned
               paths and estimated size from input shards)

Other options are passed through to scripts/concatenate_fits.py.
EOF
}

human_bytes() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "${bytes}"
  else
    printf '%s B' "${bytes}"
  fi
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

SUMMARY=0
DRY_RUN=0
OVERWRITE=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --summary)
      SUMMARY=1
      ;;
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("${arg}")
      ;;
    --overwrite)
      OVERWRITE=1
      EXTRA_ARGS+=("${arg}")
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

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

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_DIR}"
fi

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
echo "dry_run=${DRY_RUN} summary=${SUMMARY}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "extra_args=${EXTRA_ARGS[*]}"
fi

failed=0
succeeded=0
skipped=0
dry_run_count=0

# Parallel arrays for --summary rows: path, size_bytes, n_files, status
declare -a SUM_PATHS=()
declare -a SUM_BYTES=()
declare -a SUM_NFILES=()
declare -a SUM_STATUS=()

for folder_path in "${folders[@]}"; do
  folder_name="$(basename "${folder_path}")"
  output_path="${OUTPUT_DIR}/${SURVEY}_${folder_name}_merged.fits"

  echo "----"
  echo "folder=${folder_name}"
  echo "output=${output_path}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    dry_out="$(mktemp)"
    if ! python "${CONCAT}" "${folder_path}" -o "${output_path}" "${EXTRA_ARGS[@]}" \
        >"${dry_out}"; then
      echo "FAILED: ${folder_name}" >&2
      failed=$((failed + 1))
      SUM_PATHS+=("${output_path}")
      SUM_BYTES+=(0)
      SUM_NFILES+=(0)
      SUM_STATUS+=("failed")
      rm -f "${dry_out}"
      continue
    fi

    n_files=0
    est_bytes=0
    while IFS= read -r fits_path || [[ -n "${fits_path}" ]]; do
      [[ -z "${fits_path}" ]] && continue
      n_files=$((n_files + 1))
      if [[ -f "${fits_path}" ]]; then
        # portable byte size (GNU/BSD)
        sz="$(wc -c < "${fits_path}" | tr -d ' ')"
        est_bytes=$((est_bytes + sz))
      fi
    done < "${dry_out}"
    rm -f "${dry_out}"

    dry_run_count=$((dry_run_count + 1))
    SUM_PATHS+=("${output_path}")
    SUM_BYTES+=("${est_bytes}")
    SUM_NFILES+=("${n_files}")
    SUM_STATUS+=("planned")
  else
    if [[ -f "${output_path}" && "${OVERWRITE}" -eq 0 ]]; then
      echo "SKIPPING: ${folder_name} (exists: ${output_path}; pass --overwrite to replace)"
      skipped=$((skipped + 1))
      out_bytes="$(wc -c < "${output_path}" | tr -d ' ')"
      SUM_PATHS+=("${output_path}")
      SUM_BYTES+=("${out_bytes}")
      SUM_NFILES+=("-")
      SUM_STATUS+=("skipped")
      continue
    fi

    if ! python "${CONCAT}" "${folder_path}" -o "${output_path}" "${EXTRA_ARGS[@]}"; then
      echo "FAILED: ${folder_name}" >&2
      failed=$((failed + 1))
      SUM_PATHS+=("${output_path}")
      SUM_BYTES+=(0)
      SUM_NFILES+=(0)
      SUM_STATUS+=("failed")
      continue
    fi

    succeeded=$((succeeded + 1))
    out_bytes=0
    if [[ -f "${output_path}" ]]; then
      out_bytes="$(wc -c < "${output_path}" | tr -d ' ')"
    fi
    SUM_PATHS+=("${output_path}")
    SUM_BYTES+=("${out_bytes}")
    SUM_NFILES+=("-")
    SUM_STATUS+=("written")
  fi
done

echo "----"
echo "done: succeeded=${succeeded} skipped=${skipped} dry_run=${dry_run_count} failed=${failed}"

if [[ "${SUMMARY}" -eq 1 ]]; then
  echo "==== summary ===="
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "# planned outputs (size ≈ sum of input shard bytes)"
  else
    echo "# written outputs"
  fi
  total_bytes=0
  total_inputs=0
  printf '%-10s %8s %12s  %s\n' "status" "n_in" "size" "path"
  for i in "${!SUM_PATHS[@]}"; do
    total_bytes=$((total_bytes + SUM_BYTES[i]))
    if [[ "${SUM_NFILES[i]}" =~ ^[0-9]+$ ]]; then
      total_inputs=$((total_inputs + SUM_NFILES[i]))
    fi
    printf '%-10s %8s %12s  %s\n' \
      "${SUM_STATUS[i]}" \
      "${SUM_NFILES[i]}" \
      "$(human_bytes "${SUM_BYTES[i]}")" \
      "${SUM_PATHS[i]}"
  done
  echo "----"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'total: folders=%d input_files=%d est_size=%s\n' \
      "${#SUM_PATHS[@]}" \
      "${total_inputs}" \
      "$(human_bytes "${total_bytes}")"
  else
    printf 'total: folders=%d size=%s\n' \
      "${#SUM_PATHS[@]}" \
      "$(human_bytes "${total_bytes}")"
  fi
fi

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
