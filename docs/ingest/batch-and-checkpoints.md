# Batch ingest, checkpoints, and duplicate policy

#### Duplicate / resume flags by command

| Command | Flag | Values | Notes |
|---------|------|--------|--------|
| `dl-ingest-catalog` | `--on-duplicate-id` | `skip`, `error`, `last` | Only when `--tile-mode append` (Parquet rows) |
| `dl-ingest-catalog` | `--allow-incomplete-link-id` | flag | Null `_source_id` for rows with missing composite/label parts; row kept, `_spectrum_index` = -1 |
| `dl-ingest-catalog` | `--tile-mode` | `append`, `replace` | Default `append`; `replace` overwrites existing tile |
| `dl-ingest-catalog` | `--streaming` | flag | FITS-only; bounded RAM; not available on batch path |
| `dl-ingest-catalog-from-list` | `--on-duplicate-id` | same | same |
| `dl-ingest-catalog-from-list` | `--allow-incomplete-link-id` | flag | Same semantics as single-file command; forwarded to parallel path when `--n-workers > 1` |
| `dl-ingest-catalog-from-list` | `--tile-mode` | `skip`, `overwrite`, `append` | Sequential default `skip`; parallel default `append` (when omitted with `--n-workers > 1`) |
| `dl-ingest-catalog-batch` | `--on-duplicate-id` | same | Parallel decode; default `--tile-mode append`; writes manifest at finalize |
| `dl-ingest-catalog-batch` | `--allow-incomplete-link-id` | flag | Same semantics as single-file command |
| `dl-ingest-catalog-batch` | `--tile-mode` | `skip`, `overwrite`, `append` | Default `append` (recommended for multi-file ingest) |
| `dl-finalize-catalog` | — | — | Rebuild ``catalog_info.json``, ``_metadata``, ``schema_manifest.json`` from tiles |
| `dl-repair-catalog-metadata` | `--rebuild-link-id` | column name | Recompute catalog ``_source_id`` from column (catalog only); then run ``dl-rebuild-catalog-indices`` |
| `dl-repair-catalog-metadata` | `--allow-incomplete-link-id` | flag | Allow null `_source_id` when rebuilding link IDs |
| `dl-repair-catalog-metadata` | — | — | Repair ``catalog_info.json``; ``--check-only``; ``--rebuild-link-id`` |
| `dl-ingest-cutouts` | `--on-duplicate` | `skip`, `error`, `append` | Default **`skip`**; per `source_id` in each `Npix=*.zarr` |
| `dl-ingest-cutouts-from-list` | `--on-duplicate` | same | same |
| `dl-ingest-spectra` | `--on-duplicate` | same | same |
| `dl-ingest-spectra-from-list` | `--on-duplicate` | same | Also ``--on-length-mismatch``, ``--wavelength-mode`` |
| `dl-ingest-spectra-from-list` | `--n-workers` | `1` (default) | `>1` parallel decode + single Zarr writer (not for ``desi_coadd``) |
| `dl-ingest-spectra-from-list` | `--max-in-flight` | — | Buffered decodes when ``--n-workers > 1`` (default: ``n_workers``) |
| `dl-ingest-spectra-batch-desi-coadds` | `--on-duplicate` | same | DESI parallel batch; default **`skip`** |

Cutout/spectrum ingest defaults to **`--on-duplicate skip`** so file-list and batch re-runs
are idempotent. Use **`append`** only when you intentionally want duplicate Zarr rows.
Catalog append uses **`--on-duplicate-id skip`** (default) with **`--tile-mode append`**.

#### Parallel catalog batch (large file lists)

For surveys shipped as **many catalog files** (e.g. Gaia `GaiaSource_*.csv.gz`), use
parallel decode with a **single-thread Parquet writer** so overlapping HEALPix tiles
are merged safely:

```bash
dl-ingest-catalog-batch gaia_files.txt --survey GAIA_DR3_source \
  --ra-col ra --dec-col dec --link-id-col source_id --norder 5 \
  --tile-mode append --on-duplicate-id skip --n-workers 8
```

`dl-ingest-catalog-from-list` with `--n-workers > 1` uses the same parallel engine and accepts the same flags as `dl-ingest-catalog-batch` (including `--allow-incomplete-link-id` and `--tile-mode`). The sequential path (`--n-workers 1`, the default) additionally supports `--streaming`. When `--tile-mode` is omitted, the parallel path defaults to `append` and the sequential path defaults to `skip`.

Each batch exit (including when every file is already in the checkpoint) runs
**finalize**: ``catalog_info.json``, Parquet ``_metadata``, and
``schema_manifest.json``. If a Slurm job is killed mid-run, tiles may exist without
a manifest — run ``dl-finalize-catalog --survey <name>`` before
``dl-refresh-lake-registry``.

Peak RAM scales roughly as **`O(n_workers × largest catalog file)`** — each worker
still decodes a full file in memory, but tile tables are **spooled to temp Parquet
in the worker** (not pickled back to the parent). Lower `--n-workers` and use
`--columns` on wide surveys (ALLWISE). Default **`--max-in-flight`** is **`n_workers`**
(not `n_workers + 2` like spectra batch).

If the OS kills a worker (**OOM**), you may see `BrokenProcessPool`; the batch tool
**restarts the pool** and re-queues in-flight files. Persistent OOM → fewer workers
and a smaller `--columns` set.

#### Sequential file-list ingest (catalogs, cutouts, spectra)

```bash
find /data/cats -name '*.fits' > cat_files.txt
dl-ingest-catalog-from-list cat_files.txt --survey my_survey --ra-col RA --dec-col DEC

find /data/cutouts -name '*.fits' > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --band-names r,i,z --on-duplicate skip

ls coadds.txt  # one DESI coadd path per line
dl-ingest-spectra-batch-desi-coadds --survey desi_dr1 --file-list coadds.txt --n-workers 16 \
  --on-duplicate skip

# Generic / SDSS spectra (sequential, not parallel DESI batch):
dl-ingest-spectra-from-list spec_files.txt --survey sdss_dr17 \
  --link-id-col SPECOBJID --on-duplicate skip

# spPlate file lists (same flags as dl-ingest-spectra):
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup /path/to/lookup.parquet --on-duplicate skip

SDSS spec lists: pixel lengths differ slightly; ingest auto-pads (or widens the
existing tile) when format is ``sdss_boss`` or ``sdss_spplate``.  Override with
``--on-length-mismatch pad`` (default; widens tiles for longer incoming spectra)
or ``truncate`` (same as ``dl-ingest-spectra``).

# Patch _cutout_index / _spectrum_index when --update-catalog (default).
```

Checkpoints default to ``catalogs/<survey>/.ingest_checkpoint.json`` or
``cutouts/<survey>/.ingest_checkpoint.json`` under the lake root.  Optional
``--failures-log`` writes JSONL per-file errors.

