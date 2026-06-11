# Catalog ingest

### Check FITS layout before ingest

Before starting catalog ingest on FITS (especially large file lists), run
**`dl-check-fits-table-format`**. It reads **headers only** — no column I/O —
and reports how the lake will read each file:

| Report field | Meaning |
|--------------|---------|
| `format` | `standard-bintable` (one row per source) or `packed-vector` (column-oriented / STILTS **colfits** layout, `NAXIS2=1`) |
| `sources` | Estimated source count from `NAXIS2` or `TDIMn` |
| `file size` | On-disk bytes (useful for parallel worker RAM planning) |
| `ingest` | Fast memmap/streaming path vs slow column-by-column reader |

With a file list, the summary line aggregates **total size** and **total estimated
sources** across all files — use that baseline to verify ingestion completed as
expected (see below).

```bash
# One file
dl-check-fits-table-format /data/galex/photoobjall_001.fits

# Same paths as batch ingest
dl-check-fits-table-format --file-list galex_files.txt > galex.summary.txt

# Machine-readable (one JSON object per file)
dl-check-fits-table-format --file-list gaia_files.txt --json
```

**Verify after ingest:** compare the pre-check **total sources** to on-disk rows.
With `--on-duplicate-id skip` (default on batch append), `dl-describe-survey`
counts **unique rows in Parquet tiles** and may be **lower** than the sum of input
file rows when duplicates were skipped. Ingest progress logs often report **rows
read from FITS**, not rows written — a small gap is normal under dedup.

If any file is **`packed-vector`**, expect long ingest times and high RAM per
worker; parallel whole-file ingest is a poor fit. Re-export as row-normal FITS
(STILTS `col=false`) or Parquet, then ingest. See [Performance tuning](../performance.md).

### Ingest a survey catalog

With a deployment config in place (`$DATA_LAKE_CONFIG` set), the
`OUTPUT_ROOT` argument is optional — it is filled in from the config:

```bash
dl-ingest-catalog survey_catalog.fits --survey des_dr2 --ra-col RA --dec-col DEC
# Also: .csv, .csv.gz, .tsv, .tsv.gz, .parquet, VOTable (in-memory path; --streaming is FITS-only).
# Delimited text: auto-detects comma vs tab vs semicolon (tab-in-.csv.gz works for GAIA-style exports).
```

Without a config you can still pass the path explicitly (legacy mode):

```bash
dl-ingest-catalog survey_catalog.fits /data/lake --survey des_dr2 --ra-col RA --dec-col DEC
```

Explicit CLI flags (`--norder`, etc.) override config defaults.

For very large FITS catalogs pass `--streaming`. The streaming path memory-maps
the FITS, sorts only RA/Dec/source_id columns, then writes one HEALPix
tile at a time via per-tile fancy indexing into the memmap:

```bash
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
    --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
    --streaming
```

Memory peak is bounded to ~one tile's worth of rows (tens of MB at
Norder=5) instead of the full table + sorted copy (~3× the raw size).
The on-disk Parquet output is identical to the in-memory path
(round-trip-tested), so consumers don't care which mode was used.
Trade-off: per-tile scattered I/O makes the streaming path ~1.5–2×
slower wall-clock; use only when memory is a constraint.

**Disk footprint:** Parquet is often **larger than a compressed FITS** file
because FITS may use internal compression, while we store a full typed,
queryable columnar layout (~12k tile files, ZSTD, per-column statistics).
To reduce size on re-ingest:

```bash
# Smaller tiles (ZSTD-9, no stats/dictionary, narrow strings per tile)
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --overwrite --compact

# Largest win: drop unused DESI columns (keep what you query/join on)
dl-ingest-catalog ... --columns TARGETID,TARGET_RA,TARGET_DEC,Z,MAG_G,MAG_R,MAG_Z,SPECTYPE
```

Expect **~2–4×** smaller than default ingest when combining `--compact` with a
sensible `--columns` list; exact ratio depends on which FITS columns you keep.

**Multiple FITS into one survey:** by default existing `Npix=*.parquet` tiles are
**skipped** (`--tile-mode skip`). To add rows from another file into the same
HEALPix pixel, use **append** (read–concat–write per tile):

```bash
dl-ingest-catalog-from-list desi_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --tile-mode append
```

When appending with a native ID column (`TARGETID`), control duplicates with
`--on-duplicate-id skip|error|last` (default `skip`). Re-ingesting the **same**
FITS with ``append`` + ``skip`` is idempotent: rows already on disk (matched by
``source_id``) are dropped per tile; unchanged tiles are not rewritten. Use
``--tile-mode overwrite`` to rebuild a tile from one file only. ``--overwrite`` is
use ``--tile-mode overwrite`` to replace an existing tile. After every ingest (and at
the end of ``dl-ingest-catalog-batch``), ``_metadata``, ``catalog_info.json``
(``total_rows``), and ``schema_manifest.json`` are refreshed from **all** tiles on
disk. If a batch job was killed before finalize, or the manifest is missing after a
checkpoint-only re-run, use ``dl-finalize-catalog --survey <name>``.

**Mixed numeric dtypes across files** (common in AllWISE/GALEX batches): the same
column may be ``E`` (float32) in one FITS and ``D`` (float64) in another. Ingest
promotes floats to **float64**, integers to **int64**, and inner elements of
``FixedSizeList`` columns likewise, before writing or appending tiles — so append
no longer fails on dtype mismatch and the first file no longer locks a narrower
Parquet type.  When rebuilding ``_metadata``, any tiles still on legacy dtypes
(e.g. float32 ``flux`` in one ``Npix=`` file and float64 in another) are rewritten
automatically before the aggregate footer is written.

#### Choosing HEALPix order (`--norder` / `hats_order`)

Catalogs, cutouts, and spectra for a survey share one **HEALPix nested** order
(`assign_healpix` in ingest). Each non-empty sky pixel becomes one on-disk shard
(`Norder=<N>/Dir=<D>/Npix=<P>.parquet` or `.zarr`). **Disk footprint and inode
count scale with the number of shard files**, not only with row count — sparse
all-sky catalogs at high order can be much larger than the source FITS.

**Mean tile area on the full sphere** (equal-area pixels; same convention as
`healpy.order2nside` / this repo’s `--norder`):

| `--norder` | `nside` | Max tiles (full sky) | Mean tile area |
|------------|---------|----------------------|----------------|
| 0 | 1 | 12 | 3438 deg² |
| 1 | 2 | 48 | 859 deg² |
| 2 | 4 | 192 | 215 deg² |
| 3 | 8 | 768 | 53.7 deg² |
| 4 | 16 | 3 072 | 13.4 deg² |
| 5 | 32 | 12 288 | **3.36 deg²** (default) |
| 6 | 64 | 49 152 | 0.84 deg² |
| 7 | 128 | 196 608 | 0.21 deg² |
| 8 | 256 | 786 432 | 0.052 deg² |
| 9 | 512 | 3 145 728 | 0.013 deg² |
| 10 | 1024 | 12 582 912 | 0.0033 deg² |

Only tiles that contain at least one source are written, but for a **sparse
all-sky** catalog the number of files still grows quickly with order (often
approaching one file per object at very high order).

**How to choose**

1. **Target rows per shard** — for TB-scale catalogs, aim for roughly **10⁴–10⁵
   rows per non-empty** `Npix=*.parquet` file. Too many tiny files → metadata and
   filesystem overhead (e.g. multi‑MB on-disk size from a 2 MB FITS when order is
   too high); too few huge tiles → slow append and heavy single-tile RAM.

2. **Survey density, not survey name** — use a **lower** order for sparse
   all-sky tables; keep **5** for dense survey footprints (DESI, deep drills)
   and Rubin/LSST **lsdb** interoperability.

3. **Independent orders per modality** — set `--norder` separately for catalog,
   spectra, and cutouts. The catalog stores
   `_spectrum_npix` / `_cutout_npix` (modality-specific tile pixel) alongside
   `_healpix_norder{N}` (catalog partition key) so accessors can open the right
   Zarr tile regardless of order differences. Linkage uses `_source_id`, not
   positional HEALPix equality. Tile-aligned **cross-survey** join
   (`crossmatch.py`) does not require identical catalog orders on both sides.

4. **Benchmark before TB ingests** — use **`dl-recommend-catalog-norder`** (quick
   FITS scan of RA/Dec only) or ingest a subset, then check tile counts and
   `du -sh catalogs/<survey>`.

| Catalog type | Typical `--norder` | Why |
|--------------|-------------------|-----|
| Sparse all-sky (10⁶–10⁷ rows, full sphere) | **3–4** | Fewer, larger Parquet tiles; smaller inode footprint |
| Dense survey footprint (DESI, deep fields) | **5** (default) | ~3.4 deg² tiles; matches common HATS/LSST practice |
| Very local, high density | **6–7** | Only if tiles still hold many rows per file |

Override the deployment default per run:

```bash
dl-ingest-catalog sparse_allsky.fits --survey my_sparse --norder 4 \
  --ra-col RA --dec-col DEC --link-id-col ID --streaming
```

**Pre-ingest norder scan** (reads FITS headers + a subsample of RA/Dec; no Parquet write):

```bash
# One file or directory of FITS
dl-recommend-catalog-norder /path/to/catalogs/*.fits \
  --ra-col TARGET_RA --dec-col TARGET_DEC

# File list (same paths as batch ingest)
dl-recommend-catalog-norder --file-list allwise_files.txt \
  --ra-col ra --dec-col dec --max-files 32 --sample-rows 500000
```

Prints a table of candidate orders with estimated **rows/tile**, **tile count**, and
**pixel_sky_frac**, and highlights a recommended `--norder` near 10⁴–10⁵ rows per
occupied pixel (default target 50 000). Re-run with more `--sample-rows` for large,
clustered footprints.

#### Link identifiers (`--link-id-col`)

The lake uses a single internal join column ``_source_id`` (int64) in Parquet
catalogs and Zarr ``_source_id/`` arrays. Survey-native columns (e.g. ultraVISTA
``SOURCE_ID``, DESI ``TARGETID``) are kept unchanged. Anything starting with ``_``
is lake-owned bookkeeping (like ``_healpix_norder*``, ``_cutout_index``,
``_spectrum_index``).

Catalog ingest records ``link_id_mode`` in ``catalog_info.json``; the join
column is always ``link_id_column: "_source_id"``. When you pass
``--link-id-col``, that native column is also stored as ``native_id_column``.

| Input type | Example | Parquet columns | ``link_id_mode`` |
|------------|---------|-----------------|---------------------|
| Integer column | DESI ``TARGETID`` | ``TARGETID`` (int64) + ``_source_id`` (same values) | ``column:TARGETID`` |
| Unsigned / uint64 column | SDSS ``objid`` (> ``2**63-1``) | Native cast + ``_source_id`` | ``column:objid`` |
| Decimal string column | ``"39627658462934656"`` in FITS ASCII | Parsed native + ``_source_id`` | ``column:TARGETID`` |
| Vector ID column | SDSS ``OBJID`` shape ``(5,)`` | **Error** — use scalar ``objid`` | — |
| Alphanumeric labels | ``J000000.00-314627.5`` in ``NAME`` | ``NAME`` kept; ``_source_id`` = stable hash | ``label:NAME`` |
| Composite labels | ``targetname`` + ``obsid_v`` + ``obsid_r`` (6dF) | Columns kept; ``_source_id`` = hash of ``target\|obsid_v\|obsid_r`` | ``composite:targetname,obsid_v,obsid_r`` |
| Composite with header (2dF) | ``SPFILE`` + ``FIBRE`` per extension | Columns kept; ``_source_id`` = hash of ``spfile\|fibre`` | ``composite:SPFILE,FIBRE`` |

**Required:** ``--link-id-col`` must name an existing catalog column (or comma-separated composite). Auto-inference and sequential ``_source_id`` are not supported.

**Sky coordinates:** every row's ``--ra-col`` / ``--dec-col`` values must pass ``is_valid_sky_position`` (finite, Dec in [-90, 90], not SDSS-style sentinels ≤ −9000). Invalid rows abort ingest before HEALPix assignment.

**Incomplete composite IDs:** If a catalog row is missing one or more parts of a composite or label link key (e.g. the `SPFILE` column is blank for some rows), use ``--allow-incomplete-link-id``. Affected rows keep all science columns but ``_source_id`` is set to ``null`` and ``_spectrum_index`` stays ``-1``. Without the flag the ingest aborts on the first missing part.

```bash
# Ingest 2dF catalog where some rows have no SPFILE
dl-ingest-catalog 2dF_cat.fits --survey 2DFGRS_DR3 \
  --link-id-col SPFILE,FIBRE --allow-incomplete-link-id

# Same flag is available on batch / repair commands
dl-ingest-catalog-batch --allow-incomplete-link-id ...
dl-repair-catalog-metadata /lake --survey 2DFGRS_DR3 \
  --rebuild-link-id SPFILE,FIBRE --allow-incomplete-link-id
```

**Whitespace — cell values:** leading and trailing spaces are stripped from
link-ID values before parsing or hashing (common for fixed-width FITS strings).
Internal spaces are preserved.

**Whitespace — column names:** leading and trailing spaces in FITS TTYPE keywords
(e.g. ``' dec'`` instead of ``'dec'``) are **stripped at ingest** so Parquet
column names always match the CLI flags you passed (``--ra-col``, ``--dec-col``).
If you have a survey that was ingested before this normalization, columns like
``' dec'`` may exist on disk and cause DuckDB query failures (``Referenced column
"dec" not found; Candidate bindings: " dec"``).

Fix with:

```bash
# See which columns need normalizing
dl-repair-catalog-metadata /data/lake --survey ALLWISE --check-padded-columns

# Rewrite tiles in-place and refresh metadata
dl-repair-catalog-metadata /data/lake --survey ALLWISE --normalize-column-names
```

**Alphanumeric labels:** the human-readable name stays in your column (e.g.
``NAME``). ``_source_id`` holds the deterministic hash (BLAKE2b → 64-bit signed int).

```python
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, normalize_object_id

label = "J000000.00-314627.5"
sid = normalize_object_id(label)   # same int64 as catalog _source_id / Zarr row
acc.get_spectrum(sid)
```

SQL on names: ``SELECT * FROM catalog WHERE NAME = 'J000000.00-314627.5'``.
Use the **same** spelling (after strip) in cutout/spectrum FITS headers via
``--link-id-col NAME`` so ingest hashes match the catalog.

For surveys where the join key spans multiple catalog columns (6dF ``targetname`` +
``obsid_v`` + ``obsid_r``), pass a comma-separated spec:
``--link-id-col targetname,obsid_v,obsid_r``.  All columns are preserved;
``_source_id`` is the stable hash of ``targetname|obsid_v|obsid_r`` (same string
built from spectrum filename stem + FITS ``OBSID_V`` on the V header and
``OBSID_R`` on the R header).

**Reassign catalog link column** (recompute ``_source_id`` from another column without FITS re-ingest; catalog Parquet only):

```bash
dl-repair-catalog-metadata /lake --survey zCOSMOS_DR3 --rebuild-link-id filename
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

Science columns (e.g. ``id``) are unchanged; ``_spectrum_index`` is reset and must be repatched.

