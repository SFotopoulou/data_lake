"""Shared operational defaults (phase-1 consistency).

These are the single source of truth for duplicated magic numbers. Lake-config
overrides land in a later knobs-config pass; until then callers import from here.

Intentional asymmetries (do not "unify" without a design pass)
--------------------------------------------------------------
* **Read vs write open Zarr tiles** — extract/query accessors keep few tiles open
  (``READ_MAX_OPEN_TILES``) so ``ulimit -n`` stays safe on multi-thousand-tile
  extracts. Parallel *ingest writers* keep more tiles open
  (``WRITE_MAX_OPEN_TILES``) because append/widen amortizes better and writer
  processes typically touch fewer distinct tiles concurrently.
* **Spectra vs cutout shards** — spectra use smaller row shards
  (``SPECTRA_CHUNKS_PER_SHARD``); cutout image stacks use larger ones
  (``CUTOUT_CHUNKS_PER_SHARD``) because each row is a multi-band postage stamp.
* **ZSTD tiers** — ingest/crossmatch favour speed (``ZSTD_INGEST``); gathered
  products favour size (``ZSTD_GATHER``); ``compact()`` uses ``ZSTD_COMPACT``.
* **CLI ID / sky columns** — catalog ingest defaults to ``ra``/``dec``; spectra
  subset defaults ``--target-id-col TARGETID`` (native survey lists); cutout
  subset defaults ``_source_id`` (lake join key). See docs/cli-reference.md.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Open Zarr tile LRUs (file descriptors + metadata)
# ---------------------------------------------------------------------------

READ_MAX_OPEN_TILES: int = 8
"""SpectrumAccessor / CutoutAccessor default (extract & query)."""

WRITE_MAX_OPEN_TILES: int = 64
"""Parallel ingest writers (DESI / spectra / spplate / cutouts). 0 = unlimited."""

# ---------------------------------------------------------------------------
# Partner / survey-B tile frame caches (Parquet → Polars in RAM)
# ---------------------------------------------------------------------------

PARTNER_TILE_CACHE_MAX_TILES: int = 48
PARTNER_TILE_CACHE_MAX_BYTES: int = 512 * 1024 * 1024  # 512 MiB
"""Shared by sky crossmatch B-cache, column-XM B-cache, and dl-gather partner cache."""

# ---------------------------------------------------------------------------
# DuckDB per-worker limits (see data_lake.io.duckdb_limits)
# ---------------------------------------------------------------------------

DUCKDB_WORKER_MEMORY_FRACTION: float = 0.25
"""Share of each worker's memory budget given to DuckDB; the rest is left for
Polars frames, partner caches, and the matcher."""

# ---------------------------------------------------------------------------
# Zarr shard sizing
# ---------------------------------------------------------------------------

SPECTRA_CHUNKS_PER_SHARD: int = 512
"""Rows per Zarr shard for spectrum tiles (~512 MB class for typical N_pix)."""

CUTOUT_CHUNKS_PER_SHARD: int = 1024
"""Rows per Zarr shard for cutout image stacks (larger footprint per row)."""

# Lake config ``partitioning.chunks_per_shard`` historically matches spectra.
DEFAULT_CHUNKS_PER_SHARD: int = SPECTRA_CHUNKS_PER_SHARD

# ---------------------------------------------------------------------------
# Parquet ZSTD compression tiers
# ---------------------------------------------------------------------------

ZSTD_INGEST: int = 3
"""Catalog ingest, crossmatch tiles, Zarr Blosc-zstd companion level."""

ZSTD_GATHER: int = 7
"""dl-gather / product Parquet (denser, slower write)."""

ZSTD_COMPACT: int = 9
"""``CatalogParquetOptions.compact()`` / offline re-encode."""

# ---------------------------------------------------------------------------
# FITS catalog export (see also export.to_spectrum_fits)
# ---------------------------------------------------------------------------

# Re-exported name kept stable for callers; value lives with FITS module too.
DEFAULT_FITS_CHUNK_ROWS: int = 50_000

# ---------------------------------------------------------------------------
# Misc caps reused in multiple places
# ---------------------------------------------------------------------------

MAX_PARALLEL_WORKERS_CAP: int = 64
"""Hard ceiling for homogenize / discovery thread or process pools."""

CROSSMATCH_DENSE_TILE_WARN: int = 50_000
"""Warn when a survey-A tile has at least this many sources."""

DEFAULT_HATS_ORDER: int = 5
"""Fallback HEALPix order when catalog/spectrum info is missing."""
