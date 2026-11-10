# lake_config.toml reference

`lake_config.toml` describes one lake deployment: where its data lives, its
default HEALPix order and storage settings, and ingest defaults. `dl-init`
creates it; you can edit it by hand afterwards. Command-line tools find it
automatically, so most commands do not need `--lake-root`.

```toml
schema_version = "1"                # data-lake config schema; bumped on breaking changes

[lake]
name        = "my_lake"             # short identifier (no spaces)
description = "Free-text"
root        = "/abs/path/to/data"   # where catalogs/spectra/cutouts live
created_utc = "2026-05-13T11:43:00Z"

[partitioning]
hats_order       = 5                # default HEALPix order (~3.36 deg²/tile at 5)
chunks_per_shard = 512              # rows per spectrum Zarr shard

[defaults]
wavelength_mode  = "shared"         # "shared" | "per_source"
mask_dtype       = "uint8"          # "uint8" | "uint16"
with_resolution  = false            # default for DESI spectra ingest

[paths]                             # relative to lake.root
catalogs = "catalogs"
products = "products"
spectra  = "spectra"
cutouts  = "cutouts"
shared   = "shared"

[ingest]
num_workers = "auto"                # "auto" (= cpu_count) | <int>
log_level   = "INFO"

[guardrails]
ingest_token_file = ".ingest_token_hash"   # SHA-256 of the ingest token; set via dl-init
```

Only `[lake]` with `name` and `root` is required; any other missing key takes
the default shown above. A relative `root` is resolved against the directory
that contains the config file. `schema_version` must be `"1"`; any other value
is rejected.

- `hats_order`: see
  [Choosing HEALPix order](ingest/catalog.md#choosing-healpix-order---norder--hats_order)
  in the catalog ingest guide.
- `chunks_per_shard`: matches `data_lake.knobs.SPECTRA_CHUNKS_PER_SHARD`.
  Cutout ingest uses `CUTOUT_CHUNKS_PER_SHARD` (1024) instead.
- `[paths]`: exposed to Python as `cfg.catalogs_root`, `cfg.products_root`
  and so on (see below). Derived catalogs from `dl-gather` and `dl-homogenize`
  go under `products/`.
- `ingest_token_file`: ingest commands that write to the lake check the ingest
  token (`--ingest-token` or `$LAKE_INGEST_TOKEN`) against the hash stored in
  this file. Set or change it with `dl-init` or `dl-set-ingest-token`.

## How the config is found

Highest priority first:

1. `--config /path/to/lake_config.toml` on the command line
2. The environment variable `$DATA_LAKE_CONFIG=/path/to/lake_config.toml`
3. The first `lake_config.toml` found walking up from the current working
   directory

From Python:

```python
from data_lake.config import LakeConfig
cfg = LakeConfig.discover()                # explicit > env > cwd walk-up
print(cfg.lake.name, cfg.spectra_root)
```

`LakeConfig.discover()` raises `LakeConfigNotFound` when no config is found.
`cfg.catalogs_root`, `cfg.products_root`, `cfg.spectra_root`,
`cfg.cutouts_root` and `cfg.shared_root` give absolute paths under the lake
root.

## Settings that are not in the config file

Operational defaults such as open-tile LRU sizes, partner and survey-B tile
caches, the DuckDB footer cache, ZSTD compression levels, FITS chunk rows and
worker caps live in [`data_lake/knobs.py`](../data_lake/knobs.py), not in
`lake_config.toml`. See
[Performance → Operational knobs](performance.md#operational-knobs-data_lakeknobs).
