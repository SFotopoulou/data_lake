# Glossary

Short definitions of terms used throughout the data lake documentation. Each entry links to the guide that covers it in full.

---

**area**
A named region saved in `areas/<id>.json`. An area stores a sky region selector (cone, bounding box, MOC or HEALPix pixel list), plus optional plans for crossmatch, gather and homogenization. Areas can span surveys and modalities, and they hold no tile data themselves.
See [Regions and areas](discovery/regions-and-areas.md).

**`_common_metadata`**
A small Parquet file (a few KiB) in each catalog directory that holds only the schema: column names and types. Readers use it to learn a catalog's columns without opening a tile. It is written by `dl-finalize-catalog`, which runs at the end of every non-deferred ingest, alongside `_metadata` (the aggregate footer of all tiles). Older catalogs without it still work, because readers fall back to one tile's schema. Run `dl-finalize-catalog` once to add it. Joined products from `dl-gather` contain only tiles and `catalog_info.json`.
See [Data layout on disk](layout/data-on-disk.md).

**crossmatch tree**
A precomputed positional match between two catalogs, stored under `crossmatch/<surveyA>_x_<surveyB>__r<radius>/`. It is created by `dl-crossmatch` and used as input to `dl-gather`.
See [Crossmatch and associations](discovery/crossmatch.md).

**deployment / lake root**
A directory that contains a `lake_config.toml` and the `catalogs/`, `spectra/`, `cutouts/` and `shared/` trees. One person or team runs one deployment, and several deployments can exist on the same machine.
See [Step 3 in the quickstart](quickstart.md#step-3--initialise-your-lake-dl-init).

**gather**
Building a joined "wide" product catalog from one base survey and one or more partner surveys, linked through crossmatch tiles. Run it with `dl-gather`. The output has `kind: product` and `product_subtype: joined`.
See [Gather](discovery/gather.md).

**HATS (HEALPix Adaptive Tiling Scheme)**
The partitioning scheme for catalog Parquet tiles. Each tile is one HEALPix pixel at a chosen `norder` and holds every source whose position falls inside that pixel. This makes spatial queries fast and allows incremental ingest.
See [hats.readthedocs.io](https://hats.readthedocs.io/en/stable/).

**HEALPix / norder**
HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) divides the sphere into equal-area pixels. `norder` (also `hats_order`) sets the resolution: the sky has \(12 \times 4^{\text{norder}}\) pixels, so `norder=5` gives 12,288 pixels and `norder=7` gives 196,608. A higher norder means smaller pixels and more tiles. The default catalog norder is 5.

**homogenize / product catalog (homogenized)**
Converting native survey columns (Vega magnitudes, Jy fluxes and so on) to a common system, such as AB magnitudes or uniform flux units, using versioned per-survey recipes. Run it with `dl-homogenize`. The output has `kind: product` and `product_subtype: homogenized`.
See [Homogenization](homogenization.md).

**ingest token**
A plaintext secret, passed in `$LAKE_INGEST_TOKEN` (or `--ingest-token`), that `dl-ingest-*` and other write commands check before writing. The lake stores only its SHA-256 hash. It is a lightweight "I am the operator" check. Real protection comes from filesystem permissions.

**`locate()`**
A method on `CutoutAccessor` and `SpectrumAccessor` that resolves many `_source_id` values in one pass. It returns two arrays, `npix` (the tile) and `local` (the row within the tile), with `-1` for IDs that were not found. Pass those positions to `get_image_at` / `get_cutout_at` / `get_spectrum_at(npix, local)` to read data. For bulk work this is much faster than calling `get_cutout(id)` or `get_spectrum(id)` once per source, which you can still use for one-off reads.

**modality**
One of the three data types stored in the lake: catalog (Parquet, `catalogs/`), spectra (Zarr, `spectra/`) and cutout (Zarr, `cutouts/`). A survey can have data in one or more modalities, and each ingest command targets one modality.

**MOC (Multi-Order Coverage map)**
An International Virtual Observatory Alliance (IVOA) standard that describes any sky region as a hierarchical set of HEALPix cells. It is used by `dl-region --export-moc` and `dl-export-moc`, which can write FITS, JSON or ASCII.
See [Regions and areas](discovery/regions-and-areas.md#export-as-moc).

**`_source_id`**
A stable int64 join key assigned to every object at catalog ingest. It links a catalog row to its spectrum (`_spectrum_index`) and cutout (`_cutout_index`) tile positions. Catalog ingest requires `--link-id-col`. If that column is a native integer ID, `_source_id` is a copy of it. If it is a string label or a comma-separated composite of columns, `_source_id` is a stable hash of the label. Once written, a source's `_source_id` does not change.
See [data-on-disk.md](layout/data-on-disk.md) for the full column set.

**native ID vs `_source_id`**
The native ID is the survey's own identifier, for example DESI `TARGETID` or SDSS `SPECOBJID`. The `_source_id` is the lake's internal join key. When `--link-id-col` names a native integer column, the two have the same value. Otherwise `_source_id` is a hash of the native label. Catalogs ingested by older versions may use a sequential counter instead, and `catalog_info.json` records which mode was used in `link_id_mode`.

**region**
A sky geometry used to select data: a cone (`--cone RA DEC --radius-arcsec R`), a bounding box (`--bbox RA_MIN RA_MAX DEC_MIN DEC_MAX`), HEALPix pixels (`--npix … --norder N`) or a MOC file (`--moc`, `dl-region` only; needs the `moc` extra). `dl-region`, `dl-gather` and `dl-homogenize` accept these selectors directly. `dl-crossmatch` takes a region through `--from-area`.
See [Regions and areas](discovery/regions-and-areas.md).
