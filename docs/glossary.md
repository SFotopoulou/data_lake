# Glossary

Short definitions of terms used throughout the data lake documentation. Links point to the canonical description in each topic guide.

---

**area**
A named logical region defined in `areas/<id>.json`. An area stores a sky region selector (cone, bbox, MOC, or npix list) plus optional plans for crossmatch, gather, and homogenization. Areas span surveys and modalities and contain no tile data.
See [Regions and areas](discovery/regions-and-areas.md).

**crossmatch tree**
A precomputed catalog-to-catalog positional match stored under `crossmatch/<surveyA>_x_<surveyB>__r<radius>/`. Created by `dl-crossmatch`. Used as input to `dl-gather`.
See [Crossmatch and associations](discovery/crossmatch.md).

**deployment / lake root**
A directory that contains a `lake_config.toml` and the `catalogs/`, `spectra/`, `cutouts/`, `shared/` tree. One person or team runs one deployment; multiple deployments can co-exist on the same machine.
See [Step 3 in the quickstart](quickstart.md#step-3--initialise-your-lake-dl-init).

**gather**
The process of materialising a joined "wide" product catalog from one base survey and one or more partner surveys connected via crossmatch tiles. Run with `dl-gather`; output has `kind: product`, `product_subtype: joined`.
See [Gather](discovery/gather.md).

**HATS (HEALPix Adaptive Tiling Scheme)**
The partitioning scheme used for catalog Parquet tiles. Each tile is a HEALPix pixel at a chosen `norder` that contains all sources whose sky position falls within that pixel. HATS enables fast spatial queries and incremental ingest.
See [hats.readthedocs.io](https://hats.readthedocs.io/en/stable/).

**HEALPix / norder**
HEALPix (*Hierarchical Equal Area isoLatitude Pixelisation*) divides the sphere into equal-area pixels. `norder` (or `hats_order`) determines the resolution: `norder=5` → 12 288 pixels, `norder=7` → 196 608 pixels. Higher norder means smaller pixels and more tiles.

**homogenize / product catalog (homogenized)**
The process of converting native survey columns (Vega mags, Jy fluxes, …) to a standardised system (AB magnitudes, uniform flux units) using versioned transform recipes. Run with `dl-homogenize`; output has `kind: product`, `product_subtype: homogenized`.
See [Homogenization](homogenization.md).

**ingest token**
A plaintext secret passed via `$LAKE_INGEST_TOKEN` that every `dl-ingest-*` command checks before writing. The lake stores only the SHA-256 hash. Provides a lightweight "I am the operator" gate; real security is filesystem ACLs.

**modality**
One of the three data types stored by the lake: **catalog** (Parquet, `catalogs/`), **spectra** (Zarr, `spectra/`), or **cutout** (Zarr, `cutouts/`). A survey may have data in one or more modalities; ingest commands target a single modality at a time.

**MOC (Multi-Order Coverage map)**
An IVOA standard for representing arbitrary sky regions as a hierarchical set of HEALPix cells. Used by `dl-region --export-moc` and `dl-export-moc`. Formats: FITS, JSON, ASCII.
See [Regions and areas](discovery/regions-and-areas.md#export-as-moc).

**`_source_id`**
A stable int64 join key assigned to every object at catalog ingest time. It links a catalog row to its spectrum tile row (`_spectrum_index`) and cutout tile row (`_cutout_index`). It can be a copy of a native integer ID column (`--link-id-col TARGETID`) or a hash of a string label. Once written, `_source_id` values never change for a given source.
See [data-on-disk.md](layout/data-on-disk.md) for the full column set.

**native ID vs `_source_id`**
The *native ID* is the survey's own identifier (e.g. DESI `TARGETID`, SDSS `SPECOBJID`). The `_source_id` is the lake's internal join key. When `--link-id-col` names the native integer column, the two are the same value; otherwise `_source_id` is a sequential counter or a hash.

**region**
A sky geometry used to select data: cone (`--cone RA DEC RADIUS`), bounding box (`--bbox`), HEALPix pixel(s) (`--npix`), or a MOC file (`--moc`). Used by `dl-region`, `dl-gather`, `dl-crossmatch`, and `dl-homogenize`.
See [Regions and areas](discovery/regions-and-areas.md).
