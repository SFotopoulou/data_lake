# Lake registry and overlays

### Lake registry and master metadata (P1)

**Lake index** — scan what is deployed:

```bash
dl-refresh-lake-registry              # write shared/registry/surveys.parquet
dl-describe-lake                      # print survey × modality summary
dl-describe-lake --modality catalog   # catalogs only (or spectra / cutout)
dl-describe-lake --count-total        # footer: per-modality totals + grand total (registry sums)
dl-describe-lake --modality catalog --count-total
dl-describe-lake --json               # {"entries": [...]} per survey × modality
dl-describe-lake --json --count-total # entries + summary object
dl-describe-lake --json --pair-surveys  # entries + catalog/spectra hats_order pairing
dl-describe-lake --verbose            # tiles, ingest sidecars, meta_fields, spectrum_sky_meta, path, …
dl-describe-lake --pair-surveys       # footer: catalog vs spectra hats_order per survey name
dl-describe-lake --refresh            # rebuild registry from disk first, then print
```

**`--count-total` does not rescan tiles.** Counts are read from the registry (`surveys.parquet`), which stores row counts recorded at ingest time. If you have ingested new data since the last `dl-refresh-lake-registry`, run `dl-describe-lake --refresh --count-total` to get up-to-date numbers.

The default table adds **sky** (`ra_column`/`dec_column`), a **detail** column (`link_id_mode` for catalogs; `n_pix` + `wavelength_mode` for spectra; band stack shape for cutouts), and **manifest** (`Y`/`·`). Full registry fields (including `link_id_mode`, `native_id_column`, `n_tiles`, ingest checkpoint flags, `hats_order_match`, `created_utc`, …) are in `shared/registry/surveys.parquet` and `--json`.

| Question | Registry field | Also see |
|----------|----------------|----------|
| How are catalog IDs defined? | `link_id_mode`, `native_id_column` | `catalog_info.json`, `dl-describe-survey` |
| Sky columns for joins / crossmatch | `ra_column`, `dec_column` | `dl-extract-catalog`, STILTS |
| Catalog vs spectra HEALPix order | `hats_order_match`, `--pair-surveys` | `dl-validate-catalog-spectra-link` |
| Spectrum pixel width / wavelength layout | `n_pix`, `wavelength_mode` | `spectrum_info.json` |
| Per-spectrum Zarr meta (redshift, sky provenance, …) | `meta_fields`, `spectrum_sky_meta_fields`, `has_spectrum_sky_meta` | `dl-describe-lake` detail/`--verbose`; `dl-describe-survey NAME --modality spectra` lists `meta.*` columns with dtypes and `meta.sky` group |
| Ingest still running? | `has_ingest_checkpoint`, `has_ingest_inflight` | survey `.ingest_*.json` sidecars |

For **spectra**, ``total_rows`` in the registry is the sum of ``source_id`` lengths
across all ``Npix=*.zarr`` tiles (same count as the ingestion report notebook’s
``n_spectra_in_zarr``). Catalog rows use ``catalog_info.json`` ``total_rows``.

**Master association** — map master ID columns to catalog join keys. Keep the
master Parquet thin; store mapping in a sidecar ``<master>.meta.json``:

```json
{
  "meta_version": "1",
  "master_parquet": "associations/master_desi_euclid.parquet",
  "primary_survey": "DESI_DR1",
  "partners": [
    {"survey": "DESI_DR1", "master_column": "desi_targetid", "catalog_id_column": "TARGETID"},
    {"survey": "EUCLID_DR1", "master_column": "euclid_source_id", "catalog_id_column": "SOURCE_ID"}
  ]
}
```

```bash
dl-describe-master associations/master.parquet
dl-describe-master associations/master.parquet --write-meta   # save guessed .meta.json
```

If ``.meta.json`` is missing, columns are matched heuristically against on-disk
``schema_manifest.json`` files (run ``dl-describe-survey <name> --rebuild`` first
for surveys without a manifest).

### Spectra, cutouts, and column overlays (P2)

**Spectra and cutout layers** get ``schema_manifest.json`` at ingest finalize
(from ``spectrum_info.json`` / ``cutout_info.json``): Zarr array names, dtypes,
roles (`flux`, `ivar`, `mask`, `wavelength`, `image`, `metadata`, …).

```bash
dl-describe-survey DESI_DR1 --modality catalog    # default
dl-describe-survey DESI_DR1 --modality spectra
dl-describe-survey LSST_DR1 --modality cutout
dl-describe-survey ALLWISE --rebuild
```

**Optional overlays** — analyst JSON under ``shared/registry/overlays/``
(see ``shared/registry/overlays/README.md``). Merged at describe time with
``unit``, ``description``, and homogenization hints (e.g. WISE Vega → AB offset).

**Discovery workflow** (master → columns → SQL):

1. ``dl-refresh-lake-registry`` then ``dl-describe-lake``
2. ``dl-describe-master associations/master.parquet`` (``--write-meta`` once)
3. ``dl-describe-survey <partner>`` for each catalog; ``--modality spectra`` if needed
4. DuckDB ``want → master → catalog`` (``notebooks/11_duckdb_catalog_query.ipynb`` §9–§10)

