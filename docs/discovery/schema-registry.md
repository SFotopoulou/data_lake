# Schema registry

### Schema registry (column discovery)

Each ingested catalog gets **`catalogs/<survey>/schema_manifest.json`** at finalize
(ingest or batch end): every column name, Arrow dtype, heuristic **role**
(`id`, `sky`, `redshift`, `photometry`, `healpix`, `index`, …), join columns, and
prefix **column_groups** (e.g. `MAG` → `MAG_G`, `MAG_R`).

```bash
dl-describe-survey DESI_DR1
dl-describe-survey EUCLID_DR1 --role photometry
dl-describe-survey DESI_DR1 --modality spectra
dl-describe-survey ALLWISE --rebuild   # refresh manifest from on-disk data
dl-describe-survey DESI_DR1 --json     # full manifest for tooling
```

Use this to pick columns before joining a **master association** table (see below).
Spectra/cutout manifests are written at ingest finalize; catalogs also get manifests from Parquet schema.
The master file should stay ID-centric; science columns come from per-survey catalogs.

