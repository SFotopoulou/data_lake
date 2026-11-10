# Troubleshooting

Look up an error message or symptom here to find its usual cause and the
command that fixes it. Each row links the symptom to one action; follow the
links at the bottom for background on catalog–spectrum linkage and
duplicate handling.

## Ingest and catalog metadata

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Catalog ingest very slow, or tqdm progress stuck at 0/N | The FITS file is **packed-vector** (STILTS colfits, `NAXIS2=1`) | Run `dl-check-fits-table-format`; re-export as row-normal FITS or Parquet ([details](ingest/catalog.md#check-fits-layout-before-ingest)) |
| Out of memory (OOM) on a catalog batch | `n_workers × largest file` exceeds RAM | Lower `--n-workers`; use `--columns` on wide surveys |
| `BrokenProcessPool` on a catalog batch | A worker ran out of memory | Same as above; see [Parallel catalog batch](ingest/batch-and-checkpoints.md#parallel-catalog-batch-large-file-lists) |
| Batch job killed mid-run; catalog tiles exist but `_metadata` / `schema_manifest.json` are missing | The job ended before `dl-finalize-catalog` ran (or the ingest used `--defer-finalize`) | `dl-finalize-catalog --survey NAME`, then `dl-refresh-lake-registry` |
| Opening a catalog or resolving columns is slow (seconds) | Older catalog lacks `_common_metadata`, so readers fall back to a large `_metadata` or a tile | `dl-finalize-catalog --survey NAME` once; see [Data on disk](layout/data-on-disk.md) |
| Row count differs between the ingest log and `dl-describe-survey` | The ingest log sums **input** rows; `dl-describe-survey` counts rows **on disk** after de-duplication | Compare with the pre-check source total; expect fewer rows with `--on-duplicate-id skip` (the default) |
| `Referenced column "dec" not found; Candidate bindings: " dec"` in a crossmatch or query | The survey's FITS `TTYPE` column names were stored with padding | `dl-repair-catalog-metadata /data/lake --survey ALLWISE --normalize-column-names` |
| `composite_link_label` ID does not match the catalog | Only some parts of a composite key are present (for example `SPFILE` without `FIBRE`) | Add `--allow-incomplete-link-id` to the ingest or repair command; those rows get a null `_source_id` |

## Catalog ↔ spectrum linkage

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `_spectrum_index` is all `-1` after ingest | `_source_id` differs between catalog and spectra (different link key) | Check `--link-id-col` on the catalog against the spectrum format; run `dl-validate-catalog-spectra-link --survey NAME` |
| `_spectrum_index` is all `-1` after a repair | Spectra not ingested yet, or indices not rebuilt (for example after `--rebuild-link-id`) | `dl-rebuild-catalog-indices --survey NAME --kind spectrum` |
| spPlate `_spectrum_index` mismatch | Wrong `specObjID` layout (DR7 vs DR8+) | Set `--specobj-id-layout auto\|dr7\|dr8plus`; see [SDSS spPlate ingest](ingest/spectra.md#sdss-spplate-ingest-640-or-1000-fibers-per-file) |

See [Validate and repair](validate-and-repair.md) for the full check-and-fix
workflow.

## Registry

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `dl-describe-lake` shows no entries | The registry file is missing | Run `dl-refresh-lake-registry` first |
| Registry row counts are stale | Registry not refreshed since the last ingest | `dl-describe-lake --refresh --count-total` |
| Column-crossmatch tree missing, or its detail shows `sky` | Registry built before the tree existed, or without the crossmatch fields | `dl-describe-lake --refresh`; look for `__col_<col_a>__<col_b>` under modality `crossmatch` (detail shows `col:<col_a>:<col_b>`) |

## Crossmatch, gather and homogenize

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `dl-crossmatch` worker killed / `BrokenProcessPool` | Workers together exceed the job's memory (Slurm or cgroup limit) | Lower `--n-workers`, lower `--b-cache-mib`, or set `--memory-budget-gb` to the memory you actually have; see [Performance](performance.md#crossmatch) |
| Crossmatch finishes but some workers were idle (summed phase time ÷ workers ≪ wall time) | Uneven sky density; too few work strips | Raise `--strips-per-worker` (default 4); see [Work strips](discovery/crossmatch.md#work-strips) |
| `get_spectrum` / `get_cutout` is slow in a loop over many IDs | Each call looks up one ID | Call `locate(ids)` once, then `get_spectrum_at` / `get_cutout_at`; or use `CutoutDataset` / `SpectrumDataset` ([Performance](performance.md#retrieval)) |
| `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | A Python script starts worker processes (crossmatch, or a PyTorch `DataLoader` with `num_workers > 0`) without a main guard | Put the calling code under `if __name__ == "__main__":` |
| `dl-homogenize --from-product` fails with `AppendRowGroups requires equal schemas` | Sparse gather tiles had different columns or dtypes, and the product was homogenized by a version that wrote unequal Parquet schemas | Re-run `dl-homogenize … --overwrite`; sparse partner bands become null AB-magnitude columns with a shared schema |
| Derived products are under `catalogs/` instead of `products/` | The products were generated by an older version | They are still readable (the resolver checks `catalogs/` first). To move them, delete or rename `catalogs/<name>` and re-run `dl-gather` / `dl-homogenize --overwrite`; writing to `products/<name>` is refused while `catalogs/<name>` exists |

For catalog–spectrum linkage issues, see [Catalog vs spectrum CLI flags](ingest/spectra.md#catalog-vs-spectrum-cli-flags) and [1-D spectrum readers reference](ingest/spectra.md#1-d-spectrum-readers-reference). For duplicate/resume logic, see [Duplicate / resume flags by command](ingest/batch-and-checkpoints.md#duplicate--resume-flags-by-command).
