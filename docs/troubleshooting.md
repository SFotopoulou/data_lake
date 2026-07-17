# Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `dl-homogenize --from-product` fails with `AppendRowGroups requires equal schemas` | Sparse gather tiles had different columns/dtypes; older homogenize wrote unequal Parquet schemas | Upgrade and re-run `dl-homogenize … --overwrite`; sparse partner bands become null AB columns with a shared schema |
| `_spectrum_index` all -1 after ingest | `_source_id` mismatch between catalog and spectra (different link key) | Check `--link-id-col` on catalog vs spectrum format; run `dl-validate-catalog-spectra-link --survey NAME` |
| `_spectrum_index` all -1 after repair | Spectra not ingested yet, or `dl-rebuild-catalog-indices` not run | Run `dl-rebuild-catalog-indices --survey NAME --kind spectrum` |
| Registry row counts stale | Registry not refreshed since last ingest | `dl-describe-lake --refresh --count-total` |
| Column crossmatch tree missing / detail shows `sky` | Registry built before the tree existed, or crossmatch fields dropped from registry schema | `dl-describe-lake --refresh`; look for `__col_<col_a>__<col_b>` under modality `crossmatch` (`detail` shows `col:<col_a>:<col_b>`) |
| Batch job killed mid-run, catalog tiles exist but no manifest | `dl-finalize-catalog` not run | `dl-finalize-catalog --survey NAME` then `dl-refresh-lake-registry` |
| Catalog ingest very slow / tqdm stuck at 0/N | FITS is **packed-vector** (STILTS colfits, `NAXIS2=1`) | Run `dl-check-fits-table-format`; re-export row-normal FITS or Parquet |
| `Referenced column "dec" not found; Candidate bindings: " dec"` during crossmatch or query | Survey was ingested with padded FITS TTYPE column names before automatic stripping was added | Run `dl-repair-catalog-metadata /data/lake --survey ALLWISE --normalize-column-names` |
| Row count mismatch (ingest log vs `dl-describe-survey`) | Ingest log sums **input** rows; describe counts **on-disk** after dedup | Compare to pre-check total sources; expect lower counts with `--on-duplicate-id skip` |
| OOM on catalog batch | `n_workers × largest file` exceeds RAM | Lower `--n-workers`; use `--columns` on wide surveys |
| `BrokenProcessPool` on catalog batch | Worker OOM | Same — reduce workers; see parallel batch note in README |
| `composite_link_label` ID does not match catalog | Partial parts — e.g. only `SPFILE` present, `FIBRE` absent | Add `--allow-incomplete-link-id` to ingest/repair; those rows get null `_source_id` |
| `dl-describe-lake` shows no entries | Registry file missing | `dl-refresh-lake-registry` first |
| spPlate `_spectrum_index` mismatch | Wrong specObjID layout (DR7 vs DR8+) | Set `--specobj-id-layout auto\|dr7\|dr8plus`; see spPlate section |

For catalog–spectrum linkage issues, see [Catalog vs spectrum CLI flags](ingest/spectra.md#catalog-vs-spectrum-cli-flags) and [1-D spectrum readers reference](ingest/spectra.md#1-d-spectrum-readers-reference). For duplicate/resume logic, see [Duplicate / resume flags by command](ingest/batch-and-checkpoints.md#duplicate--resume-flags-by-command).

