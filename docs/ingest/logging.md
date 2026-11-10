# Logging and long runs

This page explains how to control log output from `dl-*` commands and how to keep an audit
trail during long ingest runs. Read it before starting a multi-hour job, where per-file
INFO lines can flood the terminal and hide overall progress.

For most long runs, use `-q --log-file PATH`: the terminal shows only warnings (plus a
periodic heartbeat on file-list commands), and the file keeps the full INFO log.

```bash
dl-ingest-spectra-from-list gama.txt /data/lake \
    --survey GAMA_DR4 --fmt gama \
    -q --log-file /data/lake/spectra/GAMA_DR4/.ingest.log
```

## Verbosity levels

Every `dl-*` command accepts `--log-file` (plus `-q`/`--quiet` and, where it does not
clash with a product flag, `-v`/`--verbose`) via the shared `logging_options` helper.
Most commands log at INFO to stderr by default. Two exceptions: the DESI and spPlate
batch commands write logs only to a file (see
[Parallel worker logs](#parallel-worker-logs)), and `dl-recommend-catalog-norder` defaults
to WARNING.

| Mode | Flags | Terminal (stderr) | Optional `--log-file` |
|------|-------|-------------------|------------------------|
| **Normal** (default) | (none) | INFO | INFO if `--log-file` set |
| **Quiet** | `-q` / `--quiet` | WARNING + heartbeat | INFO (full audit trail) |
| **Verbose** | `-v` / `--verbose` | DEBUG | DEBUG |

`-q` and `-v` are mutually exclusive; passing both is an error. The heartbeat appears only
on the commands listed under [Heartbeat summaries](#heartbeat-summaries). The tqdm progress
bar is separate from logging; turn it off with `--no-progress` where the command offers it.

### `dl-describe-lake --verbose` is not DEBUG logging

On most CLIs, `-v` / `--verbose` means DEBUG log level. **`dl-describe-lake` is different:**
`--verbose` only widens the printed registry table (extra columns such as path, tile
counts, ingest sidecars). That command does **not** expose a logging `-v`; use `-q` /
`--log-file` for logging control. See [Lake registry](../discovery/registry.md).

The terminal level is resolved in this order: the `-q` / `-v` flag, then `log_level` in the
`[ingest]` section of `lake_config.toml`, then INFO.

## Heartbeat summaries

The three file-list commands (`dl-ingest-catalog-from-list`, `dl-ingest-cutouts-from-list`,
`dl-ingest-spectra-from-list`) can print a one-line progress summary to stderr at a fixed
interval when they run sequentially (`--n-workers 1`, the default). With `-q` the interval
is 300 seconds (5 minutes); without `-q` the heartbeat is off. `--heartbeat-interval SEC`
sets the interval and also turns the heartbeat on without `-q`.

```
[spectra-ingest] 12000/50000 files (24%), ETA ~4.2h, 1,230,000 spectra, 2 failed, 2.10 files/s (37.2 min elapsed)
```

The heartbeat is printed even while the tqdm progress bar is active, so shell history and
Slurm output files keep time-stamped progress points. A final summary is printed when the
run ends.

## Recommended patterns for long runs

**12-hour spPlate run (parallel batch):**

```bash
# Full INFO log in the file; -q also hides the progress bar
dl-ingest-spectra-batch-spplate /data/lake \
    --survey sdss_dr17 \
    --file-list plates.txt \
    --n-workers 8 \
    -q --log-file /data/lake/spectra/sdss_dr17/.ingest.log
```

**Sequential file-list run (GAMA / single object per file):**

```bash
dl-ingest-spectra-from-list gama.txt /data/lake \
    --survey GAMA_DR4 --fmt gama \
    -q --log-file /data/lake/spectra/GAMA_DR4/.ingest.log \
    --heartbeat-interval 120
```

**Map-reduce catalog ingest (Slurm array, one log per task):**

```bash
dl-ingest-from-shard --phase map --file-list "$CHUNK" \
    --survey Legacy_DR11_north --shard-root "$SCRATCH/shards" \
    --link-id-col ls_id_dr11 --norder 7 --n-workers 8 \
    --no-progress -q \
    --log-file "logs/map_${SLURM_ARRAY_TASK_ID}.log" \
    --failures-log "logs/map_${SLURM_ARRAY_TASK_ID}_failures.jsonl"

# Failed inputs across all tasks
cat logs/map_*_failures.jsonl | jq -r .path | sort -u
```

Use a separate log file per array task, because concurrent tasks appending to one file can
interleave lines. Reduce failure lines carry `npix` and `parts` instead of `path`. See
[Two-phase map-reduce ingest](batch-and-checkpoints.md#two-phase-map-reduce-ingest-dl-ingest-from-shard).

## Config-file default

To make quiet the default for every command in a deployment, set the level in
`lake_config.toml`:

```toml
[ingest]
log_level = "WARNING"   # quiet by default; -v to override at runtime
```

This setting changes the terminal level only. Unlike `-q`, it does not force INFO into
`--log-file` or enable the heartbeat, and the DESI and spPlate batch commands ignore it.

## Parallel worker logs

`dl-ingest-spectra-batch-desi-coadds` and `dl-ingest-spectra-batch-spplate` send all logging,
including worker logs, to a file, so the terminal shows only the tqdm progress bar and a
start and end summary line. The file defaults to `<lake>/spectra/<survey>/.ingest.log`; override
it with `--log-file`. The file receives INFO, or DEBUG with `-v`. With `-q` these commands
also hide the progress bar; they have no heartbeat.
