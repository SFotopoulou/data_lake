# Logging and long runs

By default all `dl-*` CLIs log at **INFO** to stderr.  On 12-hour ingest runs this creates thousands of per-file lines that slow down the terminal and make it hard to track overall progress.

### Verbosity levels

| Mode | Flags | Terminal (stderr) | Optional `--log-file` |
|------|-------|-------------------|------------------------|
| **Normal** (default) | (none) | INFO | INFO if `--log-file` set |
| **Quiet** | `-q` / `--quiet` | WARNING + heartbeat | INFO (full audit trail) |
| **Verbose** | `-v` / `--verbose` | DEBUG | DEBUG |

`-q` and `-v` are mutually exclusive.

Resolution order: **CLI flag > `cfg.ingest.log_level` > default INFO**.  Set `log_level = "WARNING"` in `lake_config.toml` to make quiet the default for all commands in a deployment.

### Heartbeat summaries

When `-q` is active, a one-line summary is printed to stderr every **5 minutes** (configurable with `--heartbeat-interval SEC`):

```
[spectra-ingest] 12000/50000 files (24%), 1,230,000 spectra, 2 failed, 2.10 files/s, ETA ~4.2h (37.2 min elapsed)
```

Heartbeat fires even when the tqdm progress bar is active, so you always have a time-stamped checkpoint in shell history.

### Recommended patterns for long runs

**12-hour spPlate / GAMA run:**

```bash
# Quiet mode with full INFO audit log
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

Use a separate log file per array task: concurrent tasks appending to one file can
interleave lines. Reduce failure lines carry `npix` and `parts` instead of `path`. See
[Two-phase map-reduce ingest](batch-and-checkpoints.md#two-phase-map-reduce-ingest-dl-ingest-from-shard).

### Config-file default

Set in `lake_config.toml` to make quiet the deployment default:

```toml
[ingest]
log_level = "WARNING"   # quiet by default; -v to override at runtime
```

### Parallel worker logs

Batch CLIs (`dl-ingest-spectra-batch-desi-coadds`, `dl-ingest-spectra-batch-spplate`) already route all worker logs to `--log-file` (stderr is reserved for tqdm).  With `-q` the terminal shows only the progress bar and heartbeat; the log file still captures all INFO.

