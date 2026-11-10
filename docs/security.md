# Security and trust boundaries

This page records which lake inputs are treated as **trusted operator
input**. The package is built for HPC and lab deployments where operators and
analysts run `dl-*` against a shared lake; it is not a multi-tenant SQL service.

## DuckDB predicates (`--where`, `--where-joined`)

Flags such as `--where` and `--where-joined` on extract, gather, homogenize and
related CLIs take a DuckDB boolean expression that is interpolated into SQL and
executed against catalog views. Typical examples:

```bash
dl-extract-catalog --survey DESI_DR1 \
  --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 \
  -o /scratch/desi_stars.parquet

dl-gather … --where-joined "DESI_DR1_z > 0.5"
```

**Treat these strings as trusted.** Do not pass untrusted user text into
`--where` / `--where-joined` (or into `CatalogAccessor.query` / gather filters
from a web form or agent) without a separate validation layer. There is no
prepared-statement binding for free-form predicates today.

Safe patterns for operators:

- Write the predicate yourself, or take it from a reviewed notebook / script.
- Prefer ID lists (`--target-list`, `--ids`) when the selection comes from an
  external pipeline that you do not fully control.
- Keep MCP explorers read-only: `build_query` returns SQL as text and does not
  execute it (see [MCP deployment](mcp-deploy.md#read-only-boundary)).

## Path-built view DDL

Helpers such as `query_from_master` and catalog accessors build
`CREATE VIEW … AS SELECT * FROM read_parquet('…')` (or `parquet_scan`) with the
resolved lake path embedded in the SQL string. Paths come from the deployment
(`lake_config.toml` / `DATA_LAKE_CONFIG`) or from CLI arguments supplied by the
operator. Do not point those roots at attacker-controlled paths.

## Ingest writes

Write paths (`dl-ingest-*`) additionally require a deployment ingest token
(`LAKE_INGEST_TOKEN` / `--ingest-token`). See [lake-config.md](lake-config.md).

## See also

- [Export and sharing](export-and-sharing.md) — `--where` extract examples
- [Gather](discovery/gather.md) — `--where` / `--where-joined`
- [MCP deployment](mcp-deploy.md) — read-only agent boundary
- [CLI reference](cli-reference.md)
