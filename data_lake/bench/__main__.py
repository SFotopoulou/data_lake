"""Performance benchmark CLI for data_lake (JSON lines output)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np


def _emit(record: dict) -> None:
    click.echo(json.dumps(record, sort_keys=True))


@click.group("dl-bench")
def cli() -> None:
    """Run lightweight lake performance scenarios."""


@cli.command("lookup-spectra")
@click.argument("lake_root", type=click.Path(exists=True, path_type=Path))
@click.option("--survey", required=True)
@click.option("--n-ids", default=1000, show_default=True, type=int)
def bench_lookup_spectra(lake_root: Path, survey: str, n_ids: int) -> None:
    """Benchmark bulk vs loop spectrum ID lookup."""
    from data_lake.io.spectra import SpectrumAccessor

    acc = SpectrumAccessor(lake_root, survey)
    tiles = acc.available_tiles()
    if not tiles:
        raise click.ClickException("No spectrum tiles found.")
    store = acc._get_tile_store(tiles[0])
    sids = store.get_source_ids()
    if sids.size == 0:
        raise click.ClickException("Empty tile.")
    rng = np.random.default_rng(0)
    ids = rng.choice(sids, size=min(n_ids, sids.size), replace=True).tolist()

    t0 = time.perf_counter()
    acc.get_batch(ids)
    batch_s = time.perf_counter() - t0

    _emit({
        "scenario": "lookup-spectra",
        "survey": survey,
        "n_ids": len(ids),
        "elapsed_s": round(batch_s, 4),
        "ids_per_s": round(len(ids) / batch_s, 1) if batch_s else None,
    })


@cli.command("catalog-streaming-columns")
@click.argument("fits_path", type=click.Path(exists=True, path_type=Path))
@click.option("--columns", required=True, help="Comma-separated column subset.")
@click.option("--ra-col", default="RA", show_default=True)
@click.option("--dec-col", default="DEC", show_default=True)
@click.option("--link-id-col", default="TARGETID", show_default=True)
def bench_streaming_columns(
    fits_path: Path,
    columns: str,
    ra_col: str,
    dec_col: str,
    link_id_col: str,
) -> None:
    """Time streaming ingest with vs without column projection (dry read)."""
    from data_lake.io.fits_read import FitsReadPolicy, materialize_fits_columns, materialize_fits_rows, open_fits

    col_list = [c.strip() for c in columns.split(",") if c.strip()]
    policy = FitsReadPolicy(memmap="on")
    with open_fits(str(fits_path), policy) as hdul:
        from astropy.io import fits

        data = next(hdu.data for hdu in hdul if isinstance(hdu, fits.BinTableHDU))
        n = min(10_000, len(data))
        idx = np.arange(n, dtype=np.int64)
        read_cols = col_list + [ra_col, dec_col, link_id_col]

        t0 = time.perf_counter()
        materialize_fits_rows(data, idx)
        full_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        materialize_fits_columns(data, idx, read_cols)
        subset_s = time.perf_counter() - t0

    _emit({
        "scenario": "catalog-streaming-columns",
        "n_rows": n,
        "n_columns_full": len(data.dtype.names),
        "n_columns_subset": len(read_cols),
        "full_elapsed_s": round(full_s, 4),
        "subset_elapsed_s": round(subset_s, 4),
    })


if __name__ == "__main__":
    cli()
