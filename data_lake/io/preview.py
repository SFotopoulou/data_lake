"""Peek helpers: show the first N sources from a lake modality.

Usage
-----
>>> from data_lake.io.preview import preview_sources
>>> preview_sources("/data/lake", "DESI_DR1", modality="catalog", n=5)
>>> preview_sources("/data/lake", "DESI_DR1", modality="spectra", n=5)
>>> preview_sources("/data/lake", "DESI_DR1", modality="cutouts", n=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl

Modality = Literal["catalog", "spectra", "cutouts", "spectrum", "cutout"]

_MODALITY_ALIASES: dict[str, str] = {
    "catalog": "catalog",
    "spectra": "spectra",
    "spectrum": "spectra",
    "cutouts": "cutouts",
    "cutout": "cutouts",
}

# Preferred catalog peek columns when present (order preserved).
_DEFAULT_CATALOG_COLS = (
    "_source_id",
    "TARGETID",
    "SOURCE_ID",
    "SPECOBJID",
    "ra",
    "RA",
    "TARGET_RA",
    "dec",
    "DEC",
    "TARGET_DEC",
    "z",
    "Z",
)


def _normalize_modality(modality: str) -> str:
    key = modality.strip().lower()
    if key not in _MODALITY_ALIASES:
        raise ValueError(
            f"modality must be one of {sorted(set(_MODALITY_ALIASES))}, got {modality!r}"
        )
    return _MODALITY_ALIASES[key]


def _quote_sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _catalog_peek_columns(
    available: Sequence[str],
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in available]
        if missing:
            raise KeyError(
                f"Requested columns not in catalog: {missing}. "
                f"Available (first 30): {list(available)[:30]}"
            )
        return list(requested)

    cols: list[str] = []
    for name in _DEFAULT_CATALOG_COLS:
        if name in available and name not in cols:
            cols.append(name)
    if "_source_id" not in cols and "_source_id" in available:
        cols.insert(0, "_source_id")
    if not cols:
        # Fall back to a short prefix of the schema so peeks never return empty schema.
        cols = list(available)[:8]
    return cols


def _peek_catalog(
    lake_root: Path,
    survey_name: str,
    n: int,
    columns: Sequence[str] | None,
) -> pl.DataFrame:
    from data_lake.io.catalog import CatalogAccessor

    cat = CatalogAccessor(lake_root, survey_name)
    cols = _catalog_peek_columns(cat.columns, columns)
    select = ", ".join(_quote_sql_ident(c) for c in cols)
    sql = f"SELECT {select} FROM catalog LIMIT {int(n)}"
    return cat.query(sql, fmt="polars")


def _peek_zarr_modality(
    lake_root: Path,
    survey_name: str,
    *,
    modality: str,
    n: int,
) -> pl.DataFrame:
    if modality == "spectra":
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(lake_root, survey_name)
        tiles = acc.available_tiles()
        get_store = acc._get_tile_store
    else:
        from data_lake.io.cutouts import CutoutAccessor

        acc = CutoutAccessor(lake_root, survey_name)
        tiles = acc.available_tiles()
        get_store = acc._get_tile_store

    source_ids: list[int] = []
    npix_out: list[int] = []
    for npix in tiles:
        store = get_store(npix)
        sids = np.asarray(store.get_source_ids(), dtype=np.int64).ravel()
        for sid in sids.tolist():
            source_ids.append(int(sid))
            npix_out.append(int(npix))
            if len(source_ids) >= n:
                return pl.DataFrame(
                    {
                        "_source_id": source_ids,
                        "npix": npix_out,
                        "modality": [modality] * len(source_ids),
                    }
                )

    return pl.DataFrame(
        {
            "_source_id": source_ids,
            "npix": npix_out,
            "modality": [modality] * len(source_ids),
        }
    )


def preview_sources(
    lake_root: Path | str,
    survey_name: str,
    *,
    modality: Modality = "catalog",
    n: int = 10,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return the first *n* sources from a survey modality (peek / head).

    Parameters
    ----------
    lake_root:
        Data lake root directory.
    survey_name:
        Survey name under ``catalogs/``, ``spectra/``, or ``cutouts/``.
    modality:
        ``"catalog"``, ``"spectra"`` (alias ``"spectrum"``), or ``"cutouts"``
        (alias ``"cutout"``).
    n:
        Maximum number of rows to return (must be ``>= 1``).
    columns:
        Catalog only: explicit column list.  When omitted, peeks
        ``_source_id`` plus common ID / sky / redshift columns when present.

    Returns
    -------
    polars.DataFrame
        Catalog peeks include the selected columns.  Spectra / cutout peeks
        include ``_source_id``, ``npix``, and ``modality``.

    Notes
    -----
    This is a non-ranked peek: catalog uses ``LIMIT n`` (DuckDB scan order);
    spectra / cutouts walk HEALPix tiles in ascending ``npix`` order and take
    the first *n* ``_source_id`` values.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    root = Path(lake_root)
    kind = _normalize_modality(modality)
    if kind == "catalog":
        return _peek_catalog(root, survey_name, n, columns)
    if columns is not None:
        raise ValueError("columns= is only supported for modality='catalog'")
    return _peek_zarr_modality(root, survey_name, modality=kind, n=n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from data_lake.cli_utils import config_option, load_optional_config, require_output_root

    @click.command("dl-preview-sources")
    @click.argument("survey", type=str)
    @click.argument("lake_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--modality",
        default="catalog",
        type=click.Choice(
            ["catalog", "spectra", "spectrum", "cutouts", "cutout"],
            case_sensitive=False,
        ),
        show_default=True,
        help="Lake layer to peek.",
    )
    @click.option(
        "-n",
        "--limit",
        "n",
        default=10,
        show_default=True,
        type=click.IntRange(min=1),
        help="Number of sources to show.",
    )
    @click.option(
        "-c",
        "--column",
        "columns",
        multiple=True,
        help="Catalog columns to include (repeatable). Default: common ID/sky/z cols.",
    )
    @click.option(
        "-o",
        "--output",
        type=click.Path(path_type=Path),
        default=None,
        help="Write table to CSV/Parquet (suffix selects format). Default: print.",
    )
    @click.option(
        "--format",
        "out_format",
        type=click.Choice(["table", "csv", "parquet", "json"], case_sensitive=False),
        default="table",
        show_default=True,
        help="Stdout format when -o is omitted (table = aligned text).",
    )
    def cli(
        survey: str,
        lake_root: Path | None,
        config_path: Path | None,
        modality: str,
        n: int,
        columns: tuple[str, ...],
        output: Path | None,
        out_format: str,
    ) -> None:
        """Peek at the first N sources from a survey modality.

        \b
          dl-preview-sources DESI_DR1 --modality catalog -n 5
          dl-preview-sources DESI_DR1 --modality spectra -n 10
          dl-preview-sources DESI_DR1 -c _source_id -c TARGETID -c SURVEY -n 5
        """
        cfg = load_optional_config(config_path)
        root = require_output_root(lake_root, cfg)
        col_list = list(columns) if columns else None
        try:
            df = preview_sources(
                root,
                survey,
                modality=modality,  # type: ignore[arg-type]
                n=n,
                columns=col_list,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            suffix = output.suffix.lower()
            if suffix == ".parquet":
                df.write_parquet(output)
            elif suffix in (".json",):
                df.write_json(output)
            else:
                df.write_csv(output)
            click.echo(f"Wrote {len(df)} row(s) to {output}", err=True)
            return

        fmt = out_format.lower()
        if fmt == "csv":
            click.echo(df.write_csv())
        elif fmt == "json":
            click.echo(df.write_json())
        elif fmt == "parquet":
            raise click.UsageError("Use -o path.parquet for parquet output.")
        else:
            with pl.Config(tbl_rows=n, tbl_cols=-1, fmt_str_lengths=40):
                click.echo(str(df))

except ImportError:  # pragma: no cover - click always present in installed env
    pass
