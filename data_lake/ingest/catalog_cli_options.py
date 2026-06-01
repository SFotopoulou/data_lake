"""
Shared Click options and helpers for catalog file-list ingest CLIs.

Keeps dl-ingest-catalog-from-list and dl-ingest-catalog-batch in sync on
the two flags where they were previously diverging: --tile-mode and
--allow-incomplete-link-id.
"""

from __future__ import annotations

import functools
from typing import Callable

try:
    import click

    from data_lake.ingest.fits_to_parquet import TileMode

    # ---------------------------------------------------------------------------
    # --allow-incomplete-link-id
    # ---------------------------------------------------------------------------

    def allow_incomplete_link_id_option(f: Callable) -> Callable:
        """Add --allow-incomplete-link-id to a Click command."""
        return click.option(
            "--allow-incomplete-link-id",
            is_flag=True,
            help=(
                "For composite or string --link-id-col: leave _source_id null when "
                "any link part is missing (row stays in catalog, unlinked to spectra)."
            ),
        )(f)

    # ---------------------------------------------------------------------------
    # --tile-mode
    # ---------------------------------------------------------------------------

    def tile_mode_option(*, default: str | None, help: str) -> Callable[[Callable], Callable]:
        """Return a --tile-mode Click decorator with the given default and help."""
        def decorator(f: Callable) -> Callable:
            return click.option(
                "--tile-mode",
                type=click.Choice(["skip", "overwrite", "append"], case_sensitive=False),
                default=default,
                show_default=default is not None,
                help=help,
            )(f)
        return decorator

    # Canonical help strings for each command context
    TILE_MODE_HELP_FROM_LIST = (
        "Existing Npix tile: skip (default for --n-workers 1), replace, or append "
        "(multi-FITS). Omitting this flag on the parallel path (--n-workers > 1) "
        "defaults to append."
    )
    TILE_MODE_HELP_BATCH = "Use append for multi-file ingest (recommended)."

    def from_list_tile_mode_option(f: Callable) -> Callable:
        return tile_mode_option(default=None, help=TILE_MODE_HELP_FROM_LIST)(f)

    def batch_tile_mode_option(f: Callable) -> Callable:
        return tile_mode_option(default="append", help=TILE_MODE_HELP_BATCH)(f)

except ImportError:
    # click not installed: keep stubs so the module can be imported
    allow_incomplete_link_id_option = None  # type: ignore[assignment]
    from_list_tile_mode_option = None  # type: ignore[assignment]
    batch_tile_mode_option = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tile-mode resolver
# ---------------------------------------------------------------------------

def resolve_catalog_tile_mode(tile_mode: str | None, *, parallel: bool) -> "TileMode | None":
    """Resolve a raw --tile-mode CLI value to the effective TileMode.

    Args:
        tile_mode: Value from Click (a lowercased/trimmed choice string, or None).
        parallel:  True when running the parallel decode path (n_workers > 1 or
                   dl-ingest-catalog-batch); False for sequential per-file ingest.

    Returns:
        - The explicit value (lowercased) when tile_mode is not None.
        - ``"append"`` when omitted on the parallel path (safe default for
          overlapping HEALPix tiles across multiple input files).
        - ``None`` when omitted on the sequential path, which lets
          :func:`data_lake.ingest.fits_to_parquet.ingest_catalog` fall back to
          its own default of ``"skip"``.
    """
    if tile_mode is not None:
        return tile_mode.lower()  # type: ignore[return-value]
    return "append" if parallel else None
