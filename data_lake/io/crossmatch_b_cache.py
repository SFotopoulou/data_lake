"""Bounded LRU cache for survey-B HEALPix Parquet tiles during sky crossmatch.

Caches **full** B tiles (no RA/Dec bbox) so neighbouring survey-A tiles reuse
overlapping B pixels. Callers apply the A-tile bbox filter in memory.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor

log = logging.getLogger(__name__)

_MIB = 1024 * 1024

EMPTY_TILE_KEYS_MAX = 65_536
"""Bound on remembered empty / absent B tiles (a few MiB of keys)."""


def tile_file_bytes(acc: CatalogAccessor, npix: int) -> int:
    """On-disk size of one catalog tile Parquet (0 when absent)."""
    path = acc._tile_parquet_path(int(npix))
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


@dataclass
class CrossmatchBTileCacheConfig:
    max_bytes: int = 0  # filled in __post_init__ from knobs when 0
    max_tiles: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        from data_lake.knobs import (
            PARTNER_TILE_CACHE_MAX_BYTES,
            PARTNER_TILE_CACHE_MAX_TILES,
        )

        if self.max_bytes <= 0:
            self.max_bytes = PARTNER_TILE_CACHE_MAX_BYTES
        if self.max_tiles <= 0:
            self.max_tiles = PARTNER_TILE_CACHE_MAX_TILES


def _estimate_df_bytes(df: Any) -> int:
    try:
        return int(df.estimated_size())
    except Exception:
        return 0


def filter_sky_bbox(
    df: Any,
    *,
    ra_col: str,
    dec_col: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> Any:
    """Apply the same RA/Dec box filter used by ``sources_in_healpix_pixels`` SQL."""
    import polars as pl

    if df is None or df.is_empty():
        return df
    parts: list[pl.Expr] = [
        pl.col(dec_col) >= dec_min,
        pl.col(dec_col) <= dec_max,
    ]
    if ra_min <= ra_max:
        parts.append((pl.col(ra_col) >= ra_min) & (pl.col(ra_col) <= ra_max))
    else:
        # RA wrap across 0/360
        parts.append((pl.col(ra_col) >= ra_min) | (pl.col(ra_col) <= ra_max))
    expr = parts[0]
    for p in parts[1:]:
        expr = expr & p
    return df.filter(expr)


class CrossmatchBTileCache:
    """LRU of full survey-B tile DataFrames keyed by ``(npix, columns)``."""

    def __init__(self, config: CrossmatchBTileCacheConfig | None = None) -> None:
        self._config = config or CrossmatchBTileCacheConfig()
        self._entries: OrderedDict[tuple[int, tuple[str, ...]], Any] = OrderedDict()
        # Tiles absent on disk or with no rows: remembered so they are not
        # re-queried, but kept out of max_tiles (they hold no data).
        self._empty: OrderedDict[tuple[int, tuple[str, ...]], None] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.max_tile_bytes = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def n_bytes(self) -> int:
        return self._bytes

    @property
    def n_tiles(self) -> int:
        return len(self._entries)

    def summary(self) -> str:
        return (
            f"{self.hits} hit(s), {self.misses} miss(es), {self.evictions} eviction(s); "
            f"largest tile {self.max_tile_bytes / _MIB:.1f} MiB in RAM; "
            f"holding {self.n_tiles}/{self._config.max_tiles} tile(s), "
            f"{self._bytes / _MIB:.0f}/{self._config.max_bytes / _MIB:.0f} MiB, "
            f"{len(self._empty)} empty tile(s) remembered"
        )

    def _log_fetched(self, acc_b: CatalogAccessor, npix: int, df: Any) -> None:
        if not log.isEnabledFor(logging.DEBUG):
            return
        disk = tile_file_bytes(acc_b, npix)
        if disk == 0:
            return
        log.debug(
            "B tile Npix=%d: %d row(s), %.1f MiB on disk, %.1f MiB in RAM; "
            "cache %d/%d tile(s), %.0f/%.0f MiB",
            npix,
            df.height,
            disk / _MIB,
            _estimate_df_bytes(df) / _MIB,
            self.n_tiles,
            self._config.max_tiles,
            self._bytes / _MIB,
            self._config.max_bytes / _MIB,
        )

    def get_tiles(
        self,
        acc_b: CatalogAccessor,
        npix_list: Sequence[int],
        columns: list[str],
    ) -> Any:
        """Return concatenated rows for *npix_list*, using the LRU when enabled.

        Each missing tile is read with its own query, so a miss holds one copy
        of each tile (the cached frame) rather than a combined frame plus a
        per-tile split of it.
        """
        import polars as pl

        npix_ints = [int(n) for n in npix_list]
        if not npix_ints:
            return acc_b._empty_result("polars", columns)

        if not self._config.enabled:
            return acc_b.sources_in_healpix_pixels(
                npix_ints, columns=columns, fmt="polars",
            )

        col_key = tuple(columns)
        parts: list[Any] = []

        for npix in npix_ints:
            key = (npix, col_key)
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                parts.append(cached)
                continue
            if key in self._empty:
                self._empty.move_to_end(key)
                self.hits += 1
                continue
            self.misses += 1
            df = acc_b.sources_in_healpix_pixels([npix], columns=columns, fmt="polars")
            if df.is_empty():
                self._put_empty(key)
                continue
            self._put(key, df)
            self._log_fetched(acc_b, npix, df)
            parts.append(df)

        if not parts:
            return acc_b._empty_result("polars", columns)
        if len(parts) == 1:
            return parts[0]
        return pl.concat(parts, how="vertical_relaxed")

    def _put_empty(self, key: tuple[int, tuple[str, ...]]) -> None:
        self._empty[key] = None
        self._empty.move_to_end(key)
        while len(self._empty) > EMPTY_TILE_KEYS_MAX:
            self._empty.popitem(last=False)

    def _put(self, key: tuple[int, tuple[str, ...]], df: Any) -> None:
        if key in self._entries:
            self._bytes -= _estimate_df_bytes(self._entries[key])
            del self._entries[key]
        self._entries[key] = df
        nbytes = _estimate_df_bytes(df)
        self._bytes += nbytes
        self.max_tile_bytes = max(self.max_tile_bytes, nbytes)
        self._entries.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while self._entries and (
            len(self._entries) > self._config.max_tiles
            or self._bytes > self._config.max_bytes
        ):
            (old_npix, _), old_df = self._entries.popitem(last=False)
            old_bytes = _estimate_df_bytes(old_df)
            self._bytes -= old_bytes
            self.evictions += 1
            log.debug("B-cache evicted Npix=%d (%.1f MiB)", old_npix, old_bytes / _MIB)
