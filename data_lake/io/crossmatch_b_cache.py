"""Bounded LRU cache for survey-B HEALPix Parquet tiles during sky crossmatch.

Caches **full** B tiles (no RA/Dec bbox) so neighbouring survey-A tiles reuse
overlapping B pixels. Callers apply the A-tile bbox filter in memory.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor


@dataclass
class CrossmatchBTileCacheConfig:
    max_bytes: int = 512 * 1024 * 1024
    max_tiles: int = 48
    enabled: bool = True


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
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def get_tiles(
        self,
        acc_b: CatalogAccessor,
        npix_list: Sequence[int],
        columns: list[str],
        *,
        hp_col: str,
    ) -> Any:
        """Return concatenated rows for *npix_list*, using the LRU when enabled."""
        import polars as pl

        npix_ints = [int(n) for n in npix_list]
        if not npix_ints:
            return acc_b._empty_result("polars", columns)

        if not self._config.enabled:
            return acc_b.sources_in_healpix_pixels(
                npix_ints, columns=columns, fmt="polars",
            )

        col_key = tuple(columns)
        need_fetch: list[int] = []
        parts: list[Any] = []

        for npix in npix_ints:
            key = (npix, col_key)
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                if not cached.is_empty():
                    parts.append(cached)
            else:
                self.misses += 1
                need_fetch.append(npix)

        if need_fetch:
            fetched = acc_b.sources_in_healpix_pixels(
                need_fetch, columns=columns, fmt="polars",
            )
            if fetched.is_empty():
                for npix in need_fetch:
                    self._put((npix, col_key), fetched.head(0))
            elif hp_col in fetched.columns:
                for npix in need_fetch:
                    chunk = fetched.filter(pl.col(hp_col) == npix)
                    self._put((npix, col_key), chunk)
                    if not chunk.is_empty():
                        parts.append(chunk)
            elif len(need_fetch) == 1:
                self._put((need_fetch[0], col_key), fetched)
                parts.append(fetched)
            else:
                # No healpix column to split — do not cache; return fetched as-is.
                return fetched

        if not parts:
            return acc_b._empty_result("polars", columns)
        if len(parts) == 1:
            return parts[0]
        return pl.concat(parts, how="vertical_relaxed")

    def _put(self, key: tuple[int, tuple[str, ...]], df: Any) -> None:
        if key in self._entries:
            self._bytes -= _estimate_df_bytes(self._entries[key])
            del self._entries[key]
        self._entries[key] = df
        self._bytes += _estimate_df_bytes(df)
        self._entries.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while self._entries and (
            len(self._entries) > self._config.max_tiles
            or self._bytes > self._config.max_bytes
        ):
            _, old_df = self._entries.popitem(last=False)
            self._bytes -= _estimate_df_bytes(old_df)
