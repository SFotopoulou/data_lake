"""Bounded LRU cache of whole partner catalog tiles during ``dl-gather``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor


@dataclass
class PartnerTileCacheConfig:
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


def _estimate_df_bytes(df) -> int:
    try:
        return int(df.estimated_size())
    except Exception:
        return 0


class PartnerTileCache:
    """LRU of whole partner tiles keyed by ``(survey, partner_npix, columns)``.

    Neighbouring base tiles share partner tiles, so each partner tile is read
    from disk once (with only the requested columns) and filtered in memory for
    every later base tile.  Tiles with no rows are remembered separately and do
    not count toward ``max_tiles``.
    """

    def __init__(self, config: PartnerTileCacheConfig | None = None) -> None:
        from data_lake.io.crossmatch_b_cache import EMPTY_TILE_KEYS_MAX

        self._config = config or PartnerTileCacheConfig()
        self._entries: OrderedDict[tuple, object] = OrderedDict()
        self._empty: OrderedDict[tuple, None] = OrderedDict()
        self._empty_max = EMPTY_TILE_KEYS_MAX
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def lookup(
        self,
        pacc: CatalogAccessor,
        survey: str,
        npix: int,
        columns: list[str],
        ids: Sequence[int],
    ):
        """Return partner rows for *ids* in one HEALPix tile, using cache when enabled."""
        return self.lookup_many(pacc, survey, [int(npix)], columns, ids)

    def lookup_many(
        self,
        pacc: CatalogAccessor,
        survey: str,
        npix_list: Sequence[int],
        columns: list[str],
        ids: Sequence[int],
    ):
        """Return partner rows for *ids* found in the tiles *npix_list*."""
        import polars as pl

        id_list = [int(i) for i in ids]
        npix_ints = sorted({int(n) for n in npix_list})
        if not id_list or not npix_ints:
            return pacc._empty_result("polars", columns)

        if not self._config.enabled:
            return pacc.get_sources_by_id_in_healpix_pixels(
                id_list, npix_ints, columns=columns, fmt="polars",
            )

        pid = pacc.link_id_column
        col_key = tuple(columns)
        parts: list[pl.DataFrame] = []
        for npix in npix_ints:
            tile = self._tile(pacc, survey, npix, columns, col_key)
            if tile is not None:
                parts.append(tile)
        if not parts:
            return pacc._empty_result("polars", columns)
        frame = parts[0] if len(parts) == 1 else pl.concat(parts, how="vertical_relaxed")
        return frame.filter(pl.col(pid).is_in(id_list))

    def _tile(self, pacc: CatalogAccessor, survey: str, npix: int, columns, col_key):
        key = (survey, npix, col_key)
        if key in self._empty:
            self._empty.move_to_end(key)
            self.hits += 1
            return None
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        df = pacc.sources_in_healpix_pixels([npix], columns=list(columns), fmt="polars")
        if df.is_empty():
            self._empty[key] = None
            while len(self._empty) > self._empty_max:
                self._empty.popitem(last=False)
            return None
        self._put(key, df)
        return df

    def summary(self) -> str:
        return (
            f"{self.hits} hit(s), {self.misses} miss(es); holding "
            f"{len(self._entries)}/{self._config.max_tiles} tile(s), "
            f"{self._bytes / 2**20:.0f}/{self._config.max_bytes / 2**20:.0f} MiB"
        )

    def _put(self, key: tuple, df) -> None:
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
