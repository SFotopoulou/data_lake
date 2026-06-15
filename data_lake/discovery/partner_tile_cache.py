"""Bounded LRU cache for partner catalog Parquet tiles during ``dl-gather``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor


@dataclass
class PartnerTileCacheConfig:
    max_bytes: int = 512 * 1024 * 1024
    max_tiles: int = 48
    enabled: bool = True


def _estimate_df_bytes(df) -> int:
    try:
        return int(df.estimated_size())
    except Exception:
        return 0


class PartnerTileCache:
    """LRU cache keyed by ``(survey, partner_npix, columns)``."""

    def __init__(self, config: PartnerTileCacheConfig | None = None) -> None:
        self._config = config or PartnerTileCacheConfig()
        self._entries: OrderedDict[tuple, object] = OrderedDict()
        self._bytes = 0

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
        import polars as pl

        id_list = [int(i) for i in ids]
        if not id_list:
            return pacc._empty_result("polars", columns)

        pid = pacc.link_id_column
        col_key = tuple(columns)

        if not self._config.enabled:
            return pacc.get_sources_by_id_in_healpix_pixels(
                id_list, [npix], columns=columns, fmt="polars",
            )

        key = (survey, int(npix), col_key)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)

        present: set[int] = set()
        if cached is not None and not cached.is_empty():
            present = set(cached[pid].to_list())

        missing = [i for i in id_list if i not in present]
        if missing:
            fetched = pacc.get_sources_by_id_in_healpix_pixels(
                missing, [npix], columns=columns, fmt="polars",
            )
            if cached is None or cached.is_empty():
                cached = fetched
            elif fetched.is_empty():
                pass
            else:
                cached = pl.concat([cached, fetched], how="vertical_relaxed")
            self._put(key, cached)

        if cached is None or cached.is_empty():
            return pacc._empty_result("polars", columns)
        return cached.filter(pl.col(pid).is_in(id_list))

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
