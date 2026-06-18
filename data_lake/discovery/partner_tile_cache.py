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
        return self.lookup_many(pacc, survey, [int(npix)], columns, ids)

    def lookup_many(
        self,
        pacc: CatalogAccessor,
        survey: str,
        npix_list: Sequence[int],
        columns: list[str],
        ids: Sequence[int],
    ):
        """Return partner rows for *ids*, batching disk reads across *npix_list*."""
        import polars as pl

        id_list = [int(i) for i in ids]
        npix_ints = [int(n) for n in npix_list]
        if not id_list or not npix_ints:
            return pacc._empty_result("polars", columns)

        pid = pacc.link_id_column
        col_key = tuple(columns)

        if not self._config.enabled:
            return pacc.get_sources_by_id_in_healpix_pixels(
                id_list, npix_ints, columns=columns, fmt="polars",
            )

        hp_col = f"_healpix_norder{pacc.norder}"
        need_fetch: list[int] = []
        parts: list[pl.DataFrame] = []

        for npix in npix_ints:
            key = (survey, npix, col_key)
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
            present: set[int] = set()
            if cached is not None and not cached.is_empty():
                present = set(cached[pid].to_list())
            if [i for i in id_list if i not in present]:
                need_fetch.append(npix)
            elif cached is not None and not cached.is_empty():
                parts.append(cached.filter(pl.col(pid).is_in(id_list)))

        if need_fetch:
            fetch_cols = list(columns)
            if hp_col not in fetch_cols:
                fetch_cols.append(hp_col)
            fetched = pacc.get_sources_by_id_in_healpix_pixels(
                id_list, need_fetch, columns=fetch_cols, fmt="polars",
            )
            if not fetched.is_empty():
                if hp_col in fetched.columns:
                    for npix in need_fetch:
                        key = (survey, npix, col_key)
                        chunk = fetched.filter(pl.col(hp_col) == npix)
                        if hp_col not in columns:
                            chunk = chunk.drop(hp_col)
                        if chunk.is_empty():
                            continue
                        cached = self._entries.get(key)
                        if cached is not None and not cached.is_empty():
                            merged = pl.concat([cached, chunk], how="vertical_relaxed")
                        else:
                            merged = chunk
                        self._put(key, merged)
                        parts.append(merged.filter(pl.col(pid).is_in(id_list)))
                elif len(need_fetch) == 1:
                    npix = need_fetch[0]
                    key = (survey, npix, col_key)
                    cached = self._entries.get(key)
                    if cached is not None and not cached.is_empty():
                        merged = pl.concat([cached, fetched], how="vertical_relaxed")
                    else:
                        merged = fetched
                    self._put(key, merged)
                    parts.append(merged.filter(pl.col(pid).is_in(id_list)))
                else:
                    parts.append(fetched.filter(pl.col(pid).is_in(id_list)))

        if not parts:
            return pacc._empty_result("polars", columns)
        if len(parts) == 1:
            return parts[0]
        return pl.concat(parts, how="vertical_relaxed")

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
