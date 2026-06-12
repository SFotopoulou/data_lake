"""Spatial discovery and selection over the data lake.

This package provides:

- :mod:`data_lake.discovery.region` – the ``Region`` selector (npix / cone /
  bbox / MOC) that resolves to HEALPix NESTED pixels at any target order.
- :mod:`data_lake.discovery.areas` – flat ``areas/<area_id>.json`` definitions.
- :mod:`data_lake.discovery.selection` – generalised base-source selection
  (region | ids file | DuckDB predicate) used by ``dl-gather``.
- :mod:`data_lake.discovery.tile_index` – cached per-survey/per-modality npix
  index used to avoid full-tree ``rglob`` during discovery.
"""

from __future__ import annotations

from data_lake.discovery.region import (
    Region,
    rescale_npix_nested,
)

__all__ = ["Region", "rescale_npix_nested"]
