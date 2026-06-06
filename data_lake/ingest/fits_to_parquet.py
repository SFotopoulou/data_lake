"""
fits_to_parquet – ingest FITS / VOTable survey catalogs into HATS-partitioned Parquet.

Layout produced
---------------
<root>/
  Norder=<order>/Dir=<dir>/Npix=<pix>.parquet
  _metadata                    ← Parquet aggregate footer (pyarrow)
  catalog_info.json            ← HATS-style descriptor
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

TileMode = Literal["skip", "overwrite", "append"]
DuplicateIdMode = Literal["skip", "error", "last"]

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.table import Table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HATS_DIR_STRIDE = 10_000  # tiles per Dir= folder (HATS convention)
_ZSTD_LEVEL = 3             # default balance of speed vs ratio (catalog ingest)

# Lake-internal int64 join key (Parquet catalogs and Zarr); not a survey column name.
LAKE_JOIN_ID_COLUMN = "_source_id"
# ``string`` uses int32 offsets (2 GiB UTF-8 cap per column chunk). Dense tiles
# (e.g. DESI zall-pix) can exceed that when narrowing ``large_string``.
_MAX_ROWS_STRING_SHRINK = 2_000_000


@dataclass(frozen=True)
class CatalogParquetOptions:
    """Parquet write tuning for HATS catalog tiles."""

    compression_level: int = _ZSTD_LEVEL
    write_statistics: bool = True
    use_dictionary: bool = True
    shrink_strings_for_tiles: bool = True

    @classmethod
    def compact(cls) -> CatalogParquetOptions:
        """Smaller on-disk footprint; slower ingest, still ZSTD-compressed."""
        return cls(
            compression_level=9,
            write_statistics=False,
            use_dictionary=False,
            shrink_strings_for_tiles=True,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def healpix_dir(norder: int, npix: int) -> str:
    """Return the HATS directory path fragment for a pixel (e.g. ``Norder=5/Dir=0``)."""
    dir_index = (npix // _HATS_DIR_STRIDE) * _HATS_DIR_STRIDE
    return f"Norder={norder}/Dir={dir_index}"


_COMMON_ID_COLUMNS = (
    "TARGETID",
    "targetid",
    "OBJID",
    "objid",
    "OBJECT_ID",
    "object_id",
    "SOURCE_ID",
    "id",
    "ID",
    "Id",
)

# Header keywords tried for Zarr ingest when --link-id-col is not set (in order).
_FITS_HEADER_ID_KEYWORDS = (
    "SOURCE_ID", "OBJ_ID", "OBJID", "TARGETID", "targetid",
    "FIBERID", "fiberid", "source_id",
)

# float64 only represents integers exactly up to 2**53; DESI TARGETIDs exceed that.
_FLOAT64_SAFE_INTEGER = 2**53
_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_UINT64_MAX = int(np.iinfo(np.uint64).max)


def stable_object_id_from_string(text: str) -> int:
    """Map an opaque string label (e.g. ``J000000.00-314627.5``) to a stable int64.

  Used when survey catalogs use alphanumeric names instead of numeric
  ``TARGETID``s.  The same UTF-8 text always yields the same integer for
  joins between Parquet catalog rows and Zarr ``_source_id`` arrays.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("object ID string is empty")
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def _object_id_text(value: object) -> str:
    if value is None:
        raise ValueError("object ID is None")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _object_id_text_optional(value: object) -> str | None:
    """Like :func:`_object_id_text` but returns ``None`` for null/blank/non-finite values."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    try:
        text = _object_id_text(value)
    except ValueError:
        return None
    return text if text else None


def _text_is_integer_object_id(text: str) -> bool:
    return bool(text) and text.lstrip("+-").isdigit()


def storage_int64_from_integer(value: int) -> int:
    """Map a logical integer object ID to int64 storage for Parquet/Zarr.

    Values in ``[0, 2**63-1]`` are stored as-is.  SDSS-style ``objid`` and other
    **uint64** IDs above ``2**63-1`` are stored by **bit pattern** in signed
    int64 (same convention as ``numpy.int64(numpy.uint64(x))``) so joins stay
    stable and unique.
    """
    if _INT64_MIN <= value <= _INT64_MAX:
        return value
    if 0 <= value <= _UINT64_MAX:
        return int.from_bytes(
            value.to_bytes(8, byteorder="little", signed=False),
            byteorder="little",
            signed=True,
        )
    raise ValueError(f"object ID {value} is outside uint64 range")


def normalize_object_id(value: object) -> int:
    """Coerce one catalog/Zarr object ID to int64 **storage** (signed int64 column).

    All cross-store matching (catalog Parquet ↔ Zarr ``_source_id`` ↔
    ``index_map`` keys) should use this so ``numpy.int64``, ``int``, and
    accidental string forms compare consistently.

    Decimal string digits are parsed as integers.  Values above ``2**63-1``
    (SDSS ``objid`` / uint64 IDs) are stored by **bit pattern** via
    :func:`storage_int64_from_integer`.  Other strings (e.g.
    ``J000000.00-314627.5``) are mapped with :func:`stable_object_id_from_string`.

    Raises ``ValueError`` for ``None``, booleans, floats (precision loss), and
    integers outside uint64 range.
    """
    if value is None:
        raise ValueError("object ID is None")
    if isinstance(value, bool):
        raise ValueError(f"invalid object ID (bool): {value!r}")

    if isinstance(value, (int, np.integer)):
        if isinstance(value, (np.bool_,)):
            raise ValueError(f"invalid object ID (bool): {value!r}")
        if isinstance(value, np.unsignedinteger):
            return storage_int64_from_integer(int(value))
        return storage_int64_from_integer(int(value))
    elif isinstance(value, (float, np.floating)):
        raise ValueError(
            f"object ID {value!r} is floating-point; DESI-scale TARGETIDs "
            f"(> {_FLOAT64_SAFE_INTEGER}) lose precision in float64. "
            f"Re-ingest the catalog with the ID column stored as int64."
        )
    elif isinstance(value, (str, bytes)):
        text = _object_id_text(value)
        if not text:
            raise ValueError(f"object ID string is empty: {value!r}")
        if _text_is_integer_object_id(text):
            return storage_int64_from_integer(int(text))
        return stable_object_id_from_string(text)
    else:
        raise TypeError(f"unsupported object ID type {type(value).__name__}: {value!r}")


def object_id_from_fits_header(
    header,
    link_id_col: str | None = None,
    *,
    hdu_index: int = 0,
) -> int:
    """Read one integer object ID from a FITS header for cutout/spectrum ingest.

    When *link_id_col* is set, only that keyword is used (same convention as
    catalog ``--link-id-col``).  Otherwise a fixed fallback chain ending in
    *hdu_index* when no ID keyword is present.
    """
    if link_id_col:
        if link_id_col not in header:
            keys = [k for k in header.keys() if k and not str(k).startswith("HISTORY")]
            raise KeyError(
                f"Header keyword {link_id_col!r} not found for object ID. "
                f"Sample keys: {keys[:25]}{'…' if len(keys) > 25 else ''}"
            )
        return normalize_object_id(header[link_id_col])

    for key in _FITS_HEADER_ID_KEYWORDS:
        if key in header:
            return normalize_object_id(header[key])

    log.warning(
        "No object-ID keyword in FITS header (tried %s); using HDU index %d. "
        "Pass --link-id-col to match the catalog (e.g. TARGETID).",
        _FITS_HEADER_ID_KEYWORDS,
        hdu_index,
    )
    return int(hdu_index)


def fits_header_keyword(header, name: str) -> object | None:
    """Return a FITS header keyword value (case-insensitive), or None if absent."""
    target = name.upper()
    for key in header.keys():
        if key and str(key).upper() == target:
            return header[key]
    return None


def sky_from_header_chain(
    *headers,
    pairs: tuple[tuple[str, str], ...],
    context: str = "FITS",
) -> tuple[float, float, str, str]:
    """Resolve (RA, Dec) in degrees from the first matching keyword pair.

    Uses key *presence* only (never treats missing keys as 0,0).  Returns the
    matched keyword names for provenance metadata.
    """
    for hdr in headers:
        for ra_key, dec_key in pairs:
            ra_raw = fits_header_keyword(hdr, ra_key)
            dec_raw = fits_header_keyword(hdr, dec_key)
            if ra_raw is None or dec_raw is None:
                continue
            ra = float(ra_raw)
            dec = float(dec_raw)
            if not is_valid_sky_position(ra, dec):
                continue
            return ra, dec, ra_key, dec_key
    tried = ", ".join(f"{a}/{b}" for a, b in pairs)
    raise ValueError(
        f"{context}: missing or invalid sky coordinates; tried keyword pairs: {tried}"
    )


def sky_from_fits_header(
    header,
    ra_col: str,
    dec_col: str,
    *,
    required: bool = False,
) -> tuple[float, float]:
    """Return (RA, Dec) in degrees from a cutout/spectrum image header."""
    if required:
        ra, dec, _, _ = sky_from_header_chain(
            header,
            pairs=(
                (ra_col, dec_col),
                ("RA_TARG", "DEC_TARG"),
                ("TARGET_RA", "TARGET_DEC"),
            ),
            context="FITS header",
        )
        return ra, dec
    ra = float(header.get(
        ra_col,
        header.get("RA_TARG", header.get("TARGET_RA", header.get("CRVAL1", 0.0))),
    ))
    dec = float(header.get(
        dec_col,
        header.get("DEC_TARG", header.get("TARGET_DEC", header.get("CRVAL2", 0.0))),
    ))
    return ra, dec


def object_id_column_is_integer_ids(column: pa.Array | pa.ChunkedArray) -> bool:
    """True if every non-null value in *column* is an integer-like object ID."""
    if isinstance(column, pa.ChunkedArray):
        return all(object_id_column_is_integer_ids(chunk) for chunk in column.chunks)
    for value in column.to_pylist():
        text = _object_id_text(value)
        if not _text_is_integer_object_id(text):
            return False
    return True


def _cast_uint64_arrow_to_int64(column: pa.Array) -> pa.Array:
    """Bit-cast Arrow unsigned integers to int64 (PyArrow ``cast`` rejects > INT64_MAX)."""
    arr = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.uint64)
    out = arr.view(np.int64)
    if column.null_count:
        return pa.array(out, type=pa.int64(), mask=pc.is_null(column).to_numpy(zero_copy_only=False))
    return pa.array(out, type=pa.int64())


def cast_object_id_column_to_int64(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast a catalog object-ID column to Arrow ``int64``.

    Integer columns are cast directly.  String, binary, and Python-object
    columns are parsed with :func:`normalize_object_id` (digit strings and
    DESI-scale integers).  Floating columns are cast with Arrow (caller should
    run the >2**53 safety check first via :func:`warn_if_id_column_unsafe`).
    """
    if isinstance(column, pa.ChunkedArray):
        if column.num_chunks == 0:
            return pa.array([], type=pa.int64())
        if column.num_chunks == 1:
            return cast_object_id_column_to_int64(column.chunk(0))
        return pa.chunked_array(
            [cast_object_id_column_to_int64(chunk) for chunk in column.chunks]
        ).combine_chunks()

    if column.type == pa.int64():
        return column

    if pa.types.is_unsigned_integer(column.type):
        return _cast_uint64_arrow_to_int64(column)

    if pa.types.is_integer(column.type):
        return column.cast(pa.int64())

    if pa.types.is_floating(column.type):
        return column.cast(pa.int64())

    if (
        pa.types.is_string(column.type)
        or pa.types.is_large_string(column.type)
        or pa.types.is_binary(column.type)
        or pa.types.is_large_binary(column.type)
    ):
        out = [normalize_object_id(v) for v in column.to_pylist()]
        return pa.array(out, type=pa.int64())

    if pa.types.is_null(column.type):
        return pa.array([], type=pa.int64())

    raise TypeError(
        f"Cannot convert object-ID column with Arrow type {column.type} to int64; "
        f"expected integer, float, string, or binary."
    )


def match_schema_column(requested: str, schema_names: Sequence[str]) -> str | None:
    """Return the exact Parquet/FITS column name matching *requested* (case-insensitive)."""
    if requested in schema_names:
        return requested
    by_upper = {n.upper(): n for n in schema_names}
    return by_upper.get(requested.upper())


def composite_link_label(*parts: object, sep: str = "|") -> str:
    """Join non-empty label parts into one stable link string (e.g. ``a|b``)."""
    values = [_object_id_text(p) for p in parts if _object_id_text(p)]
    if not values:
        raise ValueError("composite link label requires at least one non-empty part")
    return sep.join(values)


def composite_link_label_optional(
    *parts: object,
    require_all_parts: bool,
    sep: str = "|",
) -> str | None:
    """Build a composite link label, or ``None`` when parts are incomplete.

    When *require_all_parts* is True, every part must be non-empty; otherwise
    returns ``None``.  When False, uses the same partial-join rule as
    :func:`composite_link_label` (empty parts dropped).
    """
    if require_all_parts:
        texts = [_object_id_text_optional(p) for p in parts]
        if any(t is None for t in texts):
            return None
        return sep.join(texts)  # type: ignore[arg-type]

    values = [t for p in parts if (t := _object_id_text_optional(p))]
    if not values:
        return None
    return sep.join(values)


def _stable_join_hashes(labels: Sequence[str | None]) -> pa.Array:
    """Map link labels to nullable int64 ``_source_id`` hashes."""
    out: list[int | None] = []
    for text in labels:
        if text is None:
            out.append(None)
        else:
            out.append(stable_object_id_from_string(text))
    return pa.array(out, type=pa.int64())


def parse_link_id_column_spec(link_id_col: str) -> list[str]:
    """Split a catalog ``--link-id-col`` spec into one or more column names."""
    return [part.strip() for part in link_id_col.split(",") if part.strip()]


def _link_id_columns_present(link_id_col: str | None, schema_names: list[str]) -> bool:
    """True when every column in a link-id spec exists in *schema_names*."""
    if not link_id_col:
        return False
    if "," in link_id_col:
        try:
            resolve_link_id_column_names(link_id_col, schema_names)
            return True
        except ValueError:
            return False
    return match_schema_column(link_id_col, schema_names) is not None


def resolve_link_id_column_names(
    link_id_col: str,
    schema_names: list[str],
) -> list[str]:
    """Resolve a single or comma-separated source-ID column spec against *schema_names*."""
    matched: list[str] = []
    missing: list[str] = []
    for part in parse_link_id_column_spec(link_id_col):
        name = match_schema_column(part, schema_names)
        if name is None:
            missing.append(part)
        else:
            matched.append(name)
    if missing:
        raise ValueError(
            f"Source-ID column(s) not in catalog schema: {missing!r}. "
            f"Available: {sorted(schema_names)[:30]}"
            f"{'…' if len(schema_names) > 30 else ''}"
        )
    if not matched:
        raise ValueError("source-id column spec is empty")
    return matched


# First match wins (case-insensitive against catalog Parquet columns).
_REDSHIFT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "Z",
    "ZCOSMO",
    "Z_HP",
    "Z_PHOT",
    "REDSHIFT",
    "Z_QSO",
    "Z_RED",
)


def resolve_redshift_column(column_names: list[str]) -> str | None:
    """Return the catalog column name to use for spectroscopic redshift."""
    by_upper = {n.upper(): n for n in column_names}
    for cand in _REDSHIFT_COLUMN_CANDIDATES:
        if cand in by_upper:
            return by_upper[cand]
    return None


def native_id_column_from_mode(mode: str) -> str | None:
    """Survey-native ID column from ``link_id_mode``, or ``None`` for sequential."""
    if isinstance(mode, str) and (
        mode.startswith("column:")
        or mode.startswith("label:")
        or mode.startswith("composite:")
    ):
        return mode.split(":", 1)[1]
    return None


def infer_native_id_column(schema_names: Sequence[str]) -> str | None:
    """Return the first plausible native object-ID column name in *schema_names*."""
    names_set = set(schema_names) - {LAKE_JOIN_ID_COLUMN}
    for cand in _COMMON_ID_COLUMNS:
        if cand in names_set:
            return cand
    by_upper = {n.upper(): n for n in schema_names}
    for cand in _COMMON_ID_COLUMNS:
        hit = by_upper.get(cand.upper())
        if hit is not None:
            return hit
    return None


def catalog_tile_schema_names(catalog_root: Path | str) -> list[str] | None:
    """Column names from the first on-disk ``Npix=*.parquet`` tile, if any."""
    catalog_root = Path(catalog_root)
    first_tile = next(catalog_root.rglob("Npix=*.parquet"), None)
    if first_tile is None:
        return None
    return list(pq.read_schema(str(first_tile)).names)


def link_id_column_from_mode(mode: str) -> str:
    """Map ``catalog_info.json`` ``link_id_mode`` to the lake join column name."""
    return LAKE_JOIN_ID_COLUMN


def _drop_join_id_columns(table: pa.Table) -> pa.Table:
    if LAKE_JOIN_ID_COLUMN in table.schema.names:
        table = table.drop_columns([LAKE_JOIN_ID_COLUMN])
    return table


def _set_lake_join_id_column(table: pa.Table, join_values: pa.Array) -> pa.Table:
    """Attach or replace ``_source_id`` with *join_values* (int64)."""
    table = _drop_join_id_columns(table)
    return table.append_column(LAKE_JOIN_ID_COLUMN, join_values)


def _sync_lake_join_id_from_native(table: pa.Table, native_col: str) -> pa.Table:
    """Copy cast native IDs into ``_source_id`` (``column:`` ingest)."""
    join = cast_object_id_column_to_int64(table.column(native_col))
    return _set_lake_join_id_column(table, join)


def ensure_catalog_source_ids(
    table: pa.Table,
    link_id_col: str | None,
    *,
    allow_incomplete_link_id: bool = False,
) -> tuple[pa.Table, str]:
    """Ensure the table has int64 object IDs for Zarr/spectrum joins.

    Returns ``(table, link_id_mode)`` where *link_id_mode* is written to
    ``catalog_info.json``:

    * ``sequential`` — auto-generated ``_source_id`` column
    * ``column:COL`` — native integer column *COL* (cast to int64 in place) plus ``_source_id``
    * ``label:COL`` — human-readable labels stay in *COL*; ``_source_id`` is a
      stable hash of each label (for cross-store matching)

    When *allow_incomplete_link_id* is True and ``--link-id-col`` is composite or
    a string label column, rows with any missing/blank link part get null
    ``_source_id`` (row stays in Parquet; spectrum ingest will not link them).

    *link_id_col* is required; auto-inference and sequential ``_source_id`` are
    not supported.
    """
    if not link_id_col:
        raise ValueError(
            "Catalog ingest requires --link-id-col (e.g. TARGETID, SOURCE_ID). "
            "Pass the survey's native object ID column."
        )

    if "," in link_id_col:
        col_names = resolve_link_id_column_names(link_id_col, table.schema.names)
        row_iter = zip(*(table.column(name).to_pylist() for name in col_names))
        if allow_incomplete_link_id:
            labels = [
                composite_link_label_optional(
                    *row, require_all_parts=True,
                )
                for row in row_iter
            ]
            n_unlinked = sum(1 for text in labels if text is None)
            if n_unlinked:
                log.warning(
                    "Composite link-id: %d / %d row(s) missing a link part in %r; "
                    "_source_id left null (use --allow-incomplete-link-id).",
                    n_unlinked,
                    len(labels),
                    col_names,
                )
        else:
            labels = [composite_link_label(*row) for row in row_iter]
        hashes = _stable_join_hashes(labels)
        table = _set_lake_join_id_column(table, hashes)
        spec = ",".join(col_names)
        log.info(
            "Composite link IDs from columns %r → _source_id (mode composite:%s).",
            col_names,
            spec,
        )
        return table, f"composite:{spec}"

    matched = match_schema_column(link_id_col, table.schema.names)
    if matched is None:
        raise KeyError(
            f"Link-ID column {link_id_col!r} not found in catalog table. "
            f"Available columns: {table.schema.names[:30]}"
            f"{'…' if len(table.schema.names) > 30 else ''}"
        )
    link_id_col = matched

    sid_field = table.schema.field(link_id_col)
    col = table.column(link_id_col)

    if pa.types.is_fixed_size_list(sid_field.type) or pa.types.is_list(sid_field.type):
        hints = [
            n
            for n in table.schema.names
            if n != link_id_col
            and n.lower() in ("objid", "obj_id", "bestobjid", "targetid", "thingid", "source_id")
        ]
        raise ValueError(
            f"Column {link_id_col!r} is a vector column ({sid_field.type}); "
            f"--link-id-col must name a scalar integer ID column. "
            f"For SDSS specObj use the scalar ``objid`` field, not the "
            f"multidim ``OBJID`` array. "
            f"Other scalar ID-like columns in this table: {hints[:12]}"
            f"{'…' if len(hints) > 12 else ''}"
        )

    if sid_field.type == pa.int64():
        table = _sync_lake_join_id_from_native(table, link_id_col)
        return table, f"column:{link_id_col}"

    if pa.types.is_unsigned_integer(sid_field.type):
        log.info(
            "Object-ID column %r is unsigned; storing values as int64 bit patterns "
            "(SDSS-style objid > 2**63-1).",
            link_id_col,
        )
        table = table.set_column(
            table.schema.get_field_index(link_id_col),
            link_id_col,
            cast_object_id_column_to_int64(col),
        )
        table = _sync_lake_join_id_from_native(table, link_id_col)
        return table, f"column:{link_id_col}"

    if pa.types.is_integer(sid_field.type):
        table = table.set_column(
            table.schema.get_field_index(link_id_col),
            link_id_col,
            cast_object_id_column_to_int64(col),
        )
        table = _sync_lake_join_id_from_native(table, link_id_col)
        return table, f"column:{link_id_col}"

    if pa.types.is_floating(sid_field.type):
        warn_if_id_column_unsafe(link_id_col, sid_field.type)
        col_np = col.to_numpy(zero_copy_only=False)
        finite = col_np[np.isfinite(col_np)]
        if finite.size and np.max(np.abs(finite)) > _FLOAT64_SAFE_INTEGER:
            raise ValueError(
                f"Column {link_id_col!r} contains values above 2**53 "
                f"but is stored as {sid_field.type}; casting to int64 would "
                f"corrupt TARGETIDs. Fix the FITS dtype or read as int64 "
                f"before ingest."
            )
        table = table.set_column(
            table.schema.get_field_index(link_id_col),
            link_id_col,
            cast_object_id_column_to_int64(col),
        )
        table = _sync_lake_join_id_from_native(table, link_id_col)
        return table, f"column:{link_id_col}"

    if object_id_column_is_integer_ids(col):
        table = table.set_column(
            table.schema.get_field_index(link_id_col),
            link_id_col,
            cast_object_id_column_to_int64(col),
        )
        table = _sync_lake_join_id_from_native(table, link_id_col)
        return table, f"column:{link_id_col}"

    if allow_incomplete_link_id:
        labels = [_object_id_text_optional(v) for v in col.to_pylist()]
        n_unlinked = sum(1 for text in labels if text is None)
        if n_unlinked:
            log.warning(
                "Label link-id column %r: %d / %d row(s) blank; _source_id left null.",
                link_id_col,
                n_unlinked,
                len(labels),
            )
    else:
        labels = [_object_id_text(v) for v in col.to_pylist()]
    sample = next((t for t in labels if t), "")
    log.warning(
        "Object-ID column %r has non-integer labels (e.g. %r). Keeping it "
        "unchanged and adding int64 column _source_id (stable hash) for "
        "spectrum/cutout joins. SQL: filter on %r; Python API: "
        "normalize_object_id(label) or stable_object_id_from_string(label).",
        link_id_col,
        sample,
        link_id_col,
    )
    if link_id_col == LAKE_JOIN_ID_COLUMN:
        raise ValueError(
            f"Column {link_id_col!r} cannot hold non-integer labels; use "
            "--link-id-col with your survey name column (e.g. NAME or SOURCE_ID)."
        )
    hashes = _stable_join_hashes(labels)
    table = _set_lake_join_id_column(table, hashes)
    return table, f"label:{link_id_col}"


def warn_if_id_column_unsafe(
    column_name: str,
    arrow_type: pa.DataType,
    *,
    context: str = "catalog",
) -> None:
    """Log when an ID column is not stored as int64 (risk for large TARGETIDs)."""
    if pa.types.is_unsigned_integer(arrow_type):
        log.info(
            "%s ID column %r is unsigned integer; values above 2**63-1 are stored "
            "as int64 bit patterns at ingest (SDSS objid).",
            context, column_name,
        )
        return
    if pa.types.is_integer(arrow_type):
        return
    if pa.types.is_floating(arrow_type):
        log.warning(
            "%s ID column %r has floating Arrow type %s; values above 2**53 "
            "cannot be represented exactly. Casting to int64 may corrupt "
            "TARGETIDs — re-ingest from FITS with an integer column.",
            context, column_name, arrow_type,
        )
        return
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        log.warning(
            "%s ID column %r is stored as string; matching still works but "
            "prefer int64 at ingest for performance and type safety.",
            context, column_name,
        )


def resolve_link_id_column(
    catalog_root: Path | str,
    *,
    schema_names: list[str] | None = None,
    override: str | None = None,
) -> str:
    """Return the lake join column for catalog ↔ Zarr / cross-match (``_source_id``).

    Reads ``catalog_info.json`` when present.  The join column is always
    :data:`LAKE_JOIN_ID_COLUMN`.
    """
    catalog_root = Path(catalog_root)

    if override:
        matched = (
            match_schema_column(override, schema_names)
            if schema_names is not None
            else override
        )
        if schema_names is None or matched is not None:
            return matched or override
        raise KeyError(
            f"Requested link-ID column {override!r} not in catalog schema. "
            f"Available: {sorted(schema_names)[:30]}"
            f"{'…' if len(schema_names) > 30 else ''}"
        )

    if schema_names is None:
        schema_names = catalog_tile_schema_names(catalog_root)

    if schema_names is not None:
        if LAKE_JOIN_ID_COLUMN in schema_names:
            return LAKE_JOIN_ID_COLUMN
        raise KeyError(
            f"No lake join column {LAKE_JOIN_ID_COLUMN!r} in catalog under {catalog_root}. "
            f"Parquet columns include: "
            f"{sorted(schema_names)[:25]}{'…' if len(schema_names) > 25 else ''}"
        )

    return LAKE_JOIN_ID_COLUMN


def read_parquet_tile(tile_path: Path | str) -> pa.Table:
    """Read one catalog tile file without hive partition discovery.

    ``pq.read_table`` on paths under ``Norder=…/Dir=…/`` can attach partition
    columns and fail when ``Norder`` types differ across tiles.
    """
    return pq.ParquetFile(tile_path).read()


def rebuild_parquet_tile_link_id(
    tile_path: Path | str,
    link_col: str,
    *,
    catalog_parquet_options: CatalogParquetOptions | None = None,
    reset_indices: bool = True,
    allow_incomplete_link_id: bool = False,
) -> tuple[str, str]:
    """Recompute ``_source_id`` from *link_col* on one catalog tile.

    This always replaces an existing ``_source_id`` column (e.g. switch zCOSMOS from ``id`` to
    ``filename`` hash).  Returns ``(status, link_id_mode)`` where *status*
    is ``"rebuilt"``.
    """
    tile_path = Path(tile_path)
    table = read_parquet_tile(tile_path)
    if "," in link_col:
        table, mode = ensure_catalog_source_ids(
            table, link_col, allow_incomplete_link_id=allow_incomplete_link_id,
        )
    else:
        matched = match_schema_column(link_col, table.schema.names)
        if matched is None:
            raise KeyError(
                f"Link column {link_col!r} not in tile {tile_path.name}; "
                f"columns: {sorted(table.schema.names)[:25]}"
            )
        table, mode = ensure_catalog_source_ids(
            table,
            matched,
            allow_incomplete_link_id=allow_incomplete_link_id,
        )
    if reset_indices:
        n = len(table)
        minus_one = pa.array(np.full(n, -1, dtype=np.int64), type=pa.int64())
        for _reset_col in (
            "_spectrum_index", "_spectrum_npix",
            "_cutout_index", "_cutout_npix",
        ):
            if _reset_col in table.schema.names:
                _idx = table.schema.get_field_index(_reset_col)
                table = table.set_column(_idx, _reset_col, minus_one)
    opts = catalog_parquet_options or CatalogParquetOptions()
    pq.write_table(
        table,
        str(tile_path),
        compression="zstd",
        compression_level=opts.compression_level,
        write_statistics=opts.write_statistics,
    )
    return "rebuilt", mode


def is_valid_sky_position(ra: float, dec: float) -> bool:
    """True when RA/Dec in degrees can be mapped to a HEALPix pixel."""
    if not np.isfinite(ra) or not np.isfinite(dec):
        return False
    if dec < -90.0 or dec > 90.0:
        return False
    # SDSS / pipeline sentinels for unused plug-map fibers
    if ra <= -9000.0 or dec <= -9000.0:
        return False
    return True


def valid_sky_position_mask(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Element-wise :func:`is_valid_sky_position` for RA/Dec arrays (vectorized)."""
    ra_deg = np.asarray(ra_deg, dtype=np.float64).reshape(-1)
    dec_deg = np.asarray(dec_deg, dtype=np.float64).reshape(-1)
    if ra_deg.shape != dec_deg.shape:
        raise ValueError(
            f"RA and Dec length mismatch: {ra_deg.shape[0]} vs {dec_deg.shape[0]}"
        )
    return (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & (dec_deg >= -90.0)
        & (dec_deg <= 90.0)
        & (ra_deg > -9000.0)
        & (dec_deg > -9000.0)
    )


def assign_healpix(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    norder: int = 5,
) -> np.ndarray:
    """Return HEALPix NESTED pixel indices for arrays of RA/Dec (degrees)."""
    nside = hp.order2nside(norder)
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    return hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def _astropy_col_to_pyarrow(col) -> pa.Array:
    """Convert one astropy ``Column`` (or ``MaskedColumn``) to a PyArrow Array.

    Preserves multidim columns as ``FixedSizeListArray`` of the same inner
    size — essential for FITS BINTABLE vector columns such as DESI's
    ``COEFF`` (shape ``(N_rows, 10)``).  A rectangular dataframe-style export
    cannot represent arbitrary 2-D cells per row; Arrow ``FixedSizeList`` does.

    Handles:
    * big-endian FITS dtypes → cast to native byte-order (PyArrow requires it)
    * fixed-width byte strings (``|S<n>``) → decode to UTF-8 strings
    * **all** string columns → stored as ``large_string`` (int64 offsets) so
      that downstream ``Table.take`` / ``Table.filter`` operations on
      multi-million-row catalogs cannot hit the 2 GB offset overflow of the
      default ``string`` type
    * 1-D masked columns → propagate the null mask
    * >2-D columns → flattened to a single FixedSizeList whose inner length
      is ``prod(shape[1:])`` (the original inner shape is recorded as
      schema metadata by :func:`_astropy_table_to_arrow`).
    """
    from astropy.table import MaskedColumn

    data = np.asarray(col)
    # GALEX / some VO exports: one logical row per source but stored as (N, 1).
    if data.ndim == 2 and data.shape[1] == 1:
        data = data.reshape(-1)
        if isinstance(col, MaskedColumn) and col.mask is not None:
            col = MaskedColumn(data, mask=np.asarray(col.mask).reshape(-1))

    # FITS BINTABLE is big-endian; PyArrow needs native byte-order.
    if data.dtype.kind in "biufc" and data.dtype.byteorder not in ("=", "|", ""):
        data = data.astype(data.dtype.newbyteorder("="), copy=False)

    if data.ndim == 1:
        if data.dtype.kind == "S":
            data = np.char.decode(data, "utf-8", errors="replace")
        # Strings: force large_string so 28M+ row catalogs don't blow up
        # `table.take` with `offset overflow while concatenating arrays`.
        pa_type = pa.large_string() if data.dtype.kind == "U" else None
        if isinstance(col, MaskedColumn) and col.mask is not None and np.any(col.mask):
            return pa.array(data, type=pa_type, mask=np.asarray(col.mask, dtype=bool))
        return pa.array(data, type=pa_type)

    # ndim >= 2 → FixedSizeList(inner_size)
    n_rows = data.shape[0]
    inner_size = int(np.prod(data.shape[1:]))
    flat = np.ascontiguousarray(data).reshape(n_rows * inner_size)
    if flat.dtype.kind == "S":
        flat = np.char.decode(flat, "utf-8", errors="replace")
    inner_type = pa.large_string() if flat.dtype.kind == "U" else None
    inner = pa.array(flat, type=inner_type)
    return pa.FixedSizeListArray.from_arrays(inner, list_size=inner_size)


def _astropy_table_to_arrow(tbl: Table) -> pa.Table:
    """Convert an astropy Table to a PyArrow Table, preserving multidim columns.

    Unlike conversions that round-trip through a flat dataframe layout, this path
    handles vector/matrix BINTABLE columns (e.g. DESI ``COEFF`` shape (N, 10)
    or per-band fluxes shape (N, 4)) by storing them as Arrow
    ``FixedSizeList`` arrays.  Inner shapes for >2-D columns are recorded
    in ``schema.metadata['data_lake.inner_shapes']`` as a JSON map so the
    original tensor shape can be reconstructed if needed.
    """
    arrays: list[pa.Array] = []
    names: list[str] = []
    inner_shapes: dict[str, list[int]] = {}
    multidim_cols: list[str] = []

    for name in tbl.colnames:
        col = tbl[name]
        data = np.asarray(col)
        if data.ndim > 2:
            inner_shapes[name] = list(map(int, data.shape[1:]))
        if data.ndim >= 2:
            multidim_cols.append(f"{name}{tuple(int(d) for d in data.shape[1:])}")
        arrays.append(_astropy_col_to_pyarrow(col))
        names.append(name)

    table = pa.Table.from_arrays(arrays, names=names)

    if inner_shapes:
        schema_meta = dict(table.schema.metadata or {})
        schema_meta[b"data_lake.inner_shapes"] = json.dumps(inner_shapes).encode()
        table = table.replace_schema_metadata(schema_meta)

    if multidim_cols:
        log.info("Preserved %d multidim column(s) as FixedSizeList: %s",
                 len(multidim_cols), ", ".join(multidim_cols))

    return table


_TFORM_REPEAT_RE = re.compile(r"^\d+[A-Za-z]$")


def _squeeze_fits_vector_column(arr: np.ndarray) -> np.ndarray:
    """Flatten GALEX-style ``(1, N)`` / ``(1, N, 1)`` vector columns to 1-D length ``N``."""
    out = np.asarray(arr)
    while out.ndim > 1 and out.shape[0] == 1:
        out = out[0]
    return np.squeeze(out)


def _bintable_hdu_index(hdul) -> int:
    """Return the index of the first table HDU (skip PRIMARY)."""
    from astropy.io.fits.hdu.table import _TableLikeHDU

    for idx, hdu in enumerate(hdul):
        if isinstance(hdu, _TableLikeHDU):
            return idx
    raise ValueError("No BINTABLE / table HDU found in FITS file.")


def _is_packed_vector_bintable(hdu) -> bool:
    """True when the table is one FITS row of long per-column vectors (GALEX photoobjall)."""
    from astropy.io.fits.hdu.table import _TableLikeHDU

    if not isinstance(hdu, _TableLikeHDU) or hdu.columns is None:
        return False
    if int(hdu.header.get("NAXIS2", 0)) != 1:
        return False
    for col in hdu.columns:
        fmt = str(col.format).strip()
        if _TFORM_REPEAT_RE.match(fmt):
            return True
        dim = col.dim
        if dim:
            parts = [int(x) for x in str(dim).strip("()").split(",") if x.strip()]
            if len(parts) >= 2 and parts[1] > 1:
                return True
    return False


def _read_packed_vector_fits(path: Path, *, hdu_index: int) -> Table:
    """Read one-row vector-packed FITS (e.g. GALEX photoobjall) column-by-column."""
    try:
        import fitsio
    except ImportError as exc:
        raise ImportError(
            "This FITS file stores one row of per-source vectors (GALEX photoobjall layout). "
            "Install the optional fitsio dependency: uv sync --extra fitsio  "
            "(or pip install fitsio)."
        ) from exc

    log.info(
        "Reading %s as packed-vector FITS (NAXIS2=1; column-by-column via fitsio) …",
        path.name,
    )
    cols: dict[str, np.ndarray] = {}
    with fitsio.FITS(str(path)) as fits:
        hdu = fits[hdu_index]
        for name in hdu.get_colnames():
            cols[name] = _squeeze_fits_vector_column(hdu[name][:])
    n = len(next(iter(cols.values())))
    for name, arr in cols.items():
        if len(arr) != n:
            raise ValueError(
                f"Packed FITS column {name!r} has length {len(arr)}, expected {n}."
            )
    return Table(cols)


def is_catalog_fits_path(path: Path | str) -> bool:
    """True for suffixes handled by the FITS catalog ingest path."""
    path = Path(path)
    name = path.name.lower()
    return name.endswith(".fits.gz") or path.suffix.lower() in {".fit", ".fits", ".fz"}


def resolve_catalog_column_name(available: Sequence[str], requested: str) -> str:
    """Match catalog column name exactly or case-insensitively."""
    names = list(available)
    if requested in names:
        return requested
    by_upper = {n.upper(): n for n in names}
    hit = by_upper.get(requested.upper())
    if hit is not None:
        return hit
    preview = ", ".join(names[:12])
    if len(names) > 12:
        preview += f", … (+{len(names) - 12} more)"
    raise KeyError(
        f"Column {requested!r} not in catalog. Available: {preview}"
    )


def _sky_arrays_to_float64(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if isinstance(out, np.ma.MaskedArray):
        out = out.filled(np.nan)
    return out


def read_catalog_sky_columns(
    path: Path | str,
    ra_col: str,
    dec_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read RA/Dec arrays the same way catalog ingest does.

    FITS: uses the catalog BINTABLE HDU and astropy column scaling (BSCALE/TZERO).
    Other formats: full read via :func:`_read_source_table` then sky columns.
    """
    path = Path(path)
    if is_catalog_fits_path(path):
        from data_lake.io.fits_read import open_fits

        with open_fits(path) as hdul:
            idx = _bintable_hdu_index(hdul)
            hdu = hdul[idx]
            if _is_packed_vector_bintable(hdu):
                tbl = _read_packed_vector_fits(path, hdu_index=idx)
            else:
                # Same reader as ingest: correct HDU + FITS BSCALE/TZERO handling.
                tbl = Table.read(hdul, hdu=idx, format="fits", memmap=True)
            ra_name = resolve_catalog_column_name(tbl.colnames, ra_col)
            dec_name = resolve_catalog_column_name(tbl.colnames, dec_col)
            return (
                _sky_arrays_to_float64(tbl[ra_name]),
                _sky_arrays_to_float64(tbl[dec_name]),
            )

    table = _read_source_table(path)
    names = list(table.schema.names)
    ra_name = resolve_catalog_column_name(names, ra_col)
    dec_name = resolve_catalog_column_name(names, dec_col)

    def _arrow_sky(name: str) -> np.ndarray:
        col = table.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
            col = pc.list_flatten(col)
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1)

    return _arrow_sky(ra_name), _arrow_sky(dec_name)


def catalog_source_row_count(path: Path | str) -> int:
    """Row count for a catalog input file (FITS header or Parquet footer)."""
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".parquet", ".pq"}:
        return int(pq.read_metadata(str(path)).num_rows)
    if is_catalog_fits_path(path):
        from data_lake.io.fits_read import open_fits

        with open_fits(path) as hdul:
            idx = _bintable_hdu_index(hdul)
            hdu = hdul[idx]
            if _is_packed_vector_bintable(hdu):
                return len(_read_packed_vector_fits(path, hdu_index=idx))
            data = hdu.data
            if data is not None:
                return len(data)
            return int(hdu.header.get("NAXIS2", 0))
    return len(_read_source_table(path))


def _read_fits_catalog_table(path: Path, *, fits_read_policy=None) -> Table:
    """Read a catalog FITS BINTABLE, including GALEX packed-vector layout."""
    from astropy.table import Table

    from data_lake.io.fits_read import FitsReadPolicy, open_fits, resolve_memmap

    policy = fits_read_policy or FitsReadPolicy.from_env()
    with open_fits(path, policy) as hdul:
        idx = _bintable_hdu_index(hdul)
        hdu = hdul[idx]
        if _is_packed_vector_bintable(hdu):
            return _read_packed_vector_fits(path, hdu_index=idx)
        log.info("Reading %s as FITS (HDU %d) …", path.name, idx)
        return Table.read(hdul, hdu=idx, format="fits", memmap=resolve_memmap(path, policy))


def _open_text_for_sniff(path: Path):
    """Text read handle; transparent gzip for ``*.gz`` paths."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _sniff_text_delimiter(path: Path, *, default: str = ",") -> str:
    """Guess field delimiter from the first lines (comma, tab, semicolon, pipe)."""
    with _open_text_for_sniff(path) as fh:
        sample = fh.read(65536)
    lines = [ln for ln in sample.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return default
    header = lines[0]
    if header.count("\t") > header.count(",") and header.count("\t") >= 1:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return default


def _read_delimited_text_table(path: Path) -> Table:
    """Read CSV/TSV (optionally ``.gz``) with delimiter sniffing and a safe fallback."""
    name = path.name.lower()
    if name.endswith(".tsv.gz") or name.endswith(".tsv"):
        delimiter = "\t"
    else:
        delimiter = _sniff_text_delimiter(path, default=",")

    log.info("Reading %s as ascii.csv (delimiter=%r) …", path.name, delimiter)
    read_kwargs: dict = {
        "format": "ascii.csv",
        "delimiter": delimiter,
        "comment": "#",
        "fast_reader": False,
    }
    try:
        return Table.read(str(path), **read_kwargs)
    except Exception as first_exc:
        log.warning(
            "Delimiter %r failed for %s (%s); retrying with astropy guess=True.",
            delimiter,
            path.name,
            first_exc,
        )
        return Table.read(str(path), format="ascii.csv", guess=True, fast_reader=False)


def _read_source_table(path: Path, *, fits_read_policy=None) -> pa.Table:
    """Read a catalog file into a PyArrow Table.

    Supported inputs (by file suffix):
    * ``.fits`` / ``.fit`` / ``.fz`` / ``.fits.gz``  – FITS BINTABLE via astropy
    * ``.xml`` / ``.vot`` / ``.votable``             – VOTable via astropy
    * ``.parquet`` / ``.pq``                         – Parquet via pyarrow (direct)
    * ``.ecsv``                                      – ECSV via astropy
    * ``.csv`` / ``.tsv`` / ``.csv.gz`` / ``.tsv.gz`` – delimited text via astropy
    * anything else                                  – astropy auto-detect

    Multidim FITS columns (e.g. DESI ``COEFF``) are preserved as
    ``FixedSizeList`` arrays rather than failing on non-scalar cells.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        log.info("Reading %s as Parquet …", path.name)
        return pq.read_table(str(path))

    if name.endswith(".fits.gz") or suffix in {".fit", ".fits", ".fz"}:
        return _astropy_table_to_arrow(
            _read_fits_catalog_table(path, fits_read_policy=fits_read_policy)
        )
    elif suffix in {".xml", ".vot", ".votable"}:
        fmt = "votable"
    elif suffix == ".ecsv":
        fmt = "ascii.ecsv"
    elif (
        name.endswith(".csv.gz")
        or name.endswith(".tsv.gz")
        or suffix in {".csv", ".tsv"}
    ):
        return _astropy_table_to_arrow(_read_delimited_text_table(path))

    fmt = None  # let astropy auto-detect

    log.info("Reading %s as %s …", path.name, fmt or "auto-detect")
    astropy_table = Table.read(str(path)) if fmt is None else Table.read(str(path), format=fmt)
    return _astropy_table_to_arrow(astropy_table)


def _add_healpix_columns(
    table: pa.Table,
    ra_col: str,
    dec_col: str,
    norder: int,
) -> pa.Table:
    """Append ``_healpix_order<N>`` and ``_cutout_index`` placeholder columns."""

    def _sky_to_float64(col_name: str) -> np.ndarray:
        col = table.column(col_name).combine_chunks()
        if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
            col = pc.list_flatten(col)
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1)

    ra = _sky_to_float64(ra_col)
    dec = _sky_to_float64(dec_col)
    valid = valid_sky_position_mask(ra, dec)
    if not valid.all():
        bad_idx = np.flatnonzero(~valid)[:5]
        n_bad = int((~valid).sum())
        examples = "; ".join(
            f"row {int(i)}: {ra_col}={ra[i]!r}, {dec_col}={dec[i]!r}"
            for i in bad_idx
        )
        raise ValueError(
            f"{n_bad} catalog row(s) have invalid sky coordinates "
            f"({ra_col!r}/{dec_col!r}; checked with is_valid_sky_position). "
            f"Examples: {examples}"
        )
    pix = assign_healpix(ra, dec, norder)
    col_name = f"_healpix_norder{norder}"
    table = table.append_column(col_name, pa.array(pix, type=pa.int64()))
    # cutout_index, spectrum_index, and their modality-specific Npix columns are
    # filled by the respective ingest steps; initialise all to -1.
    for colname in (
        "_cutout_index",
        "_cutout_npix",
        "_spectrum_index",
        "_spectrum_npix",
    ):
        if colname not in table.schema.names:
            table = table.append_column(
                colname, pa.array(np.full(len(table), -1, dtype=np.int64), type=pa.int64())
            )
    return table


def _filter_table_columns(
    table: pa.Table,
    columns: Sequence[str] | None,
    *,
    ra_col: str,
    dec_col: str,
    link_id_col: str | None,
    norder: int,
) -> pa.Table:
    """Keep only requested columns plus sky, ID, HEALPix, and index placeholders."""
    if not columns:
        return table
    hp_col = f"_healpix_norder{norder}"
    required = {
        ra_col, dec_col, hp_col,
        "_cutout_index", "_cutout_npix",
        "_spectrum_index", "_spectrum_npix",
    }
    if link_id_col:
        required.add(link_id_col)
    required.add(LAKE_JOIN_ID_COLUMN)
    keep: list[str] = []
    seen: set[str] = set()
    for name in list(columns) + sorted(required):
        if name in table.schema.names and name not in seen:
            keep.append(name)
            seen.add(name)
    missing = required - seen
    if missing:
        raise KeyError(
            f"Required column(s) missing after column filter: {sorted(missing)}. "
            f"Available: {table.schema.names[:30]}"
        )
    log.info("Column subset: keeping %d / %d columns", len(keep), len(table.schema))
    return table.select(keep)


def _streaming_fits_read_columns(
    columns: Sequence[str] | None,
    *,
    col_names: list[str],
    ra_col: str,
    dec_col: str,
    link_id_col: str | None,
) -> list[str] | None:
    """Column names to read from FITS memmap when ``--columns`` is set."""
    if not columns:
        return None
    required = {ra_col, dec_col}
    if link_id_col:
        for part in link_id_col.split("+"):
            part = part.strip()
            if part:
                required.add(part)
    read = []
    seen: set[str] = set()
    for name in list(columns) + sorted(required):
        if name in col_names and name not in seen:
            read.append(name)
            seen.add(name)
    return read


def _shrink_tile_table_for_disk(table: pa.Table) -> pa.Table:
    """Narrow ``large_string`` → ``string`` per tile when safe (smaller on disk).

    Skips narrowing when the tile is very large or PyArrow reports the UTF-8
    payload would exceed the 2 GiB ``string`` offset limit (common on dense
    DESI HEALPix tiles with many string columns).
    """
    if table.num_rows > _MAX_ROWS_STRING_SHRINK:
        log.info(
            "Keeping large_string columns for tile with %d rows "
            "(>%d row shrink threshold).",
            table.num_rows, _MAX_ROWS_STRING_SHRINK,
        )
        return table

    arrays: list[pa.Array] = []
    for name in table.schema.names:
        col = table.column(name)
        if not pa.types.is_large_string(col.type):
            arrays.append(col)
            continue
        try:
            arrays.append(pc.cast(col, pa.string()))
        except pa.ArrowInvalid as exc:
            if "too large" not in str(exc).lower():
                raise
            log.warning(
                "Column %r: cannot narrow large_string to string for %d-row tile "
                "(%s); keeping large_string on disk.",
                name, table.num_rows, exc,
            )
            arrays.append(col)
    return pa.Table.from_arrays(arrays, names=table.schema.names)


def _parquet_tile_tmp_path(out_file: Path) -> Path:
    return out_file.with_name(out_file.name + ".tmp")


def _is_valid_parquet_tile(path: Path) -> bool:
    """Return False for missing, empty, or truncated (aborted) Parquet tiles."""
    if not path.is_file() or path.stat().st_size < 8:
        return False
    try:
        pq.read_metadata(str(path))
        return True
    except Exception:
        return False


def _remove_stale_parquet_tmp_files(catalog_root: Path) -> None:
    """Drop leftover ``*.parquet.tmp`` from killed ingest jobs."""
    for tmp in catalog_root.rglob("*.parquet.tmp"):
        log.warning("Removing stale temporary tile %s", tmp)
        try:
            tmp.unlink()
        except OSError as exc:
            log.warning("Could not remove %s: %s", tmp, exc)


def _read_catalog_tile(path: Path) -> pa.Table:
    """Read one HATS tile without injecting Hive partition columns."""
    if not _is_valid_parquet_tile(path):
        raise ValueError(f"Not a valid Parquet tile: {path}")
    return pq.ParquetFile(str(path)).read()


def canonical_arrow_type(dtype: pa.DataType) -> pa.DataType:
    """Map a column type to the catalog-wide canonical numeric storage type."""
    if pa.types.is_floating(dtype):
        return pa.float64()
    if pa.types.is_integer(dtype) or pa.types.is_unsigned_integer(dtype):
        return pa.int64()
    if pa.types.is_boolean(dtype):
        return pa.bool_()
    if pa.types.is_fixed_size_list(dtype):
        inner = canonical_arrow_type(dtype.value_type)
        if inner.equals(dtype.value_type):
            return dtype
        return pa.list_(inner, dtype.list_size)
    return dtype


def _canonical_merge_types(existing: pa.DataType, incoming: pa.DataType) -> pa.DataType:
    """Pick one Arrow type for append when two files disagree (e.g. float32 vs float64)."""
    if existing.equals(incoming):
        return canonical_arrow_type(existing)
    ce, ci = canonical_arrow_type(existing), canonical_arrow_type(incoming)
    if ce.equals(ci):
        return ce
    if pa.types.is_floating(existing) and pa.types.is_floating(incoming):
        return pa.float64()
    if (
        pa.types.is_integer(existing) or pa.types.is_unsigned_integer(existing)
    ) and (pa.types.is_integer(incoming) or pa.types.is_unsigned_integer(incoming)):
        return pa.int64()
    return ce


def normalize_catalog_table_types(table: pa.Table) -> pa.Table:
    """Coerce numeric columns to float64 / int64 so multi-file ingest shares one schema.

    Survey batches (AllWISE, GALEX, …) often ship the same column as ``E`` (float32)
    in one FITS file and ``D`` (float64) in another.  Normalizing on ingest avoids
    append failures and stops the first file from locking a narrower Parquet type.
    """
    if table.num_rows == 0:
        return table
    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for field in table.schema:
        canon = canonical_arrow_type(field.type)
        col = table.column(field.name).combine_chunks()
        if not col.type.equals(canon):
            col = pc.cast(col, canon)
        arrays.append(col)
        fields.append(pa.field(field.name, canon, nullable=field.nullable))
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields, metadata=table.schema.metadata),
    )


def _schema_needs_canonicalization(schema: pa.Schema) -> bool:
    """True when any column is not already at catalog-wide canonical dtypes."""
    return any(
        not field.type.equals(canonical_arrow_type(field.type))
        for field in schema
    )


def _reconcile_catalog_tile_dtypes(
    tile_path: Path,
    parquet_options: CatalogParquetOptions,
) -> bool:
    """Rewrite one tile when on-disk dtypes are not canonical (e.g. float32 ``flux``).

    Returns True if the file was rewritten.
    """
    schema = pq.read_schema(str(tile_path))
    if not _schema_needs_canonicalization(schema):
        return False
    table = _read_catalog_tile(tile_path)
    normalized = normalize_catalog_table_types(table)
    _write_catalog_parquet_tile(normalized, tile_path, parquet_options)
    return True


def reconcile_catalog_tile_dtypes(
    catalog_root: Path | str,
    *,
    parquet_options: CatalogParquetOptions | None = None,
) -> int:
    """Rewrite tiles whose Parquet dtypes are not canonical so ``_metadata`` can be built.

    Multi-file ingest normalizes **incoming** rows before each tile write, but tiles
    that were never appended again (or were written before normalization) can still
    store float32 while newer tiles use float64.  ``_regenerate_metadata_from_all_tiles``
    calls this automatically; use directly to repair an existing survey directory.
    """
    catalog_root = Path(catalog_root)
    pq_opts = parquet_options or CatalogParquetOptions()
    n_rewritten = 0
    for path in _iter_valid_parquet_tiles(catalog_root):
        if _reconcile_catalog_tile_dtypes(path, pq_opts):
            n_rewritten += 1
    if n_rewritten:
        log.info(
            "Reconciled dtypes on %d catalog tile(s) under %s",
            n_rewritten,
            catalog_root,
        )
    return n_rewritten


def _unified_append_schema(existing: pa.Schema, incoming: pa.Schema) -> pa.Schema:
    """Build a target schema that can hold both *existing* and *incoming* tiles."""
    if existing.names != incoming.names:
        raise ValueError(
            "Cannot align tables: column name mismatch.\n"
            f"On disk: {existing.names}\nIncoming: {incoming.names}"
        )
    fields: list[pa.Field] = []
    for name in existing.names:
        ex_f = existing.field(name)
        in_f = incoming.field(name)
        merged_type = _canonical_merge_types(ex_f.type, in_f.type)
        fields.append(
            pa.field(
                name,
                merged_type,
                nullable=ex_f.nullable or in_f.nullable,
            )
        )
    return pa.schema(fields, metadata=incoming.metadata)


def _schemas_compatible(existing: pa.Schema, incoming: pa.Schema) -> bool:
    if existing.names != incoming.names:
        return False
    return all(
        existing.field(i).type.equals(incoming.field(i).type)
        for i in range(len(existing))
    )


def _schema_type_mismatches(existing: pa.Schema, incoming: pa.Schema) -> list[str]:
    """Human-readable per-column type diffs (same names required)."""
    mismatches: list[str] = []
    for i, name in enumerate(existing.names):
        ex_t = existing.field(i).type
        in_t = incoming.field(i).type
        if not ex_t.equals(in_t):
            mismatches.append(f"  {name!r}: on disk {ex_t} vs incoming {in_t}")
    return mismatches


def _align_incoming_to_schema(incoming: pa.Table, target: pa.Schema) -> pa.Table:
    """Cast *incoming* columns to match a target tile schema.

    Handles common re-ingest drift (e.g. ``large_string`` in RAM vs ``string``
    written by :func:`_shrink_tile_table_for_disk`, or float32 vs float64).
    """
    if incoming.schema.names != target.names:
        raise ValueError(
            "Cannot align tables: column name mismatch.\n"
            f"On disk: {target.names}\nIncoming: {incoming.schema.names}"
        )
    arrays: list[pa.Array] = []
    for i, name in enumerate(target.names):
        col = incoming.column(name).combine_chunks()
        tgt_type = target.field(i).type
        if col.type.equals(tgt_type):
            arrays.append(col)
        else:
            try:
                arrays.append(pc.cast(col, tgt_type))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                diffs = _schema_type_mismatches(target, incoming.schema)
                detail = "\n".join(diffs) if diffs else str(exc)
                raise ValueError(
                    "Cannot append catalog tile: incompatible column types.\n"
                    f"{detail}"
                ) from exc
    return pa.Table.from_arrays(arrays, schema=target)


def _id_column_for_dedup(table: pa.Table, link_id_col: str | None) -> str | None:
    if LAKE_JOIN_ID_COLUMN in table.schema.names:
        return LAKE_JOIN_ID_COLUMN
    if link_id_col and link_id_col in table.schema.names:
        return link_id_col
    return None


def _object_ids_from_column(table: pa.Table, col: str) -> set[int]:
    out: set[int] = set()
    for v in table.column(col).to_pylist():
        if v is None:
            continue
        out.add(normalize_object_id(v))
    return out


def _filter_table_exclude_ids(table: pa.Table, col: str, exclude: set[int]) -> pa.Table:
    if not exclude:
        return table
    keep = [
        v is None or normalize_object_id(v) not in exclude
        for v in table.column(col).to_pylist()
    ]
    return table.filter(pa.array(keep, type=pa.bool_()))


def _merge_tile_tables(
    existing: pa.Table,
    incoming: pa.Table,
    *,
    sid_col: str | None,
    on_duplicate_id: DuplicateIdMode,
) -> pa.Table:
    """Concatenate tile tables, optionally deduplicating on an object-ID column."""
    existing = normalize_catalog_table_types(existing)
    incoming = normalize_catalog_table_types(incoming)
    target = _unified_append_schema(existing.schema, incoming.schema)
    existing = _align_incoming_to_schema(existing, target)
    incoming = _align_incoming_to_schema(incoming, target)
    if not _schemas_compatible(existing.schema, incoming.schema):
        diffs = _schema_type_mismatches(existing.schema, incoming.schema)
        raise ValueError(
            "Cannot append catalog tile: schema mismatch between existing tile "
            "and incoming rows.\n"
            + ("\n".join(diffs) if diffs else (
                f"Existing columns: {existing.schema.names}\n"
                f"Incoming columns: {incoming.schema.names}"
            ))
        )
    if sid_col is None:
        return pa.concat_tables([existing, incoming])

    existing_ids = _object_ids_from_column(existing, sid_col)
    incoming_ids = _object_ids_from_column(incoming, sid_col)
    overlap = existing_ids & incoming_ids

    if overlap:
        if on_duplicate_id == "error":
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"Duplicate object ID(s) when appending catalog tile: "
                f"{len(overlap)} overlap(s), e.g. {sample}"
                f"{'…' if len(overlap) > 5 else ''}"
            )
        if on_duplicate_id == "skip":
            incoming = _filter_table_exclude_ids(incoming, sid_col, existing_ids)
            if incoming.num_rows == 0:
                return existing
        else:  # last
            existing = _filter_table_exclude_ids(existing, sid_col, incoming_ids)

    return pa.concat_tables([existing, incoming])


def _write_tile_for_mode(
    out_file: Path,
    incoming: pa.Table,
    *,
    tile_mode: TileMode,
    on_duplicate_id: DuplicateIdMode,
    link_id_col: str | None,
    parquet_options: CatalogParquetOptions,
) -> pq.FileMetaData | None:
    """Write one ``Npix=*.parquet`` tile; return metadata if written, else None."""
    incoming = normalize_catalog_table_types(incoming)
    if not out_file.exists():
        return _write_catalog_parquet_tile(incoming, out_file, parquet_options)

    if tile_mode == "skip":
        log.debug("Skip existing tile %s", out_file)
        return None

    if tile_mode == "overwrite":
        return _write_catalog_parquet_tile(incoming, out_file, parquet_options)

    if not _is_valid_parquet_tile(out_file):
        log.warning(
            "Existing tile %s is missing or corrupt (likely aborted write); "
            "replacing with incoming rows only.",
            out_file,
        )
        return _write_catalog_parquet_tile(incoming, out_file, parquet_options)

    existing = _read_catalog_tile(out_file)
    sid = _id_column_for_dedup(existing, link_id_col)
    merged = _merge_tile_tables(
        existing,
        incoming,
        sid_col=sid,
        on_duplicate_id=on_duplicate_id,
    )
    if merged.num_rows == existing.num_rows:
        log.debug("Append left tile unchanged %s", out_file)
        return None
    return _write_catalog_parquet_tile(merged, out_file, parquet_options)


def _iter_valid_parquet_tiles(catalog_root: Path) -> list[Path]:
    """Return ``Npix=*.parquet`` paths that pass a footer read (skip corrupt tiles)."""
    valid: list[Path] = []
    for path in sorted(catalog_root.rglob("Npix=*.parquet")):
        if _is_valid_parquet_tile(path):
            valid.append(path)
        else:
            log.warning("Skipping corrupt/incomplete tile for metadata: %s", path)
    return valid


def _regenerate_metadata_from_all_tiles(catalog_root: Path) -> None:
    """Rebuild ``_metadata`` from every ``Npix=*.parquet`` under *catalog_root*."""
    reconcile_catalog_tile_dtypes(catalog_root)
    tile_paths = _iter_valid_parquet_tiles(catalog_root)
    if not tile_paths:
        return
    file_metadata: list[pq.FileMetaData] = []
    schema: pa.Schema | None = None
    for path in tile_paths:
        meta = pq.read_metadata(str(path))
        file_metadata.append(meta)
        arrow_schema = meta.schema.to_arrow_schema()
        if schema is None:
            schema = arrow_schema
        else:
            schema = _unified_append_schema(schema, arrow_schema)
    if schema is not None:
        _write_aggregate_metadata(catalog_root, file_metadata, schema)


def _count_catalog_rows(catalog_root: Path) -> int:
    return sum(
        pq.read_metadata(str(p)).num_rows
        for p in _iter_valid_parquet_tiles(catalog_root)
    )


def finalize_catalog_survey(
    catalog_root: Path | str,
    survey_name: str,
    norder: int,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    link_id_mode: str | None = None,
    streaming: bool = False,
    fallback_n_cols: int = 0,
    allow_incomplete_link_id: bool | None = None,
) -> bool:
    """Refresh ``catalog_info.json``, ``_metadata``, and ``schema_manifest.json`` from tiles.

    Merges ``ra``/``dec``/``link_id_mode``/``hats_order`` from existing
    ``catalog_info.json`` when present.  Returns False when there are no valid tiles.
    """
    catalog_root = Path(catalog_root)
    ra = ra_col
    dec = dec_col
    sid = link_id_mode if link_id_mode is not None else "sequential"
    hats_order = norder
    stream = streaming
    allow_incomplete = allow_incomplete_link_id
    info_path = catalog_root / "catalog_info.json"
    if info_path.is_file():
        with open(info_path) as fh:
            info = json.load(fh)
        ra = info.get("ra_column", ra)
        dec = info.get("dec_column", dec)
        if link_id_mode is None:
            sid = info.get("link_id_mode", sid)
        hats_order = int(info.get("hats_order", hats_order))
        stream = bool(info.get("ingest_streaming", stream))
        if allow_incomplete is None:
            allow_incomplete = bool(info.get("allow_incomplete_link_id", False))

    if not _iter_valid_parquet_tiles(catalog_root):
        return False

    _finalize_catalog_writes(
        catalog_root,
        survey_name,
        hats_order,
        ra_col=ra,
        dec_col=dec,
        link_id_mode=sid,
        streaming=stream,
        fallback_n_cols=fallback_n_cols,
        allow_incomplete_link_id=allow_incomplete,
    )
    return True


def _finalize_catalog_writes(
    catalog_root: Path,
    survey_name: str,
    norder: int,
    *,
    ra_col: str,
    dec_col: str,
    link_id_mode: str,
    streaming: bool,
    fallback_n_cols: int,
    allow_incomplete_link_id: bool | None = None,
) -> None:
    """Refresh ``_metadata`` and ``catalog_info.json`` from all on-disk tiles."""
    catalog_root.mkdir(parents=True, exist_ok=True)
    tile_paths = _iter_valid_parquet_tiles(catalog_root)
    total_rows = _count_catalog_rows(catalog_root)
    n_cols = (
        len(pq.read_schema(str(tile_paths[0])))
        if tile_paths
        else fallback_n_cols
    )
    try:
        _regenerate_metadata_from_all_tiles(catalog_root)
    except Exception:
        log.exception(
            "Could not rebuild catalog _metadata under %s; "
            "catalog_info and schema_manifest will still be updated",
            catalog_root,
        )
    native_col: str | None = native_id_column_from_mode(link_id_mode)
    if tile_paths:
        tile_schema = pq.read_schema(str(tile_paths[0])).names
        try:
            resolve_link_id_column(catalog_root, schema_names=tile_schema)
        except KeyError:
            pass
        if LAKE_JOIN_ID_COLUMN in tile_schema and link_id_mode == "sequential":
            inferred = infer_native_id_column(tile_schema)
            if inferred is not None:
                link_id_mode = f"column:{inferred}"
                native_col = inferred
                log.info(
                    "Catalog %s: recording link_id_mode=%r from tile schema.",
                    survey_name,
                    link_id_mode,
                )
        if native_col is None and link_id_mode.startswith(("column:", "label:")):
            native_col = native_id_column_from_mode(link_id_mode)

    join_col = LAKE_JOIN_ID_COLUMN
    info_path = catalog_root / "catalog_info.json"
    if info_path.exists():
        with open(info_path) as fh:
            info = json.load(fh)
        info["total_rows"] = total_rows
        info["total_columns"] = n_cols
        info["hats_order"] = norder
        info["link_id_column"] = join_col
        info["link_id_mode"] = link_id_mode
        if native_col:
            info["native_id_column"] = native_col
        elif "native_id_column" in info:
            del info["native_id_column"]
        if allow_incomplete_link_id is not None:
            if allow_incomplete_link_id:
                info["allow_incomplete_link_id"] = True
            else:
                info.pop("allow_incomplete_link_id", None)
        with open(info_path, "w") as fh:
            json.dump(info, fh, indent=2)
    else:
        _write_catalog_info(
            catalog_root,
            survey_name,
            norder,
            total_rows,
            n_cols,
            ra_column=ra_col,
            dec_column=dec_col,
            link_id_mode=link_id_mode,
            link_id_column=join_col,
            native_id_column=native_col,
            streaming=streaming,
            allow_incomplete_link_id=bool(allow_incomplete_link_id),
        )

    if tile_paths:
        from data_lake.schema_registry import write_catalog_schema_manifest

        write_catalog_schema_manifest(
            catalog_root,
            survey_name,
            hats_order=norder,
            ra_column=ra_col,
            dec_column=dec_col,
            link_id_mode=link_id_mode,
            total_rows=total_rows,
        )


def _write_catalog_parquet_tile(
    tile_table: pa.Table,
    out_file: Path,
    options: CatalogParquetOptions,
) -> pq.FileMetaData:
    """Write one tile atomically (``*.parquet.tmp`` then ``os.replace``)."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if options.shrink_strings_for_tiles:
        tile_table = _shrink_tile_table_for_disk(tile_table)
    tmp = _parquet_tile_tmp_path(out_file)
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    try:
        writer = pq.ParquetWriter(
            str(tmp),
            tile_table.schema,
            compression="zstd",
            compression_level=options.compression_level,
            write_statistics=options.write_statistics,
            use_dictionary=options.use_dictionary,
        )
        try:
            writer.write_table(tile_table)
        finally:
            writer.close()
        os.replace(tmp, out_file)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return pq.read_metadata(str(out_file))


def ingest_catalog(
    source_path: Path | str,
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "ra",
    dec_col: str = "dec",
    norder: int = 5,
    link_id_col: str | None = None,
    tile_mode: TileMode | None = None,
    on_duplicate_id: DuplicateIdMode = "skip",
    streaming: bool = False,
    columns: Sequence[str] | None = None,
    parquet_options: CatalogParquetOptions | None = None,
    compact: bool = False,
    allow_incomplete_link_id: bool = False,
    fits_memmap: str = "auto",
    streaming_parallel: int = 0,
) -> None:
    """
    Ingest a single FITS/VOTable file into HATS-partitioned Parquet.

    Parameters
    ----------
    source_path:
        Path to the input FITS or VOTable file.
    output_root:
        Root of the data lake (e.g. ``/data/lake``). Survey files are written
        under ``<output_root>/catalogs/<survey_name>/``.
    survey_name:
        Short name used for the output directory (e.g. ``"des_dr2"``).
    ra_col / dec_col:
        Column names for right ascension and declination in degrees.
    norder:
        HEALPix order for partitioning (default 5 → ~12k tiles of ~3.7 deg²).
    link_id_col:
        **Required.** Survey object ID column (e.g. ``TARGETID``, ``SOURCE_ID``).
        Integer columns are stored as ``int64`` in ``_source_id``.  Non-integer
        string labels are kept in that column and ``_source_id`` is a stable hash.
        Decimal ASCII strings (DESI ``TARGETID``) are parsed as integers.
    tile_mode:
        How to handle an existing ``Npix=*.parquet`` tile: ``skip`` (default),
        ``overwrite`` (replace), or ``append`` (read–concat–write).
    on_duplicate_id:
        When ``tile_mode="append"`` and an ID column exists (``link_id_col``
        or auto ``source_id``): ``skip`` drops incoming duplicates, ``error``
        fails, ``last`` replaces existing rows with the same ID.
    streaming:
        When ``True`` (FITS input only), memory-map the input and write
        one tile at a time using per-tile fancy indexing.  Memory peak is
        bounded to ~one tile's worth of rows, not the full table.  Use this
        for catalogs ≳ 50 M rows on machines where you don't want to spend
        ~3× the raw table size in RAM (full copy + sorted copy + per-tile
        slice).  Slightly slower per-tile due to scattered I/O.  See
        :func:`_ingest_catalog_streaming` for the implementation.
    columns:
        If set, only these FITS columns (plus required sky/ID/HEALPix/index
        columns) are written.  Largest win for DESI's ~140-column tables.
    parquet_options:
        ZSTD level, statistics, dictionary encoding, and per-tile string
        narrowing.  Ignored when ``compact=True`` (uses :meth:`CatalogParquetOptions.compact`).
    compact:
        Preset for smaller files: ZSTD level 9, no column statistics, no
        dictionary encoding, narrow string type per tile.
    allow_incomplete_link_id:
        When True, composite or string ``--link-id-col`` rows with any missing
        link part keep null ``_source_id`` (catalog row retained, not linked to
        spectra).  Recorded in ``catalog_info.json`` for rebuild/repair.
    """
    source_path = Path(source_path)
    output_root = Path(output_root)
    catalog_root = output_root / "catalogs" / survey_name
    catalog_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_parquet_tmp_files(catalog_root)
    pq_opts = (
        CatalogParquetOptions.compact()
        if compact
        else (parquet_options or CatalogParquetOptions())
    )
    resolved_tile_mode: TileMode = tile_mode if tile_mode is not None else "skip"

    from data_lake.io.fits_read import FitsReadPolicy, parse_memmap_mode

    env_policy = FitsReadPolicy.from_env()
    fits_read_policy = FitsReadPolicy(
        memmap=parse_memmap_mode(fits_memmap),
        small_file_bytes=env_policy.small_file_bytes,
        parallel_catalog_max_bytes=env_policy.parallel_catalog_max_bytes,
    )

    if streaming:
        if streaming_parallel > 1:
            if resolved_tile_mode != "append":
                raise ValueError(
                    "--streaming-parallel requires tile_mode='append' (use --tile-mode append)."
                )
            _ingest_catalog_streaming_parallel(
                source_path,
                catalog_root,
                survey_name,
                streaming_parallel=streaming_parallel,
                ra_col=ra_col,
                dec_col=dec_col,
                norder=norder,
                link_id_col=link_id_col,
                tile_mode=resolved_tile_mode,
                on_duplicate_id=on_duplicate_id,
                columns=columns,
                parquet_options=pq_opts,
                allow_incomplete_link_id=allow_incomplete_link_id,
                fits_read_policy=fits_read_policy,
            )
            return
        _ingest_catalog_streaming(
            source_path=source_path,
            catalog_root=catalog_root,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=norder,
            link_id_col=link_id_col,
            tile_mode=resolved_tile_mode,
            on_duplicate_id=on_duplicate_id,
            columns=columns,
            parquet_options=pq_opts,
            allow_incomplete_link_id=allow_incomplete_link_id,
            fits_read_policy=fits_read_policy,
        )
        return

    table = _read_source_table(source_path, fits_read_policy=fits_read_policy)
    log.info("Loaded %d rows × %d columns", len(table), len(table.schema))

    table, sid_mode = ensure_catalog_source_ids(
        table, link_id_col, allow_incomplete_link_id=allow_incomplete_link_id,
    )

    table = _add_healpix_columns(table, ra_col, dec_col, norder)
    table = _filter_table_columns(
        table, columns, ra_col=ra_col, dec_col=dec_col,
        link_id_col=link_id_col, norder=norder,
    )
    hp_col = f"_healpix_norder{norder}"

    # Sort by healpix for locality (so per-tile slicing is contiguous and O(1))
    sort_indices = pa.compute.sort_indices(table, sort_keys=[(hp_col, "ascending")])
    table = table.take(sort_indices)

    # Find tile boundaries vectorised; replaces a per-tile O(N) filter+mask
    # scan that became prohibitive at 28M rows.  np.unique on a sorted
    # int64 column with return_index gives the start offset of each tile;
    # pa.Table.slice is O(1) (just adjusts column offsets, no data copy).
    pix_np = np.asarray(table.column(hp_col))
    unique_pixels, group_starts = np.unique(pix_np, return_index=True)
    group_starts = np.append(group_starts, len(pix_np))
    log.info("Writing %d HEALPix tiles at Norder=%d …", len(unique_pixels), norder)

    tiles_written = 0
    t0 = time.perf_counter()
    for g, npix in enumerate(unique_pixels.tolist()):
        s, e = int(group_starts[g]), int(group_starts[g + 1])
        tile_table = table.slice(s, e - s)

        out_dir = catalog_root / healpix_dir(norder, int(npix))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"Npix={int(npix)}.parquet"

        if _write_tile_for_mode(
            out_file,
            tile_table,
            tile_mode=resolved_tile_mode,
            on_duplicate_id=on_duplicate_id,
            link_id_col=link_id_col,
            parquet_options=pq_opts,
        ) is not None:
            tiles_written += 1

    elapsed = time.perf_counter() - t0
    log.info(
        "Processed %d HEALPix tiles (%d written/updated) in %.1f s",
        len(unique_pixels), tiles_written, elapsed,
    )

    _finalize_catalog_writes(
        catalog_root,
        survey_name,
        norder,
        ra_col=ra_col,
        dec_col=dec_col,
        link_id_mode=sid_mode,
        streaming=False,
        fallback_n_cols=len(table.schema),
        allow_incomplete_link_id=allow_incomplete_link_id,
    )
    log.info("Catalog written to %s", catalog_root)


def catalog_table_to_tile_batches(
    table: pa.Table,
    norder: int,
) -> list[tuple[int, pa.Table]]:
    """Split a prepared catalog table into per-HEALPix ``(npix, slice)`` pairs."""
    hp_col = f"_healpix_norder{norder}"
    if hp_col not in table.schema.names:
        raise KeyError(f"Table missing {hp_col!r}; call _add_healpix_columns first.")

    sort_indices = pa.compute.sort_indices(table, sort_keys=[(hp_col, "ascending")])
    table = table.take(sort_indices)
    pix_np = np.asarray(table.column(hp_col))
    unique_pixels, group_starts = np.unique(pix_np, return_index=True)
    group_starts = np.append(group_starts, len(pix_np))

    batches: list[tuple[int, pa.Table]] = []
    for g, npix in enumerate(unique_pixels.tolist()):
        s, e = int(group_starts[g]), int(group_starts[g + 1])
        batches.append((int(npix), table.slice(s, e - s)))
    return batches


def decode_catalog_file_to_batches(
    source_path: Path | str,
    *,
    ra_col: str,
    dec_col: str,
    norder: int,
    link_id_col: str | None = None,
    columns: Sequence[str] | None = None,
    allow_incomplete_link_id: bool = False,
    fits_read_policy=None,
) -> tuple[list[tuple[int, pa.Table]], str, int]:
    """Read one catalog file and partition rows by HEALPix tile (in-memory).

    Returns ``(tile_batches, link_id_mode, n_rows)``.  Used by the parallel
    file-list ingest; each worker holds one full file in RAM.
    """
    from data_lake.io.fits_read import FitsReadPolicy, check_parallel_catalog_file_size

    source_path = Path(source_path)
    policy = fits_read_policy or FitsReadPolicy.from_env()
    if is_catalog_fits_path(source_path):
        check_parallel_catalog_file_size(source_path, policy)
    table = _read_source_table(source_path, fits_read_policy=policy)
    table, sid_mode = ensure_catalog_source_ids(
        table, link_id_col, allow_incomplete_link_id=allow_incomplete_link_id,
    )
    table = _add_healpix_columns(table, ra_col, dec_col, norder)
    table = _filter_table_columns(
        table,
        columns,
        ra_col=ra_col,
        dec_col=dec_col,
        link_id_col=link_id_col,
        norder=norder,
    )
    table = normalize_catalog_table_types(table)
    batches = catalog_table_to_tile_batches(table, norder)
    return batches, sid_mode, int(table.num_rows)


def _ingest_catalog_streaming(
    source_path: Path,
    catalog_root: Path,
    survey_name: str,
    ra_col: str,
    dec_col: str,
    norder: int,
    link_id_col: str | None,
    tile_mode: TileMode,
    on_duplicate_id: DuplicateIdMode = "skip",
    columns: Sequence[str] | None = None,
    parquet_options: CatalogParquetOptions | None = None,
    allow_incomplete_link_id: bool = False,
    fits_read_policy=None,
    row_start: int = 0,
    row_end: int | None = None,
    catalog_root_override: Path | None = None,
    skip_finalize: bool = False,
) -> None:
    """Stream-write per-tile Parquet from a FITS BINTABLE without materialising
    the full catalog as a PyArrow Table in RAM.

    Pipeline (FITS only — other input formats fall back to the in-memory path):

    1. ``fits.open(memmap=True)`` exposes the BINTABLE as a memory-mapped
       numpy recarray; no decompression / copy yet.
    2. Read only ``ra_col``, ``dec_col``, and (optionally) ``link_id_col``
       into RAM.  Compute HEALPix and an argsort permutation by tile.
    3. For each HEALPix tile (contiguous slice of the sorted order):
       a. Use numpy fancy indexing into the memmap to materialise only
          the rows for this tile (triggers paged disk reads — random I/O
          on spinning rust, fine on SSD).
       b. Wrap in an astropy Table and route through
          :func:`_astropy_table_to_arrow` so multidim columns and the
          large_string fix carry over.
       c. Append source_id (if not in the FITS), ``_healpix_norder<N>``,
          ``_cutout_index = -1``, ``_spectrum_index = -1``.
       d. Write one Parquet file for the tile; drop the slice from RAM.
    4. Aggregate ``_metadata`` + ``catalog_info.json`` from the per-tile
       FileMetaData objects (same as the in-memory path).

    Memory peak: bounded by the largest tile's row count × column widths.
    At Norder=5 over the DESI footprint this is typically a few thousand
    rows → tens of MB, vs ~tens of GB for the full in-memory table.
    """
    from astropy.table import Table

    from data_lake.io.fits_read import (
        FitsReadPolicy,
        materialize_fits_columns,
        materialize_fits_rows,
        open_fits,
    )

    if source_path.suffix.lower() not in {".fit", ".fits", ".fz"} \
            and not source_path.name.lower().endswith(".fits.gz"):
        raise ValueError(
            f"streaming=True is supported only for FITS inputs; "
            f"got {source_path.name!r}.  Use the default in-memory path for "
            "Parquet / VOTable / CSV / ECSV inputs."
        )

    pq_opts = parquet_options or CatalogParquetOptions()

    policy = fits_read_policy or FitsReadPolicy.from_env()
    log.info("Reading %s as fits (streaming, memmapped) …", source_path.name)

    with open_fits(str(source_path), policy) as hdul:
        from astropy.io import fits

        bintable_hdu = next(
            (hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)), None
        )
        if bintable_hdu is None:
            raise ValueError(f"No BINTABLE HDU found in {source_path.name}")

        data = bintable_hdu.data
        n_rows = len(data)
        row_end_eff = n_rows if row_end is None else min(int(row_end), n_rows)
        row_start_eff = max(0, int(row_start))
        if row_start_eff >= row_end_eff:
            log.info("Streaming row range [%d, %d) is empty; nothing to do.", row_start_eff, row_end_eff)
            return
        if row_start_eff > 0 or row_end_eff < n_rows:
            data = data[row_start_eff:row_end_eff]
            n_rows = len(data)
            log.info(
                "Streaming row shard [%d, %d): %d rows × %d columns",
                row_start_eff, row_end_eff, n_rows, len(data.dtype.names),
            )
        col_names = list(data.dtype.names)
        log.info(
            "FITS BINTABLE: %d rows × %d columns (memmapped)",
            n_rows, len(col_names),
        )

        for required in (ra_col, dec_col):
            if required not in col_names:
                raise KeyError(
                    f"Required column {required!r} not in FITS BINTABLE.  "
                    f"Available columns: {col_names[:20]}"
                    f"{'…' if len(col_names) > 20 else ''}"
                )

        ra = np.ascontiguousarray(np.asarray(data[ra_col], dtype=np.float64))
        dec = np.ascontiguousarray(np.asarray(data[dec_col], dtype=np.float64))

        if not link_id_col:
            raise ValueError(
                "Catalog ingest requires --link-id-col (e.g. TARGETID, SOURCE_ID). "
                "Pass the survey's native object ID column."
            )
        if not _link_id_columns_present(link_id_col, col_names):
            raise KeyError(
                f"Link-ID column {link_id_col!r} not found in FITS BINTABLE. "
                f"Available columns: {col_names[:30]}"
                f"{'…' if len(col_names) > 30 else ''}"
            )

        valid = valid_sky_position_mask(ra, dec)
        if not valid.all():
            bad_idx = np.flatnonzero(~valid)[:5]
            n_bad = int((~valid).sum())
            examples = "; ".join(
                f"row {int(i)}: {ra_col}={ra[i]!r}, {dec_col}={dec[i]!r}"
                for i in bad_idx
            )
            raise ValueError(
                f"{n_bad} catalog row(s) have invalid sky coordinates "
                f"({ra_col!r}/{dec_col!r}; checked with is_valid_sky_position). "
                f"Examples: {examples}"
            )

        npix_arr = assign_healpix(ra, dec, norder)
        sort_order = np.argsort(npix_arr, kind="stable")
        npix_sorted = npix_arr[sort_order]
        unique_pixels, group_starts = np.unique(npix_sorted, return_index=True)
        group_starts = np.append(group_starts, n_rows)
        log.info(
            "Writing %d HEALPix tiles at Norder=%d (streaming) …",
            len(unique_pixels), norder,
        )

        catalog_root.mkdir(parents=True, exist_ok=True)
        write_root = catalog_root_override or catalog_root
        hp_col = f"_healpix_norder{norder}"

        fits_read_cols = _streaming_fits_read_columns(
            columns,
            col_names=col_names,
            ra_col=ra_col,
            dec_col=dec_col,
            link_id_col=link_id_col,
        )

        tile_schema: pa.Schema | None = None
        sid_mode: str | None = None
        tiles_written = 0
        t0 = time.perf_counter()

        for g, npix in enumerate(unique_pixels.tolist()):
            s, e = int(group_starts[g]), int(group_starts[g + 1])
            n_tile = e - s
            row_idx = sort_order[s:e]

            # Materialise only this tile's rows from the memmap using sequential
            # file-order reads when row indices are not contiguous.
            if fits_read_cols is not None:
                chunk = materialize_fits_columns(data, row_idx, fits_read_cols)
            else:
                chunk = materialize_fits_rows(data, row_idx)
            astropy_chunk = Table(chunk, copy=False)
            tile_table = _astropy_table_to_arrow(astropy_chunk)

            tile_table, tile_sid_mode = ensure_catalog_source_ids(
                tile_table,
                link_id_col,
                allow_incomplete_link_id=allow_incomplete_link_id,
            )
            if sid_mode is None:
                sid_mode = tile_sid_mode

            tile_table = tile_table.append_column(
                hp_col,
                pa.array(np.full(n_tile, npix, dtype=np.int64), type=pa.int64()),
            )
            for _col in (
                "_cutout_index",
                "_cutout_npix",
                "_spectrum_index",
                "_spectrum_npix",
            ):
                tile_table = tile_table.append_column(
                    _col,
                    pa.array(np.full(n_tile, -1, dtype=np.int64), type=pa.int64()),
                )

            tile_table = _filter_table_columns(
                tile_table, columns, ra_col=ra_col, dec_col=dec_col,
                link_id_col=link_id_col, norder=norder,
            )

            if tile_schema is None:
                tile_schema = tile_table.schema

            out_dir = write_root / healpix_dir(norder, int(npix))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"Npix={int(npix)}.parquet"

            if _write_tile_for_mode(
                out_file,
                tile_table,
                tile_mode=tile_mode,
                on_duplicate_id=on_duplicate_id,
                link_id_col=link_id_col,
                parquet_options=pq_opts,
            ) is not None:
                tiles_written += 1

        elapsed = time.perf_counter() - t0
        log.info(
            "Processed %d tiles (%d written/updated) in %.1f s "
            "(streaming, peak ≈ tile-sized)",
            len(unique_pixels), tiles_written, elapsed,
        )

        if tile_schema is not None and not skip_finalize:
            if sid_mode is None:
                sid_mode = "sequential"
            _finalize_catalog_writes(
                catalog_root,
                survey_name,
                norder,
                ra_col=ra_col,
                dec_col=dec_col,
                link_id_mode=sid_mode,
                streaming=True,
                fallback_n_cols=len(tile_schema),
                allow_incomplete_link_id=allow_incomplete_link_id,
            )
        if not skip_finalize:
            log.info("Catalog written to %s", catalog_root)


def _merge_spooled_catalog_tiles(
    spool_root: Path,
    catalog_root: Path,
    *,
    tile_mode: TileMode,
    on_duplicate_id: DuplicateIdMode,
    link_id_col: str | None,
    parquet_options: CatalogParquetOptions,
) -> None:
    """Merge worker-local streaming shards into the survey catalog root."""
    import pyarrow.parquet as pq

    for spool_file in sorted(spool_root.rglob("Npix=*.parquet")):
        rel = spool_file.relative_to(spool_root)
        dest = catalog_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(spool_file)
        _write_tile_for_mode(
            dest,
            table,
            tile_mode=tile_mode,
            on_duplicate_id=on_duplicate_id,
            link_id_col=link_id_col,
            parquet_options=parquet_options,
        )


def _ingest_catalog_streaming_parallel(
    source_path: Path,
    catalog_root: Path,
    survey_name: str,
    *,
    streaming_parallel: int,
    ra_col: str,
    dec_col: str,
    norder: int,
    link_id_col: str | None,
    tile_mode: TileMode,
    on_duplicate_id: DuplicateIdMode,
    columns: Sequence[str] | None,
    parquet_options: CatalogParquetOptions,
    allow_incomplete_link_id: bool,
    fits_read_policy,
) -> None:
    """Parallel row-range shards of streaming FITS catalog ingest."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from data_lake.cli_utils import init_parallel_ingest_subprocess
    from data_lake.io.fits_read import open_fits

    if streaming_parallel < 2:
        raise ValueError("streaming_parallel must be >= 2")
    if tile_mode != "append":
        raise ValueError("streaming parallel ingest requires tile_mode='append'")

    with open_fits(str(source_path), fits_read_policy) as hdul:
        from astropy.io import fits

        bintable_hdu = next(
            (hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)), None
        )
        if bintable_hdu is None:
            raise ValueError(f"No BINTABLE HDU found in {source_path.name}")
        n_rows = len(bintable_hdu.data)

    chunk = (n_rows + streaming_parallel - 1) // streaming_parallel
    ranges = [
        (i * chunk, min(n_rows, (i + 1) * chunk))
        for i in range(streaming_parallel)
        if i * chunk < n_rows
    ]

    import shutil
    import tempfile

    spool_dirs: list[Path] = []

    def _shard_worker(row_start: int, row_end: int) -> str:
        from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

        apply_parallel_worker_logging_after_heavy_imports()
        spool = Path(tempfile.mkdtemp(prefix="dl_stream_spool_"))
        _ingest_catalog_streaming(
            source_path,
            catalog_root,
            survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=norder,
            link_id_col=link_id_col,
            tile_mode="append",
            on_duplicate_id=on_duplicate_id,
            columns=columns,
            parquet_options=parquet_options,
            allow_incomplete_link_id=allow_incomplete_link_id,
            fits_read_policy=fits_read_policy,
            row_start=row_start,
            row_end=row_end,
            catalog_root_override=spool,
            skip_finalize=True,
        )
        return str(spool)

    with ProcessPoolExecutor(
        max_workers=len(ranges),
        initializer=init_parallel_ingest_subprocess,
    ) as pool:
        futures = {
            pool.submit(_shard_worker, rs, re): (rs, re)
            for rs, re in ranges
        }
        for fut in as_completed(futures):
            spool_dirs.append(Path(fut.result()))

    try:
        for spool in spool_dirs:
            _merge_spooled_catalog_tiles(
                spool,
                catalog_root,
                tile_mode=tile_mode,
                on_duplicate_id=on_duplicate_id,
                link_id_col=link_id_col,
                parquet_options=parquet_options,
            )
    finally:
        for spool in spool_dirs:
            shutil.rmtree(spool, ignore_errors=True)

    _finalize_catalog_writes(
        catalog_root,
        survey_name,
        norder,
        ra_col=ra_col,
        dec_col=dec_col,
        link_id_mode=None,
        streaming=True,
        fallback_n_cols=0,
        allow_incomplete_link_id=allow_incomplete_link_id,
    )
    log.info("Parallel streaming catalog written to %s", catalog_root)


def _write_aggregate_metadata(
    catalog_root: Path,
    file_metadata: list[pq.FileMetaData],
    schema: pa.Schema,
) -> None:
    if not file_metadata:
        return
    combined = file_metadata[0]
    for m in file_metadata[1:]:
        combined.append_row_groups(m)
    combined.write_metadata_file(str(catalog_root / "_metadata"))


def _write_catalog_info(
    catalog_root: Path,
    survey_name: str,
    norder: int,
    n_rows: int,
    n_cols: int,
    *,
    ra_column: str,
    dec_column: str,
    link_id_mode: str,
    link_id_column: str | None = None,
    native_id_column: str | None = None,
    streaming: bool,
    allow_incomplete_link_id: bool = False,
) -> None:
    join_col = link_id_column or LAKE_JOIN_ID_COLUMN
    info = {
        "catalog_name": survey_name,
        "catalog_type": "object",
        "hats_order": norder,
        "total_rows": n_rows,
        "total_columns": n_cols,
        "schema_version": "1",
        "epoch": "J2000",
        "ra_column": ra_column,
        "dec_column": dec_column,
        "link_id_mode": link_id_mode,
        "link_id_column": join_col,
        "ingest_streaming": streaming,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if native_id_column:
        info["native_id_column"] = native_id_column
    if allow_incomplete_link_id:
        info["allow_incomplete_link_id"] = True
    with open(catalog_root / "catalog_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


# ---------------------------------------------------------------------------
# Batch ingest: multiple files → single survey
# ---------------------------------------------------------------------------


def ingest_catalog_batch(
    source_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "ra",
    dec_col: str = "dec",
    norder: int = 5,
    link_id_col: str | None = None,
    tile_mode: TileMode | None = None,
    on_duplicate_id: DuplicateIdMode = "skip",
    allow_incomplete_link_id: bool = False,
) -> None:
    """Ingest multiple source files into the same survey catalog.

    Each file is passed to :func:`ingest_catalog` with the same tile policy.
    For overlapping sky, use ``tile_mode="append"`` (and ``--on-duplicate-id``
    as needed).
    """
    for path in source_paths:
        ingest_catalog(
            source_path=path,
            output_root=output_root,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=norder,
            link_id_col=link_id_col,
            tile_mode=tile_mode,
            on_duplicate_id=on_duplicate_id,
            allow_incomplete_link_id=allow_incomplete_link_id,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        ingest_token_option,
        load_optional_config,
        logging_options,
        pick,
        require_ingest_permission,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
        fits_memmap_option,
    )

    @click.command("dl-ingest-catalog")
    @click.argument("source_path", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--norder", default=None, type=int,
                  help="HEALPix order (overrides config; default 5).")
    @click.option(
        "--link-id-col",
        required=True,
        help="Survey object ID column (e.g. TARGETID, SOURCE_ID). Required.",
    )
    @click.option(
        "--allow-incomplete-link-id",
        is_flag=True,
        help="For composite or string --link-id-col: leave _source_id null when "
        "any link part is missing (row stays in catalog, unlinked to spectra).",
    )
    @click.option(
        "--tile-mode",
        type=click.Choice(["skip", "overwrite", "append"], case_sensitive=False),
        default=None,
        help="Existing Npix tile: skip (default), replace, or read-concat-write.",
    )
    @click.option(
        "--on-duplicate-id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
        help="When --tile-mode=append and an ID column exists.",
    )
    @click.option(
        "--streaming/--no-streaming", default=False, show_default=True,
        help="FITS-only: memmap the input and write one tile at a time. "
             "Bounds memory peak to ~one tile's worth of rows (tens of MB) "
             "instead of holding the full table + sorted copy in RAM. "
             "Recommended for catalogs >~ 50 M rows.",
    )
    @click.option(
        "--streaming-parallel",
        default=0,
        show_default=True,
        type=int,
        help="FITS streaming only: shard row ranges across N worker processes "
             "(requires --streaming and --tile-mode append).",
    )
    @fits_memmap_option
    @click.option(
        "--columns",
        default=None,
        help="Comma-separated FITS columns to keep (sky, ID, HEALPix, and index "
             "columns are always added). Omit rarely used columns to cut disk use.",
    )
    @click.option(
        "--compact",
        is_flag=True,
        help="Smaller Parquet tiles: ZSTD-9, no per-column statistics, no "
             "dictionary encoding, narrow string type per tile.",
    )
    @click.option(
        "--compression-level",
        default=None,
        type=int,
        help="ZSTD level for catalog tiles (default 3; ignored if --compact).",
    )
    @click.option(
        "--log-file",
        "log_file",
        type=click.Path(path_type=Path),
        default=None,
        help="File for INFO+ logs (useful with --quiet for an audit trail).",
    )
    @logging_options
    def cli(
        source_path: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int | None,
        link_id_col: str | None,
        allow_incomplete_link_id: bool,
        tile_mode: str | None,
        on_duplicate_id: str,
        streaming: bool,
        streaming_parallel: int,
        fits_memmap: str,
        columns: str | None,
        compact: bool,
        compression_level: int | None,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Ingest FITS/VOTable SOURCE_PATH into HATS-partitioned Parquet.

        OUTPUT_ROOT is optional when a lake config is available
        (via --config or $DATA_LAKE_CONFIG); in that case it defaults to
        ``<lake.root>/<paths.catalogs>``.
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            log_file=log_file,
            quiet=quiet,
        )
        require_ingest_permission(cfg, ingest_token)
        resolved_output = require_output_root(output_root, cfg, kind="catalogs")

        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        if streaming_parallel > 0 and not streaming:
            raise click.UsageError("--streaming-parallel requires --streaming.")
        if streaming_parallel > 1 and (tile_mode or "skip").lower() != "append":
            raise click.UsageError(
                "--streaming-parallel requires --tile-mode append."
            )
        pq_opts = None
        if compression_level is not None and not compact:
            pq_opts = CatalogParquetOptions(compression_level=compression_level)

        ingest_catalog(
            source_path=source_path,
            output_root=resolved_output,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=pick(norder,
                        cfg.partitioning.hats_order if cfg else None, 5),
            link_id_col=link_id_col,
            allow_incomplete_link_id=allow_incomplete_link_id,
            tile_mode=tile_mode.lower() if tile_mode else None,  # type: ignore[arg-type]
            on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
            streaming=streaming,
            streaming_parallel=streaming_parallel,
            columns=col_list,
            parquet_options=pq_opts,
            compact=compact,
            fits_memmap=fits_memmap.lower(),
        )

    @click.command("dl-finalize-catalog")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--norder", default=None, type=int)
    @logging_options
    def cli_finalize(
        output_root: Path | None,
        config_path: Path | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Rebuild ``catalog_info.json``, ``_metadata``, and ``schema_manifest.json`` from tiles."""
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg, kind="catalogs")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        catalog_root = lake / "catalogs" / survey_name
        if not finalize_catalog_survey(
            catalog_root,
            survey_name,
            n,
            ra_col=ra_col,
            dec_col=dec_col,
        ):
            raise click.ClickException(f"No Parquet tiles under {catalog_root}")
        click.echo(f"Finalized {catalog_root}")

except ImportError:
    cli = None  # type: ignore[assignment]
    cli_finalize = None  # type: ignore[assignment]
