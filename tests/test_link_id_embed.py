"""Unit tests for embedding catalog link-id columns into subset extracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_values_to_numpy_int_and_str() -> None:
    from data_lake.export.link_id_embed import values_to_numpy

    assert values_to_numpy([1, 2, None]).dtype == np.int64
    assert values_to_numpy([1, 2, None])[2] == -1
    s = values_to_numpy(["main", "sv3", None])
    assert s.dtype.kind == "S"
    assert s[0].decode("utf-8") == "main"
    assert s[2].decode("utf-8") == ""


def test_write_link_id_arrays_zarr_roundtrip(tmp_path: Path) -> None:
    import zarr

    from data_lake.export.link_id_embed import (
        ATTR_LINK_ID_COLUMNS,
        write_link_id_arrays_zarr,
    )

    out = tmp_path / "t.zarr"
    root = zarr.open_group(
        store=zarr.storage.LocalStore(str(out)), mode="w", zarr_format=3,
    )
    sids = np.array([101, 202], dtype=np.int64)
    root.create_array("_source_id", data=sids)
    write_link_id_arrays_zarr(
        root,
        sids,
        ["TARGETID", "SURVEY"],
        {
            "TARGETID": {101: 11, 202: 22},
            "SURVEY": {101: "main", 202: "sv3"},
        },
        link_id_mode="composite:TARGETID,SURVEY",
    )
    r = zarr.open_group(
        store=zarr.storage.LocalStore(str(out)), mode="r", zarr_format=3,
    )
    assert list(r.attrs[ATTR_LINK_ID_COLUMNS]) == ["TARGETID", "SURVEY"]
    assert r.attrs["link_id_mode"] == "composite:TARGETID,SURVEY"
    np.testing.assert_array_equal(np.asarray(r["TARGETID"][:]), [11, 22])
    assert np.asarray(r["SURVEY"][:])[0].decode("utf-8") == "main"


def test_reader_prefers_embedded_over_lake(tmp_path: Path, monkeypatch) -> None:
    """resolve_id_columns must use subset arrays and not touch the lake."""
    import zarr

    from data_lake.export.link_id_embed import write_link_id_arrays_zarr

    out = tmp_path / "sub.zarr"
    root = zarr.open_group(
        store=zarr.storage.LocalStore(str(out)), mode="w", zarr_format=3,
    )
    sids = np.array([1], dtype=np.int64)
    root.create_array("_source_id", data=sids)
    # minimal spectra-like arrays so modality detection works if needed
    root.create_array("flux", data=np.zeros((1, 4), dtype=np.float32))
    root.create_array("ivar", data=np.ones((1, 4), dtype=np.float32))
    root.create_array("mask", data=np.zeros((1, 4), dtype=np.uint8))
    root.create_array("wavelength", data=np.linspace(1, 4, 4))
    root.create_array("redshift", data=np.array([0.1], dtype=np.float32))
    root.attrs["source_survey"] = "X"
    root.attrs["source_lake_root"] = str(tmp_path / "missing_lake")
    write_link_id_arrays_zarr(
        root, sids, ["TARGETID"], {"TARGETID": {1: 99}}, link_id_mode="column:TARGETID",
    )

    # Load the standalone script as a module (scripts/ is not a package)
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "read_zarr_subset.py"
    spec = importlib.util.spec_from_file_location("read_zarr_subset", script)
    assert spec is not None and spec.loader is not None
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def _fail(*_a, **_k):
        raise AssertionError("lake fallback should not be used when embedded")

    monkeypatch.setattr(reader, "resolve_id_columns_from_lake", _fail)
    r = zarr.open_group(
        store=zarr.storage.LocalStore(str(out)), mode="r", zarr_format=3,
    )
    cols, arrays = reader.resolve_id_columns(r, sids)
    assert cols == ["TARGETID"]
    assert int(arrays["TARGETID"][0]) == 99
