"""Unit tests for target-list identity bridge helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def test_build_bridge_without_via() -> None:
    from data_lake.export.target_id_bridge import build_target_id_bridge

    identity = pa.table({
        "TARGETID": pa.array([11, 13], type=pa.int64()),
        "SURVEY": pa.array(["main", "sv3"], type=pa.string()),
    })
    bridge = build_target_id_bridge(
        np.array([11, 13], dtype=np.int64),
        identity,
    )
    assert bridge.column_names == ["_source_id", "TARGETID", "SURVEY"]
    assert bridge.column("_source_id").to_pylist() == [11, 13]
    assert bridge.column("TARGETID").to_pylist() == [11, 13]


def test_build_bridge_via_pairs_expands() -> None:
    from data_lake.export.target_id_bridge import build_target_id_bridge

    identity = pa.table({"TARGETID": pa.array([11, 12], type=pa.int64())})
    pairs = pa.table({
        "partner_id": pa.array([11, 12], type=pa.int64()),
        "extract_id": pa.array([101, 102], type=pa.int64()),
    })
    bridge = build_target_id_bridge(
        np.array([11, 12], dtype=np.int64),
        identity,
        via_pairs=pairs,
    )
    assert bridge.column("_source_id").to_pylist() == [101, 102]
    assert bridge.column("TARGETID").to_pylist() == [11, 12]


def test_attach_bridge_to_table_expands_rows() -> None:
    from data_lake.export.target_id_bridge import (
        attach_bridge_to_table,
        build_target_id_bridge,
    )

    extract = pa.table({
        "_source_id": pa.array([101], type=pa.int64()),
        "flux_g": pa.array([1.5], type=pa.float64()),
    })
    # Two partner targets map to the same extract source.
    identity = pa.table({"TARGETID": pa.array([11, 99], type=pa.int64())})
    pairs = pa.table({
        "partner_id": pa.array([11, 99], type=pa.int64()),
        "extract_id": pa.array([101, 101], type=pa.int64()),
    })
    bridge = build_target_id_bridge(
        np.array([11, 99], dtype=np.int64),
        identity,
        via_pairs=pairs,
    )
    merged = attach_bridge_to_table(extract, bridge, via_crossmatch=True)
    assert merged.num_rows == 2
    assert sorted(merged.column("TARGETID").to_pylist()) == [11, 99]
    assert merged.column("flux_g").to_pylist() == [1.5, 1.5]


def test_write_target_id_sidecar(tmp_path: Path) -> None:
    from data_lake.export.target_id_bridge import (
        build_target_id_bridge,
        target_id_sidecar_path,
        write_target_id_sidecar,
    )

    out = tmp_path / "subset.zarr"
    out.mkdir()
    bridge = build_target_id_bridge(
        np.array([1], dtype=np.int64),
        pa.table({"TARGETID": pa.array([1], type=pa.int64())}),
    )
    side = write_target_id_sidecar(out, bridge)
    assert side == target_id_sidecar_path(out)
    assert pq.read_table(side).column("TARGETID").to_pylist() == [1]
