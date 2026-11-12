"""Region clips (--from-area / --cone / …) on dl-extract-catalog."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.areas import make_area, save_area
from data_lake.discovery.region import Region
from data_lake.export.catalog_extract import ExtractResult, cli, extract_catalog
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir


def _write_catalog_tile(lake, survey, *, norder, npix, source_ids, ra, dec, extra=None):
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    cols = {
        LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        hp_col: pa.array([npix] * len(source_ids), type=pa.int64()),
    }
    if extra:
        for k, v in extra.items():
            cols[k] = pa.array(v)
    pq.write_table(pa.table(cols), tile_dir / f"Npix={npix}.parquet")
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_mode": "sequential",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": len(source_ids),
        })
    )


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    norder = 5
    ra, dec = 120.0, 45.0
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    _write_catalog_tile(
        lake, "ALLWISE",
        norder=norder, npix=npix,
        source_ids=[1, 2, 3],
        ra=[ra, ra + 0.01, ra + 0.02],
        dec=[dec, dec + 0.01, dec + 0.02],
        extra={"mag": [10.0, 12.0, 14.0]},
    )
    ti.write_tile_index(lake, "ALLWISE", "catalog")
    area = make_area("MyCone", Region.cone(ra, dec, 3600.0))
    save_area(lake, area)
    return lake


class TestExtractRegion:
    def test_cone_keeps_overlapping_tile(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "in.parquet"
        result = extract_catalog(
            output=out,
            specs=["ra", "dec", "mag"],
            lake_root=lake,
            survey="ALLWISE",
            engine="tiles",
            region=Region.cone(120.0, 45.0, 3600.0),
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 3
        assert result.n_tiles == 1

    def test_empty_region_writes_zero_rows(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "empty.parquet"
        result = extract_catalog(
            output=out,
            specs=["ra", "dec"],
            lake_root=lake,
            survey="ALLWISE",
            engine="tiles",
            region=Region.cone(0.0, 0.0, 1.0),
        )
        assert result.n_rows == 0
        assert out.is_file()
        assert pq.read_table(out).num_rows == 0

    def test_from_area_with_where(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "bright.parquet"
        result = extract_catalog(
            output=out,
            specs=["mag"],
            lake_root=lake,
            survey="ALLWISE",
            engine="tiles",
            region=Region.cone(120.0, 45.0, 3600.0),
            where_sql="mag < 13",
        )
        assert result.n_rows == 2

    def test_cli_from_area(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "cli.parquet"
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "--lake-root", str(lake), "--survey", "ALLWISE",
                "--from-area", "MyCone",
                "-c", "ra,dec,mag",
                "-o", str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        assert pq.read_table(out).num_rows == 3

    def test_cli_rejects_region_on_raw_file(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw.parquet"
        pq.write_table(pa.table({"ra": [1.0], "dec": [2.0]}), raw)
        out = tmp_path / "out.parquet"
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                str(raw), "-c", "ra,dec", "-o", str(out),
                "--cone", "120", "45", "--radius-arcsec", "60",
            ],
        )
        assert r.exit_code != 0
        assert "Region clip" in r.output or "requires --survey" in r.output

    def test_cli_rejects_two_selectors(self, lake: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "--lake-root", str(lake), "--survey", "ALLWISE",
                "--from-area", "MyCone",
                "--cone", "120", "45", "--radius-arcsec", "60",
                "-c", "ra", "-o", str(tmp_path / "x.parquet"),
            ],
        )
        assert r.exit_code != 0
        assert "at most one" in r.output.lower()
