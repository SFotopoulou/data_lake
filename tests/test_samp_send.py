"""Tests for dl-samp-send-catalog and SAMP helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.areas import make_area, save_area
from data_lake.discovery.region import Region
from data_lake.export.samp_send import SampSendError, find_topcat_client_id, send_votable_to_topcat
from data_lake.export.samp_send_cli import cli as samp_cli
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


class TestFindTopcat:
    def test_matches_samp_name(self) -> None:
        client = MagicMock()
        client.get_registered_clients.return_value = ["hub", "c1", "c2"]
        client.get_metadata.side_effect = lambda cid: {
            "hub": {"samp.name": "Hub"},
            "c1": {"samp.name": "topcat"},
            "c2": {"samp.name": "ds9"},
        }[cid]
        assert find_topcat_client_id(client) == "c1"


class TestSendVotable:
    def test_no_hub(self, tmp_path: Path) -> None:
        vot = tmp_path / "t.vot"
        vot.write_text("<VOTABLE/>")
        with patch("astropy.samp.SAMPIntegratedClient") as cls:
            inst = cls.return_value
            inst.connect.side_effect = RuntimeError("no hub")
            with pytest.raises(SampSendError, match="No SAMP hub"):
                send_votable_to_topcat(vot)

    def test_no_topcat(self, tmp_path: Path) -> None:
        vot = tmp_path / "t.vot"
        vot.write_text("<VOTABLE/>")
        with patch("astropy.samp.SAMPIntegratedClient") as cls:
            inst = cls.return_value
            inst.get_registered_clients.return_value = ["hub"]
            inst.get_metadata.return_value = {"samp.name": "Hub"}
            with pytest.raises(SampSendError, match="TOPCAT is not connected"):
                send_votable_to_topcat(vot)

    def test_call_and_wait(self, tmp_path: Path) -> None:
        vot = tmp_path / "t.vot"
        vot.write_text("<VOTABLE/>")
        with patch("astropy.samp.SAMPIntegratedClient") as cls:
            inst = cls.return_value
            inst.get_registered_clients.return_value = ["c1"]
            inst.get_metadata.return_value = {"samp.name": "topcat"}
            cid = send_votable_to_topcat(vot, name="demo", timeout_sec=5)
            assert cid == "c1"
            inst.call_and_wait.assert_called_once()
            args = inst.call_and_wait.call_args[0]
            assert args[0] == "c1"
            assert args[1]["samp.mtype"] == "table.load.votable"
            assert args[1]["samp.params"]["name"] == "demo"
            assert args[2] == "5"
            inst.disconnect.assert_called_once()


class TestCli:
    def test_cli_input_mocked(self, tmp_path: Path) -> None:
        vot = tmp_path / "in.vot"
        # Minimal real VOTable via write_catalog_extract
        from data_lake.export.catalog_extract import write_catalog_extract

        write_catalog_extract(
            pa.table({"ra": [1.0], "dec": [2.0]}),
            vot,
            output_format="votable",
        )
        runner = CliRunner()
        with patch(
            "data_lake.export.samp_send_cli.send_votable_to_topcat",
            return_value="c1",
        ) as send:
            r = runner.invoke(
                samp_cli,
                ["--input", str(vot), "--name", "demo"],
            )
        assert r.exit_code == 0, r.output
        send.assert_called_once()
        assert "demo" in r.output

    def test_cli_survey_from_area(self, lake: Path) -> None:
        runner = CliRunner()
        with patch(
            "data_lake.export.samp_send_cli.send_votable_to_topcat",
            return_value="c1",
        ) as send:
            r = runner.invoke(
                samp_cli,
                [
                    str(lake), "--survey", "ALLWISE",
                    "-c", "ra,dec,mag",
                    "--from-area", "MyCone",
                    "--where", "mag < 13",
                    "--name", "wise_bright",
                ],
            )
        assert r.exit_code == 0, r.output
        send.assert_called_once()
        assert send.call_args.kwargs.get("name") == "wise_bright" or (
            send.call_args[1].get("name") == "wise_bright"
        )

    def test_cli_rejects_input_plus_filters(self, tmp_path: Path) -> None:
        vot = tmp_path / "in.vot"
        vot.write_text("<VOTABLE/>")
        runner = CliRunner()
        r = runner.invoke(
            samp_cli,
            ["--input", str(vot), "--where", "mag > 0"],
        )
        assert r.exit_code != 0
        assert "--input" in r.output
