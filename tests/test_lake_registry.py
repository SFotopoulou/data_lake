"""Tests for lake registry and master association metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.table import Table

from concurrent.futures import ThreadPoolExecutor

from data_lake.ingest.desi_parallel_ingest import TileBatch, WorkerResult, ingest_spectra_parallel
from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE, _meta_to_bytes
from data_lake.lake_registry import (
    REGISTRY_FILENAME,
    build_lake_registry_table,
    filter_lake_registry_table,
    format_lake_registry_table,
    format_registry_pair_footer,
    guess_master_meta,
    load_lake_registry,
    load_master_meta,
    master_meta_path,
    refresh_lake_registry,
    registry_path,
    summarize_registry_row_counts,
    write_master_meta,
)
from data_lake.schema_registry import MODALITY_CATALOG, MODALITY_CROSSMATCH, MODALITY_SPECTRA


def _ingest_mini_catalog(
    tmp_path: Path,
    survey: str,
    *,
    ra: float = 10.0,
    dec: float = 20.0,
    id_offset: int = 0,
) -> None:
    n = 6
    tbl = Table({
        "TARGETID": np.arange(id_offset, id_offset + n, dtype=np.int64),
        "TARGET_RA": np.full(n, ra),
        "TARGET_DEC": np.full(n, dec),
        "Z": np.linspace(0.1, 0.3, n),
        "MAG_G": np.linspace(18.0, 19.0, n),
    })
    fits = tmp_path / f"{survey}.fits"
    tbl.write(fits, overwrite=True)
    lake = tmp_path / "lake"
    ingest_catalog(
        source_path=fits,
        output_root=lake,
        survey_name=survey,
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        link_id_col="TARGETID",
        norder=5,
        tile_mode="overwrite",
    )


def test_refresh_lake_registry(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_A", id_offset=0)
    _ingest_mini_catalog(tmp_path, "SURV_B", id_offset=100, ra=50.0, dec=60.0)
    lake = tmp_path / "lake"

    out = refresh_lake_registry(lake)
    assert out == registry_path(lake)
    table = load_lake_registry(lake)
    names = set(table.column("survey").to_pylist())
    assert "SURV_A" in names
    assert "SURV_B" in names


def test_registry_catalog_extended_fields(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_META")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)
    row = next(
        r for r in load_lake_registry(lake).to_pylist()
        if r["survey"] == "SURV_META" and r["modality"] == MODALITY_CATALOG
    )
    assert row["link_id_mode"] == "column:TARGETID"
    assert row["native_id_column"] == "TARGETID"
    assert row["ra_column"] == "TARGET_RA"
    assert row["dec_column"] == "TARGET_DEC"
    assert row["n_tiles"] is not None and row["n_tiles"] >= 1
    assert row["has_aggregate_metadata"] is True
    assert row["registry_generated_utc"]
    assert row["hats_order_match"] is None


def test_master_meta_guess_and_write(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "DESI_DR1", id_offset=0)
    _ingest_mini_catalog(tmp_path, "EUCLID_DR1", id_offset=1000, ra=50.0)
    lake = tmp_path / "lake"

    master = pa.table({
        "desi_targetid": pa.array([1, 2], type=pa.int64()),
        "euclid_source_id": pa.array([1000, 1001], type=pa.int64()),
        "sep_arcsec": pa.array([0.1, 0.2], type=pa.float32()),
    })
    master_path = lake / "associations" / "master.parquet"
    master_path.parent.mkdir(parents=True)
    pq.write_table(master, str(master_path))

    meta = guess_master_meta(master_path, lake)
    assert len(meta["partners"]) >= 1
    surveys = {p["survey"] for p in meta["partners"]}
    assert "DESI_DR1" in surveys or "EUCLID_DR1" in surveys

    write_master_meta(master_path, meta)
    assert master_meta_path(master_path).is_file()
    loaded = load_master_meta(master_path, lake, allow_guess=False)
    assert loaded["partners"][0]["master_column"]


def test_build_registry_table_empty_lake(tmp_path: Path) -> None:
    lake = tmp_path / "empty_lake"
    (lake / "catalogs").mkdir(parents=True)
    table = build_lake_registry_table(lake)
    assert table.num_rows == 0


def _mini_spectrum_decoder(path_str: str, norder: int) -> WorkerResult:
    label = Path(path_str).stem
    npix = 100 if label == "fileA" else 200
    sid = 101 if label == "fileA" else 202
    n_pix = 8
    flux = np.full((1, n_pix), float(sid), dtype=np.float32)
    ivar = np.ones_like(flux)
    mask = np.zeros((1, n_pix), dtype=np.uint8)
    sids = np.array([sid], dtype=np.int64)
    mb = _meta_to_bytes({
        "z": 0.1, "z_err": 0.01, "snr": 5.0,
        "exptime": 100.0, "R": 3000.0, "instr": "TEST",
    })
    wave = np.linspace(3600.0, 3700.0, n_pix, dtype=np.float64)
    wcs = {
        "ctype": "WAVE", "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]), "crpix": 1.0,
        "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": n_pix,
    }
    return WorkerResult(
        path=path_str,
        ok=True,
        batches=[TileBatch(npix=npix, flux=flux, ivar=ivar, mask=mask, source_ids=sids, meta_bytes=mb)],
        wavelength=wave,
        wcs_attrs=wcs,
        n_pix=n_pix,
        n_spectra=1,
        elapsed_s=0.0,
    )


def test_registry_spectra_total_rows(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    paths = []
    for label in ("fileA", "fileB"):
        p = tmp_path / f"{label}.fits"
        p.touch()
        paths.append(p)

    ingest_spectra_parallel(
        file_paths=paths,
        output_root=lake,
        survey_name="SPEC_SURV",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt.json",
        failures_log=tmp_path / "fail.jsonl",
        show_progress=False,
        decoder=_mini_spectrum_decoder,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )

    refresh_lake_registry(lake)
    table = load_lake_registry(lake)
    spec_rows = [
        r
        for r in table.to_pylist()
        if r["survey"] == "SPEC_SURV" and r["modality"] == "spectra"
    ]
    assert len(spec_rows) == 1
    assert spec_rows[0]["total_rows"] == 2
    assert spec_rows[0]["n_pix"] == 8
    assert spec_rows[0]["wavelength_mode"] == "shared"
    assert spec_rows[0]["n_tiles"] == 2
    assert spec_rows[0]["wcs_summary"]
    assert spec_rows[0]["meta_fields"] == list(_META_DTYPE.names)
    assert spec_rows[0]["has_spectrum_sky_meta"] is True
    assert set(spec_rows[0]["spectrum_sky_meta_fields"]) == {
        "ra_key", "dec_key", "ra", "dec", "source_file",
    }

    table_text = format_lake_registry_table(load_lake_registry(lake))
    assert "sky_meta" in table_text
    verbose_text = format_lake_registry_table(load_lake_registry(lake), verbose=True)
    assert "spectrum_sky_meta:" in verbose_text
    assert "ra_key" in verbose_text


def test_registry_hats_order_match(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "PAIR_SURV")
    lake = tmp_path / "lake"
    paths = [tmp_path / "a.fits", tmp_path / "b.fits"]
    for p in paths:
        p.touch()
    ingest_spectra_parallel(
        file_paths=paths,
        output_root=lake,
        survey_name="PAIR_SURV",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt.json",
        failures_log=tmp_path / "fail.jsonl",
        show_progress=False,
        decoder=_mini_spectrum_decoder,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    refresh_lake_registry(lake)
    table = load_lake_registry(lake)
    rows = {r["modality"]: r for r in table.to_pylist() if r["survey"] == "PAIR_SURV"}
    assert rows[MODALITY_CATALOG]["hats_order_match"] is True
    assert rows[MODALITY_SPECTRA]["hats_order_match"] is True
    footer = format_registry_pair_footer(table)
    assert "PAIR_SURV" in footer
    assert "match" in footer


def test_filter_lake_registry_by_modality(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_A")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)
    table = load_lake_registry(lake)

    cat_only = filter_lake_registry_table(table, MODALITY_CATALOG)
    assert cat_only.num_rows >= 1
    assert all(m == MODALITY_CATALOG for m in cat_only.column("modality").to_pylist())

    spec_only = filter_lake_registry_table(table, MODALITY_SPECTRA)
    assert spec_only.num_rows == 0


def test_summarize_registry_row_counts(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_A")
    _ingest_mini_catalog(tmp_path, "SURV_B", id_offset=100)
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)
    table = load_lake_registry(lake)

    summary = summarize_registry_row_counts(table)
    assert summary["grand_total"] == 12
    assert summary["surveys_listed"] == 2
    assert summary["by_modality"][MODALITY_CATALOG] == 12

    text = format_lake_registry_table(table, count_total=True)
    assert "By modality:" in text
    assert f"  {MODALITY_CATALOG}" in text
    assert "Total: 12 rows" in text



def test_registry_crossmatch_total_rows(tmp_path: Path) -> None:
    """Crossmatch registry totals come from on-disk Parquet, not the sidecar."""
    from data_lake.ingest.fits_to_parquet import healpix_dir

    _ingest_mini_catalog(tmp_path, "SURV_REG")
    lake = tmp_path / "lake"
    xm_root = lake / "crossmatch" / "A_x_B__r1"
    # Two tiles: 2 + 3 rows; sidecar lies with a smaller total_rows.
    for npix, n_rows in ((100, 2), (200, 3)):
        tile_dir = xm_root / healpix_dir(5, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                "source_id_a": pa.array(list(range(n_rows)), type=pa.int64()),
                "source_id_b": pa.array(list(range(n_rows)), type=pa.int64()),
            }),
            tile_dir / f"Npix={npix}.parquet",
        )
    (xm_root / "crossmatch_info.json").write_text(json.dumps({
        "catalog_name": "A_x_B__r1",
        "modality": "crossmatch",
        "match_mode": "sky",
        "survey_a": "A",
        "survey_b": "B",
        "match_radius_arcsec": 1.0,
        "hats_order": 5,
        "total_rows": 1,
        "n_match_rows": 1,
        "n_tiles": 1,
        "schema_version": "1",
    }))
    col_root = lake / "crossmatch" / "A_x_C__col_ID__ID"
    col_tile = col_root / healpix_dir(5, 10)
    col_tile.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "source_id_a": pa.array([1, 2, 3], type=pa.int64()),
            "source_id_b": pa.array([4, 5, 6], type=pa.int64()),
        }),
        col_tile / "Npix=10.parquet",
    )
    (col_root / "crossmatch_info.json").write_text(json.dumps({
        "catalog_name": "A_x_C__col_ID__ID",
        "modality": "crossmatch",
        "match_mode": "column",
        "survey_a": "A",
        "survey_b": "C",
        "match_col_a": "ID",
        "match_col_b": "ID",
        "hats_order": 5,
        "total_rows": 1,
        "n_match_rows": 1,
        "n_tiles": 1,
        "schema_version": "1",
    }))

    refresh_lake_registry(lake)
    table = load_lake_registry(lake)
    assert "match_mode" in table.schema.names
    assert "match_col_a" in table.schema.names
    rows = {
        r["survey"]: r
        for r in table.to_pylist()
        if r["modality"] == MODALITY_CROSSMATCH
    }
    assert rows["A_x_B__r1"]["total_rows"] == 5
    assert rows["A_x_B__r1"]["n_tiles"] == 2
    assert rows["A_x_C__col_ID__ID"]["total_rows"] == 3
    assert rows["A_x_C__col_ID__ID"]["match_mode"] == "column"
    assert rows["A_x_C__col_ID__ID"]["match_col_a"] == "ID"

    text = format_lake_registry_table(table)
    assert "5" in text
    assert "3" in text
    assert "r=1" in text
    assert "col:ID:ID" in text
    assert "A_x_C__col_ID__ID" in text


def test_describe_lake_count_total_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "SURV_SUM")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)

    assert cli_describe_lake is not None
    result = CliRunner().invoke(cli_describe_lake, [str(lake), "--count-total"])
    assert result.exit_code == 0
    assert "By modality:" in result.output
    assert "catalog" in result.output
    assert "Total: 6 rows" in result.output


def test_describe_lake_modality_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "SURV_CLI")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)

    assert cli_describe_lake is not None
    result = CliRunner().invoke(
        cli_describe_lake,
        [str(lake), "--modality", MODALITY_CATALOG],
    )
    assert result.exit_code == 0
    assert "SURV_CLI" in result.output
    assert "catalog" in result.output


def test_describe_lake_kind_filter_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "INGESTED_SURV")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)

    runner = CliRunner()
    # The mini catalog is an ingested catalog -> appears under --kind ingested.
    res_ing = runner.invoke(cli_describe_lake, [str(lake), "--kind", "ingested"])
    assert res_ing.exit_code == 0
    assert "INGESTED_SURV" in res_ing.output
    # No product catalogs exist -> product filter yields none of our survey.
    res_prod = runner.invoke(cli_describe_lake, [str(lake), "--kind", "product"])
    assert res_prod.exit_code == 0
    assert "INGESTED_SURV" not in res_prod.output


def test_describe_lake_areas_block_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.discovery.areas import make_area, save_area
    from data_lake.discovery.region import Region
    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "AREA_SURV")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)
    save_area(lake, make_area("MyArea", Region.cone(10.0, 20.0, 60.0)))

    result = CliRunner().invoke(cli_describe_lake, [str(lake), "--areas"])
    assert result.exit_code == 0
    assert "Areas:" in result.output
    assert "MyArea" in result.output


def test_describe_lake_verbose_and_json_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "SURV_VERB")
    lake = tmp_path / "lake"
    refresh_lake_registry(lake)

    assert cli_describe_lake is not None
    runner = CliRunner()
    verbose = runner.invoke(cli_describe_lake, [str(lake), "--verbose"])
    assert verbose.exit_code == 0
    assert "link_mode" in verbose.output or "column:TARGETID" in verbose.output
    assert "tiles" in verbose.output.lower() or "ckpt" in verbose.output

    json_out = runner.invoke(cli_describe_lake, [str(lake), "--json"])
    assert json_out.exit_code == 0
    payload = json.loads(json_out.output)
    assert "entries" in payload
    assert payload["entries"][0]["survey"] == "SURV_VERB"


def test_describe_lake_pair_surveys_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from data_lake.lake_registry import cli_describe_lake

    _ingest_mini_catalog(tmp_path, "PAIR_CLI")
    lake = tmp_path / "lake"
    for label in ("a", "b"):
        (tmp_path / f"{label}.fits").touch()
    ingest_spectra_parallel(
        file_paths=[tmp_path / "a.fits", tmp_path / "b.fits"],
        output_root=lake,
        survey_name="PAIR_CLI",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt_pair.json",
        failures_log=tmp_path / "fail_pair.jsonl",
        show_progress=False,
        decoder=_mini_spectrum_decoder,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    refresh_lake_registry(lake)

    assert cli_describe_lake is not None
    result = CliRunner().invoke(
        cli_describe_lake,
        [str(lake), "--pair-surveys", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "pairing" in payload
    assert any(p["survey"] == "PAIR_CLI" for p in payload["pairing"])


def test_registry_scans_products_directory(tmp_path: Path) -> None:
    """Products under products/<name>/ appear in the registry with kind=product."""
    import json

    from data_lake.lake_registry import build_lake_registry_table
    from data_lake.schema_registry import CATALOG_KIND_PRODUCT

    lake = tmp_path / "lake"
    # Ingested survey under catalogs/
    cat_dir = lake / "catalogs" / "WISE"
    cat_dir.mkdir(parents=True)
    (cat_dir / "catalog_info.json").write_text(json.dumps({
        "hats_order": 5,
        "kind": "ingested",
        "ra_column": "ra",
        "dec_column": "dec",
    }))

    # Derived product under products/
    prod_dir = lake / "products" / "EUCLID_wise_ab"
    prod_dir.mkdir(parents=True)
    (prod_dir / "catalog_info.json").write_text(json.dumps({
        "hats_order": 5,
        "kind": "product",
        "product_subtype": "homogenized",
        "ra_column": "ra",
        "dec_column": "dec",
    }))

    table = build_lake_registry_table(lake)
    surveys = dict(zip(table["survey"].to_pylist(), table["path"].to_pylist()))
    assert "WISE" in surveys
    assert "EUCLID_wise_ab" in surveys
    assert surveys["EUCLID_wise_ab"].startswith("products/")
    kinds = dict(zip(table["survey"].to_pylist(), table["kind"].to_pylist()))
    assert kinds["EUCLID_wise_ab"] == CATALOG_KIND_PRODUCT


def test_describe_lake_version_flag() -> None:
    import tomllib
    from click.testing import CliRunner

    from data_lake import __version__
    from data_lake.lake_registry import cli_describe_lake

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        expected = tomllib.load(fh)["project"]["version"]

    assert cli_describe_lake is not None
    assert __version__ == expected
    result = CliRunner().invoke(cli_describe_lake, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"data-lake, version {expected}"
