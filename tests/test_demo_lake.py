"""dl-demo-lake: synthetic deployment built through the real ingest path."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from data_lake import demo
from data_lake.config import LakeConfig
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.cutouts import CutoutAccessor
from data_lake.io.spectra import SpectrumAccessor
from data_lake.lake_registry import load_lake_registry

SMALL = demo.DemoSizes(n_phot=80, n_spec=12, n_cutouts=6, n_ir_only=5)


@pytest.fixture(scope="module")
def demo_config(tmp_path_factory) -> Path:
    return demo.build_demo_lake(tmp_path_factory.mktemp("demo") / "lake", sizes=SMALL)


def _lake(config: Path) -> Path:
    return Path(LakeConfig.discover(config).lake.root)


def test_registry_lists_every_survey_and_modality(demo_config: Path) -> None:
    reg = load_lake_registry(_lake(demo_config)).to_pylist()
    rows = {(r["survey"], r["modality"]): r["total_rows"] for r in reg}
    assert rows[(demo.PHOT_SURVEY, "catalog")] == SMALL.n_phot
    assert rows[(demo.SPEC_SURVEY, "catalog")] == SMALL.n_spec
    assert rows[(demo.SPEC_SURVEY, "spectra")] == SMALL.n_spec
    assert rows[(demo.PHOT_SURVEY, "cutout")] == SMALL.n_cutouts
    assert rows[(demo.IR_SURVEY, "catalog")] > SMALL.n_ir_only


def test_spectra_and_cutouts_are_linked(demo_config: Path) -> None:
    lake = _lake(demo_config)
    with CatalogAccessor(lake, demo.SPEC_SURVEY) as cat:
        ids = cat.query("SELECT _source_id FROM catalog WHERE _spectrum_index >= 0")["_source_id"].to_list()
        assert len(ids) == SMALL.n_spec
        with SpectrumAccessor(lake, demo.SPEC_SURVEY, catalog_accessor=cat) as acc:
            npix, local = acc.locate(ids)
            assert (npix >= 0).all()
            sp = acc.get_spectrum_at(int(npix[0]), int(local[0]))
            assert sp.flux.size == demo._wavelength().size

    with CatalogAccessor(lake, demo.PHOT_SURVEY) as cat:
        ids = cat.query("SELECT _source_id FROM catalog WHERE _cutout_index >= 0")["_source_id"].to_list()
        assert len(ids) == SMALL.n_cutouts
        with CutoutAccessor(lake, demo.PHOT_SURVEY, catalog_accessor=cat) as acc:
            npix, local = acc.locate(ids)
            img = acc.get_image_at(int(npix[0]), int(local[0]))
            assert img.shape == (len(demo.CUTOUT_BANDS), demo.CUTOUT_SIZE, demo.CUTOUT_SIZE)


def test_recipes_and_raw_inputs_are_written(demo_config: Path) -> None:
    recipes = _lake(demo_config) / "shared" / "registry" / "homogenize"
    assert {p.stem for p in recipes.glob("*.json")} == {demo.PHOT_SURVEY, demo.IR_SURVEY}
    raw = demo_config.parent / "raw"
    assert (raw / f"{demo.PHOT_SURVEY}.fits").is_file()
    assert len(list((raw / "spectra").glob("*.fits"))) == SMALL.n_spec


def test_same_seed_is_reproducible(tmp_path: Path) -> None:
    a = demo.write_demo_inputs(tmp_path / "a", seed=7, sizes=SMALL)
    b = demo.write_demo_inputs(tmp_path / "b", seed=7, sizes=SMALL)
    from astropy.table import Table

    ta, tb = Table.read(a.phot_catalog), Table.read(b.phot_catalog)
    assert (ta["MAG_Z"] == tb["MAG_Z"]).all()


def test_ensure_reuses_complete_and_rebuilds_partial(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []
    real = demo.build_demo_lake

    def counting(path, **kw):
        calls.append(Path(path))
        return real(path, sizes=SMALL, **{k: v for k, v in kw.items() if k != "sizes"})

    monkeypatch.setattr(demo, "build_demo_lake", counting)
    target = tmp_path / "lake"
    cfg = demo.ensure_demo_lake(target)
    assert demo.ensure_demo_lake(target) == cfg
    assert len(calls) == 1

    (target / demo._COMPLETE_MARKER).unlink()
    demo.ensure_demo_lake(target)
    assert len(calls) == 2


def test_default_dir_honours_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(demo.DEMO_DIR_ENV, str(tmp_path / "x"))
    assert demo.default_demo_dir() == (tmp_path / "x").resolve()


def test_cli_refuses_existing_path_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "lake").mkdir()
    result = CliRunner().invoke(demo.cli, [str(tmp_path / "lake"), "--n-sources", "80"])
    assert result.exit_code != 0
    assert "exists" in result.output


def test_sizes_are_validated() -> None:
    with pytest.raises(ValueError):
        demo.DemoSizes(n_phot=50, n_spec=60)
