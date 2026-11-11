# Source visualisation: SED + 1D spectrum

Two commands draw quick-look figures for one source:

- `dl-plot-spectrum` plots a single 1D spectrum from a survey's spectrum store. It needs only the survey name and the source's lake `_source_id`.
- `dl-plot-source` draws a two-panel figure. The top panel is the spectral energy distribution (SED), built from the AB magnitudes in a homogenized product catalog. The bottom panel is the 1D spectrum.

Use `dl-plot-spectrum` to check a spectrum. Use `dl-plot-source` once you have a homogenized product and want to compare photometry with spectroscopy.

## Install the viz extra

Plotting needs matplotlib, which is an optional dependency:

```bash
pip install "data-lake[viz]"
# or with uv:
uv sync --extra viz
```

## Quick start

```bash
export DATA_LAKE_CONFIG=/home/user/como/lake_config.toml

# Spectrum only (survey + _source_id)
dl-plot-spectrum DESI_DR1 --id 1234567890 -o desi_123.png

# Two-panel SED + spectrum (needs a homogenized product)
dl-plot-source \
    --id 1234567890 \
    --product EDFF_cone_joined \
    --spectra-survey SDSS_DR17 \
    -o source_1234567890.png
```

Both commands save the figure and print its path to stdout. Without `-o`, `dl-plot-spectrum` writes `spectrum_<survey>_<id>.png` and `dl-plot-source` writes `source_<id>.png` in the current directory. Wavelengths are shown in micron. The SED panel shows each band at its effective wavelength with error bars. The spectrum panel shows observed-frame flux with the 1σ uncertainty shaded.

## Source ID semantics

`--id` must be the lake `_source_id`, the int64 key that links a survey's catalog Parquet tiles to its Zarr spectrum stacks. It is not the survey's native identifier (such as `TARGETID` or `SOURCE_ID`), unless the catalog was ingested with that integer column as `--link-id-col` (for example `--link-id-col TARGETID`). In that case the two values are the same.

To find the `_source_id` for a native identifier, query the catalog:

```python
from data_lake.io.catalog import CatalogAccessor
cat = CatalogAccessor("/shared/como", "SDSS_DR17")
df = cat.query("SELECT _source_id, SPECOBJID FROM catalog WHERE SPECOBJID = 123")
```

Or from the command line:

```bash
dl-extract-catalog --survey SDSS_DR17 -c _source_id,SPECOBJID \
    --where "SPECOBJID = 123" -o /tmp/lookup.csv

# Peek a few IDs from the spectrum store
dl-preview-sources DESI_DR1 --modality spectra -n 5
```

## Spectrum-only options (`dl-plot-spectrum`)

The lake root is an optional positional argument. You can leave it out when `--config` or `$DATA_LAKE_CONFIG` is set.

```
dl-plot-spectrum SURVEY [LAKE_ROOT] --id INTEGER [OPTIONS]

Options:
  --id INTEGER             Lake _source_id (required)
  -o, --output PATH        Output file: .png, .pdf, .svg
                           [default: spectrum_<survey>_<id>.png]
  --dpi INTEGER            DPI for raster output [default: 150]
  --title TEXT             Figure title override
  --rest-frame             Convert to rest-frame wavelength using the stored redshift
  --linear-x               Linear wavelength axis (default: log)
  --clip-sigma FLOAT       Y-axis median ± N×MAD-σ clip [default: 10]; 0 disables
  --config PATH            Path to lake_config.toml (or $DATA_LAKE_CONFIG)
  -q / -v                  Quiet / verbose logging
```

If the survey has a catalog, it is used to find the spectrum's tile. Otherwise the command scans the Zarr tiles, which is slower but works.

## SED + spectrum options (`dl-plot-source`)

The lake root is passed as `--lake-root`, not as a positional argument.

```
dl-plot-source [OPTIONS]

Options:
  --id INTEGER             Lake _source_id integer (required)
  --product TEXT           Homogenized product catalog name (required)
  --spectra-survey TEXT    Spectra survey name (required)
  --lake-root PATH         Lake root directory (or set $DATA_LAKE_CONFIG)
  -o, --output PATH        Output file: .png, .pdf, .svg [default: source_<id>.png]
  --dpi INTEGER            DPI for raster output [default: 150]
  --title TEXT             Figure suptitle [default: "Source <id>"]
  --rest-frame             Convert spectrum to rest-frame wavelength using the stored redshift
  --no-curves              Suppress bandpass transmission-curve shading
  --config PATH            Path to lake_config.toml (or $DATA_LAKE_CONFIG)
  -q, --quiet              Suppress INFO output
  -v, --verbose            Print DEBUG output
  --help                   Show this message and exit.
```

If no bands with a known wavelength are found for the source, the command prints a warning and draws an empty SED panel. If the source is missing from the product or the spectrum store, it exits with an error.

## Photometry source: homogenized product catalog

The SED panel reads every `phot_ab_*` column in the product, together with its matching `phot_ab_*_err` column. Bands with a null value are skipped, and so are bands without an effective wavelength in the `BandpassRegistry`. Neither case produces a warning.

Make the product with `dl-homogenize`, or with `dl-gather` followed by `dl-homogenize`, using the `phot_ab_v1` transform (see [homogenization.md](homogenization.md)). Run `dl-describe-lake --kind product` to list the available products.

## Bandpass registry

`BandpassRegistry` (`data_lake.homogenize.bandpass`) provides two things:

- The effective wavelength (`lambda_eff_um`) and width (`fwhm_um`) of each `phot_ab_*` band, from `data_lake/homogenize/bandpass.json`.
- Optional transmission curves stored as ECSV files, which are drawn as semi-transparent shading behind the SED.

### Resolution rules

Curve files are looked up in this order, and the first match wins:

1. `<lake_root>/shared/registry/bandpasses/<file>.ecsv` (lake-local override)
2. `data_lake/homogenize/bandpasses/<file>.ecsv` (bundled default)

Use the lake directory for deployment-specific curves, so you don't have to modify the package. If a curve file is missing, the band is plotted without shading.

### Adding transmission curves

`data_lake/homogenize/bandpasses/README.md` describes the ECSV format and how to download authoritative curves from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/).

To add a curve for a band that is already in `bandpass.json`:

1. Save the curve as `<Survey>_<Band>.ecsv` in `data_lake/homogenize/bandpasses/` or in the lake override directory.
2. Add `"curve": "<Survey>_<Band>.ecsv"` to the band's entry in `data_lake/homogenize/bandpass.json`.

Only `WISE_W1.ecsv` and `Gaia_G.ecsv` are bundled, and they are illustrative rather than authoritative. `bandpass.json` also points `phot_ab_w2` to `WISE_W2.ecsv`, which is not bundled. That band is drawn without shading unless you add the file.

### All currently registered bands

| `phot_ab_*` column | λ_eff (µm) | FWHM (µm) | Reference |
|--------------------|------------|-----------|-----------|
| `phot_ab_bp` | 0.532 | 0.244 | Gaia BP |
| `phot_ab_g` | 0.6735 | 0.4434 | Gaia G |
| `phot_ab_rp` | 0.797 | 0.294 | Gaia RP |
| `phot_ab_z` | 0.882 | 0.100 | VISTA Z |
| `phot_ab_y` | 1.021 | 0.101 | VISTA Y |
| `phot_ab_j` | 1.253 | 0.172 | 2MASS / VISTA J |
| `phot_ab_h` | 1.643 | 0.291 | 2MASS / VISTA H |
| `phot_ab_k` | 2.159 | 0.262 | 2MASS Ks |
| `phot_ab_ks` | 2.151 | 0.312 | VISTA Ks |
| `phot_ab_w1` | 3.368 | 0.662 | WISE W1 |
| `phot_ab_w2` | 4.618 | 1.042 | WISE W2 |
| `phot_ab_w3` | 12.082 | 5.509 | WISE W3 |
| `phot_ab_w4` | 22.194 | 4.098 | WISE W4 |

## Python API

The same figure can be built in Python. This is what `dl-plot-source` does internally:

```python
import matplotlib
matplotlib.use("Agg")

from data_lake.io.catalog import CatalogAccessor
from data_lake.io.spectra import SpectrumAccessor
from data_lake.homogenize.bandpass import BandpassRegistry
from data_lake.plot.sed import assemble_sed
from data_lake.plot.source_figure import plot_source_sed_spectrum

lake = "/shared/como"
source_id = 1234567890

product = CatalogAccessor(lake, "EDFF_cone_joined")
spec_acc = SpectrumAccessor(lake, "SDSS_DR17", catalog_accessor=product)
bp = BandpassRegistry(lake_root=lake)

sed = assemble_sed(product, source_id, bp)
spectrum = spec_acc.get_spectrum(source_id)
fig = plot_source_sed_spectrum(sed, spectrum, bandpass_registry=bp)
fig.savefig("source.png", dpi=150, bbox_inches="tight")
```

The optional `catalog_accessor` argument of `SpectrumAccessor` is used to find a spectrum's tile from the catalog's `_spectrum_index` column. If that lookup fails, `get_spectrum` falls back to scanning the spectrum tiles. A joined product may not carry `_spectrum_index` for the spectra survey, so pass `CatalogAccessor(lake, "SDSS_DR17")` instead when you need faster lookups. For many sources, call `spec_acc.locate(ids)` once and then `spec_acc.get_spectrum_at(npix, local)` for each source, instead of calling `get_spectrum` per ID. `plot_spectrum` in `data_lake.plot.source_figure` draws the spectrum-only figure.

## Known limitations

These features are not implemented yet:

- There is no `--y-unit fnu|flam` option to convert SED points to flux density and overlay them on the spectrum in a single panel.
- Authoritative SVO transmission curves are not bundled for every registered band.
- EUCLID, DESI and SDSS have no bundled `phot_ab_*` catalog recipes, so their bands do not appear in the SED until you add a recipe (see [homogenization.md](homogenization.md)).

See also: [homogenization.md](homogenization.md), [`dl-homogenize`](cli-reference.md).
