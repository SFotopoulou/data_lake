# Source visualisation: SED + 1D spectrum

`dl-plot-source` renders a two-panel figure for a single astronomical source:

- **Top panel** — spectral energy distribution (SED): AB magnitudes from a homogenized product catalog, plotted at each band's effective wavelength with error bars.  Optional semi-transparent transmission-curve shading when ECSV curve files are registered.
- **Bottom panel** — 1D spectrum: observed-frame flux from the Zarr spectrum store, with 1σ uncertainty shaded.

Both panels share a logarithmic wavelength axis in micron.

## Install the viz extra

Plotting requires matplotlib, which is an optional dependency:

```bash
pip install "data-lake[viz]"
# or with uv:
uv sync --extra viz
```

## Quick start

```bash
export DATA_LAKE_CONFIG=/home/user/como/lake_config.toml

dl-plot-source \
    --id 1234567890 \
    --product EDFF_cone_joined \
    --spectra-survey SDSS_DR17 \
    -o source_1234567890.png
```

This saves `source_1234567890.png` (or prints the path to stdout).

## Source ID semantics

`--id` must be the **lake `_source_id`** integer — the stable int64 join key shared between the catalog Parquet tiles and the Zarr spectrum stacks for a given survey.  This is *not* a native survey identifier like `TARGETID` or `SOURCE_ID` unless the survey was ingested with `--link-id-mode column:TARGETID`.

To look up the `_source_id` for a native identifier, query the catalog:

```python
from data_lake.io.catalog import CatalogAccessor
cat = CatalogAccessor("/shared/como", "SDSS_DR17")
df = cat.query("SELECT _source_id, SPECOBJID FROM catalog WHERE SPECOBJID = 123")
```

Or from the CLI:

```bash
dl-extract-catalog --survey SDSS_DR17 -c _source_id -c SPECOBJID \
    --where "SPECOBJID = 123" -o /tmp/lookup.csv
```

## All options

```
dl-plot-source [OPTIONS]

Options:
  --id INTEGER             Lake _source_id integer (required)
  --product TEXT           Homogenized product catalog name (required)
  --spectra-survey TEXT    Spectra survey name (required)
  --lake-root PATH         Lake root directory (or set $DATA_LAKE_CONFIG)
  -o, --output PATH        Output file: .png (default), .pdf, .svg
  --dpi INTEGER            DPI for raster output [default: 150]
  --title TEXT             Figure suptitle
  --rest-frame             Convert spectrum to rest-frame wavelength
  --no-curves              Suppress bandpass transmission-curve shading
  --config PATH            Path to lake_config.toml (or $DATA_LAKE_CONFIG)
  -q, --quiet              Suppress INFO output
  -v, --verbose            Print DEBUG output
  --help                   Show this message and exit.
```

## Photometry source: homogenized product catalog

The SED panel reads all `phot_ab_*` columns (and matching `phot_ab_*_err` columns) present in the product catalog.  Bands without a known effective wavelength in the `BandpassRegistry` are silently skipped.

The product catalog must have been produced by `dl-homogenize` (or `dl-gather` followed by `dl-homogenize`) and contain `phot_ab_*` columns from the `phot_ab_v1` transform.  Use `dl-describe-lake --kind product` to list available products.

## Bandpass registry

The `BandpassRegistry` provides:

- `lambda_eff_um` and `fwhm_um` for each `phot_ab_*` band (from `data_lake/homogenize/bandpass.json`)
- Optional transmission-curve ECSV files (from `data_lake/homogenize/bandpasses/`)

### Resolution rules

1. `<lake_root>/shared/registry/bandpasses/<file>.ecsv` — lake-local override
2. `data_lake/homogenize/bandpasses/<file>.ecsv` — bundled default

Lake overrides win, enabling deployment-specific curves without modifying the package.

### Adding transmission curves

See `data_lake/homogenize/bandpasses/README.md` for the ECSV format and instructions for downloading authoritative curves from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/).

To add a curve for a band already in `bandpass.json`:

1. Save the curve as `<Survey>_<Band>.ecsv` in `data_lake/homogenize/bandpasses/` (or a lake override directory).
2. Add `"curve": "<Survey>_<Band>.ecsv"` to the band entry in `data_lake/homogenize/bandpass.json`.

Currently bundled (illustrative, not authoritative): `WISE_W1.ecsv`, `Gaia_G.ecsv`.

### All currently registered bands

| `phot_ab_*` column | λ_eff (µm) | FWHM (µm) | Reference |
|--------------------|------------|-----------|-----------|
| `phot_ab_bp` | 0.532 | 0.244 | Gaia BP |
| `phot_ab_g` | 0.674 | 0.443 | Gaia G |
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

## Follow-ups (not in v0.5.0)

- `--y-unit fnu|flam` overlay: convert SED points to flux density units for direct overlay with the spectrum on a single axis.
- Authoritative SVO transmission curves for all registered bands.
- EUCLID/DESI/SDSS catalog mag → `phot_ab_*` recipes (needed to populate the SED for those surveys; see [homogenization.md](homogenization.md)).

See also: [homogenization.md](homogenization.md), [`dl-homogenize`](cli-reference.md).
