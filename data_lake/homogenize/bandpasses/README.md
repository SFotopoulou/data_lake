# Bandpass transmission curves

Per-band transmission curves (also called filter response functions) for photometric bands
used by `phot_ab_v1` homogenization.

## File format

Each file is an [ECSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) table with:

| Column | Unit | Description |
|--------|------|-------------|
| `wavelength` | Angstrom | Wavelength grid |
| `throughput` | (dimensionless, 0-1) | Total system response (atmosphere + optics + detector) or filter-only response as stated in the header |

ECSV header metadata keys:

| Key | Required | Example |
|-----|----------|---------|
| `band` | yes | `phot_ab_w1` |
| `system` | yes | `AB` or `Vega-as-AB` |
| `source` | yes | URL, paper reference, or `illustrative` |
| `description` | no | Human-readable note |

## Naming convention

```
<Survey>_<Band>.ecsv
```

Examples: `WISE_W1.ecsv`, `Gaia_G.ecsv`, `VISTA_Y.ecsv`.

The band name in the file header (`band:`) must match the `phot_ab_*` column name used in
homogenized catalogs, e.g. `phot_ab_w1`.

## Resolution rules

The `BandpassRegistry` (see `data_lake/homogenize/bandpass.py`) loads curves with
lake-override semantics:

1. `<lake_root>/shared/registry/bandpasses/<file>.ecsv` — lake-local override
2. `data_lake/homogenize/bandpasses/<file>.ecsv` — bundled default (this folder)

Lake overrides win over bundled defaults, enabling deployment-specific or more accurate
curves without modifying the package.

## Obtaining authoritative curves

The [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/) provides
machine-readable transmission curves for thousands of filters. To fetch a curve and
convert it to ECSV:

```python
from astropy.io.votable import parse
from astropy.table import Table
import urllib.request

url = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=WISE/WISE.W1"
t = Table.read(url, format="ascii")
t.rename_column("col1", "wavelength")  # Angstrom
t.rename_column("col2", "throughput")
t.meta["band"] = "phot_ab_w1"
t.meta["system"] = "AB"
t.meta["source"] = url
t.write("WISE_W1.ecsv", format="ascii.ecsv", overwrite=True)
```

## Adding new curves

1. Obtain the curve (SVO or survey documentation).
2. Save as `<Survey>_<Band>.ecsv` in this folder (or in `<lake_root>/shared/registry/bandpasses/` for a lake override).
3. Add a `"curve": "<Survey>_<Band>.ecsv"` entry to the matching band in `bandpass.json`.

Curves are optional — plotting falls back to a vertical error bar at `lambda_eff` when
a curve file is missing or not yet registered.
