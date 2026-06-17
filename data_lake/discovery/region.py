"""Region selector: resolve a sky region to HEALPix NESTED pixels.

A :class:`Region` is one of four spatial selectors (user stories A-D):

- ``npix`` – explicit pixel list/ranges at a declared source order.
- ``cone`` – ``ra, dec, radius`` circle.
- ``bbox`` – ``ra_min/ra_max, dec_min/dec_max`` box (RA wrap-around aware).
- ``moc`` – Multi-Order Coverage map (requires the optional ``mocpy`` dependency).

Every region resolves to a set of pixels **at a target HEALPix order** via
:meth:`Region.to_npix`. Because catalog, spectra and cutout layers may use
different ``hats_order``, callers resolve the region per modality using that
modality's order.

Pixel indices are HEALPix **NESTED** scheme throughout, matching the on-disk
``Npix=*`` partitioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import healpy as hp
import numpy as np

RegionType = Literal["npix", "cone", "bbox", "moc"]


def rescale_npix_nested(
    npix: Iterable[int],
    source_norder: int,
    target_norder: int,
) -> set[int]:
    """Rescale NESTED pixel indices between HEALPix orders.

    - ``target == source``: returned unchanged.
    - ``target > source`` (finer): each pixel expands to its ``4**(t-s)`` nested
      children ``[npix << 2*delta, (npix+1) << 2*delta)``.
    - ``target < source`` (coarser): each pixel maps to its parent
      ``npix >> 2*(s-t)`` (deduplicated).

    This is exact for the NESTED indexing scheme.
    """
    if source_norder < 0 or target_norder < 0:
        raise ValueError("HEALPix orders must be >= 0")
    src = {int(p) for p in npix}
    if target_norder == source_norder:
        return src
    if target_norder > source_norder:
        delta = target_norder - source_norder
        shift = 2 * delta
        out: set[int] = set()
        for p in src:
            base = p << shift
            out.update(range(base, base + (1 << shift)))
        return out
    shift = 2 * (source_norder - target_norder)
    return {p >> shift for p in src}


def _cone_npix(ra_deg: float, dec_deg: float, radius_arcsec: float, norder: int) -> set[int]:
    nside = hp.order2nside(norder)
    theta = math.radians(90.0 - dec_deg)
    phi = math.radians(ra_deg)
    vec = hp.ang2vec(theta, phi)
    radius_rad = math.radians(radius_arcsec / 3600.0)
    pixels = hp.query_disc(nside, vec, radius_rad, nest=True, inclusive=True)
    return {int(p) for p in pixels}


def _normalize_dec(dec: float) -> float:
    return max(-90.0, min(90.0, float(dec)))


def _bbox_polygon_npix(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    norder: int,
) -> set[int]:
    """Pixels inside an RA/Dec box at *norder*, handling RA wrap-around.

    When ``ra_min > ra_max`` the box is interpreted as wrapping through 0/360 and
    is split into two sub-boxes. Dec is clamped to [-90, 90].
    """
    dec_lo = _normalize_dec(min(dec_min, dec_max))
    dec_hi = _normalize_dec(max(dec_min, dec_max))
    nside = hp.order2nside(norder)

    def _one(a0: float, a1: float) -> set[int]:
        # Degenerate width -> nothing.
        if a1 <= a0:
            return set()
        # query_polygon needs corners in order; build a CCW rectangle.
        corners = [
            (a0, dec_lo),
            (a1, dec_lo),
            (a1, dec_hi),
            (a0, dec_hi),
        ]
        thetas = [math.radians(90.0 - d) for _, d in corners]
        phis = [math.radians(r) for r, _ in corners]
        vecs = np.array([hp.ang2vec(t, p) for t, p in zip(thetas, phis)])
        try:
            pix = hp.query_polygon(nside, vecs, nest=True, inclusive=True)
        except Exception:
            # Fall back to a disc covering the box (robust near poles / thin boxes).
            return _bbox_disc_fallback(a0, a1, dec_lo, dec_hi, norder)
        return {int(p) for p in pix}

    ra0 = float(ra_min) % 360.0
    ra1 = float(ra_max) % 360.0
    if ra0 == ra1:
        # Full RA range at these decs.
        return _one(0.0, 359.999999) | _one(0.0, 180.0) | _one(180.0, 359.999999)
    if ra0 < ra1:
        return _one(ra0, ra1)
    # Wrap-around: [ra0, 360) U [0, ra1]
    return _one(ra0, 360.0) | _one(0.0, ra1)


def _bbox_disc_fallback(
    ra_min: float, ra_max: float, dec_min: float, dec_max: float, norder: int
) -> set[int]:
    ra_c = 0.5 * (ra_min + ra_max)
    dec_c = 0.5 * (dec_min + dec_max)
    # Half-diagonal in degrees (approximate, with cos(dec) RA scaling).
    cos_dec = max(math.cos(math.radians(dec_c)), 1e-6)
    dra = 0.5 * (ra_max - ra_min) * cos_dec
    ddec = 0.5 * (dec_max - dec_min)
    radius_deg = math.hypot(dra, ddec) + hp.nside2resol(hp.order2nside(norder), arcmin=True) / 60.0
    return _cone_npix(ra_c, dec_c, radius_deg * 3600.0, norder)


@dataclass(frozen=True)
class Region:
    """A sky-region selector resolvable to HEALPix NESTED pixels at any order.

    Construct via the classmethods (:meth:`from_npix`, :meth:`cone`,
    :meth:`bbox`, :meth:`from_moc`) or :meth:`from_dict`.
    """

    type: RegionType
    # npix
    npix: tuple[int, ...] = ()
    source_norder: int | None = None
    # cone
    ra_deg: float | None = None
    dec_deg: float | None = None
    radius_arcsec: float | None = None
    # bbox
    ra_min: float | None = None
    ra_max: float | None = None
    dec_min: float | None = None
    dec_max: float | None = None
    # moc
    moc_path: str | None = None
    moc_string: str | None = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_npix(cls, npix: Iterable[int], source_norder: int) -> "Region":
        pix = tuple(sorted({int(p) for p in npix}))
        return cls(type="npix", npix=pix, source_norder=int(source_norder))

    @classmethod
    def cone(cls, ra_deg: float, dec_deg: float, radius_arcsec: float) -> "Region":
        if radius_arcsec <= 0:
            raise ValueError("cone radius must be > 0")
        return cls(
            type="cone",
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            radius_arcsec=float(radius_arcsec),
        )

    @classmethod
    def bbox(cls, ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> "Region":
        return cls(
            type="bbox",
            ra_min=float(ra_min),
            ra_max=float(ra_max),
            dec_min=float(dec_min),
            dec_max=float(dec_max),
        )

    @classmethod
    def from_moc(cls, *, path: str | Path | None = None, moc_string: str | None = None) -> "Region":
        if path is None and moc_string is None:
            raise ValueError("from_moc requires either path or moc_string")
        return cls(
            type="moc",
            moc_path=str(path) if path is not None else None,
            moc_string=moc_string,
        )

    # -- resolution ---------------------------------------------------------

    def to_npix(self, target_norder: int) -> set[int]:
        """Resolve this region to a set of NESTED pixel indices at *target_norder*."""
        if target_norder < 0:
            raise ValueError("target_norder must be >= 0")
        if self.type == "npix":
            if self.source_norder is None:
                raise ValueError("npix region requires source_norder")
            return rescale_npix_nested(self.npix, self.source_norder, target_norder)
        if self.type == "cone":
            return _cone_npix(
                self.ra_deg, self.dec_deg, self.radius_arcsec, target_norder  # type: ignore[arg-type]
            )
        if self.type == "bbox":
            return _bbox_polygon_npix(
                self.ra_min, self.ra_max, self.dec_min, self.dec_max, target_norder  # type: ignore[arg-type]
            )
        if self.type == "moc":
            return self._moc_to_npix(target_norder)
        raise ValueError(f"unknown region type {self.type!r}")

    def _moc_to_npix(self, target_norder: int) -> set[int]:
        try:
            from mocpy import MOC  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "MOC regions require the optional 'mocpy' dependency. "
                "Install with: pip install mocpy"
            ) from exc

        if self.moc_path is not None:
            moc = MOC.from_fits(self.moc_path)
        else:
            moc = MOC.from_string(self.moc_string)  # type: ignore[arg-type]

        # Degrade/keep to the target order, then enumerate NESTED pixels.
        max_order = getattr(moc, "max_order", target_norder)
        if max_order > target_norder:
            moc = moc.degrade_to_order(target_norder)
        out: set[int] = set()
        # mocpy exposes (order, npix) cell iteration differently across versions;
        # flatten to target order via the uniq/cells interface.
        try:
            cells = moc.flatten()  # array of NESTED npix at moc.max_order
            cell_order = moc.max_order
            for p in cells:
                out |= rescale_npix_nested([int(p)], cell_order, target_norder)
        except AttributeError:  # pragma: no cover - version fallback
            for order, ipix in moc.to_json().items():  # type: ignore[attr-defined]
                out |= rescale_npix_nested([int(p) for p in ipix], int(order), target_norder)
        return out

    # -- (de)serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        if self.type == "npix":
            return {"type": "npix", "npix": list(self.npix), "source_norder": self.source_norder}
        if self.type == "cone":
            return {
                "type": "cone",
                "ra_deg": self.ra_deg,
                "dec_deg": self.dec_deg,
                "radius_arcsec": self.radius_arcsec,
            }
        if self.type == "bbox":
            return {
                "type": "bbox",
                "ra_min": self.ra_min,
                "ra_max": self.ra_max,
                "dec_min": self.dec_min,
                "dec_max": self.dec_max,
            }
        if self.type == "moc":
            d: dict[str, Any] = {"type": "moc"}
            if self.moc_path is not None:
                d["moc_path"] = self.moc_path
            if self.moc_string is not None:
                d["moc_string"] = self.moc_string
            return d
        raise ValueError(f"unknown region type {self.type!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Region":
        rtype = d.get("type")
        if rtype == "npix":
            npix = _parse_npix_spec(d.get("npix", []))
            source_norder = d.get("source_norder")
            if source_norder is None:
                raise ValueError("npix region requires 'source_norder'")
            return cls.from_npix(npix, int(source_norder))
        if rtype == "cone":
            return cls.cone(
                float(d["ra_deg"]), float(d["dec_deg"]), float(d["radius_arcsec"])
            )
        if rtype == "bbox":
            return cls.bbox(
                float(d["ra_min"]),
                float(d["ra_max"]),
                float(d["dec_min"]),
                float(d["dec_max"]),
            )
        if rtype == "moc":
            return cls.from_moc(path=d.get("moc_path"), moc_string=d.get("moc_string"))
        raise ValueError(f"unknown region type {rtype!r}")


def _parse_npix_spec(spec: Any) -> list[int]:
    """Accept a list of ints and/or ``"start-end"`` range strings."""
    if isinstance(spec, (str, bytes)):
        spec = [spec]
    out: list[int] = []
    for item in spec:
        if isinstance(item, str) and "-" in item:
            lo, hi = item.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return out


def parse_npix_arg(text: str) -> list[int]:
    """Parse a CLI ``--npix`` string: comma-separated ints and ``a-b`` ranges."""
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    return _parse_npix_spec(tokens)
