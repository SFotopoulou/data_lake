#!/usr/bin/env python3
"""Wrapper so you can run ``python scripts/validate_spectra_ingest.py ...`` from a clone."""
from __future__ import annotations

from data_lake.ingest.validate_spectra_ingest import cli

if __name__ == "__main__":
    cli()
