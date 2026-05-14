-- Cross-survey join: LSST photometry (g_mag filter) + master associations.
-- Tokens __LAKE__, __LSST_PARQUET_GLOB__, __MASTER_PARQUET__ are replaced by demo.py
-- (or substitute manually and run: duckdb < query.sql).

CREATE OR REPLACE VIEW lsst AS
SELECT * FROM read_parquet('__LSST_PARQUET_GLOB__', hive_partitioning=false);

CREATE OR REPLACE VIEW master AS
SELECT * FROM read_parquet('__MASTER_PARQUET__');

SELECT
    l.source_id AS lsst_id,
    l.g_mag,
    l.ra,
    l.dec,
    m.desi_targetid,
    m.euclid_source_id,
    m.healpix_npix,
    m.norder,
    m.desi_zarr_row,
    m.euclid_zarr_row,
    '__LAKE__/spectra/__DESI_SURVEY__/Norder=' || m.norder::VARCHAR
        || '/Dir=' || ((m.healpix_npix / 10000)::BIGINT * 10000)::VARCHAR
        || '/Npix=' || m.healpix_npix::VARCHAR || '.zarr' AS spectrum_zarr_path,
    '__LAKE__/cutouts/__EUCLID_SURVEY__/Norder=' || m.norder::VARCHAR
        || '/Dir=' || ((m.healpix_npix / 10000)::BIGINT * 10000)::VARCHAR
        || '/Npix=' || m.healpix_npix::VARCHAR || '.zarr' AS cutout_zarr_path
FROM lsst l
INNER JOIN master m ON l.source_id = m.lsst_source_id
WHERE l.g_mag < 22.0
ORDER BY l.source_id;
