CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.offline AS
SELECT * FROM read_parquet(?);