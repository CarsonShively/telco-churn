"""Project-wide constant paths and defaults for telco-churn pipelines."""

REPO_ID = "Carson-Shively/telco-churn"
REVISION = "main"

BRONZE_OFFLINE_PARQUET = "data/bronze/train.parquet"
BRONZE_ONLINE_PARQUET = "data/bronze/demo.parquet"

GOLD_TRAIN_PARQUET = "data/gold/train.parquet"

DUCKDB_PATH = ":memory:"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

REDIS_BASE_PREFIX = "telco:features:"
REDIS_CURRENT_POINTER_KEY = "telco:features:CURRENT"
REDIS_RUN_META_PREFIX = "telco:features:RUN_META:"

REDIS_TTL_SECONDS = 0

FEATURES_TABLE = "gold.features"
ENTITY_COL = "customer_id"

TRAIN_HF_PATH = GOLD_TRAIN_PARQUET

FLAG_RATE = 0.05

CURRENT_ARTIFACT_VERSION = 7