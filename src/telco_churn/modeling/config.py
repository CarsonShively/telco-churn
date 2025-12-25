"""Model training constants."""

TARGET_COL = "churn"

PRIMARY_METRIC = "average_precision"
METRIC_DIRECTION = "maximize"
DEFAULT_THRESHOLD = 0.5

HOLDOUT_SIZE = 0.2
SEED = 42
CV_SPLITS = 5