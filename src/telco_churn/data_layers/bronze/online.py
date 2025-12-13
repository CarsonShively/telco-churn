import pyarrow.parquet as pq
import pyarrow as pa

def read_parquet_arrow(path: str) -> pa.Table:
    return pq.read_table(path)
