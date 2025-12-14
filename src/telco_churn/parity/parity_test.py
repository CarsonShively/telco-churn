import pyarrow as pa
import pyarrow.compute as pc

def _nullsafe_equal(a: pa.Array, b: pa.Array) -> bool:
    eq = pc.equal(a, b)
    eq = pc.fill_null(eq, True)
    return bool(pc.all(eq).as_py())

def _align(table: pa.Table, cols: list[str]) -> pa.Table:
    return table.select(cols)

def _sort(table: pa.Table, key: str) -> pa.Table:
    return table.sort_by([(key, "ascending")])

def assert_silver_parity(
    silver_offline: pa.Table,
    silver_online: pa.Table,
    key: str = "customer_id",
) -> None:
    off_cols = silver_offline.column_names
    on_cols = silver_online.column_names

    missing_in_online = [c for c in off_cols if c not in on_cols]
    missing_in_offline = [c for c in on_cols if c not in off_cols]
    if missing_in_online or missing_in_offline:
        raise AssertionError(
            f"Column mismatch. missing_in_online={missing_in_online}, missing_in_offline={missing_in_offline}"
        )

    silver_offline = _align(silver_offline, off_cols)
    silver_online  = _align(silver_online,  off_cols)

    silver_offline = _sort(silver_offline, key)
    silver_online  = _sort(silver_online,  key)

    if silver_offline.num_rows != silver_online.num_rows:
        raise AssertionError(
            f"Row count mismatch: offline={silver_offline.num_rows}, online={silver_online.num_rows}"
        )

    for col in off_cols:
        a = silver_offline[col].combine_chunks()
        b = silver_online[col].combine_chunks()

        if a.type != b.type:
            raise AssertionError(f"Type mismatch in '{col}': offline={a.type}, online={b.type}")

        if not _nullsafe_equal(a, b):
            raise AssertionError(f"Value mismatch in column '{col}'")
