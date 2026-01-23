import pandas as pd
import dagster as dg
import json
from pathlib import Path

@dg.asset(name="batch_report", required_resource_keys={"batch_ctx"})
def batch_report(context: dg.AssetExecutionContext, batch_scored_df: pd.DataFrame, batch_action_df: pd.DataFrame, batch_summary: dict) -> dict:
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()

    batch_scored_df.to_parquet(ctx.scored_path, index=False)
    batch_action_df.to_parquet(ctx.actions_path, index=False)

    with open(ctx.summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    for p in [ctx.scored_path, ctx.actions_path, ctx.summary_path]:
        p = Path(p)
        if not p.exists():
            raise RuntimeError(f"Missing output file: {p}")

    return {
        "scored_path": str(ctx.scored_path),
        "actions_path": str(ctx.actions_path),
        "summary_path": str(ctx.summary_path),
        "hf_batch_path": ctx.hf_batch_path,
        "batch_id": ctx.batch_id,
    }