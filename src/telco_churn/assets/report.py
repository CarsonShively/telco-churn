import pandas as pd
import dagster as dg
import json
from pathlib import Path

@dg.asset(name="report", required_resource_keys={"batch_ctx"})
def report(context: dg.AssetExecutionContext, score: pd.DataFrame, action: pd.DataFrame, summary: dict) -> dict:
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()

    score.to_parquet(ctx.scored_path, index=False)
    action.to_parquet(ctx.actions_path, index=False)

    with open(ctx.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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