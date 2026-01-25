from pathlib import Path
import dagster as dg
from telco_churn.batch.latest_batch import write_latest_pointer

@dg.asset(
    name="upload_batch_report",
    required_resource_keys={"hf_data", "batch_ctx"},
    config_schema={"upload": dg.Field(bool, default_value=False)},
)
def upload_batch_report(context: dg.AssetExecutionContext, batch_report: dict) -> dict:
    if not context.op_config["upload"]:
        context.add_output_metadata({
            "uploaded": False,
            "batch_id": str(batch_report.get("batch_id")),
        })
        return batch_report
    
    hf_data = context.resources.hf_data
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()

    scored_path = Path(batch_report["scored_path"])
    actions_path = Path(batch_report["actions_path"])
    summary_path = Path(batch_report["summary_path"])
    hf_batch_path = batch_report.get("hf_batch_path", ctx.hf_batch_path)

    hf_data.upload_data(local_path=str(scored_path),  hf_path=f"{hf_batch_path}/scored.parquet")
    hf_data.upload_data(local_path=str(actions_path), hf_path=f"{hf_batch_path}/actions.parquet")
    hf_data.upload_data(local_path=str(summary_path), hf_path=f"{hf_batch_path}/summary.json")

    latest_local = write_latest_pointer(
        reports_root=ctx.reports_root,
        batch_id=ctx.batch_id,
    )
    hf_data.upload_data(local_path=str(latest_local), hf_path="reports/latest.json")

    context.add_output_metadata({
        "uploaded": True,
        "batch_id": str(ctx.batch_id),

        "hf_batch_path": hf_batch_path,
        "scored_hf": f"{hf_batch_path}/scored.parquet",
        "actions_hf": f"{hf_batch_path}/actions.parquet",
        "summary_hf": f"{hf_batch_path}/summary.json",
        "latest_hf": "reports/latest.json",
    })

    return {
        **batch_report,
        "hf_batch_path": hf_batch_path,
        "scored_hf": f"{hf_batch_path}/scored.parquet",
        "actions_hf": f"{hf_batch_path}/actions.parquet",
        "summary_hf": f"{hf_batch_path}/summary.json",
        "latest_hf": "reports/latest.json",
    }
