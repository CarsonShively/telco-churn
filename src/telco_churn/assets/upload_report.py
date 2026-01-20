from pathlib import Path
import dagster as dg
from telco_churn.batch.latest_batch import write_latest_pointer

@dg.asset(name="upload_report", required_resource_keys={"hf_data", "batch_ctx"}, config_schema={"upload": dg.Field(bool, default_value=False)})
def upload_report(context: dg.AssetExecutionContext, report: dict) -> dict:
    if not context.op_config["upload"]:
        return report
    
    hf_data = context.resources.hf_data
    batch_ctx = context.resources.batch_ctx
    
    ctx = batch_ctx.get()

    scored_path = Path(report["scored_path"])
    actions_path = Path(report["actions_path"])
    summary_path = Path(report["summary_path"])
    hf_batch_path = report.get("hf_batch_path", ctx.hf_batch_path)

    hf_data.upload_data(local_path=str(scored_path),  hf_path=f"{hf_batch_path}/scored.parquet")
    hf_data.upload_data(local_path=str(actions_path), hf_path=f"{hf_batch_path}/actions.parquet")
    hf_data.upload_data(local_path=str(summary_path), hf_path=f"{hf_batch_path}/summary.json")

    latest_local = write_latest_pointer(reports_root=ctx.reports_root, batch_id=ctx.batch_id)
    hf_data.upload_data(local_path=str(latest_local), hf_path="reports/latest.json")

    return {
        **report,
        "hf_batch_path": hf_batch_path,
        "scored_hf": f"{hf_batch_path}/scored.parquet",
        "actions_hf": f"{hf_batch_path}/actions.parquet",
        "summary_hf": f"{hf_batch_path}/summary.json",
        "latest_hf": "reports/latest.json",
        }