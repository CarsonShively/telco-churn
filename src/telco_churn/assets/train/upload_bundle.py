import dagster as dg
from telco_churn.modeling.types import BundleOut

@dg.asset(name="upload_bundle", required_resource_keys={"hf_model", "train_cfg"})
def upload_bundle(context: dg.AssetExecutionContext, artifact_bundle: BundleOut) -> str:
    cfg = context.resources.train_cfg
    if cfg.upload:
        hf_model = context.resources.hf_model
        hf_model.bundle_upload(bundle_dir=artifact_bundle.bundle_dir, run_id=artifact_bundle.run_id)
        return "run uploaded"
    return "no upload, local only"