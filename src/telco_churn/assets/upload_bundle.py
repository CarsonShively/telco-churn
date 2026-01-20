import dagster as dg
from telco_churn.modeling.types import BundleOut, TrainConfig

@dg.asset(name="upload_bundle", required_resource_keys={"hf_model"})
def upload_bundle(context: dg.AssetExecutionContext, bundle: BundleOut, config: TrainConfig) -> str:
    if config.upload:
        hf_model = context.resources.hf_model
        hf_model.bundle_upload(bundle_dir=bundle.bundle_dir, run_id=bundle.run_id)
        return "run uploaded"
    return "no upload, local only"