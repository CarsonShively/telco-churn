import dagster as dg
from telco_churn.modeling.types import BundleOut

@dg.asset(name="upload_bundle", required_resource_keys={"hf_model", "train_cfg"})
def upload_bundle(context: dg.AssetExecutionContext, artifact_bundle: BundleOut) -> str:
    """Upload run bundle to model runs archive."""
    cfg = context.resources.train_cfg
    if cfg.upload:
        hf_model = context.resources.hf_model
        hf_model.bundle_upload(bundle_dir=artifact_bundle.bundle_dir, run_id=artifact_bundle.run_id)

        context.add_output_metadata({
            "uploaded": True,
            "run_id": str(artifact_bundle.run_id),
            "bundle_dir": dg.MetadataValue.path(str(artifact_bundle.bundle_dir)),
            "result": "run uploaded",
        })
        return "run uploaded"

    context.add_output_metadata({
        "uploaded": False,
        "run_id": str(artifact_bundle.run_id),
        "bundle_dir": dg.MetadataValue.path(str(artifact_bundle.bundle_dir)),
        "result": "no upload, local only",
    })
    return "no upload, local only"
