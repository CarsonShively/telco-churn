import dagster as dg
from telco_churn.promotion.type import PromotionDecision, RunRow, ChampionRef, PromotionConfig
from telco_churn.promotion.registry import write_champion_json
from telco_churn.paths import REPO_ROOT

@dg.asset(name="promote", required_resource_keys={"hf_model"})
def promote(
    context: dg.AssetExecutionContext,
    best_contender: RunRow,
    promotion_decision: PromotionDecision,
    config: PromotionConfig,
):
    hf_model = context.resources.hf_model

    if not promotion_decision.promote:
        context.add_output_metadata({
            "promoted": False,
            "reason": promotion_decision.reason,
            "upload_requested": bool(config.upload),
            "result": "No promotion",
        })
        return "No promotion"

    run_id = best_contender.run_id
    ref = ChampionRef(run_id=run_id, path_in_repo=f"runs/{run_id}")
    local_path = write_champion_json(ref, out_path=REPO_ROOT / "champion.json")

    if config.upload:
        hf_model.upload_model_json(local_path, path_in_repo="champion.json")
        context.add_output_metadata({
            "promoted": True,
            "run_id": run_id,
            "champion_ref_path_in_repo": ref.path_in_repo,
            "local_champion_json": dg.MetadataValue.path(str(local_path)),
            "uploaded": True,
            "uploaded_path_in_repo": "champion.json",
            "result": "Upload sucess",
        })
        return "Upload sucess"

    context.add_output_metadata({
        "promoted": True,
        "run_id": run_id,
        "champion_ref_path_in_repo": ref.path_in_repo,
        "local_champion_json": dg.MetadataValue.path(str(local_path)),
        "uploaded": False,
        "uploaded_path_in_repo": "champion.json",
        "result": "No upload, dry run",
    })
    return "No upload, dry run"
