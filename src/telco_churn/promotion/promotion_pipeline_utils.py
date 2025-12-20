from __future__ import annotations

from telco_churn.promotion.new_champion import (
    ChampionRef,
    write_champion_json_local,
    upload_existing_champion_json,
)

def get_run_id(best_row) -> str:
    run_id = getattr(best_row, "run_id", None)
    if run_id:
        return run_id

    metrics = getattr(best_row, "metrics", {}) or {}
    run_id = metrics.get("run_id")
    if run_id:
        return run_id

    raise ValueError("Best contender missing run_id")

def promote_run_as_champion(
    *,
    run_id: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str = "champion.json",
) -> ChampionRef:
    ref = ChampionRef(run_id=run_id, path_in_repo=f"runs/{run_id}")
    local_path = write_champion_json_local(ref)

    upload_existing_champion_json(
        local_path,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
    )
    return ref