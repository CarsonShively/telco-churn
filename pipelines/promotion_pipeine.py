from __future__ import annotations

from telco_churn.io.hf_run_metrics import fetch_all_run_metrics
from telco_churn.promotion.best_candidate import pick_best_contender 
from telco_churn.promotion.contender_vs_champion import try_load_champion_ref, load_champion_metrics
from telco_churn.promotion.promotion_decision import decide_promotion
from telco_churn.promotion.promotion_pipeline_utils import get_run_id, promote_run_as_champion
from telco_churn.config import REPO_ID, REVISION


def main() -> None:

    rows = fetch_all_run_metrics(
        repo_id=REPO_ID,
        revision=REVISION,
    )
    
    ok_rows = []
    for r in rows:
        if r.error:
            print(f"{r.run_id} FAILED: {r.error}")
            continue
        ok_rows.append(r)
    try:
        best_row = pick_best_contender(ok_rows)
    except ValueError as e:
        print(f"Promotion pipeline stopped: {e}")
        return

    ref = try_load_champion_ref(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
    )
    
    if ref is not None:
        champ_metrics = load_champion_metrics(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            ref=ref,
        )

        cont_metrics = best_row.metrics

        decision = decide_promotion(
            contender_metrics=cont_metrics,
            champion_metrics=champ_metrics,
        )

        if decision.promote:
            run_id = get_run_id(best_row)
            promote_run_as_champion(
                run_id=run_id,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )
        else:
            print("Champion wins:", decision.reason)
        
    else:
        run_id = get_run_id(best_row)
        promote_run_as_champion(
            run_id=run_id,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )


if __name__ == "__main__":
    main()
