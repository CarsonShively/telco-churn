from __future__ import annotations
import argparse
from pathlib import Path
from telco_churn.io.hf_run_metrics import fetch_all_run_metrics
from telco_churn.promotion.best_candidate import get_best_contender 
from telco_churn.io.hf import read_model_json, upload_model_json_hf
from telco_churn.promotion.registry import ChampionRef, write_champion_json
from telco_churn.promotion.decision import decide_promotion
from telco_churn.config import REPO_ID, REVISION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = PROJECT_ROOT / "champion.json"
EPSILON = 0.001

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best contender and optionally promote it to champion."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If set, apply the promotion (write/upload champion.json). Otherwise, dry-run only.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    rows = fetch_all_run_metrics(
        repo_id=REPO_ID,
        revision=REVISION,
    )
    
    try:
        best_row = get_best_contender(rows)
    except ValueError as e:
        print(f"Promotion pipeline stopped: {e}")
        return

    champion_metrics = read_model_json(
        repo_id=REPO_ID,
        revision=REVISION,
        path_in_repo=CHAMPION_PATH
    )
    
    contender_metrics = best_row.metrics
    
    decision = decide_promotion(
        contender_metrics=contender_metrics,
        champion_metrics=champion_metrics,
        epsilon=EPSILON,
    )


    if decision.promote:
        ref = ChampionRef(
            run_id=best_row["run_id"],
            path_in_repo=f"runs/{best_row['run_id']}",
        )

        local_path = write_champion_json(ref, out_path=CHAMPION_PATH)
        if args.promote:
            upload_model_json_hf(
                local_path,
                repo_id=REPO_ID,
                path_in_repo=CHAMPION_PATH,
                revision=REVISION,
                commit_message=f"Update champion pointer -> {best_row.run_id}",
            )
    else:
        print("Champion wins:", decision.reason)

if __name__ == "__main__":
    main()