"""Write full local bundle ready to be uploaded to hf model repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from telco_churn.modeling.bundle.write_metrics import (
    assemble_metrics_payload,
    write_metrics_json,
)
from telco_churn.modeling.bundle.write_metadata import (
    assemble_metadata_payload,
    write_metadata_json,
)
from telco_churn.modeling.bundle.write_model import write_model_joblib
from telco_churn.modeling.bundle.model_artifact import ModelArtifact


def write_bundle(
    *,
    bundle_dir: Path,
    artifact_version: int,
    artifact_obj: ModelArtifact,
    best_params: Dict[str, Any],
    cv_summary: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    primary_metric: str,
    direction: str,
    cfg: Any = None,
    feature_names: list[str] | None = None,
) -> Path:
    write_model_joblib(bundle_dir, artifact_obj)

    metrics_payload = assemble_metrics_payload(
        run_id=artifact_obj.run_id,
        artifact_version=artifact_version,
        model_type=artifact_obj.model_type,
        primary_metric=primary_metric,
        direction=direction,
        threshold=artifact_obj.threshold,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
    )
    write_metrics_json(bundle_dir, metrics_payload)

    meta_payload = assemble_metadata_payload(
        run_id=artifact_obj.run_id,
        artifact_version=artifact_version,
        model_type=artifact_obj.model_type,
        best_params=best_params,
        cfg=cfg,
        feature_names=feature_names,
    )
    write_metadata_json(bundle_dir, meta_payload)

    return bundle_dir
