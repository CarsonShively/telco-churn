from dataclasses import dataclass
from typing import Any, Optional
import dagster as dg
from telco_churn.modeling.types import BundleOut
from telco_churn.io.hf import read_model_json, load_model_hf, upload_model_bundle, upload_model_json_hf
from telco_churn.io.hf_run_metrics import fetch_all_run_metrics, RunRow

@dataclass(frozen=True)
class ModelBundle:
    model_version: str
    model: Any
    threshold: Optional[float]
    feature_names: Optional[list[str]]

class HFModelResource(dg.ConfigurableResource):
    repo_id: str
    revision: str

    _bundle: ModelBundle | None = None

    def model_json(self, path_in_repo: str) -> Optional[dict[str, Any]]:
        return read_model_json(repo_id=self.repo_id, revision=self.revision, path_in_repo=path_in_repo)

    def model_artifact(self, path_in_repo):
        return load_model_hf(repo_id=self.repo_id, revision=self.revision, path_in_repo=path_in_repo)
    
    def bundle_upload(self, bundle_dir: str, run_id: str):
        return upload_model_bundle(bundle_dir=bundle_dir, repo_id=self.repo_id, run_id=run_id, revision=self.revision)

    def get_model_bundle(self) -> BundleOut:
        if self._bundle is not None:
            return self._bundle

        champion_ptr = self.model_json("champion.json")
        model_version = champion_ptr["path_in_repo"]

        artifact = self.model_artifact(f"{model_version}/model.joblib")
        model = getattr(artifact, "model", artifact)

        meta = self.model_json(f"{model_version}/metadata.json")
        threshold = meta.get("cfg", {}).get("threshold")
        feature_names = meta.get("feature_names")

        self._bundle = ModelBundle(
            model_version=model_version,
            model=model,
            threshold=threshold,
            feature_names=feature_names,
        )
        return self._bundle

    def run_metrics(self) -> list[RunRow]:
        return fetch_all_run_metrics(repo_id=self.repo_id, revision=self.revision)
    
    def upload_model_json(self, local_path: str, path_in_repo: str):
        return upload_model_json_hf(local_path=local_path, path_in_repo=path_in_repo, repo_id=self.repo_id, revision=self.revision)