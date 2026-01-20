import dagster as dg
from telco_churn.io.hf import download_dataset_hf, upload_dataset_hf

class HFDataResource(dg.ConfigurableResource):
    repo_id: str
    revision: str

    def download_data(self, filename: str) -> str:
        return download_dataset_hf(repo_id=self.repo_id, filename=filename, revision=self.revision)
    
    def upload_data(self, local_path: str, hf_path: str) -> None:
        return upload_dataset_hf(repo_id=self.repo_id, local_path=local_path, hf_path=hf_path)