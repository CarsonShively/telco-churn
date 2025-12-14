# telco-churn

## Setup

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Hugging Face Datasets

This project uses a Hugging Face **dataset repository** as the artifact store for Parquet data (bronze/gold layers).

- Dataset repo: `Carson-Shively/telco-churn`
- Generated Parquet files are written locally and optionally uploaded to Hugging Face.
- Large data artifacts are not committed to git.

### Authentication
For uploading datasets, authenticate once with:
```bash
huggingface-cli login

Alternatively (for CI/automation), set:
export HF_TOKEN=hf_...