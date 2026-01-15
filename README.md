# telco-churn

This project implements a **deterministic, production-style machine learning system** for customer churn prediction.  
The goal is to mirror real-world ML workflows—**data pipelines, feature stores, reproducible training, and model promotion**—rather than notebook-only experimentation or ad-hoc tuning.

The system is designed so that **models, features, and data artifacts are reproducible, comparable, and promotable** in a controlled way.

---

## System Overview

**End-to-end flow:**

1. **Data Pipeline**
   - Raw (bronze) → cleaned/typed (silver) → modeled (gold)

2. **Feature Store Pipeline**
   - Materializes features into Redis
   - Ensures offline (training) and online (serving) feature parity by reusing the same feature definitions and transformations

3. **Training Pipeline**
   - Deterministic training and evaluation across model families
   - Supports LR, XGBoost, and LightGBM under a consistent interface

4. **Promotion Pipeline**
   - Selects the best contender deterministically
   - Evaluates against the current champion
   - Promotes only when criteria are met

---

## Setup

## Create and activate a virtual environment
python3 -m venv .venv

source .venv/bin/activate

## Install dependencies
pip install -U pip

pip install -e .

## Start the local demo app 
python3 space/app.py

## Data pipeline
python3 pipelines/data_pipeline.py

python3 pipelines/data_pipeline.py --upload

## Feature store pipeline
python3 pipelines/feature_store_pipeline.py

## Train pipeline - determinstic training and scoring accross models
### Supported models
- `lr`  — Logistic Regression
- `xgb` — XGBoost
- `lgb` — LightGBM

python3 pipelines/train_pipeline.py --model-type xgb

python3 pipelines/train_pipeline.py --model-type xgb --upload

## Promotion pipeline - Select the best contender and compare it against the current champion
python3 pipelines/promotion_pipeline.py

python3 pipelines/promotion_pipeline.py --promote


## External artifact storage

This project uses a Hugging Face **dataset repository** and **model repository** as the artifact store for Parquet data and model runs .

- Dataset repo: `Carson-Shively/telco-churn`
- Model repo: `Carson-Shively/telco-churn`

## HF Authentication
huggingface-cli login

## Feature Store Reasoning
customer Churn is often handled via batch inference.  
Here, a feature store is used to demonstrate online ML architecture and
production-oriented system design.