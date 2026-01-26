# telco-churn

This project is intended as a portfolio grade **ML batch system**. 

The goal is to mirror real-world ML workflows—**data pipelines, reproducible training, model promotion, and result reporting**—rather than notebook-only experimentation or ad-hoc tuning.

---

## Dagster Jobs

1. **Data**
   - Ingest raw churn history.
   - Apply bronze -> silver -> gold transformations.
   - Produce train-ready feature tables.

2. **Train**
   - Config-driven modeling produces trained artifacts and metadata.
   - Supported models: Logistic Regression, XGBoost, LightGBM.

3. **Promotion**
   - Deterministically select the best contender. 
   - Evaluate the best contender against the current champion.
   - Promote only if performance improves by an epsilon threshold.

4. **Batch Report**
   - Score a new batch partition of customer data.
   - Emit a structured batch report including:
      - Churn risk buckets and customer priority ranks.
      - Decision codes and suggested actions for predicted churners.
      - Aggregate batch-level summaries.

---

## Artifact Storage

This system uses **Hugging Face Hub** to store immutable artifact history.

   **dataset repository** 
   - [link](https://huggingface.co/datasets/Carson-Shively/telco-churn)

   **model repository** 
   - [link](https://huggingface.co/Carson-Shively/telco-churn)

> **Safe by Default**
>
> All artifact upload flags are **disabled by default**.
> This allows local dry runs and experimentation **without modifying any remote
> dataset or model repositories**.
>
> Uploads must be explicitly enabled via configuration.
> If you enable uploads, you must be authenticated to upload.

## Running the system
```bash
make install
make dagster
```

1. Open the URL printed in your terminal (usually http://127.0.0.1:3000)
2. Navigate to **Jobs**
3. Select a job
4. Materialize the job
5. Inspect asset-level metadata and artifacts in the Dagster UI



## Planned System Improvements
1. Partitioned batch data ingestion
2. Scheduled Dagster jobs
3. Dagster sensors for automated batch triggering