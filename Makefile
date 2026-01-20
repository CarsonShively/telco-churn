.PHONY: venv install data train batch promote dagster

SHELL := /bin/bash

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
DAG  := $(VENV)/bin/dagster

UPLOAD  ?= 0
PROMOTE ?= 0
MODEL   ?= lr

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip

install: venv
	$(PIP) install -e ".[dev,dagster]"

data: ## Run data job (UPLOAD=1 to upload)
	@$(PY) pipelines/data_pipeline.py $(if $(filter 1,$(UPLOAD)),--upload,)

train: ## Train (MODEL=lr|lgb|xgb) (UPLOAD=1 to upload)
	@$(PY) pipelines/train_pipeline.py \
		--model $(MODEL) \
		$(if $(filter 1,$(UPLOAD)),--upload,)

batch:
	@$(PY) pipelines/batch_pipeline.py

promote: ## Promotion job (PROMOTE=1 to promote)
	@$(PY) pipelines/promotion_pipeline.py $(if $(filter 1,$(PROMOTE)),--promote,)

dagster:
	@$(DAG) dev -m telco_churn.definitions
