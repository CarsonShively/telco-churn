.PHONY: venv install dagster

SHELL := /bin/bash

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
DAG  := $(VENV)/bin/dagster

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip

install: venv
	$(PIP) install -e ".[dev,dagster]"

dagster:
	@$(DAG) dev -m telco_churn.definitions