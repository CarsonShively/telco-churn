.PHONY: venv install install-dev lock upgrade dagster dagster-home hf-login hf-logout

SHELL := /bin/bash

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
UV   := $(VENV)/bin/uv
DG := $(VENV)/bin/dg
HF   := $(VENV)/bin/hf

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DAGSTER_HOME_DIR := $(MAKEFILE_DIR).dagster_home

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip uv

install: venv
	$(UV) sync

install-dev: venv
	$(UV) sync --extra dev

lock: venv
	$(UV) lock

dagster-home:
	mkdir -p "$(DAGSTER_HOME_DIR)"

dagster: dagster-home
	@echo "DAGSTER_HOME=$(DAGSTER_HOME_DIR)"
	DAGSTER_HOME="$(DAGSTER_HOME_DIR)" $(DG) dev -m telco_churn.definitions

hf-login:
	$(HF) auth login

hf-logout:
	$(HF) auth logout
