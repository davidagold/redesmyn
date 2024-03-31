SHELL := /bin/bash

.PHONY: install-py build-rs build-py run-rs run-py docs-py clean-docs-py

PYO3_PRINT_CONFIG = 0
RUST_LOG ?= INFO
FLAGS ?= ""
PY_PKG_GROUPS ?= "packages dev-packages"


install-py:
	pipenv install --categories $(PY_PKG_GROUPS)


MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

develop-py:
	pipenv run maturin develop $(MATURIN_OPTIONS) \
		--pip-path="$(shell pipenv --venv)/bin/pip3"

build-py:
	pipenv run maturin build $(MATURIN_OPTIONS)

build-rs:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh $(FLAGS) \
		-i $(shell pipenv --py)


run-rs:
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run --package examples

run-py:
	cd py-redesmyn && \
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	python -m tests.test_server


docs-py:
	pipenv run sphinx-build -M html ./py-redesmyn/docs/src ./py-redesmyn/docs/build \
		-c ./py-redesmyn/docs -vv

clean-docs-py:
	@rm -rf py-redesmyn/docs/src/api/
	@rm -rf py-redesmyn/docs/src/**/api/
	@rm -rf py-redesmyn/docs/build/*
