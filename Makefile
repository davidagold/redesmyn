SHELL := /bin/bash

.PHONY: install-py develop-py build-rs build-py run-rs run-py docs-py clean-docs-py


PY_PKG_GROUPS ?= "packages dev-packages"

install-py:
	pipenv install --categories $(PY_PKG_GROUPS)


FLAGS ?=
MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

develop-py:
	pipenv run maturin develop $(MATURIN_OPTIONS) $(FLAGS)

build-py:
	pipenv run maturin build $(MATURIN_OPTIONS) $(FLAGS)


PYO3_PRINT_CONFIG = 0

build-rs:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh $(FLAGS)


RUST_LOG ?= INFO

run-rs:
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run --package examples

run-py: develop-py
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
