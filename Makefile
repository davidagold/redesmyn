SHELL := /bin/bash

.PHONY: install-py develop-py build-rs build-py run-rs run-py docs-py clean-docs-py


PY_PKG_GROUPS ?= "packages dev-packages"

install-py:
	poetry install --directory py-redesmyn --with dev --no-root


FLAGS ?=
MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

develop-py:
	poetry run --directory=py-redesmyn maturin develop $(MATURIN_OPTIONS) $(FLAGS)

build-py:
	poetry run --directory=py-redesmyn maturin build $(MATURIN_OPTIONS) $(FLAGS)


PYO3_PRINT_CONFIG = 0

build-rs:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh $(FLAGS)


RUST_LOG ?= INFO

run-rs:
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell poetry env info --directory=py-redesmyn)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run --package examples


docs-py: clean-docs-py
	@poetry run --directory=py-redesmyn sphinx-build -M html ./py-redesmyn/docs/src ./py-redesmyn/docs/build \
		-c ./py-redesmyn/docs -vv

clean-docs-py:
	@rm -rf py-redesmyn/docs/src/api/
	@rm -rf py-redesmyn/docs/src/**/api/
	@rm -rf py-redesmyn/docs/build/*

test-py:
	@poetry run --directory=py-redesmyn python -m pytest py-redesmyn/tests
