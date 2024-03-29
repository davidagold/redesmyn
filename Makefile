SHELL := /bin/bash

.PHONY: build build-python run run-python

PYO3_PRINT_CONFIG = 0
RUST_LOG ?= INFO
FLAGS ?= ""

build:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh $(FLAGS)

build-python:
	maturin develop -m py-redesmyn/Cargo.toml --target-dir target/py-redesmyn

run:
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run --package examples

run-python:
	cd py-redesmyn && \
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	python -m tests.test_server
