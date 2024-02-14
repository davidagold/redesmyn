SHELL := /bin/bash

.PHONY: run build

PYO3_PRINT_CONFIG = 0

build:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh

run: build
	RUST_LOG=INFO \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlfow \
	MLFLOW_REGISTRY_DIR=$(shell pwd)/data/models/mlfow \
	cargo run
