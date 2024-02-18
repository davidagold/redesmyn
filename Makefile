SHELL := /bin/bash

.PHONY: run build

PYO3_PRINT_CONFIG = 0
RUST_LOG ?= INFO
FLAGS ?= ""

build:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) . scripts/build.sh $(FLAGS)

run: build
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run
