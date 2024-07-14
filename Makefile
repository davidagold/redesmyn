SHELL := /bin/bash

.PHONY: build run


PYTHON_EXECUTABLE = $(shell poetry env info --executable --directory py-redesmyn)
PYO3_PRINT_CONFIG = 0

build:
	PYO3_PRINT_CONFIG=$(PYO3_PRINT_CONFIG) \
	PYO3_PYTHON=$(PYTHON_EXECUTABLE) \
	PYTHON_SYS_EXECUTABLE=$(PYTHON_EXECUTABLE) \
	cargo build $(FLAGS)


RUST_LOG ?= INFO
PYTHON_VERSION ?= 3.12
FLAGS ?=

run:
	RUST_LOG=$(RUST_LOG) \
	PYTHONPATH=$(shell poetry env info --path --directory py-redesmyn)/lib/python$(PYTHON_VERSION)/site-packages \
	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
	cargo run --package examples --target-dir examples/target $(FLAGS)
