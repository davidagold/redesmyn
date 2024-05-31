SHELL := /bin/bash

.PHONY: install-py develop-py run


PY_PKG_GROUPS ?= "packages dev-packages"

install-py:
	pipenv install --categories $(PY_PKG_GROUPS)


FLAGS ?=
MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

develop-py:
	cd ../../.. && poetry run --directory py-redesmyn/examples/iris maturin develop $(MATURIN_OPTIONS) $(FLAGS)


RUN_ID ?= 9b129f65953a4ca1add7ec22b454bcc4

serve:
	poetry run python -m serve $(RUN_ID)

predict:
	poetry run python -m predict



# RUST_LOG ?= INFO

# run-py: develop-py
# 	cd py-redesmyn && \
# 	RUST_LOG=$(RUST_LOG) \
# 	PYTHONPATH=$(shell pipenv --venv)/lib/python3.11/site-packages \
# 	MLFLOW_TRACKING_DIR=$(shell pwd)/data/models/mlflow \
# 	MLFLOW_TRACKING_URI=$(shell pwd)/data/models/mlflow \
# 	python -m tests.test_server
