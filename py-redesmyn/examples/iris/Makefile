SHELL := /bin/bash

.PHONY: install build run predict


PY_PKG_GROUPS ?= packages dev-packages

install:
	poetry install --with dev


FLAGS ?=
MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

build:
	cd ../../.. && poetry run --directory py-redesmyn/examples/iris maturin develop $(MATURIN_OPTIONS) $(FLAGS)


serve:
	RUST_LOG=debug poetry run python -m serve

predict:
	poetry run python -m predict
