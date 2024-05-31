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


RUN_ID ?= 9b129f65953a4ca1add7ec22b454bcc4

serve:
	poetry run python -m serve $(RUN_ID)

predict:
	poetry run python -m predict
