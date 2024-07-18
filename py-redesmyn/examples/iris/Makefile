SHELL := /bin/bash

.PHONY: install build serve predict

VIRTUAL_ENV = $(shell poetry env info --path)
install:
	poetry install --with dev --no-root

# develop: sync-deps
# 	@poetry run maturin develop $(MATURIN_OPTIONS)

develop: install
	VIRTUAL_ENV=$(VIRTUAL_ENV) make -f ../../Makefile develop FP_CARGO_TOML=../../Cargo.toml

build:
	@poetry run maturin build $(MATURIN_OPTIONS)

serve:
	RUST_LOG=debug RUST_BACKTRACE=1 poetry run python -m serve

predict:
	poetry run python -m predict
