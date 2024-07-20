SHELL := /bin/bash

.PHONY: install build serve predict

VIRTUAL_ENV = $(shell poetry env info --path)


install:
	poetry install --with dev --no-root

develop: install
	VIRTUAL_ENV=$(VIRTUAL_ENV) make -f ../../Makefile develop FP_CARGO_TOML=../../Cargo.toml

build:
	@poetry run maturin build $(MATURIN_OPTIONS)

serve:
	RUST_LOG=debug poetry run python -m serve

predict:
	poetry run python -m predict
