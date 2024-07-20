SHELL := /bin/bash

.PHONY: install sync-deps develop build build-release test docs clean-docs

FP_CARGO_TOML ?= ./Cargo.toml
MATURIN_FLAGS ?=
MATURIN_OPTIONS = -m $(FP_CARGO_TOML) \
	$(MATURIN_FLAGS) \
	--strip \
	--target-dir ./target/py-redesmyn


install:
	@poetry install --with dev --no-root

sync-deps:
	@poetry run python -m bin.sync_dependencies

develop: sync-deps
	@poetry run maturin develop $(MATURIN_OPTIONS)

build: sync-deps
	poetry run maturin build $(MATURIN_OPTIONS)

build-release:
	@make build MATURIN_FLAGS=--release

test:
	@RUST_LOG=debug poetry run python -m pytest tests

docs: clean-docs
	@poetry run sphinx-build -M html ./docs/src ./docs/build -c ./docs -vv

clean-docs:
	@rm -rf docs/src/api/
	@rm -rf docs/src/**/api/
	@rm -rf docs/build/*
