SHELL := /bin/bash

.PHONY: install develop build test docs clean-docs

MATURIN_FLAGS ?=
MATURIN_OPTIONS = -m Cargo.toml \
	$(MATURIN_FLAGS) \
	--strip \
	--target-dir ./target/py-redesmyn


install:
	@poetry install --with dev --no-root

develop: sync-deps
	@poetry run maturin develop $(MATURIN_OPTIONS)

test:
	@poetry run python -m pytest py-redesmyn/tests

sync-deps:
	poetry run python -m bin.sync_dependencies

build: sync-deps
	poetry run maturin build $(MATURIN_OPTIONS)

build-release:
	@make build MATURIN_FLAGS=--release

docs: clean-docs
	@poetry run sphinx-build -M html ./docs/src ./docs/build -c ./docs -vv

clean-docs:
	@rm -rf docs/src/api/
	@rm -rf docs/src/**/api/
	@rm -rf docs/build/*
