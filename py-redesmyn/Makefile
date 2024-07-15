SHELL := /bin/bash

.PHONY: install develop build test docs clean-docs


FLAGS ?=
MATURIN_OPTIONS = -m Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn


install:
	@poetry install --with dev --no-root

develop:
	@poetry run maturin develop $(MATURIN_OPTIONS) $(FLAGS)

test:
	@poetry run python -m pytest py-redesmyn/tests

build:
	@poetry run maturin build $(MATURIN_OPTIONS) $(FLAGS)

docs: clean-docs
	@poetry run sphinx-build -M html ./docs/src ./docs/build -c ./docs -vv

clean-docs:
	@rm -rf docs/src/api/
	@rm -rf docs/src/**/api/
	@rm -rf docs/build/*
