SHELL := /bin/bash

.PHONY: install develop build test docs clean-docs


install:
	poetry install --directory py-redesmyn --with dev --no-root


FLAGS ?=
MATURIN_OPTIONS = -m py-redesmyn/Cargo.toml \
	--strip \
	--target-dir target/py-redesmyn

develop:
	@poetry run --directory py-redesmyn maturin develop $(MATURIN_OPTIONS) $(FLAGS)

test:
	@poetry run --directory py-redesmyn python -m pytest py-redesmyn/tests

build:
	@poetry run --directory py-redesmyn maturin build $(MATURIN_OPTIONS) $(FLAGS)

docs: clean-docs
	@poetry run --directory py-redesmyn sphinx-build -M html ./py-redesmyn/docs/src ./py-redesmyn/docs/build \
		-c ./py-redesmyn/docs -vv

clean-docs:
	@rm -rf py-redesmyn/docs/src/api/
	@rm -rf py-redesmyn/docs/src/**/api/
	@rm -rf py-redesmyn/docs/build/*
