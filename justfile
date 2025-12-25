set shell := ["bash", "-cu"]

default: check

install:
    uv sync
    uv tool install --editable . --force
    @echo "If 'rn' is not found, run: uv tool update-shell (then restart your terminal)"

format:
    uv run ruff format .
    cd dashboard && npm run format

check:
    uv run ruff check .
    uv run ty check .
    cd dashboard && npm run lint
    cd dashboard && npm run typecheck
