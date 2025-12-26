set shell := ["bash", "-cu"]

default: check

install:
    uv sync
    uv tool install --editable . --force
    @if [ ! -d dashboard/node_modules ]; then cd dashboard && npm ci; fi
    cd dashboard && npm run api:update
    cd dashboard && npm run build
    @echo "If 'rn' is not found, run: uv tool update-shell (then restart your terminal)"

dev:
    uv run rn dev

format:
    uv run ruff format .
    cd dashboard && npm run format

check:
    uv run ruff check .
    uv run ty check .
    cd dashboard && npm run lint
    cd dashboard && npm run typecheck

run:
    rn daemon run

