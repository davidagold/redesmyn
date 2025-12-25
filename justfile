set shell := ["bash", "-cu"]

default: check

format:
    uv run ruff format .
    cd dashboard && npm run format

check:
    uv run ruff check .
    uv run ty check .
    cd dashboard && npm run lint
    cd dashboard && npm run typecheck
