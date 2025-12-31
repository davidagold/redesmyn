set shell := ["bash", "-cu"]

default: check

install:
    uv sync
    uv tool install --editable . --force
    @cd dashboard && if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] || ! cmp -s package-lock.json node_modules/.package-lock.json; then npm ci; fi
    cd dashboard && npm run api:update
    cd dashboard && npm run build
    @echo "If 'rn' is not found, run: uv tool update-shell (then restart your terminal)"

dev:
    uv run rn dev

hooks:
    hk install

format:
    uv run ruff format .
    cd dashboard && npm run format

check:
    uv run ruff check .
    uv run ty check .
    cd dashboard && npm run lint
    cd dashboard && npm run typecheck
    cd dashboard && npm run test

run:
    rn daemon run

# Database (Alembic)
db-current:
    uv run alembic -c alembic.ini current

db-history:
    uv run alembic -c alembic.ini history

db-upgrade rev="head":
    uv run alembic -c alembic.ini upgrade {{rev}}

db-downgrade rev="-1":
    uv run alembic -c alembic.ini downgrade {{rev}}

db-revision msg:
    uv run alembic -c alembic.ini revision -m "{{msg}}" --autogenerate
