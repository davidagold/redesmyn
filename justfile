set shell := ["bash", "-cu"]

default: check

install:
    uv sync
    uv tool install --editable . --force
    @cd dashboard && if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] || ! cmp -s package-lock.json node_modules/.package-lock.json; then npm ci && cp package-lock.json node_modules/.package-lock.json; fi
    cd dashboard && npm run api:update
    cd dashboard && npm run build
    uv run python scripts/build_packaged_dashboard_assets.py --copy-only
    @echo "If 'rn' is not found, run: uv tool update-shell (then restart your terminal)"

dev:
    uv run rn debug dev

hooks:
    hk install

format:
    uv run ruff format .
    cd dashboard && npm run format

check:
    uv run ruff check .
    uv run ty check --extra-search-path . .
    @cd dashboard && if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] || ! cmp -s package-lock.json node_modules/.package-lock.json; then npm ci && cp package-lock.json node_modules/.package-lock.json; fi
    cd dashboard && npm run lint
    cd dashboard && npm run typecheck
    cd dashboard && npm run test

test:
    uv run pytest

e2e:
    uv sync
    @cd dashboard && if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] || ! cmp -s package-lock.json node_modules/.package-lock.json; then npm ci && cp package-lock.json node_modules/.package-lock.json; fi
    @if [ ! -d dashboard/dist ]; then cd dashboard && npm run build; fi
    uv run playwright install chromium
    uv run pytest -m e2e -o addopts="--strict-markers --tb=short -ra"

run flags="":
    @if [[ "{{flags}}" == "--local" ]]; then \
      rn server run & server_pid=$$!; \
      rn daemon run & daemon_pid=$$!; \
      trap 'kill $$server_pid $$daemon_pid 2>/dev/null || true' INT TERM EXIT; \
      while kill -0 $$server_pid 2>/dev/null && kill -0 $$daemon_pid 2>/dev/null; do sleep 0.2; done; \
      if ! kill -0 $$server_pid 2>/dev/null; then wait $$server_pid; status=$$?; else wait $$daemon_pid; status=$$?; fi; \
      kill $$server_pid $$daemon_pid 2>/dev/null || true; \
      wait $$server_pid 2>/dev/null || true; \
      wait $$daemon_pid 2>/dev/null || true; \
      exit $$status; \
    elif [[ -z "{{flags}}" ]]; then \
      rn server run; \
    else \
      echo "error: unknown flag for just run: {{flags}}" 1>&2; \
      exit 2; \
    fi

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

db-check:
    uv run python scripts/check_migrations.py
