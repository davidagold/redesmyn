# Redesmyn

Redesmyn is a local-first cockpit for orchestrating multi-agent work on a git repository using stacked branches (as a tree), with a real-time web UI and first-class GitHub + Linear integration.

## Development quickstart

**Prereqs**

- Python 3.11+
- Node (see `.nvmrc`; Vite currently expects Node `>=22.12.0`)
- `uv` (install via Homebrew: `brew install uv`)

**Dev (recommended)**

- Run dashboard HMR + daemon reload on a single origin: `rn dev`
  - If `rn` isn’t installed yet: `uv run rn dev`
  - Dashboard: `http://127.0.0.1:9234/` (Vite)
  - API: `http://127.0.0.1:9234/v1/*` (proxied to the daemon on `:9235`)

**Backend**

- Install: `just install` (runs `uv sync` + `uv tool install --editable . --force`)
- Or install deps (recommended: `uv`): `uv sync`
- Make `rn` available without `uv run` (pick one):
  - Recommended (global install): `uv tool install --editable .` (one-time)
  - Or per-shell: `source .venv/bin/activate`
- Initialize repo state: `rn init`
- Run daemon (dev): `rn daemon run --reload` (serves API on `http://127.0.0.1:9234`)

**Dashboard**

- `cd dashboard && npm install`
- Run dev server (separate origin): `npm run dev` (proxies `/v1/*` to the daemon)
- Or build + serve from daemon: `npm run build` then open `http://127.0.0.1:9234/`

## Epics

- `epics/README.md`
- `epics/redesmyn/README.md` (bootstrapping epic; we dogfood Redesmyn to build Redesmyn)
