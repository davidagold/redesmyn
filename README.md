# Redesmyn

Redesmyn is a local-first cockpit for orchestrating multi-agent work on a git repository using stacked branches (as a tree), with a real-time web UI and first-class GitHub + Linear integration.

## Development quickstart

**Prereqs**

- Python 3.11+
- Node (see `.nvmrc`; Vite currently expects Node `>=22.12.0`)
- `uv` (install via Homebrew: `brew install uv`)

**Backend**

- Install (recommended: `uv`): `uv sync`
- Initialize repo state: `rn init`
- Run daemon (dev): `rn daemon run --reload` (serves API on `http://127.0.0.1:9234`)

**Dashboard**

- `cd dashboard && npm install`
- Run dev server: `npm run dev` (proxies `/v1/*` to the daemon)
- Or build + serve from daemon: `npm run build` then open `http://127.0.0.1:9234/`

## Epics

- `epics/README.md`
- `epics/redesmyn/README.md` (bootstrapping epic; we dogfood Redesmyn to build Redesmyn)
