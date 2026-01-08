# Redesmyn Dashboard

React + TypeScript + Vite (using `rolldown-vite`) + Tailwind + shadcn/ui.

## Dev

- Install deps: `npm install`
- Node: `dashboard/package.json` expects Node `>=22.12.0`; if you see an engine warning (e.g. on Node `22.5.1`), upgrade/switch Node to avoid confusing tooling failures.
- Recommended (single origin): from repo root, run `just dev`
  - Dashboard: `http://127.0.0.1:9234/`
  - API: `http://127.0.0.1:9234/v1/*` (proxied to the daemon on `:9235`)
- Dashboard only: `npm run dev`
  - Proxies `/v1/*` to `REDESMYN_DAEMON_ORIGIN` (defaults to `http://127.0.0.1:9234`; see `dashboard/vite.config.ts`).

## Build

- `npm run build` (outputs to `dashboard/dist/`)
- `npm run preview` (serve the static build locally)

## Tooling

- Lint: `npm run lint` (`oxlint`)
- Format: `npm run format` / `npm run format:check` (`oxfmt`)
- Typecheck: `npm run typecheck` (`tsc`)

## OpenAPI types

- Generate spec + types: `npm run api:update`
  - Requires the backend deps installed via `uv sync --dev`.
