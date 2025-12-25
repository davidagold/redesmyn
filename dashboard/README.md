# Redesmyn Dashboard

React + TypeScript + Vite (using `rolldown-vite`) + Tailwind + shadcn/ui.

## Dev

- Install deps: `npm install`
- Run dev server: `npm run dev`
  - Proxies `/api/*` to the daemon at `http://127.0.0.1:9234` (see `dashboard/vite.config.ts`).

## Build

- `npm run build` (outputs to `dashboard/dist/`)
- `npm run preview` (serve the static build locally)

## Tooling

- Lint: `npm run lint` (`oxlint`)
- Format: `npm run format` / `npm run format:check` (`oxfmt`)
- Typecheck: `npm run typecheck` (`tsc`)
