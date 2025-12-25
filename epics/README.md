# Epics

An **epic** is a coherent objective/initiative inside a git repository (a “project” is often narrower than the repo itself). A single repo may contain many epics.

Each epic gets a directory under `epics/<slug>/`:

- `README.md`: canonical “control doc” (vision, scope, invariants, architecture, decisions)
- Optional: `notes/`, `artifacts/`, `decisions/`, `tasks/`

By default, prefer git history + PRs + Linear/GitHub metadata as the changelog. Add `CHANGELOG.md` only if a human-curated changelog becomes necessary.

## Current epics

- `epics/redesmyn/README.md`
