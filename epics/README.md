# Epics

An **epic** is a coherent objective/initiative inside a git repository (a “project” is often narrower than the repo itself). A single repo may contain many epics.

Each epic gets a directory under `epics/<slug>/`:

- `README.md`: canonical “control doc” (vision, scope, invariants, architecture, decisions)
- Optional: `notes/`, `artifacts/`, `decisions/`, `tasks/` (task directories under `tasks/<task>/README.md`)

By default, prefer git history + PRs + Linear/GitHub metadata as the changelog. Add `CHANGELOG.md` only if a human-curated changelog becomes necessary.

## Current epics

- `epics/redesmyn/README.md`
- `epics/graph-viz/README.md`
- `epics/agent-orchestration/README.md`
- `epics/harness-interface-v0/README.md`
- `epics/ui-v0/README.md`
- `epics/messages-commands/README.md`
- `epics/revise-architecture/README.md`
- `epics/git-mechanics-v0/README.md`
- `epics/linear-integration/README.md`
- `epics/github-integration/README.md`
- `epics/tests-v0/README.md`
- `epics/v0-launch/README.md`
- `epics/backlog/README.md`
