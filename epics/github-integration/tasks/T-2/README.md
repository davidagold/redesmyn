---
rn:
  node:
    branch: rn/github-integration/T-2-github-repo-association
  parent: null
  after: []
---

# T-2 GitHub repo association (auto-detect + overrides) + epic-level badge

## Brief (local)

- Associate GitHub repository identity with an epic, with an optional per-task override.
- Default to auto-detection from the local git repo (remote parsing).
- Provide an epic-level badge UI similar to the Linear project badge.

## Acceptance Criteria

- Auto-detect `owner/repo` from the local git repository remote (likely `origin`), supporting common URL forms:
  - `git@github.com:owner/repo.git`
  - `https://github.com/owner/repo.git`
- Persist epic-level GitHub repo mapping (and optional per-task override) in the DB.
- Dashboard subheader shows an epic-level GitHub badge next to the Linear badge:
  - icon-only in the header row,
  - click opens a small settings popover (no “configure screen”).
- v0 keeps PRs to same-repo branches, but the data model should not paint us into a corner for forks later.

## Notes / Design

- This ticket is intentionally about *identity + mapping*, not PR operations.
- Prefer explicit-but-unobtrusive override controls (auto-detection should cover the common case).
